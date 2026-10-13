"""The agent loop (§05).

    planner -> step queue -> executor -> observer -> critic
                   ^                                    |
                   +------------ re-planner <-----------+

An agent is not a chat model with tools bolted on; it is a loop with state,
budgets and a way to notice it has failed.

Three properties this file exists to guarantee:

  BUDGETS ARE TERMINAL. Exhausted means stop and report. It never means loop
  again, and it never means "one more try". A run that cannot finish inside its
  budget produces a partial result and says so.

  APPROVAL PAUSES, IT DOES NOT BLOCK. A step needing sign-off parks the run in
  AWAITING_APPROVAL and returns. The process can exit; the run resumes later from
  SQLite. An approval gate that only holds while the process is up is theatre.

  FAILURE REPLANS ONCE PER STEP. The critic's verdict feeds the re-planner, which
  amends the remaining steps. Bounded, because an unbounded replan loop on a
  12 GB card is a demo watching a model change its mind.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..audit import get_audit
from ..config import get_settings
from ..tools import (
    ApprovalRequired,
    PolicyDenied,
    Workspace,
    get_executor,
)
from ..tools.registry import ToolError
from ..types import (
    Budget,
    Label,
    Observation,
    Run,
    RunStatus,
    Step,
    StepStatus,
    WorkMode,
)
from .critic import Verdict, get_critic
from .planner import PlanError, get_planner
from .references import resolve_references
from .state import get_run_store

MAX_REPLANS = 2

# Failures that NO PLAN CAN ROUTE AROUND.
#
# Replanning exists for a model that chose badly: wrong tool, wrong argument,
# wrong order. It is useless against a missing dependency, and worse than
# useless in the timeline - a run against a box with no Docker produced the
# same code.exec step three times, failed it three times with the identical
# message, spent both replans, and buried the one line that mattered ("install
# Docker") under two copies of itself.
#
# The executor stamps the exception class onto the observation's stderr
# (`f"{type(exc).__name__}: {exc}"` in ToolExecutor.invoke), so the class name
# is the prefix. Matched on the prefix rather than on the message, because the
# message is written for a human and will be reworded; the class name is an
# API.
#
# Deliberately narrow. ToolNotFound and InvalidArguments are NOT here: a model
# that called a tool that does not exist, or passed the wrong argument, is
# exactly what a replan is for.
UNFIXABLE_BY_REPLANNING = (
    "SandboxUnavailable:",    # code.exec with no Docker - install it
    "DeliverableError:",      # doc.write with no python-docx - install it
)


def _needs_an_operator(observation: Observation) -> str | None:
    """The reason this run should stop now, or None to replan as usual."""
    text = (observation.stderr or "").lstrip()
    for prefix in UNFIXABLE_BY_REPLANNING:
        if text.startswith(prefix):
            return text
    return None


@dataclass
class RunEvent:
    """One thing that happened, for the timeline and for SSE in Phase 7."""

    kind: str
    run_id: str
    step_idx: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "detail": self.detail,
            "at": self.at,
        }


class AgentLoop:
    def __init__(self, planner=None, critic=None, executor=None, store=None):
        self._planner = planner
        self._critic = critic
        self._executor = executor
        self._store = store

    @property
    def planner(self):
        return self._planner or get_planner()

    @property
    def critic(self):
        return self._critic or get_critic()

    @property
    def executor(self):
        return self._executor or get_executor()

    @property
    def store(self):
        return self._store or get_run_store()

    # ---------------------------------------------------------------- start

    def start(
        self,
        goal: str,
        *,
        label: Label = Label.INTERNAL,
        user: str = "local",
        budget: Budget | None = None,
        run_id: str | None = None,
        mode: WorkMode = WorkMode.AUTO,
    ) -> Run:
        """Plan a run and persist it. Does not execute - call `advance`."""
        run = Run(goal=goal, label=label, user=user, budget=budget or Budget(), mode=mode)
        if run_id:
            run.id = run_id

        audit = get_audit()
        audit.append(
            "run.start", actor=user, run_id=run.id, data_class=label.value,
            goal=goal[:400],
            # In the chain, not just in SQLite. Which deliverable was asked for
            # is part of what an auditor reconstructs about a run, and the run
            # row can be re-read while the chain cannot be rewritten.
            mode=mode.value,
        )

        self.store.create(run)

        try:
            plan = self.planner.plan(goal, label=label, run_id=run.id, mode=mode)
        except PlanError as exc:
            run.status = RunStatus.FAILED
            run.error = f"planning failed: {exc}"
            self.store.save_run(run)
            audit.append("run.failed", run_id=run.id, reason=run.error)
            return run

        run.steps = plan.steps
        run.status = RunStatus.RUNNING
        self.store.create(run)
        return run

    # ---------------------------------------------------------------- advance

    def advance(
        self,
        run_id: str,
        *,
        max_steps: int | None = None,
        approvals: dict[int, str] | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> Run:
        """Run until done, blocked on approval, or out of budget.

        `approvals` maps step index -> approver name, supplying sign-off for steps
        that requested it. Re-calling advance with an approval resumes the run.
        """
        run = self.store.load(run_id)
        approvals = approvals or {}
        emit = on_event or (lambda _e: None)
        audit = get_audit()

        if run.status in (RunStatus.DONE, RunStatus.FAILED):
            return run

        workspace = Workspace.for_run(get_settings().var_dir, run_id)
        replans = 0
        executed = 0

        while True:
            exhausted = run.budget.exhausted()
            if exhausted:
                run.status = RunStatus.BUDGET_EXHAUSTED
                run.error = exhausted
                self.store.save_run(run)
                audit.append("run.budget_exhausted", run_id=run_id, reason=exhausted)
                emit(RunEvent("budget_exhausted", run_id, detail={"reason": exhausted}))
                return run

            if max_steps is not None and executed >= max_steps:
                self.store.save_run(run)
                return run

            step = self._next_step(run)
            if step is None:
                run.status = RunStatus.DONE
                self.store.save_run(run)
                audit.append(
                    "run.done", run_id=run_id,
                    steps=len(run.steps),
                    artifacts=len(self.store.artifacts(run_id)),
                )
                emit(RunEvent("done", run_id))
                return run

            # ---- approval gate -------------------------------------------
            approver = approvals.get(step.idx)
            if step.requires_approval and approver is None:
                step.status = StepStatus.AWAITING_APPROVAL
                run.status = RunStatus.AWAITING_APPROVAL
                self.store.save_step(run_id, step)
                self.store.save_run(run)
                audit.append(
                    "run.awaiting_approval", run_id=run_id, step_idx=step.idx,
                    tool=step.tool, intent=step.intent,
                )
                emit(RunEvent("awaiting_approval", run_id, step.idx,
                              {"intent": step.intent, "tool": step.tool}))
                return run

            # ---- execute --------------------------------------------------
            step.status = StepStatus.RUNNING
            step.attempts += 1
            self.store.save_step(run_id, step)
            emit(RunEvent("step_started", run_id, step.idx,
                          {"intent": step.intent, "tool": step.tool}))

            observation = self._execute(run, step, workspace, approver)
            self.store.add_observation(run_id, observation)
            run.budget.tool_calls_used += 1
            executed += 1

            # ---- observe + critique --------------------------------------
            verdict = self.critic.judge(step, observation, workspace, run_id=run_id)
            audit.append(
                "step.judged", run_id=run_id, step_idx=step.idx,
                ok=verdict.ok, layer=verdict.layer, reason=verdict.reason[:300],
            )

            if verdict.ok:
                step.status = StepStatus.PASSED
                run.budget.steps_used += 1
                self.store.save_step(run_id, step)
                self._record_artifacts(run_id, step, observation, workspace)
                emit(RunEvent("step_passed", run_id, step.idx, verdict.as_dict()))
                continue

            # ---- failed --------------------------------------------------
            step.status = StepStatus.FAILED
            run.budget.steps_used += 1
            self.store.save_step(run_id, step)
            emit(RunEvent("step_failed", run_id, step.idx, verdict.as_dict()))

            # Stop NOW if no plan could have avoided this. Two more attempts
            # would produce two more identical failures and hide the fix.
            blocked = _needs_an_operator(observation)
            if blocked:
                run.status = RunStatus.FAILED
                run.error = (
                    f"step {step.idx} ({step.intent}) cannot run on this "
                    f"computer, and no change to the plan would help - so the "
                    f"remaining attempts were not spent repeating it. "
                    f"{blocked}"
                )
                self.store.save_run(run)
                audit.append(
                    "run.failed", run_id=run_id, reason=run.error,
                    unfixable_by_replanning=True,
                )
                emit(RunEvent("failed", run_id, step.idx, {"reason": run.error}))
                return run

            if replans >= MAX_REPLANS:
                run.status = RunStatus.FAILED
                run.error = (
                    f"step {step.idx} ({step.intent}) failed and the replan budget "
                    f"({MAX_REPLANS}) is spent: {verdict.reason}"
                )
                self.store.save_run(run)
                audit.append("run.failed", run_id=run_id, reason=run.error)
                emit(RunEvent("failed", run_id, step.idx, {"reason": run.error}))
                return run

            replans += 1
            if not self._replan(run, step, verdict, emit):
                return run

    # ---------------------------------------------------------------- pieces

    @staticmethod
    def _next_step(run: Run) -> Step | None:
        for step in sorted(run.steps, key=lambda s: s.idx):
            if step.status in (StepStatus.PENDING, StepStatus.AWAITING_APPROVAL):
                return step
        return None

    def _observations_by_index(self, run_id: str) -> dict[int, Any]:
        """Latest observation per step. A retried step should be quoted by what
        it finally produced, not by its first attempt."""
        latest: dict[int, Any] = {}
        for obs in self.store.load_observations(run_id):
            latest[obs.step_idx] = obs
        return latest

    def _execute(

        self, run: Run, step: Step, workspace: Workspace, approver: str | None
    ) -> Observation:
        if not step.tool:
            return Observation(
                step_idx=step.idx, ok=False,
                stderr="step has no tool; the planner should never emit one",
            )

        # A plan is written before anything runs, so a step that needs what an
        # earlier step FOUND can only say so by reference. Resolve those now,
        # when the observation exists. Without this a generated approval note
        # could only contain what the model knew at planning time - which is why
        # they came out a line long.
        args, notes = resolve_references(
            step.args, self._observations_by_index(run.id), step.idx
        )
        if notes:
            get_audit().append(
                "step.references_resolved",
                run_id=run.id, step_idx=step.idx, detail="; ".join(notes[:6]),
            )

        try:
            result = self.executor.invoke(
                step.tool,
                args,
                workspace,
                label=run.label,
                role="engineer",
                run_id=run.id,
                step_idx=step.idx,
                approved_by=approver,
            )
        except ApprovalRequired as exc:
            # The policy engine wants sign-off even though the plan did not
            # anticipate it - mark the step so the next advance() pauses on it.
            step.requires_approval = True
            return Observation(step_idx=step.idx, ok=False, stderr=str(exc))
        except PolicyDenied as exc:
            return Observation(step_idx=step.idx, ok=False, stderr=f"denied: {exc}")
        except ToolError as exc:
            return Observation(step_idx=step.idx, ok=False, stderr=str(exc))

        data = result.data or {}

        # A run inherits the strictest label of anything it has read (§02.5).
        # Until Phase 6 nothing could raise it, because no tool returned
        # classified content; kb.search does, and a run that has read one
        # CONFIDENTIAL clause is a CONFIDENTIAL run for every step after it.
        inherited = data.get("effective_label")
        if isinstance(inherited, str):
            try:
                promoted = Label.max(run.label, Label(inherited))
            except ValueError:
                promoted = run.label
            if promoted is not run.label:
                run.label = promoted
                self.store.save_run(run)
                get_audit().append(
                    "run.label_raised", run_id=run.id, step_idx=step.idx,
                    data_class=promoted.value, via=step.tool,
                )

        return Observation(
            step_idx=step.idx,
            ok=result.ok,
            stdout=str(data.get("stdout", "") or ""),
            stderr=(result.error or data.get("stderr", "") or ""),
            exit_code=int(data.get("exit_code", 0) or 0),
            artifacts=[data["path"]] if isinstance(data.get("path"), str) else [],
            data={k: v for k, v in data.items() if k not in ("stdout", "stderr")},
        )

    def _record_artifacts(
        self, run_id: str, step: Step, observation: Observation, workspace: Workspace
    ) -> None:
        import hashlib

        for rel in observation.artifacts:
            try:
                resolved = workspace.resolve(rel, must_exist=True)
            except (OSError, ValueError):
                continue
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            self.store.add_artifact(run_id, rel, resolved.suffix.lstrip(".") or "file", digest)
            get_audit().append(
                "artifact.write", run_id=run_id, step_idx=step.idx,
                path=rel, sha256=digest,
            )

    def _replan(
        self, run: Run, failed: Step, verdict: Verdict, emit: Callable[[RunEvent], None]
    ) -> bool:
        completed = [s for s in run.steps if s.status in
                     (StepStatus.PASSED, StepStatus.FAILED)]
        try:
            plan = self.planner.replan(
                run.goal, completed, verdict.reason, label=run.label, run_id=run.id,
                mode=run.mode,
            )
        except PlanError as exc:
            run.status = RunStatus.FAILED
            run.error = f"replan failed after step {failed.idx}: {exc}"
            self.store.save_run(run)
            get_audit().append("run.failed", run_id=run.id, reason=run.error)
            emit(RunEvent("failed", run.id, failed.idx, {"reason": run.error}))
            return False

        run.steps = completed + plan.steps
        self.store.replace_steps(run.id, run.steps)
        self.store.save_run(run)
        get_audit().append(
            "run.replanned", run_id=run.id, after_step=failed.idx,
            new_steps=len(plan.steps), reason=verdict.reason[:200],
        )
        emit(RunEvent("replanned", run.id, failed.idx,
                      {"new_steps": len(plan.steps), "reason": verdict.reason[:200]}))
        return True

    # ---------------------------------------------------------------- convenience

    def run_to_completion(
        self,
        goal: str,
        *,
        label: Label = Label.INTERNAL,
        budget: Budget | None = None,
        auto_approve: str | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
        mode: WorkMode = WorkMode.AUTO,
    ) -> Run:
        """Plan and execute in one call.

        `auto_approve` supplies a name for every approval gate. It exists for
        tests and for the unattended rehearsal; a real run should call advance()
        with explicit per-step approvals so a human actually sees them.
        """
        run = self.start(goal, label=label, budget=budget, mode=mode)
        if run.status is RunStatus.FAILED:
            return run

        while True:
            approvals = (
                {s.idx: auto_approve for s in run.steps if s.requires_approval}
                if auto_approve
                else {}
            )
            run = self.advance(run.id, approvals=approvals, on_event=on_event)
            if run.status is not RunStatus.AWAITING_APPROVAL or not auto_approve:
                return run


_loop: AgentLoop | None = None


def get_loop() -> AgentLoop:
    global _loop
    if _loop is None:
        _loop = AgentLoop()
    return _loop


def reset_loop() -> None:
    """Tests only."""
    global _loop
    _loop = None

"""Run state (§05).

A run is inspectable and resumable because every step, observation and artifact
is written to SQLite as it happens - not held in memory and flushed at the end.

That matters for three reasons, in ascending order of importance:

  1. The UI can render a run timeline by polling, with no shared memory.
  2. A run interrupted by a crash can be resumed rather than restarted.
  3. A run that pauses for human approval has to survive the wait. An approval
     gate that only works while the process stays up is not an approval gate.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..store import Database, get_db
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


class RunNotFound(KeyError):
    pass


def _mode_of(row: Any) -> WorkMode:
    """The run's work mode, tolerant of what an older row can contain.

    Two things can go wrong reading this and neither should lose a run: the
    column may be absent (a database written before the migration, opened by a
    build that somehow skipped it), or it may hold a value this version no
    longer knows. Both fall back to AUTO, which is what every pre-toggle run
    genuinely was. Raising here would make a run from last week unopenable.
    """
    try:
        raw = row["mode"]
    except (IndexError, KeyError):
        return WorkMode.AUTO
    try:
        return WorkMode(raw)
    except ValueError:
        return WorkMode.AUTO


class RunStore:
    def __init__(self, db: Database | None = None):
        self.db = db or get_db()

    # ---------------------------------------------------------------- write

    def create(self, run: Run) -> Run:
        self.db.upsert_run(run)
        for step in run.steps:
            self.db.upsert_step(run.id, step)
        return run

    def save_run(self, run: Run) -> None:
        self.db.upsert_run(run)

    def reconcile_interrupted(self) -> list[str]:
        """Fail runs that were mid-flight when the process died. Returns their ids.

        A run executes on a thread. If the gateway is restarted or killed while
        that thread is working, SQLite keeps the last status it was given -
        PLANNING or RUNNING - and nothing ever moves it again. The panel then
        shows a run as live forever, and `GET /live` says live=false, which is
        the same subsystem disagreeing with itself. A run that cannot progress
        must not look like one that is thinking.

        Called at boot, where the invariant is unambiguous: no run thread can
        exist yet, so any run in an executing state is a leftover.

        AWAITING_APPROVAL is deliberately untouched. Parking for a human is
        exactly what is meant to survive a restart - that is the property
        test_a_paused_run_survives_a_process_restart pins.
        """
        stranded = [
            row["id"] for row in self.list_runs(limit=500)
            if row["status"] in (RunStatus.PLANNING.value, RunStatus.RUNNING.value)
        ]
        for run_id in stranded:
            run = self.load(run_id)
            run.status = RunStatus.FAILED
            run.error = (
                "the gateway restarted while this run was executing, so its "
                "thread is gone. The plan and every completed step are still "
                "here; start a new run to continue."
            )
            self.db.upsert_run(run)
        return stranded

    def save_step(self, run_id: str, step: Step) -> None:
        self.db.upsert_step(run_id, step)

    def replace_steps(self, run_id: str, steps: list[Step]) -> None:
        """Used by the re-planner, which amends the REMAINING steps.

        Completed steps are never rewritten - a plan that edits its own history
        is not a plan, it is a story. Only pending work is replaced.
        """
        done = {
            row["idx"]
            for row in self.db.query(
                "SELECT idx FROM steps WHERE run_id=? AND status IN "
                "('passed','failed','skipped')",
                (run_id,),
            )
        }
        self.db.execute(
            "DELETE FROM steps WHERE run_id=? AND status NOT IN "
            "('passed','failed','skipped')",
            (run_id,),
        )
        for step in steps:
            if step.idx not in done:
                self.db.upsert_step(run_id, step)

    def add_observation(self, run_id: str, obs: Observation) -> None:
        self.db.add_observation(run_id, obs)

    def add_artifact(self, run_id: str, path: str, kind: str, sha256: str) -> None:
        self.db.add_artifact(run_id, path, kind, sha256)

    def approve_artifact(self, run_id: str, path: str, approver: str) -> None:
        self.db.approve_artifact(run_id, path, approver)

    # ---------------------------------------------------------------- read

    def load(self, run_id: str) -> Run:
        row = self.db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if row is None:
            raise RunNotFound(f"no run {run_id!r}")

        budget_raw = json.loads(row["budget_json"])
        budget = Budget(
            max_steps=budget_raw.get("max_steps", 12),
            max_wall_s=budget_raw.get("max_wall_s", 480),
            max_tool_calls=budget_raw.get("max_tool_calls", 20),
            steps_used=budget_raw.get("steps_used", 0),
            tool_calls_used=budget_raw.get("tool_calls_used", 0),
        )

        run = Run(
            id=row["id"],
            user=row["user"],
            goal=row["goal"],
            label=Label(row["label"]),
            status=RunStatus(row["status"]),
            budget=budget,
            created_at=row["created_at"],
            error=row["error"],
            mode=_mode_of(row),
        )
        run.steps = self.load_steps(run_id)
        run.observations = self.load_observations(run_id)
        run.artifacts = [
            r["path"] for r in self.db.query(
                "SELECT path FROM artifacts WHERE run_id=? ORDER BY created_at", (run_id,)
            )
        ]
        return run

    def load_steps(self, run_id: str) -> list[Step]:
        return [
            Step(
                idx=r["idx"],
                intent=r["intent"],
                tool=r["tool"],
                args=json.loads(r["args_json"]),
                acceptance=r["acceptance"],
                status=StepStatus(r["status"]),
                attempts=r["attempts"],
                requires_approval=bool(r["requires_approval"]),
            )
            for r in self.db.query(
                "SELECT * FROM steps WHERE run_id=? ORDER BY idx", (run_id,)
            )
        ]

    def load_observations(self, run_id: str) -> list[Observation]:
        return [
            Observation(
                step_idx=r["step_idx"],
                ok=bool(r["ok"]),
                stdout=r["stdout"],
                stderr=r["stderr"],
                exit_code=r["exit_code"],
                artifacts=json.loads(r["artifacts_json"]),
                data=json.loads(r["data_json"]),
            )
            for r in self.db.query(
                "SELECT * FROM observations WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "goal": r["goal"],
                "status": r["status"],
                "label": r["label"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "error": r["error"],
                "mode": _mode_of(r).value,
            }
            for r in self.db.query(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        ]

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query(
                "SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at", (run_id,)
            )
        ]

    # ---------------------------------------------------------------- timeline

    def timeline(self, run_id: str) -> dict[str, Any]:
        """Everything the centre panel renders: the plan as it was produced, each
        step's current state, the critic's verdict and the retries."""
        run = self.load(run_id)
        observations: dict[int, list[Observation]] = {}
        for obs in run.observations:
            observations.setdefault(obs.step_idx, []).append(obs)

        return {
            "run_id": run.id,
            "goal": run.goal,
            "status": run.status.value,
            "label": run.label.value,
            "mode": run.mode.value,
            "error": run.error,
            "budget": run.budget.as_dict(),
            "created_at": run.created_at,
            "artifacts": self.artifacts(run_id),
            "steps": [
                {
                    "idx": step.idx,
                    "intent": step.intent,
                    "tool": step.tool,
                    "args": step.args,
                    "acceptance": step.acceptance,
                    "status": step.status.value,
                    "attempts": step.attempts,
                    "requires_approval": step.requires_approval,
                    "observations": [
                        {
                            "ok": o.ok,
                            "stdout": o.stdout[:4000],
                            "stderr": o.stderr[:2000],
                            "exit_code": o.exit_code,
                            "artifacts": o.artifacts,
                            "data": o.data,
                        }
                        for o in observations.get(step.idx, [])
                    ],
                }
                for step in run.steps
            ],
        }


_store: RunStore | None = None


def get_run_store() -> RunStore:
    global _store
    if _store is None:
        _store = RunStore()
    return _store


def reset_run_store() -> None:
    """Tests only."""
    global _store
    _store = None


def touch_updated(run: Run) -> None:
    run.budget.steps_used = sum(
        1 for s in run.steps if s.status in (StepStatus.PASSED, StepStatus.FAILED)
    )
    run.created_at = run.created_at or time.time()

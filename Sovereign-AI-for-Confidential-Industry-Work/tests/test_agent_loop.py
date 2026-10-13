"""The agent loop (§05).

Three properties this file exists to protect, in order of how badly their absence
would show in a live demo:

  BUDGETS ARE TERMINAL      exhausted means stop and report, never "one more try"
  APPROVAL PAUSES          the run survives the wait, including a process restart
  FAILURE REPLANS, BOUNDED the critic's verdict drives a replan, twice at most
"""

from __future__ import annotations

import pytest

from kavach.agent import get_loop, get_run_store
from kavach.audit import get_audit
from kavach.types import Budget, Label, RunStatus, StepStatus


@pytest.fixture()
def loop():
    return get_loop()


@pytest.fixture()
def store():
    return get_run_store()


# ---------------------------------------------------------------- planning

def test_a_run_produces_a_plan(loop):
    run = loop.start("Calculate the pressure drop and write it down")
    assert run.status is RunStatus.RUNNING
    assert run.steps
    assert all(s.status is StepStatus.PENDING for s in run.steps)


def test_every_planned_step_has_an_acceptance_test(loop):
    """A plan without acceptance tests is a list of intentions, and a loop over
    intentions cannot tell success from a confident failure."""
    run = loop.start("do some engineering work")
    assert all(s.acceptance.strip() for s in run.steps)


def test_every_planned_step_names_a_real_tool(loop):
    from kavach.tools import get_tool_registry

    names = set(get_tool_registry().names())
    run = loop.start("do some engineering work")
    assert all(s.tool in names for s in run.steps)


def test_planning_is_audited(loop):
    loop.start("something")
    assert "plan.accepted" in [r["event"] for r in get_audit().tail(20)]


# ---------------------------------------------------------------- execution

def test_a_run_completes(loop):
    run = loop.run_to_completion("Calculate something and write it to a file")
    assert run.status is RunStatus.DONE
    assert all(s.status is StepStatus.PASSED for s in run.steps)


def test_events_describe_the_timeline(loop):
    events = []
    loop.run_to_completion("Calculate and write", on_event=events.append)
    kinds = [e.kind for e in events]
    assert "step_started" in kinds
    assert "step_passed" in kinds
    assert kinds[-1] == "done"


def test_artifacts_are_recorded_with_a_digest(loop, store):
    run = loop.run_to_completion("Calculate and write it to a file")
    artifacts = store.artifacts(run.id)
    assert artifacts
    assert len(artifacts[0]["sha256"]) == 64


def test_the_run_is_persisted_not_held_in_memory(loop, store):
    """The UI polls, the process may restart, and an approval gate has to
    survive the wait - all of which need the run on disk."""
    run = loop.run_to_completion("Calculate and write")
    reloaded = store.load(run.id)
    assert reloaded.status is run.status
    assert len(reloaded.steps) == len(run.steps)
    assert reloaded.goal == run.goal


# ---------------------------------------------------------------- budgets

def test_step_budget_is_terminal(loop):
    run = loop.start("do a lot of things", budget=Budget(max_steps=1))
    run = loop.advance(run.id)
    assert run.status is RunStatus.BUDGET_EXHAUSTED
    assert "step budget" in run.error


def test_tool_call_budget_is_terminal(loop):
    run = loop.start("do a lot of things", budget=Budget(max_tool_calls=1))
    run = loop.advance(run.id)
    assert run.status is RunStatus.BUDGET_EXHAUSTED
    assert "tool-call budget" in run.error


def test_wall_clock_budget_is_terminal(loop):
    run = loop.start("do a lot of things", budget=Budget(max_wall_s=0))
    run = loop.advance(run.id)
    assert run.status is RunStatus.BUDGET_EXHAUSTED
    assert "wall-clock" in run.error


def test_exhaustion_stops_rather_than_retrying(loop):
    """The failure mode this prevents: a loop that keeps going because it is
    'nearly there'. Exhausted is a full stop."""
    run = loop.start("do a lot", budget=Budget(max_steps=1))
    run = loop.advance(run.id)
    before = [s.status for s in run.steps]
    run = loop.advance(run.id)
    assert run.status is RunStatus.BUDGET_EXHAUSTED
    assert [s.status for s in run.steps] == before


def test_exhaustion_is_audited(loop):
    run = loop.start("x", budget=Budget(max_steps=1))
    loop.advance(run.id)
    assert "run.budget_exhausted" in [r["event"] for r in get_audit().tail(20)]


# ---------------------------------------------------------------- approval

def test_a_gated_step_pauses_the_run(loop, store):
    run = loop.start("do the thing")
    step = run.steps[0]
    step.requires_approval = True
    store.save_step(run.id, step)

    run = loop.advance(run.id)
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert run.steps[0].status is StepStatus.AWAITING_APPROVAL


def test_a_paused_run_survives_a_process_restart(loop, store):
    """An approval gate that only holds while the process is up is theatre."""
    run = loop.start("do the thing")
    step = run.steps[0]
    step.requires_approval = True
    store.save_step(run.id, step)
    loop.advance(run.id)

    # Nothing in memory: load purely from SQLite, as a fresh process would.
    reloaded = store.load(run.id)
    assert reloaded.status is RunStatus.AWAITING_APPROVAL
    assert reloaded.steps[0].requires_approval is True


def test_approval_resumes_the_run(loop, store):
    run = loop.start("do the thing")
    step = run.steps[0]
    step.requires_approval = True
    store.save_step(run.id, step)
    loop.advance(run.id)

    run = loop.advance(run.id, approvals={0: "pverma"})
    assert run.status is RunStatus.DONE


def test_a_gated_step_does_not_execute_before_approval(loop, store):
    run = loop.start("Calculate and write it to a file")
    write_step = next(s for s in run.steps if s.tool == "fs.write")
    write_step.requires_approval = True
    store.save_step(run.id, write_step)

    loop.advance(run.id)
    from kavach.config import get_settings
    from kavach.tools import Workspace

    ws = Workspace.for_run(get_settings().var_dir, run.id)
    assert not (ws.root / "result.txt").exists(), "the gated write ran anyway"


def test_pausing_is_audited(loop, store):
    run = loop.start("do the thing")
    step = run.steps[0]
    step.requires_approval = True
    store.save_step(run.id, step)
    loop.advance(run.id)
    assert "run.awaiting_approval" in [r["event"] for r in get_audit().tail(20)]


# ---------------------------------------------------------------- the critic

def test_a_failed_acceptance_test_triggers_a_replan(loop, store):
    run = loop.start("do the thing")
    step = run.steps[0]
    step.acceptance = "file never_created.txt exists"
    store.save_step(run.id, step)

    events = []
    run = loop.advance(run.id, on_event=events.append)
    assert "replanned" in [e.kind for e in events]


def test_the_failed_step_stays_in_history(loop, store):
    """A plan that edits its own history is not a plan, it is a story."""
    run = loop.start("do the thing")
    step = run.steps[0]
    step.acceptance = "file never_created.txt exists"
    store.save_step(run.id, step)

    run = loop.advance(run.id)
    assert run.steps[0].status is StepStatus.FAILED
    assert len(run.steps) > 1


def test_replanning_is_bounded(loop, store, monkeypatch):
    """An unbounded replan loop on a 12 GB card is a demo watching a model
    change its mind. Every step failing must terminate, not spin."""
    from kavach.agent.critic import Critic, Verdict

    monkeypatch.setattr(
        Critic, "judge",
        lambda self, step, obs, ws, run_id=None: Verdict(False, "always fails"),
    )
    run = loop.start("do the thing")
    run = loop.advance(run.id)
    assert run.status is RunStatus.FAILED
    assert "replan budget" in run.error


def test_a_tool_failure_is_observed_not_raised(loop, store):
    run = loop.start("do the thing")
    step = run.steps[0]
    step.tool = "fs.read"
    step.args = {"path": "definitely-missing.txt"}
    step.acceptance = "the step produces output"
    store.save_step(run.id, step)

    run = loop.advance(run.id)
    assert run.status in (RunStatus.DONE, RunStatus.FAILED)
    observations = store.load_observations(run.id)
    assert any(not o.ok for o in observations)


# ---------------------------------------------------------------- policy

def test_the_run_label_reaches_the_tool_gate(loop, store):
    """A CONFIDENTIAL run's tool calls must be audited at that classification."""
    loop.run_to_completion("Calculate and write", label=Label.CONFIDENTIAL)
    records = [r for r in get_audit().tail(60) if r["event"] == "tool.invoke"]
    assert records
    assert any(r["data_class"] == "CONFIDENTIAL" for r in records)


# ---------------------------------------------------------------- timeline

def test_timeline_has_what_the_panel_renders(loop, store):
    run = loop.run_to_completion("Calculate and write")
    timeline = store.timeline(run.id)
    assert timeline["status"] == "done"
    assert timeline["steps"]
    step = timeline["steps"][0]
    for key in ("idx", "intent", "tool", "acceptance", "status", "attempts",
                "observations"):
        assert key in step


def test_runs_are_listed_newest_first(loop, store):
    a = loop.start("first goal")
    b = loop.start("second goal")
    listed = [r["id"] for r in store.list_runs(10)]
    assert listed.index(b.id) < listed.index(a.id)


def test_the_audit_chain_survives_a_full_run(loop):
    loop.run_to_completion("Calculate and write it down")
    assert get_audit().verify().ok


# ------------------------------------------- failures no plan can route around


def test_a_missing_dependency_does_not_burn_the_replan_budget(monkeypatch):
    """Replanning is for a model that chose badly, not for a box missing Docker.

    Against a machine with no sandbox, the loop produced the same code.exec
    step three times, failed it three times with the identical message, spent
    both replans, and buried the one line that mattered - "install Docker" -
    under two copies of itself. A run that cannot proceed should say so once.
    """
    from kavach.agent.loop import _needs_an_operator
    from kavach.types import Observation

    blocked = Observation(
        step_idx=0, ok=False,
        stderr="SandboxUnavailable: code.exec needs the Docker sandbox and it "
               "is not available.\n  docker present: False",
    )
    assert _needs_an_operator(blocked)


def test_a_model_mistake_is_still_worth_replanning():
    """The narrow half of the same rule. A tool that does not exist, or an
    argument that is wrong, is precisely what the replan budget is for -
    treating those as unfixable would remove the loop's only repair."""
    from kavach.agent.loop import _needs_an_operator
    from kavach.types import Observation

    for stderr in (
        "ToolNotFound: 'fs.appendx' does not exist. Available: fs.write, ...",
        "InvalidArguments: fs.write is missing required argument 'content'",
        "the file was empty",
        "",
    ):
        assert _needs_an_operator(Observation(step_idx=0, ok=False, stderr=stderr)) is None


def test_the_match_is_on_the_class_name_not_the_wording():
    """The message is written for a human and will be reworded; the exception
    class name is an API. Matching prose would make the guard fail silently the
    next time somebody improves the sentence."""
    from kavach.agent.loop import UNFIXABLE_BY_REPLANNING

    for prefix in UNFIXABLE_BY_REPLANNING:
        assert prefix.endswith(":"), prefix
        assert prefix[0].isupper(), f"{prefix} is not an exception class name"

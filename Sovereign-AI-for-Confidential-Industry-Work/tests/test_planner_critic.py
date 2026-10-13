"""Planner validation and the critic (§05).

The planner tests are mostly about ONE failure mode: a model emitting
structurally perfect JSON that calls a tool which does not exist, or passes
arguments that do not match its schema. Constrained decoding does not prevent
this - a grammar can force valid JSON containing a fictional tool name - so the
plan is validated against the real registry before it is accepted.

The critic tests are about the ordering of its two layers. A critic that asks the
model first will accept "I have successfully created the file" as evidence that a
file exists.
"""

from __future__ import annotations

import pytest

from kavach.agent.critic import Critic, Verdict
from kavach.agent.planner import PlanError, _extract_json_array, validate_plan
from kavach.tools.workspace import Workspace
from kavach.types import Observation, Step


def raw_step(**overrides):
    base = {
        "intent": "compute something",
        "tool": "calc",
        "args": '{"expression": "2+2"}',
        "acceptance": "the step produces output",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- extraction

def test_plain_json_array_parses():
    assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]


def test_fenced_json_parses():
    text = 'Here is the plan:\n```json\n[{"a": 1}]\n```\nHope that helps.'
    assert _extract_json_array(text) == [{"a": 1}]


def test_array_surrounded_by_prose_parses():
    """Leniency about SHAPE is safe because every field is validated next."""
    assert _extract_json_array('Sure! [{"a": 1}] Done.') == [{"a": 1}]


def test_no_array_is_an_error():
    with pytest.raises(PlanError, match="no JSON array"):
        _extract_json_array("I would start by reading the file.")


def test_malformed_json_is_an_error():
    with pytest.raises(PlanError, match="invalid JSON"):
        _extract_json_array('[{"a": 1,}]')


# ---------------------------------------------------------------- validation

def test_a_valid_plan_is_accepted():
    steps = validate_plan([raw_step()])
    assert len(steps) == 1
    assert steps[0].tool == "calc"
    assert steps[0].args == {"expression": "2+2"}


def test_an_invented_tool_is_rejected():
    """THE failure mode. A grammar cannot prevent this; only the registry can."""
    with pytest.raises(PlanError, match="does not exist"):
        validate_plan([raw_step(tool="database.query")])


def test_the_rejection_lists_the_real_tools():
    """The message is fed back to the model for its one repair attempt, so it
    has to say what IS available."""
    with pytest.raises(PlanError, match="calc"):
        validate_plan([raw_step(tool="nope")])


def test_arguments_are_validated_against_the_tool_schema():
    with pytest.raises(PlanError, match="missing required"):
        validate_plan([raw_step(tool="fs.write", args='{"path": "a.txt"}')])


def test_unknown_arguments_are_rejected():
    with pytest.raises(PlanError, match="unknown argument"):
        validate_plan([raw_step(args='{"expression": "1+1", "nonsense": true}')])


def test_a_step_without_an_acceptance_test_is_rejected():
    """The critic would have nothing to check, so the step could never be known
    to have worked."""
    with pytest.raises(PlanError, match="no acceptance test"):
        validate_plan([raw_step(acceptance="")])


def test_a_step_without_an_intent_is_rejected():
    with pytest.raises(PlanError, match="no intent"):
        validate_plan([raw_step(intent="")])


def test_an_empty_plan_is_rejected():
    with pytest.raises(PlanError, match="empty"):
        validate_plan([])


def test_an_over_long_plan_is_rejected():
    with pytest.raises(PlanError, match="budget allows"):
        validate_plan([raw_step() for _ in range(30)])


def test_args_may_be_an_object_or_a_string():
    from_string = validate_plan([raw_step(args='{"expression": "1+1"}')])
    from_object = validate_plan([raw_step(args={"expression": "1+1"})])
    assert from_string[0].args == from_object[0].args


def test_malformed_args_string_is_rejected():
    with pytest.raises(PlanError, match="not valid JSON"):
        validate_plan([raw_step(args="{expression: 1+1}")])


def test_steps_are_indexed_in_order():
    steps = validate_plan([raw_step(), raw_step(), raw_step()])
    assert [s.idx for s in steps] == [0, 1, 2]


# ---------------------------------------------------------------- critic

@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_critic")


@pytest.fixture()
def critic():
    return Critic(semantic=False)


def step_with(acceptance: str) -> Step:
    return Step(idx=0, intent="do it", tool="calc", acceptance=acceptance)


def test_a_failed_tool_needs_no_interpretation(critic, ws):
    verdict = critic.judge(
        step_with("anything"),
        Observation(step_idx=0, ok=False, stderr="it broke"),
        ws,
    )
    assert verdict.ok is False
    assert verdict.layer == "trivial"


def test_file_existence_is_checked_on_disk(critic, ws):
    """Never taken from the model's word."""
    verdict = critic.judge(
        step_with("file report.txt exists"),
        Observation(step_idx=0, ok=True, stdout="I have created report.txt"),
        ws,
    )
    assert verdict.ok is False, "the critic believed the model instead of the disk"


def test_file_existence_passes_when_the_file_is_really_there(critic, ws):
    (ws.root / "report.txt").write_text("content", encoding="utf-8")
    verdict = critic.judge(
        step_with("file report.txt exists"),
        Observation(step_idx=0, ok=True),
        ws,
    )
    assert verdict.ok is True


def test_an_empty_file_does_not_count_as_existing(critic, ws):
    (ws.root / "report.txt").write_text("", encoding="utf-8")
    verdict = critic.judge(
        step_with("file report.txt exists"), Observation(step_idx=0, ok=True), ws
    )
    assert verdict.ok is False


def test_exit_code_is_checked(critic, ws):
    bad = critic.judge(
        step_with("the command exits 0"),
        Observation(step_idx=0, ok=True, exit_code=3),
        ws,
    )
    good = critic.judge(
        step_with("the command exits 0"),
        Observation(step_idx=0, ok=True, exit_code=0),
        ws,
    )
    assert bad.ok is False
    assert good.ok is True


def test_non_empty_output_is_checked(critic, ws):
    empty = critic.judge(
        step_with("the step produces output"),
        Observation(step_idx=0, ok=True, stdout="   "),
        ws,
    )
    assert empty.ok is False


def test_required_substring_is_checked(critic, ws):
    verdict = critic.judge(
        step_with('the output contains "Reynolds"'),
        Observation(step_idx=0, ok=True, stdout="the Reynolds number is 182514"),
        ws,
    )
    assert verdict.ok is True


def test_missing_substring_fails(critic, ws):
    verdict = critic.judge(
        step_with('the output contains "Reynolds"'),
        Observation(step_idx=0, ok=True, stdout="nothing relevant here"),
        ws,
    )
    assert verdict.ok is False


def test_multiple_checks_are_all_reported(critic, ws):
    verdict = critic.judge(
        step_with("file out.txt exists and the command exits 0"),
        Observation(step_idx=0, ok=True, exit_code=1),
        ws,
    )
    assert verdict.ok is False
    assert len(verdict.checks) >= 2


def test_an_uncheckable_acceptance_falls_through(critic, ws):
    """With the semantic layer off, nothing mechanical applies -> trivial pass.
    With it on, this is where a model judgement would be requested."""
    verdict = critic.judge(
        step_with("the result looks reasonable"),
        Observation(step_idx=0, ok=True, stdout="something"),
        ws,
    )
    assert verdict.layer == "trivial"


def test_a_broken_critic_fails_closed(ws):
    """An unjudgeable step must get a human's attention, not a silent pass."""
    class Exploding(Critic):
        @property
        def router(self):
            raise RuntimeError("no model available")

    verdict = Exploding(semantic=True).judge(
        step_with("the result looks reasonable"),
        Observation(step_idx=0, ok=True, stdout="something"),
        ws,
    )
    assert verdict.ok is False
    assert "could not evaluate" in verdict.reason


def test_verdict_serialises_for_the_panel():
    payload = Verdict(True, "fine", checks={"a": True}).as_dict()
    assert set(payload) == {"ok", "reason", "layer", "checks"}

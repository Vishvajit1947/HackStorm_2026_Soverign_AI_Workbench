"""critic.py: "X exists and contains 'Y'" is a claim about the file, not stdout.

Found at the last step of the first compute-then-save run that got past both
the sandbox refusal and the plan-time repair: fs.write wrote the file correctly
and returned a write confirmation, the acceptance said the file "contains
'12000.0'", and the mechanical layer searched the confirmation for the value.
Every compute-then-save plan failed at its final step, with the deliverable
sitting correctly on disk.
"""

from __future__ import annotations

import pytest

from kavach.agent.critic import Critic
from kavach.tools.workspace import Workspace
from kavach.types import Observation, Step


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_critic")


@pytest.fixture()
def critic():
    return Critic(semantic=False)


def _step(acceptance: str, tool: str = "fs.write") -> Step:
    return Step(idx=0, intent="save it", tool=tool, acceptance=acceptance)


def _write_confirmation() -> Observation:
    # What fs.write actually returns: where it wrote and how much, never the content.
    return Observation(step_idx=0, ok=True, stdout="wrote 7 bytes to report.txt")


def test_a_value_in_the_file_satisfies_a_file_scoped_contains(critic, ws):
    (ws.root / "report.txt").write_text("1200 kPa", encoding="utf-8")
    verdict = critic.judge(
        _step("report.txt exists and contains '1200'"), _write_confirmation(), ws
    )
    assert verdict.ok is True, verdict.reason


def test_a_value_missing_from_both_file_and_output_still_fails(critic, ws):
    (ws.root / "report.txt").write_text("nothing useful", encoding="utf-8")
    verdict = critic.judge(
        _step("report.txt exists and contains '1200'"), _write_confirmation(), ws
    )
    assert verdict.ok is False
    assert "the file or the output" in verdict.reason


def test_the_file_still_has_to_exist(critic, ws):
    verdict = critic.judge(
        _step("report.txt exists and contains '1200'"), _write_confirmation(), ws
    )
    assert verdict.ok is False
    assert "does not exist" in verdict.reason


def test_output_contains_without_a_file_is_unchanged(critic, ws):
    verdict = critic.judge(
        _step("output contains '182514'", tool="calc"),
        Observation(step_idx=0, ok=True, stdout="Re = 182514"),
        ws,
    )
    assert verdict.ok is True
    verdict = critic.judge(
        _step("output contains '182514'", tool="calc"),
        Observation(step_idx=0, ok=True, stdout="Re = 0"),
        ws,
    )
    assert verdict.ok is False
    assert verdict.reason.startswith("the output does not contain")

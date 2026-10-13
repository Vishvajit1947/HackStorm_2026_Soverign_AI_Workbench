"""critic.py: a successful tool with an empty result has not failed.

fs.list on a fresh workspace returns zero entries with exit code 0. The tool did
exactly its job - the emptiness is a fact about the workspace, not a fault in the
step. The acceptance the planner writes for it ("fs.list returns a list of files
in the workspace") matches no mechanical pattern, so judgement falls to the
semantic layer, which was handed the payload alone: no files in, "FAIL" out.

That cost a replan on a step that was never wrong, and the run then died two
replans later having done nothing incorrect. So the judge is now told what the
tool itself reported, and told to judge whether the step did its job rather than
whether the result was interesting.
"""

from __future__ import annotations

import pytest

from kavach.agent.critic import Critic
from kavach.tools.workspace import Workspace
from kavach.types import Observation, Step


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_empty")


def _step(acceptance: str, tool: str = "fs.list") -> Step:
    return Step(idx=0, intent="see what is here", tool=tool, acceptance=acceptance)


class _Recorder(Critic):
    """Captures the prompt instead of calling a model."""

    def __init__(self):
        super().__init__(semantic=True)
        self.prompt: str | None = None

    @property
    def router(self):
        recorder = self

        class _R:
            def route(self, request, **kw):
                recorder.prompt = request.prompt

                class _O:
                    text = "PASS - the tool ran and returned its result."
                return _O()
        return _R()


def test_the_judge_is_told_whether_the_tool_succeeded(ws):
    c = _Recorder()
    verdict = c.judge(
        _step("fs.list returns a list of files in the workspace"),
        Observation(step_idx=0, ok=True, exit_code=0, data={"count": 0, "entries": []}),
        ws,
    )
    assert c.prompt is not None, "the semantic layer never ran"
    assert "SUCCESS" in c.prompt, "the judge cannot see that the tool succeeded"
    assert "exit code 0" in c.prompt
    assert verdict.ok is True


def test_the_judge_is_told_emptiness_can_be_correct(ws):
    c = _Recorder()
    c.judge(
        _step("fs.list returns a list of files in the workspace"),
        Observation(step_idx=0, ok=True, exit_code=0, data={"count": 0, "entries": []}),
        ws,
    )
    lowered = c.prompt.lower()
    assert "empty" in lowered, "no guidance that an empty result can be correct"
    assert "did its job" in lowered


def test_a_failing_tool_never_reaches_the_judge_at_all(ws):
    """A tool that errored is failed outright, without spending a model call -
    and without giving anything the chance to talk the verdict back up."""
    c = _Recorder()
    verdict = c.judge(
        _step("fs.read returns the contents", tool="fs.read"),
        Observation(step_idx=0, ok=False, exit_code=1, stderr="no such file"),
        ws,
    )
    assert verdict.ok is False
    assert c.prompt is None, "a failed tool was sent to the model for an opinion"


def test_a_mechanical_verdict_still_wins_over_the_judge(ws):
    """The cheap layer decides when it can: a named file that does not exist is
    a failure no model opinion should overturn."""
    verdict = Critic(semantic=False).judge(
        _step("report.txt exists", tool="fs.write"),
        Observation(step_idx=0, ok=True, exit_code=0, stdout="wrote it"),
        ws,
    )
    assert verdict.ok is False
    assert verdict.layer == "mechanical"

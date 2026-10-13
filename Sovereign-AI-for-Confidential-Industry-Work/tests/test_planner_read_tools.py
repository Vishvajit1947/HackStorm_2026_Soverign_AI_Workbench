"""A read-only tool cannot be asked to prove a file exists.

Small models put the DELIVERABLE's acceptance test on the FIRST step: kb.search
is told to prove approval_note.docx exists. The search returns exactly the right
passages, fails anyway because it was never going to create a file, and burns a
replan. Saying so in the planner prompt does not hold at 1.5B - it wrote the same
acceptance again on the very next run - so the rule is enforced at plan time.

Derived from the manifest's declared side effect rather than a list of tool
names, so a READ tool added later is covered the day it lands.
"""

from __future__ import annotations

import pytest

from kavach.agent.planner import PlanError, _expects_a_file, validate_plan


def _step(tool: str, acceptance: str, args: dict) -> dict:
    return {"intent": "do it", "tool": tool, "acceptance": acceptance, "args": args}


@pytest.mark.parametrize("tool,args", [
    ("kb.search", {"query": "pump findings"}),
    ("fs.list", {}),
    ("fs.read", {"path": "notes.txt"}),
])
def test_a_read_tool_cannot_be_asked_to_create_a_file(tool, args):
    with pytest.raises(PlanError, match="only reads"):
        validate_plan([_step(tool, "approval_note.docx is created", args)])


def test_the_refusal_says_where_the_file_check_belongs(tool_hint="writes the file"):
    with pytest.raises(PlanError, match=tool_hint):
        validate_plan([_step("kb.search", "report.docx exists", {"query": "x"})])


def test_a_read_tool_judged_on_what_it_returned_is_fine():
    steps = validate_plan([
        _step("kb.search", "the search returns relevant passages", {"query": "x"})])
    assert steps[0].tool == "kb.search"


def test_a_writing_tool_may_of_course_promise_a_file():
    steps = validate_plan([
        _step("doc.write", "approval_note.docx is created",
              {"path": "approval_note.docx", "body": "text"})])
    assert steps[0].tool == "doc.write"


@pytest.mark.parametrize("acceptance,expected", [
    ("A Word file named approval_note.docx is created", True),
    ("calculations.xlsx exists in the workspace", True),
    ("briefing.pptx is written", True),
    ("result.txt exists", True),
    ("stdout contains 1200", False),
    ("the search returned three passages", False),
])
def test_deliverable_extensions_are_recognised_as_files(acceptance, expected):
    """The check listed txt/csv/json but not docx/xlsx/pptx, so every acceptance
    naming a real deliverable slipped past it - including the code.exec guard."""
    assert _expects_a_file(acceptance) is expected, acceptance

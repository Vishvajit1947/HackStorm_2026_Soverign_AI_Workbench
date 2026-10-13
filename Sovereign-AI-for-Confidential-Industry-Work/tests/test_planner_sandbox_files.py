"""planner.py: a code.exec step that expects a file can never pass.

The sandbox's working directory is a tmpfs wiped on exit and nothing is copied
out, so "run code, then check the file exists" fails on every attempt however
good the code is. Found by watching a run print the correct answer (1200 kPa)
and still fail, then burn both replans before it reached the fs.write step that
would have worked. Refusing the shape at plan time costs nothing.
"""

from __future__ import annotations

import pytest

from kavach.agent.planner import PlanError, validate_plan


def _step(tool: str, acceptance: str, args: dict) -> dict:
    return {"intent": "do the thing", "tool": tool, "acceptance": acceptance, "args": args}


def test_code_exec_that_expects_a_file_is_refused_at_plan_time():
    with pytest.raises(PlanError, match="fs.write"):
        validate_plan([_step("code.exec", "pressure_kpa.txt exists in the workspace",
                             {"code": "print(1200)"})])


def test_the_refusal_names_the_split_the_model_should_make():
    with pytest.raises(PlanError, match="checks stdout"):
        validate_plan([_step("code.exec", "result saved in the workspace",
                             {"code": "print(1200)"})])


def test_code_exec_that_checks_stdout_is_accepted():
    steps = validate_plan([_step("code.exec", "stdout contains 1200",
                                 {"code": "print(1200)"})])
    assert steps[0].tool == "code.exec"


def test_fs_write_that_expects_a_file_is_still_fine():
    steps = validate_plan([_step("fs.write", "pressure_kpa.txt exists",
                                 {"path": "pressure_kpa.txt", "content": "1200"})])
    assert steps[0].tool == "fs.write"


def test_the_refusal_names_doc_write_for_office_files():
    """Telling a model to use fs.write for a .pptx sends it round the same loop
    twice: fs.write writes UTF-8 text, so the plan is refused again and the run
    dies on advice that could never have worked."""
    with pytest.raises(PlanError, match="doc.write"):
        validate_plan([_step("code.exec", "A file named 'report.pptx' exists",
                             {"code": "print(1)"})])


@pytest.mark.parametrize("acceptance", [
    "approval_note.docx is created",
    "calculations.xlsx exists in the workspace",
    "briefing.pptx is written",
])
def test_every_office_format_points_at_doc_write(acceptance):
    with pytest.raises(PlanError, match="doc.write"):
        validate_plan([_step("code.exec", acceptance, {"code": "print(1)"})])


def test_plain_text_still_points_at_fs_write():
    """doc.write cannot produce a .txt, so the advice must stay correct both
    ways rather than naming the newest tool for everything."""
    with pytest.raises(PlanError, match="fs.write"):
        validate_plan([_step("code.exec", "result.txt exists",
                             {"code": "print(1)"})])


# ------------------------------------------------------- a format that exists

def test_doc_write_is_refused_a_pdf_at_plan_time():
    """There is no PDF writer on this box, and doc.write says so - but only once
    the step RUNS, which on a real backend is minutes after the search step that
    precedes it. Found on a run that spent ten minutes reaching a deliverable
    that was never producible: `company_details.pdf is created in the workspace`.
    """
    with pytest.raises(PlanError, match="no PDF writer"):
        validate_plan([_step("doc.write", "the report is written",
                             {"path": "company_details.pdf", "body": "x"})])


def test_the_pdf_refusal_names_the_formats_that_do_exist():
    """A refusal a model cannot act on costs a replan and arrives at the same
    dead end. It has to name the way out."""
    with pytest.raises(PlanError) as exc:
        validate_plan([_step("doc.write", "the report is written",
                             {"path": "report.pdf", "body": "x"})])
    for fmt in (".docx", ".xlsx", ".pptx"):
        assert fmt in str(exc.value)


@pytest.mark.parametrize("path", ["note.docx", "sheet.xlsx", "deck.pptx"])
def test_every_format_the_tool_can_write_is_accepted(path):
    steps = validate_plan([_step("doc.write", f"{path} exists", {"path": path, "body": "x"})])
    assert steps[0].tool == "doc.write"


def test_a_doc_write_step_with_no_path_is_left_to_the_schema_check():
    """Refusing it here would report a missing format for an argument that is
    missing entirely, which sends the model to fix the wrong thing."""
    with pytest.raises(PlanError, match="missing required argument"):
        validate_plan([_step("doc.write", "something is written", {"body": "x"})])

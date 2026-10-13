"""A defect with one correct repair is repaired, not bounced back to the model.

A 1.7B planner put the deliverable's acceptance test on the kb.search step -
"A Word file named 'report.txt' is created." - and wrote it again on the retry
after being told exactly what was wrong. Two attempts, both rejected, and a run
died over one sentence while every tool, argument and ordering in the plan was
correct.

There is only one way to judge a search, so the system can write that sentence
itself. What it must not do is hide that it did: every repair is reported on the
PlanResult, in the API payload and in the audit chain.

Defects with more than one possible fix stay rejected, because guessing there
would be inventing intent.
"""

from __future__ import annotations

import pytest

from kavach.agent.planner import PlanError, repair_plan, validate_plan


def _step(tool: str, acceptance: str, args: dict) -> dict:
    return {"intent": "do it", "tool": tool, "acceptance": acceptance, "args": args}


SEARCH = {"query": "pump inspection findings"}


@pytest.mark.parametrize("tool,args", [
    ("kb.search", SEARCH),
    ("fs.list", {}),
    ("fs.read", {"path": "notes.txt"}),
])
def test_a_read_step_judged_by_a_file_is_repaired_rather_than_refused(tool, args):
    steps, notes = repair_plan([_step(tool, "approval_note.docx is created", args)])
    assert notes, "a silent repair is the thing this must never be"
    # The repaired plan must now survive the very validator that rejected it.
    validated = validate_plan(steps)
    assert validated[0].tool == tool
    assert ".docx" not in validated[0].acceptance


def test_the_repair_note_quotes_what_the_model_actually_wrote():
    """Without the original text the audit record cannot show what was changed,
    and a reviewer has no way to tell a repair from a hallucination."""
    original = "A Word file named 'report.txt' is created."
    _, notes = repair_plan([_step("kb.search", original, SEARCH)])
    assert original in notes[0]
    assert "kb.search" in notes[0]


def test_a_writing_step_keeps_its_file_test():
    """doc.write really does create the file, so its acceptance is correct and
    rewriting it would destroy the only check that proves the deliverable."""
    steps, notes = repair_plan([
        _step("doc.write", "approval_note.docx is created",
              {"path": "approval_note.docx", "body": "text"})])
    assert notes == []
    assert steps[0]["acceptance"] == "approval_note.docx is created"


def test_a_sound_plan_is_returned_untouched():
    plan = [
        _step("kb.search", "the search returns relevant passages", SEARCH),
        _step("doc.write", "approval_note.docx is created",
              {"path": "approval_note.docx", "body": "{{step0}}"}),
    ]
    steps, notes = repair_plan(plan)
    assert notes == []
    assert steps == plan


def test_the_search_to_document_plan_that_failed_now_survives_end_to_end():
    """The exact shape the FAST profile emitted for the MRPL demo goal."""
    steps, notes = repair_plan([
        _step("kb.search", "A Word file named 'report.txt' is created.", SEARCH),
        _step("doc.write", "approval_note.docx is created",
              {"path": "approval_note.docx", "body": "Findings:\n\n{{step0}}"}),
    ])
    validated = validate_plan(steps)
    assert len(notes) == 1
    assert [s.tool for s in validated] == ["kb.search", "doc.write"]


def test_a_code_exec_step_judged_by_a_file_has_its_acceptance_rewritten():
    """This used to go back to the model, on the grounds that the plan needs
    SPLITTING in two and where to split it is the model's call. Watching a 1.7B
    burn its last repair on it changed that: the split is the model's call, but
    the acceptance is not a judgement call at all. A code.exec step can never be
    judged by a file, whatever was intended, because the sandbox is a tmpfs
    destroyed with the container - so rewriting it is not guessing intent.

    Only the acceptance. Inventing the second step would be guessing, and the
    note says plainly that the file still needs a writer.
    """
    steps, notes = repair_plan([
        _step("code.exec", "results.csv is created", {"code": "print(1)"})])
    assert len(notes) == 1
    assert "stdout" in notes[0]
    assert "fs.write or doc.write" in notes[0]
    validated = validate_plan(steps)          # and it now survives validation
    assert validated[0].acceptance == "the program runs and prints its result to stdout"


def test_the_repair_does_not_invent_the_step_that_writes_the_file():
    """The other half of the advice stays with the model. A plan that grew a
    step nobody wrote is a plan its author cannot recognise."""
    steps, _ = repair_plan([
        _step("code.exec", "results.csv is created", {"code": "print(1)"})])
    assert len(steps) == 1


def test_a_code_exec_step_already_judged_by_stdout_is_left_alone():
    steps, notes = repair_plan([
        _step("code.exec", "stdout contains 1200", {"code": "print(1200)"})])
    assert notes == []
    assert steps[0]["acceptance"] == "stdout contains 1200"


def test_an_invented_tool_is_still_refused():
    steps, _ = repair_plan([_step("pump.inspect", "it ran", {})])
    with pytest.raises(PlanError, match="does not exist"):
        validate_plan(steps)


def test_a_read_step_with_no_acceptance_is_not_quietly_given_one():
    """An absent acceptance test means the model never decided how to judge the
    step. Inventing one would manufacture a pass out of nothing."""
    steps, notes = repair_plan([_step("kb.search", "", SEARCH)])
    assert notes == []
    with pytest.raises(PlanError, match="no acceptance test"):
        validate_plan(steps)


def test_a_malformed_step_survives_repair_for_the_validator_to_report():
    """repair_plan must not crash on garbage; validate_plan owns that message."""
    steps, notes = repair_plan(["not a step", 42, None])
    assert notes == []
    with pytest.raises(PlanError, match="not an object"):
        validate_plan(steps)


# ------------------------------------------------- a format this box can write

def test_a_pdf_is_rewritten_as_a_docx_rather_than_killing_the_run():
    """There is no PDF writer on this box, and a 1.7B does not stop asking for
    one: told in the prompt, refused at plan time with the three formats named,
    it produced .docx on the next attempt and asked for .pdf again on the one
    after - exhausting the repair budget on a run that had already written a
    perfectly good document.

    The content is the model's; only the container changes, which is the only
    part that was impossible.
    """
    steps, notes = repair_plan([
        _step("doc.write", "summary.pdf is created in the workspace",
              {"path": "summary.pdf", "body": "Findings: ..."})])
    assert steps[0]["args"]["path"] == "summary.docx"
    assert steps[0]["args"]["body"] == "Findings: ..."
    assert len(notes) == 1 and "no .pdf writer" in notes[0]
    validate_plan(steps)


def test_the_acceptance_test_follows_the_file_it_names():
    """Renaming the file and leaving the acceptance naming the old one swaps a
    plan-time refusal for a step that can never pass."""
    steps, _ = repair_plan([
        _step("doc.write", "report.pdf is created in the workspace",
              {"path": "report.pdf", "body": "x"})])
    assert "report.docx" in steps[0]["acceptance"]
    assert "report.pdf" not in steps[0]["acceptance"]


@pytest.mark.parametrize(("asked", "written"), [
    (".pdf", ".docx"), (".doc", ".docx"), (".rtf", ".docx"),
    (".csv", ".xlsx"), (".xls", ".xlsx"), (".ppt", ".pptx"),
])
def test_each_substitute_keeps_the_shape_of_what_was_asked_for(asked, written):
    """A mapping, not a guess: a PDF and a .docx are both a document somebody
    reads; a .csv and a .xlsx are both a grid of values. A spreadsheet must
    never be delivered as prose."""
    steps, _ = repair_plan([
        _step("doc.write", "it is written", {"path": f"out{asked}", "body": "x"})])
    assert steps[0]["args"]["path"] == f"out{written}"


def test_a_text_file_is_not_rewritten_because_it_needs_a_different_TOOL():
    """`.txt` is fs.write's job. Changing the container would be wrong and
    changing the tool would be inventing intent, so this still goes back to the
    model with the refusal that names fs.write."""
    steps, notes = repair_plan([
        _step("doc.write", "notes.txt is created", {"path": "notes.txt", "body": "x"})])
    assert notes == []
    with pytest.raises(PlanError, match="fs.write"):
        validate_plan(steps)


def test_a_format_the_tool_already_writes_is_untouched():
    steps, notes = repair_plan([
        _step("doc.write", "note.docx is created", {"path": "note.docx", "body": "x"})])
    assert notes == []
    assert steps[0]["args"]["path"] == "note.docx"


def test_args_arriving_as_a_string_of_json_do_not_crash_the_repair():
    """Small models emit `args` as a STRING of JSON constantly - validate_plan
    parses it, so repair_plan runs BEFORE it is a dict. Reading it as one raised
    AttributeError, which surfaced as a 500 from the run endpoint rather than as
    a plan error anybody could act on."""
    steps, notes = repair_plan([{
        "intent": "write it", "tool": "doc.write",
        "acceptance": "summary.pdf is created",
        "args": '{"path": "summary.pdf", "body": "x"}',
    }])
    assert steps[0]["args"]["path"] == "summary.docx"
    assert len(notes) == 1


def test_args_that_are_not_valid_json_are_left_for_the_validator():
    """It reports the parse error with the position; guessing here would bury
    it."""
    steps, notes = repair_plan([{
        "intent": "write it", "tool": "doc.write",
        "acceptance": "x.pdf is created", "args": "{not json",
    }])
    assert notes == []
    with pytest.raises(PlanError, match="not valid JSON"):
        validate_plan(steps)

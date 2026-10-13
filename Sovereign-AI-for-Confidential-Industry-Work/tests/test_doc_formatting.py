"""Formatted deliverables, and the promise that content cannot destroy one.

Two separate claims are tested here.

THE DOCUMENT IS TYPESET. Headings are heading styles, numbered steps are a Word
numbered list, a formula is monospaced and evaluated where it can be, a column
heading is something a person would circulate. Every one of these is checked by
opening the file with the library that reads that format - a .docx with the
right magic bytes and one wall of 11pt prose is not an approval note.

NOTHING IN THE BODY CAN COST THE RUN ITS DELIVERABLE. This is the stronger
claim and most of the file. A deliverable is the last step of a long run: the
corpus was searched, a model was loaded, a plan was judged. Losing all of that
to a malformed table, an unbalanced bracket or a symbolic equation is the worst
bargain in the system. So every hostile body below must still produce a file
that opens - and must still SAY what did not render, because a document that
quietly dropped a section is worse than one that shows it as text.
"""

from __future__ import annotations

import json

import pytest

from kavach.tools.impl.deliverables import clean_filename, doc_write
from kavach.tools.impl.docmodel import (
    BULLETS,
    FIELDS,
    FORMULA,
    HEADING,
    STEPS,
    TABLE,
    parse_body,
    parse_rows,
    render_formula,
)
from kavach.tools.workspace import Workspace


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_fmt")


def _paragraphs(path):
    from docx import Document

    return [(p.style.name, p.text) for p in Document(path).paragraphs]


# ----------------------------------------------------------------- the model


def test_every_markup_form_becomes_its_own_block():
    body = (
        "# Findings\n"
        "Equipment: P-101A\n"
        "- below t-min\n"
        "1. isolate the line\n"
        "| tag | mm |\n| --- | --- |\n| 6-P-1203 | 8.89 |\n"
        "$$ dP = 2*3 $$\n"
        "> WARNING: do not restart\n"
    )
    kinds = [b.kind for b in parse_body(body)[0]]
    for expected in (HEADING, FIELDS, BULLETS, STEPS, TABLE, FORMULA, "note"):
        assert expected in kinds, f"{expected} was not recognised; got {kinds}"


def test_a_bare_equation_is_a_formula_and_a_labelled_value_is_not():
    """Models write `dP = f*(L/D)*rho*v^2/2` far more often than they remember a
    delimiter. But `Prepared by: R. Kumar` is a field, and rendering it as an
    equation would put a plant engineer's name in a formula box."""
    blocks = parse_body("dP = 2*3.5\nPrepared by: R. Kumar\n")[0]
    kinds = [b.kind for b in blocks]
    assert FORMULA in kinds
    assert FIELDS in kinds


def test_the_trailing_colon_heading_convention_still_works():
    """`Background:` on its own line has meant a heading since Phase 4 and every
    plan written against the manifest uses it. A markup upgrade that breaks the
    plans written for it is a breaking change wearing an improvement."""
    blocks = parse_body("Background:\n\nThe pump was inspected.")[0]
    assert blocks[0].kind == HEADING
    assert blocks[0].text == "Background"


def test_steps_are_renumbered_rather_than_trusted():
    """Small models write `1. 2. 2. 4.`. Printed verbatim that reads as a
    mistake in the deliverable, and nobody can tell whether a step is missing."""
    blocks = parse_body("1. one\n2. two\n2. three\n4. four\n")[0]
    steps = [b for b in blocks if b.kind == STEPS]
    assert steps and steps[0].items == ["one", "two", "three", "four"]


# -------------------------------------------------------------- formulas


def test_a_formula_with_numbers_is_evaluated_and_shows_its_value():
    out = render_formula("dP = 2 * 3.5")
    assert out["value"] == pytest.approx(7.0)


def test_engineering_caret_notation_is_understood():
    """`v^2` is v squared to every engineer alive. Python reads it as XOR."""
    assert render_formula("E = 3^2")["value"] == pytest.approx(9.0)


def test_a_symbolic_relation_is_not_an_error():
    """`dP = f*(L/D)*rho*v^2/2` with no numbers is a relation, and a relation is
    a perfectly good thing to put in a report. It must be typeset as written,
    with a note saying why there is no number - never refused, and never given
    an invented value."""
    out = render_formula("dP = f*(L/D)*rho*v^2/2")
    assert out["value"] is None
    assert out["note"] and "relation" in out["note"]
    assert out["expression"]


@pytest.mark.parametrize("expression", [
    "x = ",
    "= = =",
    "dP = (2 * (3 + ",
    "sqrt(-1)",
    "1/0",
    "x = 10**10**10",
    "ρ * v^2 / 2",
    "x = __import__('os').system('echo hi')",
    "f(x) = integral of y dx",
    "A = pi*r^2 where r is unknown",
    "" ,
    "   ",
    "=SUM(A1:A9)",
    "x" * 5000,
])
def test_no_formula_of_any_shape_can_raise(expression):
    """The user-facing promise: a mathematics or engineering expression, however
    written, never produces an error. It is evaluated when it can be and
    typeset as written when it cannot, with the reason attached."""
    out = render_formula(expression)
    assert isinstance(out, dict)
    assert "expression" in out
    assert out["value"] is None or isinstance(out["value"], (int, float))
    if out["value"] is None:
        assert out["note"], f"no value AND no explanation for {expression!r}"


def test_an_injection_dressed_as_a_formula_is_refused_not_executed():
    """calc runs UNSANDBOXED - its side effect is COMPUTE. A document body is
    untrusted text, so the same AST allowlist that protects the tool has to
    protect the typesetter. Refused WITH a reason, never executed, and never a
    crash that loses the document."""
    out = render_formula("__import__('os').system('echo pwned')")
    assert out["value"] is None
    assert out["note"]


def test_a_formula_lands_in_the_word_file_with_its_value(ws):
    doc_write(ws, "calc.docx", title="Pressure drop",
              body="Working:\n\n$$ dP = 2 * 3.5 $$\n")
    text = "\n".join(t for _s, t in _paragraphs(ws.root / "calc.docx"))
    assert "dP = 2 * 3.5" in text
    assert "7" in text, "the computed value is not in the document"


def test_an_unevaluable_formula_still_lands_in_the_word_file(ws):
    """The whole point: the document exists either way."""
    doc_write(ws, "rel.docx", title="Relation",
              body="$$ dP = f*(L/D)*rho*v^2/2 $$\n")
    text = "\n".join(t for _s, t in _paragraphs(ws.root / "rel.docx"))
    assert "f*(L/D)*rho*v^2/2" in text
    assert "relation" in text.lower(), "the reader is not told why there is no number"


# ------------------------------------------------------------------ JSON


def test_a_json_list_of_objects_becomes_a_table_with_readable_headings():
    """A model that has just read a tool result hands the list straight through.
    The old writer turned it into one paragraph of braces."""
    body = json.dumps([
        {"tag": "6-P-1203", "thickness_mm": 8.89},
        {"tag": "6-P-1204", "thickness_mm": 9.12, "note": "recheck"},
    ])
    blocks = parse_body(body)[0]
    assert blocks[0].kind == TABLE
    assert blocks[0].rows[0] == ["Tag", "Thickness Mm", "Note"], blocks[0].rows[0]


def test_a_ragged_json_list_keeps_every_column():
    """Taking the first record's keys silently drops the columns only later
    records carry - and a dropped column in an inspection sheet is a dropped
    finding."""
    body = json.dumps([{"a": 1}, {"a": 2, "b": 3}])
    rows = parse_rows(body)[0]
    assert rows[0] == ["A", "B"]


def test_a_json_object_becomes_labelled_fields():
    blocks = parse_body(json.dumps({"equipment": "P-101A", "status": "ok"}))[0]
    assert blocks[0].kind == FIELDS
    assert ["Equipment", "P-101A"] in blocks[0].rows


def test_json_that_has_no_document_shape_is_shown_rather_than_guessed_at():
    """Inventing a layout for an arbitrary nesting produces a confident-looking
    document about nothing."""
    blocks, warnings = parse_body(json.dumps({"a": {"b": {"c": [1, 2]}}}))
    assert blocks
    assert any("shown" in w or "lay out" in w for w in warnings) or blocks[0].kind == FIELDS


# ------------------------------------------------------------------ Excel


def test_a_cell_written_as_a_formula_stays_a_live_formula(ws):
    """A calculation sheet that shows its own arithmetic as text is not a
    calculation sheet."""
    from openpyxl import load_workbook

    doc_write(ws, "sheet.xlsx", title="Totals",
              body="item,qty,unit,total\npipe,3,10,=B2*C2\n")
    sheet = load_workbook(ws.root / "sheet.xlsx").active
    assert sheet["D2"].value == "=B2*C2"


def test_a_tag_number_is_not_turned_into_something_else(ws):
    """`007` is a tag, not the number seven. Excel mangling identifiers is the
    classic way a plant spreadsheet goes wrong."""
    from openpyxl import load_workbook

    doc_write(ws, "tags.xlsx", body="tag,reading\n007,1.5\n6-P-1203,2.5\n")
    sheet = load_workbook(ws.root / "tags.xlsx").active
    assert sheet["A2"].value == "007"
    assert sheet["A3"].value == "6-P-1203"


def test_the_sheet_is_usable_not_merely_present(ws):
    from openpyxl import load_workbook

    doc_write(ws, "survey.xlsx", title="Wall thickness survey",
              body="tag,thickness_mm\n" + "".join(f"6-P-{i},8.{i}\n" for i in range(20)))
    sheet = load_workbook(ws.root / "survey.xlsx").active
    assert sheet.freeze_panes == "A2", "a 20-row survey scrolls its header away"
    assert sheet.auto_filter.ref, "no filter on a survey nobody can sort"
    assert sheet.column_dimensions["A"].width > 8, "columns are not sized to content"


def test_an_illegal_sheet_name_is_cleaned_rather_than_raising(ws):
    """openpyxl raises on `: \\ / ? * [ ]` and on anything over 31 characters. A
    calculation sheet lost because its title contained a slash is exactly the
    failure this module exists to prevent."""
    from openpyxl import load_workbook

    out = doc_write(ws, "s.xlsx", title="P-101A: suction / discharge [rev 3] survey",
                    body="a,b\n1,2\n")
    sheet = load_workbook(ws.root / "s.xlsx").active
    assert len(sheet.title) <= 31
    assert out["sheet"] == sheet.title


# ---------------------------------------------------- nothing loses the file


HOSTILE_BODIES = [
    "",
    "   \n\n  \n",
    "| unclosed | table\n| a | b |",
    "```\nnever closed",
    "#" * 200,
    "- " * 3000,
    "| " * 500 + "|",
    "1. " + "x" * 10000,
    "$$ $$",
    "$$ (((((( $$",
    "Label: " + "v" * 9000,
    "\x00\x01\x02 control characters",
    "ρ Δ °C non-ASCII engineering text",
    json.dumps({"deeply": {"nested": {"and": {"pointless": [1, 2, 3]}}}}),
    "[1, 2, {\"a\": 3}]",
    "{not really json}",
    "\n".join(f"{i}. step {i}" for i in range(500)),
]


@pytest.mark.parametrize("body", HOSTILE_BODIES)
@pytest.mark.parametrize("extension", [".docx", ".xlsx", ".pptx"])
def test_no_body_can_stop_a_deliverable_from_being_produced(ws, body, extension):
    """The promise, across every format and every shape of input.

    `doc.write` is the last step of a long run. If it can be made to raise by
    its own content, every earlier step is thrown away over a line of markup.
    """
    out = doc_write(ws, f"out{extension}", title="Test", body=body)
    assert out["size_bytes"] > 0
    assert (ws.root / f"out{extension}").is_file()


@pytest.mark.parametrize("extension", [".docx", ".xlsx", ".pptx"])
def test_the_file_that_survives_is_one_its_own_library_can_open(ws, extension):
    """Producing bytes is not the claim; producing a document is."""
    readers = {
        ".docx": lambda p: __import__("docx").Document(str(p)),
        ".xlsx": lambda p: __import__("openpyxl").load_workbook(str(p)),
        ".pptx": lambda p: __import__("pptx").Presentation(str(p)),
    }
    doc_write(ws, f"hostile{extension}", title="Test",
              body="| broken\n```\n$$ ((( $$\n" + "Δ" * 50)
    readers[extension](ws.root / f"hostile{extension}")


def test_a_degraded_render_is_reported_and_not_hidden(ws):
    """A document that quietly dropped a section is worse than one that shows it
    as text and says why."""
    out = doc_write(ws, "warn.docx", title="T", body="```\nunclosed fence")
    assert out["warnings"], "a degraded render reported nothing"
    text = "\n".join(t for _s, t in _paragraphs(ws.root / "warn.docx"))
    assert "Rendering notes" in text, "the caveat is not in the document itself"


def test_a_clean_document_carries_no_rendering_notes(ws):
    """The counterpart: a caveat section on every document is a caveat nobody
    reads on the one that needs it."""
    out = doc_write(ws, "clean.docx", title="T", body="# Findings\n\nAll good.")
    assert out["warnings"] == []
    text = "\n".join(t for _s, t in _paragraphs(ws.root / "clean.docx"))
    assert "Rendering notes" not in text


# ------------------------------------------------------------- file names


@pytest.mark.parametrize("given,expected", [
    ("P-101A: inspection.docx", "P-101A- inspection.docx".replace(" ", "_")),
    ("note?.docx", "note-.docx"),
    ("a/b/plain.docx", "a/b/plain.docx"),
    ("already_fine.xlsx", "already_fine.xlsx"),
])
def test_a_filename_the_filesystem_would_refuse_is_cleaned(given, expected):
    """Windows refuses `< > : " | ? *`. A model naming a deliverable
    `P-101A: inspection.docx` is being perfectly reasonable and would otherwise
    get an OSError at the very end of the run."""
    assert clean_filename(given)[0] == expected


def test_cleaning_a_filename_never_touches_the_directory():
    """The workspace layout a plan asked for is the layout it gets."""
    path, _ = clean_filename("reports/2026/note:1.docx")
    assert path.startswith("reports/2026/")


def test_a_cleaned_filename_is_reported(ws):
    out = doc_write(ws, "P-101A: note.docx", title="T", body="x")
    assert out.get("renamed") is True
    assert any("file name" in w for w in out["warnings"])
    assert ":" not in out["path"]

"""Real deliverables: Word, Excel and PowerPoint (SS 26117).

The brief asks for approval notes, board packs and calculation sheets - "real
deliverables ... not just chat replies". A .txt file is not an approval note
anyone can sign, and neither is a .docx of undifferentiated 11pt paragraphs. So
this module is a typesetter, not a dumper: one document model (`docmodel.py`),
three renderers, and a house style that makes a heading look like a heading and
a formula look like a formula.

Four deliberate choices:

  The FORMAT COMES FROM THE EXTENSION. One tool, three formats, no `format`
  argument for a model to get wrong - it already had to name the file.

  The SCHEMA IS FLAT: three strings. Nested tool arguments are where small
  models start failing (§05), and the whole point of this box is that a 1.5B
  can drive it. Structure lives INSIDE `body`, in a markup a small model can
  type, or as JSON it can pass straight through from a tool result.

  A DELIVERABLE IS NEVER LOST TO ITS OWN CONTENT. Every block is rendered
  inside a guard. A table that will not lay out becomes its own text, a formula
  that will not evaluate is typeset as written, and the document is still
  produced. This is the last step of a long run - trading a searched corpus, a
  loaded model and a judged plan for one malformed line of markup is the worst
  bargain in the system.

  BUT THE DEGRADATION IS ON THE PAGE. Anything that did not render as intended
  is listed in `warnings`, returned to the agent, and printed into the document
  under "Rendering notes". A document that quietly dropped a table is worse
  than one that shows it as text and says so.

Every path goes through Workspace.resolve, so a deliverable cannot be written
outside the run's workspace any more than fs.write can.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..workspace import Workspace
from .docmodel import (
    BULLETS,
    CODE,
    FIELDS,
    FORMULA,
    HEADING,
    NOTE,
    PARAGRAPH,
    STEPS,
    TABLE,
    Block,
    parse_body,
    parse_rows,
    render_formula,
    scrub,
)

SUPPORTED = {".docx", ".xlsx", ".pptx"}

# ---------------------------------------------------------------- house style
#
# Fonts are named, never embedded - the file only carries the name, and Word
# resolves it on the machine that opens it. These three are present on every
# Windows and Office install this will ever be opened on, which is the whole
# requirement: a deliverable that renders in a substituted font on the plant's
# PC is a deliverable whose layout nobody can predict.
FONT_BODY = "Calibri"
FONT_HEAD = "Calibri Light"
FONT_MONO = "Consolas"

# The workbench palette, darkened for paper. Amber on a screen is an accent; on
# a laser printer at 600dpi it is a pale smudge, so headings and rules use the
# darker amber-700 and body text is near-black rather than the UI's #FAFAFA.
INK = "18181B"
INK_SOFT = "52525B"
ACCENT = "B45309"
ACCENT_SOFT = "FDF4E3"
RULE = "D8D8D2"
NOTE_TONE = {
    "NOTE": ("1D4ED8", "EFF4FE"),
    "IMPORTANT": ("1D4ED8", "EFF4FE"),
    "WARNING": ("B45309", "FDF4E3"),
    "CAUTION": ("B45309", "FDF4E3"),
    "DANGER": ("B91C1C", "FEF0F0"),
}


class DeliverableError(RuntimeError):
    """The deliverable could not be produced. Never a half-written file."""


def _require(module: str, package: str):
    try:
        return __import__(module)
    except ImportError as exc:                       # pragma: no cover - env
        raise DeliverableError(
            f"{package} is not installed, so this box cannot produce that "
            f"format. Install it with: pip install -e \".[deliverables]\" "
            f"(offline: pip install --no-index --find-links wheels {package})"
        ) from exc


def _maybe_number(cell: str) -> Any:
    """A calculation sheet whose numbers are text is not a calculation sheet -
    the reader cannot sum a column of strings.

    Leading zeros and `+` prefixes are left alone: `007` is a tag number and
    `P-101A` is equipment, and Excel turning either into something else is the
    classic way a plant spreadsheet goes wrong.
    """
    text = (cell or "").strip()
    if not text or (len(text) > 1 and text[0] == "0" and text[1] not in ".eE"):
        return cell
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return cell


# ------------------------------------------------------------------ Word

def _docx_styles(doc) -> None:
    """Set the house style once, on the document's own styles.

    Styling every run individually produces a file whose formatting cannot be
    changed afterwards - and the first thing anyone does with an approval note
    is adjust it. Setting the STYLES means the reader can restyle the whole
    document from Word's own gallery, which is what makes this a document
    rather than a picture of one.
    """
    from docx.enum.text import WD_LINE_SPACING
    from docx.shared import Pt, RGBColor

    normal = doc.styles["Normal"]
    normal.font.name = FONT_BODY
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.15

    for name, size, colour, bold in (
        ("Title", 26, INK, False),
        ("Heading 1", 16, ACCENT, False),
        ("Heading 2", 13, INK, True),
        ("Heading 3", 11.5, INK_SOFT, True),
    ):
        try:
            style = doc.styles[name]
        except KeyError:                              # pragma: no cover - env
            continue
        style.font.name = FONT_HEAD if name != "Heading 3" else FONT_BODY
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor.from_string(colour)
        style.paragraph_format.space_before = Pt(16 if name.startswith("Heading") else 0)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True


def _docx_rule(paragraph, colour: str = RULE, size: int = 6) -> None:
    """A bottom border on a paragraph. python-docx has no API for it, so this
    is the raw OOXML - isolated here so nothing else has to know that."""
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), colour)
    borders.append(bottom)
    paragraph._p.get_or_add_pPr().append(borders)


def _docx_shade(cell, colour: str) -> None:
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), colour)
    cell._tc.get_or_add_tcPr().append(shading)


def _docx_block(doc, block: Block, warnings: list[str]) -> None:
    from docx.shared import Pt, RGBColor

    if block.kind == HEADING:
        level = min(max(block.level, 1), 3)
        paragraph = doc.add_heading(block.text, level=level)
        if level == 1:
            # A rule under every H1. On a printed note this is what separates
            # "Findings" from the paragraph above it once the colour is gone.
            _docx_rule(paragraph, ACCENT, 8)
        return

    if block.kind == PARAGRAPH:
        doc.add_paragraph(block.text)
        return

    if block.kind == BULLETS:
        for item in block.items:
            doc.add_paragraph(item, style="List Bullet")
        return

    if block.kind == STEPS:
        # Renumbered by Word rather than by the model. A plan that writes
        # "1. 2. 2. 4." - which small models do - reads as a mistake in the
        # deliverable, and the reader cannot tell whether a step is missing.
        for item in block.items:
            doc.add_paragraph(item, style="List Number")
        return

    if block.kind == FIELDS:
        table = doc.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        for label, value in block.rows:
            cells = table.add_row().cells
            cells[0].text = ""
            run = cells[0].paragraphs[0].add_run(label)
            run.bold = True
            run.font.color.rgb = RGBColor.from_string(INK_SOFT)
            _docx_shade(cells[0], ACCENT_SOFT)
            cells[1].text = value
        doc.add_paragraph()
        return

    if block.kind == TABLE:
        rows = [r for r in block.rows if r]
        if not rows:
            return
        width = max(len(r) for r in rows)
        table = doc.add_table(rows=0, cols=width)
        table.style = "Table Grid"
        for index, row in enumerate(rows):
            cells = table.add_row().cells
            for position in range(width):
                text = row[position] if position < len(row) else ""
                cells[position].text = ""
                run = cells[position].paragraphs[0].add_run(text)
                if index == 0 and block.meta.get("header", True):
                    run.bold = True
                    run.font.color.rgb = RGBColor.from_string(ACCENT)
                    _docx_shade(cells[position], ACCENT_SOFT)
        doc.add_paragraph()
        return

    if block.kind == FORMULA:
        rendered = render_formula(block.text, block.meta.get("substitutions"))
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Pt(18)
        paragraph.paragraph_format.space_before = Pt(6)
        run = paragraph.add_run(rendered["expression"])
        run.font.name = FONT_MONO
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor.from_string(INK)
        tail = None
        if rendered["value"] is not None:
            tail = f"   =  {rendered['value']:g}"
        elif rendered["simplified"]:
            # The simplified relation is a better line in a report than the one
            # that was typed, and showing it is not a claim that it has a value.
            tail = f"   =  {rendered['simplified']}"
        if tail:
            value = paragraph.add_run(tail)
            value.font.name = FONT_MONO
            value.font.size = Pt(11)
            value.bold = True
            value.font.color.rgb = RGBColor.from_string(ACCENT)
        _docx_rule(paragraph, RULE, 4)
        if rendered["note"]:
            # The working, or the reason there is none, in the document. A
            # number with no derivation is not checkable, and in this domain an
            # uncheckable number is worse than no answer.
            caption = doc.add_paragraph()
            caption.paragraph_format.left_indent = Pt(18)
            note = caption.add_run(rendered["note"])
            note.font.size = Pt(8.5)
            note.italic = True
            note.font.color.rgb = RGBColor.from_string(INK_SOFT)
        return

    if block.kind == NOTE:
        label = block.meta.get("label", "NOTE")
        colour, fill = NOTE_TONE.get(label, NOTE_TONE["NOTE"])
        table = doc.add_table(rows=1, cols=1)
        table.style = "Table Grid"
        cell = table.rows[0].cells[0]
        _docx_shade(cell, fill)
        cell.text = ""
        tag = cell.paragraphs[0].add_run(label + "  ")
        tag.bold = True
        tag.font.color.rgb = RGBColor.from_string(colour)
        body = cell.paragraphs[0].add_run(block.text)
        body.font.color.rgb = RGBColor.from_string(INK)
        doc.add_paragraph()
        return

    if block.kind == CODE:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Pt(12)
        run = paragraph.add_run(block.text)
        run.font.name = FONT_MONO
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor.from_string(INK_SOFT)
        return

    warnings.append(f"block type {block.kind!r} has no Word rendering; shown as text")
    doc.add_paragraph(block.text or " ".join(block.items))


def _write_docx(target: Path, title: str, body: str) -> dict[str, Any]:
    _require("docx", "python-docx")
    from docx import Document
    from docx.shared import Pt, RGBColor

    blocks, warnings = parse_body(body)

    doc = Document()
    _docx_styles(doc)

    if title:
        doc.add_paragraph(title, style="Title")
        subtitle = doc.add_paragraph()
        stamp = subtitle.add_run(
            "Prepared offline by the Kavach Sovereign Workbench. "
            "No content left this machine."
        )
        stamp.font.size = Pt(9)
        stamp.italic = True
        stamp.font.color.rgb = RGBColor.from_string(INK_SOFT)
        _docx_rule(subtitle, ACCENT, 12)

    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1
        try:
            _docx_block(doc, block, warnings)
        except Exception as exc:                      # noqa: BLE001 - see module docstring
            warnings.append(
                f"a {block.kind} block could not be laid out "
                f"({type(exc).__name__}); its text is shown as a paragraph"
            )
            doc.add_paragraph(block.text or "\n".join(block.items))

    # Same measure as the deck. A .docx of three headings and no paragraphs is
    # a document that opens, prints and says nothing - and looks finished.
    thin = _too_thin(blocks)
    if thin:
        warnings.append(thin)

    _docx_notes(doc, warnings)
    doc.save(target)
    return {
        "format": "docx",
        "content_chars": _content_chars(blocks),
        # `paragraphs` is kept because the acceptance tests and the run sheet
        # read it; the block census is the more useful number now.
        "paragraphs": sum(counts.values()),
        "blocks": counts,
        "warnings": warnings,
    }


def _docx_notes(doc, warnings: list[str]) -> None:
    if not warnings:
        return
    from docx.shared import Pt, RGBColor

    heading = doc.add_paragraph("Rendering notes", style="Heading 2")
    _docx_rule(heading, RULE, 6)
    intro = doc.add_paragraph()
    run = intro.add_run(
        "This document was produced, but the following parts of it did not "
        "render as intended. Nothing was dropped - each is shown as written."
    )
    run.font.size = Pt(9)
    run.italic = True
    run.font.color.rgb = RGBColor.from_string(INK_SOFT)
    for note in warnings:
        doc.add_paragraph(note, style="List Bullet")


# ------------------------------------------------------------------ Excel

# `thickness_mm` is a key; `Thickness Mm` is not a column heading either. A
# trailing unit token is the one case worth special-casing, because every
# engineering dataset has them and "Thickness (mm)" is what the reader expects
# on a sheet they will print.
_UNIT_SUFFIX = re.compile(
    r"^(.*?)[ _-]+(mm|cm|m|km|in|ft|kg|g|t|s|min|h|hr|degc|degf|c|f|k|"
    r"bar|barg|bara|kpa|mpa|pa|psi|psig|m3h|m3|lpm|gpm|kw|mw|w|hp|hz|rpm|pct|"
    r"percent|usd|inr)$",
    re.IGNORECASE,
)
_UNIT_LABEL = {"degc": "degC", "degf": "degF", "pct": "%", "percent": "%",
               "m3h": "m3/h", "barg": "barg", "bara": "bara", "psig": "psig"}


def _heading_text(raw: str) -> str:
    """A column heading a person would put on a circulated spreadsheet."""
    text = (raw or "").strip()
    if not text or text.startswith("="):
        return text
    # Already written for humans: spaces and capitals, leave it alone.
    if " " in text and not text.islower():
        return text
    match = _UNIT_SUFFIX.match(text)
    unit = ""
    if match:
        text, token = match.group(1), match.group(2)
        unit = _UNIT_LABEL.get(token.lower(), token)
    words = re.split(r"[_\-\s]+|(?<=[a-z0-9])(?=[A-Z])", text)
    label = " ".join(w[:1].upper() + w[1:] for w in words if w) or raw
    return f"{label} ({unit})" if unit else label


def _sheet_name(title: str, fallback: str) -> str:
    """A legal, readable sheet name.

    Excel refuses `: \\ / ? * [ ]` and anything over 31 characters, and openpyxl
    raises rather than trimming. A calculation sheet lost because its title
    contained a slash is the kind of failure this module exists to prevent.
    """
    cleaned = re.sub(r"[:\\/?*\[\]]", " ", title or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned or fallback)[:31]


def _write_xlsx(target: Path, title: str, body: str) -> dict[str, Any]:
    _require("openpyxl", "openpyxl")
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    rows, warnings = parse_rows(body)
    wb = Workbook()
    ws = wb.active
    ws.title = _sheet_name(title, "Data")

    if not rows:
        warnings.append("the body contained no rows; the sheet has headings only")

    # THE HEADER STAYS ON ROW 1.
    #
    # A title banner above the data looks better and breaks everything that
    # reads the sheet: "row 1 is the header" is how every consumer, every
    # pandas.read_excel and every acceptance test written against this tool
    # finds the columns. So the title goes where it costs no data row - the
    # sheet tab, the workbook properties, and the PRINT header, which is where
    # a signed-off calculation sheet actually needs it.
    width = max((len(r) for r in rows), default=1)
    first_data_row = 1
    if title:
        wb.properties.title = title
        ws.oddHeader.center.text = f"&\"{FONT_HEAD},Bold\"&12{title}"
        ws.oddFooter.left.text = (
            "Prepared offline by the Kavach Sovereign Workbench - "
            "no content left this machine"
        )
        ws.oddFooter.right.text = "Page &P of &N"

    header_fill = PatternFill("solid", fgColor=ACCENT)
    header_font = Font(name=FONT_BODY, size=11, bold=True, color="FFFFFF")
    band = PatternFill("solid", fgColor="FBFBF9")
    hairline = Side(style="thin", color=RULE)
    edges = Border(left=hairline, right=hairline, top=hairline, bottom=hairline)

    widest = [0] * width
    for offset, row in enumerate(rows):
        excel_row = first_data_row + offset
        is_header = offset == 0
        for column in range(width):
            raw = row[column] if column < len(row) else ""
            text = str(raw).strip()
            # A cell the author wrote as a formula stays a formula. Excel needs
            # the leading "=" and openpyxl passes it through untouched; writing
            # it as text would hand the engineer a spreadsheet that shows its
            # own arithmetic instead of doing it.
            value = text if text.startswith("=") else _maybe_number(text)
            if is_header:
                value = _heading_text(text)
            cell = ws.cell(row=excel_row, column=column + 1, value=value)
            cell.border = edges
            if is_header:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="left", vertical="center",
                                           wrap_text=True)
            else:
                cell.font = Font(name=FONT_BODY, size=10.5, color=INK)
                if offset % 2 == 0:
                    cell.fill = band
                if isinstance(value, float):
                    cell.number_format = "#,##0.000"
                    cell.alignment = Alignment(horizontal="right")
                elif isinstance(value, int):
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right")
            widest[column] = max(widest[column], len(text))

    for column in range(width):
        # Clamped: one long cell must not push a column off the screen, and a
        # heading must still fit. These are the numbers Excel's own autofit
        # lands near for Calibri 11.
        ws.column_dimensions[get_column_letter(column + 1)].width = min(
            max(widest[column] + 4, 12), 52
        )
    if rows:
        ws.row_dimensions[first_data_row].height = 22
        # Freeze under the header and turn on the filter, so a 400-row wall
        # thickness survey is usable rather than merely present.
        ws.freeze_panes = ws.cell(row=first_data_row + 1, column=1)
        ws.auto_filter.ref = (
            f"A{first_data_row}:"
            f"{get_column_letter(width)}{first_data_row + len(rows) - 1}"
        )

    if warnings:
        _xlsx_notes(wb, warnings)
    wb.save(target)
    return {
        "format": "xlsx",
        # `rows` counts DATA rows, as it always has - the banner is chrome.
        "rows": max(len(rows) - 1, 0) if rows else 0,
        "header_row": first_data_row if rows else None,
        "columns": width,
        "sheet": ws.title,
        "warnings": warnings,
    }


def _xlsx_notes(wb, warnings: list[str]) -> None:
    """Rendering notes go on their own sheet, never mixed into the data.

    An extra row appended to a calculation sheet is a row somebody sums.
    """
    from openpyxl.styles import Font

    ws = wb.create_sheet("Rendering notes")
    ws.cell(row=1, column=1, value="These parts of the body did not render as "
                                   "intended. Nothing was dropped.").font = Font(
        name=FONT_BODY, size=10, italic=True, color=INK_SOFT
    )
    for index, note in enumerate(warnings, start=3):
        ws.cell(row=index, column=1, value=note)
    ws.column_dimensions["A"].width = 110


# ------------------------------------------------------------- PowerPoint

def _pptx_text(frame, block: Block, warnings: list[str]) -> None:
    from pptx.dml.color import RGBColor
    from pptx.util import Pt

    def line(text: str, *, level: int = 0, mono: bool = False, bold: bool = False,
             size: float = 18, colour: str = INK):
        para = frame.paragraphs[0] if not frame.paragraphs[0].runs and \
            not frame.paragraphs[0].text else frame.add_paragraph()
        para.level = min(level, 4)
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT_MONO if mono else FONT_BODY
        run.font.color.rgb = RGBColor.from_string(colour)

    if block.kind in (BULLETS, STEPS):
        for index, item in enumerate(block.items, start=1):
            prefix = f"{index}. " if block.kind == STEPS else ""
            line(prefix + item)
        return
    if block.kind == FIELDS:
        for label, value in block.rows:
            line(f"{label}: {value}")
        return
    if block.kind == TABLE:
        for row in block.rows:
            line(" | ".join(row), mono=True, size=14)
        return
    if block.kind == FORMULA:
        rendered = render_formula(block.text)
        text = rendered["expression"]
        if rendered["value"] is not None:
            text += f"   =  {rendered['value']:g}"
        elif rendered["simplified"]:
            text += f"   =  {rendered['simplified']}"
        line(text, mono=True, size=16, colour=ACCENT)
        return
    if block.kind == NOTE:
        label = block.meta.get("label", "NOTE")
        colour = NOTE_TONE.get(label, NOTE_TONE["NOTE"])[0]
        line(f"{label}: {block.text}", bold=True, colour=colour)
        return
    if block.kind == CODE:
        for row in block.text.split("\n"):
            line(row, mono=True, size=13, colour=INK_SOFT)
        return
    if block.kind == HEADING:
        line(block.text, bold=True, size=20)
        return
    if block.kind == PARAGRAPH:
        line(block.text)
        return
    warnings.append(f"block type {block.kind!r} has no slide rendering; shown as text")
    line(block.text or " ".join(block.items))


# How much a deliverable actually SAYS, and the threshold below which saying so
# matters more than the file existing.
#
# A run can produce a .pptx that opens, has the right title and contains two
# words, because the body it was given was a template whose {{stepN}} resolved
# to almost nothing. The file is then evidence of a finished task and is not
# one. This does not refuse to write it - a thin deliverable is still the
# deliverable, and refusing would lose the little there is - it makes the
# thinness part of the result, in `warnings`, in the rendering notes inside the
# document, and in the panel.
# Measured against the document's OWN STRUCTURE rather than a flat character
# count, because "how short is too short" has no absolute answer: "# Findings /
# All good." is a legitimate nine-character note, and three headings with two
# words under each is a hollow deck. What is wrong in the second case is the
# RATIO - the model laid out sections and did not fill them.
THIN_CHARS_PER_SECTION = 25

# The longest a line may be and still be treated as a slide title. PowerPoint's
# own title placeholder stops being readable well before this; past it, the text
# is a paragraph that happened to be on its own line.
TITLE_MAX_CHARS = 90


def _content_chars(blocks: list[Block]) -> int:
    """Characters of real content - headings excluded, because a document made
    entirely of headings is the case this measure exists to catch."""
    total = 0
    for block in blocks:
        if block.kind == HEADING:
            continue
        total += len(block.text or "")
        total += sum(len(item) for item in (block.items or []))
        for row in (block.rows or []):
            total += sum(len(cell) for cell in row)
    return total


def _too_thin(blocks: list[Block]) -> str | None:
    chars = _content_chars(blocks)
    sections = sum(1 for block in blocks if block.kind == HEADING)

    if chars == 0 and blocks or sections >= 2 and chars < THIN_CHARS_PER_SECTION * sections:
        hollow = True
    else:
        hollow = False
    if not hollow:
        return None

    return (
        f"this deliverable lays out {sections} section(s) and fills them with "
        f"{chars} characters in total, which is not a deliverable - it is an "
        "outline. Check that the step feeding this returned real text: a "
        "{{stepN}} reference resolves to what that step FOUND, and a search "
        "that matched little resolves to little."
    )


def _looks_like_a_title(text: str) -> bool:
    """Short, and not a sentence.

    Length alone was not enough. Real corpus lines are short and are still
    sentences - "4 Standby pump P-101B vibration is high (7.65 mm/s on NDE)."
    and "at 180 bar(g) for 4 hrs. Zero pressure drop recorded." both became
    slide TITLES, so a briefing read as a stack of half-thoughts with empty
    bodies under them. A line that ends in a full stop is prose; a title is a
    label ("Findings", "Recommendations for GM Approval Note").

    The `Heading:` convention is unaffected - a line ending in a colon is
    parsed as a HEADING block long before it reaches here.
    """
    stripped = (text or "").strip()
    if not stripped or len(stripped) > TITLE_MAX_CHARS:
        return False
    return not stripped.endswith((".", "!", "?", ";", ","))


def _write_pptx(target: Path, title: str, body: str) -> dict[str, Any]:
    _require("pptx", "python-pptx")
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Pt

    blocks, warnings = parse_body(body)
    prs = Presentation()

    if title:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
        for paragraph in slide.shapes.title.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.name = FONT_HEAD
                run.font.color.rgb = RGBColor.from_string(INK)
        if len(slide.placeholders) > 1:
            sub = slide.placeholders[1].text_frame
            sub.text = ("Prepared offline by the Kavach Sovereign Workbench. "
                        "No content left this machine.")
            for run in sub.paragraphs[0].runs:
                run.font.size = Pt(12)
                run.font.italic = True
                run.font.color.rgb = RGBColor.from_string(INK_SOFT)

    # Two ways a slide starts, and both have to keep working:
    #
    #   a heading (`# Findings`), the new markup; and
    #   a plain line that opens a blank-line-separated block, which is what the
    #   manifest documents and what every existing plan writes.
    #
    # The second is why this is not simply "split on HEADING": dropping it
    # would collapse "Findings\n- x\n\nActions\n- y" into a single slide, and
    # a deck that silently merges its sections is worse than no deck.
    groups: list[tuple[str, list[Block]]] = []
    for block in blocks:
        if block.kind == HEADING:
            groups.append((block.text, []))
            continue
        # ...but only a SHORT line. The rule is "a blank-line-separated block
        # starts a slide", and it was written for label lines like "Findings" -
        # every existing plan writes those and they must keep working. Applied
        # to a 400-character retrieved passage it made that passage the slide's
        # TITLE, truncated at 90 characters with the rest repeated underneath,
        # so a deck built from real corpus text opened on slides that began
        # mid-sentence: "ude Distillation Unit (CDU-II), comprehensive..."
        #
        # A passage is content. It belongs on the slide that is already open.
        if (
            block.kind == PARAGRAPH
            and (not groups or groups[-1][1])
            and _looks_like_a_title(block.text)
        ):
            groups.append((block.text, []))
            continue
        if not groups:
            groups.append(("", []))
        groups[-1][1].append(block)

    # A HEADING WITH NOTHING UNDER IT IS NOT A SLIDE.
    #
    # The guard below used to be `not heading and not content` - both empty -
    # so a section that had a title and no body produced a slide with a title
    # and an empty box. A deck built from a body of "Findings:", a step
    # reference and "Recommendation:" - where the retrieved passage came back
    # thin - is then three headings and no content, which is exactly what a
    # reader reported: blank slides, and one or two words where findings should
    # be.
    #
    # Empty sections are NOT silently dropped either - a section the model
    # meant to fill and did not is a fact about the run. Each one is named on
    # the slide and counted in `warnings`, so the gap is visible in the deck and
    # in the tool result rather than looking like a design choice.
    empty_sections: list[str] = []
    slides = 0
    for heading, content in groups:
        if not heading and not content:
            continue
        if heading and not content:
            empty_sections.append(heading)
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        spill = None
        if len(heading) > 90:
            # A paragraph used as a title. Truncating alone would LOSE the rest
            # of the sentence, so the full text goes into the body.
            spill, heading = heading, heading[:87].rstrip() + "..."
        slide.shapes.title.text = heading or "Detail"
        for paragraph in slide.shapes.title.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.name = FONT_HEAD
                run.font.size = Pt(28)
                run.font.color.rgb = RGBColor.from_string(ACCENT)

        frame = slide.placeholders[1].text_frame
        frame.clear()
        frame.word_wrap = True
        if spill:
            content = [Block(PARAGRAPH, text=spill), *content]
        for block in content:
            try:
                _pptx_text(frame, block, warnings)
            except Exception as exc:                  # noqa: BLE001
                warnings.append(
                    f"a {block.kind} block could not be laid out on a slide "
                    f"({type(exc).__name__}); its text is shown instead"
                )
                para = frame.add_paragraph()
                para.text = block.text or " ".join(block.items)
        if not content and not spill:
            para = frame.add_paragraph()
            para.text = "(no content was written for this section)"
            for run in para.runs:
                run.font.size = Pt(14)
                run.font.italic = True
                run.font.color.rgb = RGBColor.from_string(INK_SOFT)
        slides += 1

    for name in empty_sections:
        warnings.append(
            f"the section {name!r} has a heading and no content - the deck says so "
            "on that slide rather than showing an empty one"
        )
    thin = _too_thin(blocks)
    if thin:
        warnings.append(thin)

    if warnings:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Rendering notes"
        frame = slide.placeholders[1].text_frame
        frame.clear()
        frame.text = ("These parts did not render as intended. "
                      "Nothing was dropped.")
        for note in warnings:
            para = frame.add_paragraph()
            para.text = note
            para.level = 1
            for run in para.runs:
                run.font.size = Pt(12)

    prs.save(target)
    return {
        "format": "pptx",
        "slides": slides,
        "empty_sections": empty_sections,
        "content_chars": _content_chars(blocks),
        "warnings": warnings,
    }


_WRITERS = {".docx": _write_docx, ".xlsx": _write_xlsx, ".pptx": _write_pptx}


# --------------------------------------------------------------- file names

# Characters Windows refuses in a filename. A model that names a deliverable
# "P-101A: inspection.docx" is being perfectly reasonable and would otherwise
# get an OSError from the filesystem at the very end of the run.
_ILLEGAL = re.compile(r'[<>:"|?*\x00-\x1f]')


def clean_filename(path: str) -> tuple[str, str | None]:
    """(usable path, what changed) - only ever touches the FILE NAME.

    Directories are left alone so the workspace layout a plan asked for is the
    layout it gets, and the stem is preserved rather than rewritten: renaming
    `approval_note.docx` to something prettier would break the acceptance test
    of the very step that wrote it.
    """
    raw = (path or "").replace("\\", "/")
    head, _, name = raw.rpartition("/")
    cleaned = _ILLEGAL.sub("-", name).strip().strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned:
        return path, None
    if cleaned == name:
        return path, None
    rebuilt = f"{head}/{cleaned}" if head else cleaned
    return rebuilt, (
        f"the file name {name!r} contains characters this filesystem refuses; "
        f"written as {cleaned!r}"
    )


def doc_write(
    workspace: Workspace,
    path: str,
    body: str,
    title: str = "",
    **_: Any,
) -> dict[str, Any]:
    path, renamed = clean_filename(path)
    resolved = workspace.resolve(path)
    suffix = resolved.suffix.lower()
    if suffix not in _WRITERS:
        raise DeliverableError(
            f"{path!r} has extension {suffix or '(none)'}, which this tool does "
            f"not produce. Use one of {', '.join(sorted(SUPPORTED))} - or "
            "fs.write for plain text."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # The title does not go through the body parser, and it reaches the Word
    # title, the Excel sheet tab and the opening slide - three more places a
    # stray control character would raise at the last step of the run.
    title, scrubbed = scrub(title or "")
    detail = _WRITERS[suffix](resolved, title, body)
    if scrubbed:
        detail.setdefault("warnings", []).append(
            f"{scrubbed} control character(s) were removed from the title"
        )
    if renamed:
        detail.setdefault("warnings", []).append(renamed)
        detail["renamed"] = True
    return {
        "path": workspace.relative(resolved),
        "size_bytes": resolved.stat().st_size,
        **detail,
    }

"""One document model behind Word, Excel and PowerPoint (SS 26117, L4).

`doc.write` takes three flat strings, because nested tool arguments are where
small models start failing (§05). That constraint is about the TOOL SCHEMA, not
about the document: a body can still carry structure, as long as the structure
is something a 1.5B can type without escaping anything.

So this module parses `body` into an ordered list of typed blocks, and the three
renderers in `deliverables.py` each draw that same list their own way. A heading
is a heading whether it becomes a Word Heading 1, an Excel sheet title or a
slide title - and adding a block type once gives all three formats.

WHAT IT ACCEPTS

  Markup, line-oriented and forgiving:

    # H1 / ## H2 / ### H3        headings
    - item   * item              bullets
    1. step  2) step             numbered steps, renumbered as a sequence
    Label: value                 a labelled field, when the line is short
    | a | b |                    a table (a separator row of dashes is ignored)
    $$ dP = f*(L/D)*rho*v**2/2   a display formula
    > Note: ...                  a callout (NOTE / WARNING / CAUTION / CAUTION)
    ```...```                    a verbatim block
    plain text                   a paragraph

  JSON, when the whole body parses as JSON. A model that has just read a tool
  result holding a list of dicts will hand that straight through, and the old
  writer turned it into one paragraph of braces. A list of objects becomes a
  table with the keys as headings; an object with `sections` or `blocks`
  becomes a structured document.

THE TWO RULES

  PARSING NEVER RAISES. A deliverable is the last step of a long run. Losing it
  to a malformed table - after the corpus was searched, the model was loaded and
  the plan was judged - trades the whole run for a line of markup. Anything this
  parser cannot read becomes a paragraph of its own literal text and appends a
  note to `warnings`.

  BUT IT IS NEVER SILENT. Every degraded block is named in `warnings`, which
  `doc.write` returns to the agent AND prints into the document itself under
  "Rendering notes". A document that quietly dropped a table is worse than one
  that shows the table as text and says why.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

# Block kinds. Each renderer maps these to its own primitives; a renderer that
# meets one it does not know falls back to PARAGRAPH rather than dropping it.
HEADING = "heading"
PARAGRAPH = "paragraph"
BULLETS = "bullets"
STEPS = "steps"
FIELDS = "fields"
TABLE = "table"
FORMULA = "formula"
NOTE = "note"
CODE = "code"


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 1
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ patterns

_HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
_STEP = re.compile(r"^\s*(\d{1,3})[.)]\s+(.*\S)\s*$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_RULE = re.compile(r"^[\s|:-]+$")
_FENCE = re.compile(r"^\s*```+\s*(\w+)?\s*$")
_CALLOUT = re.compile(r"^\s*>\s*(.*)$")
_CALLOUT_KIND = re.compile(r"^(note|warning|caution|important|danger)\b[:\-\s]*(.*)$", re.IGNORECASE)

# A field line: "Equipment: P-101A". Deliberately narrow - a long sentence that
# happens to contain a colon is prose, and turning it into a two-column field
# row makes an approval note unreadable. 48 characters is about the longest a
# real field label runs ("Maximum allowable working pressure").
_FIELD = re.compile(r"^\s*([A-Z][^:|]{2,48}?)\s*:\s+(\S.*)$")

# The ORIGINAL convention, which `doc.write`'s manifest documents and which
# every plan written against it uses: a short line ending in ":" and nothing
# else is a heading. "Background:" on its own line has meant Heading 2 since
# Phase 4, so it still does - a markup upgrade that invalidates the plans
# already written for it is a breaking change wearing an improvement.
_TRAILING_COLON = re.compile(r"^\s*(\S.{0,76}?)\s*:\s*$")

# $$ ... $$ on one line, or a line that is plainly an equation. The second form
# matters because models write bare "dP = f * (L/D) * rho * v^2 / 2" far more
# often than they remember a delimiter.
_FORMULA_DELIM = re.compile(r"^\s*\$\$\s*(.+?)\s*\$\$\s*$")
_FORMULA_BARE = re.compile(
    r"^\s*([A-Za-z_][\w\s.]{0,40}?)\s*=\s*([^=].*?)\s*$"
)
# What makes a bare "a = b" line an equation rather than a field: the right side
# has to look like arithmetic. "Status: = pending" is not a formula, and neither
# is "Prepared by = R. Kumar".
_LOOKS_LIKE_MATHS = re.compile(r"[0-9]").search
_MATHS_OPERATOR = re.compile(r"[+\-*/^()]|\b(sqrt|log|ln|exp|sin|cos|tan|pi)\b")


def _is_equation(right: str) -> bool:
    return bool(_MATHS_OPERATOR.search(right)) and bool(_LOOKS_LIKE_MATHS(right))


# Characters no OOXML file can carry. XML 1.0 permits tab, newline and carriage
# return out of the C0 range and nothing else, plus the two non-characters at
# the end of the BMP and the unpaired surrogates.
#
# This is not theoretical: a body assembled from OCR output, from a file read
# with the wrong encoding, or from a model echoing bytes it was shown, arrives
# with a NUL in it - and python-docx writes it happily while openpyxl raises
# IllegalCharacterError. One byte nobody can see would otherwise cost the run
# its deliverable at the last step.
_ILLEGAL_XML = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff\ufdd0-\ufdef\ufffe\uffff]"
)


def scrub(text: str) -> tuple[str, int]:
    """(text an OOXML file can carry, how many characters were replaced).

    Replaced with a space rather than deleted: removing them silently joins two
    words that were separate, and a report that reads "thepumpfailed" is a
    report whose corruption is invisible.
    """
    cleaned, count = _ILLEGAL_XML.subn(" ", text or "")
    return cleaned, count


# --------------------------------------------------------------- formulas

def render_formula(expression: str, substitutions: str | None = None) -> dict[str, Any]:
    """Present one formula, and evaluate it if it can be evaluated.

    Returns `{"expression", "value", "simplified", "unresolved", "steps",
    "note"}` and NEVER raises. That is the whole contract: an engineering
    document has to be able to carry a formula this box cannot compute - a
    symbolic relation, a formula with unbound symbols, a unit expression -
    without the document failing to exist.

    Three outcomes, all useful and all honest:

      A NUMBER when every symbol has a value. `value` is set and the document
      shows the working.

      A SIMPLIFIED RELATION when symbols remain. `dP = f*(L/D)*rho*v**2/2`
      comes back as `L*f*rho*v**2/(2*D)` with `unresolved` naming every symbol
      still free. That is a better line in a report than the one that was typed,
      and it is not a failure.

      THE EXPRESSION AS WRITTEN when nothing can be done with it, with `note`
      saying why.

    What it will never do is invent a number. An unchecked value in front of a
    plant engineer is the one genuinely dangerous outcome here, so a failure to
    evaluate is always reported rather than filled in.
    """
    shown = (expression or "").strip()
    out: dict[str, Any] = {
        "expression": shown,
        "value": None,
        "simplified": None,
        "unresolved": [],
        "steps": [],
        "note": None,
    }
    if not shown:
        out["note"] = "the formula was empty"
        return out

    try:
        from .calc import CalcError, calc

        # `result`, not `value` - calc's own key. Reading the wrong one made
        # every formula in every document look unevaluable, including the ones
        # that had computed perfectly.
        outcome = calc(shown, substitutions=substitutions)
        result = outcome.get("result")
        out["steps"] = list(outcome.get("steps") or [])
        out["unresolved"] = list(outcome.get("unresolved") or [])

        if isinstance(result, (int, float)) and not isinstance(result, bool):
            out["value"] = float(result)
        elif isinstance(result, str) and result.strip():
            out["simplified"] = result.strip()
            if out["unresolved"]:
                out["note"] = (
                    "shown as a relation - no value, because "
                    + ", ".join(out["unresolved"])
                    + " were not given"
                )
            else:
                out["note"] = "shown as written - this did not reduce to a number"
        elif isinstance(result, list):
            # solve_for returns a list of roots. Not reachable from a document
            # today, but a list rendered as "None" would be a silent loss.
            out["simplified"] = ", ".join(str(r) for r in result)
        else:
            out["note"] = "shown as written - this did not reduce to a number"
    except CalcError as exc:
        out["note"] = f"shown as written - it could not be evaluated here: {exc}"
    except Exception as exc:                          # noqa: BLE001
        # calc is the only thing called here and it is well behaved, but a
        # deliverable must not be lost to a surprise from a dependency.
        out["note"] = f"shown as written - the evaluator raised {type(exc).__name__}"
    return out


# ------------------------------------------------------------------- JSON

def _cell(value: Any) -> str:
    """One JSON value as one cell of text.

    Nested structures are kept as compact JSON rather than flattened: a reader
    can at least see what was there, and flattening invents a shape the data
    did not have.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, str)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    except (TypeError, ValueError):
        return str(value)


def _humanise(key: str) -> str:
    """`suction_pressure_barg` -> `Suction Pressure Barg`.

    JSON keys are written for machines. A spreadsheet whose headings are
    snake_case is a spreadsheet nobody circulates, and the brief asks for real
    deliverables rather than a dump.
    """
    words = re.split(r"[_\-\s]+|(?<=[a-z0-9])(?=[A-Z])", str(key))
    return " ".join(w[:1].upper() + w[1:] for w in words if w) or str(key)


def _table_from_records(records: list[dict[str, Any]]) -> Block:
    """A list of objects -> one table, with every key that appears anywhere.

    Union of keys rather than the first record's keys: records from a real tool
    are ragged, and taking the first row's shape silently drops the columns that
    only later rows carry.
    """
    headers: list[str] = []
    for record in records:
        for key in record:
            if key not in headers:
                headers.append(key)
    rows = [[_humanise(h) for h in headers]]
    for record in records:
        rows.append([_cell(record.get(h)) for h in headers])
    return Block(TABLE, rows=rows, meta={"header": True, "source": "json"})


def _blocks_from_json(data: Any, warnings: list[str]) -> list[Block] | None:
    """Structured JSON -> blocks, or None if this is not a shape we can lay out.

    Returning None rather than guessing matters: a JSON scalar or a deeply
    nested object has no obvious document shape, and inventing one produces a
    confident-looking document about nothing. The caller falls back to treating
    the text as markup, which at worst shows the JSON verbatim.
    """
    if isinstance(data, list) and data:
        if all(isinstance(item, dict) for item in data):
            return [_table_from_records(data)]
        if all(isinstance(item, (list, tuple)) for item in data):
            rows = [[_cell(c) for c in row] for row in data]
            return [Block(TABLE, rows=rows, meta={"header": True, "source": "json"})]
        if all(isinstance(item, (str, int, float, bool)) for item in data):
            return [Block(BULLETS, items=[_cell(i) for i in data])]
        warnings.append(
            "the JSON body is a list of mixed types; each entry is shown as text"
        )
        return [Block(BULLETS, items=[_cell(i) for i in data])]

    if isinstance(data, dict):
        blocks: list[Block] = []
        sections = data.get("sections") or data.get("blocks")
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    heading = section.get("heading") or section.get("title")
                    if heading:
                        blocks.append(Block(HEADING, text=_cell(heading), level=2))
                    nested = _blocks_from_json(
                        section.get("content") or section.get("body") or section,
                        warnings,
                    )
                    if nested:
                        blocks.extend(nested)
                else:
                    blocks.append(Block(PARAGRAPH, text=_cell(section)))
            return blocks

        # A flat object is a set of labelled fields, which is exactly what the
        # header of an approval note or a datasheet is.
        pairs = [(_humanise(k), _cell(v)) for k, v in data.items()]
        return [Block(FIELDS, rows=[[k, v] for k, v in pairs])]

    return None


# ------------------------------------------------------------------ markup

def _flush(pending: dict[str, Any], blocks: list[Block]) -> None:
    """Close whichever list/table is being accumulated."""
    kind = pending.get("kind")
    if kind in (BULLETS, STEPS) and pending.get("items"):
        blocks.append(Block(kind, items=list(pending["items"])))
    elif kind in (TABLE, FIELDS) and pending.get("rows"):
        blocks.append(
            Block(kind, rows=[list(r) for r in pending["rows"]],
                  meta={"header": kind == TABLE})
        )
    elif kind == PARAGRAPH and pending.get("lines"):
        blocks.append(Block(PARAGRAPH, text=" ".join(pending["lines"])))
    pending.clear()


def _start(pending: dict[str, Any], blocks: list[Block], kind: str) -> None:
    if pending.get("kind") != kind:
        _flush(pending, blocks)
        pending["kind"] = kind
        pending.setdefault("items", [])
        pending.setdefault("rows", [])
        pending.setdefault("lines", [])


def parse_markup(body: str, warnings: list[str]) -> list[Block]:
    blocks: list[Block] = []
    pending: dict[str, Any] = {}
    lines = (body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    fence: list[str] | None = None
    fence_lang = ""

    for raw in lines:
        line = raw.rstrip()

        # ---- verbatim block
        fenced = _FENCE.match(line)
        if fenced:
            if fence is None:
                _flush(pending, blocks)
                fence, fence_lang = [], fenced.group(1) or ""
            else:
                blocks.append(Block(CODE, text="\n".join(fence),
                                    meta={"language": fence_lang}))
                fence, fence_lang = None, ""
            continue
        if fence is not None:
            fence.append(raw)
            continue

        if not line.strip():
            _flush(pending, blocks)
            continue

        match = _HEADING.match(line)
        if match:
            _flush(pending, blocks)
            blocks.append(Block(HEADING, text=match.group(2), level=len(match.group(1))))
            continue

        match = _CALLOUT.match(line)
        if match:
            _flush(pending, blocks)
            text = match.group(1).strip()
            kind_match = _CALLOUT_KIND.match(text)
            label = kind_match.group(1).upper() if kind_match else "NOTE"
            text = kind_match.group(2).strip() if kind_match else text
            blocks.append(Block(NOTE, text=text, meta={"label": label}))
            continue

        match = _FORMULA_DELIM.match(line)
        if match:
            _flush(pending, blocks)
            blocks.append(Block(FORMULA, text=match.group(1)))
            continue

        match = _TABLE_ROW.match(line)
        if match:
            if _TABLE_RULE.match(line):
                continue                      # the |---|---| separator
            _start(pending, blocks, TABLE)
            pending["rows"].append([c.strip() for c in match.group(1).split("|")])
            continue

        match = _STEP.match(line)
        if match:
            _start(pending, blocks, STEPS)
            pending["items"].append(match.group(2))
            continue

        match = _BULLET.match(line)
        if match:
            _start(pending, blocks, BULLETS)
            pending["items"].append(match.group(1))
            continue

        match = _FORMULA_BARE.match(line)
        if match and _is_equation(match.group(2)):
            _flush(pending, blocks)
            blocks.append(Block(FORMULA, text=line.strip()))
            continue

        match = _FIELD.match(line)
        if match:
            _start(pending, blocks, FIELDS)
            pending["rows"].append([match.group(1).strip(), match.group(2).strip()])
            continue

        match = _TRAILING_COLON.match(line)
        if match:
            _flush(pending, blocks)
            blocks.append(Block(HEADING, text=match.group(1), level=2))
            continue

        _start(pending, blocks, PARAGRAPH)
        pending["lines"].append(line.strip())

    if fence is not None:
        # An unclosed fence is a model that forgot the closing backticks. The
        # content is still the content; losing it would be the worse reading.
        blocks.append(Block(CODE, text="\n".join(fence), meta={"language": fence_lang}))
        warnings.append("a verbatim block was never closed; it is shown to the end")
    _flush(pending, blocks)
    return blocks


def parse_body(body: str, *, warnings: list[str] | None = None) -> tuple[list[Block], list[str]]:
    """`body` -> blocks. Never raises; degrades with a note instead."""
    notes = warnings if warnings is not None else []
    text = body if isinstance(body, str) else _cell(body)
    text, scrubbed = scrub(text)
    if scrubbed:
        notes.append(
            f"{scrubbed} control character(s) that no Word, Excel or PowerPoint "
            "file can contain were replaced with spaces"
        )

    stripped = text.strip()
    if stripped[:1] in "[{":
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            pass                              # not JSON after all; read as markup
        else:
            blocks = _blocks_from_json(data, notes)
            if blocks:
                return blocks, notes
            notes.append(
                "the body is valid JSON but has no shape this can lay out; it is "
                "shown as written"
            )
            return [Block(CODE, text=stripped, meta={"language": "json"})], notes

    try:
        return parse_markup(text, notes), notes
    except Exception as exc:                          # noqa: BLE001
        # The parser is pure regex over lines and should not fail. If it ever
        # does, the deliverable still exists and carries every character of what
        # was meant to go in it.
        notes.append(
            f"the body could not be parsed ({type(exc).__name__}); it is shown "
            "as plain text"
        )
        return [Block(PARAGRAPH, text=text)], notes


# ------------------------------------------------------------------ tabular

def parse_rows(body: str, warnings: list[str] | None = None) -> tuple[list[list[str]], list[str]]:
    """`body` -> rows, for the spreadsheet writer.

    Accepts, in order of preference: JSON (records or rows), a markdown table,
    and CSV. CSV stays last and stays supported - it is what the tool manifest
    has always documented, and a plan written against that must keep working.
    """
    notes = warnings if warnings is not None else []
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    text, scrubbed = scrub(text)
    if scrubbed:
        notes.append(
            f"{scrubbed} control character(s) that no spreadsheet can contain "
            "were replaced with spaces"
        )

    stripped = text.strip()
    if stripped[:1] in "[{":
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            pass
        else:
            blocks = _blocks_from_json(data, notes)
            if blocks and blocks[0].kind == TABLE:
                return blocks[0].rows, notes
            if blocks and blocks[0].kind == FIELDS:
                return [["Field", "Value"], *blocks[0].rows], notes
            notes.append("the JSON body is not tabular; each entry is one row")
            return [[line] for line in stripped.split("\n")], notes

    if any(_TABLE_ROW.match(line) for line in text.split("\n")):
        rows = []
        for line in text.split("\n"):
            match = _TABLE_ROW.match(line)
            if match and not _TABLE_RULE.match(line):
                rows.append([c.strip() for c in match.group(1).split("|")])
        if rows:
            return rows, notes

    try:
        rows = [
            [c.strip() for c in row]
            for row in csv.reader(io.StringIO(text))
            if any(c.strip() for c in row)
        ]
        return rows, notes
    except csv.Error as exc:
        notes.append(f"the rows could not be read as CSV ({exc}); shown one per line")
        return [[line] for line in text.split("\n") if line.strip()], notes

"""Splitting a finished run into its two kinds of output (SS 26117, L5/L6).

The brief asks for a workbench people use the way they use Claude or Codex, and
those are two different reading modes: you read CODE by looking at the source
and what it printed when it ran, and you read a DOCUMENT by opening the file.
A single "artifacts" list serves neither - an .xlsx and a .py next to each other
as two grey filenames with digests is a manifest, not a deliverable.

So the run panel has one toggle per kind, and this module is what each side of
it renders. Two properties matter more than the formatting:

  THE SPLIT IS A PARTITION. Every artifact the run produced appears under
  exactly one tab. A file that matched neither rule would be invisible in both
  views while still counting in the artifact total, which is the worst of all
  worlds - a deliverable that exists and cannot be found. Code extensions are
  named explicitly; everything else is a document, so an extension nobody
  anticipated lands somewhere a human will see it.

  AN EMPTY TAB SAYS WHY IT IS EMPTY. A run in Code mode that produced no code
  is a real and important failure - the toggle did not deliver what it promised
  - and it is precisely the case where a blank panel reads as "still loading".
  Every empty side carries a sentence naming what happened, and an empty side of
  a run that was ASKED for that kind says so in stronger terms.

Previews are read back OUT OF THE FILE wherever a reader exists, not echoed from
the arguments the model passed to doc.write. Those two are not the same claim:
the second proves what was requested, the first proves what is in the bytes a
human will open. Where no reader is installed the preview falls back to the
requested text and SAYS it is the request rather than the file - the one thing
it must never do is show the request while implying the file was read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import get_settings
from ..tools.impl.languages import UnknownLanguage
from ..tools.impl.languages import detect as detect_language
from ..tools.impl.languages import resolve as resolve_language
from ..tools.workspace import Workspace, WorkspaceViolation
from ..types import WorkMode

# Source the Code tab renders with a language tag. Anything not listed here is a
# document - see the partition note above. `.ipynb` is deliberately absent: it is
# JSON on disk, and showing it as source shows a reader machine noise.
CODE_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".r": "r",
    ".m": "matlab",
    ".jl": "julia",
    ".st": "structured-text",   # PLC logic - refinery instrumentation work
}

# Read straight off disk as text. Everything else needs a format reader below,
# or admits it has none.
PLAIN_TEXT = {".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml", ".xml", ".htm", ".html"}

CODE_EXEC = "code.exec"

# Caps. Large enough that a real deliverable is shown whole, small enough that
# one runaway file cannot make the panel unresponsive. A cut is always declared
# on the item - a preview that silently stops is a preview that lies about the
# end of the document.
MAX_SOURCE_CHARS = 40_000
MAX_PREVIEW_CHARS = 12_000
MAX_STDOUT_CHARS = 8_000


def _language(suffix: str) -> str | None:
    return CODE_LANGUAGE.get(suffix.lower())


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


# ------------------------------------------------------------------ previews

def _read_text_file(path: Path) -> tuple[str | None, str | None]:
    """Decoded contents, or None and the reason. Never a partial decode passed
    off as the file: `errors="replace"` marks what could not be decoded rather
    than dropping it, because a silently shortened engineering note is worse
    than a visible replacement character."""
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"the file could not be read from disk: {exc}"


def _read_docx(path: Path) -> tuple[str | None, str | None]:
    """Paragraphs AND tables, in the order they appear in the document.

    `Document.paragraphs` skips tables entirely, so an approval note whose
    findings are in a table and whose header is a field grid previewed as a
    heading followed by nothing - the document looked like it had lost its
    data. Walking the body's own child elements is the only way to get the two
    interleaved in reading order, which is also the order a reader needs them.
    """
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        return None, "python-docx is not installed on this box, so the .docx cannot be read back"
    try:
        doc = Document(str(path))
    except Exception as exc:                          # noqa: BLE001 - any reader failure
        return None, f"the .docx could not be opened: {exc}"

    parts: list[str] = []
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells]
                # A one-column table is a callout, not a grid; joining its
                # single cell with pipes would invent a column that is not
                # there.
                line = cells[0] if len(cells) == 1 else "  |  ".join(cells)
                if line.strip():
                    parts.append(line)
    return "\n\n".join(parts), None


def _read_xlsx(path: Path) -> tuple[str | None, str | None]:
    try:
        from openpyxl import load_workbook
    except ImportError:
        return None, "openpyxl is not installed on this box, so the .xlsx cannot be read back"
    try:
        # read_only keeps a large calculation sheet from being held in memory
        # whole just to show the first screenful of it.
        wb = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:                          # noqa: BLE001
        return None, f"the .xlsx could not be opened: {exc}"
    try:
        lines = []
        for sheet in wb.worksheets:
            lines.append(f"[{sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                lines.append(
                    ", ".join("" if c is None else str(c) for c in row).rstrip(", ")
                )
        return "\n".join(lines), None
    finally:
        wb.close()


def _read_pptx(path: Path) -> tuple[str | None, str | None]:
    try:
        from pptx import Presentation
    except ImportError:
        return None, "python-pptx is not installed on this box, so the .pptx cannot be read back"
    try:
        deck = Presentation(str(path))
    except Exception as exc:                          # noqa: BLE001
        return None, f"the .pptx could not be opened: {exc}"
    lines = []
    for number, slide in enumerate(deck.slides, start=1):
        lines.append(f"[slide {number}]")
        for shape in slide.shapes:
            text = getattr(shape, "text", "") or ""
            if text.strip():
                lines.append(text)
    return "\n".join(lines), None


_READERS = {".docx": _read_docx, ".xlsx": _read_xlsx, ".pptx": _read_pptx}


def _preview(path: Path, suffix: str) -> tuple[str | None, str, str | None]:
    """(text, source_of_the_text, why_there_is_none).

    `source_of_the_text` is what the panel labels the preview with. "file" means
    these characters came out of the bytes on disk; anything else is a weaker
    claim and has to be shown as one.
    """
    if not path.is_file():
        return None, "none", "the file is no longer in the run workspace"

    reader = _READERS.get(suffix)
    if reader is not None:
        text, why = reader(path)
        return (text, "file", None) if text is not None else (None, "none", why)

    if suffix in PLAIN_TEXT or suffix == "":
        text, why = _read_text_file(path)
        return (text, "file", None) if text is not None else (None, "none", why)

    return None, "none", (
        f"there is no reader on this box for {suffix or 'a file with no extension'}; "
        "download it to open it"
    )


# ------------------------------------------------------------------ the split

def _requested_body(steps: list[dict[str, Any]], rel_path: str) -> str | None:
    """The text a doc.write/fs.write step ASKED for, for this path.

    Only ever used as a labelled fallback when the file itself cannot be read.
    Kept separate from the file preview on purpose: confusing "what the run
    intended to write" with "what is in the file" would let a failed write show
    a perfect document.
    """
    for step in steps:
        args = step.get("args") or {}
        if str(args.get("path", "")).replace("\\", "/") != rel_path:
            continue
        for key in ("body", "content"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _exec_language(data: dict[str, Any], args: dict[str, Any], source: str) -> str:
    """What a code.exec step ran as, in order of authority.

    The observation is the record of what actually happened and wins outright.
    A step that never reached the sandbox has no observation, so the argument
    the plan passed is the next best thing, and detection is the last - a step
    still queued is drawn before anything has run it.
    """
    reported = data.get("language")
    if isinstance(reported, str) and reported:
        return reported
    requested = args.get("language")
    if isinstance(requested, str) and requested:
        try:
            return resolve_language(requested).id
        except UnknownLanguage:
            pass
    return detect_language(source)[0].id


def _code_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every snippet the run actually executed, with what it printed.

    These are not artifacts and never will be: code.exec runs on a tmpfs that is
    destroyed with the container, so nothing it writes reaches the workspace.
    The source and the stdout are the only surviving record that a sandboxed
    execution happened, which makes them the whole of the "a coding task run and
    verified in a sandbox" demo beat.
    """
    items: list[dict[str, Any]] = []
    for step in steps:
        if step.get("tool") != CODE_EXEC:
            continue
        args = step.get("args") or {}
        source = str(args.get("code", ""))
        source, source_truncated = _clip(source, MAX_SOURCE_CHARS)

        observations = step.get("observations") or []
        latest = observations[-1] if observations else {}
        data = latest.get("data") or {}
        stdout, stdout_truncated = _clip(str(latest.get("stdout") or ""), MAX_STDOUT_CHARS)

        items.append({
            "kind": "executed",
            "step_idx": step.get("idx"),
            "intent": step.get("intent", ""),
            "status": step.get("status"),
            # What the sandbox ACTUALLY ran it as, which is the only authority
            # on the question: the tool reports it back on every result. This
            # was hardcoded to "python" when code.exec was Python-only, so a C
            # snippet was syntax-highlighted as Python in the one panel a
            # reviewer reads the source in.
            "language": _exec_language(data, args, source),
            "source": source,
            "truncated": source_truncated,
            "stdout": stdout,
            "stdout_truncated": stdout_truncated,
            "stderr": str(latest.get("stderr") or "")[:MAX_STDOUT_CHARS],
            "exit_code": latest.get("exit_code"),
            # The tool's own verdict, which is NOT derivable from exit_code.
            # An Observation defaults exit_code to 0, so a code.exec step that
            # never reached the sandbox at all - no Docker, a policy denial, a
            # timeout - reports exit 0 and would be drawn with a green tick over
            # the stderr saying it failed.
            "ok": latest.get("ok") if observations else None,
            "ran": bool(observations),
            # False is not a detail. An unsandboxed execution is the dev
            # fallback, and the panel has to shout it rather than show the same
            # green tick a container run gets.
            "sandboxed": data.get("sandboxed"),
            # The tool's own note about WHY this failed the way it did - a
            # missing entry point, a toolchain that is not in the image. It is
            # the difference between "the model wrote bad code" and "this box
            # cannot run that language", and it is invisible unless carried.
            "hint": data.get("hint"),
        })
    return items


# A run that has not finished is missing nothing yet. This check comes FIRST in
# both notes below, ahead of the mode-specific wording: a code run is empty for
# its first ten seconds by construction, and telling the operator then that the
# toggle was not honoured makes every run look broken while it is working.
IN_PROGRESS = ("planning", "running", "awaiting_approval")


def _code_note(items: list[dict[str, Any]], mode: WorkMode, status: str) -> str | None:
    if items:
        return None
    if status in IN_PROGRESS:
        return "Nothing has been written or executed yet."
    if mode is WorkMode.CODE:
        return (
            "This run was started in Code mode and produced no code - no snippet "
            "was executed in the sandbox and no source file was written. The "
            "plan did not deliver what the toggle asked for; read the timeline "
            "for what it did instead."
        )
    return (
        "This run wrote no source files and executed nothing in the sandbox. "
        "Switch the toggle to Code before starting a run whose deliverable is a "
        "program."
    )


def _document_note(items: list[dict[str, Any]], mode: WorkMode, status: str) -> str | None:
    if items:
        return None
    if status in IN_PROGRESS:
        return "No document has been written yet."
    if mode is WorkMode.DOCUMENT:
        return (
            "This run was started in Document mode and produced no document "
            "file. Whatever it did, there is nothing here for anyone to open, "
            "read or sign; read the timeline for where it stopped."
        )
    return (
        "This run produced no document. Switch the toggle to Document before "
        "starting a run whose deliverable is a note, sheet or briefing."
    )


def split_outputs(timeline: dict[str, Any]) -> dict[str, Any]:
    """Partition a run timeline into its code side and its document side.

    Takes the timeline rather than a run id so it is pure with respect to the
    database: the API already has the timeline, and a test can hand this a dict
    without standing up a store.
    """
    run_id = timeline.get("run_id", "")
    status = str(timeline.get("status", ""))
    try:
        mode = WorkMode(timeline.get("mode", WorkMode.AUTO.value))
    except ValueError:
        mode = WorkMode.AUTO

    steps = list(timeline.get("steps") or [])
    workspace = Workspace(get_settings().workspace_root / run_id) if run_id else None

    code = _code_from_steps(steps)
    documents: list[dict[str, Any]] = []

    for art in timeline.get("artifacts") or []:
        rel = str(art.get("path", ""))
        suffix = Path(rel).suffix.lower()
        # Resolved THROUGH the workspace, exactly as the download route is. A
        # second way to turn a stored path into bytes is a second way out of the
        # workspace if it decides containment for itself.
        resolved: Path | None = None
        if workspace is not None:
            try:
                resolved = workspace.resolve(rel)
            except (WorkspaceViolation, ValueError):
                resolved = None

        common = {
            "path": rel,
            "name": Path(rel).name,
            "sha256": art.get("sha256", ""),
            "approved_by": art.get("approved_by"),
            "download": (
                f"/api/runs/{run_id}/artifacts/" + "/".join(rel.replace("\\", "/").split("/"))
            ),
        }

        language = _language(suffix)
        if language is not None:
            source: str | None = None
            why: str | None = None
            if resolved is None:
                why = "the artifact path does not resolve inside this run's workspace"
            else:
                source, why = _read_text_file(resolved) if resolved.is_file() else (
                    None, "the file is no longer in the run workspace"
                )
            truncated = False
            if source is not None:
                source, truncated = _clip(source, MAX_SOURCE_CHARS)
            code.append({
                **common,
                "kind": "file",
                "language": language,
                "source": source,
                "source_note": why,
                "truncated": truncated,
            })
            continue

        text: str | None = None
        text_source = "none"
        why = "the artifact path does not resolve inside this run's workspace"
        if resolved is not None:
            text, text_source, why = _preview(resolved, suffix)

        if text is None:
            # Last resort, and labelled as the weaker claim it is.
            requested = _requested_body(steps, rel.replace("\\", "/"))
            if requested is not None:
                text, text_source = requested, "requested"

        truncated = False
        if text is not None:
            text, truncated = _clip(text, MAX_PREVIEW_CHARS)

        documents.append({
            **common,
            "kind": "file",
            "format": suffix.lstrip(".") or "file",
            "preview": text,
            # "file"      - read back out of the bytes on disk
            # "requested" - the text the run ASKED to write; the file itself
            #               could not be read, and the panel must say so
            # "none"      - no preview; `preview_note` is why
            "preview_source": text_source,
            "preview_note": why if text_source != "file" else None,
            "truncated": truncated,
        })

    return {
        "run_id": run_id,
        "mode": mode.value,
        "status": status,
        "code": {
            "count": len(code),
            "items": code,
            "note": _code_note(code, mode, status),
        },
        "documents": {
            "count": len(documents),
            "items": documents,
            "note": _document_note(documents, mode, status),
        },
    }

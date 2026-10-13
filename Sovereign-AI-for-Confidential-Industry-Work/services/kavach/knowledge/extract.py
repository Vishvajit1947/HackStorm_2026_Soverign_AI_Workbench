"""The ingest fork (§07).

One decision, made per page, that everything downstream depends on:

    does this page carry its own text, or is it a picture of text?

A native PDF page is read exactly - the characters in the index are the
characters in the file. A scanned page has to be transcribed by something, and
whatever transcribes it can be wrong. Those two are not interchangeable, so they
are not recorded as the same thing: every page carries how it was read, every
chunk inherits it, and every citation renders it.

The fork, in order:

    1. text layer >= min_native_chars      -> NATIVE
    2. a real OCR engine on this box       -> OCR
    3. a real vision model (not simulated) -> VLM
    4. the mock backend + explicit opt-in  -> SIMULATED
    5. nothing                             -> UNREADABLE

Step 5 is the one that matters. It would be trivial to return "" for a page we
cannot read; the page would join the index as an empty document, contribute
nothing, and the corpus would report 11 documents ingested with no asterisk. The
knowledge plane would then answer questions from a corpus with a hole in it and
have no way to mention the hole. So an unreadable page is recorded as unreadable,
counted, listed by page number, and surfaced on every search that touches its
document.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import GenRequest, Label
from .types import DocumentRecord, Extraction, PageRecord, slugify_doc_id

try:  # pymupdf ships in the `knowledge` extra; the rest of Kavach does not need it
    import pymupdf

    HAVE_PYMUPDF = True
except ImportError:  # pragma: no cover - exercised on a box without the extra
    pymupdf = None
    HAVE_PYMUPDF = False

# Mirrors KAVACH_ALLOW_UNSANDBOXED_EXEC: the degraded path exists, it is off by
# default, it must be asked for by name, and everything it produces is stamped.
SIMULATED_OCR_ENV = "KAVACH_KB_ALLOW_SIMULATED_OCR"

# The MRPL header block carries the marking on the document's own face.
_MARKING = re.compile(
    r"CLASSIFICATION\s*[:\-]?\s*(PUBLIC|INTERNAL|CONFIDENTIAL|RESTRICTED)",
    re.IGNORECASE,
)

# A numbered heading at the start of a line: "2. SAFETY INTERLOCKS", "4.1 Scope".
_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{3,80})$", re.MULTILINE)

VLM_PROMPT = (
    "Transcribe every character visible in this page image, preserving headings, "
    "table rows and instrument tags. Do not summarise, do not explain, do not "
    "invent text that is not visible. If part of the page is illegible, write "
    "[illegible] rather than guessing."
)


def _vlm_prompt(hint: str) -> str:
    """The transcription instruction, plus the filename LABELLED as context.

    The hint used to be appended as a bare line:

        Transcribe every character visible in this page image...

        PID_01.png

    which is an instruction followed by a string - and a vision model reading
    that transcribes the string. A real P&ID went in and `PID_01.png` came out,
    19 characters, recorded as a successful VLM extraction. The page was never
    looked at and nothing said so.

    Labelling it costs one sentence and removes the ambiguity entirely.
    """
    if not hint.strip():
        return VLM_PROMPT
    return (
        f"{VLM_PROMPT}\n\n"
        f"(Context only - this page came from a file named {hint.strip()}. "
        f"Do not include that filename in your transcription.)"
    )


def _is_echo(text: str, hint: str) -> bool:
    """Did the model hand the hint back instead of reading the page?

    The failure this catches is the worst kind available here: a confident,
    successful-looking extraction of a page nobody looked at. It is cheap to
    detect and it must never be recorded as VLM output.
    """
    if not hint.strip():
        return False
    answer = text.strip().strip('".\'').lower()
    name = hint.strip().lower()
    stem = name.rsplit(".", 1)[0]
    return answer in {name, stem} or (len(answer) <= len(name) + 8 and stem in answer)


# (page just finished, pages in this document, what is happening) -> None.
# Optional at every level: nothing in the read path may depend on somebody
# watching, and a callback that raises must never cost a page.
ProgressFn = Callable[[int, int, str], None]


def _report(on_progress: ProgressFn | None, page: int, pages: int, detail: str) -> None:
    if on_progress is None:
        return
    try:
        on_progress(page, pages, detail)
    except Exception:                       # noqa: BLE001 - see below
        # Swallowed on purpose and not logged: this runs once per page, a
        # listener that raises would raise on every one of them, and a document
        # must never be lost because somebody was watching it being read.
        return


class IngestError(RuntimeError):
    """A file could not be opened at all. Distinct from a page we could not read."""


def simulated_ocr_allowed() -> bool:
    return os.environ.get(SIMULATED_OCR_ENV, "").strip().lower() in {"1", "true", "yes"}


# Where the official Windows installer puts Tesseract. It does NOT add itself
# to PATH unless the box is ticked, and almost nobody ticks it - so a machine
# with Tesseract properly installed reported "not installed", and the panel
# told its owner to install the thing they had just installed.
#
# A short list of standard locations, tried AFTER PATH and after the explicit
# setting, is not the same as hardcoding what the box may run: it finds a
# default install and nothing else, and `KAVACH_TESSERACT` overrides it.
_TESSERACT_DEFAULTS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def tesseract_binary() -> str | None:
    """The OCR engine's executable, or None. PATH, then config, then defaults.

    Returns the PATH so callers run the binary they found rather than the bare
    name - resolving it and then invoking "tesseract" would find it on one
    machine and not on the next.
    """
    found = shutil.which("tesseract")
    if found:
        return found

    configured = os.environ.get("KAVACH_TESSERACT", "").strip()
    if configured:
        candidate = Path(configured)
        # Either the executable itself or the folder holding it: an operator
        # setting this will reasonably do either.
        for option in (candidate, candidate / "tesseract.exe", candidate / "tesseract"):
            if option.is_file():
                return str(option)

    for default in _TESSERACT_DEFAULTS:
        if Path(default).is_file():
            return default
    return None


def tesseract_available() -> bool:
    """A real OCR engine on this box, not a Python wrapper around a missing one."""
    return tesseract_binary() is not None


# ---------------------------------------------------------------- the image path


@dataclass
class ImagePathStatus:
    """What this box can do with a page it cannot read as text."""

    ocr_engine: str | None
    vision_model: str | None
    vision_is_simulated: bool
    simulated_allowed: bool

    @property
    def available(self) -> bool:
        if self.ocr_engine:
            return True
        if self.vision_model and not self.vision_is_simulated:
            return True
        return bool(self.vision_model and self.simulated_allowed)

    @property
    def mode(self) -> Extraction:
        if self.ocr_engine:
            return Extraction.OCR
        if self.vision_model and not self.vision_is_simulated:
            return Extraction.VLM
        if self.vision_model and self.simulated_allowed:
            return Extraction.SIMULATED
        return Extraction.UNREADABLE

    @property
    def detail(self) -> str:
        if self.ocr_engine:
            return f"OCR via {self.ocr_engine}"
        if self.vision_model and not self.vision_is_simulated:
            return f"transcribed by {self.vision_model}"
        if self.vision_model and self.simulated_allowed:
            return (
                f"{self.vision_model} is SIMULATED - this text was not read off "
                "the page and may not be quoted"
            )
        if self.vision_model:
            return (
                f"no OCR engine on this box, and the only vision model "
                f"({self.vision_model}) is SIMULATED. Set {SIMULATED_OCR_ENV}=1 to "
                "index synthetic transcriptions anyway - they are stamped and "
                "excluded from grounded answers."
            )
        return "no OCR engine and no vision model available on this profile"

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "mode": self.mode.value,
            "ocr_engine": self.ocr_engine,
            "vision_model": self.vision_model,
            "vision_is_simulated": self.vision_is_simulated,
            "simulated_allowed": self.simulated_allowed,
            "detail": self.detail,
        }


class ImageReader:
    """Turns page pixels into text, or says honestly that it cannot.

    Holds the registry/warden/backend rather than reaching for the singletons at
    call time, so ingest can be pointed at fakes in tests without monkeypatching
    module globals.
    """

    def __init__(self, registry=None, warden=None, backend=None):
        self._registry = registry
        self._warden = warden
        self._backend = backend

    # ------------------------------------------------------------ availability

    @property
    def registry(self):
        if self._registry is None:
            from ..runtime import get_registry

            self._registry = get_registry()
        return self._registry

    @property
    def warden(self):
        if self._warden is None:
            from ..runtime import get_warden

            self._warden = get_warden()
        return self._warden

    @property
    def backend(self):
        if self._backend is None:
            from ..runtime import get_backend

            self._backend = get_backend()
        return self._backend

    def _vision_spec(self):
        """The best vision model this profile may use, or None."""
        try:
            available = self.warden.profile.available
        except Exception:  # noqa: BLE001 - a missing warden must degrade, not raise
            available = None
        specs = [
            s
            for s in self.registry.available(available)
            # has_projector, not just has_vision: a VL model with no projector
            # loads fine and never looks at the picture, so picking one here
            # would send every scan down a path that refuses it - or worse,
            # answers about an image it never saw.
            if s.role == "generation" and s.has_vision and s.routable
            and s.has_projector
        ]
        if not specs:
            return None
        # Highest tier wins. Transcription quality is the whole job here and a
        # page is read once and indexed forever, so the swap penalty that
        # dominates a per-request routing decision (§04) does not apply.
        return max(specs, key=lambda s: (s.tier, s.quality_for("visual_doc")))

    def status(self) -> ImagePathStatus:
        engine = "tesseract" if tesseract_available() else None
        spec = self._vision_spec()
        simulated = False
        if spec is not None:
            from ..runtime.registry import Provenance

            simulated = spec.provenance is Provenance.SIMULATED
        return ImagePathStatus(
            ocr_engine=engine,
            vision_model=spec.id if spec else None,
            vision_is_simulated=simulated,
            simulated_allowed=simulated_ocr_allowed(),
        )

    # ------------------------------------------------------------------- read

    def read(self, png: bytes, *, hint: str = "") -> tuple[str, Extraction, str]:
        """Transcribe one page image.

        Never raises: a failed transcription is an UNREADABLE page with a reason,
        not an exception that aborts the whole corpus half-ingested. That contract
        is load-bearing - extract_document has no handler of its own - so EVERY
        call out of this method is guarded, including the ones that only look up
        which model to use. A warden that throws while picking a model would
        otherwise take down an eleven-document ingest on its seventh file.
        """
        try:
            status = self.status()
        except Exception as exc:  # noqa: BLE001
            return "", Extraction.UNREADABLE, f"image path unavailable: {exc}"
        mode = status.mode

        if mode is Extraction.UNREADABLE:
            return "", Extraction.UNREADABLE, status.detail

        if mode is Extraction.OCR:
            text, detail = self._run_tesseract(png)
            if not text.strip():
                return "", Extraction.UNREADABLE, detail or "OCR produced no text"
            return text, Extraction.OCR, status.detail

        try:
            spec = self._vision_spec()
            self.warden.admit(spec.id)
            result = self.backend.generate(
                spec,
                GenRequest(
                    prompt=_vlm_prompt(hint),
                    max_tokens=2048,
                    temperature=0.0,
                    images=[base64.b64encode(png).decode("ascii")],
                ),
            )
        except Exception as exc:  # noqa: BLE001 - one bad page must not kill a corpus
            return "", Extraction.UNREADABLE, f"{type(exc).__name__}: {exc}"

        text = result.text.strip()
        if not text:
            return "", Extraction.UNREADABLE, "the vision model returned nothing"
        if _is_echo(text, hint):
            # A page recorded as VLM whose "transcription" is its own filename
            # is a page nobody read, filed as one that was read. UNREADABLE is
            # the honest verdict, and the gap count is what surfaces it.
            return "", Extraction.UNREADABLE, (
                f"the vision model returned the file name ({text[:60]!r}) rather "
                "than the contents of the page, so nothing was actually read "
                "from it"
            )
        return text, mode, status.detail

    @staticmethod
    def _run_tesseract(png: bytes) -> tuple[str, str]:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory(prefix="kavach-ocr-") as tmp:
            src = Path(tmp) / "page.png"
            src.write_bytes(png)
            try:
                binary = tesseract_binary()
                if binary is None:
                    return "", "the OCR engine went away between the check and the run"
                out = subprocess.run(
                    [binary, str(src), "stdout", "--psm", "1"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                return "", f"tesseract failed: {exc}"
        if out.returncode != 0:
            return "", f"tesseract exit {out.returncode}: {out.stderr.strip()[:200]}"
        return out.stdout, ""


# ---------------------------------------------------------------- the fork


@dataclass
class ExtractedPage:
    page: int
    text: str
    extraction: Extraction
    detail: str = ""

    def as_record(self) -> PageRecord:
        return PageRecord(
            page=self.page,
            extraction=self.extraction,
            chars=len(self.text),
            detail=self.detail,
        )


@dataclass
class ExtractedDocument:
    record: DocumentRecord
    pages: list[ExtractedPage]


def detect_label(text: str) -> Label | None:
    """The marking on the document's own face. None when there is no marking."""
    match = _MARKING.search(text)
    if not match:
        return None
    return Label(match.group(1).upper())


def section_headings(text: str) -> list[tuple[int, str]]:
    """(offset, section number) for each numbered heading, in document order."""
    return [(m.start(), m.group(1)) for m in _HEADING.finditer(text)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 8:
            return stripped[:160]
    return fallback


def extract_document(
    path: Path,
    config: dict[str, Any],
    *,
    reader: ImageReader | None = None,
    label: Label | None = None,
    doc_id: str | None = None,
    on_progress: ProgressFn | None = None,
) -> ExtractedDocument:
    """Read one file into pages, forking per page between text and image paths."""
    if not HAVE_PYMUPDF:
        raise IngestError(
            "the knowledge plane needs PyMuPDF to open documents. Install it "
            'offline with: pip install -e ".[knowledge]"'
        )

    path = Path(path)
    if not path.exists():
        raise IngestError(f"{path} does not exist")

    ingest = config.get("ingest", {})
    min_native = int(ingest.get("min_native_chars", 120))
    dpi = int(ingest.get("render_dpi", 200))
    image_exts = {e.lower() for e in ingest.get("image_extensions", [])}
    pdf_exts = {e.lower() for e in ingest.get("pdf_extensions", [".pdf"])}
    suffix = path.suffix.lower()

    if suffix not in pdf_exts and suffix not in image_exts:
        raise IngestError(
            f"{path.name}: {suffix or 'no extension'} is not a document type this "
            f"box can open; expected one of {sorted(pdf_exts | image_exts)}"
        )

    reader = reader or ImageReader()
    pages: list[ExtractedPage] = []

    if suffix in image_exts:
        # A bare image is the image path by definition - there is no text layer
        # to fork on.
        _report(on_progress, 1, 1,
                "reading the picture - this is the slow part")
        text, extraction, detail = reader.read(path.read_bytes(), hint=path.name)
        pages.append(ExtractedPage(1, text, extraction, detail))
    else:
        try:
            opened = pymupdf.open(path)
        except Exception as exc:
            # PyMuPDF raises several unrelated types here.
            # Re-raised as IngestError so the orchestrator's one handler catches
            # it. A truncated or password-locked PDF that escapes as its own
            # exception type aborts the whole corpus load on file seven, and
            # everything already ingested looks like the complete set.
            raise IngestError(
                f"{path.name}: PyMuPDF could not open this file "
                f"({type(exc).__name__}: {exc}); it may be truncated, encrypted "
                "or not a PDF at all"
            ) from exc
        with opened as doc:
            total = doc.page_count
            for number, page in enumerate(doc, start=1):
                _report(on_progress, number, total,
                        f"reading page {number} of {total}")
                native = page.get_text().strip()
                if len(native) >= min_native:
                    pages.append(
                        ExtractedPage(number, native, Extraction.NATIVE, "PDF text layer")
                    )
                    continue
                png = page.get_pixmap(dpi=dpi).tobytes("png")
                # Named separately because this is the slow one: a page with no
                # text layer goes to OCR or a vision model, and on a mid-range
                # card that is minutes. The person waiting should be told which
                # page is costing them the time.
                _report(on_progress, number, total,
                        f"page {number} of {total} has no text layer - "
                        "reading the picture, this is the slow part")
                text, extraction, detail = reader.read(
                    png, hint=f"{path.name} page {number}"
                )
                if native and extraction is Extraction.UNREADABLE:
                    # There WAS a text layer, just a thin one. Say so: "no text at
                    # all" and "18 chars of scanner footer" are different diagnoses
                    # and the second one is the tell that min_native_chars is
                    # earning its keep.
                    detail = (
                        f"{detail} (page carries only {len(native)} chars of text "
                        f"layer, under the {min_native}-char native threshold)"
                    )
                pages.append(ExtractedPage(number, text, extraction, detail))

    readable_text = "\n".join(p.text for p in pages if p.extraction.is_real)
    marking = detect_label(readable_text)
    # Not just UNREADABLE: a page whose only text is a SIMULATED transcription
    # was not read either, so a marking sitting on it was not read either. Both
    # cases are "nobody knows what this document is classified as".
    unread = any(not p.extraction.is_real for p in pages)

    if label is not None:
        resolved = label
    elif marking is not None:
        resolved = marking
    elif unread:
        # See knowledge.yaml: over-classify what could not be read.
        resolved = Label(ingest.get("unclassified_label", "CONFIDENTIAL"))
    else:
        resolved = Label(ingest.get("default_label", "INTERNAL"))

    record = DocumentRecord(
        id=doc_id or slugify_doc_id(path.stem),
        path=str(path),
        title=_title_from(readable_text, path.stem),
        sha256=_sha256(path),
        label=resolved,
        pages=[p.as_record() for p in pages],
    )
    return ExtractedDocument(record=record, pages=pages)

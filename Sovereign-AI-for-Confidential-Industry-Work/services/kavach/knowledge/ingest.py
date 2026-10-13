"""Ingest orchestration (§07).

Walks a file or a directory, runs the fork in `extract.py` on every page, chunks
what came back, writes it to the store and rebuilds the index.

Every document is audited - including the ones that failed. A corpus load that
half-worked and reported "9 documents ingested" is the kind of thing nobody
notices until a question comes back with no answer and no explanation. Here, a
file that could not be opened is a row in the report and a record in the chain,
with the reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audit import get_audit
from ..types import Label
from .chunk import chunk_document
from .config import load_knowledge_config
from .extract import (
    SIMULATED_OCR_ENV,
    ImageReader,
    IngestError,
    ProgressFn,
    extract_document,
)
from .index import HybridIndex, get_index
from .store import KnowledgeStore, get_knowledge_store
from .types import Extraction


@dataclass
class DocumentOutcome:
    path: str
    doc_id: str | None = None
    ok: bool = False
    skipped: bool = False
    reason: str = ""
    pages: int = 0
    chunks: int = 0
    label: str = ""
    extraction: dict[str, int] = field(default_factory=dict)
    unreadable_pages: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "doc_id": self.doc_id,
            "ok": self.ok,
            "skipped": self.skipped,
            "reason": self.reason,
            "pages": self.pages,
            "chunks": self.chunks,
            "label": self.label,
            "extraction": self.extraction,
            "unreadable_pages": self.unreadable_pages,
        }


@dataclass
class IngestReport:
    documents: list[DocumentOutcome] = field(default_factory=list)
    elapsed_s: float = 0.0
    index: dict[str, Any] = field(default_factory=dict)

    @property
    def ingested(self) -> int:
        return sum(1 for d in self.documents if d.ok and not d.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for d in self.documents if d.skipped)

    @property
    def failed(self) -> list[DocumentOutcome]:
        return [d for d in self.documents if not d.ok and not d.skipped]

    @property
    def unreadable_pages(self) -> int:
        return sum(len(d.unreadable_pages) for d in self.documents)

    @property
    def simulated_pages(self) -> int:
        return sum(d.extraction.get("SIMULATED", 0) for d in self.documents)

    @property
    def warning(self) -> str | None:
        bits: list[str] = []
        if self.failed:
            bits.append(f"{len(self.failed)} file(s) could not be opened at all")
        if self.unreadable_pages:
            docs = [d for d in self.documents if d.unreadable_pages]
            bits.append(
                f"{self.unreadable_pages} page(s) across {len(docs)} document(s) "
                "could not be read - they are NOT in the index"
            )
        if self.simulated_pages:
            # An ingest that reports "3 documents, 3 pages" and then indexes
            # nothing is the most confusing outcome this code can produce. Name
            # the reason here rather than leaving it to be inferred from a chunk
            # count of zero two lines further down.
            bits.append(
                f"{self.simulated_pages} page(s) hold SIMULATED transcriptions "
                f"({SIMULATED_OCR_ENV} is set) - they are NOT indexed and nothing "
                "can be cited to them"
            )
        return "; ".join(bits) or None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ingested": self.ingested,
            "skipped": self.skipped,
            "failed": len(self.failed),
            "unreadable_pages": self.unreadable_pages,
            "simulated_pages": self.simulated_pages,
            "elapsed_s": round(self.elapsed_s, 2),
            "warning": self.warning,
            "documents": [d.as_dict() for d in self.documents],
            "index": self.index,
        }


def _report_file(on_progress: ProgressFn, position: int, total: int, name: str) -> None:
    try:
        on_progress(0, 0, f"file {position} of {total}: {name}")
    except Exception:                       # noqa: BLE001
        # Same rule as _report in extract.py: watching must not be able to
        # break the thing being watched.
        return


class Ingestor:
    def __init__(
        self,
        store: KnowledgeStore | None = None,
        index: HybridIndex | None = None,
        reader: ImageReader | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._store = store
        self._index = index
        self.reader = reader or ImageReader()
        self._config = config

    @property
    def store(self) -> KnowledgeStore:
        if self._store is None:
            self._store = get_knowledge_store()
        return self._store

    @property
    def index(self) -> HybridIndex:
        if self._index is None:
            self._index = get_index()
        return self._index

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_knowledge_config()
        return self._config

    # ---------------------------------------------------------------- walk

    def discover(self, target: Path) -> list[Path]:
        """Every file under `target` this box knows how to open, sorted."""
        ingest = self.config.get("ingest", {})
        known = {e.lower() for e in ingest.get("pdf_extensions", [".pdf"])}
        known |= {e.lower() for e in ingest.get("image_extensions", [])}

        target = Path(target)
        if target.is_file():
            return [target]
        return sorted(
            p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in known
        )

    # ---------------------------------------------------------------- ingest

    def ingest_file(
        self, path: Path, *, label: Label | None = None, force: bool = False,
        on_progress: ProgressFn | None = None,
    ) -> DocumentOutcome:
        audit = get_audit()
        path = Path(path)
        outcome = DocumentOutcome(path=str(path))

        try:
            extracted = extract_document(
                path, self.config, reader=self.reader, label=label,
                on_progress=on_progress,
            )
        except IngestError as exc:
            outcome.reason = str(exc)
            audit.append("kb.ingest_failed", path=str(path), reason=str(exc))
            return outcome

        record = extracted.record
        outcome.doc_id = record.id

        existing = self.store.document(record.id)
        if existing and existing.sha256 == record.sha256 and not force:
            outcome.ok = True
            outcome.skipped = True
            outcome.reason = "already ingested, identical sha256"
            outcome.pages = existing.page_count
            outcome.label = existing.label.value
            outcome.unreadable_pages = existing.unreadable_pages
            return outcome

        chunks = chunk_document(
            extracted.pages,
            doc_id=record.id,
            label=record.label,
            config=self.config,
        )
        self.store.put_document(record, chunks)

        outcome.ok = True
        outcome.pages = record.page_count
        outcome.chunks = len(chunks)
        outcome.label = record.label.value
        outcome.unreadable_pages = record.unreadable_pages
        outcome.extraction = {
            kind.value: sum(1 for p in record.pages if p.extraction is kind)
            for kind in Extraction
            if any(p.extraction is kind for p in record.pages)
        }

        # The document's own text is CONFIDENTIAL more often than not, so the
        # chain records the sha256 and the shape - never the content. Same rule
        # the tool executor applies to arguments (§06).
        audit.append(
            "kb.document_ingested",
            data_class=record.label.value,
            doc_id=record.id,
            path=str(path),
            sha256=record.sha256,
            pages=record.page_count,
            chunks=len(chunks),
            extraction=outcome.extraction,
            unreadable_pages=record.unreadable_pages,
        )
        return outcome

    def ingest(
        self,
        target: Path,
        *,
        label: Label | None = None,
        force: bool = False,
        rebuild: bool = True,
        on_progress: ProgressFn | None = None,
    ) -> IngestReport:
        started = time.perf_counter()
        audit = get_audit()
        target = Path(target)

        files = self.discover(target)
        audit.append("kb.ingest_start", path=str(target), files=len(files))

        report = IngestReport()
        for position, path in enumerate(files, start=1):
            # A folder ingest reports which FILE it is on as well as which page,
            # so a corpus of eleven documents does not look stuck on the third.
            if on_progress and len(files) > 1:
                _report_file(on_progress, position, len(files), path.name)
            report.documents.append(
                self.ingest_file(path, label=label, force=force,
                                 on_progress=on_progress)
            )

        if rebuild:
            report.index = self.index.build().as_dict()
        else:
            report.index = self.index.state().as_dict()

        report.elapsed_s = time.perf_counter() - started
        audit.append(
            "kb.ingest_done",
            path=str(target),
            ingested=report.ingested,
            skipped=report.skipped,
            failed=len(report.failed),
            unreadable_pages=report.unreadable_pages,
            chunks_indexed=report.index.get("chunks"),
        )
        return report

    # ---------------------------------------------------------------- maintenance

    def forget(self, doc_id: str, *, rebuild: bool = True) -> bool:
        removed = self.store.forget(doc_id)
        if removed:
            get_audit().append("kb.document_forgotten", doc_id=doc_id)
            if rebuild:
                self.index.build()
        return removed

    def clear(self) -> int:
        count = self.store.clear()
        get_audit().append("kb.cleared", documents=count)
        self.index.build()
        return count


_ingestor: Ingestor | None = None


def get_ingestor() -> Ingestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = Ingestor()
    return _ingestor


def reset_ingestor() -> None:
    """Tests only."""
    global _ingestor
    _ingestor = None

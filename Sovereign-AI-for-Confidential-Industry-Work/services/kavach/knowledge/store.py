"""Durable record of what was ingested (§07).

The index in `index.py` is derived data - delete it and it rebuilds. This is not:
it is the answer to "which document is `[SOP_01 p.2]`, was that page actually
read, and what is it classified as". A citation that cannot be resolved back to
a row here is not a citation.

Kept in the same SQLite file as the runs and the audit chain, deliberately: one
file is what an auditor can be handed.
"""

from __future__ import annotations

import time
from typing import Any

from ..store import Database, get_db
from ..types import Label
from .types import Chunk, DocumentRecord, Extraction, PageRecord


class KnowledgeStore:
    def __init__(self, db: Database | None = None):
        self.db = db or get_db()

    # ---------------------------------------------------------------- write

    def put_document(self, record: DocumentRecord, chunks: list[Chunk]) -> None:
        """Insert or replace a document, its pages and its chunks, atomically.

        Re-ingesting a document REPLACES it rather than adding a second copy.
        Two copies of the same SOP in the index means a retrieval returns the
        same clause twice, which reads to a user as corroboration from two
        sources - the most quietly wrong thing a knowledge plane can do.
        """
        record.ingested_at = record.ingested_at or time.time()
        conn = self.db.conn
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (record.id,))
            conn.execute("DELETE FROM kb_pages WHERE doc_id=?", (record.id,))
            conn.execute("DELETE FROM kb_documents WHERE id=?", (record.id,))
            conn.execute(
                """INSERT INTO kb_documents (id, path, title, sha256, label, ingested_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    record.id,
                    record.path,
                    record.title,
                    record.sha256,
                    record.label.value,
                    record.ingested_at,
                ),
            )
            conn.executemany(
                """INSERT INTO kb_pages (doc_id, page, extraction, chars, detail)
                   VALUES (?,?,?,?,?)""",
                [
                    (record.id, p.page, p.extraction.value, p.chars, p.detail)
                    for p in record.pages
                ],
            )
            conn.executemany(
                """INSERT INTO kb_chunks (id, doc_id, page, ordinal, section, label,
                                          extraction, text)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (
                        c.id,
                        c.doc_id,
                        c.page,
                        c.ordinal,
                        c.section,
                        c.label.value,
                        c.extraction.value,
                        c.text,
                    )
                    for c in chunks
                ],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def forget(self, doc_id: str) -> bool:
        """Remove one document. Returns False when there was nothing to remove."""
        row = self.db.one("SELECT id FROM kb_documents WHERE id=?", (doc_id,))
        if row is None:
            return False
        self.db.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM kb_pages WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM kb_documents WHERE id=?", (doc_id,))
        return True

    def clear(self) -> int:
        count = len(self.documents())
        self.db.execute("DELETE FROM kb_chunks")
        self.db.execute("DELETE FROM kb_pages")
        self.db.execute("DELETE FROM kb_documents")
        return count

    # ---------------------------------------------------------------- read

    def _pages(self, doc_id: str) -> list[PageRecord]:
        return [
            PageRecord(
                page=r["page"],
                extraction=Extraction(r["extraction"]),
                chars=r["chars"],
                detail=r["detail"],
            )
            for r in self.db.query(
                "SELECT * FROM kb_pages WHERE doc_id=? ORDER BY page", (doc_id,)
            )
        ]

    def _document(self, row: Any) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            path=row["path"],
            title=row["title"],
            sha256=row["sha256"],
            label=Label(row["label"]),
            pages=self._pages(row["id"]),
            ingested_at=row["ingested_at"],
        )

    def document(self, doc_id: str) -> DocumentRecord | None:
        row = self.db.one("SELECT * FROM kb_documents WHERE id=?", (doc_id,))
        return self._document(row) if row else None

    def documents(self) -> list[DocumentRecord]:
        return [
            self._document(r)
            for r in self.db.query("SELECT * FROM kb_documents ORDER BY id")
        ]

    def document_by_sha(self, sha256: str) -> DocumentRecord | None:
        row = self.db.one("SELECT * FROM kb_documents WHERE sha256=?", (sha256,))
        return self._document(row) if row else None

    @staticmethod
    def _chunk(row: Any) -> Chunk:
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            page=row["page"],
            ordinal=row["ordinal"],
            text=row["text"],
            section=row["section"],
            label=Label(row["label"]),
            extraction=Extraction(row["extraction"]),
        )

    def chunks(self) -> list[Chunk]:
        return [
            self._chunk(r)
            for r in self.db.query("SELECT * FROM kb_chunks ORDER BY doc_id, ordinal")
        ]

    def chunk(self, chunk_id: str) -> Chunk | None:
        row = self.db.one("SELECT * FROM kb_chunks WHERE id=?", (chunk_id,))
        return self._chunk(row) if row else None

    def chunks_on_page(self, doc_id: str, page: int) -> list[Chunk]:
        return [
            self._chunk(r)
            for r in self.db.query(
                "SELECT * FROM kb_chunks WHERE doc_id=? AND page=? ORDER BY ordinal",
                (doc_id, page),
            )
        ]

    def page(self, doc_id: str, page: int) -> PageRecord | None:
        row = self.db.one(
            "SELECT * FROM kb_pages WHERE doc_id=? AND page=?", (doc_id, page)
        )
        if row is None:
            return None
        return PageRecord(
            page=row["page"],
            extraction=Extraction(row["extraction"]),
            chars=row["chars"],
            detail=row["detail"],
        )

    # ---------------------------------------------------------------- summary

    def stats(self) -> dict[str, Any]:
        """What the corpus panel renders - including, prominently, the gaps."""
        docs = self.documents()
        extraction: dict[str, int] = {}
        for row in self.db.query(
            "SELECT extraction, COUNT(*) AS n FROM kb_pages GROUP BY extraction"
        ):
            extraction[row["extraction"]] = row["n"]

        labels: dict[str, int] = {}
        for row in self.db.query(
            "SELECT label, COUNT(*) AS n FROM kb_documents GROUP BY label"
        ):
            labels[row["label"]] = row["n"]

        chunk_row = self.db.one("SELECT COUNT(*) AS n FROM kb_chunks")
        unreadable = extraction.get(Extraction.UNREADABLE.value, 0)
        simulated = extraction.get(Extraction.SIMULATED.value, 0)
        total_pages = sum(extraction.values())

        gaps = [
            {"doc_id": d.id, "pages": d.unreadable_pages}
            for d in docs
            if d.unreadable_pages
        ]

        warning = None
        if unreadable:
            warning = (
                f"{unreadable} of {total_pages} page(s) across {len(gaps)} document(s) "
                "could not be read on this box. Those pages are NOT in the index and "
                "no answer can be grounded in them - see `kavach kb doctor`."
            )
        elif simulated:
            warning = (
                f"{simulated} page(s) hold SIMULATED transcriptions, not text read "
                "off the page. They are excluded from grounded answers."
            )

        return {
            "documents": len(docs),
            "pages": total_pages,
            "chunks": chunk_row["n"] if chunk_row else 0,
            "extraction": extraction,
            "labels": labels,
            "unreadable_pages": unreadable,
            "simulated_pages": simulated,
            "gaps": gaps,
            "warning": warning,
        }


_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


def reset_knowledge_store() -> None:
    """Tests only."""
    global _store
    _store = None

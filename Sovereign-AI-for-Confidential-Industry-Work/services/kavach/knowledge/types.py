"""Vocabulary of the knowledge plane (§07).

The type that matters here is `Extraction`. Everything else is bookkeeping.

A chunk's text either came off the page or it did not, and the difference is not
a quality gradient - it is the difference between a quotation and a paraphrase of
something nobody read. The registry made the same distinction for weights
(VERIFIED vs SIMULATED, §03) and for the same reason: a value that can be
mistaken for its stronger neighbour will be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..types import Label


class Extraction(str, Enum):
    """How a page's text was obtained. Carried on every chunk and every hit."""

    NATIVE = "NATIVE"          # the PDF's own text layer - exact, byte for byte
    OCR = "OCR"                # a real OCR engine on this box read the pixels
    VLM = "VLM"                # a real vision model transcribed the page
    SIMULATED = "SIMULATED"    # the VLM path ran on the mock backend: synthetic
    UNREADABLE = "UNREADABLE"  # nothing on this box could read it. A known gap.

    @property
    def is_real(self) -> bool:
        """True only when a human could open the page and find this text on it."""
        return self in (Extraction.NATIVE, Extraction.OCR, Extraction.VLM)

    @property
    def is_indexed(self) -> bool:
        """UNREADABLE pages have no text, so there is nothing to index. SIMULATED
        pages do get indexed - behind an opt-in - so the pipeline can be
        exercised end to end, but every hit and every citation is stamped."""
        return self is not Extraction.UNREADABLE


@dataclass
class PageRecord:
    """One page of one document, and how it was read."""

    page: int
    extraction: Extraction
    chars: int = 0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "extraction": self.extraction.value,
            "chars": self.chars,
            "detail": self.detail,
        }


@dataclass
class Chunk:
    """An indexed span of one page. Never spans two pages - see knowledge.yaml."""

    id: str
    doc_id: str
    page: int
    ordinal: int
    text: str
    section: str = ""
    label: Label = Label.INTERNAL
    extraction: Extraction = Extraction.NATIVE

    @property
    def citation(self) -> str:
        """The inline form a model is told to emit. ASCII only."""
        base = f"{self.doc_id} p.{self.page}"
        if self.section:
            base = f"{base} s.{self.section}"
        if self.extraction is Extraction.SIMULATED:
            # A simulated transcription is not a quotation. Anything rendering
            # this string - CLI, panel, model context, the run sheet - says so.
            return f"[SIMULATED {base}]"
        return f"[{base}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "page": self.page,
            "ordinal": self.ordinal,
            "section": self.section,
            "label": self.label.value,
            "extraction": self.extraction.value,
            "citation": self.citation,
            "text": self.text,
        }


@dataclass
class DocumentRecord:
    """One ingested file."""

    id: str
    path: str
    title: str
    sha256: str
    label: Label
    pages: list[PageRecord] = field(default_factory=list)
    ingested_at: float = 0.0

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def readable_pages(self) -> int:
        return sum(1 for p in self.pages if p.extraction.is_real)

    @property
    def unreadable_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.extraction is Extraction.UNREADABLE]

    @property
    def simulated_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.extraction is Extraction.SIMULATED]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "sha256": self.sha256,
            "label": self.label.value,
            "pages": self.page_count,
            "readable_pages": self.readable_pages,
            "unreadable_pages": self.unreadable_pages,
            "simulated_pages": self.simulated_pages,
            "ingested_at": self.ingested_at,
            "extraction": {
                kind.value: sum(1 for p in self.pages if p.extraction is kind)
                for kind in Extraction
                if any(p.extraction is kind for p in self.pages)
            },
        }


@dataclass
class Hit:
    """One retrieved chunk with the arithmetic that put it there."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_similarity: float | None = None
    rerank_score: float | None = None

    @property
    def arms(self) -> list[str]:
        found = []
        if self.bm25_rank is not None:
            found.append("bm25")
        if self.dense_rank is not None:
            found.append("dense")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "page": self.chunk.page,
            "section": self.chunk.section,
            "label": self.chunk.label.value,
            "extraction": self.chunk.extraction.value,
            "simulated": self.chunk.extraction is Extraction.SIMULATED,
            "citation": self.chunk.citation,
            "score": round(self.score, 5),
            "arms": self.arms,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "dense_similarity": (
                round(self.dense_similarity, 4)
                if self.dense_similarity is not None
                else None
            ),
            "text": self.chunk.text,
        }


# A document id is used verbatim inside a citation, so it may not contain
# anything that would break parsing the citation back out again.
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_doc_id(stem: str) -> str:
    slug = _SLUG.sub("-", stem).strip("-._")
    return slug or "document"

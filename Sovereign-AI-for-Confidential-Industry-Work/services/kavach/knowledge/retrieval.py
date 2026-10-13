"""Hybrid retrieval (§07).

    query -> BM25 arm  ->\
                          Reciprocal Rank Fusion -> clearance filter -> top_k
    query -> dense arm ->/

Three things happen here that are worth reading the code for.

FUSION IS BY RANK, NOT BY SCORE. BM25 scores are unbounded and corpus-dependent;
cosine similarity is 0..1. Adding them with weights produces a number that looks
tuneable and is not: on a corpus like MRPL's, where a query shares a rare token
(`P-101A`) with exactly one document, the BM25 term is an order of magnitude
larger and the dense arm stops mattering at all. The router learned the same
lesson about raw seconds versus normalised terms (§04). RRF only ever compares
positions within one arm, so neither arm can swamp the other by having bigger
numbers.

CLEARANCE IS APPLIED AFTER FUSION AND IS COUNTED. A chunk the caller may not read
is dropped, and the count of what was dropped is returned. Silently returning
five hits instead of six teaches the user their query was weak; saying "2 chunks
withheld: role 'operator' is cleared to INTERNAL" teaches them the truth. The
policy engine decides, not this file.

COVERAGE TRAVELS WITH THE RESULT. If any document in the corpus has pages nothing
on this box could read, every search says so. An answer drawn from a corpus with
a hole in it must not be presentable as an answer drawn from a whole one, and the
only moment anyone will look is when the answer is in front of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..sovereignty import get_policy
from ..types import Label
from .index import HybridIndex, get_index
from .store import KnowledgeStore, get_knowledge_store
from .types import Chunk, Extraction, Hit


@dataclass
class Coverage:
    """What the corpus could not read, carried on every result."""

    documents: int = 0
    pages: int = 0
    unreadable_pages: int = 0
    simulated_pages: int = 0
    gaps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.unreadable_pages == 0 and self.simulated_pages == 0

    @property
    def warning(self) -> str | None:
        if self.unreadable_pages:
            names = ", ".join(
                f"{g['doc_id']} p.{','.join(str(p) for p in g['pages'])}"
                for g in self.gaps[:4]
            )
            more = "" if len(self.gaps) <= 4 else f" (+{len(self.gaps) - 4} more)"
            return (
                f"{self.unreadable_pages} of {self.pages} pages in this corpus could "
                f"not be read on this box and are NOT searchable: {names}{more}"
            )
        if self.simulated_pages:
            return (
                f"{self.simulated_pages} page(s) hold SIMULATED transcriptions and "
                "are excluded from retrieval"
            )
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "pages": self.pages,
            "unreadable_pages": self.unreadable_pages,
            "simulated_pages": self.simulated_pages,
            "complete": self.complete,
            "gaps": self.gaps,
            "warning": self.warning,
        }


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    coverage: Coverage
    withheld: int = 0
    withheld_reason: str = ""
    degraded: bool = False
    detail: str = ""
    arms: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def effective_label(self) -> Label:
        """A run inherits the strictest label of anything it has read (§02.5)."""
        if not self.hits:
            return Label.PUBLIC
        return Label.max(*[h.chunk.label for h in self.hits])

    @property
    def citations(self) -> list[str]:
        """Distinct citations, in rank order.

        Two chunks from the same page and section render the same citation - the
        page is the unit a reader can check, so that is correct, and listing it
        twice reads as two independent sources agreeing.
        """
        return list(dict.fromkeys(h.chunk.citation for h in self.hits))

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [h.as_dict() for h in self.hits],
            "count": len(self.hits),
            "citations": self.citations,
            "effective_label": self.effective_label.value,
            "withheld": self.withheld,
            "withheld_reason": self.withheld_reason,
            "degraded": self.degraded,
            "detail": self.detail,
            "arms": self.arms,
            "coverage": self.coverage.as_dict(),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class Retriever:
    def __init__(
        self,
        index: HybridIndex | None = None,
        store: KnowledgeStore | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._index = index
        self._store = store
        self._config = config

    @property
    def index(self) -> HybridIndex:
        if self._index is None:
            self._index = get_index()
        return self._index

    @property
    def store(self) -> KnowledgeStore:
        if self._store is None:
            self._store = get_knowledge_store()
        return self._store

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            from .config import load_knowledge_config

            self._config = load_knowledge_config()
        return self._config

    # ---------------------------------------------------------------- coverage

    def coverage(self) -> Coverage:
        stats = self.store.stats()
        return Coverage(
            documents=stats["documents"],
            pages=stats["pages"],
            unreadable_pages=stats["unreadable_pages"],
            simulated_pages=stats["simulated_pages"],
            gaps=stats["gaps"],
        )

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        role: str = "engineer",
        label_ceiling: Label | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        cfg = self.config.get("retrieval", {})
        top_k = int(top_k or cfg.get("top_k", 6))
        per_arm = int(cfg.get("candidates_per_arm", 25))
        rrf_k = int(cfg.get("rrf_k", 60))
        min_sim = float(cfg.get("min_dense_similarity", 0.15))

        state = self.index.state()
        query = (query or "").strip()

        if not query:
            return RetrievalResult(
                query=query,
                hits=[],
                coverage=self.coverage(),
                degraded=True,
                detail="empty query",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        lexical = self.index.bm25_ranking(query, per_arm) if state.lexical_ready else []
        dense = self.index.dense_ranking(query, per_arm) if state.dense_ready else []

        fused: dict[str, dict[str, Any]] = {}
        for rank, (chunk_id, _score) in enumerate(lexical, start=1):
            fused.setdefault(chunk_id, {"score": 0.0})
            fused[chunk_id]["score"] += 1.0 / (rrf_k + rank)
            fused[chunk_id]["bm25_rank"] = rank

        for rank, (chunk_id, similarity) in enumerate(dense, start=1):
            # A dense neighbour below the floor is noise; it keeps its position in
            # the dense arm's ranking but contributes nothing to the fused set, so
            # a query with no real match returns nothing rather than the
            # least-bad chunk in the corpus.
            if similarity < min_sim:
                continue
            fused.setdefault(chunk_id, {"score": 0.0})
            fused[chunk_id]["score"] += 1.0 / (rrf_k + rank)
            fused[chunk_id]["dense_rank"] = rank
            fused[chunk_id]["dense_similarity"] = similarity

        ordered = sorted(
            fused.items(),
            # Ties break toward the earlier chunk so the same query always
            # produces the same citations - a demo that reorders between runs
            # looks broken even when it is not.
            key=lambda kv: (-kv[1]["score"], kv[0]),
        )

        policy = get_policy()
        hits: list[Hit] = []
        withheld = 0
        withheld_reason = ""

        for chunk_id, terms in ordered:
            chunk = self.index.chunks.get(chunk_id) or self.store.chunk(chunk_id)
            if chunk is None:
                continue

            verdict = policy.role_may_read(role, chunk.label)
            over_ceiling = (
                label_ceiling is not None and chunk.label.rank > label_ceiling.rank
            )
            if verdict.denied or over_ceiling:
                withheld += 1
                withheld_reason = (
                    verdict.reason
                    if verdict.denied
                    else f"above the requested ceiling {label_ceiling.value}"
                )
                continue

            hits.append(
                Hit(
                    chunk=chunk,
                    score=terms["score"],
                    bm25_rank=terms.get("bm25_rank"),
                    dense_rank=terms.get("dense_rank"),
                    dense_similarity=terms.get("dense_similarity"),
                )
            )
            if len(hits) >= top_k:
                break

        arms = [a for a, on in (("bm25", state.lexical_ready), ("dense", state.dense_ready)) if on]

        return RetrievalResult(
            query=query,
            hits=hits,
            coverage=self.coverage(),
            withheld=withheld,
            withheld_reason=withheld_reason,
            degraded=state.degraded,
            detail=state.detail,
            arms=arms,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def context_block(hits: list[Hit], max_chunks: int = 8) -> str:
    """The retrieved text as it is handed to a model.

    Every passage is prefixed with the exact citation string the model is told to
    reproduce. The model is never asked to construct a citation from a filename
    and a page number - it is asked to copy one it can see, because a model that
    assembles citations will assemble plausible ones for text it did not read.
    """
    lines: list[str] = []
    for hit in hits[:max_chunks]:
        marker = (
            " (SIMULATED TRANSCRIPTION - NOT READ OFF THE PAGE)"
            if hit.chunk.extraction is Extraction.SIMULATED
            else ""
        )
        lines.append(f"{hit.chunk.citation}{marker}\n{hit.chunk.text}")
    return "\n\n---\n\n".join(lines)


def retrievable(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if c.extraction.is_real]


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reset_retriever() -> None:
    """Tests only."""
    global _retriever
    _retriever = None

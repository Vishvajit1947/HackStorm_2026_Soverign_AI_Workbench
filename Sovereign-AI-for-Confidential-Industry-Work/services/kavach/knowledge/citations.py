"""Mandatory citations (§07).

The requirement is "every knowledge answer carries citations". The requirement as
implemented is stricter, because the loose version is trivially satisfiable by a
model that has learned what a citation looks like:

    every citation in the answer must RESOLVE to a chunk that was actually in
    that answer's context.

Four verdicts, and the middle two are the interesting ones:

    RESOLVED         points at a chunk that was in the context. Checkable: the
                     reader can open that page and find that text.

    OUT_OF_CONTEXT   the document and page exist in the corpus, but were not
                     retrieved for this answer. The model is citing from its
                     weights, not from what it was given. The citation may even
                     be correct - and it is still not evidence, because nothing
                     in this system checked it.

    UNKNOWN_PAGE     the document exists; that page does not, or could not be
                     read. Frequently a model extrapolating "p.7" from a
                     four-page SOP.

    UNKNOWN_DOCUMENT no such document. Invented outright.

Only RESOLVED counts. An answer whose citations are all plausible and none
resolvable is the exact failure this module exists to make impossible to ship,
and it is the failure mode that survives every check that only looks for square
brackets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .store import KnowledgeStore, get_knowledge_store
from .types import Extraction, Hit

# [SOP_01 p.2]  ·  [SOP_01 p.2 s.4.1]  ·  [SIMULATED inspection_report_01 p.1]
CITATION = re.compile(
    r"\[\s*(?P<simulated>SIMULATED\s+)?"
    r"(?P<doc>[A-Za-z0-9][A-Za-z0-9._-]*)\s+"
    r"p\.\s*(?P<page>\d+)"
    r"(?:\s+s\.\s*(?P<section>[\d.]+))?\s*\]"
)

CITATION_INSTRUCTIONS = """\
Answer ONLY from the passages above. Every factual sentence must end with the
citation of the passage it came from, copied EXACTLY as it appears above - for
example [SOP_01 p.2]. Do not invent a citation, do not adjust a page number, and
do not cite a document that is not shown above. If the passages do not contain
the answer, say so plainly and cite nothing.\
"""


class Verdict(str, Enum):
    RESOLVED = "resolved"
    OUT_OF_CONTEXT = "out_of_context"
    UNKNOWN_PAGE = "unknown_page"
    UNKNOWN_DOCUMENT = "unknown_document"

    @property
    def ok(self) -> bool:
        return self is Verdict.RESOLVED


@dataclass
class CitationCheck:
    raw: str
    doc_id: str
    page: int
    section: str | None
    verdict: Verdict
    detail: str = ""
    simulated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "doc_id": self.doc_id,
            "page": self.page,
            "section": self.section,
            "verdict": self.verdict.value,
            "detail": self.detail,
            "simulated": self.simulated,
        }


@dataclass
class GroundingResult:
    ok: bool
    reason: str
    citations: list[CitationCheck] = field(default_factory=list)

    @property
    def resolved(self) -> list[CitationCheck]:
        return [c for c in self.citations if c.verdict.ok]

    @property
    def unresolved(self) -> list[CitationCheck]:
        return [c for c in self.citations if not c.verdict.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "cited": len(self.citations),
            "resolved": len(self.resolved),
            "unresolved": [c.as_dict() for c in self.unresolved],
            "citations": [c.as_dict() for c in self.citations],
        }


def parse_citations(text: str) -> list[tuple[str, str, int, str | None, bool]]:
    """(raw, doc_id, page, section, simulated) for each citation, in order."""
    found = []
    for match in CITATION.finditer(text or ""):
        found.append(
            (
                match.group(0),
                match.group("doc"),
                int(match.group("page")),
                match.group("section"),
                bool(match.group("simulated")),
            )
        )
    return found


def check_grounding(
    text: str,
    hits: list[Hit],
    *,
    store: KnowledgeStore | None = None,
    require_citation: bool = True,
) -> GroundingResult:
    """Resolve every citation in `text` against `hits` and the corpus."""
    store = store or get_knowledge_store()
    in_context = {(h.chunk.doc_id, h.chunk.page) for h in hits}
    parsed = parse_citations(text)

    if not parsed:
        if not require_citation:
            return GroundingResult(True, "no citation required for this answer")
        return GroundingResult(
            False,
            "the answer carries no citation, and an uncited claim about plant "
            "documentation is worse than no answer",
        )

    checks: list[CitationCheck] = []
    for raw, doc_id, page, section, simulated in parsed:
        document = store.document(doc_id)
        if document is None:
            checks.append(
                CitationCheck(
                    raw, doc_id, page, section, Verdict.UNKNOWN_DOCUMENT,
                    f"no document {doc_id!r} has been ingested", simulated,
                )
            )
            continue

        page_record = store.page(doc_id, page)
        if page_record is None:
            checks.append(
                CitationCheck(
                    raw, doc_id, page, section, Verdict.UNKNOWN_PAGE,
                    f"{doc_id} has {document.page_count} page(s); there is no p.{page}",
                    simulated,
                )
            )
            continue

        if page_record.extraction is Extraction.UNREADABLE:
            checks.append(
                CitationCheck(
                    raw, doc_id, page, section, Verdict.UNKNOWN_PAGE,
                    f"{doc_id} p.{page} could not be read on this box, so nothing "
                    "was indexed from it and nothing can be cited to it",
                    simulated,
                )
            )
            continue

        if (doc_id, page) not in in_context:
            checks.append(
                CitationCheck(
                    raw, doc_id, page, section, Verdict.OUT_OF_CONTEXT,
                    f"{doc_id} p.{page} exists but was not retrieved for this "
                    "answer - this claim came from the model, not the corpus",
                    simulated,
                )
            )
            continue

        checks.append(
            CitationCheck(
                raw, doc_id, page, section, Verdict.RESOLVED,
                "", simulated or page_record.extraction is Extraction.SIMULATED,
            )
        )

    bad = [c for c in checks if not c.verdict.ok]
    if bad:
        summary = "; ".join(f"{c.raw} {c.verdict.value}: {c.detail}" for c in bad[:3])
        more = "" if len(bad) <= 3 else f" (+{len(bad) - 3} more)"
        return GroundingResult(
            False,
            f"{len(bad)} of {len(checks)} citation(s) do not resolve - {summary}{more}",
            checks,
        )

    simulated_used = [c for c in checks if c.simulated]
    if simulated_used:
        return GroundingResult(
            False,
            f"{len(simulated_used)} citation(s) point at SIMULATED transcriptions. "
            "That text was not read off the page and may not be used as evidence.",
            checks,
        )

    return GroundingResult(
        True, f"all {len(checks)} citation(s) resolve to retrieved passages", checks
    )


def render_sources(hits: list[Hit]) -> list[dict[str, Any]]:
    """The source list a UI renders under an answer."""
    seen: dict[str, dict[str, Any]] = {}
    for hit in hits:
        key = hit.chunk.citation
        if key in seen:
            continue
        seen[key] = {
            "citation": key,
            "doc_id": hit.chunk.doc_id,
            "page": hit.chunk.page,
            "section": hit.chunk.section,
            "label": hit.chunk.label.value,
            "extraction": hit.chunk.extraction.value,
            "score": round(hit.score, 5),
            "excerpt": hit.chunk.text[:280],
        }
    return list(seen.values())

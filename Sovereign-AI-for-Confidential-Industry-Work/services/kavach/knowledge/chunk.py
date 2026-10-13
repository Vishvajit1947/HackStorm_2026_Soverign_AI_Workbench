"""Chunking (§07).

Two rules, and the second one is the one people get wrong.

  SIZE     ~1000 characters with 150 of overlap. Characters, not tokens: the
           tokeniser is a property of whichever model happens to be resident, and
           the index must not change shape when the router swaps qwen3-8b for the
           coder. An index whose chunk boundaries move when a model is evicted is
           an index you cannot reproduce.

  PAGES    a chunk never spans a page boundary. Not "rarely" - never.

The page rule costs a little retrieval quality at page joins and buys the only
thing that makes a citation checkable. A chunk covering the bottom of page 3 and
the top of page 4 can only be cited as "[doc p.3]" or "[doc p.4]", and either way
a reader who turns to that page to verify the claim finds half of the text there
and no way to tell which half. In a plant context, a citation you cannot follow
is worse than no citation, because it looks like diligence.

Section numbers are tracked as the text is walked, so a chunk in the middle of
"4.1 SUCTION CONDITIONS" cites `s.4.1` even though the heading is 600 characters
back and not in the chunk.
"""

from __future__ import annotations

import re
from typing import Any

from ..types import Label
from .extract import ExtractedPage, section_headings
from .types import Chunk, Extraction

# Prefer to break at a paragraph, then a sentence, then a line, then anywhere.
# Splitting mid-sentence is the difference between a quotable chunk and one that
# starts "...and shall be isolated before".
_BREAKS = (
    re.compile(r"\n\s*\n"),
    re.compile(r"(?<=[.!?])\s+(?=[A-Z(])"),
    re.compile(r"\n"),
)


def _best_break(text: str, lower: int, upper: int) -> int:
    """The latest acceptable break point in text[lower:upper], or upper."""
    window = text[lower:upper]
    for pattern in _BREAKS:
        positions = [m.end() for m in pattern.finditer(window)]
        if positions:
            return lower + positions[-1]
    return upper


def _word_start(text: str, offset: int) -> int:
    """The next offset at or after `offset` that is not inside a word.

    Forward, never back: moving backwards would re-emit text the previous chunk
    already ended with, which is what the overlap is for and does not need
    doing twice.
    """
    if offset <= 0 or offset >= len(text):
        return offset
    if text[offset - 1].isspace() or text[offset].isspace():
        return offset
    space = offset
    limit = min(len(text), offset + 40)      # a word, not a paragraph
    while space < limit and not text[space].isspace():
        space += 1
    return space + 1 if space < limit else offset


def _section_at(headings: list[tuple[int, str]], offset: int) -> str:
    """The most recent numbered heading at or before `offset`."""
    current = ""
    for start, number in headings:
        if start > offset:
            break
        current = number
    return current


def chunk_page(
    page: ExtractedPage,
    *,
    doc_id: str,
    label: Label,
    config: dict[str, Any],
    start_ordinal: int = 0,
) -> list[Chunk]:
    """Split one page's text. An UNREADABLE page yields nothing - there is no
    text, and emitting an empty chunk would put a hole in the index that looks
    exactly like a page about nothing."""
    if not page.extraction.is_indexed:
        return []

    text = page.text.strip()
    if not text:
        return []

    cfg = config.get("chunk", {})
    target = int(cfg.get("target_chars", 1000))
    overlap = int(cfg.get("overlap_chars", 150))
    minimum = int(cfg.get("min_chars", 80))

    headings = section_headings(text)
    chunks: list[Chunk] = []
    cursor = 0
    ordinal = start_ordinal

    while cursor < len(text):
        remaining = len(text) - cursor
        if remaining <= target:
            end = len(text)
        else:
            # Look for a break in the last quarter of the window, so a chunk is
            # never cut much shorter than the target just to land on a newline.
            end = _best_break(text, cursor + (target * 3) // 4, cursor + target)

        body = text[cursor:end].strip()
        # A trailing scrap shorter than min_chars is glued onto the previous
        # chunk rather than indexed alone: a 20-character chunk matches noise in
        # BM25 and its embedding is meaningless.
        if body and (len(body) >= minimum or not chunks):
            chunks.append(
                Chunk(
                    id=f"{doc_id}#p{page.page}c{ordinal}",
                    doc_id=doc_id,
                    page=page.page,
                    ordinal=ordinal,
                    text=body,
                    section=_section_at(headings, cursor),
                    label=label,
                    extraction=page.extraction,
                )
            )
            ordinal += 1
        elif body and chunks:
            chunks[-1].text = f"{chunks[-1].text}\n{body}"

        if end >= len(text):
            break
        # The overlap rewind lands on an arbitrary character offset, which is
        # very often the middle of a word: a chunk then begins "ude Distillation
        # Unit (CDU-II)" and every quotation of it - in a search result, in a
        # briefing deck, in an approval note - starts two letters into a word.
        # Nudging forward to the next space costs a few characters of overlap
        # and is the difference between a citation a reader trusts and one they
        # have to go and check.
        cursor = _word_start(text, max(end - overlap, cursor + 1))

    return chunks


def chunk_document(
    pages: list[ExtractedPage],
    *,
    doc_id: str,
    label: Label,
    config: dict[str, Any],
) -> list[Chunk]:
    """Chunk every readable page. Ordinals are continuous across the document so
    a chunk id is stable and sortable, but no chunk crosses a page."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(
            chunk_page(
                page,
                doc_id=doc_id,
                label=label,
                config=config,
                start_ordinal=len(chunks),
            )
        )
    return chunks


def coverage(chunks: list[Chunk]) -> dict[str, int]:
    """How many chunks came from each extraction path. Rendered on `kb stats`."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.extraction.value] = counts.get(chunk.extraction.value, 0) + 1
    return counts


def simulated_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if c.extraction is Extraction.SIMULATED]

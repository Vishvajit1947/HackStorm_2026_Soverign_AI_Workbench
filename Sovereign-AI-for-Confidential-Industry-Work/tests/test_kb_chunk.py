"""Chunking and citation rendering (§07).

The invariant with teeth is the page one: every chunk belongs to exactly one page,
so the citation it carries is a claim a reader can check by turning to that page.
Everything else here is quality; that one is correctness.
"""

from __future__ import annotations

import pytest

from kavach.knowledge import Extraction, chunk_document, chunk_page
from kavach.knowledge.chunk import coverage
from kavach.knowledge.config import load_knowledge_config
from kavach.knowledge.extract import ExtractedPage, section_headings
from kavach.types import Label
from kbfixtures import SOP_TEXT


@pytest.fixture()
def config():
    return load_knowledge_config()


def page(number: int, text: str, extraction: Extraction = Extraction.NATIVE):
    return ExtractedPage(number, text, extraction, "test")


LONG = " ".join(f"Clause {i} states an operating requirement for the unit." for i in range(120))


# ---------------------------------------------------------------- the page rule

def test_no_chunk_ever_spans_two_pages(config):
    """A chunk covering p.3 and p.4 can only be cited as one of them, and a
    reader who turns to that page finds half the text there."""
    pages = [page(1, LONG), page(2, LONG), page(3, LONG)]
    chunks = chunk_document(pages, doc_id="DOC", label=Label.INTERNAL, config=config)

    assert len(chunks) > 3
    for chunk in chunks:
        source = pages[chunk.page - 1]
        assert chunk.text.replace("\n", " ") in " ".join(source.text.split()) or \
            chunk.text in source.text


def test_every_page_that_has_text_produces_at_least_one_chunk(config):
    pages = [page(1, "Short but real content about the pump."), page(2, LONG)]
    chunks = chunk_document(pages, doc_id="DOC", label=Label.INTERNAL, config=config)
    assert {c.page for c in chunks} == {1, 2}


def test_an_unreadable_page_produces_no_chunk_rather_than_an_empty_one(config):
    """An empty chunk in the index is a page about nothing, which is exactly what
    a page nobody could read is not."""
    chunks = chunk_page(
        page(1, "", Extraction.UNREADABLE),
        doc_id="DOC", label=Label.INTERNAL, config=config,
    )
    assert chunks == []


def test_a_page_of_whitespace_produces_no_chunk(config):
    assert chunk_page(
        page(1, "   \n\n  "), doc_id="DOC", label=Label.INTERNAL, config=config
    ) == []


# ---------------------------------------------------------------- shape

def test_chunks_overlap_so_a_clause_split_at_a_boundary_is_still_findable(config):
    chunks = chunk_page(page(1, LONG), doc_id="DOC", label=Label.INTERNAL, config=config)
    assert len(chunks) >= 3
    tail = chunks[0].text[-60:]
    assert any(word in chunks[1].text for word in tail.split()[:4])


def test_a_trailing_scrap_is_glued_on_rather_than_indexed_alone(config):
    """A 20-character chunk matches noise in BM25 and embeds to nothing."""
    minimum = config["chunk"]["min_chars"]
    chunks = chunk_page(
        page(1, LONG + "\n\nEND."), doc_id="DOC", label=Label.INTERNAL, config=config
    )
    assert all(len(c.text) >= minimum for c in chunks)
    assert "END." in chunks[-1].text


def test_ordinals_are_continuous_across_a_document(config):
    pages = [page(1, LONG), page(2, LONG)]
    chunks = chunk_document(pages, doc_id="DOC", label=Label.INTERNAL, config=config)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique_and_carry_the_page(config):
    chunks = chunk_document(
        [page(1, LONG), page(2, LONG)], doc_id="DOC", label=Label.INTERNAL, config=config
    )
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(f"#p{c.page}" in c.id for c in chunks)


# ---------------------------------------------------------------- sections

def test_headings_are_found_in_document_order():
    numbers = [n for _, n in section_headings(SOP_TEXT)]
    assert numbers == ["1", "2", "3"]


def test_a_chunk_inherits_the_heading_above_it_even_when_it_is_far_back(config):
    body = "1. PURPOSE\n" + ("Filler sentence about the unit. " * 60) + \
           "\n2. INTERLOCKS\n" + ("Interlock detail sentence. " * 60)
    chunks = chunk_page(page(4, body), doc_id="DOC", label=Label.INTERNAL, config=config)
    sections = [c.section for c in chunks]
    assert "1" in sections and "2" in sections


def test_text_before_the_first_heading_has_no_section(config):
    chunks = chunk_page(
        page(1, "Front matter with no heading at all, but enough characters to index."),
        doc_id="DOC", label=Label.INTERNAL, config=config,
    )
    assert chunks[0].section == ""


# ---------------------------------------------------------------- citations

def test_the_rendered_citation_is_ascii_and_names_doc_and_page(config):
    chunk = chunk_page(
        page(2, LONG), doc_id="SOP_01", label=Label.INTERNAL, config=config
    )[0]
    assert chunk.citation.startswith("[SOP_01 p.2")
    chunk.citation.encode("ascii")          # a section sign mojibakes on cp1252


def test_a_section_appears_in_the_citation(config):
    chunk = chunk_page(
        page(1, "2.1 SCOPE\n" + LONG), doc_id="SOP_01", label=Label.INTERNAL, config=config
    )[0]
    assert chunk.citation == "[SOP_01 p.1 s.2.1]"


def test_a_simulated_chunk_can_never_render_as_an_ordinary_citation(config):
    chunk = chunk_page(
        page(1, LONG, Extraction.SIMULATED),
        doc_id="SCAN", label=Label.INTERNAL, config=config,
    )[0]
    assert chunk.citation.startswith("[SIMULATED ")


def test_chunks_inherit_the_documents_label(config):
    chunks = chunk_page(
        page(1, LONG), doc_id="DOC", label=Label.RESTRICTED, config=config
    )
    assert all(c.label is Label.RESTRICTED for c in chunks)


def test_coverage_counts_each_extraction_path(config):
    chunks = chunk_document(
        [page(1, LONG), page(2, LONG, Extraction.OCR)],
        doc_id="DOC", label=Label.INTERNAL, config=config,
    )
    counts = coverage(chunks)
    assert counts["NATIVE"] > 0 and counts["OCR"] > 0

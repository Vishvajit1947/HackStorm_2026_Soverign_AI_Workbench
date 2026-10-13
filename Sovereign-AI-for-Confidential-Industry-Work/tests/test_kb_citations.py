"""Mandatory citations (§07).

"The answer has citations" is trivially satisfiable by a model that has learned
what a citation looks like. These tests pin the stronger property: every citation
must resolve to a passage that was actually in this answer's context.

The four verdicts exist because they call for four different reactions, and
collapsing them into "bad citation" throws away the diagnosis.
"""

from __future__ import annotations

import pytest

from kavach.knowledge import Extraction, Verdict, check_grounding, parse_citations
from kavach.knowledge.store import KnowledgeStore
from kavach.knowledge.types import Chunk, DocumentRecord, Hit, PageRecord
from kavach.router.verifier import check_has_citation
from kavach.types import Label


@pytest.fixture()
def store():
    s = KnowledgeStore()
    s.clear()
    record = DocumentRecord(
        id="SOP_01", path="/corpus/SOP_01.pdf", title="SOP", sha256="a" * 64,
        label=Label.CONFIDENTIAL,
        pages=[
            PageRecord(1, Extraction.NATIVE, 900, "PDF text layer"),
            PageRecord(2, Extraction.NATIVE, 900, "PDF text layer"),
            PageRecord(3, Extraction.UNREADABLE, 0, "no OCR engine on this box"),
        ],
    )
    chunks = [
        Chunk(id="SOP_01#p1c0", doc_id="SOP_01", page=1, ordinal=0,
              text="PSLL-101 trip setpoint is 1.80 bar gauge.", section="2",
              label=Label.CONFIDENTIAL),
        Chunk(id="SOP_01#p2c1", doc_id="SOP_01", page=2, ordinal=1,
              text="The fire watch remains on station for thirty minutes.",
              label=Label.CONFIDENTIAL),
    ]
    s.put_document(record, chunks)
    yield s
    s.clear()


def hits(store, *chunk_ids):
    return [Hit(chunk=store.chunk(cid), score=0.5) for cid in chunk_ids]


# ---------------------------------------------------------------- parsing

def test_a_page_citation_parses():
    assert parse_citations("The setpoint is 1.8 bar [SOP_01 p.2].") == [
        ("[SOP_01 p.2]", "SOP_01", 2, None, False)
    ]


def test_a_section_citation_parses():
    _raw, doc, page, section, simulated = parse_citations("x [SOP_01 p.2 s.4.1]")[0]
    assert (doc, page, section, simulated) == ("SOP_01", 2, "4.1", False)


def test_a_simulated_citation_is_recognisable_as_one():
    _, _, _, _, simulated = parse_citations("x [SIMULATED SCAN p.1]")[0]
    assert simulated is True


def test_prose_in_brackets_is_not_a_citation():
    assert parse_citations("The valve [see the drawing] is open") == []
    assert parse_citations("[TODO]") == []


def test_the_routers_verifier_accepts_what_the_knowledge_plane_emits():
    """A cross-module contract: §04's citation check and §07's citation format
    are written in different files and must not drift apart."""
    ok, _ = check_has_citation("The setpoint is 1.8 bar [SOP_01 p.2].")
    assert ok
    ok, _ = check_has_citation("The setpoint is 1.8 bar [SOP_01 p.2 s.4.1].")
    assert ok


# ---------------------------------------------------------------- verdicts

def test_a_citation_to_a_retrieved_passage_resolves(store):
    result = check_grounding(
        "The trip setpoint is 1.80 bar gauge [SOP_01 p.1].",
        hits(store, "SOP_01#p1c0"),
        store=store,
    )
    assert result.ok
    assert result.citations[0].verdict is Verdict.RESOLVED


def test_a_citation_to_a_real_page_that_was_not_retrieved_is_not_evidence(store):
    """The claim may even be true. Nothing in this system checked it."""
    result = check_grounding(
        "The fire watch stays thirty minutes [SOP_01 p.2].",
        hits(store, "SOP_01#p1c0"),
        store=store,
    )
    assert not result.ok
    assert result.citations[0].verdict is Verdict.OUT_OF_CONTEXT
    assert "came from the model" in result.citations[0].detail


def test_a_page_beyond_the_end_of_the_document_is_caught(store):
    result = check_grounding("Per the annex [SOP_01 p.9].", hits(store, "SOP_01#p1c0"),
                             store=store)
    assert result.citations[0].verdict is Verdict.UNKNOWN_PAGE
    assert "there is no p.9" in result.citations[0].detail


def test_a_citation_to_an_unreadable_page_is_refused(store):
    """Nothing was indexed from that page, so nothing can be cited to it - and
    this is exactly where a model will point when it invents a source."""
    result = check_grounding("Per the scan [SOP_01 p.3].", hits(store, "SOP_01#p1c0"),
                             store=store)
    assert result.citations[0].verdict is Verdict.UNKNOWN_PAGE
    assert "could not be read" in result.citations[0].detail


def test_an_invented_document_is_caught(store):
    result = check_grounding("Per the standard [API-610 p.4].", hits(store, "SOP_01#p1c0"),
                             store=store)
    assert result.citations[0].verdict is Verdict.UNKNOWN_DOCUMENT
    assert "has been ingested" in result.citations[0].detail


def test_an_answer_with_no_citation_at_all_fails(store):
    result = check_grounding(
        "The trip setpoint is 1.80 bar gauge.", hits(store, "SOP_01#p1c0"), store=store
    )
    assert not result.ok
    assert "no citation" in result.reason


def test_a_general_answer_may_be_exempted_explicitly(store):
    result = check_grounding("I do not know.", [], store=store, require_citation=False)
    assert result.ok


def test_one_bad_citation_among_good_ones_still_fails(store):
    """Partial credit on provenance is not a thing."""
    result = check_grounding(
        "First [SOP_01 p.1]. Second [API-610 p.4].",
        hits(store, "SOP_01#p1c0"),
        store=store,
    )
    assert not result.ok
    assert len(result.resolved) == 1
    assert len(result.unresolved) == 1


def test_the_reason_names_the_offending_citations(store):
    result = check_grounding("x [API-610 p.4] y [ISO-13709 p.2]",
                             hits(store, "SOP_01#p1c0"), store=store)
    assert "API-610" in result.reason
    assert "2 of 2" in result.reason


def test_a_citation_pointing_at_a_simulated_transcription_is_refused(store):
    record = DocumentRecord(
        id="SCAN", path="/corpus/SCAN.pdf", title="scan", sha256="b" * 64,
        label=Label.CONFIDENTIAL,
        pages=[PageRecord(1, Extraction.SIMULATED, 40, "mock VLM")],
    )
    chunk = Chunk(id="SCAN#p1c0", doc_id="SCAN", page=1, ordinal=0,
                  text="invented", label=Label.CONFIDENTIAL,
                  extraction=Extraction.SIMULATED)
    store.put_document(record, [chunk])

    result = check_grounding("Per the report [SIMULATED SCAN p.1].",
                             [Hit(chunk=chunk, score=1.0)], store=store)
    assert not result.ok
    assert "may not be used as evidence" in result.reason


def test_the_summary_counts_resolved_and_unresolved(store):
    result = check_grounding("a [SOP_01 p.1] b [SOP_01 p.1]",
                             hits(store, "SOP_01#p1c0"), store=store)
    payload = result.as_dict()
    assert payload["cited"] == 2
    assert payload["resolved"] == 2
    assert payload["unresolved"] == []

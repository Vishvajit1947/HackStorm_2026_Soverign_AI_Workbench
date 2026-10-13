"""Hybrid retrieval, clearance and coverage (§07).

Three claims:

    fusion cannot be won by one arm having bigger numbers,
    a passage the caller may not read is withheld AND counted, and
    a corpus with a hole in it says so on every single search.
"""

from __future__ import annotations

import pytest

from kavach.knowledge import Extraction, HybridIndex, Retriever, context_block
from kavach.knowledge.config import load_knowledge_config
from kavach.knowledge.store import KnowledgeStore
from kavach.knowledge.types import Chunk, DocumentRecord, PageRecord
from kavach.types import Label
from kbfixtures import HSE_TEXT, SOP_TEXT


@pytest.fixture()
def config():
    return load_knowledge_config()


@pytest.fixture()
def kb(tmp_path, config):
    """A knowledge plane on its own index directory with an empty corpus."""
    store = KnowledgeStore()
    store.clear()
    index = HybridIndex(tmp_path, store=store)
    yield store, index, Retriever(index=index, store=store, config=config)
    store.clear()


def add(store, doc_id, texts, *, label=Label.INTERNAL, pages=None,
        extraction=Extraction.NATIVE):
    record = DocumentRecord(
        id=doc_id, path=f"/corpus/{doc_id}.pdf", title=doc_id,
        sha256=doc_id.encode().hex().ljust(64, "0")[:64], label=label,
        pages=pages or [PageRecord(1, extraction, 100, "test")],
    )
    chunks = [
        Chunk(id=f"{doc_id}#p1c{i}", doc_id=doc_id, page=1, ordinal=i, text=text,
              label=label, extraction=extraction)
        for i, text in enumerate(texts)
    ]
    store.put_document(record, chunks)


# ---------------------------------------------------------------- fusion

def test_a_passage_both_arms_found_outranks_one_only_a_single_arm_found(kb):
    store, index, retriever = kb
    add(store, "SOP", [
        "PSLL-101 suction pressure trip setpoint is 1.80 bar gauge to prevent dry-run.",
        "The motor is de-energised when suction falls, protecting the seal faces.",
        "Unrelated content about painting specifications for external pipework.",
    ])
    index.build()

    result = retriever.search("PSLL-101 suction pressure trip")
    assert result.hits
    top = result.hits[0]
    assert top.arms == ["bm25", "dense"]
    assert top.chunk.id == "SOP#p1c0"


def test_both_arms_are_reported_on_every_result(kb):
    store, index, retriever = kb
    add(store, "SOP", ["Content about the crude booster pump and its interlocks."])
    index.build()
    assert retriever.search("pump").arms == ["bm25", "dense"]


def test_a_query_matching_nothing_returns_nothing_rather_than_the_least_bad_chunk(kb):
    store, index, retriever = kb
    add(store, "SOP", ["Hydrocarbon feed pump startup and isolation for CDU-II."])
    index.build()

    result = retriever.search("zzzqqq unrelated xyzzy plugh")
    assert result.hits == []


def test_an_empty_query_is_degraded_not_a_crash(kb):
    _, _, retriever = kb
    result = retriever.search("   ")
    assert result.hits == []
    assert result.degraded
    assert result.detail == "empty query"


def test_the_same_query_returns_the_same_citations_twice(kb):
    """A demo that reorders between runs looks broken even when it is not."""
    store, index, retriever = kb
    add(store, "SOP", [SOP_TEXT, HSE_TEXT])
    index.build()

    first = retriever.search("permit to work and gas testing")
    second = retriever.search("permit to work and gas testing")
    assert first.citations == second.citations


def test_top_k_is_honoured(kb):
    store, index, retriever = kb
    add(store, "SOP", [f"Clause {i} about the pump and its operating envelope." for i in range(10)])
    index.build()
    assert len(retriever.search("pump", top_k=3).hits) == 3


# ---------------------------------------------------------------- clearance

def test_an_operator_does_not_receive_confidential_passages(kb):
    store, index, retriever = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."], label=Label.CONFIDENTIAL)
    add(store, "HSE", ["PSLL-101 appears in the hot work permit checklist."], label=Label.INTERNAL)
    index.build()

    result = retriever.search("PSLL-101", role="operator")

    assert {h.chunk.doc_id for h in result.hits} == {"HSE"}
    # Withheld AND counted: returning one hit instead of two teaches the user
    # their query was weak, which is not what happened.
    assert result.withheld == 1
    assert "cleared to INTERNAL" in result.withheld_reason


def test_an_engineer_receives_what_an_operator_may_not(kb):
    store, index, retriever = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."], label=Label.CONFIDENTIAL)
    index.build()

    assert retriever.search("PSLL-101", role="engineer").hits
    assert retriever.search("PSLL-101", role="operator").hits == []


def test_an_explicit_ceiling_withholds_below_the_roles_clearance(kb):
    store, index, retriever = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."], label=Label.CONFIDENTIAL)
    index.build()

    result = retriever.search("PSLL-101", role="engineer", label_ceiling=Label.INTERNAL)
    assert result.hits == []
    assert result.withheld == 1
    assert "INTERNAL" in result.withheld_reason


def test_an_unknown_role_is_refused_everything(kb):
    """The policy engine decides; retrieval does not invent a default."""
    store, index, retriever = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."])
    index.build()

    result = retriever.search("PSLL-101", role="intern")
    assert result.hits == []
    assert result.withheld == 1


def test_the_run_inherits_the_strictest_label_it_read(kb):
    store, index, retriever = kb
    add(store, "PUB", ["Pump maintenance overview for the training handbook."],
        label=Label.INTERNAL)
    add(store, "SOP", ["Pump interlock detail with the trip setpoint."],
        label=Label.CONFIDENTIAL)
    index.build()

    assert retriever.search("pump").effective_label is Label.CONFIDENTIAL


def test_a_search_that_returns_nothing_does_not_claim_a_label(kb):
    _, index, retriever = kb
    index.build()
    assert retriever.search("anything").effective_label is Label.PUBLIC


# ---------------------------------------------------------------- coverage

def test_every_search_reports_pages_that_could_not_be_read(kb):
    store, index, retriever = kb
    add(store, "SOP", ["Hydrocarbon feed pump startup for CDU-II."])
    add(store, "SCAN", [], pages=[PageRecord(1, Extraction.UNREADABLE, 0, "no OCR")],
        label=Label.CONFIDENTIAL)
    index.build()

    result = retriever.search("pump")

    assert not result.coverage.complete
    assert result.coverage.unreadable_pages == 1
    assert "SCAN p.1" in result.coverage.warning


def test_a_whole_corpus_reports_no_warning(kb):
    store, index, retriever = kb
    add(store, "SOP", ["Hydrocarbon feed pump startup for CDU-II."])
    index.build()

    result = retriever.search("pump")
    assert result.coverage.complete
    assert result.coverage.warning is None


def test_a_stale_index_makes_the_search_report_degraded(kb):
    store, index, retriever = kb
    add(store, "SOP", ["Hydrocarbon feed pump startup for CDU-II."])
    index.build()
    add(store, "HSE", ["Hot work permits expire at the end of the issuing shift."])

    result = retriever.search("pump")
    assert result.degraded
    assert "STALE" in result.detail


# ---------------------------------------------------------------- context

def test_every_passage_in_the_context_carries_its_citation(kb):
    """The model copies a citation it can see; it never assembles one."""
    store, index, retriever = kb
    add(store, "SOP", ["PSLL-101 suction pressure trip setpoint is 1.80 bar gauge."])
    index.build()

    block = context_block(retriever.search("PSLL-101").hits)
    assert "[SOP p.1]" in block
    assert "PSLL-101" in block


def test_the_context_is_capped(kb):
    store, index, retriever = kb
    add(store, "SOP", [f"Clause {i} about the pump operating envelope." for i in range(10)])
    index.build()

    hits = retriever.search("pump", top_k=10).hits
    assert context_block(hits, max_chunks=2).count("[SOP p.1") == 2


def test_a_simulated_passage_is_marked_inside_the_model_context(kb):
    from kavach.knowledge.types import Chunk as C

    hits = context_block([
        type("H", (), {"chunk": C(
            id="x", doc_id="SCAN", page=1, ordinal=0,
            text="invented", extraction=Extraction.SIMULATED,
        )})()
    ])
    assert "SIMULATED" in hits

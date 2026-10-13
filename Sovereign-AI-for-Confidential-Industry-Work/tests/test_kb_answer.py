"""Grounded answering, ingest orchestration and the kb.search tool (§07).

The answerer has three refusals, and each one guards a moment where the system
could have produced something that looked like an answer. They are the tests that
matter here; the happy path is the easy part.
"""

from __future__ import annotations

import pytest

from kavach.knowledge import (
    Answerer,
    Extraction,
    HybridIndex,
    Ingestor,
    KnowledgeStore,
    Retriever,
)
from kavach.knowledge.config import load_knowledge_config
from kavach.knowledge.types import Chunk, DocumentRecord, PageRecord
from kavach.tools import Workspace, get_executor
from kavach.tools.impl.kb import kb_search
from kavach.types import Label
from kbfixtures import HSE_TEXT, SOP_TEXT, FakeReader, native_pdf, scanned_pdf


@pytest.fixture()
def config():
    return load_knowledge_config()


@pytest.fixture()
def kb(tmp_path, config):
    store = KnowledgeStore()
    store.clear()
    index = HybridIndex(tmp_path / "idx", store=store)
    retriever = Retriever(index=index, store=store, config=config)
    ingestor = Ingestor(store=store, index=index, reader=FakeReader(), config=config)
    yield store, index, retriever, ingestor
    store.clear()


def add(store, doc_id, texts, *, label=Label.INTERNAL, extraction=Extraction.NATIVE):
    record = DocumentRecord(
        id=doc_id, path=f"/corpus/{doc_id}.pdf", title=doc_id,
        sha256=doc_id.encode().hex().ljust(64, "0")[:64], label=label,
        pages=[PageRecord(1, extraction, 100, "test")],
    )
    store.put_document(record, [
        Chunk(id=f"{doc_id}#p1c{i}", doc_id=doc_id, page=1, ordinal=i, text=t,
              label=label, extraction=extraction)
        for i, t in enumerate(texts)
    ])


# ---------------------------------------------------------------- ingest

def test_ingesting_a_folder_reports_every_file_including_the_unreadable_ones(
    tmp_path, kb
):
    _store, _, _, ingestor = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    native_pdf(corpus / "SOP_01.pdf", SOP_TEXT)
    native_pdf(corpus / "SOP_02.pdf", HSE_TEXT)
    scanned_pdf(corpus / "scan.pdf")

    report = ingestor.ingest(corpus)

    assert report.ingested == 3
    assert not report.failed
    assert report.unreadable_pages == 1
    assert "NOT in the index" in report.warning
    assert {d.doc_id for d in report.documents} == {"SOP_01", "SOP_02", "scan"}


def test_re_ingesting_identical_bytes_does_not_duplicate_the_document(tmp_path, kb):
    """Two copies of one SOP reads to a user as corroboration from two sources."""
    store, _, _, ingestor = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    native_pdf(corpus / "SOP_01.pdf", SOP_TEXT)

    first = ingestor.ingest(corpus)
    second = ingestor.ingest(corpus)

    assert first.ingested == 1
    assert second.skipped == 1 and second.ingested == 0
    assert len(store.documents()) == 1


def test_changed_bytes_replace_the_document_rather_than_adding_to_it(tmp_path, kb):
    store, _, _, ingestor = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    native_pdf(corpus / "SOP_01.pdf", SOP_TEXT)
    ingestor.ingest(corpus)
    assert any("PSLL-101" in c.text for c in store.chunks())

    native_pdf(corpus / "SOP_01.pdf", HSE_TEXT)
    ingestor.ingest(corpus)

    assert len(store.documents()) == 1
    # The old chunks are gone, not sitting alongside the new ones answering
    # questions from a revision that was superseded.
    assert not any("PSLL-101" in c.text for c in store.chunks())
    assert any("Hydrogen sulfide" in c.text for c in store.chunks())


def test_a_file_that_cannot_be_opened_is_reported_not_swallowed(tmp_path, kb):
    _store, _, _, ingestor = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    native_pdf(corpus / "SOP_01.pdf", SOP_TEXT)
    (corpus / "broken.pdf").write_bytes(b"this is not a PDF")

    report = ingestor.ingest(corpus)

    assert report.ingested == 1
    assert len(report.failed) == 1
    assert report.failed[0].path.endswith("broken.pdf")


def test_forgetting_a_document_removes_it_from_the_index(tmp_path, kb):
    _store, index, retriever, ingestor = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    native_pdf(corpus / "SOP_01.pdf", SOP_TEXT)
    ingestor.ingest(corpus)
    assert retriever.search("PSLL-101").hits

    assert ingestor.forget("SOP_01")
    assert retriever.search("PSLL-101").hits == []
    assert not index.is_stale()


def test_the_corpus_stats_lead_with_what_is_missing(tmp_path, kb):
    store, _, _, ingestor = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    native_pdf(corpus / "SOP_01.pdf", SOP_TEXT)
    scanned_pdf(corpus / "scan.pdf")
    ingestor.ingest(corpus)

    stats = store.stats()
    assert stats["unreadable_pages"] == 1
    assert stats["gaps"] == [{"doc_id": "scan", "pages": [1]}]
    assert "could not be read" in stats["warning"]


# ---------------------------------------------------------------- refusals

def test_nothing_retrieved_means_no_answer_not_the_models_recollection(kb):
    """A model asked a plant question with no context still produces fluent,
    specific, wrong text about interlock setpoints."""
    store, index, retriever, _ = kb
    add(store, "SOP", ["Hydrocarbon feed pump startup and isolation for CDU-II."])
    index.build()

    class NoRouter:
        def route(self, *a, **k):
            raise AssertionError("the model must not be called with no context")

    result = Answerer(retriever=retriever, router=NoRouter()).answer("xyzzy plugh")

    assert not result.ok
    assert result.answer is None
    assert "no passage in the corpus matched" in result.headline


def test_a_refusal_names_the_passages_that_were_withheld(kb):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."],
        label=Label.CONFIDENTIAL)
    index.build()

    class NoRouter:
        def route(self, *a, **k):
            raise AssertionError("must not be called")

    result = Answerer(retriever=retriever, router=NoRouter()).answer(
        "PSLL-101", role="operator"
    )
    assert not result.ok
    assert "withheld" in result.headline


def test_a_simulated_context_is_refused_before_a_model_is_called(kb):
    """Same refusal the eval harness makes about quality numbers (§04)."""
    store, index, _retriever, _ = kb
    add(store, "SCAN", ["Invented transcription of the inspection report."],
        extraction=Extraction.SIMULATED)
    index.build()
    # The index skips SIMULATED chunks, so put one in front of the answerer
    # directly - this is the path that exists if anything ever indexes them.
    chunk = store.chunks()[0]

    class Fake:
        def search(self, *a, **k):
            from kavach.knowledge.retrieval import Coverage, RetrievalResult
            from kavach.knowledge.types import Hit

            return RetrievalResult(
                query="q", hits=[Hit(chunk=chunk, score=1.0)], coverage=Coverage()
            )

    class NoRouter:
        def route(self, *a, **k):
            raise AssertionError("must not be called on simulated context")

    result = Answerer(retriever=Fake(), router=NoRouter()).answer("anything")
    assert not result.ok
    assert "SIMULATED" in result.headline


def test_an_ungrounded_answer_is_never_returned_as_the_answer(kb):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."])
    index.build()

    class Router:
        def route(self, req, **k):
            return _outcome("The setpoint is 3 bar [API-610 p.7].")

    result = Answerer(retriever=retriever, router=Router()).answer("PSLL-101")

    assert not result.ok
    assert result.answer is None
    assert result.rejected_answer == "The setpoint is 3 bar [API-610 p.7]."
    assert "unknown_document" in result.grounding.as_dict()["unresolved"][0]["verdict"]


def test_an_answer_with_no_citation_is_rejected(kb):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."])
    index.build()

    class Router:
        def route(self, req, **k):
            return _outcome("The setpoint is 1.8 bar.")

    result = Answerer(retriever=retriever, router=Router()).answer("PSLL-101")
    assert not result.ok
    assert result.answer is None


# ---------------------------------------------------------------- the good path

def test_a_grounded_answer_keeps_its_sources_and_inherits_the_label(kb):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."],
        label=Label.CONFIDENTIAL)
    index.build()

    seen = {}

    class Router:
        def route(self, req, **k):
            seen["label"] = req.label
            seen["kb_grounded"] = k.get("kb_grounded")
            return _outcome("The setpoint is 1.80 bar gauge [SOP p.1].")

    result = Answerer(retriever=retriever, router=Router()).answer("PSLL-101")

    assert result.ok
    assert result.answer == "The setpoint is 1.80 bar gauge [SOP p.1]."
    assert result.label is Label.CONFIDENTIAL
    # The run inherits the label, and the verifier is told to demand a citation.
    assert seen["label"] is Label.CONFIDENTIAL
    assert seen["kb_grounded"] is True
    assert result.sources[0]["citation"] == "[SOP p.1]"


def test_the_context_handed_to_the_model_carries_the_citations_to_copy(kb):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."])
    index.build()

    seen = {}

    class Router:
        def route(self, req, **k):
            seen["prompt"] = req.prompt
            return _outcome("x [SOP p.1]")

    Answerer(retriever=retriever, router=Router()).answer("PSLL-101")

    assert "[SOP p.1]" in seen["prompt"]
    assert "copied EXACTLY" in seen["prompt"]


# ---------------------------------------------------------------- the tool

def test_kb_search_returns_citations_and_the_label_to_inherit(kb, monkeypatch):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."],
        label=Label.CONFIDENTIAL)
    index.build()
    monkeypatch.setattr("kavach.tools.impl.kb.get_retriever", lambda: retriever)

    data = kb_search("PSLL-101", role="engineer")

    assert data["count"] == 1
    assert data["citations"] == ["[SOP p.1]"]
    assert data["effective_label"] == "CONFIDENTIAL"


def test_kb_search_applies_the_callers_clearance(kb, monkeypatch):
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."],
        label=Label.CONFIDENTIAL)
    index.build()
    monkeypatch.setattr("kavach.tools.impl.kb.get_retriever", lambda: retriever)

    assert kb_search("PSLL-101", role="operator")["count"] == 0
    assert kb_search("PSLL-101", role="operator")["withheld"] == 1


def test_kb_search_top_k_is_capped(kb, monkeypatch):
    store, index, retriever, _ = kb
    add(store, "SOP", [f"Clause {i} about the pump envelope." for i in range(40)])
    index.build()
    monkeypatch.setattr("kavach.tools.impl.kb.get_retriever", lambda: retriever)

    assert kb_search("pump", top_k=999)["count"] <= 20


def test_the_executor_hands_the_tool_the_callers_role(tmp_path, kb, monkeypatch):
    """The clearance filter must be fed by the invocation, never by a global."""
    store, index, retriever, _ = kb
    add(store, "SOP", ["PSLL-101 trip setpoint is 1.80 bar gauge."],
        label=Label.CONFIDENTIAL)
    index.build()
    monkeypatch.setattr("kavach.tools.impl.kb.get_retriever", lambda: retriever)

    workspace = Workspace(tmp_path / "run_kb")
    result = get_executor().invoke(
        "kb.search", {"query": "PSLL-101"}, workspace, role="operator"
    )
    assert result.ok
    assert result.data["count"] == 0
    assert result.data["withheld"] == 1


def _outcome(text: str):
    """A minimal stand-in for RouteOutcome - only what the answerer reads."""
    from kavach.types import GenResult, RouteDecision, TaskClass

    class Outcome:
        def __init__(self):
            self.result = GenResult(text=text, model_id="fake-model")
            self.decision = RouteDecision(
                chosen="fake-model", task_class=TaskClass.KB_QA,
                confidence=1.0, candidates=[], reason="test",
            )

        @property
        def text(self):
            return text

    return Outcome()


# ---------------------------------------------------------------- the opt-in

def test_a_simulated_transcription_is_stored_stamped_and_never_indexed(tmp_path, kb):
    """The KAVACH_KB_ALLOW_SIMULATED_OCR path, end to end.

    It exists so the image path can be exercised before real vision weights are
    staged. Everything it produces has to stay visibly synthetic all the way to
    the citation, or it is just a way to make the corpus panel look full.
    """
    from kavach.knowledge import Ingestor

    store, index, retriever, _ = kb
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    scanned_pdf(corpus / "scan.pdf")

    ingestor = Ingestor(
        store=store, index=index, config=load_knowledge_config(),
        reader=FakeReader(text="INSPECTION REPORT 2026", extraction=Extraction.SIMULATED),
    )
    report = ingestor.ingest(corpus)

    # Stored, so the panel can count it...
    assert report.simulated_pages == 1
    assert store.document("scan").simulated_pages == [1]
    assert store.chunks()[0].citation.startswith("[SIMULATED ")
    # ...over-classified, because nobody actually read the marking...
    assert store.document("scan").label is Label.CONFIDENTIAL
    # ...never indexed, and the ingest says so rather than leaving a chunk count
    # of zero to be interpreted.
    assert index.state().chunks == 0
    assert retriever.search("INSPECTION").hits == []
    assert "SIMULATED" in report.warning

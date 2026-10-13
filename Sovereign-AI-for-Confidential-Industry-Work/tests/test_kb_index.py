"""The hybrid index (§07).

Two failure modes get the most attention here, because both produce numbers that
look fine:

    comparing vectors from two different embedders (cosine returns a float, it
    just does not mean anything), and

    searching an index that no longer covers the corpus.
"""

from __future__ import annotations

import pytest

from kavach.knowledge import Extraction, HybridIndex, tokenise
from kavach.knowledge.config import load_knowledge_config
from kavach.knowledge.store import KnowledgeStore
from kavach.knowledge.types import Chunk, DocumentRecord, PageRecord
from kavach.types import Label


@pytest.fixture()
def config():
    return load_knowledge_config()


@pytest.fixture()
def store():
    s = KnowledgeStore()
    s.clear()
    yield s
    s.clear()


def document(store, doc_id: str, texts: list[str], extraction=Extraction.NATIVE,
             label=Label.INTERNAL):
    record = DocumentRecord(
        id=doc_id, path=f"/corpus/{doc_id}.pdf", title=doc_id,
        sha256=doc_id.encode().hex().ljust(64, "0")[:64], label=label,
        pages=[PageRecord(1, extraction, sum(len(t) for t in texts), "test")],
    )
    chunks = [
        Chunk(
            id=f"{doc_id}#p1c{i}", doc_id=doc_id, page=1, ordinal=i, text=text,
            label=label, extraction=extraction,
        )
        for i, text in enumerate(texts)
    ]
    store.put_document(record, chunks)
    return record, chunks


# ---------------------------------------------------------------- tokenisation

def test_an_instrument_tag_survives_tokenisation_whole():
    """`P-101A` is the highest-signal token an engineer types and the one an
    embedder has never seen."""
    tokens = tokenise("Trip PSLL-101 on pump P-101A")
    assert "psll-101" in tokens
    assert "p-101a" in tokens


def test_a_tag_is_also_emitted_in_parts():
    """Emitting only the whole tag makes a document unfindable by any query that
    does not reproduce it verbatim."""
    tokens = tokenise("MRPL-SOP-CDU-0101")
    assert "mrpl-sop-cdu-0101" in tokens
    assert {"mrpl", "sop", "cdu", "0101"} <= set(tokens)


def test_a_plain_word_is_not_duplicated():
    assert tokenise("pump").count("pump") == 1


def test_punctuation_and_case_do_not_change_the_tokens():
    assert tokenise("Pump, P-101A.") == tokenise("pump p-101a")


# ---------------------------------------------------------------- both arms

def test_the_lexical_arm_finds_an_exact_tag(tmp_path, store):
    document(store, "SOP", [
        "PSLL-101 suction pressure trip setpoint is 1.80 bar gauge on CDU-II.",
        "The fire watch must remain on station for thirty minutes after welding.",
    ])
    index = HybridIndex(tmp_path, store=store)
    index.build()

    ranking = index.bm25_ranking("PSLL-101", 5)
    assert ranking and ranking[0][0] == "SOP#p1c0"


def test_the_dense_arm_finds_a_paraphrase_with_no_shared_rare_token(tmp_path, store):
    document(store, "SOP", [
        "The motor trips instantly to prevent dry-run damage to the seal faces.",
        "Permits expire at the end of the issuing shift and are not carried over.",
    ])
    index = HybridIndex(tmp_path, store=store)
    index.build()

    assert index.state().dense_ready
    ranking = index.dense_ranking("what stops the motor to protect the seals", 5)
    assert ranking


def test_both_arms_report_themselves_live_on_a_healthy_index(tmp_path, store):
    document(store, "SOP", ["Content about the crude booster pump and its interlocks."])
    state = HybridIndex(tmp_path, store=store).build()
    assert state.lexical_ready and state.dense_ready
    assert not state.degraded


# ---------------------------------------------------------------- refusals

def test_vectors_from_a_different_embedder_are_never_compared(tmp_path, store):
    """Cosine between bge-m3 and minilm space is a number, not a similarity."""
    document(store, "SOP", ["Content about the crude booster pump and its interlocks."])
    index = HybridIndex(tmp_path, store=store)
    index.build()
    assert index.state().dense_ready

    index.embed_model = "some-other-embedder"
    state = index.state()

    assert not state.dense_ready
    assert state.degraded
    assert "not comparable" in state.detail
    assert index.dense_ranking("pump", 5) == []


def test_an_index_that_no_longer_covers_the_corpus_says_so(tmp_path, store):
    document(store, "SOP", ["Content about the crude booster pump and its interlocks."])
    index = HybridIndex(tmp_path, store=store)
    index.build()
    assert not index.state().stale

    document(store, "HSE", ["Hot work permits expire at the end of the issuing shift."])

    assert index.is_stale()
    state = index.state()
    assert state.degraded
    assert "STALE" in state.detail


def test_forgetting_a_document_also_makes_the_index_stale(tmp_path, store):
    document(store, "SOP", ["Content about the crude booster pump and its interlocks."])
    index = HybridIndex(tmp_path, store=store)
    index.build()

    store.forget("SOP")
    assert index.is_stale()


def test_simulated_chunks_are_never_indexed(tmp_path, store):
    """A hit on invented text is indistinguishable from a hit on the document."""
    document(store, "REAL", ["Genuine clause text about the pump suction line."])
    document(store, "FAKE", ["Invented transcription of a page nobody read."],
             extraction=Extraction.SIMULATED)

    index = HybridIndex(tmp_path, store=store)
    state = index.build()

    assert state.chunks == 1
    assert all(cid.startswith("REAL") for cid in index.chunk_ids)
    assert index.bm25_ranking("invented transcription", 5) == []


def test_an_empty_corpus_is_degraded_not_broken(tmp_path, store):
    state = HybridIndex(tmp_path, store=store).build()
    assert state.chunks == 0
    assert state.degraded
    assert not state.lexical_ready


# ---------------------------------------------------------------- persistence

def test_an_index_survives_a_process_restart(tmp_path, store):
    document(store, "SOP", [
        "PSLL-101 suction pressure trip setpoint is 1.80 bar gauge on CDU-II.",
    ])
    HybridIndex(tmp_path, store=store).build()

    # A fresh object, as a new process would build it - nothing carried over.
    reloaded = HybridIndex(tmp_path, store=store)

    assert reloaded.chunk_ids == ["SOP#p1c0"]
    assert reloaded.embed_model is not None
    assert not reloaded.is_stale()
    assert reloaded.bm25_ranking("PSLL-101", 5)
    assert reloaded.dense_ranking("suction pressure trip", 5)


def test_a_vector_file_that_does_not_match_the_meta_is_discarded(tmp_path, store):
    document(store, "SOP", ["Content about the crude booster pump and its interlocks."])
    index = HybridIndex(tmp_path, store=store)
    index.build()

    (tmp_path / "kb" / "vectors.npy").write_bytes(b"not a numpy file")
    reloaded = HybridIndex(tmp_path, store=store)

    assert not reloaded.state().dense_ready


def test_a_fresh_install_is_not_reported_as_stale(tmp_path, store):
    """No index file and no corpus agree with each other.

    Saying STALE here sends an operator to `kb reindex` to fix a corpus with
    nothing wrong with it, and teaches them to ignore the word by the time it
    means something.
    """
    index = HybridIndex(tmp_path, store=store)
    assert not index.is_stale()
    assert "STALE" not in index.state().detail

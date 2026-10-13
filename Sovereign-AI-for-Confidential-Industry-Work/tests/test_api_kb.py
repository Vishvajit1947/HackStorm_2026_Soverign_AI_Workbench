"""L3 gateway endpoints (§07).

`GET /api/kb` is the corpus panel's single poll, and the thing it must never do
is show only what worked. A judge asking "what is in there?" gets the extraction
breakdown and the gaps in the same response.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app
from kavach.knowledge import get_index, get_ingestor, get_knowledge_store
from kbfixtures import HSE_TEXT, SOP_TEXT, native_pdf, scanned_pdf


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def corpus(tmp_path):
    """A three-document corpus: two readable, one that nothing here can read."""
    store = get_knowledge_store()
    store.clear()
    folder = tmp_path / "corpus"
    folder.mkdir()
    native_pdf(folder / "SOP_01.pdf", SOP_TEXT)
    native_pdf(folder / "SOP_02.pdf", HSE_TEXT)
    scanned_pdf(folder / "inspection.pdf")
    get_ingestor().ingest(folder)
    yield folder
    store.clear()
    get_index().build()


# ---------------------------------------------------------------- the panel

def test_the_corpus_panel_is_one_poll(client, corpus):
    body = client.get("/api/kb").json()
    assert set(body) == {"corpus", "index", "image_path"}
    assert body["corpus"]["documents"] == 3
    assert body["index"]["chunks"] > 0


def test_the_panel_leads_with_the_pages_it_could_not_read(client, corpus):
    body = client.get("/api/kb").json()["corpus"]
    assert body["unreadable_pages"] == 1
    assert body["gaps"] == [{"doc_id": "inspection", "pages": [1]}]
    assert "could not be read" in body["warning"]


def test_the_extraction_breakdown_distinguishes_read_from_unread(client, corpus):
    extraction = client.get("/api/kb").json()["corpus"]["extraction"]
    assert extraction["NATIVE"] == 2
    assert extraction["UNREADABLE"] == 1


def test_doctor_names_its_blockers(client, corpus):
    body = client.get("/api/kb/doctor").json()
    assert body["ok"] is False
    assert any("could not be read" in b for b in body["blockers"])
    assert body["image_path"]["mode"] in {"UNREADABLE", "SIMULATED", "OCR", "VLM"}


# ---------------------------------------------------------------- documents

def test_documents_carry_their_classification(client, corpus):
    body = client.get("/api/kb/documents").json()
    labels = {d["id"]: d["label"] for d in body["documents"]}
    assert labels["SOP_01"] == "CONFIDENTIAL"      # the marking on its own face
    assert labels["SOP_02"] == "INTERNAL"
    assert labels["inspection"] == "CONFIDENTIAL"  # unread -> over-classified


def test_one_document_exposes_how_each_page_was_read(client, corpus):
    body = client.get("/api/kb/documents/inspection").json()
    assert body["page_detail"][0]["extraction"] == "UNREADABLE"
    assert body["page_detail"][0]["detail"]


def test_an_unknown_document_is_a_404(client, corpus):
    assert client.get("/api/kb/documents/nope").status_code == 404
    assert client.delete("/api/kb/documents/nope").status_code == 404
    assert client.get("/api/kb/chunks", params={"doc_id": "nope"}).status_code == 404


def test_chunks_are_reachable_from_a_citation(client, corpus):
    body = client.get("/api/kb/chunks", params={"doc_id": "SOP_01", "page": 1}).json()
    assert body["count"] > 0
    assert body["chunks"][0]["citation"].startswith("[SOP_01 p.1")


# ---------------------------------------------------------------- ingest

def test_ingesting_a_missing_path_is_a_404(client):
    body = client.post("/api/kb/ingest", json={"path": "no/such/folder"})
    assert body.status_code == 404


def test_ingest_reports_the_unreadable_pages_it_created(client, tmp_path):
    get_knowledge_store().clear()
    folder = tmp_path / "one"
    folder.mkdir()
    scanned_pdf(folder / "scan.pdf")

    body = client.post("/api/kb/ingest", json={"path": str(folder)}).json()

    assert body["ingested"] == 1
    assert body["unreadable_pages"] == 1
    assert "NOT in the index" in body["warning"]
    get_knowledge_store().clear()
    get_index().build()


def test_reindex_and_reload_are_live(client, corpus):
    assert client.post("/api/kb/reindex").json()["chunks"] > 0
    assert "retrieval" in client.post("/api/kb/reload").json()


# ---------------------------------------------------------------- search

def test_search_returns_citations_and_coverage(client, corpus):
    body = client.post("/api/kb/search", json={"query": "PSLL-101 trip setpoint"}).json()
    assert body["count"] > 0
    assert body["citations"][0].startswith("[SOP_01 p.1")
    assert body["coverage"]["complete"] is False


def test_search_withholds_by_clearance_and_says_how_much(client, corpus):
    body = client.post(
        "/api/kb/search", json={"query": "PSLL-101 trip setpoint", "role": "operator"}
    ).json()
    assert body["withheld"] >= 1
    assert "cleared to INTERNAL" in body["withheld_reason"]
    assert all(h["label"] != "CONFIDENTIAL" for h in body["hits"])


def test_an_empty_query_is_rejected_by_validation(client, corpus):
    assert client.post("/api/kb/search", json={"query": ""}).status_code == 422


# ---------------------------------------------------------------- answer

def test_a_refusal_to_answer_is_a_result_not_an_error(client, corpus):
    """A 4xx here becomes 'something went wrong' in the UI. The refusal IS the
    answer, and the panel has to render it with its reason."""
    body = client.post("/api/kb/answer", json={"question": "xyzzy plugh nothing"})
    assert body.status_code == 200
    payload = body.json()
    assert payload["ok"] is False
    assert payload["answer"] is None
    assert payload["headline"]


def test_an_answer_carries_its_sources_and_its_grounding_verdict(client, corpus):
    payload = client.post(
        "/api/kb/answer", json={"question": "What is the PSLL-101 trip setpoint?"}
    ).json()

    assert payload["sources"]
    assert payload["label"] == "CONFIDENTIAL"
    if payload["ok"]:
        # The mock backend copies citations out of the context, so the citations
        # resolve - and the prose between them is still synthetic. Those are two
        # different facts and the payload keeps them apart.
        assert payload["simulated"] is True
        assert "SIMULATED" in payload["headline"]
        assert payload["grounding"]["resolved"] > 0
    else:
        assert payload["rejected_answer"] is not None


def test_an_answer_never_hides_a_corpus_with_a_hole_in_it(client, corpus):
    payload = client.post(
        "/api/kb/answer", json={"question": "What is the PSLL-101 trip setpoint?"}
    ).json()
    assert payload["retrieval"]["coverage"]["complete"] is False

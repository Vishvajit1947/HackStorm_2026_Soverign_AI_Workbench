"""L2 gateway endpoints - the explainer panel's data source (§04)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------- explain

def test_explain_returns_everything_the_panel_renders(client):
    body = client.post("/api/route/explain", json={
        "prompt": "Script the NPSH-available calculation and unit-test it"
    }).json()

    assert body["decision"]["chosen"] == "qwen2_5-coder-7b-q4"
    assert body["decision"]["class"] == "code_generation"
    assert body["classification"]["method"] == "embedding"
    assert body["weights"]["w_quality"] == 1.0
    assert body["decision"]["candidates"]


def test_explain_shows_score_terms_per_candidate(client):
    body = client.post("/api/route/explain", json={"prompt": "Write a parser"}).json()
    winner = next(
        c for c in body["decision"]["candidates"]
        if c["model"] == body["decision"]["chosen"]
    )
    for term in ("quality", "confidence_fit", "latency", "swap", "vram"):
        assert term in winner["terms"]


def test_explain_shows_eliminations_with_reasons(client):
    body = client.post("/api/route/explain", json={
        "prompt": "Read this scan", "has_images": True
    }).json()
    eliminated = [c for c in body["decision"]["candidates"] if c["eliminated_by"]]
    assert eliminated
    assert any("vision" in c["eliminated_by"] for c in eliminated)


def test_explain_declares_where_its_numbers_came_from(client):
    """The panel must answer "where does 0.65 come from?" without being asked.
    Which answer it gives depends on whether the harness has run; that it gives
    one, and warns while they are guesses, does not."""
    body = client.post("/api/route/explain",
                       json={"prompt": "summarise this document"}).json()
    provenance = body["quality_provenance"]
    assert provenance in {"seeded", "partially_measured", "measured"}
    if provenance == "seeded":
        assert body.get("quality_warning"), "seeded numbers with no warning"
    else:
        assert "eval_harness" in json.dumps(body) or body.get("quality_warning") is None


def test_the_task_class_reaches_the_decision_over_http(client):
    """Two different task types must be CLASSIFIED differently and each carry a
    full score breakdown. They no longer have to choose different models: the
    measured table says these 7B models are equivalent at code and summary, and
    a test that demands they differ is asserting that the numbers are wrong."""
    code = client.post("/api/route/explain", json={
        "prompt": "Script the NPSH-available calculation and unit-test it"}).json()
    summary = client.post("/api/route/explain", json={
        "prompt": "Summarise this 40-page SOP and list the safety interlocks"}).json()
    assert code["decision"]["class"] != summary["decision"]["class"]
    for body in (code, summary):
        assert body["decision"]["chosen"]
        assert body["decision"]["candidates"], "no score breakdown for the panel"


# ---------------------------------------------------------------- classify

def test_classify_endpoint(client):
    body = client.post("/api/route/classify", json={
        "prompt": "Draft an approval note to the GM"}).json()
    assert body["class"] == "official_drafting"
    assert body["neighbours"]


def test_forced_class_is_honoured_over_http(client):
    body = client.post("/api/route/classify", json={
        "prompt": "anything", "forced_class": "visual_doc"}).json()
    assert body["class"] == "visual_doc"
    assert body["method"] == "forced"


# ---------------------------------------------------------------- generate

def test_route_generates(client):
    body = client.post("/api/route", json={
        "prompt": "Summarise this 40-page SOP and list the safety interlocks",
        "max_tokens": 128,
    }).json()
    assert body["text"]
    # Not a named model: which one wins a tie is a measurement result, not a
    # contract. That it chose SOMETHING routable and verified is the contract.
    assert body["model"], "no model was chosen"
    assert body["verification"]["ok"] is True
    assert body["tokens"]["completion"] > 0


def test_impossible_request_is_422_with_the_decision(client):
    resp = client.post("/api/route", json={
        "prompt": "anything", "min_context": 10000000})
    assert resp.status_code == 422
    assert resp.json()["detail"]["decision"]["chosen"] is None


def test_empty_prompt_is_rejected(client):
    assert client.post("/api/route/explain", json={"prompt": ""}).status_code == 422


# ---------------------------------------------------------------- prototypes

def test_prototypes_are_listed(client):
    body = client.get("/api/route/prototypes").json()
    assert body["total"] > 100
    assert all(n > 0 for n in body["per_class"].values())


def test_prototypes_reload(client):
    """The 'fix a misroute in seconds' loop, over HTTP."""
    body = client.post("/api/route/prototypes/reload").json()
    assert body["loaded"] > 100


def test_config_endpoint_exposes_the_weights(client):
    body = client.get("/api/route/config").json()
    assert body["weights"]["w_swap"] == 0.20
    assert body["verifier"]["max_retries"] == 1
    assert body["quality_provenance"] in {"seeded", "partially_measured", "measured"}

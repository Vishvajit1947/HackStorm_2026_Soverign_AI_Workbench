"""L1 gateway endpoints.

Covers the extensibility beat (§09, 10:45) end to end: paste a manifest entry, hit
reload, and it is a routing candidate on the next request with no restart.
"""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from kavach.api.app import create_app
from kavach.config import get_settings


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def manifest_path():
    """Restores config/models.yaml after any test that edits it."""
    path = get_settings().config_dir / "models.yaml"
    original = path.read_text(encoding="utf-8")
    yield path
    path.write_text(original, encoding="utf-8")


# ---------------------------------------------------------------- registry

def test_list_models(client):
    body = client.get("/api/models").json()
    assert body["count"] > 0
    ids = {m["id"] for m in body["models"]}
    assert "qwen3-8b-q4" in ids


def test_listed_models_carry_routing_metadata(client):
    """The router (Phase 3) reads exactly these fields."""
    body = client.get("/api/models").json()
    entry = next(m for m in body["models"] if m["id"] == "qwen3-8b-q4")
    for key in ("resident", "swap_cost_ms", "fits", "quality", "vram_gb", "provenance"):
        assert key in entry, f"missing {key}"


def test_simulated_models_are_flagged_not_hidden(client):
    body = client.get("/api/models").json()
    assert body["simulate"] is True
    assert body["all_weights_real"] is False
    assert "SIMULATED" in body["warning"]


def test_verify_reports_not_demo_ready_without_weights(client):
    body = client.get("/api/models/verify").json()
    assert body["ok"] is False
    assert body["simulated"]
    assert "not demo-ready" in body["detail"]


# ---------------------------------------------------------------- warden

def test_residency_shows_the_pinned_embedder(client):
    body = client.get("/api/models/residency").json()
    assert any(r["pinned"] for r in body["resident"])
    assert body["budget_gb"] > 0


def test_load_admits_a_model(client):
    body = client.post("/api/models/load", json={"model_id": "qwen3-8b-q4"}).json()
    assert body["decision"]["admitted"] is True
    assert any(r["model_id"] == "qwen3-8b-q4" for r in body["residency"]["resident"])


def test_loading_twice_is_a_free_hit(client):
    client.post("/api/models/load", json={"model_id": "qwen3-8b-q4"})
    body = client.post("/api/models/load", json={"model_id": "qwen3-8b-q4"}).json()
    assert body["decision"]["was_resident"] is True
    assert body["decision"]["load_ms"] == 0.0


def test_loading_a_second_model_evicts_the_first(client):
    """One swap slot on MID: the vision model must evict the text model."""
    client.post("/api/models/evict", json={"include_pinned": False})
    client.post("/api/models/load", json={"model_id": "qwen3-8b-q4"})
    body = client.post("/api/models/load", json={"model_id": "qwen3-vl-8b-q4"}).json()
    assert "qwen3-8b-q4" in body["decision"]["evicted"]


def test_unknown_model_is_404(client):
    assert client.post("/api/models/load", json={"model_id": "nope"}).status_code == 404


def test_evict_spares_pinned(client):
    client.post("/api/models/load", json={"model_id": "qwen3-8b-q4"})
    body = client.post("/api/models/evict", json={"include_pinned": False}).json()
    assert any(r["pinned"] for r in body["residency"]["resident"])


# ---------------------------------------------------------------- extensibility

def test_reload_picks_up_a_pasted_manifest_entry(client, manifest_path):
    """The 10:45 demo beat: paste, reload, it is a candidate. No restart."""
    assert not any(
        m["id"] == "demo-newcomer-3b" for m in client.get("/api/models").json()["models"]
    )

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw.append({
        "id": "demo-newcomer-3b",
        "file": "demo-newcomer-3b.gguf",
        "sha256": None,
        "license": "apache-2.0",
        "tier": 2,
        "capabilities": ["text", "tools"],
        "context": 16384,
        "vram_gb": 2.4,
        "load_ms": 1800,
        "tok_per_s": 44,
        "quality": {"code_generation": 0.97},
    })
    manifest_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    body = client.post("/api/models/reload").json()
    assert "demo-newcomer-3b" in body["added"]

    listed = client.get("/api/models").json()["models"]
    newcomer = next(m for m in listed if m["id"] == "demo-newcomer-3b")
    assert newcomer["quality"]["code_generation"] == 0.97
    assert newcomer["fits"] is True

    # Put it back so the rest of the suite sees the real registry.
    client.post("/api/models/reload")


def test_reload_is_audited(client):
    client.post("/api/models/reload")
    events = [r["event"] for r in client.get("/api/audit/tail?n=30").json()["records"]]
    assert "model.registry_reload" in events


# ---------------------------------------------------------------- dashboard

def test_health_reports_vram(client):
    body = client.get("/api/health").json()
    assert "vram_used_gb" in body
    assert body["models_registered"] > 0


def test_sovereignty_panel_includes_the_vram_meter(client):
    body = client.get("/api/sovereignty").json()
    assert "vram" in body
    assert "resident" in body["vram"]
    assert body["models"]["backend"]["simulated"] is True


def test_backend_health_declares_mock(client):
    body = client.get("/api/models/backend").json()
    assert body["backend"] == "mock"
    assert body["simulated"] is True

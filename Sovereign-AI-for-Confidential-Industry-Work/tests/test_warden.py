"""VRAM warden (§03).

The warden is the component that stops the card OOMing mid-demo, so the tests are
mostly about pressure: what gets evicted, what never gets evicted, and whether the
swap cost the router reads is actually true.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import Profile
from kavach.runtime.backends.mock import MockBackend
from kavach.runtime.registry import ModelRegistry
from kavach.runtime.warden import VramExhausted, VramWarden

MANIFEST = [
    {"id": "emb", "file": "emb.gguf", "role": "embedding", "tier": 0,
     "capabilities": ["embedding"], "context": 512, "vram_gb": 1.0,
     "load_ms": 200, "dim": 64},
    {"id": "small", "file": "small.gguf", "tier": 1, "capabilities": ["text"],
     "context": 8192, "vram_gb": 2.0, "load_ms": 1000},
    {"id": "text", "file": "text.gguf", "tier": 2, "capabilities": ["text", "tools"],
     "context": 8192, "vram_gb": 4.0, "load_ms": 2000},
    {"id": "vision", "file": "vision.gguf", "tier": 2, "capabilities": ["text", "vision"],
     "context": 8192, "vram_gb": 5.0, "load_ms": 3000},
    {"id": "huge", "file": "huge.gguf", "tier": 3, "capabilities": ["text"],
     "context": 8192, "vram_gb": 40.0, "load_ms": 9000},
]


def make_profile(vram_gb=12.0, reserve_gb=2.0, swap_slots=1, pinned=("emb",)):
    return Profile(
        name="TEST",
        description="test profile",
        vram_gb=vram_gb,
        reserve_gb=reserve_gb,
        cpu_only=False,
        context=8192,
        pinned=list(pinned),
        available=[m["id"] for m in MANIFEST],
        swap_slots=swap_slots,
    )


@pytest.fixture()
def warden(tmp_path):
    config_dir = tmp_path / "config"
    model_dir = tmp_path / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")

    registry = ModelRegistry(manifest, model_dir, simulate=True)
    backend = MockBackend(time_scale=0.0)
    w = VramWarden(make_profile(), registry, backend=backend)
    w.pin_profile_models()
    return w


# ---------------------------------------------------------------- budget

def test_budget_subtracts_reserve_and_kv_cache(warden):
    # 12 GB card - 2 GB reserve - 0.2 GB KV (8192 ctx) = 9.8 GB
    assert warden.budget_gb == pytest.approx(9.8, abs=0.01)


def test_pinned_model_is_loaded_at_boot(warden):
    assert warden.is_resident("emb")
    assert warden.used_gb == pytest.approx(1.0)


# ---------------------------------------------------------------- admission

def test_admitting_loads_the_model(warden):
    decision = warden.admit("text")
    assert decision.admitted
    assert not decision.was_resident
    assert warden.is_resident("text")


def test_second_admit_is_a_free_hit(warden):
    warden.admit("text")
    decision = warden.admit("text")
    assert decision.was_resident
    assert decision.evicted == []
    assert decision.load_ms == 0.0


def test_admit_evicts_lru_when_slots_are_full(warden):
    """One swap slot: admitting vision must evict text."""
    warden.admit("text")
    decision = warden.admit("vision")
    assert decision.evicted == ["text"]
    assert warden.is_resident("vision")
    assert not warden.is_resident("text")


def test_pinned_models_are_never_evicted(warden):
    warden.admit("text")
    warden.admit("vision")
    warden.admit("small")
    assert warden.is_resident("emb"), "the pinned embedder was evicted"


def test_model_too_large_for_the_card_is_rejected(warden):
    with pytest.raises(VramExhausted) as exc:
        warden.admit("huge")
    assert "huge" in str(exc.value)
    assert "budget" in str(exc.value).lower()


def test_rejection_does_not_disturb_residency(warden):
    warden.admit("text")
    with pytest.raises(VramExhausted):
        warden.admit("huge")
    assert warden.is_resident("text"), "a failed admission evicted a healthy model"


def test_unknown_model_raises(warden):
    with pytest.raises(KeyError):
        warden.admit("does-not-exist")


# ---------------------------------------------------------------- swap slots

def test_two_slots_keep_both_models_resident(tmp_path):
    """HIGH profile: vision must not evict the text model mid-run."""
    config_dir = tmp_path / "config"
    model_dir = tmp_path / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")

    registry = ModelRegistry(manifest, model_dir, simulate=True)
    w = VramWarden(
        make_profile(vram_gb=16.0, reserve_gb=2.2, swap_slots=2),
        registry,
        backend=MockBackend(time_scale=0.0),
    )
    w.pin_profile_models()

    w.admit("text")
    w.admit("vision")
    assert w.is_resident("text") and w.is_resident("vision")


# ---------------------------------------------------------------- swap cost

def test_resident_model_costs_nothing_to_swap(warden):
    warden.admit("text")
    assert warden.swap_cost("text") == 0.0


def test_absent_model_costs_at_least_its_load_time(warden):
    assert warden.swap_cost("text") >= 2000.0


def test_swap_cost_includes_eviction_when_full(warden):
    """This is the term that makes the router honest about a 12 GB card."""
    warden.admit("vision")
    cost = warden.swap_cost("text")
    assert cost > 2000.0, "eviction was not priced into the swap cost"


def test_unknown_model_has_infinite_swap_cost(warden):
    assert warden.swap_cost("nope") == float("inf")


def test_fits_reports_what_could_ever_be_admitted(warden):
    assert warden.fits("text") is True
    assert warden.fits("huge") is False


# ---------------------------------------------------------------- backend

def test_backend_actually_loads_and_unloads(warden):
    backend = warden.backend
    warden.admit("text")
    assert "text" in backend.loaded_models()
    warden.admit("vision")          # evicts text
    assert "text" not in backend.loaded_models()
    assert "vision" in backend.loaded_models()


def test_evict_all_spares_pinned_by_default(warden):
    warden.admit("text")
    evicted = warden.evict_all()
    assert "text" in evicted
    assert "emb" not in evicted
    assert warden.is_resident("emb")


def test_evict_all_can_include_pinned(warden):
    warden.admit("text")
    evicted = warden.evict_all(include_pinned=True)
    assert "emb" in evicted
    assert not warden.is_resident("emb")


# ---------------------------------------------------------------- reporting

def test_state_reports_the_vram_meter(warden):
    warden.admit("text")
    state = warden.state()
    assert state["used_gb"] == pytest.approx(5.0)
    assert state["slots_in_use"] == 1
    assert state["swap_slots"] == 1
    assert any(r["pinned"] for r in state["resident"])


def test_pin_failure_is_reported_not_raised(tmp_path):
    """A missing embedder should degrade the router, not stop the box booting."""
    config_dir = tmp_path / "config"
    model_dir = tmp_path / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")

    registry = ModelRegistry(manifest, model_dir, simulate=True)
    w = VramWarden(
        make_profile(pinned=("no-such-embedder",)),
        registry,
        backend=MockBackend(time_scale=0.0),
    )
    pinned = w.pin_profile_models()

    assert pinned == []
    assert any("not in the registry" in f for f in w.pin_failures)

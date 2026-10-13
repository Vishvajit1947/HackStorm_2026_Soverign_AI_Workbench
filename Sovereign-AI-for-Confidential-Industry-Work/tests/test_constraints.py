"""Stage 1 - the constraint filter (§04).

Stage 1 removes impossible candidates rather than scoring them. That distinction
is the point: a model with no vision tower is not a bad choice for a scanned page,
it is not a choice at all, and letting it into the scoring stage means a high
quality number can win an argument it should never have been in.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import Profile
from kavach.router.constraints import apply_constraints, summarise, survivors
from kavach.runtime.backends.mock import MockBackend
from kavach.runtime.registry import ModelRegistry
from kavach.runtime.warden import VramWarden
from kavach.types import Label, RouteRequest

MANIFEST = [
    {"id": "emb", "file": "emb.gguf", "role": "embedding", "tier": 0,
     "capabilities": ["embedding"], "context": 512, "vram_gb": 1.0,
     "load_ms": 200, "dim": 64},
    {"id": "text", "file": "text.gguf", "tier": 2, "capabilities": ["text", "tools"],
     "context": 8192, "vram_gb": 4.0, "load_ms": 2000,
     "quality": {"code_generation": 0.8}},
    {"id": "vision", "file": "vision.gguf", "tier": 2,
     "capabilities": ["text", "vision"], "context": 8192, "vram_gb": 5.0,
     "load_ms": 3000, "quality": {"visual_doc": 0.9}},
    {"id": "notools", "file": "nt.gguf", "tier": 1, "capabilities": ["text"],
     "context": 4096, "vram_gb": 2.0, "load_ms": 1000,
     "quality": {"code_generation": 0.5}},
    {"id": "huge", "file": "huge.gguf", "tier": 3, "capabilities": ["text", "tools"],
     "context": 32768, "vram_gb": 40.0, "load_ms": 9000,
     "quality": {"code_generation": 0.99}},
]


@pytest.fixture()
def env(tmp_path):
    config_dir, model_dir = tmp_path / "config", tmp_path / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")

    registry = ModelRegistry(manifest, model_dir, simulate=True)
    profile = Profile(
        name="TEST", description="", vram_gb=12.0, reserve_gb=2.0, cpu_only=False,
        context=8192, pinned=["emb"], available=[m["id"] for m in MANIFEST],
        swap_slots=1,
    )
    warden = VramWarden(profile, registry, backend=MockBackend(time_scale=0.0))
    warden.pin_profile_models()
    return registry, warden


def run(env, req, max_tier=None):
    registry, warden = env
    return apply_constraints(registry.all(), req, warden=warden, max_tier=max_tier)


def surviving_ids(candidates):
    return sorted(c.model_id for c in survivors(candidates))


def reason_for(candidates, model_id):
    return next(c.eliminated_by for c in candidates if c.model_id == model_id)


# ---------------------------------------------------------------- role

def test_embedders_are_never_candidates(env):
    cands = run(env, RouteRequest(prompt="anything"))
    assert "emb" not in surviving_ids(cands)
    assert "role=embedding" in reason_for(cands, "emb")


# ---------------------------------------------------------------- vision

def test_images_eliminate_models_without_a_vision_tower(env):
    cands = run(env, RouteRequest(prompt="read this scan", has_images=True))
    assert surviving_ids(cands) == ["vision"]
    assert reason_for(cands, "text") == "images attached, no vision tower"


def test_no_images_keeps_text_models(env):
    cands = run(env, RouteRequest(prompt="write code"))
    assert "text" in surviving_ids(cands)


# ---------------------------------------------------------------- tools

def test_tool_requirement_eliminates_models_without_tool_calling(env):
    cands = run(env, RouteRequest(prompt="call a tool", needs_tools=True))
    assert "notools" not in surviving_ids(cands)
    assert reason_for(cands, "notools") == "tool calling required, not supported"


# ---------------------------------------------------------------- context

def test_context_requirement_eliminates_short_context_models(env):
    cands = run(env, RouteRequest(prompt="long doc", min_context=8000))
    assert "notools" not in surviving_ids(cands)          # 4096 context
    assert "needs 8000 ctx, has 4096" in reason_for(cands, "notools")


def test_context_exactly_at_the_limit_is_allowed(env):
    cands = run(env, RouteRequest(prompt="x", min_context=4096))
    assert "notools" in surviving_ids(cands)


# ---------------------------------------------------------------- vram

def test_a_model_that_cannot_fit_is_eliminated(env):
    """Routing to it would only produce an OOM, so it is not a candidate."""
    cands = run(env, RouteRequest(prompt="write code"))
    assert "huge" not in surviving_ids(cands)
    assert "will not fit" in reason_for(cands, "huge")


def test_the_best_model_is_still_eliminated_if_it_cannot_fit(env):
    """huge has quality 0.99 - the highest by far. Stage 1 must remove it anyway,
    which is precisely why elimination happens before scoring."""
    cands = run(env, RouteRequest(prompt="write code"))
    assert "huge" not in surviving_ids(cands)


# ---------------------------------------------------------------- provenance

def test_unverified_weights_are_never_routable(tmp_path):
    config_dir, model_dir = tmp_path / "config", tmp_path / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")

    # simulate=False and no staged files -> everything UNVERIFIED
    registry = ModelRegistry(manifest, model_dir, simulate=False)
    cands = apply_constraints(registry.all(), RouteRequest(prompt="x"))
    assert survivors(cands) == []
    assert "UNVERIFIED" in reason_for(cands, "text")


# ---------------------------------------------------------------- policy tier

def test_policy_tier_ceiling_eliminates_high_tiers(env):
    cands = run(env, RouteRequest(prompt="x", label=Label.RESTRICTED), max_tier=2)
    assert "huge" not in surviving_ids(cands)           # tier 3
    assert "text" in surviving_ids(cands)               # tier 2


def test_no_ceiling_allows_every_tier(env):
    cands = run(env, RouteRequest(prompt="x"), max_tier=None)
    ids = surviving_ids(cands)
    assert "text" in ids and "vision" in ids


# ---------------------------------------------------------------- reporting

def test_eliminated_candidates_are_kept_for_the_panel(env):
    """The panel shows '5 -> 3' with reasons, not a silent shortlist."""
    cands = run(env, RouteRequest(prompt="read this", has_images=True))
    assert len(cands) == len(MANIFEST)
    assert all(c.eliminated_by for c in cands if not c.survived)


def test_summarise_reports_the_narrowing(env):
    cands = run(env, RouteRequest(prompt="read this", has_images=True))
    assert summarise(cands) == "5 -> 1"


def test_the_first_failure_is_the_reported_one(env):
    """A model can fail several checks; the message names the most fundamental."""
    cands = run(env, RouteRequest(prompt="x", has_images=True, min_context=99999))
    assert reason_for(cands, "notools") == "images attached, no vision tower"

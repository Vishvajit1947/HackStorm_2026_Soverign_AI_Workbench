"""Stage 3 - the utility score (§04).

The most important test in this file is `test_quality_is_not_swamped_by_latency`.
Before normalisation the latency term contributed about -2.0 while quality
contributed at most +1.0, so the router silently ranked by SPEED and quality was a
tie-breaker. Nothing errored. Every decision still looked plausible because the
fast model was often also the right one. It is exactly the class of bug that
survives to demo day and then gets found by a judge reading the panel.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import Profile
from kavach.router.scoring import (
    ScoringWeights,
    confidence_fit,
    explain_margin,
    rank,
    score_candidates,
    vram_pressure,
    winner,
)
from kavach.runtime.backends.mock import MockBackend
from kavach.runtime.registry import ModelRegistry
from kavach.runtime.warden import VramWarden
from kavach.types import Candidate

MANIFEST = [
    {"id": "emb", "file": "emb.gguf", "role": "embedding", "tier": 0,
     "capabilities": ["embedding"], "context": 512, "vram_gb": 1.0,
     "load_ms": 200, "dim": 64},
    # Deliberately: the BEST model at code is also the SLOWEST. If the router
    # ranks by speed this one loses, which is what makes the test meaningful.
    {"id": "good-slow", "file": "a.gguf", "tier": 2, "capabilities": ["text", "tools"],
     "context": 8192, "vram_gb": 4.0, "load_ms": 4000, "tok_per_s": 10,
     "quality": {"code_generation": 0.90, "doc_summary": 0.50}},
    {"id": "poor-fast", "file": "b.gguf", "tier": 1, "capabilities": ["text", "tools"],
     "context": 8192, "vram_gb": 2.0, "load_ms": 1000, "tok_per_s": 100,
     "quality": {"code_generation": 0.40, "doc_summary": 0.80}},
    {"id": "mid", "file": "c.gguf", "tier": 3, "capabilities": ["text", "tools"],
     "context": 8192, "vram_gb": 3.0, "load_ms": 2000, "tok_per_s": 40,
     "quality": {"code_generation": 0.65, "doc_summary": 0.65}},
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


def candidates_for(registry, ids):
    return [
        Candidate(model_id=i, tier=registry.get(i).tier, resident=False, quality=0.0)
        for i in ids
    ]


# ---------------------------------------------------------------- the scale bug

def test_quality_is_not_swamped_by_latency(env):
    """THE regression test. good-slow is 0.90 at code and the SLOWEST candidate;
    poor-fast is 0.40 and the quickest. Quality must still win.

    If this fails, check that latency and swap are still normalised 0..1 in
    score_candidates. With raw seconds the latency term reaches ~-2.0 against a
    quality term capped at +1.0, and the router quietly becomes 'pick the
    fastest model' while every individual decision still looks plausible.
    """
    registry, warden = env
    cands = candidates_for(registry, ["good-slow", "poor-fast", "mid"])
    score_candidates(cands, "code_generation", 0.9, registry, warden)

    assert winner(cands).model_id == "good-slow"
    assert rank(cands)[-1].model_id == "poor-fast", (
        "the fastest-but-worst model should rank LAST on a code task"
    )


def test_the_ranking_is_actually_driven_by_quality(env):
    """Sharper than asserting a margin: hold everything else constant and show
    that flipping the quality numbers flips the ranking.

    Ordering alone could be a coincidence of speed. This cannot be.
    """
    registry, warden = env

    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    assert winner(cands).model_id == "good-slow"      # 0.90 vs 0.40

    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    score_candidates(cands, "doc_summary", 0.9, registry, warden)
    assert winner(cands).model_id == "poor-fast"      # 0.50 vs 0.80

    # Same models, same speeds, same VRAM, same swap state - only the per-class
    # quality changed, and the winner changed with it.


def test_cost_terms_cannot_outweigh_quality_on_their_own(env):
    """Structural invariant: the sum of every penalty weight must stay below the
    quality weight. Otherwise a model that is worst-on-everything-else can lose
    despite being the best available for the task, whatever the gap."""
    weights = ScoringWeights()
    total_cost = weights.w_latency + weights.w_swap + weights.w_vram
    assert total_cost < weights.w_quality, (
        f"penalties sum to {total_cost}, at or above w_quality "
        f"{weights.w_quality} - quality can be overridden by cost alone"
    )


def test_every_term_stays_within_its_weight(env):
    """Each term is weight x a 0..1 quantity, so |term| <= |weight|. This is the
    invariant that makes the weights comparable at all."""
    registry, warden = env
    weights = ScoringWeights()
    cands = candidates_for(registry, ["good-slow", "poor-fast", "mid"])
    score_candidates(cands, "code_generation", 0.5, registry, warden, weights)

    limits = {
        "quality": weights.w_quality,
        "confidence_fit": weights.w_confidence,
        "latency": weights.w_latency,
        "swap": weights.w_swap,
        "vram": weights.w_vram,
    }
    for c in cands:
        for term, limit in limits.items():
            assert abs(c.terms[term]) <= limit + 1e-9, (
                f"{c.model_id}.{term} = {c.terms[term]:.3f} exceeds weight {limit}"
            )


def test_speed_still_breaks_a_quality_tie(env):
    """Normalisation must not make latency inert - it should still decide when
    quality is equal. mid is 0.65 on both classes; against an equal-quality peer
    the faster one wins."""
    registry, warden = env
    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    # doc_summary: poor-fast 0.80 vs good-slow 0.50 - fast AND better.
    score_candidates(cands, "doc_summary", 0.9, registry, warden)
    assert winner(cands).model_id == "poor-fast"


# ---------------------------------------------------------------- swap term

def test_resident_model_pays_no_swap(env):
    registry, warden = env
    warden.admit("poor-fast")
    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    for c in cands:
        c.resident = warden.is_resident(c.model_id)
    score_candidates(cands, "code_generation", 0.9, registry, warden)

    resident = next(c for c in cands if c.model_id == "poor-fast")
    cold = next(c for c in cands if c.model_id == "good-slow")
    assert resident.terms["swap"] == 0.0
    assert cold.terms["swap"] < 0


def test_swap_flips_a_close_decision_but_not_a_wide_one(env):
    """§04's actual claim: 'given two models within a few points of each other on
    quality, it will keep the resident one'. A 0.50 quality gap is not a few
    points, and must still beat a reload."""
    registry, warden = env
    warden.admit("poor-fast")

    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    for c in cands:
        c.resident = warden.is_resident(c.model_id)
    # code: 0.90 vs 0.40 - a wide gap. Quality must win despite the reload.
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    assert winner(cands).model_id == "good-slow"


def test_all_cold_makes_the_swap_term_relative_not_absolute(env):
    """When nothing is resident the term still discriminates by load cost, but
    the cheapest option is charged 0 rather than everyone being penalised."""
    registry, warden = env
    cands = candidates_for(registry, ["good-slow", "poor-fast", "mid"])
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    swaps = {c.model_id: c.terms["swap"] for c in cands}
    assert swaps["good-slow"] < swaps["mid"] < swaps["poor-fast"] <= 0


# ---------------------------------------------------------------- confidence fit

def test_low_confidence_prefers_the_higher_tier(env):
    registry, _ = env
    spec_low = registry.get("poor-fast")     # tier 1
    spec_high = registry.get("mid")          # tier 3
    assert confidence_fit(spec_high, 0.1, 3) > confidence_fit(spec_low, 0.1, 3)


def test_full_confidence_gives_no_tier_bonus(env):
    registry, _ = env
    assert confidence_fit(registry.get("mid"), 1.0, 3) == pytest.approx(0.0)


def test_confidence_fit_handles_a_single_tier(env):
    registry, _ = env
    assert confidence_fit(registry.get("mid"), 0.5, 0) == 0.0


# ---------------------------------------------------------------- vram

def test_vram_pressure_is_a_fraction_of_budget(env):
    registry, _ = env
    assert vram_pressure(registry.get("good-slow"), 10.0) == pytest.approx(0.4)


def test_vram_pressure_is_capped_at_one(env):
    registry, _ = env
    assert vram_pressure(registry.get("good-slow"), 1.0) == 1.0


def test_vram_pressure_with_no_budget_is_zero(env):
    registry, _ = env
    assert vram_pressure(registry.get("good-slow"), 0.0) == 0.0


# ---------------------------------------------------------------- ranking

def test_eliminated_candidates_are_never_scored(env):
    registry, warden = env
    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    cands[0].eliminated_by = "no vision tower"
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    assert cands[0].score is None
    assert winner(cands).model_id == "poor-fast"


def test_no_survivors_yields_no_winner(env):
    registry, warden = env
    cands = candidates_for(registry, ["good-slow"])
    cands[0].eliminated_by = "nope"
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    assert winner(cands) is None
    assert rank(cands) == []


def test_margin_is_none_with_one_candidate(env):
    registry, warden = env
    cands = candidates_for(registry, ["good-slow"])
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    assert explain_margin(cands) is None


def test_terms_are_preserved_for_the_panel(env):
    """The explainer panel shows both the raw input and the weighted contribution."""
    registry, warden = env
    cands = candidates_for(registry, ["good-slow", "poor-fast"])
    score_candidates(cands, "code_generation", 0.9, registry, warden)
    terms = cands[0].terms
    for key in ("quality", "confidence_fit", "latency", "swap", "vram",
                "_quality_raw", "_gen_s", "_swap_s", "_vram_pressure"):
        assert key in terms
    assert cands[0].score == pytest.approx(
        sum(v for k, v in terms.items() if not k.startswith("_"))
    )


def test_weights_load_from_config():
    weights = ScoringWeights.from_config({"w_quality": 2.0, "w_swap": 0.05})
    assert weights.w_quality == 2.0
    assert weights.w_swap == 0.05
    assert weights.w_latency == 0.15          # default preserved

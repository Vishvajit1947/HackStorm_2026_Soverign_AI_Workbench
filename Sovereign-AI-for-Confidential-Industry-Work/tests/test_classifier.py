"""Stage 2 - the task classifier (§04).

Two things are being protected here:

  1. Different task types actually classify differently. That is the headline
     claim of the whole problem statement ("auto-selection across >= 2 task
     types"), and it is checked against the SHIPPED prototypes, not fixtures.

  2. A missing embedder degrades to keywords and SAYS SO. A silently-degraded
     classifier that still returns a confident-looking class is worse than one
     that fails, because nobody investigates a number that looks fine.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import Profile, get_settings
from kavach.router.classifier import TaskClassifier
from kavach.runtime.backends.mock import MockBackend
from kavach.runtime.registry import ModelRegistry
from kavach.runtime.warden import VramWarden
from kavach.types import TaskClass

MANIFEST = [
    {"id": "emb", "file": "emb.gguf", "role": "embedding", "tier": 0,
     "capabilities": ["embedding"], "context": 512, "vram_gb": 1.0,
     "load_ms": 200, "dim": 512},
    {"id": "text", "file": "text.gguf", "tier": 2, "capabilities": ["text", "tools"],
     "context": 8192, "vram_gb": 4.0, "load_ms": 2000},
]


def build(tmp_path, *, with_embedder=True, prototypes=None):
    config_dir, model_dir = tmp_path / "config", tmp_path / "models"
    config_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    manifest_entries = MANIFEST if with_embedder else MANIFEST[1:]
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(manifest_entries), encoding="utf-8")

    proto_path = config_dir / "prototypes.yaml"
    if prototypes is None:
        proto_path = get_settings().config_dir / "prototypes.yaml"
    else:
        proto_path.write_text(yaml.safe_dump(prototypes), encoding="utf-8")

    registry = ModelRegistry(manifest, model_dir, simulate=True)
    profile = Profile(
        name="TEST", description="", vram_gb=12.0, reserve_gb=2.0, cpu_only=False,
        context=8192, pinned=["emb"] if with_embedder else [],
        available=[m["id"] for m in manifest_entries], swap_slots=1,
    )
    backend = MockBackend(time_scale=0.0)
    warden = VramWarden(profile, registry, backend=backend)
    warden.pin_profile_models()

    return TaskClassifier(proto_path, registry, warden, backend)


@pytest.fixture()
def classifier(tmp_path):
    return build(tmp_path)


# ---------------------------------------------------------------- prototypes

def test_shipped_prototypes_load(classifier):
    assert classifier.prototype_count > 100


def test_every_task_class_has_examples(classifier):
    counts = classifier.counts_per_class()
    empty = [label for label, n in counts.items() if n == 0]
    assert not empty, f"no prototypes for {empty} - those classes can never be chosen"


def test_every_class_has_enough_examples_to_outvote_k(classifier):
    """With k=5 a class holding fewer than 5 examples can be outvoted by its
    neighbours even on a verbatim match. Keep a floor."""
    thin = {label: n for label, n in classifier.counts_per_class().items() if n < 5}
    assert not thin, f"too few prototypes for {thin}"


def test_unknown_label_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not task classes"):
        build(tmp_path, prototypes={"not_a_real_class": ["hello"]})


def test_reload_picks_up_new_examples(tmp_path):
    path = tmp_path / "config" / "prototypes.yaml"
    c = build(tmp_path, prototypes={"code_generation": ["write a script"] * 5})
    before = c.prototype_count
    path.write_text(
        yaml.safe_dump({"code_generation": ["write a script"] * 9}), encoding="utf-8"
    )
    assert c.load_prototypes() == 9
    assert c.prototype_count > before


# ---------------------------------------------------------------- classifying

@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Script the NPSH-available calculation and unit-test it", TaskClass.CODE_GENERATION),
        ("Summarise this 40-page SOP and list the safety interlocks", TaskClass.DOC_SUMMARY),
        ("Read this scanned inspection report", TaskClass.VISUAL_DOC),
        ("Calculate the NPSH available for P-101 at 65 degrees C", TaskClass.NUMERICAL_CALC),
        ("Draft an approval note to the GM Mechanical for this finding",
         TaskClass.OFFICIAL_DRAFTING),
        ("What does our SOP say about hot work near P-101?", TaskClass.KB_QA),
        ("Read this inspection report and draft the approval note as a Word file",
         TaskClass.AGENTIC_MULTI),
    ],
)
def test_verbatim_prototypes_classify_to_their_own_label(classifier, prompt, expected):
    result = classifier.classify(prompt)
    assert result.task_class is expected
    assert result.method == "embedding"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("can you write me a small python helper that converts flow units",
         TaskClass.CODE_GENERATION),
        ("what do our procedures require before entering a confined space",
         TaskClass.KB_QA),
        ("draft a note to the general manager asking for shutdown approval",
         TaskClass.OFFICIAL_DRAFTING),
        ("give me the key points from this turnaround report", TaskClass.DOC_SUMMARY),
    ],
)
def test_novel_phrasings_generalise(classifier, prompt, expected):
    """Not in prototypes.yaml. If these regress, add the failing prompt to
    prototypes.yaml under the right label - that is the intended fix, and it is
    the property that makes the kNN head worth choosing over a fine-tune."""
    assert classifier.classify(prompt).task_class is expected


def test_confidence_is_a_probability(classifier):
    result = classifier.classify("write a python script to parse the log")
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_neighbours_are_reported_for_the_panel(classifier):
    result = classifier.classify("write a python script")
    assert result.neighbours
    assert {"label", "similarity", "example"} <= set(result.neighbours[0])


def test_forced_class_bypasses_the_classifier(classifier):
    result = classifier.classify("total nonsense", forced=TaskClass.VISUAL_DOC)
    assert result.task_class is TaskClass.VISUAL_DOC
    assert result.confidence == 1.0
    assert result.method == "forced"


# ---------------------------------------------------------------- degradation

def test_no_embedder_falls_back_to_keywords(tmp_path):
    c = build(tmp_path, with_embedder=False)
    result = c.classify("write a python script and unit test it")
    assert result.method == "keyword"
    assert result.task_class is TaskClass.CODE_GENERATION


def test_the_fallback_announces_that_it_is_degraded(tmp_path):
    """A degraded classifier that looks confident is worse than one that fails."""
    c = build(tmp_path, with_embedder=False)
    result = c.classify("summarise this report")
    assert result.degraded is True
    assert "DEGRADED" in result.detail


def test_unmatched_prompt_defaults_to_the_planner_tier(tmp_path):
    """When nothing matches, hand it to the planner rather than guessing a
    specialist - the wrong specialist is worse than a generalist."""
    c = build(tmp_path, with_embedder=False)
    result = c.classify("zzzz qqqq xxxx")
    assert result.task_class is TaskClass.AGENTIC_MULTI
    assert result.confidence == 0.0
    assert result.degraded is True


def test_embedding_path_is_not_marked_degraded(classifier):
    assert classifier.classify("write a python script").degraded is False

"""The shipped quality numbers must still be able to pick a winner.

Runs against the SHIPPED config/models.yaml, like test_profiles.py, because the
failure this pins is a config failure rather than a code one.

A real measurement run wrote 1.00 into nearly every class of every model - the
eval set of the day was 1-3 easy prompts per class and every model passed them
all. Written into the manifest that made the quality term a tie, so scoring fell
through to latency and the router sent a summarisation job to the coder model:
the "ranks by speed while every individual decision still looks plausible"
failure the scoring notes warn about. Nothing failed. No test went red. The
router simply stopped being a router.

So this asserts the property the numbers exist to provide - that they separate
models - rather than asserting any particular value, which would just have to be
edited every time a measurement lands.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import get_settings

# The classes the MID profile's models are actually asked to compete on. visual_doc
# is excluded: an honest 0.00 for every text model is a tie by design, and that
# tie is what stage 1 uses to eliminate them.
CONTESTED = ["code_generation", "doc_summary", "numerical_calc",
             "official_drafting", "kb_qa", "agentic_multi"]


@pytest.fixture(scope="module")
def models():
    settings = get_settings()
    with (settings.config_dir / "models.yaml").open(encoding="utf-8") as fh:
        return {m["id"]: m for m in yaml.safe_load(fh)}


@pytest.fixture(scope="module")
def contenders(models):
    """The generation models MID actually routes between."""
    settings = get_settings()
    with (settings.config_dir / "profiles.yaml").open(encoding="utf-8") as fh:
        available = yaml.safe_load(fh)["MID"]["available"]
    return [models[m] for m in available
            if models[m].get("role") not in ("embedding", "reranker")]


def test_the_profile_has_something_to_choose_between(contenders):
    assert len(contenders) >= 2, "a router with one candidate is a lookup"


def test_quality_is_not_uniform_across_every_class(contenders):
    """A table that says every model is identical everywhere cannot pick a
    winner, so the router silently ranks on speed - the failure that made a
    saturated eval dangerous.

    Individual ties are FINE and often true: measured, these three 7B models
    really are equivalent at code generation and drafting. What must not happen
    is that NOTHING separates them anywhere.
    """
    discriminating = [
        cls for cls in CONTESTED
        if len({m["quality"].get(cls) for m in contenders}) > 1
    ]
    assert discriminating, (
        "every model scores identically on every contested class - quality "
        "cannot break any decision, so the router is ranking on speed alone"
    )


def test_a_class_that_discriminates_has_a_defensible_spread(contenders):
    """A 0.01 gap is noise dressed as a decision. Where the table does separate
    models, the gap should be big enough to act on."""
    for cls in CONTESTED:
        values = sorted({m["quality"].get(cls, 0.0) for m in contenders})
        if len(values) > 1:
            assert values[-1] - values[0] >= 0.05, (
                f"{cls} separates models by {values[-1] - values[0]:.3f} - "
                "too fine a margin to be evidence of anything"
            )


def test_quality_covers_every_contested_class(contenders):
    for m in contenders:
        missing = [c for c in CONTESTED if c not in m.get("quality", {})]
        assert not missing, f"{m['id']} has no quality for {missing}"


def test_a_measured_flag_names_its_classes(models):
    """quality_measured must never be a bare true: a run that measured 4 of 7
    classes and stamped `true` is claiming the 3 it never touched."""
    for m in models.values():
        flag = m.get("quality_measured")
        if flag is None:
            continue
        assert flag is not True, (
            f"{m['id']} claims quality_measured: true - it must list the "
            "classes that were actually measured"
        )
        assert isinstance(flag, list) and flag, flag

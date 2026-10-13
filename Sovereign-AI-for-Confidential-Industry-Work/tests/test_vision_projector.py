"""A vision model without its projector is a text model.

llama.cpp will happily load a VL model with no --mmproj, answer a question about
an attached image, and never have looked at it. For a P&ID that is the worst
thing this backend could return, so the projector - not the `vision` capability
flag - is what decides whether images may be sent at all.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.config import get_settings
from kavach.runtime.registry import ModelSpec


def _spec(**kw) -> ModelSpec:
    base = {"id": "vl", "file": "vl.gguf", "license": "apache-2.0", "tier": 2,
                "capabilities": ["text", "vision"], "context": 8192, "vram_gb": 6.1,
                "load_ms": 5100}
    base.update(kw)
    return ModelSpec(**base)


def test_no_mmproj_declared_means_no_projector():
    assert _spec().has_projector is False


def test_a_declared_but_absent_projector_is_still_no_projector():
    """The manifest naming a file proves nothing; the file has to be here."""
    assert _spec(mmproj="not-on-this-box.gguf").has_projector is False


def test_vision_capability_alone_does_not_imply_a_projector():
    spec = _spec(mmproj="missing.gguf")
    assert spec.has_vision is True
    assert spec.has_projector is False, \
        "has_vision must not be read as 'can actually see'"


def test_the_shipped_vl_model_declares_its_projector():
    """If this fails, the VL model would be picked for scans and then refuse."""
    with (get_settings().config_dir / "models.yaml").open(encoding="utf-8") as fh:
        models = {m["id"]: m for m in yaml.safe_load(fh)}
    vl = models["qwen3-vl-8b-q4"]
    assert vl.get("mmproj"), "the vision model has no mmproj declared"
    assert "mmproj" in vl["mmproj"].lower()


@pytest.mark.parametrize("declared", [None, "", "absent.gguf"])
def test_the_ingest_fork_will_not_choose_a_blind_vision_model(declared):
    """extract.py filters on has_projector. This pins the property it relies on
    so a refactor cannot quietly turn it back into has_vision."""
    assert _spec(mmproj=declared).has_projector is False

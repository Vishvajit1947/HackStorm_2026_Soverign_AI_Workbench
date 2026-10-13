"""Model registry + provenance (§02.4, §03).

The claim under test is "how do you know the weights on this box are the weights
you approved?" - so most of these stage a fake weight file and then corrupt it.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.runtime.registry import (
    DigestMismatch,
    ModelRegistry,
    Provenance,
    sha256_file,
)

MANIFEST = [
    {
        "id": "tiny-text",
        "file": "tiny-text.gguf",
        "license": "apache-2.0",
        "tier": 1,
        "capabilities": ["text", "tools"],
        "context": 8192,
        "vram_gb": 2.0,
        "load_ms": 1000,
        "quality": {"code_generation": 0.5, "doc_summary": 0.6},
    },
    {
        "id": "tiny-emb",
        "file": "tiny-emb.gguf",
        "license": "mit",
        "tier": 0,
        "role": "embedding",
        "capabilities": ["embedding"],
        "context": 512,
        "vram_gb": 0.2,
        "load_ms": 200,
        "dim": 64,
    },
]


@pytest.fixture()
def workspace(tmp_path):
    config_dir = tmp_path / "config"
    model_dir = tmp_path / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")
    return manifest, model_dir


def stage(model_dir, name: str, content: bytes = b"fake gguf weights") -> str:
    path = model_dir / name
    path.write_bytes(content)
    return sha256_file(path)


def approve(manifest, model_id: str, digest: str | None) -> None:
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    for entry in raw:
        if entry["id"] == model_id:
            entry["sha256"] = digest
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")


# ---------------------------------------------------------------- provenance

def test_absent_weights_are_unverified(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir)
    assert registry.get("tiny-text").provenance is Provenance.UNVERIFIED


def test_absent_weights_are_simulated_on_mock_backend(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    assert registry.get("tiny-text").provenance is Provenance.SIMULATED


def test_staged_but_unapproved_records_observed_digest(workspace):
    """We hash it anyway so an operator can paste the value in deliberately."""
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf")
    registry = ModelRegistry(manifest, model_dir)
    spec = registry.get("tiny-text")
    assert spec.provenance is Provenance.UNVERIFIED
    assert spec.observed_sha256 == digest


def test_approved_and_matching_is_verified(workspace):
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf")
    approve(manifest, "tiny-text", digest)
    registry = ModelRegistry(manifest, model_dir)
    assert registry.get("tiny-text").provenance is Provenance.VERIFIED


def test_a_staged_file_beats_the_simulate_flag(workspace):
    """simulate=True must never mask a real file that is present."""
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf")
    approve(manifest, "tiny-text", digest)
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    assert registry.get("tiny-text").provenance is Provenance.VERIFIED


# ---------------------------------------------------------------- the halt

def test_substituted_weights_halt_startup(workspace):
    """The auditor's question, answered: a swapped file stops the box booting."""
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf", b"the approved weights")
    approve(manifest, "tiny-text", digest)

    # Someone replaces the file after approval.
    stage(model_dir, "tiny-text.gguf", b"different weights entirely")

    with pytest.raises(DigestMismatch) as exc:
        ModelRegistry(manifest, model_dir, enforce_digests=True)
    assert "tiny-text" in str(exc.value)
    assert "tiny-text.gguf" in str(exc.value)


def test_mismatch_names_the_file(workspace):
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf", b"approved")
    approve(manifest, "tiny-text", digest)
    stage(model_dir, "tiny-text.gguf", b"substituted")

    with pytest.raises(DigestMismatch) as exc:
        ModelRegistry(manifest, model_dir, enforce_digests=True)
    message = str(exc.value)
    assert "hashes to" in message and "approved" in message


def test_enforcement_can_be_disabled_for_dev(workspace):
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf", b"approved")
    approve(manifest, "tiny-text", digest)
    stage(model_dir, "tiny-text.gguf", b"substituted")

    registry = ModelRegistry(manifest, model_dir, enforce_digests=False)
    assert registry.get("tiny-text").provenance is Provenance.MISMATCH


def test_mismatched_model_is_never_routable(workspace):
    manifest, model_dir = workspace
    digest = stage(model_dir, "tiny-text.gguf", b"approved")
    approve(manifest, "tiny-text", digest)
    stage(model_dir, "tiny-text.gguf", b"substituted")

    registry = ModelRegistry(manifest, model_dir, enforce_digests=False)
    assert registry.get("tiny-text").routable is False
    assert registry.routable() == []


# ---------------------------------------------------------------- routing view

def test_embedders_are_not_routable(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    assert registry.get("tiny-emb").routable is False
    assert [s.id for s in registry.routable()] == ["tiny-text"]


def test_embedder_lookup(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    assert registry.embedder().id == "tiny-emb"


def test_quality_lookup_defaults_to_zero(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    spec = registry.get("tiny-text")
    assert spec.quality_for("code_generation") == 0.5
    assert spec.quality_for("visual_doc") == 0.0


def test_vision_capability_detection(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    assert registry.get("tiny-text").has_vision is False
    assert registry.get("tiny-text").has_tools is True


# ---------------------------------------------------------------- hot reload

def test_reload_picks_up_a_new_model(workspace):
    """The extensibility beat (§09, 10:45): paste an entry, reload, it is a candidate."""
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    assert registry.get("newcomer") is None

    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw.append({
        "id": "newcomer",
        "file": "newcomer.gguf",
        "license": "apache-2.0",
        "tier": 2,
        "capabilities": ["text", "tools"],
        "context": 16384,
        "vram_gb": 3.0,
        "load_ms": 2000,
        "quality": {"code_generation": 0.95},
    })
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    registry.reload()
    assert registry.get("newcomer") is not None
    assert registry.get("newcomer").routable is True
    assert registry.get("newcomer").quality_for("code_generation") == 0.95


def test_reload_drops_a_removed_model(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    raw = [e for e in yaml.safe_load(manifest.read_text(encoding="utf-8"))
           if e["id"] != "tiny-text"]
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    registry.reload()
    assert registry.get("tiny-text") is None


def test_summary_warns_about_simulated_models(workspace):
    manifest, model_dir = workspace
    registry = ModelRegistry(manifest, model_dir, simulate=True)
    summary = registry.summary()
    assert summary["all_weights_real"] is False
    assert "SIMULATED" in summary["warning"]


def test_file_present_is_reported_separately_from_provenance(tmp_path, monkeypatch):
    """"Is the weight file here" is not a question provenance answers.

    A missing file is SIMULATED on the mock backend and UNVERIFIED on
    llamacpp - and UNVERIFIED is ALSO what a file that IS present but has no
    recorded digest reports. Anything deciding "can this model run" that reads
    the enum gets a different answer when the backend changes, which is how a
    filter meant to hide seven absent models showed all fourteen.
    """
    from kavach.runtime.registry import ModelRegistry

    models = tmp_path / "models"
    models.mkdir()
    (models / "here.gguf").write_bytes(b"GGUF" + b"\0" * 64)

    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "- id: here\n"
        "  file: here.gguf\n"
        "  license: apache-2.0\n"
        "  tier: 1\n"
        "  capabilities: [text]\n"
        "  context: 2048\n"
        "  vram_gb: 0.1\n"
        "  load_ms: 100\n"
        "- id: absent\n"
        "  file: absent.gguf\n"
        "  license: apache-2.0\n"
        "  tier: 1\n"
        "  capabilities: [text]\n"
        "  context: 2048\n"
        "  vram_gb: 0.1\n"
        "  load_ms: 100\n",
        encoding="utf-8",
    )

    registry = ModelRegistry(manifest, models, enforce_digests=False)
    registry.reload()
    by_id = {m["id"]: m for m in registry.summary()["models"]}

    assert by_id["here"]["file_present"] is True
    assert by_id["absent"]["file_present"] is False

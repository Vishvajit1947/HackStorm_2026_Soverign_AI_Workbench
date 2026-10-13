"""eval_harness --write must never make models.yaml say less than it did.

The harness protects one door: it refuses to measure the mock backend, because
stamping quality_measured on synthetic text would launder a guess into a
measurement. It did not protect the other. eval_tasks.yaml carries prompts for
some task classes and not others, and a partial run rewrote each measured
model's whole quality block from its results - silently deleting every class it
had not measured, then stamping `quality_measured: true` anyway.

Losing visual_doc is not cosmetic: an explicit 0.00 is the only reason stage 1
eliminates a text model for a scan. Losing agentic_multi removes the class the
planner routes on. And the README tells an operator to run exactly this command
before a demo.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest
import yaml

REPO_SCRIPT = "scripts/eval_harness.py"


@pytest.fixture(scope="module")
def eh():
    spec = importlib.util.spec_from_file_location("eval_harness", REPO_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def manifest(tmp_path):
    dst = tmp_path / "models.yaml"
    shutil.copy("config/models.yaml", dst)
    return dst


PARTIAL = {"code_generation": 0.71, "doc_summary": 0.88}


def _entry(path, model_id):
    return next(m for m in yaml.safe_load(path.read_text(encoding="utf-8"))
                if m["id"] == model_id)


def test_a_partial_run_keeps_the_classes_it_did_not_measure(eh, manifest):
    before = _entry(manifest, "qwen3-8b-q4")["quality"]
    eh._write_quality(manifest, {"qwen3-8b-q4": dict(PARTIAL)})
    after = _entry(manifest, "qwen3-8b-q4")["quality"]
    assert set(after) == set(before), "a class was dropped by a partial write"
    assert after["visual_doc"] == before["visual_doc"]
    assert after["agentic_multi"] == before["agentic_multi"]


def test_measured_values_are_actually_written(eh, manifest):
    eh._write_quality(manifest, {"qwen3-8b-q4": dict(PARTIAL)})
    after = _entry(manifest, "qwen3-8b-q4")["quality"]
    assert after["code_generation"] == 0.71
    assert after["doc_summary"] == 0.88


def test_the_flag_names_the_classes_rather_than_claiming_all(eh, manifest):
    eh._write_quality(manifest, {"qwen3-8b-q4": dict(PARTIAL)})
    flag = _entry(manifest, "qwen3-8b-q4").get("quality_measured")
    assert flag is not True, "a bare true claims classes that were never measured"
    assert sorted(flag) == sorted(PARTIAL), flag


def test_carried_classes_are_marked_seeded_in_the_file(eh, manifest):
    eh._write_quality(manifest, {"qwen3-8b-q4": dict(PARTIAL)})
    assert "# SEEDED - not measured" in manifest.read_text(encoding="utf-8")


def test_models_that_were_not_measured_are_untouched(eh, manifest):
    """Byte-for-byte identical, whatever it already said. Asserting the flag was
    ABSENT only held until a real measurement run put one there - the property
    that matters is that this write did not change it."""
    before = _entry(manifest, "qwen2_5-coder-7b-q4")
    eh._write_quality(manifest, {"qwen3-8b-q4": dict(PARTIAL)})
    after = _entry(manifest, "qwen2_5-coder-7b-q4")
    assert after["quality"] == before["quality"]
    assert after.get("quality_measured") == before.get("quality_measured")


def test_the_seed_warning_comment_survives(eh, manifest):
    eh._write_quality(manifest, {"qwen3-8b-q4": dict(PARTIAL)})
    text = manifest.read_text(encoding="utf-8")
    assert "REPLACE THEM with measured values" in text, \
        "a yaml round-trip destroyed the comments"

"""The evaluation harness (§04).

Two things are pinned here:

  1. The held-out set stays held out. If an eval prompt is copied verbatim into
     prototypes.yaml, routing accuracy stops measuring generalisation and starts
     measuring memorisation - and the number silently becomes worthless while
     looking better than ever.

  2. Routing accuracy does not regress. The harness found the prototype set at
     78.6% with agentic_multi swallowing eight of nine misroutes; that is the
     kind of thing that is invisible without a held-out set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from kavach.config import get_settings
from kavach.router import get_router
from kavach.types import ALL_TASK_CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def tasks():
    path = get_settings().config_dir / "eval_tasks.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.fixture(scope="module")
def prototypes():
    path = get_settings().config_dir / "prototypes.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------- held out

def test_no_eval_prompt_appears_in_the_prototypes(tasks, prototypes):
    """The whole point of a held-out set. Copying a misroute across to fix it is
    the intended workflow - but you must DELETE it from eval_tasks.yaml when you
    do, or the next run measures memorisation."""
    proto_texts = {
        str(text).strip().lower()
        for examples in prototypes.values()
        for text in (examples or [])
    }
    leaked = [
        prompt
        for prompts in tasks.get("routing", {}).values()
        for prompt in (prompts or [])
        if str(prompt).strip().lower() in proto_texts
    ]
    assert not leaked, (
        "these eval prompts are also prototypes, so accuracy is inflated: " f"{leaked}"
    )


def test_every_class_has_held_out_cases(tasks):
    routing = tasks.get("routing", {})
    missing = [c for c in ALL_TASK_CLASSES if not routing.get(c)]
    assert not missing, f"no held-out routing cases for {missing}"


def test_eval_labels_are_real_task_classes(tasks):
    unknown = set(tasks.get("routing", {})) - set(ALL_TASK_CLASSES)
    assert not unknown, f"eval_tasks.yaml has unknown labels: {unknown}"


# ---------------------------------------------------------------- accuracy

def test_routing_accuracy_does_not_regress(tasks):
    """Measured with the MOCK embedder, so this is a floor on the prototype set's
    quality rather than the production number. It still catches the failure mode
    that matters: one class quietly becoming an attractor for everything else."""
    classifier = get_router().classifier
    total = correct = 0
    for expected, prompts in tasks.get("routing", {}).items():
        for prompt in prompts or []:
            total += 1
            if classifier.classify(prompt).task_class.value == expected:
                correct += 1

    accuracy = correct / total
    assert accuracy >= 0.88, (
        f"routing accuracy fell to {accuracy:.1%} ({correct}/{total}). "
        "Run: python scripts/eval_harness.py --routing --confusion"
    )


def test_no_class_is_an_attractor(tasks):
    """The specific defect the harness found: agentic_multi absorbing prompts
    from every other class. Any class collecting a lot of foreign traffic is
    over-matching and its prototypes need tightening."""
    classifier = get_router().classifier
    wrong_into: dict[str, int] = dict.fromkeys(ALL_TASK_CLASSES, 0)
    total = 0

    for expected, prompts in tasks.get("routing", {}).items():
        for prompt in prompts or []:
            total += 1
            got = classifier.classify(prompt).task_class.value
            if got != expected:
                wrong_into[got] += 1

    worst = max(wrong_into, key=lambda c: wrong_into[c])
    assert wrong_into[worst] <= max(2, total * 0.08), (
        f"{worst} is absorbing {wrong_into[worst]} of {total} prompts from other "
        "classes - its prototypes are matching too loosely"
    )


# ---------------------------------------------------------------- checks

def test_quality_checks_all_resolve(tasks):
    from eval_harness import CHECKS

    for task_class, items in tasks.get("quality", {}).items():
        for item in items or []:
            assert item["check"] in CHECKS, (
                f"{task_class} references unknown check {item['check']!r}"
            )


def test_python_check_accepts_correct_code():
    from eval_harness import python_runs_assertion

    code = "def c_to_f(c):\n    return c * 9 / 5 + 32\n"
    assert python_runs_assertion(code, "abs(c_to_f(100) - 212) < 0.001") == 1.0


def test_python_check_rejects_wrong_code():
    from eval_harness import python_runs_assertion

    code = "def c_to_f(c):\n    return c\n"
    assert python_runs_assertion(code, "abs(c_to_f(100) - 212) < 0.001") == 0.0


def test_python_check_rejects_prose():
    from eval_harness import python_runs_assertion

    assert python_runs_assertion("I would multiply by nine fifths.", "True") == 0.0


def test_number_check_respects_tolerance():
    from eval_harness import contains_number_near

    assert contains_number_near("The velocity is 0.61 m/s", 0.61, 0.1) == 1.0
    assert contains_number_near("The velocity is 4.2 m/s", 0.61, 0.1) == 0.0


def test_uncited_answer_is_penalised():
    from eval_harness import has_citation_and_mentions

    cited = has_citation_and_mentions("A gas test is needed [SOP-114 p.12]", ["gas test"])
    uncited = has_citation_and_mentions("A gas test is needed", ["gas test"])
    assert cited == 1.0
    assert uncited < cited


# ---------------------------------------------------------------- the gate

def test_quality_mode_refuses_the_mock_backend(tasks, capsys):
    """The gate that stops a stub's output being stamped `quality_measured: true`
    and shown to judges as an evaluation table."""
    from eval_harness import run_quality

    code = run_quality(tasks, write=True)
    assert code == 2
    assert "REFUSING TO RUN" in capsys.readouterr().out


def test_models_yaml_is_untouched_after_a_refused_run(tasks):
    path = get_settings().config_dir / "models.yaml"
    before = path.read_text(encoding="utf-8")
    from eval_harness import run_quality

    run_quality(tasks, write=True)
    assert path.read_text(encoding="utf-8") == before

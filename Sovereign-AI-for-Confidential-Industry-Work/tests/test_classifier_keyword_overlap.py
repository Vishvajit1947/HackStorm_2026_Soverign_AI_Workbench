"""Overlapping keyword hints must not distort the degraded classifier.

The keyword fallback is a bag of hand-written terms, and hand-written lists
overlap in two ways that both change routing:

  WITHIN a class   "unit test" and "test" are one piece of evidence written
                   twice. Scored separately they banked 3 points for a phrase a
                   human wrote once, and a rival class holding one genuine hint
                   lost 3:1 instead of 2:1.

  ACROSS classes   the same term in two lists splits a vote it should carry
                   whole. There are none today; this file is what tells the next
                   person who adds one.

And a bag of keywords ties constantly - two classes holding one hint each is an
ordinary outcome. `max()` used to break those by dict insertion order and report
full confidence in a coin flip, which is exactly the kind of number nobody can
defend.
"""

from __future__ import annotations

import pytest

from kavach.router.classifier import ALL_TASK_CLASSES, KEYWORD_HINTS, TaskClassifier
from kavach.types import TaskClass


@pytest.fixture
def classify():
    """The fallback in isolation - it touches no embedder and no registry."""
    bare = TaskClassifier.__new__(TaskClassifier)
    return lambda prompt: bare._classify_by_keyword(prompt, "no embedder")


# ---------------------------------------------------------------- the lists

def test_no_hint_belongs_to_two_classes():
    """A duplicated hint splits the evidence it was written to carry: both
    classes gain a point, their difference is unchanged, and the term does no
    work at all. If a term genuinely marks two classes it needs a phrase that
    distinguishes them, not the same word in both lists."""
    owners: dict[str, list[str]] = {}
    for label, hints in KEYWORD_HINTS.items():
        for hint in hints:
            owners.setdefault(hint, []).append(label)

    duplicated = {h: labels for h, labels in owners.items() if len(labels) > 1}
    assert duplicated == {}, f"hints claimed by more than one class: {duplicated}"


def test_every_class_has_hints():
    """A class with no hints can never win the fallback, so a prompt it owns is
    routed somewhere else with no indication anything was missing."""
    assert set(KEYWORD_HINTS) == set(ALL_TASK_CLASSES)
    assert all(KEYWORD_HINTS[label] for label in ALL_TASK_CLASSES)


# ---------------------------------------------------------------- overlap

def test_a_phrase_and_the_token_inside_it_count_once(classify):
    """'unit test' matches the phrase AND the bare token 'test'. Scored twice,
    code_generation takes 3 of 4 points here instead of the 2 of 3 the evidence
    supports."""
    result = classify("summarise the unit test results")
    assert result.scores["code_generation"] == pytest.approx(2 / 3)
    assert result.scores["doc_summary"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("prompt,label", [
    ("write a unit test for the parser", "code_generation"),
    ("give me an executive summary", "doc_summary"),
    ("prepare a covering letter", "official_drafting"),
])
def test_an_overlapping_pair_still_wins_its_own_prompt(classify, prompt, label):
    """Collapsing the double count must not collapse the signal with it."""
    assert classify(prompt).task_class is TaskClass(label)


def test_scores_are_a_distribution(classify):
    result = classify("calculate the pressure drop and draft a note")
    assert sum(result.scores.values()) == pytest.approx(1.0)
    assert result.confidence == result.scores[result.task_class.value]


# ---------------------------------------------------------------- ties

def test_a_tie_is_named_rather_than_silently_broken(classify):
    """'debug' and 'memo' are one hint each. The winner is a coin flip, so the
    verdict has to say so - a 0.50 confidence with no explanation reads as a
    decision that was made on evidence."""
    result = classify("debug the memo")
    assert result.confidence == pytest.approx(0.5)
    assert "TIED" in result.detail
    assert "code_generation" in result.detail
    assert "official_drafting" in result.detail


def test_a_tie_with_the_planner_tier_goes_to_the_planner(classify):
    """A multi-step marker tying with a specialist means the prompt contains
    BOTH halves. The planner can run the specialist's work as one of its steps;
    the specialist cannot recover the half it was never told about."""
    result = classify("refactor the parser and prepare a circular")
    assert result.task_class is TaskClass.AGENTIC_MULTI
    assert "TIED" in result.detail


def test_an_uncontested_winner_is_not_reported_as_tied(classify):
    result = classify("write a python function to parse the log")
    assert result.task_class is TaskClass.CODE_GENERATION
    assert "TIED" not in result.detail


def test_the_tie_break_is_stable_across_calls(classify):
    """An arbitrary winner is tolerable; an unpredictable one is not - the same
    prompt routing two different ways is unreproducible in a demo."""
    verdicts = {classify("debug the memo").task_class for _ in range(8)}
    assert len(verdicts) == 1


# ---------------------------------------------------------------- unchanged

def test_nothing_matched_still_defaults_to_the_planner_at_zero_confidence(classify):
    result = classify("hello there")
    assert result.task_class is TaskClass.AGENTIC_MULTI
    assert result.confidence == 0.0
    assert "no keywords matched" in result.detail


def test_the_fallback_always_admits_it_is_degraded(classify):
    for prompt in ("debug the memo", "hello there", "calculate npsh"):
        result = classify(prompt)
        assert result.degraded is True
        assert result.method == "keyword"

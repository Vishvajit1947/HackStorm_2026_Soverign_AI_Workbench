"""Stage 5 - the output verifier (§04).

The verifier is what turns a router into a system that recovers. Two properties
matter more than the individual checks:

  - a failed check escalates ONCE, never loops. An unbounded ladder on a 12 GB
    card means swapping through every model you own while a judge watches.
  - when there is nothing better to escalate to, it returns the failure rather
    than quietly presenting a bad answer as a good one.
"""

from __future__ import annotations

import pytest

from kavach.router.verifier import (
    check_has_citation,
    check_non_trivial,
    check_not_truncated,
    check_parseable_code,
    next_tier_up,
    verify_output,
)
from kavach.types import TaskClass

# ---------------------------------------------------------------- code

def test_valid_python_passes():
    ok, _ = check_parseable_code("def f(x):\n    return x * 2\n")
    assert ok


def test_syntax_error_fails_with_a_line_number():
    ok, reason = check_parseable_code("def f(x:\n  return x")
    assert not ok
    assert "line" in reason


def test_fenced_code_is_extracted():
    ok, _ = check_parseable_code(
        "Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps."
    )
    assert ok


def test_prose_masquerading_as_code_fails():
    """The exact failure the escalation exists for: plausible-looking rubbish
    from a small model."""
    ok, reason = check_parseable_code("Sure! I would write a function that loops.")
    assert not ok
    assert "does not parse" in reason


def test_empty_response_fails():
    ok, reason = check_parseable_code("   ")
    assert not ok
    assert "no code" in reason


# ---------------------------------------------------------------- citations

@pytest.mark.parametrize("text", [
    "Hot work needs a permit [SOP-114 p. 12].",
    "The interlock trips at 8 bar [doc:ops-manual, page 44]",
    "See the isolation steps [Procedure 7 §3.2].",
])
def test_citation_shapes_are_recognised(text):
    ok, _ = check_has_citation(text)
    assert ok


def test_uncited_claim_fails():
    ok, reason = check_has_citation("Hot work needs a permit near P-101.")
    assert not ok
    assert "citation" in reason


# ---------------------------------------------------------------- basics

def test_short_output_fails():
    ok, _ = check_non_trivial("ok", 12)
    assert not ok


def test_truncated_generation_fails():
    ok, reason = check_not_truncated("length")
    assert not ok
    assert "token limit" in reason


def test_normal_finish_passes():
    ok, _ = check_not_truncated("stop")
    assert ok


# ---------------------------------------------------------------- composition

def test_code_class_requires_parseable_code():
    result = verify_output(
        "I would write a loop that does the thing you asked for.",
        TaskClass.CODE_GENERATION,
        require_parseable_code=["code_generation"],
    )
    assert not result.ok
    assert result.checks["parseable_code"] is False


def test_summary_class_does_not_require_parseable_code():
    result = verify_output(
        "The SOP lists four safety interlocks covering pressure and level.",
        TaskClass.DOC_SUMMARY,
        require_parseable_code=["code_generation"],
    )
    assert result.ok


def test_citation_is_only_required_when_grounded():
    """Demanding a citation from a model that was given nothing to cite would
    escalate every general question forever."""
    ungrounded = verify_output(
        "Generally a permit is needed for hot work.",
        TaskClass.KB_QA,
        require_citations=["kb_qa"],
        kb_grounded=False,
    )
    assert ungrounded.ok

    grounded = verify_output(
        "Generally a permit is needed for hot work.",
        TaskClass.KB_QA,
        require_citations=["kb_qa"],
        kb_grounded=True,
    )
    assert not grounded.ok


def test_disabled_verifier_skips_everything():
    result = verify_output("x", TaskClass.CODE_GENERATION, enabled=False)
    assert result.ok
    assert result.skipped


def test_multiple_failures_are_all_reported():
    result = verify_output(
        "no",
        TaskClass.CODE_GENERATION,
        finish_reason="length",
        require_parseable_code=["code_generation"],
    )
    assert not result.ok
    assert result.reason.count(";") >= 1


# ---------------------------------------------------------------- escalation

def test_escalates_to_the_next_tier_up():
    assert next_tier_up(1, [0, 1, 2, 3]) == 2


def test_escalation_skips_missing_tiers():
    assert next_tier_up(1, [0, 1, 3]) == 3


def test_no_escalation_from_the_top():
    """Returning None is what makes the router say 'already the most capable
    model available' instead of pretending the answer passed."""
    assert next_tier_up(3, [0, 1, 2, 3]) is None


def test_no_escalation_when_alone():
    assert next_tier_up(2, [2]) is None

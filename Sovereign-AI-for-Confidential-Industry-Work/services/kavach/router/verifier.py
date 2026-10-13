"""Stage 5 - the output verifier (§04).

This is what turns a router into a system that recovers: a failed check escalates
the SAME prompt to the next tier rather than returning broken code.

Escalation is bounded at one retry. Not because one is magic, but because an
unbounded escalation ladder on a 12 GB card means swapping through every model you
own while a judge watches - and the second attempt is the one that usually helps.
Past that you are burning the clock, not improving the answer.

The checks are deliberately shallow and mechanical. A deep semantic critic belongs
in the agent loop (Phase 5), where it has the step's acceptance test to work with.
Here we only catch output that is obviously unusable.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from ..types import TaskClass

# Fenced code blocks, so a model wrapping python in markdown still parses.
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# An inline citation of the shapes the knowledge plane emits (Phase 6).
_CITATION = re.compile(
    r"\[(?:[^\]]*?(?:p\.?\s*\d+|page\s*\d+|§\s*[\d.]+|doc:[^\]]+))[^\]]*\]",
    re.IGNORECASE,
)


@dataclass
class VerifierResult:
    ok: bool
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "checks": self.checks,
            "skipped": self.skipped,
        }


def _extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    return "\n\n".join(blocks) if blocks else text


def check_parseable_code(text: str) -> tuple[bool, str]:
    """Does the output actually parse as Python?

    A syntax error is the cheapest possible signal that a small model produced
    plausible-looking rubbish, and it is exactly the case the escalation exists
    for. Note this does NOT execute anything - that is code.exec's job, in a
    sandbox with no network (Phase 4).
    """
    code = _extract_code(text).strip()
    if not code:
        return False, "no code in the response"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"code does not parse: line {exc.lineno}, {exc.msg}"
    return True, ""


def check_has_citation(text: str) -> tuple[bool, str]:
    """In a plant context an uncited claim is worse than no answer (§07)."""
    if _CITATION.search(text):
        return True, ""
    return False, "no inline citation in a knowledge-grounded answer"


def check_non_trivial(text: str, min_chars: int) -> tuple[bool, str]:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False, f"output is {len(stripped)} chars, under the {min_chars} minimum"
    return True, ""


def check_not_truncated(finish_reason: str) -> tuple[bool, str]:
    if finish_reason == "length":
        return False, "generation hit the token limit mid-answer"
    return True, ""


def verify_output(
    text: str,
    task_class: TaskClass,
    *,
    finish_reason: str = "stop",
    min_output_chars: int = 12,
    require_citations: tuple[str, ...] | list[str] = (),
    require_parseable_code: tuple[str, ...] | list[str] = (),
    enabled: bool = True,
    kb_grounded: bool = False,
) -> VerifierResult:
    """Run the checks that apply to this task class."""
    if not enabled:
        return VerifierResult(ok=True, skipped=True, reason="verifier disabled")

    checks: dict[str, bool] = {}
    failures: list[str] = []

    ok, reason = check_non_trivial(text, min_output_chars)
    checks["non_trivial"] = ok
    if not ok:
        failures.append(reason)

    ok, reason = check_not_truncated(finish_reason)
    checks["not_truncated"] = ok
    if not ok:
        failures.append(reason)

    if task_class.value in require_parseable_code:
        ok, reason = check_parseable_code(text)
        checks["parseable_code"] = ok
        if not ok:
            failures.append(reason)

    # Only demand a citation when the answer was actually grounded in retrieved
    # documents. Requiring one from a model that was given nothing to cite would
    # escalate every general question forever.
    if task_class.value in require_citations and kb_grounded:
        ok, reason = check_has_citation(text)
        checks["has_citation"] = ok
        if not ok:
            failures.append(reason)

    if failures:
        return VerifierResult(ok=False, reason="; ".join(failures), checks=checks)
    return VerifierResult(ok=True, reason="all checks passed", checks=checks)


def next_tier_up(current_tier: int, available_tiers: list[int]) -> int | None:
    """The escalation target: the cheapest tier strictly above the current one.

    Returns None when the model that just failed is already the most capable
    thing on this box - at which point escalating is not an option and the
    honest move is to return the failure with its reason.
    """
    higher = sorted(t for t in available_tiers if t > current_tier)
    return higher[0] if higher else None

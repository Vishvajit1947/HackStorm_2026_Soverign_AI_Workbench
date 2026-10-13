"""calc accepts the form an engineer actually writes.

A model asked to compute something writes "delta_P = (F / A) * (1 / rho)" -
the quantity, then the formula. ast.parse in eval mode rejects an assignment
outright, so every such step failed with "invalid syntax", burned a replan and
took the run down with it, having been handed a perfectly correct formula.

The label carries no information the tool needs; the right-hand side is the sum.
These tests pin that the convenience did not open the door the AST allowlist
exists to keep shut - calc is COMPUTE, so it has no sandbox behind it.
"""

from __future__ import annotations

import pytest

from kavach.tools.impl.calc import CalcError, UnsafeExpression, calc, strip_assignment


def test_the_expression_that_kept_failing_now_evaluates():
    result = calc("delta_P = (F / A) * (1 / rho)", substitutions="F=1000,A=0.01,rho=998")
    assert result["result"] == pytest.approx(100.2, abs=0.1)


def test_the_derivation_still_shows_what_was_dropped():
    """Silently discarding half the input would make the working unreadable."""
    steps = " ".join(calc("delta_P = 2 + 2")["steps"])
    assert "delta_P" in steps, steps


def test_a_plain_expression_is_untouched():
    assert calc("2 + 2")["result"] == pytest.approx(4.0)
    assert strip_assignment("2 + 2") == ("2 + 2", None)


def test_a_comparison_is_not_mistaken_for_an_assignment():
    """`==` is a comparison. Stripping it would silently change the meaning."""
    assert strip_assignment("a == b") == ("a == b", None)


@pytest.mark.parametrize("attack", [
    "x = __import__('os').system('echo pwned')",
    "y = open('/etc/passwd').read()",
    "z = ().__class__.__bases__[0].__subclasses__()",
])
def test_an_assignment_is_not_a_way_past_the_allowlist(attack):
    """calc has no sandbox, so the gate is the whole defence. Stripping the
    label must not let anything through it that was refused before."""
    with pytest.raises((UnsafeExpression, CalcError)):
        calc(attack)


def test_a_chained_assignment_is_refused_rather_than_half_read():
    with pytest.raises(CalcError):
        calc("a = b = 1 + 1")


def test_the_stripped_name_is_reported():
    expression, name = strip_assignment("delta_P = (F / A)")
    assert expression == "(F / A)"
    assert name == "delta_P"

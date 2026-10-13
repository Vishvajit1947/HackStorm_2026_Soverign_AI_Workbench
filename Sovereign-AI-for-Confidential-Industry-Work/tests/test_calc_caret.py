"""calc reads "^" the way an engineer wrote it.

A model asked for a pressure drop writes rho*v^2/2, because that is how the
formula appears in every handbook. Python parses "^" as bitwise-XOR, so the AST
allowlist refused it with "operator BitXor is not allowed" - correct about the
syntax tree and useless to the person who wrote an ordinary formula. The step
failed, the run burned a replan, and the panel showed a security-sounding error
for a correct input.

Nothing calc is asked to evaluate wants a real XOR: every operand is a physical
quantity. So the translation is unambiguous - and it is announced in the working,
because a tool that silently rewrites its input is one whose answer cannot be
checked.
"""

from __future__ import annotations

import pytest

from kavach.tools.impl.calc import CalcError, UnsafeExpression, calc


def test_the_expression_from_the_failing_run_now_evaluates():
    result = calc("rho*v^2/2", substitutions="rho=998,v=1.2")
    assert result["result"] == pytest.approx(718.56, abs=0.01)


def test_a_plain_power_works():
    assert calc("2^3")["result"] == pytest.approx(8.0)


def test_darcy_weisbach_with_carets():
    result = calc("f*(L/D)*(rho*v^2/2)",
                  substitutions="f=0.02,L=50,D=0.1524,rho=998,v=1.2")
    assert result["result"] == pytest.approx(4715.0, abs=5)


def test_the_reinterpretation_is_stated_in_the_working():
    """Silently rewriting the input would make the derivation a lie."""
    steps = " ".join(calc("2^3")["steps"]).lower()
    assert "power" in steps and "^" in steps


def test_python_style_exponent_still_works():
    assert calc("2**10")["result"] == pytest.approx(1024.0)


def test_an_expression_with_no_caret_is_untouched():
    steps = " ".join(calc("2 + 2")["steps"]).lower()
    assert "power" not in steps


@pytest.mark.parametrize("attack", [
    "x = __import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__[0].__subclasses__()",
])
def test_the_translation_is_not_a_way_past_the_allowlist(attack):
    """calc is COMPUTE - it has no sandbox, so the gate is the whole defence."""
    with pytest.raises((UnsafeExpression, CalcError)):
        calc(attack)

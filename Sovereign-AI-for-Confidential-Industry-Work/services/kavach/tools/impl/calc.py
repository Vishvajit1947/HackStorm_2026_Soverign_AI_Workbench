"""`calc` - engineering maths that shows its working (§06).

Two reasons this exists rather than letting the model do arithmetic:

  1. A 7B model doing floating-point arithmetic in its head is the single easiest
     way to put a confidently wrong number in front of a plant engineer.
  2. The problem statement asks for "calculations with steps shown". A number
     with no derivation is not checkable, and in this domain an uncheckable
     number is worse than no answer.

SymPy is an optional dependency. Without it, the tool degrades to a restricted
arithmetic evaluator and SAYS SO in the result - it does not silently become less
capable.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

try:  # pragma: no cover - exercised by whichever branch is installed
    import sympy

    HAVE_SYMPY = True
except ImportError:  # pragma: no cover
    sympy = None
    HAVE_SYMPY = False


class CalcError(ValueError):
    pass


class UnsafeExpression(CalcError):
    """The expression contains something that is not arithmetic."""


# ---------------------------------------------------------------- the gate

# sympy.sympify() EVALUATES its input - it is documented as unsafe for untrusted
# strings, and `__import__('os').system('...')` executes happily inside it.
#
# calc takes a model-authored expression and runs UNSANDBOXED (its side effect is
# COMPUTE, not EXECUTE, so there is no container around it). A prompt-injected
# document could therefore reach the host through what looks like a calculator.
#
# So every expression is parsed and walked BEFORE it reaches any evaluator, and
# only arithmetic survives: numbers, plain symbol names, the operators below, and
# calls to a fixed list of maths functions. Attribute access, subscripting,
# lambdas, comprehensions and dunder names are all rejected outright - there is no
# legitimate engineering expression that needs them.

_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Load,
)


# A model asked to compute something writes what an engineer writes:
# "delta_P = (F / A) * (1 / rho)". That is an assignment, and ast.parse in eval
# mode rejects it outright - so every such step failed with "invalid syntax",
# burned a replan, and the run died having been handed a perfectly good formula.
# The label is not information the tool needs; the right-hand side is the sum.
#
# Only a single leading "name =" is stripped, and only when the left side is a
# bare identifier: "a == b" is a comparison and survives untouched, and anything
# with two assignments is still refused rather than silently half-read.
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", re.DOTALL)


def strip_assignment(expression: str) -> tuple[str, str | None]:
    """Return (expression, assigned name or None)."""
    match = _ASSIGNMENT.match(expression)
    if not match:
        return expression, None
    name, rhs = match.group(1), match.group(2)
    if _ASSIGNMENT.match(rhs):
        return expression, None        # chained assignment: let it fail loudly
    return rhs.strip(), name


def assert_safe_expression(expression: str) -> ast.Expression:
    """Parse and validate. Raises UnsafeExpression on anything but arithmetic."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"could not parse {expression!r}: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise UnsafeExpression(
                "attribute access is not allowed in an expression "
                f"(found .{node.attr})"
            )
        if isinstance(node, (ast.Subscript, ast.Lambda, ast.IfExp, ast.Await)):
            raise UnsafeExpression(
                f"{type(node).__name__} is not allowed in an expression"
            )
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise UnsafeExpression("comprehensions are not allowed in an expression")

        if isinstance(node, ast.Name) and (
            node.id.startswith("__") or node.id in _FORBIDDEN_NAMES
        ):
            raise UnsafeExpression(f"{node.id!r} is not allowed in an expression")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpression("only direct calls to maths functions are allowed")
            if node.func.id not in _FUNCS:
                raise UnsafeExpression(
                    f"{node.func.id!r} is not an available function; "
                    f"have {', '.join(sorted(_FUNCS))}"
                )
            if node.keywords:
                raise UnsafeExpression("keyword arguments are not allowed")

        # Numbers only. bool is an int subclass, so it needs excluding explicitly.
        if (
            isinstance(node, ast.Constant)
            and not isinstance(node.value, (int, float))
            and (isinstance(node.value, bool) or node.value is not None)
        ):
            raise UnsafeExpression(f"unsupported literal {node.value!r}")

        if isinstance(node, ast.BinOp) and type(node.op) not in _BIN_OPS:
            raise UnsafeExpression(f"operator {type(node.op).__name__} is not allowed")

        # THE ONE UNBOUNDED OPERATION IN AN ARITHMETIC ALLOWLIST.
        #
        # Everything else here costs what it looks like it costs. Exponentiation
        # does not: `10**10**10` is eleven characters that ask Python for a
        # number with ten billion digits, and it neither returns nor raises -
        # the process pins a core and allocates until the box dies. A model
        # writing a tower by accident, or a document body carrying one, would
        # take the whole workbench down with no error anywhere, which is worse
        # than any failure this module was already guarding against.
        #
        # 1000 is far past anything physical - no engineering quantity is raised
        # to a four-digit power - and it makes the cost of every surviving
        # expression bounded by inspection.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _assert_bounded_power(node)

        if isinstance(node, ast.UnaryOp) and type(node.op) not in _UNARY_OPS:
            raise UnsafeExpression(f"operator {type(node.op).__name__} is not allowed")

        if not isinstance(node, (*_ALLOWED_NODES, *_BIN_OPS, *_UNARY_OPS)):
            raise UnsafeExpression(
                f"{type(node).__name__} is not allowed in an expression"
            )

    return tree


MAX_EXPONENT = 1000


def _assert_bounded_power(node: ast.BinOp) -> None:
    """Refuse a power this box cannot afford to compute.

    Refused with a reason, never silently clamped: an exponent quietly reduced
    to 1000 returns a number that is not the answer to the expression that was
    written, and an unchecked wrong number in front of a plant engineer is the
    outcome this whole module exists to prevent.
    """
    exponent = node.right
    if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub):
        exponent = exponent.operand

    if isinstance(exponent, ast.BinOp) and isinstance(exponent.op, ast.Pow):
        raise UnsafeExpression(
            "a power of a power (a**b**c) is refused: it can ask for a number "
            "with more digits than this machine has memory, and nothing "
            "physical needs one"
        )

    if isinstance(exponent, ast.Constant) and isinstance(exponent.value, (int, float)):
        if abs(exponent.value) > MAX_EXPONENT:
            raise UnsafeExpression(
                f"the exponent {exponent.value} is above the limit of "
                f"{MAX_EXPONENT}; computing it would take longer than this run"
            )
        return

    # A symbolic exponent is fine at parse time - it has no value yet. It is
    # bounded when the substitutions arrive, below.
    return


def assert_bounded_substitutions(tree: ast.Expression, names: dict[str, float]) -> None:
    """The second half of the same guard: `x**y` with `y=10000000000` supplied
    as a substitution is the tower written in two arguments instead of one."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)):
            continue
        exponent = node.right
        if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, ast.USub):
            exponent = exponent.operand
        if isinstance(exponent, ast.Name) and exponent.id in names:
            value = names[exponent.id]
            if abs(value) > MAX_EXPONENT:
                raise UnsafeExpression(
                    f"{exponent.id} = {value:g} is used as an exponent and is "
                    f"above the limit of {MAX_EXPONENT}"
                )


_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "dir", "breakpoint", "exit", "quit",
    "memoryview", "bytearray", "object", "type", "super", "classmethod",
})


# ---------------------------------------------------------------- fallback

# A restricted evaluator for when SymPy is absent. Deliberately NOT eval():
# it walks the AST and permits only arithmetic over numbers and a short list of
# maths functions. No names, no attributes, no calls to anything else.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "atan",
                 "asin", "acos", "floor", "ceil", "fabs", "degrees", "radians")
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST, names: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"unsupported literal {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _safe_eval(node.left, names), _safe_eval(node.right, names)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand, names))
    if isinstance(node, ast.Name):
        if node.id in names:
            return float(names[node.id])
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise CalcError(f"unknown symbol {node.id!r} - give it in `substitutions`")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalcError("only basic maths functions are available")
        return _FUNCS[node.func.id](*[_safe_eval(a, names) for a in node.args])
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


# ---------------------------------------------------------------- parsing

def parse_substitutions(raw: str | None) -> dict[str, float]:
    """`"rho=998, v=1.2"` -> {"rho": 998.0, "v": 1.2}."""
    if not raw or not raw.strip():
        return {}
    out: dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            raise CalcError(
                f"cannot read substitution {part.strip()!r}; expected name=value"
            )
        name, value = part.split("=", 1)
        name = name.strip()
        try:
            out[name] = float(value.strip())
        except ValueError as exc:
            raise CalcError(f"{name}: {value.strip()!r} is not a number") from exc
    return out


# ---------------------------------------------------------------- the tool

def calc(
    expression: str,
    substitutions: str | None = None,
    solve_for: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    # THE GATE. Runs before any evaluator sees the string, on both paths.
    # sympify() would otherwise execute arbitrary Python from a model-authored
    # expression, and calc is not sandboxed.
    # Strip a leading "name =" BEFORE the gate, so the gate still sees a pure
    # expression and its guarantees are unchanged.
    expression, assigned = strip_assignment(expression)

    # In engineering notation "^" is exponent: v^2 is v squared. Python reads it
    # as bitwise-XOR, so the allowlist refused it with "operator BitXor is not
    # allowed" - accurate about the AST and useless to the engineer who wrote a
    # perfectly ordinary formula. Nothing here wants a real XOR: every operand
    # is a physical quantity, so translating is unambiguous rather than a guess.
    caret = expression.count("^")
    if caret:
        expression = expression.replace("^", "**")
        steps_note = f'read "^" as a power ({caret} occurrence' + ("s)" if caret > 1 else ")")
    else:
        steps_note = None

    tree = assert_safe_expression(expression)
    if solve_for:
        assert_safe_expression(solve_for)

    subs = parse_substitutions(substitutions)
    assert_bounded_substitutions(tree, subs)
    steps: list[str] = [f"expression: {expression}"]
    if assigned:
        steps.insert(0, f"read as: {assigned} = {expression}")
    if steps_note:
        # Say it in the working: a tool that silently rewrites its input is a
        # tool whose answer you cannot check.
        steps.insert(0, steps_note)
    if subs:
        steps.append(
            "substitute:  " + ", ".join(f"{k} = {v:g}" for k, v in sorted(subs.items()))
        )

    if not HAVE_SYMPY:
        return _calc_fallback(tree, subs, steps, solve_for)

    try:
        expr = sympy.sympify(expression)
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise CalcError(f"could not parse {expression!r}: {exc}") from exc

    if solve_for:
        symbol = sympy.Symbol(solve_for)
        substituted = expr.subs(subs) if subs else expr
        steps.append(f"solve for {solve_for}: {substituted} = 0")
        solutions = sympy.solve(sympy.Eq(substituted, 0), symbol)
        steps.append(f"solutions:   {solutions}")
        return {
            "result": [str(s) for s in solutions],
            "steps": steps,
            "engine": "sympy",
            "exact": True,
        }

    simplified = sympy.simplify(expr)
    if str(simplified) != str(expr):
        steps.append(f"simplify:    {simplified}")

    substituted = simplified.subs(subs) if subs else simplified
    if subs:
        steps.append(f"evaluate:    {substituted}")

    free = substituted.free_symbols
    if free:
        # Still symbolic: report it honestly rather than inventing values.
        steps.append(
            "result is symbolic - unresolved: " + ", ".join(sorted(str(s) for s in free))
        )
        return {
            "result": str(substituted),
            "steps": steps,
            "engine": "sympy",
            "exact": True,
            "unresolved": sorted(str(s) for s in free),
        }

    numeric = _as_real(sympy.N(substituted), expression)
    steps.append(f"result:      {numeric:.6g}")
    return {
        "result": numeric,
        "steps": steps,
        "engine": "sympy",
        "exact": False,
    }


def _as_real(value: Any, expression: str) -> float:
    """A sympy result as a real float, or a CalcError that says what happened.

    `1/0` simplifies to sympy's `zoo` (complex infinity) and `sqrt(-1)` to `I`.
    Neither converts to float, and the TypeError that came out named neither the
    expression nor the reason - a plant engineer who divided by zero got
    "TypeError: Cannot convert complex to float" from a calculator.

    Refused rather than returned as inf or nan: a spreadsheet that carries `inf`
    down a column of sums produces a page of `inf`, and the first wrong number
    is the one worth reporting.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        text = str(value)
        if "zoo" in text:
            # SymPy folds several distinct causes into complex infinity - 1/0
            # and log(0) both arrive here as `zoo`. Naming one of them would be
            # a guess, and sending an engineer to look for a denominator that
            # is not there wastes more time than saying less.
            raise CalcError(
                f"{expression!r} is undefined at these values - it divides by "
                "zero, or takes a log of zero"
            ) from exc
        if text.strip("-+") == "oo":
            # log(0) is -oo and is unbounded, not a division by zero. Naming the
            # wrong cause sends the engineer looking for a denominator that is
            # not there.
            raise CalcError(
                f"{expression!r} is unbounded - the result is infinite ({text})"
            ) from exc
        if "I" in text:
            raise CalcError(
                f"{expression!r} has no real value (the result is complex: {text})"
            ) from exc
        raise CalcError(
            f"{expression!r} did not reduce to a number (it is {text})"
        ) from exc

    if math.isnan(numeric):
        raise CalcError(f"{expression!r} is not a number (0/0 or similar)")
    if math.isinf(numeric):
        raise CalcError(
            f"{expression!r} overflows - the result is larger than this machine "
            "can represent"
        )
    return numeric


def _calc_fallback(
    tree: ast.Expression,
    subs: dict[str, float],
    steps: list[str],
    solve_for: str | None,
) -> dict[str, Any]:
    if solve_for:
        raise CalcError(
            "solve_for needs SymPy, which is not installed. "
            "Install it with: pip install sympy"
        )

    try:
        value = _safe_eval(tree, subs)
    except ZeroDivisionError as exc:
        raise CalcError("the expression divides by zero") from exc
    except ValueError as exc:
        # math.sqrt(-1), math.log(0) and friends. The domain error is the
        # useful part of the message and the user's own input explains it.
        raise CalcError(f"the expression is outside the domain of its function: {exc}") from exc
    except OverflowError as exc:
        raise CalcError("the expression overflows this machine's number range") from exc

    if math.isnan(value) or math.isinf(value):
        raise CalcError("the expression has no finite value")
    steps.append(f"result:      {value:.6g}")
    steps.append(
        "NOTE: SymPy is not installed, so this used a restricted arithmetic "
        "evaluator - no symbolic simplification was performed."
    )
    return {
        "result": value,
        "steps": steps,
        "engine": "fallback",
        "exact": False,
        "degraded": True,
    }

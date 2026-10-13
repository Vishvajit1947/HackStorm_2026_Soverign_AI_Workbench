"""JSON schema -> GBNF (§05).

A 4-8B model will produce malformed tool JSON often enough to wreck a live demo.
The architecture doc calls this "the single most common reason small-model agent
demos fall over", and the mitigation is not better prompting - it is constrained
decoding.

A GBNF grammar handed to llama.cpp restricts the sampler at every token, so
output that does not match the grammar is not unlikely, it is UNREACHABLE. The
model cannot emit a missing brace or an unquoted key because those tokens are
masked out.

What this generates is deliberately strict:

  - keys appear in a fixed order, so there is no permutation ambiguity
  - required keys are mandatory, optional keys are genuinely optional
  - types are enforced (a `number` field cannot receive a quoted string)
  - enums collapse to a literal alternation

Ordering the keys is a real trade-off: it removes a large class of grammar
ambiguity and shrinks the state space, at the cost of rejecting valid JSON that
happens to order keys differently. Since we generate the grammar and the model
only ever fills it, that costs nothing here.
"""

from __future__ import annotations

from typing import Any

# Primitive rules shared by every generated grammar.
PREAMBLE = r"""
ws      ::= [ \t\n]*
string  ::= "\"" char* "\"" ws
char    ::= [^"\\\x00-\x1F] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex     ::= [0-9a-fA-F]
number  ::= "-"? int frac? exp? ws
int     ::= "0" | [1-9] [0-9]*
frac    ::= "." [0-9]+
exp     ::= [eE] [-+]? [0-9]+
boolean ::= ("true" | "false") ws
null    ::= "null" ws
""".strip()


def _escape(literal: str) -> str:
    return literal.replace("\\", "\\\\").replace('"', '\\"')


def _value_rule(spec: dict[str, Any], rule_name: str) -> tuple[str, list[str]]:
    """Return (rule reference, extra rule definitions) for one property."""
    extra: list[str] = []

    if "enum" in spec:
        options = " | ".join(f'"\\"{_escape(str(v))}\\""' for v in spec["enum"])
        extra.append(f"{rule_name} ::= ({options}) ws")
        return rule_name, extra

    kind = spec.get("type", "string")

    if kind == "string":
        return "string", extra
    if kind in ("number", "integer"):
        return "number", extra
    if kind == "boolean":
        return "boolean", extra
    if kind == "array":
        item_ref, item_extra = _value_rule(spec.get("items", {"type": "string"}),
                                           f"{rule_name}-item")
        extra.extend(item_extra)
        extra.append(
            f'{rule_name} ::= "[" ws ({item_ref} ("," ws {item_ref})*)? "]" ws'
        )
        return rule_name, extra
    if kind == "object":
        # A schema that says "object" with no declared properties is a free-form
        # container, not a nested type - the plan schema's `args` is exactly
        # that, since its shape depends on which tool the step calls. Emitting a
        # generic JSON object for it lets a plan carry `{"path": "x.txt"}`
        # directly instead of a string with JSON escaped inside it, which is the
        # quoting a 1.5B reliably gets wrong - the plan then dies over escaping
        # rather than over reasoning.
        # Presence of the key, not its truthiness: `properties: {}` is still a
        # DECLARED shape and stays refused, while a bare {"type": "object"} is the
        # free-form container the plan schema means.
        if "properties" not in spec:
            extra.append(
                f'{rule_name} ::= "{{" ws ({rule_name}-kv ("," ws {rule_name}-kv)*)? "}}" ws'
            )
            extra.append(
                f'{rule_name}-kv ::= string ":" ws {rule_name}-val'
            )
            extra.append(
                f'{rule_name}-val ::= string | number | boolean | null | {rule_name} '
                f'| "[" ws ({rule_name}-val ("," ws {rule_name}-val)*)? "]" ws'
            )
            return rule_name, extra
        # A DECLARED nested object is still refused: tool schemas are kept flat
        # on purpose, and that is what keeps small models reliable at tool calls.
        # Nested objects are not generated: tool schemas are kept flat on purpose
        # (§05), and a nested grammar is where small models start failing again.
        raise ValueError(
            "nested objects are not supported in tool schemas - keep them flat "
            "and few-argument, which is what makes small models reliable"
        )

    return "string", extra


def schema_to_gbnf(schema: dict[str, Any], root_name: str = "root") -> str:
    """Build a GBNF grammar that admits exactly this schema's JSON objects."""
    properties: dict[str, Any] = schema.get("properties", {})
    required = list(schema.get("required", []))

    if not properties:
        return f'{root_name} ::= "{{" ws "}}" ws\n\n{PREAMBLE}'

    # Required keys first, in declaration order; optional keys after.
    ordered = [k for k in properties if k in required]
    ordered += [k for k in properties if k not in required]

    rules: list[str] = []
    parts: list[str] = []

    for index, key in enumerate(ordered):
        ref, extra = _value_rule(properties[key], f"{root_name}-{key}")
        rules.extend(extra)

        # Every pair after the first carries its own leading comma, so an omitted
        # optional key cannot leave a dangling separator.
        pair = f'"\\"{_escape(key)}\\"" ws ":" ws {ref}'
        if index > 0:
            pair = f'"," ws {pair}'

        if key in required:
            parts.append(f"({pair})")
        else:
            parts.append(f"({pair})?")

    body = " ".join(parts)
    root = f'{root_name} ::= "{{" ws {body} "}}" ws'

    return "\n".join([root, *rules, "", PREAMBLE])


def tool_call_grammar(tool_name: str, schema: dict[str, Any]) -> str:
    """A grammar for a full tool call: {"tool": "<name>", "args": {...}}.

    The tool name is pinned to a literal, so the model cannot invent a tool that
    does not exist - a failure mode no amount of prompting reliably prevents.
    """
    args_grammar = schema_to_gbnf(schema, root_name="args")
    args_root, *rest = args_grammar.split("\n")
    rest_body = "\n".join(rest)

    root = (
        'root ::= "{" ws "\\"tool\\"" ws ":" ws '
        f'"\\"{_escape(tool_name)}\\"" ws "," ws '
        '"\\"args\\"" ws ":" ws args "}" ws'
    )
    return f"{root}\n{args_root}\n{rest_body}"


def plan_grammar(schema: dict[str, Any]) -> str:
    """A grammar for an ordered ARRAY of plan steps.

    Without this the planner is the one place in the build that asks a 7B for
    structured JSON and then merely hopes: qwen2.5-7b emitted perfectly formed
    steps carrying `tool`, `args` and `acceptance`, and silently omitted
    `intent` on every one of them, so validate_plan() rejected step 0 and the
    whole run failed before a single step executed.

    A missing required key is exactly what constrained decoding removes - not
    made unlikely, made unreachable. The step count is deliberately NOT bounded
    here; validate_plan() checks it against the budget, and expressing a
    repetition limit in GBNF costs far more grammar than the check is worth.
    """
    step = schema_to_gbnf(schema, root_name="step")
    body = step.split(PREAMBLE)[0].strip()
    root = 'root ::= ws "[" ws (step ("," ws step)*)? "]" ws'
    return f"{root}\n\n{body}\n\n{PREAMBLE}"


def tool_choice_grammar(tools: list[tuple[str, dict[str, Any]]]) -> str:
    """A grammar admitting a call to any ONE of several tools.

    This is what the agent executor hands the model when several tools are
    available: it can choose, but only from tools that actually exist, and only
    with arguments matching that tool's schema.
    """
    if not tools:
        raise ValueError("no tools to build a grammar from")
    if len(tools) == 1:
        return tool_call_grammar(*tools[0])

    alternatives: list[str] = []
    rules: list[str] = []

    for index, (name, schema) in enumerate(tools):
        prefix = f"call{index}"
        args_grammar = schema_to_gbnf(schema, root_name=f"{prefix}-args")
        # Drop the shared preamble from each; it is appended once at the end.
        body = args_grammar.split(PREAMBLE)[0].strip()
        rules.append(body)
        rules.append(
            f'{prefix} ::= "{{" ws "\\"tool\\"" ws ":" ws '
            f'"\\"{_escape(name)}\\"" ws "," ws '
            f'"\\"args\\"" ws ":" ws {prefix}-args "}}" ws'
        )
        alternatives.append(prefix)

    root = "root ::= " + " | ".join(alternatives)
    return "\n".join([root, "", *rules, "", PREAMBLE])

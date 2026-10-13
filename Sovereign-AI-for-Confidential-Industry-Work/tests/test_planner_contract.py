"""The planner's prompt, grammar and validator must agree with each other.

Three bugs live here, and every one of them failed the SAME way from outside:
"planning failed ... step 0 has no intent", with no steps, before anything ran.
Nothing in that message points at a prompt, a grammar or a schema, which is why
these are pinned by contract rather than left to a live run to discover.

  1. validate_plan() required `intent`; PLANNER_SYSTEM never asked for it, so
     qwen2.5-7b emitted tool/args/acceptance and omitted intent on every step.
  2. The module docstring advertised a GBNF constraint that was never passed.
     PLAN_SCHEMA was dead code and the plan was generated unconstrained.
  3. The prompt listed tool NAMES only, so the model guessed argument names -
     `fs.list(dir=...)` instead of `subdir`, because `dir` is what it is called
     nearly everywhere else in the training data.
"""

from __future__ import annotations

import json

from kavach.agent.planner import (
    PLAN_SCHEMA,
    PLANNER_SYSTEM,
    Planner,
    _tool_catalogue,
)
from kavach.tools import get_tool_registry
from kavach.tools.grammar import PREAMBLE, plan_grammar


def _rendered_system() -> str:
    return PLANNER_SYSTEM.format(
        tools=_tool_catalogue(get_tool_registry()), max_steps=12
    )


# ------------------------------------------------- prompt / validator agreement

def test_the_prompt_asks_for_every_field_the_validator_demands():
    """Bug 1. A field the validator rejects a plan for must be a field the
    prompt actually asked for - the model cannot supply a key nobody named."""
    system = _rendered_system()
    for field in PLAN_SCHEMA["required"]:
        assert field in system, (
            f"validate_plan() requires {field!r} but the planner prompt never "
            "mentions it; the model has no way to know it is needed"
        )


def test_the_prompt_shows_the_step_shape_not_just_the_field_names():
    """Naming the keys in prose was not enough to get them emitted reliably;
    the prompt carries a literal example object."""
    system = _rendered_system()
    assert '"intent"' in system and '"acceptance"' in system


# ------------------------------------------------------------- tool catalogue

def test_the_catalogue_lists_the_real_argument_name_of_every_tool():
    """Bug 3. Every argument the registry will validate against must appear in
    the prompt, or the model invents a plausible synonym and the plan is
    rejected for a name it was never given."""
    catalogue = _tool_catalogue(get_tool_registry())
    for tool in get_tool_registry().all():
        assert tool.name in catalogue
        for arg in tool.properties:
            assert arg in catalogue, (
                f"{tool.name} takes {arg!r} but the catalogue never says so"
            )


def test_the_catalogue_distinguishes_required_from_optional_arguments():
    """Without the marker the model treats every key as mandatory and pads the
    call with invented values, which the schema then rejects."""
    catalogue = _tool_catalogue(get_tool_registry())
    for tool in get_tool_registry().all():
        for arg in tool.properties:
            if arg not in tool.required:
                assert f"{arg}?:" in catalogue, f"{tool.name}.{arg} not marked optional"


def test_fs_list_is_advertised_with_subdir_and_never_dir():
    """The concrete regression: `dir` is the name the model reached for."""
    catalogue = _tool_catalogue(get_tool_registry())
    assert "fs.list(subdir?: string)" in catalogue


# -------------------------------------------------------------- plan grammar

def test_the_grammar_makes_every_required_field_mandatory():
    """Bug 2. Under the grammar a missing key is not unlikely, it is
    unreachable - which is the only reason the prompt and the validator can be
    trusted to stay in step."""
    grammar = plan_grammar(PLAN_SCHEMA)
    for field in PLAN_SCHEMA["required"]:
        assert f'"\\"{field}\\""' in grammar
    # optional-key syntax must not appear on a required field
    assert ")?" not in grammar.split("step ::=")[1].split("\n")[0]


def test_the_grammar_admits_an_array_not_a_single_step():
    grammar = plan_grammar(PLAN_SCHEMA)
    root = grammar.split("\n")[0]
    assert root.startswith("root ::=") and "[" in root and "step" in root


def test_the_planner_actually_passes_the_grammar(monkeypatch):
    """The bug was not a wrong grammar, it was a grammar that was never handed
    over: route() accepted the argument and the planner never sent one."""
    seen = {}

    class _FakeRouter:
        def route(self, req, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here - we only care what was passed")

    planner = Planner(router=_FakeRouter())
    try:
        planner.plan("do something")
    except RuntimeError:
        pass

    assert seen.get("grammar"), "the planner sent no grammar at all"
    assert '"\\"intent\\""' in seen["grammar"]


# ------------------------------------------- JSON strings and control chars

def test_generated_json_strings_forbid_raw_control_characters():
    """A grammar that admits JSON the json module then rejects is worse than no
    grammar: it guarantees output that LOOKS constrained and will not parse.

    `char ::= [^"\\]` permitted a literal newline inside a string, so a plan
    whose `args` carried multi-line content was emitted perfectly and died in
    json.loads with "Invalid control character". Shared by every generated
    grammar, so tool calls had it too.
    """
    char_rule = next(
        line for line in PREAMBLE.splitlines() if line.startswith("char")
    )
    assert "x00-" in char_rule and "x1F" in char_rule, (
        "the char rule admits raw control characters; JSON forbids U+0000-U+001F "
        f"inside a string. Rule was: {char_rule}"
    )


def test_the_json_module_really_does_reject_those_characters():
    """Pins the premise of the test above, so it cannot quietly become vacuous."""
    import pytest

    with pytest.raises(json.JSONDecodeError):
        json.loads('{"content": "line1' + chr(10) + 'line2"}')


# ------------------------------------------- the mock reads this prompt too

def test_the_mock_backend_can_still_read_the_tool_list_out_of_the_prompt():
    """The mock parses tool names from the planner's system prompt rather than
    importing them, so it stays a backend - but that couples it to the prompt's
    SHAPE. When the prompt grew per-tool signatures its regex quietly matched
    nothing, the mock planned a single calc step, and four agent-loop tests
    failed with `StopIteration` on `next(s for s in run.steps ...)` - a very
    long way from the prompt that caused it.
    """
    from kavach.runtime.backends.mock import MockBackend

    parsed = MockBackend._tools_from_system(_rendered_system())
    assert parsed, "the mock read no tools at all out of the planner prompt"
    assert set(parsed) == set(get_tool_registry().names())

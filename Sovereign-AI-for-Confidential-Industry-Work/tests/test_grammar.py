"""JSON schema -> GBNF (§05).

The architecture doc calls malformed tool JSON "the single most common reason
small-model agent demos fall over". Constrained decoding is the mitigation: a
grammar makes non-conforming output UNREACHABLE rather than merely unlikely.

IMPORTANT LIMIT ON THESE TESTS. They assert structural properties of the
generated grammar text - they do NOT prove llama.cpp accepts it, because that
needs a real llama.cpp build. That verification is checkpoint R4, and until it
runs, "GBNF prevents malformed tool calls" is a designed property, not a
measured one.
"""

from __future__ import annotations

import pytest

from kavach.tools import get_tool_registry
from kavach.tools.grammar import (
    PREAMBLE,
    schema_to_gbnf,
    tool_call_grammar,
    tool_choice_grammar,
)

FLAT = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "count": {"type": "integer"},
        "flag": {"type": "boolean"},
    },
    "required": ["path"],
}


# ---------------------------------------------------------------- structure

def test_required_key_is_mandatory():
    grammar = schema_to_gbnf(FLAT)
    root = grammar.splitlines()[0]
    assert '("\\"path\\"" ws ":" ws string)' in root
    assert '("\\"path\\"" ws ":" ws string)?' not in root


def test_optional_keys_are_optional():
    root = schema_to_gbnf(FLAT).splitlines()[0]
    assert '"\\"count\\"" ws ":" ws number)?' in root
    assert '"\\"flag\\"" ws ":" ws boolean)?' in root


def test_required_keys_come_first():
    """Fixed key order removes a large class of grammar ambiguity. Since we
    generate the grammar and the model only fills it, that costs nothing."""
    root = schema_to_gbnf(FLAT).splitlines()[0]
    assert root.index("path") < root.index("count")


def test_every_optional_pair_carries_its_own_comma():
    """An omitted optional key must not leave a dangling separator - the classic
    way a generated grammar admits invalid JSON."""
    root = schema_to_gbnf(FLAT).splitlines()[0]
    assert '("," ws "\\"count\\"' in root
    assert '("," ws "\\"flag\\"' in root


def test_types_map_to_the_right_rules():
    root = schema_to_gbnf(FLAT).splitlines()[0]
    assert '"\\"path\\"" ws ":" ws string' in root
    assert '"\\"count\\"" ws ":" ws number' in root
    assert '"\\"flag\\"" ws ":" ws boolean' in root


def test_preamble_is_included():
    assert PREAMBLE in schema_to_gbnf(FLAT)


def test_empty_schema_admits_only_an_empty_object():
    grammar = schema_to_gbnf({"type": "object", "properties": {}})
    assert grammar.splitlines()[0] == 'root ::= "{" ws "}" ws'


# ---------------------------------------------------------------- types

def test_enum_becomes_a_literal_alternation():
    grammar = schema_to_gbnf({
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
        "required": ["mode"],
    })
    assert '"\\"fast\\"" | "\\"slow\\""' in grammar


def test_array_generates_its_own_rule():
    grammar = schema_to_gbnf({
        "type": "object",
        "properties": {"files": {"type": "array", "items": {"type": "string"}}},
    })
    assert 'root-files ::= "[" ws (string ("," ws string)*)? "]" ws' in grammar


def test_nested_objects_are_refused():
    """Tool schemas are kept flat on purpose - nesting is where small models
    start failing again, so the generator refuses rather than emitting a grammar
    that will not hold up."""
    with pytest.raises(ValueError, match="flat"):
        schema_to_gbnf({
            "type": "object",
            "properties": {"cfg": {"type": "object", "properties": {}}},
        })


# ---------------------------------------------------------------- tool calls

def test_tool_call_pins_the_tool_name():
    """The model cannot invent a tool that does not exist - a failure mode no
    amount of prompting reliably prevents."""
    grammar = tool_call_grammar("fs.write", FLAT)
    assert '"\\"fs.write\\""' in grammar
    assert grammar.startswith("root ::=")


def test_tool_call_wraps_args():
    grammar = tool_call_grammar("calc", FLAT)
    assert '"\\"args\\"" ws ":" ws args' in grammar
    assert "args ::=" in grammar


def test_choice_grammar_offers_each_tool():
    grammar = tool_choice_grammar([
        ("calc", {"type": "object", "properties": {"expression": {"type": "string"}},
                  "required": ["expression"]}),
        ("fs.read", {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}),
    ])
    assert grammar.splitlines()[0] == "root ::= call0 | call1"
    assert '"\\"calc\\""' in grammar
    assert '"\\"fs.read\\""' in grammar


def test_choice_grammar_includes_the_preamble_once():
    grammar = tool_choice_grammar([
        ("a", {"type": "object", "properties": {"x": {"type": "string"}}}),
        ("b", {"type": "object", "properties": {"y": {"type": "string"}}}),
    ])
    assert grammar.count("hex     ::= [0-9a-fA-F]") == 1


def test_single_tool_choice_delegates():
    single = tool_choice_grammar([("calc", FLAT)])
    assert single == tool_call_grammar("calc", FLAT)


def test_empty_choice_is_an_error():
    with pytest.raises(ValueError, match="no tools"):
        tool_choice_grammar([])


# ---------------------------------------------------------------- real tools

def test_every_shipped_tool_produces_a_grammar():
    """If a manifest ever grows a nested object this fails, which is the point."""
    for spec in get_tool_registry().all():
        grammar = tool_call_grammar(spec.name, spec.schema)
        assert grammar.startswith("root ::=")
        assert PREAMBLE in grammar


def test_shipped_tools_can_be_offered_together():
    tools = [(s.name, s.schema) for s in get_tool_registry().all()]
    grammar = tool_choice_grammar(tools)
    for name, _ in tools:
        assert f'\\"{name}\\"' in grammar


def test_quotes_in_a_key_are_escaped():
    grammar = schema_to_gbnf({
        "type": "object",
        "properties": {'we"ird': {"type": "string"}},
        "required": ['we"ird'],
    })
    assert '\\\\"' in grammar or 'we\\\\"ird' in grammar

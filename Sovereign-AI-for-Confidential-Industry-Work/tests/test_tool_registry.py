"""Tool registry and manifests (§06).

The claim under test is that tools are DECLARED, not hard-coded - so the policy
engine can reason about a tool it has never seen, and adding one is a manifest
plus a handler.
"""

from __future__ import annotations

import pytest
import yaml

from kavach.tools.registry import (
    InvalidArguments,
    ToolError,
    ToolNotFound,
    ToolRegistry,
    get_tool_registry,
)
from kavach.types import SideEffect


@pytest.fixture(scope="module")
def registry():
    return get_tool_registry()


# ---------------------------------------------------------------- shipped set

def test_the_shipped_tools_load(registry):
    assert set(registry.names()) == {
        "calc", "code.exec", "doc.write", "fs.read", "fs.list", "fs.write",
        "kb.search",
    }


def test_every_manifest_has_a_handler(registry):
    """An unbound manifest loads but cannot run. None should ship that way."""
    assert registry.unbound() == []


def test_side_effects_are_declared_correctly(registry):
    assert registry.get("fs.read").side_effect is SideEffect.READ
    assert registry.get("fs.write").side_effect is SideEffect.WRITE
    assert registry.get("code.exec").side_effect is SideEffect.EXECUTE
    assert registry.get("calc").side_effect is SideEffect.COMPUTE


def test_code_exec_declares_a_networkless_sandbox(registry):
    """The strongest claim in the tool plane, asserted at the manifest level."""
    spec = registry.get("code.exec")
    assert spec.sandbox is not None
    assert spec.sandbox.network == "none"
    assert spec.sandbox.networkless


def test_only_code_exec_is_sandboxed(registry):
    """The others do not execute model-authored code, so they need no container."""
    sandboxed = [t.name for t in registry.all() if t.sandbox]
    assert sandboxed == ["code.exec"]


def test_unknown_tool_raises_with_the_available_list(registry):
    with pytest.raises(ToolNotFound, match="calc"):
        registry.get("nope")


# ---------------------------------------------------------------- manifests

def write_manifest(tmp_path, body: dict) -> ToolRegistry:
    d = tmp_path / "manifests"
    d.mkdir(exist_ok=True)
    (d / f"{body['name']}.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return ToolRegistry(d)


def test_a_new_manifest_becomes_a_tool(tmp_path):
    """'Add tools without redesign' - a manifest plus a handler, no code change
    anywhere else."""
    reg = write_manifest(tmp_path, {
        "name": "invented.tool",
        "description": "something nobody has seen",
        "side_effect": "DELETE",
        "schema": {"type": "object", "properties": {"target": {"type": "string"}},
                   "required": ["target"]},
    })
    spec = reg.get("invented.tool")
    assert spec.side_effect is SideEffect.DELETE
    assert spec.handler is None          # declared, not yet implemented
    assert reg.unbound() == ["invented.tool"]


def test_a_networked_sandbox_is_refused(tmp_path):
    """A tool that could open a socket breaks the sovereignty claim, so the
    registry refuses to load it at all."""
    with pytest.raises(ToolError, match="networkless"):
        write_manifest(tmp_path, {
            "name": "leaky.tool",
            "side_effect": "EXECUTE",
            "schema": {"type": "object", "properties": {}},
            "sandbox": {"network": "bridge"},
        })


def test_invalid_side_effect_is_refused(tmp_path):
    with pytest.raises(ToolError, match="invalid manifest"):
        write_manifest(tmp_path, {
            "name": "bad.tool",
            "side_effect": "TELEPORT",
            "schema": {"type": "object", "properties": {}},
        })


def test_reload_picks_up_a_new_manifest(tmp_path):
    reg = write_manifest(tmp_path, {
        "name": "first.tool", "side_effect": "READ",
        "schema": {"type": "object", "properties": {}},
    })
    assert reg.names() == ["first.tool"]
    write_manifest(tmp_path, {
        "name": "second.tool", "side_effect": "READ",
        "schema": {"type": "object", "properties": {}},
    })
    reg.reload()
    assert reg.names() == ["first.tool", "second.tool"]


# ---------------------------------------------------------------- validation

def test_required_arguments_are_enforced(registry):
    with pytest.raises(InvalidArguments, match="missing required"):
        registry.get("calc").validate({})


def test_optional_arguments_may_be_omitted(registry):
    assert registry.get("calc").validate({"expression": "1+1"}) == {"expression": "1+1"}


def test_unknown_arguments_are_refused(registry):
    with pytest.raises(InvalidArguments, match="unknown argument"):
        registry.get("calc").validate({"expression": "1+1", "precision": 4})


def test_array_of_strings_is_checked(registry):
    spec = registry.get("code.exec")
    assert spec.validate({"code": "x", "files": ["a.txt"]})
    with pytest.raises(InvalidArguments, match="array of strings"):
        spec.validate({"code": "x", "files": [1, 2]})


def test_non_array_where_array_expected(registry):
    with pytest.raises(InvalidArguments, match="expected an array"):
        registry.get("code.exec").validate({"code": "x", "files": "a.txt"})


def test_enum_is_enforced(tmp_path):
    reg = write_manifest(tmp_path, {
        "name": "enum.tool", "side_effect": "READ",
        "schema": {"type": "object",
                   "properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}},
                   "required": ["mode"]},
    })
    spec = reg.get("enum.tool")
    assert spec.validate({"mode": "fast"})
    with pytest.raises(InvalidArguments, match="not one of"):
        spec.validate({"mode": "sideways"})


def test_non_object_arguments_are_refused(registry):
    with pytest.raises(InvalidArguments, match="must be an object"):
        registry.get("calc").validate(["expression"])


# ---------------------------------------------------------------- reporting

def test_summary_exposes_the_gate_for_each_tool(registry):
    summary = registry.summary()
    assert summary["count"] == 7
    entry = next(t for t in summary["tools"] if t["name"] == "code.exec")
    assert entry["sandboxed"] is True
    assert entry["network"] == "none"
    assert entry["bound"] is True

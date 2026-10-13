"""Tool registry (§06).

Tools are declared, not hard-coded. Each carries a JSON schema, a permission
scope, a side-effect class and whether it needs human approval - so the policy
engine can reason about a tool it has never seen, and adding one is a manifest
plus a handler.

That last property is what makes "new tools without redesign" credible rather
than aspirational: the policy engine keys off `side_effect`, never off the name,
so a brand-new tool declaring DELETE is gated on its first invocation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..types import SideEffect

MANIFEST_DIR = Path(__file__).with_name("manifests")


class ToolError(RuntimeError):
    """A tool failed in a way the agent should see and can act on."""


class ToolNotFound(ToolError):
    pass


class InvalidArguments(ToolError):
    """Arguments did not match the declared schema."""


@dataclass
class SandboxSpec:
    network: str = "none"
    rootfs: str = "read-only"
    writable: list[str] = field(default_factory=lambda: ["/work"])
    mem_gb: float = 2.0
    cpus: float = 1.5
    pids_limit: int = 128

    @property
    def networkless(self) -> bool:
        return self.network == "none"


@dataclass
class ToolSpec:
    name: str
    description: str
    side_effect: SideEffect
    schema: dict[str, Any]
    requires_approval: bool = False
    max_runtime_s: int = 30
    sandbox: SandboxSpec | None = None
    returns: str = ""
    # Set on tools whose output is a deliverable the user takes away, so the
    # policy engine can apply the final-deliverable gate.
    produces_artifact: bool = False

    handler: Callable[..., Any] | None = None

    @property
    def properties(self) -> dict[str, Any]:
        return self.schema.get("properties", {})

    @property
    def required(self) -> list[str]:
        return list(self.schema.get("required", []))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "side_effect": self.side_effect.value,
            "requires_approval": self.requires_approval,
            "max_runtime_s": self.max_runtime_s,
            "schema": self.schema,
            "returns": self.returns,
            "produces_artifact": self.produces_artifact,
            "sandboxed": self.sandbox is not None,
            "network": self.sandbox.network if self.sandbox else None,
            "bound": self.handler is not None,
        }

    # ---------------------------------------------------------------- validation

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check arguments against the declared schema.

        A deliberately small subset of JSON Schema - type, required, enum - because
        the schemas are kept flat and few-argument on purpose (§05: complex tool
        schemas are a leading cause of malformed tool calls from small models).
        Rejecting unknown keys matters: a model that invents `path` alongside
        `file` should be told, not silently ignored.
        """
        if not isinstance(args, dict):
            raise InvalidArguments(f"{self.name}: arguments must be an object")

        missing = [k for k in self.required if k not in args]
        if missing:
            raise InvalidArguments(
                f"{self.name}: missing required argument(s) {', '.join(missing)}"
            )

        unknown = [k for k in args if k not in self.properties]
        if unknown:
            raise InvalidArguments(
                f"{self.name}: unknown argument(s) {', '.join(unknown)}; "
                f"expected some of {', '.join(self.properties) or '(none)'}"
            )

        cleaned: dict[str, Any] = {}
        for key, value in args.items():
            spec = self.properties[key]
            cleaned[key] = self._coerce(key, value, spec)
        return cleaned

    def _coerce(self, key: str, value: Any, spec: dict[str, Any]) -> Any:
        expected = spec.get("type")

        if expected == "string":
            if not isinstance(value, str):
                raise InvalidArguments(f"{self.name}.{key}: expected a string")
        elif expected == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidArguments(f"{self.name}.{key}: expected an integer")
        elif expected == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidArguments(f"{self.name}.{key}: expected a number")
        elif expected == "boolean":
            if not isinstance(value, bool):
                raise InvalidArguments(f"{self.name}.{key}: expected true or false")
        elif expected == "array":
            if not isinstance(value, list):
                raise InvalidArguments(f"{self.name}.{key}: expected an array")
            item_type = spec.get("items", {}).get("type")
            if item_type == "string" and not all(isinstance(v, str) for v in value):
                raise InvalidArguments(f"{self.name}.{key}: expected an array of strings")

        if "enum" in spec and value not in spec["enum"]:
            raise InvalidArguments(
                f"{self.name}.{key}: {value!r} is not one of {spec['enum']}"
            )

        return value


class ToolRegistry:
    def __init__(self, manifest_dir: Path | None = None):
        self.manifest_dir = manifest_dir or MANIFEST_DIR
        self._tools: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        self.reload()

    # ---------------------------------------------------------------- loading

    def reload(self) -> int:
        """Re-read every manifest and re-bind handlers. Live-reloadable, so a new
        tool is a manifest plus a handler and no restart."""
        tools: dict[str, ToolSpec] = {}

        for path in sorted(self.manifest_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            spec = self._parse(raw, path)
            tools[spec.name] = spec

        with self._lock:
            self._tools = tools

        self._bind_handlers()
        return len(tools)

    def _parse(self, raw: dict[str, Any], path: Path) -> ToolSpec:
        try:
            name = raw["name"]
            side_effect = SideEffect(raw["side_effect"])
        except (KeyError, ValueError) as exc:
            raise ToolError(f"{path.name}: invalid manifest - {exc}") from exc

        sandbox = None
        if raw.get("sandbox"):
            s = raw["sandbox"]
            sandbox = SandboxSpec(
                network=s.get("network", "none"),
                rootfs=s.get("rootfs", "read-only"),
                writable=list(s.get("writable", ["/work"])),
                mem_gb=float(s.get("mem_gb", 2.0)),
                cpus=float(s.get("cpus", 1.5)),
                pids_limit=int(s.get("pids_limit", 128)),
            )
            if not sandbox.networkless:
                raise ToolError(
                    f"{path.name}: sandbox.network is {sandbox.network!r}. Kavach "
                    "only ships networkless sandboxes - a tool that can open a "
                    "socket breaks the sovereignty claim."
                )

        return ToolSpec(
            name=name,
            description=raw.get("description", ""),
            side_effect=side_effect,
            schema=raw.get("schema", {"type": "object", "properties": {}}),
            requires_approval=bool(raw.get("requires_approval", False)),
            max_runtime_s=int(raw.get("max_runtime_s", 30)),
            sandbox=sandbox,
            returns=raw.get("returns", ""),
            produces_artifact=bool(raw.get("produces_artifact", False)),
        )

    def _bind_handlers(self) -> None:
        """Attach implementations. A manifest with no handler stays listed but
        unbound, and invoking it is a clear error rather than a mystery."""
        from .impl import HANDLERS

        with self._lock:
            for name, spec in self._tools.items():
                spec.handler = HANDLERS.get(name)

    # ---------------------------------------------------------------- lookup

    def get(self, name: str) -> ToolSpec:
        with self._lock:
            spec = self._tools.get(name)
        if spec is None:
            raise ToolNotFound(
                f"no tool named {name!r}; have {', '.join(sorted(self._tools))}"
            )
        return spec

    def all(self) -> list[ToolSpec]:
        with self._lock:
            return sorted(self._tools.values(), key=lambda t: t.name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def unbound(self) -> list[str]:
        return [t.name for t in self.all() if t.handler is None]

    def summary(self) -> dict[str, Any]:
        tools = self.all()
        return {
            "count": len(tools),
            "unbound": self.unbound(),
            "tools": [t.as_dict() for t in tools],
        }


_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ToolRegistry()
    return _registry


def reset_tool_registry() -> None:
    """Tests only."""
    global _registry
    _registry = None

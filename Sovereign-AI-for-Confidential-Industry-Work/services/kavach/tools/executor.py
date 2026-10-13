"""The tool chokepoint (§06).

Every tool invocation in Kavach goes through `ToolExecutor.invoke`. There is no
second path, and that is the point: policy, the workspace jail, approval gating
and the audit record are applied in one place, so a new tool cannot forget them.

Order of operations, and why:

    1. resolve the tool          unknown tool -> clear error, nothing else runs
    2. validate arguments        against the declared schema, before any effect
    3. inspect for escapes       does this call write outside the workspace?
    4. POLICY GATE               deny / require approval / allow  (§02.5)
    5. audit the intent          recorded BEFORE execution, so a crash mid-tool
                                 still leaves evidence the call was attempted
    6. execute
    7. audit the outcome

Step 5 matters more than it looks. If the record were written only on success, a
tool that hung or killed the process would leave no trace - and "no trace" is
indistinguishable from "never happened" to an auditor.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..audit import get_audit
from ..sovereignty import get_policy
from ..types import Label, SideEffect
from .registry import InvalidArguments, ToolError, ToolRegistry, ToolSpec, get_tool_registry
from .workspace import Workspace, WorkspaceViolation


class ApprovalRequired(ToolError):
    """The call is permitted but a human must sign it off first.

    Not an error in the failure sense - the agent should surface it and wait,
    which is what the approval gates in §05 are for.
    """

    def __init__(self, message: str, invocation: ToolInvocation):
        super().__init__(message)
        self.invocation = invocation


class PolicyDenied(ToolError):
    """The policy engine refused. This is the visible refusal, not a bug."""


@dataclass
class ToolInvocation:
    tool: str
    args: dict[str, Any]
    label: Label
    role: str
    run_id: str | None = None
    step_idx: int | None = None

    @property
    def args_sha256(self) -> str:
        """Arguments are hashed into the audit record rather than stored raw:
        they can contain document content, and the chain is not the place for
        CONFIDENTIAL text. The hash still proves what was invoked."""
        blob = json.dumps(self.args, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: float = 0.0
    approved_by: str | None = None
    sandboxed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "approved_by": self.approved_by,
            "sandboxed": self.sandboxed,
        }


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or get_tool_registry()

    # ---------------------------------------------------------------- inspect

    @staticmethod
    def _writes_outside_workspace(
        spec: ToolSpec, args: dict[str, Any], workspace: Workspace
    ) -> bool:
        """Would this call touch anything outside the run's workspace?

        Answered by asking the jail, not by pattern-matching the path - the jail
        is the only component that knows what containment means.
        """
        if spec.side_effect not in (SideEffect.WRITE, SideEffect.DELETE):
            return False
        for key in ("path", "dest", "target", "output"):
            value = args.get(key)
            if isinstance(value, str) and not workspace.contains(value):
                return True
        return False

    # ---------------------------------------------------------------- invoke

    def invoke(
        self,
        tool: str,
        args: dict[str, Any],
        workspace: Workspace,
        *,
        label: Label = Label.INTERNAL,
        role: str = "engineer",
        run_id: str | None = None,
        step_idx: int | None = None,
        approved_by: str | None = None,
        actor: str = "local",
    ) -> ToolResult:
        audit = get_audit()
        spec = self.registry.get(tool)               # 1. resolve

        try:                                          # 2. validate
            cleaned = spec.validate(args)
        except InvalidArguments as exc:
            audit.append(
                "tool.rejected", actor=actor, run_id=run_id, step_idx=step_idx,
                tool=tool, data_class=label.value, reason=str(exc),
                role=role, side_effect=spec.side_effect.value,
            )
            raise

        invocation = ToolInvocation(
            tool=tool, args=cleaned, label=label, role=role,
            run_id=run_id, step_idx=step_idx,
        )

        # 3 + 4. inspect, then gate
        outside = self._writes_outside_workspace(spec, cleaned, workspace)
        verdict = get_policy().check_tool(
            tool_name=spec.name,
            side_effect=spec.side_effect,
            label=label,
            role=role,
            writes_outside_workspace=outside,
            is_final_deliverable=spec.produces_artifact,
            declared_requires_approval=spec.requires_approval,
        )

        if not verdict.allowed:
            audit.append(
                "tool.denied", actor=actor, run_id=run_id, step_idx=step_idx,
                tool=tool, data_class=label.value,
                args_sha256=invocation.args_sha256, reason=verdict.reason,
                role=role, side_effect=spec.side_effect.value,
            )
            raise PolicyDenied(f"{tool}: {verdict.reason}")

        if verdict.requires_approval and approved_by is None:
            audit.append(
                "tool.awaiting_approval", actor=actor, run_id=run_id, step_idx=step_idx,
                tool=tool, data_class=label.value,
                args_sha256=invocation.args_sha256, reason=verdict.reason,
                role=role, side_effect=spec.side_effect.value,
            )
            raise ApprovalRequired(f"{tool}: {verdict.reason}", invocation)

        if spec.handler is None:
            raise ToolError(
                f"{tool} is declared in a manifest but has no handler bound; "
                "add it to kavach.tools.impl.HANDLERS"
            )

        # 5. record the INTENT before running anything
        audit.append(
            "tool.invoke", actor=actor, run_id=run_id, step_idx=step_idx,
            tool=tool, data_class=label.value,
            args_sha256=invocation.args_sha256,
            side_effect=spec.side_effect.value,
            role=role,
            approved_by=approved_by,
            sandboxed=spec.sandbox is not None,
        )

        # 6. execute
        started = time.perf_counter()
        try:
            data = self._call(spec, cleaned, workspace, invocation)
        except (WorkspaceViolation, ToolError, OSError, ValueError) as exc:
            elapsed = (time.perf_counter() - started) * 1000
            audit.append(
                "tool.failed", actor=actor, run_id=run_id, step_idx=step_idx,
                tool=tool, data_class=label.value,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round(elapsed, 1),
                role=role, side_effect=spec.side_effect.value,
            )
            return ToolResult(
                tool=tool, ok=False, error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed, approved_by=approved_by,
            )

        elapsed = (time.perf_counter() - started) * 1000
        sandboxed = data.get("sandboxed") if isinstance(data, dict) else None

        # 7. record the outcome
        audit.append(
            "tool.result", actor=actor, run_id=run_id, step_idx=step_idx,
            tool=tool, data_class=label.value, ok=True,
            elapsed_ms=round(elapsed, 1), sandboxed=sandboxed,
            role=role, side_effect=spec.side_effect.value,
        )

        return ToolResult(
            tool=tool, ok=True, data=data, elapsed_ms=elapsed,
            approved_by=approved_by, sandboxed=sandboxed,
        )

    @staticmethod
    def _call(
        spec: ToolSpec,
        args: dict[str, Any],
        workspace: Workspace,
        invocation: ToolInvocation,
    ) -> dict[str, Any]:
        """Inject the context a handler asks for, by parameter name.

        `label` and `role` are here for tools whose RESULT depends on who is
        asking - kb.search returns fewer passages to an operator than to an
        engineer (§07). A tool that filtered on clearance by reading a global
        would be a tool that could be called from somewhere the caller's identity
        was never established.
        """
        import inspect

        params = inspect.signature(spec.handler).parameters
        kwargs = dict(args)
        if "workspace" in params:
            kwargs["workspace"] = workspace
        if "sandbox" in params:
            kwargs["sandbox"] = spec.sandbox
        if "timeout_s" in params:
            kwargs["timeout_s"] = spec.max_runtime_s
        if "label" in params:
            kwargs["label"] = invocation.label
        if "role" in params:
            kwargs["role"] = invocation.role
        return spec.handler(**kwargs)


_executor: ToolExecutor | None = None


def get_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor


def reset_executor() -> None:
    """Tests only."""
    global _executor
    _executor = None

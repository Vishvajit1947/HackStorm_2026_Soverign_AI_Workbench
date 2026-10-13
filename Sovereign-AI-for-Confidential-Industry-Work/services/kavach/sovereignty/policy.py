"""Classification-aware policy engine (§02.5).

Binds data labels to capabilities. The engine reasons over a tool's *declared*
side-effect class rather than its name, so a tool it has never seen is still
governed - which is what makes "add tools without redesign" safe rather than scary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import get_settings
from ..types import Label, SideEffect


@dataclass
class PolicyVerdict:
    allowed: bool
    requires_approval: bool
    reason: str

    @property
    def denied(self) -> bool:
        return not self.allowed

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
        }


ALLOW = "allow"
DENY = "deny"
APPROVE = "allow_with_approval"


class PolicyEngine:
    def __init__(self, policy: dict[str, Any]):
        self.raw = policy
        self.rules: dict[str, dict[str, Any]] = policy.get("rules", {})
        self.roles: dict[str, dict[str, Any]] = policy.get("roles", {})
        self.always_approve: set[str] = set(policy.get("always_approve", []))

    # ---------------------------------------------------------------- helpers

    def _rule(self, label: Label) -> dict[str, Any]:
        return self.rules.get(label.value, {})

    def role_may_read(self, role: str, label: Label) -> PolicyVerdict:
        spec = self.roles.get(role)
        if spec is None:
            return PolicyVerdict(False, False, f"unknown role {role!r}")
        ceiling = Label(spec.get("max_label", "INTERNAL"))
        if label.rank > ceiling.rank:
            return PolicyVerdict(
                False,
                False,
                f"role {role!r} is cleared to {ceiling.value}; this run is {label.value}",
            )
        return PolicyVerdict(True, False, f"role {role!r} cleared to {ceiling.value}")

    def role_may_approve(self, role: str) -> bool:
        return bool(self.roles.get(role, {}).get("may_approve", False))

    # ---------------------------------------------------------------- tool gate

    def check_tool(
        self,
        *,
        tool_name: str,
        side_effect: SideEffect,
        label: Label,
        role: str = "engineer",
        writes_outside_workspace: bool = False,
        is_final_deliverable: bool = False,
        declared_requires_approval: bool = False,
    ) -> PolicyVerdict:
        """The single chokepoint. Every tool invocation passes through here before
        the executor is allowed to touch it."""
        rule = self._rule(label)

        clearance = self.role_may_read(role, label)
        if clearance.denied:
            return clearance

        if tool_name in set(rule.get("tools_denied", [])):
            return PolicyVerdict(
                False, False, f"{tool_name} is denied for {label.value} data by policy"
            )

        if side_effect.value in self.always_approve:
            return PolicyVerdict(
                True, True, f"{side_effect.value} always stops for a human"
            )

        if is_final_deliverable and "FINAL_DELIVERABLE" in self.always_approve:
            return PolicyVerdict(
                True, True, "final deliverable requires sign-off before release"
            )

        if writes_outside_workspace:
            setting = rule.get("fs_write_outside_workspace", APPROVE)
            if setting == DENY:
                return PolicyVerdict(
                    False,
                    False,
                    f"a {label.value} run may not write outside its workspace - "
                    "this is the visible refusal, not a bug",
                )
            if setting == APPROVE:
                return PolicyVerdict(
                    True, True, f"writing outside the workspace needs approval at {label.value}"
                )

        if declared_requires_approval:
            return PolicyVerdict(True, True, f"{tool_name} declares requires_approval")

        return PolicyVerdict(True, False, f"{side_effect.value} permitted at {label.value}")

    # ---------------------------------------------------------------- export gate

    def check_export(self, label: Label, role: str = "engineer") -> PolicyVerdict:
        clearance = self.role_may_read(role, label)
        if clearance.denied:
            return clearance
        setting = self._rule(label).get("export", APPROVE)
        if setting == DENY:
            return PolicyVerdict(False, False, f"export denied at {label.value}")
        if setting == APPROVE:
            return PolicyVerdict(True, True, f"export at {label.value} needs approval")
        return PolicyVerdict(True, False, f"export permitted at {label.value}")

    # ---------------------------------------------------------------- routing gate

    def max_tier(self, label: Label) -> int | None:
        """RESTRICTED runs may be pinned away from unverified/experimental models."""
        value = self._rule(label).get("max_tier")
        return int(value) if value is not None else None


_engine: PolicyEngine | None = None


def get_policy() -> PolicyEngine:
    global _engine
    if _engine is None:
        _engine = PolicyEngine(get_settings().policy)
    return _engine

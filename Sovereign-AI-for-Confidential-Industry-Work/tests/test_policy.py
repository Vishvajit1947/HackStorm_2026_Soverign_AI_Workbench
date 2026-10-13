"""Policy engine (§02.5). The RESTRICTED refusal is a demo beat, so it is pinned."""

from __future__ import annotations

import pytest
import yaml

from kavach.sovereignty.policy import PolicyEngine
from kavach.types import Label, SideEffect


@pytest.fixture()
def engine():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "policy.yaml").open(encoding="utf-8") as fh:
        return PolicyEngine(yaml.safe_load(fh))


# ---------------------------------------------------------------- labels

def test_label_ordering():
    assert Label.PUBLIC.rank < Label.INTERNAL.rank < Label.CONFIDENTIAL.rank < Label.RESTRICTED.rank


def test_run_inherits_strictest_label():
    assert Label.max(Label.PUBLIC, Label.RESTRICTED, Label.INTERNAL) is Label.RESTRICTED


# ---------------------------------------------------------------- the refusal

def test_restricted_cannot_write_outside_workspace(engine):
    """This is the visible refusal in the demo, not a bug."""
    verdict = engine.check_tool(
        tool_name="fs.write",
        side_effect=SideEffect.WRITE,
        label=Label.RESTRICTED,
        role="admin",
        writes_outside_workspace=True,
    )
    assert verdict.denied
    assert "may not write outside" in verdict.reason


def test_confidential_can_write_outside_with_approval(engine):
    verdict = engine.check_tool(
        tool_name="fs.write",
        side_effect=SideEffect.WRITE,
        label=Label.CONFIDENTIAL,
        writes_outside_workspace=True,
    )
    assert verdict.allowed
    assert verdict.requires_approval


def test_write_inside_workspace_is_ungated(engine):
    verdict = engine.check_tool(
        tool_name="fs.write",
        side_effect=SideEffect.WRITE,
        label=Label.CONFIDENTIAL,
        writes_outside_workspace=False,
    )
    assert verdict.allowed
    assert not verdict.requires_approval


# ---------------------------------------------------------------- always-approve

def test_delete_always_stops_for_a_human(engine):
    for label in (Label.PUBLIC, Label.RESTRICTED):
        verdict = engine.check_tool(
            tool_name="fs.delete", side_effect=SideEffect.DELETE, label=label, role="admin"
        )
        assert verdict.requires_approval, f"DELETE ungated at {label}"


def test_final_deliverable_needs_signoff(engine):
    verdict = engine.check_tool(
        tool_name="doc.write",
        side_effect=SideEffect.WRITE,
        label=Label.INTERNAL,
        is_final_deliverable=True,
    )
    assert verdict.requires_approval


# ---------------------------------------------------------------- clearance

def test_operator_cannot_touch_confidential(engine):
    verdict = engine.check_tool(
        tool_name="kb.search",
        side_effect=SideEffect.READ,
        label=Label.CONFIDENTIAL,
        role="operator",
    )
    assert verdict.denied
    assert "cleared to INTERNAL" in verdict.reason


def test_engineer_can_touch_confidential(engine):
    verdict = engine.check_tool(
        tool_name="kb.search",
        side_effect=SideEffect.READ,
        label=Label.CONFIDENTIAL,
        role="engineer",
    )
    assert verdict.allowed


def test_unknown_role_is_denied(engine):
    verdict = engine.check_tool(
        tool_name="kb.search",
        side_effect=SideEffect.READ,
        label=Label.PUBLIC,
        role="intruder",
    )
    assert verdict.denied


def test_only_some_roles_may_approve(engine):
    assert engine.role_may_approve("engineer")
    assert engine.role_may_approve("admin")
    assert not engine.role_may_approve("operator")
    assert not engine.role_may_approve("auditor")


# ---------------------------------------------------------------- unseen tools

def test_unknown_tool_is_still_governed(engine):
    """The engine reasons over the declared side-effect, not the tool's name -
    which is what makes 'add tools without redesign' safe rather than scary."""
    verdict = engine.check_tool(
        tool_name="some.brand.new.tool",
        side_effect=SideEffect.DELETE,
        label=Label.INTERNAL,
    )
    assert verdict.requires_approval

"""The tool chokepoint (§06).

Every tool invocation goes through ToolExecutor.invoke. There is no second path,
and these tests exist to keep it that way - if a tool could reach the filesystem
without passing the policy gate, the classification-aware policy would be
decoration.

The audit-ordering test matters more than it looks: the intent record is written
BEFORE execution, so a tool that hangs or kills the process still leaves evidence
it was attempted. "No trace" and "never happened" look identical to an auditor.
"""

from __future__ import annotations

import pytest

from kavach.audit import get_audit
from kavach.tools import (
    ApprovalRequired,
    InvalidArguments,
    PolicyDenied,
    ToolExecutor,
    ToolNotFound,
    Workspace,
)
from kavach.types import Label


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_exec")


@pytest.fixture()
def ex():
    return ToolExecutor()


# ---------------------------------------------------------------- resolution

def test_unknown_tool_raises(ex, ws):
    with pytest.raises(ToolNotFound):
        ex.invoke("no.such.tool", {}, ws)


# ---------------------------------------------------------------- validation

def test_missing_required_argument_is_rejected(ex, ws):
    with pytest.raises(InvalidArguments, match="missing required"):
        ex.invoke("fs.write", {"path": "a.txt"}, ws)


def test_unknown_argument_is_rejected(ex, ws):
    """A model that invents `filename` alongside `path` must be told, not
    silently ignored - silent ignores are how a run quietly does the wrong thing."""
    with pytest.raises(InvalidArguments, match="unknown argument"):
        ex.invoke("fs.write", {"path": "a.txt", "content": "x", "filename": "b"}, ws)


def test_wrong_type_is_rejected(ex, ws):
    with pytest.raises(InvalidArguments, match="expected a string"):
        ex.invoke("fs.write", {"path": 42, "content": "x"}, ws)


def test_boolean_is_not_accepted_as_integer(ex, ws):
    with pytest.raises(InvalidArguments):
        ex.invoke("fs.read", {"path": "a.txt", "max_bytes": True}, ws)


def test_validation_runs_before_any_side_effect(ex, ws):
    with pytest.raises(InvalidArguments):
        ex.invoke("fs.write", {"path": "created.txt"}, ws)
    assert not (ws.root / "created.txt").exists()


def test_rejected_arguments_are_audited(ex, ws):
    with pytest.raises(InvalidArguments):
        ex.invoke("fs.write", {"path": "a.txt"}, ws)
    assert "tool.rejected" in [r["event"] for r in get_audit().tail(5)]


# ---------------------------------------------------------------- policy gate

def test_restricted_write_outside_workspace_is_denied(ex, ws):
    """The visible refusal (§02.5), enforced at the chokepoint rather than
    inside the tool."""
    with pytest.raises(PolicyDenied, match="may not write outside"):
        ex.invoke(
            "fs.write",
            {"path": "../escape.txt", "content": "x"},
            ws,
            label=Label.RESTRICTED,
            role="admin",
        )


def test_confidential_write_outside_needs_approval(ex, ws):
    with pytest.raises(ApprovalRequired):
        ex.invoke(
            "fs.write",
            {"path": "../escape.txt", "content": "x"},
            ws,
            label=Label.CONFIDENTIAL,
        )


def test_approval_lets_it_through_the_gate(ex, ws):
    """Approved, so the policy gate passes - and the workspace jail still
    refuses, because approval is not a licence to escape the sandbox."""
    result = ex.invoke(
        "fs.write",
        {"path": "../escape.txt", "content": "x"},
        ws,
        label=Label.CONFIDENTIAL,
        approved_by="pverma",
    )
    assert result.ok is False
    assert "WorkspaceViolation" in result.error


def test_writing_inside_the_workspace_is_ungated(ex, ws):
    result = ex.invoke("fs.write", {"path": "fine.txt", "content": "x"}, ws)
    assert result.ok
    assert (ws.root / "fine.txt").exists()


def test_operator_cannot_touch_confidential(ex, ws):
    with pytest.raises(PolicyDenied, match="cleared to INTERNAL"):
        ex.invoke("fs.list", {}, ws, label=Label.CONFIDENTIAL, role="operator")


def test_denied_call_never_executes(ex, ws):
    with pytest.raises(PolicyDenied):
        ex.invoke(
            "fs.write",
            {"path": "../nope.txt", "content": "x"},
            ws,
            label=Label.RESTRICTED,
            role="admin",
        )
    assert not (ws.root.parent / "nope.txt").exists()


def test_awaiting_approval_never_executes(ex, ws):
    with pytest.raises(ApprovalRequired):
        ex.invoke(
            "fs.write",
            {"path": "../pending.txt", "content": "x"},
            ws,
            label=Label.CONFIDENTIAL,
        )
    assert not (ws.root.parent / "pending.txt").exists()


# ---------------------------------------------------------------- audit

def test_intent_is_recorded_before_execution(ex, ws):
    """tool.invoke precedes tool.result, so a tool that dies mid-call still
    leaves evidence it was attempted."""
    ex.invoke("calc", {"expression": "2+2"}, ws)
    events = [r["event"] for r in reversed(get_audit().tail(10))]
    assert events.index("tool.invoke") < events.index("tool.result")


def test_arguments_are_hashed_not_stored(ex, ws):
    """Arguments can contain CONFIDENTIAL document text. The chain records a
    digest that proves what was invoked, without holding the content."""
    secret = "the crude unit trips at 8.4 barg"
    ex.invoke("fs.write", {"path": "s.txt", "content": secret}, ws)
    record = next(r for r in get_audit().tail(10) if r["event"] == "tool.invoke")
    assert "args_sha256" in record["payload"]
    assert len(record["payload"]["args_sha256"]) == 64
    assert secret not in str(record)


def test_denial_is_audited(ex, ws):
    with pytest.raises(PolicyDenied):
        ex.invoke(
            "fs.write", {"path": "../x.txt", "content": "y"}, ws,
            label=Label.RESTRICTED, role="admin",
        )
    assert "tool.denied" in [r["event"] for r in get_audit().tail(5)]


def test_approval_wait_is_audited(ex, ws):
    with pytest.raises(ApprovalRequired):
        ex.invoke(
            "fs.write", {"path": "../x.txt", "content": "y"}, ws,
            label=Label.CONFIDENTIAL,
        )
    assert "tool.awaiting_approval" in [r["event"] for r in get_audit().tail(5)]


def test_failure_is_audited(ex, ws):
    result = ex.invoke("fs.read", {"path": "missing.txt"}, ws)
    assert result.ok is False
    assert "tool.failed" in [r["event"] for r in get_audit().tail(5)]


def test_data_class_travels_with_the_record(ex, ws):
    ex.invoke("calc", {"expression": "1+1"}, ws, label=Label.CONFIDENTIAL)
    record = next(r for r in get_audit().tail(10) if r["event"] == "tool.invoke")
    assert record["data_class"] == "CONFIDENTIAL"


def test_the_chain_survives_a_session_of_tool_calls(ex, ws):
    for i in range(6):
        ex.invoke("fs.write", {"path": f"f{i}.txt", "content": str(i)}, ws)
    assert get_audit().verify().ok


# ---------------------------------------------------------------- failures

def test_tool_failure_becomes_a_result_not_an_exception(ex, ws):
    """The agent loop (Phase 5) observes failures and replans; it should not have
    to catch an exception per tool."""
    result = ex.invoke("fs.read", {"path": "nope.txt"}, ws)
    assert result.ok is False
    assert result.error
    assert "FileNotFound" in result.error


def test_workspace_violation_becomes_a_result(ex, ws):
    result = ex.invoke("fs.read", {"path": "../../../etc/passwd"}, ws)
    assert result.ok is False
    assert "WorkspaceViolation" in result.error


def test_successful_result_carries_timing(ex, ws):
    result = ex.invoke("calc", {"expression": "2+2"}, ws)
    assert result.ok
    assert result.elapsed_ms >= 0

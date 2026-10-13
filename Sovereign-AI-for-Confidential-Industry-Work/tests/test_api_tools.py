"""L4 gateway endpoints (§06).

The status codes carry meaning the UI depends on:
    403  the policy engine refused          -> show the refusal
    428  legitimate, needs human sign-off   -> show the approval gate
    422  the model produced bad arguments   -> the agent should retry
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------- listing

def test_tools_are_listed_with_their_gates(client):
    body = client.get("/api/tools").json()
    assert body["count"] == 7
    names = {t["name"] for t in body["tools"]}
    assert names == {
        "calc", "code.exec", "doc.write", "fs.read", "fs.list", "fs.write",
        "kb.search",
    }


def test_code_exec_advertises_a_networkless_sandbox(client):
    body = client.get("/api/tools").json()
    entry = next(t for t in body["tools"] if t["name"] == "code.exec")
    assert entry["sandboxed"] is True
    assert entry["network"] == "none"


def test_no_tool_ships_unbound(client):
    assert client.get("/api/tools").json()["unbound"] == []


def test_sandbox_status_is_honest_about_readiness(client):
    body = client.get("/api/tools/sandbox").json()
    assert set(body) >= {"docker_available", "image_available", "ready", "detail"}
    if not body["ready"]:
        assert "refuse" in body["detail"]


def test_reload_is_idempotent(client):
    body = client.post("/api/tools/reload").json()
    assert body["count"] == 7
    assert body["added"] == [] and body["removed"] == []


# ---------------------------------------------------------------- grammar

def test_grammar_endpoint_pins_the_tool_name(client):
    body = client.get("/api/tools/calc/grammar").json()
    assert '"\\"calc\\""' in body["grammar"]


def test_grammar_for_unknown_tool_is_404(client):
    assert client.get("/api/tools/nope/grammar").status_code == 404


# ---------------------------------------------------------------- invoke

def test_calc_returns_the_working(client):
    body = client.post("/api/tools/invoke", json={
        "tool": "calc",
        "args": {"expression": "rho*v*d/mu",
                 "substitutions": "rho=998,v=1.2,d=0.1524,mu=0.001"},
    }).json()
    assert body["ok"] is True
    assert body["data"]["result"] == pytest.approx(182514.24, rel=1e-6)
    assert len(body["data"]["steps"]) >= 3


def test_fs_write_then_read(client):
    client.post("/api/tools/invoke", json={
        "tool": "fs.write",
        "args": {"path": "api-test.txt", "content": "written over http"},
        "run_id": "api_test",
    })
    body = client.post("/api/tools/invoke", json={
        "tool": "fs.read", "args": {"path": "api-test.txt"}, "run_id": "api_test",
    }).json()
    assert body["data"]["content"] == "written over http"


def test_unknown_tool_is_404(client):
    resp = client.post("/api/tools/invoke", json={"tool": "nope", "args": {}})
    assert resp.status_code == 404


def test_bad_arguments_are_422(client):
    resp = client.post("/api/tools/invoke", json={
        "tool": "fs.write", "args": {"path": "x.txt"},
    })
    assert resp.status_code == 422
    assert "missing required" in resp.json()["detail"]


def test_policy_denial_is_403(client):
    """The visible refusal, over HTTP."""
    resp = client.post("/api/tools/invoke", json={
        "tool": "fs.write",
        "args": {"path": "../escape.txt", "content": "x"},
        "label": "RESTRICTED",
        "role": "admin",
    })
    assert resp.status_code == 403
    assert "may not write outside" in resp.json()["detail"]


def test_approval_gate_is_428(client):
    resp = client.post("/api/tools/invoke", json={
        "tool": "fs.write",
        "args": {"path": "../needs-approval.txt", "content": "x"},
        "label": "CONFIDENTIAL",
    })
    assert resp.status_code == 428
    detail = resp.json()["detail"]
    assert detail["tool"] == "fs.write"
    assert len(detail["args_sha256"]) == 64
    assert "approved_by" in detail["hint"]


def test_workspace_escape_is_a_failed_result_not_a_crash(client):
    body = client.post("/api/tools/invoke", json={
        "tool": "fs.read", "args": {"path": "../../../etc/passwd"},
    }).json()
    assert body["ok"] is False
    assert "WorkspaceViolation" in body["error"]


def test_calc_refuses_code_execution_over_http(client):
    """sympify() would otherwise evaluate this. calc is not sandboxed, so the
    AST gate is the only thing between a prompt-injected expression and the host."""
    body = client.post("/api/tools/invoke", json={
        "tool": "calc", "args": {"expression": "__import__('os').system('echo x')"},
    }).json()
    assert body["ok"] is False
    assert "UnsafeExpression" in body["error"]


# ---------------------------------------------------------------- probe

def test_sandbox_probe_reports_honestly(client):
    """Without a sandbox the probe must not claim anything was proven."""
    body = client.post("/api/tools/sandbox/probe").json()
    if not body["sandboxed"]:
        assert body["ok"] is False or "NOT SANDBOXED" in body["headline"]
    else:
        assert body["reached_network"] is False


# ---------------------------------------------------------------- audit

def test_tool_calls_reach_the_audit_chain(client):
    client.post("/api/tools/invoke", json={
        "tool": "calc", "args": {"expression": "1+1"},
    })
    events = [r["event"] for r in client.get("/api/audit/tail?n=20").json()["records"]]
    assert "tool.invoke" in events
    assert "tool.result" in events
    assert client.get("/api/audit/verify").json()["ok"] is True

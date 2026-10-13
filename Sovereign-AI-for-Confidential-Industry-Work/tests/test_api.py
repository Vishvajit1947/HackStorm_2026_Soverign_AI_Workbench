"""Phase 1 gateway smoke tests.

These exist mostly to catch the class of bug that bit us once already: request
models defined inside create_app() cannot be resolved by FastAPI under
`from __future__ import annotations`, and silently degrade from body params into
query params. A 422 in these tests usually means someone moved a model back inside.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------------------- health

def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["profile"] == "MID"


def test_sovereignty_panel_has_everything_the_ui_renders(client):
    body = client.get("/api/sovereignty").json()
    for key in ("external_packets", "blocked_packets", "sentinel", "audit", "profile"):
        assert key in body, f"dashboard is missing {key}"
    assert body["audit"]["verified"] is True


def test_boot_is_audited(client):
    events = [r["event"] for r in client.get("/api/audit/tail?n=50").json()["records"]]
    assert "system.boot" in events


# ---------------------------------------------------------------- policy

def test_policy_check_accepts_a_json_body(client):
    """The regression test for the body-vs-query bug."""
    resp = client.post(
        "/api/policy/check",
        json={
            "tool_name": "fs.write",
            "side_effect": "WRITE",
            "label": "RESTRICTED",
            "role": "admin",
            "writes_outside_workspace": True,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["allowed"] is False


def test_restricted_refusal_is_visible(client):
    body = client.post(
        "/api/policy/check",
        json={
            "tool_name": "fs.write",
            "side_effect": "WRITE",
            "label": "RESTRICTED",
            "role": "admin",
            "writes_outside_workspace": True,
        },
    ).json()
    assert body["allowed"] is False
    assert "RESTRICTED" in body["reason"]


def test_confidential_escalates_to_approval(client):
    body = client.post(
        "/api/policy/check",
        json={
            "tool_name": "fs.write",
            "side_effect": "WRITE",
            "label": "CONFIDENTIAL",
            "writes_outside_workspace": True,
        },
    ).json()
    assert body["allowed"] is True
    assert body["requires_approval"] is True


def test_policy_matrix_is_readable(client):
    body = client.get("/api/policy").json()
    assert "RESTRICTED" in body["rules"]
    assert "DELETE" in body["always_approve"]


# ---------------------------------------------------------------- canary

def test_canary_never_claims_success_on_a_timeout(client):
    """A timeout is not proof. If this ever starts returning ok=True on a dev box
    with no firewall, the canary has become a decoration."""
    body = client.post("/api/sovereignty/canary", json={"timeout_s": 0.4}).json()
    assert body["ok"] is (body["conclusive"] and body["ok"])
    for probe in body["probes"]:
        assert probe["verdict"] in {"BLOCKED", "INCONCLUSIVE", "REACHED"}
        if probe["verdict"] == "INCONCLUSIVE":
            assert body["conclusive"] is False


def test_canary_is_audited(client):
    client.post("/api/sovereignty/canary", json={"timeout_s": 0.4})
    events = [r["event"] for r in client.get("/api/audit/tail?n=20").json()["records"]]
    assert "canary.start" in events
    assert "canary.result" in events


# ---------------------------------------------------------------- audit

def test_audit_chain_survives_a_session_of_api_calls(client):
    for _ in range(5):
        client.get("/api/sovereignty")
        client.post("/api/policy/check", json={"tool_name": "calc", "side_effect": "COMPUTE"})
    result = client.get("/api/audit/verify").json()
    assert result["ok"] is True
    assert result["records"] > 0


# --------------------------------------------- capabilities, before they are asked


def test_health_reports_what_this_box_can_actually_do(client):
    """The only way to learn that code execution was unavailable used to be to
    ask for it and watch a task fail four minutes later."""
    caps = client.get("/api/health").json()["capabilities"]
    for key in ("code_execution", "sandbox", "model_weights", "text_recognition",
                "knowledge_base", "audit_chain"):
        assert key in caps, f"capabilities does not report {key}"


def test_a_missing_sandbox_does_not_make_the_gateway_unhealthy():
    """Kavach is fully operational without Docker - the knowledge base, OCR,
    routing, policy, audit, documents and the sovereignty proofs do not touch
    it. Marking the whole gateway unhealthy over one unavailable tool is a
    degraded capability dressed up as a dead system."""
    from unittest import mock

    from fastapi.testclient import TestClient

    from kavach.api.app import create_app

    with (
        mock.patch("kavach.tools.impl.sandbox.docker_available", return_value=False),
        TestClient(create_app()) as c,
    ):
        body = c.get("/api/health").json()
    assert body["ok"] is True
    assert body["capabilities"]["code_execution"] is False


def test_an_unavailable_capability_says_how_to_fix_it(client):
    """A red cross and nothing else moves the question from "what is broken" to
    "what do I do", which is not progress."""
    caps = client.get("/api/capabilities").json()
    for key in ("sandbox", "model_weights", "text_recognition"):
        entry = caps[key]
        if not entry["available"]:
            assert entry.get("fix"), f"{key} is unavailable and suggests nothing"


def test_the_sandbox_runtime_is_named_only_when_it_is_usable():
    """`runtime: "docker"` beside `available: false` reads as "Docker is the
    runtime and it is fine", which is how a capability panel becomes noise."""
    from unittest import mock

    from fastapi.testclient import TestClient

    from kavach.api.app import create_app

    with (
        mock.patch("kavach.tools.impl.sandbox.docker_available", return_value=False),
        TestClient(create_app()) as c,
    ):
        sandbox = c.get("/api/capabilities").json()["sandbox"]
    assert sandbox["available"] is False
    assert sandbox["runtime"] is None
    assert sandbox["image"] is None

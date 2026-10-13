"""L5 gateway endpoints - runs, timeline and approval (§05)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def start(client, **kwargs):
    payload = {"goal": "Calculate the pressure drop and write it to a file"}
    payload.update(kwargs)
    return client.post("/api/runs", json=payload).json()


# ---------------------------------------------------------------- start

def test_a_run_plans_and_executes(client):
    body = start(client)
    assert body["status"] == "done"
    assert body["steps"]
    assert all(s["status"] == "passed" for s in body["steps"])


def test_plan_only_does_not_execute(client):
    body = start(client, execute=False)
    assert body["status"] == "running"
    assert all(s["status"] == "pending" for s in body["steps"])


def test_the_timeline_carries_what_the_panel_renders(client):
    body = start(client)
    step = body["steps"][0]
    for key in ("idx", "intent", "tool", "args", "acceptance", "status",
                "attempts", "requires_approval", "observations"):
        assert key in step, f"panel cannot render without {key}"


def test_budget_is_reported(client):
    body = start(client, max_steps=6)
    assert body["budget"]["max_steps"] == 6
    assert body["budget"]["steps_used"] >= 1


def test_empty_goal_is_rejected(client):
    assert client.post("/api/runs", json={"goal": ""}).status_code == 422


# ---------------------------------------------------------------- budgets

def test_a_tight_budget_stops_the_run(client):
    body = start(client, max_steps=1)
    assert body["status"] == "budget_exhausted"
    assert "budget" in body["error"]


# ---------------------------------------------------------------- reading

def test_run_is_retrievable_by_id(client):
    run_id = start(client)["run_id"]
    assert client.get(f"/api/runs/{run_id}").json()["run_id"] == run_id


def test_timeline_endpoint_matches(client):
    run_id = start(client)["run_id"]
    assert client.get(f"/api/runs/{run_id}/timeline").json()["run_id"] == run_id


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/run_nope").status_code == 404
    assert client.get("/api/runs/run_nope/timeline").status_code == 404


def test_runs_are_listed(client):
    start(client)
    body = client.get("/api/runs").json()
    assert body["runs"]
    assert "goal" in body["runs"][0]


def test_artifacts_are_listed_with_digests(client):
    run_id = start(client)["run_id"]
    body = client.get(f"/api/runs/{run_id}/artifacts").json()
    assert body["artifacts"]
    assert len(body["artifacts"][0]["sha256"]) == 64


# ---------------------------------------------------------------- approval

def test_a_gated_run_pauses_and_resumes_over_http(client):
    """The demo beat: the run stops, a human signs off, it continues."""
    body = start(client, execute=False)
    run_id = body["run_id"]

    # Gate the write step directly in the store, as a policy decision would.
    from kavach.agent import get_run_store

    store = get_run_store()
    run = store.load(run_id)
    target = next(s for s in run.steps if s.tool == "fs.write")
    target.requires_approval = True
    store.save_step(run_id, target)

    paused = client.post(f"/api/runs/{run_id}/advance", json={"approvals": {}}).json()
    assert paused["status"] == "awaiting_approval"

    resumed = client.post(
        f"/api/runs/{run_id}/approve",
        json={"step_idx": target.idx, "approved_by": "pverma"},
    ).json()
    assert resumed["status"] == "done"


def test_approving_a_missing_step_is_404(client):
    run_id = start(client, execute=False)["run_id"]
    resp = client.post(
        f"/api/runs/{run_id}/approve", json={"step_idx": 99, "approved_by": "x"}
    )
    assert resp.status_code == 404


def test_approve_requires_a_name(client):
    """An approval nobody can be identified with is not an approval."""
    run_id = start(client, execute=False)["run_id"]
    resp = client.post(
        f"/api/runs/{run_id}/approve", json={"step_idx": 0, "approved_by": ""}
    )
    assert resp.status_code == 422


def test_advance_is_idempotent_on_a_finished_run(client):
    run_id = start(client)["run_id"]
    body = client.post(f"/api/runs/{run_id}/advance", json={"approvals": {}}).json()
    assert body["status"] == "done"


# ---------------------------------------------------------------- audit

def test_a_run_is_fully_audited(client):
    start(client)
    events = [r["event"] for r in client.get("/api/audit/tail?n=80").json()["records"]]
    for expected in ("run.start", "plan.accepted", "tool.invoke", "step.judged",
                     "run.done"):
        assert expected in events, f"{expected} missing from the chain"
    assert client.get("/api/audit/verify").json()["ok"] is True

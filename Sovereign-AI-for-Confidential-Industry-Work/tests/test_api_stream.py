"""The SSE run stream and the UI mount (§09).

The contract the panel is written against, pinned here because it is a contract
between two files that are never edited together:

    snapshot first, always      -> a late client is correct immediately
    stream_end is terminal      -> a quiet connection is not a thinking model
    the stream carries nothing  -> if SSE is unavailable the panel polls
    the timeline does not          /timeline and loses smoothness, not truth
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kavach.agent import get_broker
from kavach.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def read_events(response) -> list[tuple[str, dict]]:
    """Parse an `text/event-stream` body into (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    name = None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            payload = line.split(":", 1)[1].strip()
            try:
                events.append((name, json.loads(payload)))
            except json.JSONDecodeError:
                events.append((name, {}))
            name = None
    return events


def finished_run(client) -> str:
    body = client.post("/api/runs", json={
        "goal": "Calculate the pressure drop and write it to a report file",
        "execute": True,
    }).json()
    return body["run_id"]


# ---------------------------------------------------------------- the stream

def test_a_stream_opens_with_the_whole_timeline(client):
    """A client that connects late must be correct immediately, not eventually."""
    run_id = finished_run(client)
    events = read_events(client.get(f"/api/runs/{run_id}/stream"))

    assert events, "the stream produced nothing"
    kind, payload = events[0]
    assert kind == "snapshot"
    assert payload["run_id"] == run_id
    assert "steps" in payload


def test_a_finished_run_closes_its_stream_rather_than_hanging(client):
    run_id = finished_run(client)
    events = read_events(client.get(f"/api/runs/{run_id}/stream"))
    assert events[-1][0] == "stream_end"


def test_the_stream_replays_what_the_broker_still_remembers(client):
    """A run started through the broker keeps its events, so a panel that
    connects after the first two steps still sees them."""
    body = client.post("/api/runs", json={
        "goal": "Calculate the pressure drop and write it to a report file",
        "execute": True, "stream": True,
    }).json()
    run_id = body["run_id"]
    assert get_broker().join(run_id, timeout=60)

    kinds = [k for k, _ in read_events(client.get(f"/api/runs/{run_id}/stream"))]
    assert kinds[0] == "snapshot"
    assert "replay" in kinds


def test_a_run_executed_synchronously_still_streams_correctly(client):
    """The synchronous POST /api/runs path never touches the broker, so there is
    no history to replay - and the snapshot alone is still the whole truth.
    That is the property that makes the stream safe to be lossy."""
    run_id = finished_run(client)
    events = read_events(client.get(f"/api/runs/{run_id}/stream"))

    assert [k for k, _ in events] == ["snapshot", "stream_end"]
    assert events[0][1]["steps"]


def test_streaming_an_unknown_run_is_a_404(client):
    assert client.get("/api/runs/run_nope/stream").status_code == 404


def test_the_stream_carries_nothing_the_timeline_does_not(client):
    """If SSE is blocked or unsupported the panel polls /timeline. That fallback
    is only honest if the two agree."""
    run_id = finished_run(client)
    snapshot = read_events(client.get(f"/api/runs/{run_id}/stream"))[0][1]
    timeline = client.get(f"/api/runs/{run_id}/timeline").json()

    for key in ("run_id", "goal", "status", "label", "steps", "artifacts"):
        assert snapshot[key] == timeline[key]


# ---------------------------------------------------------------- liveness

def test_live_reports_whether_a_thread_is_actually_running(client):
    run_id = finished_run(client)
    body = client.get(f"/api/runs/{run_id}/live").json()
    assert body["run_id"] == run_id
    assert body["live"] is False


def test_live_on_an_unknown_run_is_a_404(client):
    assert client.get("/api/runs/run_nope/live").status_code == 404


# ---------------------------------------------------------------- start/advance

def test_starting_with_stream_returns_as_soon_as_the_plan_is_accepted(client):
    body = client.post("/api/runs", json={
        "goal": "Calculate the pressure drop and write it to a report file",
        "execute": True, "stream": True,
    }).json()

    assert body["run_id"]
    assert "live" in body        # the snapshot shape, not the plain timeline
    get_broker().join(body["run_id"], timeout=60)


def test_advancing_a_run_that_is_already_executing_is_a_409(client, monkeypatch):
    """Not a silent no-op: a caller told 'fine' would never learn why its second
    click did nothing."""
    import threading

    from kavach.agent import broker as broker_module

    release = threading.Event()
    started = threading.Event()
    run_id = finished_run(client)

    real_advance = broker_module.RunBroker.advance
    live = {"value": True}

    def fake_is_running(self, rid):
        return live["value"] and rid == run_id

    monkeypatch.setattr(broker_module.RunBroker, "is_running", fake_is_running)

    def fake_advance(self, rid, **kwargs):
        if rid == run_id and live["value"]:
            return False
        return real_advance(self, rid, **kwargs)

    monkeypatch.setattr(broker_module.RunBroker, "advance", fake_advance)

    response = client.post(f"/api/runs/{run_id}/advance", json={"background": True})
    assert response.status_code == 409
    assert "already executing" in response.json()["detail"]

    _ = (release, started)


def test_advancing_an_unknown_run_in_the_background_is_a_404(client):
    response = client.post("/api/runs/run_nope/advance", json={"background": True})
    assert response.status_code == 404


# ---------------------------------------------------------------- the UI mount

def test_the_panel_is_served_from_this_box(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # utf-8 explicitly: the timeline draws its step marks with box glyphs, and a
    # mis-declared charset turns them into mojibake on the one screen a judge
    # is looking at.
    assert "charset=utf-8" in response.headers["content-type"]
    assert "Sovereign Workbench" in response.text


def test_the_ui_assets_are_same_origin(client):
    for asset in ("/app.js", "/charts.js", "/app.css"):
        response = client.get(asset)
        assert response.status_code == 200, asset
        assert "charset=utf-8" in response.headers["content-type"], asset


def test_every_view_the_nav_offers_exists(client):
    """The nav switches views by id. A nav button pointing at a view that is not
    in the page is a dead click with nothing on screen to explain it."""
    import re

    html = client.get("/").text
    offered = set(re.findall(r'data-view="([^"]+)"', html))
    present = set(re.findall(r'id="view-([^"]+)"', html))

    assert offered, "the nav offers no views at all"
    assert offered <= present, f"nav points at missing views: {offered - present}"
    assert present <= offered, f"views no nav item reaches: {present - offered}"


def test_the_ui_mount_does_not_shadow_the_api(client):
    """A StaticFiles mount at "/" matches everything the routes above did not.
    Registered too early it would swallow the entire API."""
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/sovereignty").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_a_missing_ui_path_is_a_404_not_the_index(client):
    """html=True must not serve index.html for an unknown API path, or a typo in
    a fetch URL comes back as 200 and a page of HTML."""
    assert client.get("/api/does-not-exist").status_code == 404


def test_the_gateway_reports_whether_the_panel_shipped_with_it(client):
    body = client.get("/api/ui").json()
    assert body["present"] is True
    assert "index.html" in body["assets"]

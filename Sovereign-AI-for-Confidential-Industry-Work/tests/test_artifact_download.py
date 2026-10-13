"""A deliverable you cannot open is a claim, not a deliverable.

The panel could name an approval note and prove its digest while giving no way
to read it - the reader had to go digging in var/workspaces to check. These
tests cover the route that hands back the bytes, and the more important half:
that the route did not become a new way out of the workspace jail.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app
from kavach.config import get_settings


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def run_with_file(client):
    """A real run id with a real deliverable in its workspace."""
    body = client.post("/api/runs", json={"goal": "make a note", "execute": False}).json()
    run_id = body.get("run_id") or body.get("id")
    ws = get_settings().workspace_root / run_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "approval_note.docx").write_bytes(b"PK\x03\x04 pretend docx")
    (ws / "sub").mkdir(exist_ok=True)
    (ws / "sub" / "nested.txt").write_text("nested", encoding="utf-8")
    return run_id


def test_a_deliverable_can_actually_be_downloaded(client, run_with_file):
    r = client.get(f"/api/runs/{run_with_file}/artifacts/approval_note.docx")
    assert r.status_code == 200, r.text
    assert r.content == b"PK\x03\x04 pretend docx"


def test_word_is_served_as_word_not_octet_stream(client, run_with_file):
    """A .docx served as application/octet-stream downloads as something the
    shell will not open."""
    r = client.get(f"/api/runs/{run_with_file}/artifacts/approval_note.docx")
    assert "wordprocessingml" in r.headers["content-type"], r.headers["content-type"]


def test_a_nested_path_still_resolves(client, run_with_file):
    r = client.get(f"/api/runs/{run_with_file}/artifacts/sub/nested.txt")
    assert r.status_code == 200
    assert r.text == "nested"


@pytest.mark.parametrize("escape", [
    "../../../../windows/win.ini",
    "..%2f..%2fetc%2fpasswd",
    "sub/../../escape.txt",
])
def test_the_download_route_is_not_a_way_out_of_the_jail(client, run_with_file, escape):
    r = client.get(f"/api/runs/{run_with_file}/artifacts/{escape}")
    assert r.status_code in (400, 404), f"traversal returned {r.status_code}"
    assert b"win.ini" not in r.content.lower()


def test_a_missing_file_is_a_clear_404(client, run_with_file):
    r = client.get(f"/api/runs/{run_with_file}/artifacts/nope.docx")
    assert r.status_code == 404
    assert "workspace" in r.json()["detail"]


def test_an_unknown_run_is_404(client):
    r = client.get("/api/runs/run_doesnotexist/artifacts/x.docx")
    assert r.status_code == 404


def test_the_listing_route_still_works(client, run_with_file):
    """The new path parameter must not shadow the plain listing."""
    r = client.get(f"/api/runs/{run_with_file}/artifacts")
    assert r.status_code == 200
    assert "artifacts" in r.json()

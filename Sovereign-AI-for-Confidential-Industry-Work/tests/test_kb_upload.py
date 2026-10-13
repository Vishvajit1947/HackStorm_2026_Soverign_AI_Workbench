"""The operator can put a document into the corpus from the browser.

/ingest takes a server-side path and says there is no upload path "because there
is nowhere for an upload to come from". That holds for a remote uploader, and the
reasoning still stands - but the gateway binds loopback, so the only browser that
can reach this endpoint is one on this machine. Carrying the file across the room
on a USB stick and typing a path is the same trust boundary with more friction.

An upload is hostile input by definition, so most of this file is about the name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app
from kavach.config import get_settings


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _upload(client, name: str, data: bytes = b"Pump P-101A trip setpoint is 0.8 barg.\n"):
    return client.post("/api/kb/upload", files={"file": (name, data, "text/plain")})


def test_a_document_can_be_uploaded_and_is_ingested(client):
    r = _upload(client, "site_note.txt")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["uploaded"]["filename"] == "site_note.txt"
    assert len(body["uploaded"]["sha256"]) == 64
    assert (get_settings().var_dir / "corpus_inbox" / "site_note.txt").exists()


@pytest.mark.parametrize("hostile", [
    "../../../../config/models.yaml",
    r"..\..\windows\win.ini",
    "/etc/passwd",
])
def test_a_hostile_filename_lands_in_the_inbox_as_a_bare_name(client, hostile):
    """The upload must not be able to choose where it is written."""
    inbox = get_settings().var_dir / "corpus_inbox"
    before = set(inbox.glob("*")) if inbox.exists() else set()
    r = _upload(client, hostile)
    # Either refused for its extension, or written - but never outside the inbox.
    assert r.status_code in (200, 415, 422), r.text
    written = (set(inbox.glob("*")) - before) if inbox.exists() else set()
    for path in written:
        assert path.parent == inbox, f"escaped to {path}"
        assert ".." not in path.name


def test_a_format_this_box_cannot_read_is_refused_with_the_list(client):
    r = _upload(client, "design.dwg", b"binary")
    assert r.status_code == 415
    assert ".pdf" in r.json()["detail"]


def test_an_empty_upload_is_refused(client):
    r = _upload(client, "empty.txt", b"")
    assert r.status_code == 422


def test_a_nameless_upload_is_refused(client):
    r = _upload(client, "")
    assert r.status_code in (415, 422)


def test_the_upload_is_recorded_in_the_audit_chain(client):
    _upload(client, "audited_note.txt")
    tail = client.get("/api/audit/tail?n=30").json()["records"]
    assert any(rec["event"] == "kb.upload" for rec in tail), \
        "an upload that leaves no audit record is not evidence of anything"
    assert client.get("/api/audit/verify").json()["ok"] is True


# ------------------------------------------- a long read must not block the box


def test_a_background_upload_returns_immediately_with_a_job(client):
    """Reading a scanned page can take minutes. Done inside the request it
    occupies uvicorn's only event loop, so /health and every panel stop
    answering and the UI correctly reports the gateway as unreachable. One slow
    document must not take the workbench down."""
    response = client.post(
        "/api/kb/upload?background=1",
        files={"file": ("async_note.txt", b"pump P-101A was inspected", "text/plain")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["id"].startswith("job_")
    assert body["uploaded"]["filename"] == "async_note.txt"


def test_the_job_can_be_watched_to_completion(client):
    """The panel polls this. It must reach a terminal state and carry the
    report, or the spinner spins forever."""
    import time

    job_id = client.post(
        "/api/kb/upload?background=1",
        files={"file": ("watch_me.txt", b"suction pressure trip at 2.1 barg",
                        "text/plain")},
    ).json()["job"]["id"]

    for _ in range(120):
        job = client.get(f"/api/kb/jobs/{job_id}").json()
        if not job["running"]:
            break
        time.sleep(0.25)

    assert job["running"] is False, "the ingest job never finished"
    assert job["stage"] == "done", job
    assert job["result"], "a finished job carries no report"
    assert job["elapsed_s"] >= 0


def test_the_synchronous_path_is_unchanged(client):
    """`kavach kb ingest`, every script and every other test in this file call
    it. Adding the browser's path must not move it."""
    body = client.post(
        "/api/kb/upload",
        files={"file": ("sync_note.txt", b"still returns the report", "text/plain")},
    ).json()
    # The IngestReport itself, in the response - not a job id to go and poll.
    assert "uploaded" in body
    assert "ingested" in body and "unreadable_pages" in body
    assert "job" not in body


def test_an_unknown_job_says_where_the_durable_record_is(client):
    response = client.get("/api/kb/jobs/job_nope")
    assert response.status_code == 404
    assert "audit chain" in response.json()["detail"]


def test_a_failing_ingest_becomes_a_failed_job_not_a_hung_one(monkeypatch):
    """A thread that dies silently leaves a spinner turning forever.

    The ingestor is made to raise rather than being handed a missing path: a
    missing path is a legitimately EMPTY ingest, which finishes normally with
    nothing done. What has to be pinned is the thread's exception path.
    """
    import time

    from kavach.knowledge import jobs as jobs_module
    from kavach.knowledge.jobs import IngestJobs

    class _Exploding:
        def ingest(self, *args, **kwargs):
            raise RuntimeError("the disk went away mid-read")

    monkeypatch.setattr(jobs_module, "get_ingestor", lambda: _Exploding(),
                        raising=False)
    monkeypatch.setattr("kavach.knowledge.ingest.get_ingestor",
                        lambda: _Exploding())

    jobs = IngestJobs()
    job = jobs.start(Path("anything.pdf"), label=None, force=False,
                     filename="anything.pdf")
    for _ in range(100):
        if not job.running:
            break
        time.sleep(0.05)

    assert job.running is False, "a failing ingest never reached a terminal state"
    assert job.stage == "failed"
    assert job.error and "disk went away" in job.error

"""Ingest as a job you can watch, not a request that hangs (SS 26117, L3).

Reading a document is the one thing this box does that can take minutes. A
scanned P&ID goes to a vision model, and on a mid-range card that is a model
load plus an image encode plus a transcription - several minutes for one page,
with nothing to look at.

Done synchronously inside the request, as it was, that has two consequences and
the second one is much worse than the first:

  THE BROWSER SHOWS NOTHING. A spinner that cannot say what it is waiting for,
  for four minutes, is indistinguishable from a hang. People reload, which
  starts a second ingest of the same file.

  THE WHOLE GATEWAY STOPS ANSWERING. Uvicorn runs the app on one event loop;
  a synchronous ingest occupies it, so /health, the audit console and every
  panel stop responding - and the UI correctly reports the gateway as
  unreachable, because it is. A long read of ONE document must not take the
  workbench down.

So the work runs on a thread and the request returns a job id. The job carries
a STAGE and a PAGE COUNT, not a percentage invented from a timer: "reading page
3 of 12 with the vision model" is a true statement a person can act on, and
"73%" would not be.

Jobs are in memory on purpose. They describe work in flight; the durable record
of what was ingested is the corpus and the audit chain, which is where anything
asking "what is in this box" already looks.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import Label

# Stages, in the order they happen. The UI shows them as steps, so the order
# here is the order on screen.
QUEUED = "queued"
READING = "reading"
INDEXING = "indexing"
DONE = "done"
FAILED = "failed"

STAGE_ORDER = [QUEUED, READING, INDEXING, DONE]

# How long a finished job stays readable. Long enough that a browser which
# polls every second always sees the result, short enough that a day of demos
# does not accumulate them.
KEEP_FINISHED_S = 600


@dataclass
class IngestJob:
    id: str
    filename: str
    stage: str = QUEUED
    detail: str = "waiting to start"
    page: int = 0
    pages: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def running(self) -> bool:
        return self.stage not in (DONE, FAILED)

    def as_dict(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "id": self.id,
            "filename": self.filename,
            "stage": self.stage,
            "detail": self.detail,
            "page": self.page,
            "pages": self.pages,
            "running": self.running,
            "elapsed_s": round(elapsed, 1),
            "result": self.result,
            "error": self.error,
        }


class IngestJobs:
    """Every ingest in flight, and the ones that finished recently."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running(self) -> list[IngestJob]:
        with self._lock:
            return [j for j in self._jobs.values() if j.running]

    def _sweep(self) -> None:
        cutoff = time.time() - KEEP_FINISHED_S
        for job_id, job in list(self._jobs.items()):
            if job.finished_at and job.finished_at < cutoff:
                self._jobs.pop(job_id, None)

    def start(self, path: Path, *, label: Label | None, force: bool,
              filename: str) -> IngestJob:
        job = IngestJob(id=f"job_{uuid.uuid4().hex[:8]}", filename=filename)
        with self._lock:
            self._sweep()
            self._jobs[job.id] = job

        thread = threading.Thread(
            target=self._run, args=(job, path, label, force),
            name=f"kavach-ingest-{job.id}", daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: IngestJob, path: Path, label: Label | None,
             force: bool) -> None:
        from .index import get_index
        from .ingest import get_ingestor

        def progress(page: int, pages: int, detail: str) -> None:
            job.page, job.pages, job.detail = page, pages, detail
            job.stage = READING

        try:
            job.stage = READING
            job.detail = f"reading {job.filename}"
            report = get_ingestor().ingest(
                path, label=label, force=force, on_progress=progress
            )

            job.stage = INDEXING
            job.detail = "adding it to the search index"
            result = report.as_dict()
            result["index"] = get_index().state().as_dict()

            job.result = result
            job.stage = DONE
            # The summary a person reads, assembled once here rather than in
            # the panel - the UI should not have to work out what happened.
            gaps = report.unreadable_pages
            job.detail = (
                f"{job.filename} is in the corpus"
                + (f", but {gaps} page(s) could not be read" if gaps else "")
            )
        except Exception as exc:                      # noqa: BLE001
            # A thread that dies silently leaves a job spinning forever. Any
            # failure at all becomes a FAILED job carrying its reason.
            job.stage = FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.detail = f"{job.filename} could not be read"
        finally:
            job.finished_at = time.time()


_jobs: IngestJobs | None = None


def get_ingest_jobs() -> IngestJobs:
    global _jobs
    if _jobs is None:
        _jobs = IngestJobs()
    return _jobs


def reset_ingest_jobs() -> None:
    """Tests only."""
    global _jobs
    _jobs = None

"""The live run stream (§09).

    GET /api/runs/{id}/stream    text/event-stream

What the centre panel connects to so a step turns from `>>>` to `OK` as it
happens rather than on the next poll.

The contract the UI is written against, and the reason it is safe for the stream
to be lossy:

    snapshot     first, always. The complete timeline as SQLite holds it. A
                 client that connected halfway through a run is CORRECT
                 IMMEDIATELY, not eventually.
    replay       events the broker still remembers from before you connected.
    <run events> step_started, step_passed, step_failed, awaiting_approval,
                 replanned, budget_exhausted, done, failed.
    dropped      {"count": n} - your queue overflowed. RE-FETCH THE TIMELINE.
    error        the run's thread died. The reason is in the payload.
    stream_end   terminal. Nothing further will arrive on this connection.

`stream_end` exists because a connection that simply goes quiet is
indistinguishable from a model still thinking, and a panel that cannot tell those
apart shows a spinner forever.

The stream never carries anything the timeline endpoint does not. That is
deliberate: if SSE is blocked, proxied badly, or unsupported, the panel falls back
to polling `/timeline` and loses smoothness, not correctness.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..agent import RunNotFound, get_broker, get_run_store

router = APIRouter(prefix="/api/runs", tags=["runs"])

# How long a reader waits on the queue before looking up to check whether the
# client has gone away. Short enough that closing a tab frees the thread
# promptly, long enough not to spin.
POLL_S = 0.4

# sse-starlette sends its own comment-ping on this interval. Without it an idle
# stream is dropped by the browser after ~30-60s and the panel goes dead with no
# error anywhere - the worst kind of failure, because everything looks fine.
PING_S = 15


def _payload(kind: str, detail: dict[str, Any], step_idx: int | None = None) -> str:
    return json.dumps({"kind": kind, "step_idx": step_idx, "detail": detail})


@router.get("/{run_id}/stream")
async def stream(run_id: str, request: Request) -> EventSourceResponse:
    broker = get_broker()
    try:
        snapshot = broker.snapshot(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    subscriber = broker.subscribe(run_id)
    backlog = broker.history(run_id)

    async def events():
        try:
            yield {"event": "snapshot", "data": json.dumps(snapshot)}
            for event in backlog:
                yield {
                    "event": "replay",
                    "data": _payload(event.kind, event.detail, event.step_idx),
                }

            # A run that finished before anyone connected has no thread and will
            # emit nothing. Close the stream rather than holding a connection
            # open on a run that is over.
            if not broker.is_running(run_id) and _terminal(snapshot["status"]):
                yield {"event": "stream_end", "data": _payload("stream_end", {})}
                return

            while True:
                if await request.is_disconnected():
                    return

                dropped = subscriber.take_dropped()
                if dropped:
                    yield {
                        "event": "dropped",
                        "data": _payload(
                            "dropped",
                            {
                                "count": dropped,
                                "action": "re-fetch /timeline - this stream fell "
                                "behind and is no longer complete",
                            },
                        ),
                    }

                event = await asyncio.to_thread(subscriber.next, POLL_S)
                if event is None:
                    continue

                yield {
                    "event": event.kind,
                    "data": _payload(event.kind, event.detail, event.step_idx),
                }
                if event.kind == "stream_end":
                    return
        finally:
            broker.unsubscribe(subscriber)

    return EventSourceResponse(events(), ping=PING_S)


def _terminal(status: str) -> bool:
    return status in {"done", "failed", "budget_exhausted"}


@router.get("/{run_id}/live")
def live(run_id: str) -> dict[str, Any]:
    """Is this run executing right now, and is anyone watching?

    The polling fallback needs this: without it a panel cannot tell a run parked
    on an approval gate from one whose thread is still working.
    """
    broker = get_broker()
    try:
        get_run_store().load(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "run_id": run_id,
        "live": broker.is_running(run_id),
        "watchers": broker.watcher_count(run_id),
        "buffered_events": len(broker.history(run_id)),
    }

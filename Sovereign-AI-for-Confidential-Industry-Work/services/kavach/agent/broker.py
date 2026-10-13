"""Live run events, for the SSE stream (§09).

`AgentLoop.advance` is synchronous and blocking: it plans, calls models, runs
tools and returns when the run is done or parked. That is the right shape for a
CLI and the wrong shape for a panel that has to show a step turning from `>>>` to
`OK` while it happens. The broker is the seam - it runs `advance` on a background
thread and fans its `RunEvent`s out to whoever is watching.

**The stream is an optimisation, never the source of truth.**

That sentence is the whole design. SQLite already holds every step, observation
and artifact, and `GET /api/runs/{id}/timeline` renders the run correctly with no
stream at all. So the broker is allowed to lose events - a slow browser tab, a
reconnect, a queue that filled - as long as it never lets a client believe it saw
everything when it did not:

  Every stream OPENS with a `snapshot` of the timeline, so a client that
  connected late is correct immediately rather than correct eventually.

  A subscriber whose queue overflows gets a `dropped` event naming the count.
  The panel's response is to re-fetch the timeline. Silently discarding events
  would leave a UI that is subtly, permanently wrong with nothing on screen to
  say so - the exact failure this codebase refuses everywhere else.

  A run thread that raises publishes a terminal `error` event before it dies.
  A stream that simply stops is indistinguishable from a run still thinking.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..audit import get_audit
from ..types import Budget, Label, RunStatus, WorkMode
from .loop import AgentLoop, RunEvent, get_loop
from .state import RunNotFound, get_run_store

# Per-subscriber buffer. A run emits a handful of events per step, so 256 is a
# long way beyond any real backlog - it fills only when a client has genuinely
# stopped reading, which is the case the `dropped` event exists for.
QUEUE_SIZE = 256

# Replayed to a client that connects mid-run, after the snapshot. Bounded because
# this is a live view, not the audit log; the chain has the complete record.
HISTORY_SIZE = 512


@dataclass
class Subscriber:
    """One watcher of one run."""

    run_id: str
    queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=QUEUE_SIZE))
    dropped: int = 0

    def offer(self, event: RunEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            # Count it. The reader turns this into a visible `dropped` event
            # rather than a UI that has quietly fallen behind reality.
            self.dropped += 1

    def next(self, timeout: float) -> RunEvent | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_dropped(self) -> int:
        count, self.dropped = self.dropped, 0
        return count


class RunBroker:
    """Runs agent loops on background threads and fans their events out."""

    def __init__(self, loop: AgentLoop | None = None):
        self._loop = loop
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._history: dict[str, list[RunEvent]] = {}
        self._threads: dict[str, threading.Thread] = {}

    @property
    def loop(self) -> AgentLoop:
        return self._loop or get_loop()

    # ---------------------------------------------------------------- pub/sub

    def publish(self, event: RunEvent) -> None:
        with self._lock:
            history = self._history.setdefault(event.run_id, [])
            history.append(event)
            if len(history) > HISTORY_SIZE:
                del history[: len(history) - HISTORY_SIZE]
            watchers = list(self._subscribers.get(event.run_id, ()))
        for subscriber in watchers:
            subscriber.offer(event)

    def subscribe(self, run_id: str) -> Subscriber:
        subscriber = Subscriber(run_id=run_id)
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            watchers = self._subscribers.get(subscriber.run_id, [])
            if subscriber in watchers:
                watchers.remove(subscriber)
            if not watchers:
                self._subscribers.pop(subscriber.run_id, None)

    def history(self, run_id: str) -> list[RunEvent]:
        with self._lock:
            return list(self._history.get(run_id, ()))

    def watcher_count(self, run_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(run_id, ()))

    # ---------------------------------------------------------------- lifecycle

    def is_running(self, run_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(run_id)
        return thread is not None and thread.is_alive()

    def active_runs(self) -> list[str]:
        with self._lock:
            return [rid for rid, t in self._threads.items() if t.is_alive()]

    def start(
        self,
        goal: str,
        *,
        label: Label = Label.INTERNAL,
        user: str = "local",
        budget: Budget | None = None,
        auto_approve: str | None = None,
        mode: WorkMode = WorkMode.AUTO,
    ) -> str:
        """Plan a run synchronously, then execute it on a background thread.

        Planning stays on the caller's thread on purpose: it is the step that can
        fail outright, and a caller that gets a run id back should already know
        the plan was accepted. A run that fails to plan is returned as a FAILED
        run with its reason, not as a stream that never emits anything.
        """
        run = self.loop.start(goal, label=label, user=user, budget=budget, mode=mode)
        if run.status is RunStatus.FAILED:
            return run.id
        self.advance(run.id, auto_approve=auto_approve)
        return run.id

    def advance(
        self,
        run_id: str,
        *,
        approvals: dict[int, str] | None = None,
        auto_approve: str | None = None,
    ) -> bool:
        """Resume a run on a background thread. False if it is already running.

        Refusing to start a second thread for the same run matters: two loops
        advancing one run interleave their steps and corrupt the budget. The
        caller gets False and can say "already running" rather than quietly
        doubling the work.
        """
        with self._lock:
            existing = self._threads.get(run_id)
            if existing is not None and existing.is_alive():
                return False
            thread = threading.Thread(
                target=self._run,
                args=(run_id, approvals or {}, auto_approve),
                name=f"kavach-run-{run_id}",
                daemon=True,
            )
            self._threads[run_id] = thread
        thread.start()
        return True

    def _run(
        self, run_id: str, approvals: dict[int, str], auto_approve: str | None
    ) -> None:
        emit = self.publish
        try:
            while True:
                merged = dict(approvals)
                if auto_approve:
                    run = get_run_store().load(run_id)
                    merged.update(
                        {s.idx: auto_approve for s in run.steps if s.requires_approval}
                    )
                run = self.loop.advance(run_id, approvals=merged, on_event=emit)
                if run.status is not RunStatus.AWAITING_APPROVAL or not auto_approve:
                    break
        except RunNotFound as exc:
            emit(RunEvent("error", run_id, detail={"reason": str(exc)}))
        except Exception as exc:  # noqa: BLE001 - a dead thread must still speak
            # A stream that just stops looks exactly like a run still thinking.
            # Say what happened, in-band, and put it in the chain too.
            reason = f"{type(exc).__name__}: {exc}"
            get_audit().append("run.thread_failed", run_id=run_id, error=reason)
            emit(RunEvent("error", run_id, detail={"reason": reason}))
        finally:
            emit(RunEvent("stream_end", run_id, detail={"at": time.time()}))

    # ---------------------------------------------------------------- cleanup

    def forget(self, run_id: str) -> None:
        """Drop the replay buffer for a finished run. Tests and long sessions."""
        with self._lock:
            self._history.pop(run_id, None)
            thread = self._threads.get(run_id)
            if thread is not None and not thread.is_alive():
                self._threads.pop(run_id, None)

    def join(self, run_id: str, timeout: float = 30.0) -> bool:
        """Wait for a run's thread. Tests only - the API never blocks on this."""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def snapshot(self, run_id: str) -> dict[str, Any]:
        payload = get_run_store().timeline(run_id)
        payload["live"] = self.is_running(run_id)
        payload["watchers"] = self.watcher_count(run_id)
        return payload


_broker: RunBroker | None = None
_broker_lock = threading.Lock()


def get_broker() -> RunBroker:
    global _broker
    if _broker is None:
        with _broker_lock:
            if _broker is None:
                _broker = RunBroker()
    return _broker


def reset_broker() -> None:
    """Tests only."""
    global _broker
    _broker = None

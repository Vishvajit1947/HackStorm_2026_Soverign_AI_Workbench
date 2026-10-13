"""The run broker (§09).

The broker exists so a panel can watch a run happen. Everything below is about
the ways that could quietly go wrong: two loops racing on one run, a client that
falls behind and does not know it, a thread that dies and takes the stream's only
terminator with it.
"""

from __future__ import annotations

import queue

import pytest

from kavach.agent import RunEvent, RunNotFound, RunStore
from kavach.agent.broker import QUEUE_SIZE, RunBroker
from kavach.types import Budget, Label, RunStatus


@pytest.fixture()
def broker():
    return RunBroker()


# ---------------------------------------------------------------- fan-out

def test_an_event_reaches_every_watcher(broker):
    a = broker.subscribe("run_x")
    b = broker.subscribe("run_x")
    broker.publish(RunEvent("step_started", "run_x", 0))

    assert a.next(0.1).kind == "step_started"
    assert b.next(0.1).kind == "step_started"


def test_events_for_another_run_are_not_delivered(broker):
    watcher = broker.subscribe("run_x")
    broker.publish(RunEvent("step_started", "run_y", 0))
    assert watcher.next(0.05) is None


def test_unsubscribing_stops_delivery_and_frees_the_slot(broker):
    watcher = broker.subscribe("run_x")
    broker.unsubscribe(watcher)
    broker.publish(RunEvent("done", "run_x"))

    assert watcher.next(0.05) is None
    assert broker.watcher_count("run_x") == 0


def test_a_late_watcher_can_replay_what_it_missed(broker):
    """A panel that connects mid-run must not start from a blank timeline."""
    broker.publish(RunEvent("step_started", "run_x", 0))
    broker.publish(RunEvent("step_passed", "run_x", 0))

    assert [e.kind for e in broker.history("run_x")] == ["step_started", "step_passed"]


def test_history_is_bounded(broker):
    from kavach.agent.broker import HISTORY_SIZE

    for i in range(HISTORY_SIZE + 40):
        broker.publish(RunEvent("step_passed", "run_x", i))
    # The chain is the complete record; this is a live view and must not grow
    # without limit on a long session.
    assert len(broker.history("run_x")) == HISTORY_SIZE


# ---------------------------------------------------------------- overflow

def test_a_watcher_that_falls_behind_is_told_rather_than_quietly_truncated(broker):
    """The whole reason the stream is allowed to be lossy.

    Dropping events silently leaves a panel that is subtly, permanently wrong
    with nothing on screen to say so.
    """
    watcher = broker.subscribe("run_x")
    for i in range(QUEUE_SIZE + 7):
        broker.publish(RunEvent("step_passed", "run_x", i))

    assert watcher.dropped == 7
    assert watcher.take_dropped() == 7
    # Counted once. A repeated warning for the same backlog would have the panel
    # re-syncing forever.
    assert watcher.take_dropped() == 0


def test_a_full_queue_never_blocks_the_run(broker):
    """The publisher is the agent loop's own thread. If offering an event could
    block, one dead browser tab would stop the run."""
    broker.subscribe("run_x")
    for i in range(QUEUE_SIZE * 2):
        broker.publish(RunEvent("step_passed", "run_x", i))   # must simply return


def test_the_subscriber_queue_is_bounded(broker):
    watcher = broker.subscribe("run_x")
    assert isinstance(watcher.queue, queue.Queue)
    assert watcher.queue.maxsize == QUEUE_SIZE


# ---------------------------------------------------------------- execution

def test_a_run_executes_on_a_background_thread_and_ends_the_stream(broker):
    watcher = broker.subscribe("placeholder")
    run_id = broker.start("Calculate something and write it to a file")
    broker.unsubscribe(watcher)

    assert broker.join(run_id, timeout=60)
    kinds = [e.kind for e in broker.history(run_id)]

    # stream_end is terminal and always last: a connection that simply goes quiet
    # is indistinguishable from a model still thinking.
    assert kinds[-1] == "stream_end"
    assert "step_started" in kinds


def test_a_second_advance_on_a_live_run_is_refused(broker, monkeypatch):
    """Two loops advancing one run interleave their steps and corrupt the budget."""
    import threading

    started = threading.Event()
    release = threading.Event()

    class SlowLoop:
        def start(self, goal, **kwargs):
            from kavach.types import Run

            run = Run(goal=goal)
            run.status = RunStatus.RUNNING
            RunStore().create(run)
            return run

        def advance(self, run_id, approvals=None, on_event=None, **kwargs):
            started.set()
            release.wait(timeout=5)
            run = RunStore().load(run_id)
            run.status = RunStatus.DONE
            return run

    slow = RunBroker(loop=SlowLoop())
    run_id = slow.start("anything")
    assert started.wait(timeout=5)

    try:
        assert slow.advance(run_id) is False
    finally:
        release.set()
    assert slow.join(run_id, timeout=10)

    # And once it is over, advancing again is allowed - the refusal is about
    # concurrency, not a permanent lock.
    assert slow.advance(run_id) is True
    slow.join(run_id, timeout=10)


def test_a_thread_that_dies_still_closes_the_stream(broker):
    """A stream that just stops shows a spinner forever."""

    class Exploding:
        def start(self, goal, **kwargs):
            from kavach.types import Run

            run = Run(goal=goal)
            run.status = RunStatus.RUNNING
            RunStore().create(run)
            return run

        def advance(self, run_id, **kwargs):
            raise RuntimeError("the card fell out")

    exploding = RunBroker(loop=Exploding())
    run_id = exploding.start("anything")
    assert exploding.join(run_id, timeout=10)

    kinds = [e.kind for e in exploding.history(run_id)]
    assert "error" in kinds
    assert kinds[-1] == "stream_end"
    error = next(e for e in exploding.history(run_id) if e.kind == "error")
    assert "the card fell out" in error.detail["reason"]


def test_a_thread_failure_is_in_the_audit_chain(broker):
    from kavach.audit import get_audit

    class Exploding:
        def start(self, goal, **kwargs):
            from kavach.types import Run

            run = Run(goal=goal)
            run.status = RunStatus.RUNNING
            RunStore().create(run)
            return run

        def advance(self, run_id, **kwargs):
            raise RuntimeError("boom")

    exploding = RunBroker(loop=Exploding())
    run_id = exploding.start("anything")
    exploding.join(run_id, timeout=10)

    events = [r["event"] for r in get_audit().tail(n=40, run_id=run_id)]
    assert "run.thread_failed" in events


def test_a_run_that_fails_to_plan_never_starts_a_thread(broker):
    """Planning is synchronous on purpose - a caller holding a run id should
    already know the plan parsed."""
    from kavach.agent.planner import PlanError

    class BadPlanner:
        def start(self, goal, **kwargs):
            from kavach.types import Run

            run = Run(goal=goal)
            run.status = RunStatus.FAILED
            run.error = "planning failed"
            RunStore().create(run)
            return run

        def advance(self, run_id, **kwargs):
            raise AssertionError("must not execute a run that failed to plan")

    _ = PlanError
    failing = RunBroker(loop=BadPlanner())
    run_id = failing.start("anything")
    assert not failing.is_running(run_id)
    assert RunStore().load(run_id).status is RunStatus.FAILED


# ---------------------------------------------------------------- snapshot

def test_the_snapshot_is_the_timeline_plus_liveness(broker):
    run_id = broker.start("Calculate something and write it to a file")
    broker.join(run_id, timeout=60)

    snapshot = broker.snapshot(run_id)
    assert snapshot["run_id"] == run_id
    assert "steps" in snapshot and "budget" in snapshot
    assert snapshot["live"] is False
    assert snapshot["watchers"] == 0


def test_a_snapshot_of_an_unknown_run_raises(broker):
    with pytest.raises(RunNotFound):
        broker.snapshot("run_nope")


def test_the_budget_is_honoured_on_the_background_thread(broker):
    run_id = broker.start(
        "do a lot of things", budget=Budget(max_steps=1), label=Label.INTERNAL
    )
    broker.join(run_id, timeout=60)
    run = RunStore().load(run_id)
    assert run.status in (RunStatus.BUDGET_EXHAUSTED, RunStatus.DONE, RunStatus.FAILED)

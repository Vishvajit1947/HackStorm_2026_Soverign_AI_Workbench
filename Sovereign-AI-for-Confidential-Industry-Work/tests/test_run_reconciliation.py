"""A run whose thread died must not keep presenting itself as live.

Found by the L0-L6 sweep: restarting the gateway mid-run left the run at status
"running" in SQLite forever, while GET /runs/{id}/live reported live=false - one
subsystem contradicting another, and a panel showing a spinner for a run that
will never move again.

AWAITING_APPROVAL is the deliberate exception: parking for a human is precisely
what is supposed to survive a restart.
"""

from __future__ import annotations

import pytest

from kavach.agent.state import RunStore
from kavach.store import Database
from kavach.types import Budget, Label, Run, RunStatus


@pytest.fixture()
def store(tmp_path):
    return RunStore(Database(tmp_path / "kavach.db"))


def _run(store, run_id: str, status: RunStatus) -> Run:
    run = Run(id=run_id, goal="do the thing", label=Label.INTERNAL,
              user="local", budget=Budget(), status=status, steps=[])
    store.create(run)
    return run


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.PLANNING])
def test_an_executing_run_is_failed_at_boot(store, status):
    _run(store, "run_x", status)
    assert store.reconcile_interrupted() == ["run_x"]
    after = store.load("run_x")
    assert after.status is RunStatus.FAILED
    assert "restarted" in (after.error or "")


def test_a_parked_run_survives_boot_untouched(store):
    _run(store, "run_parked", RunStatus.AWAITING_APPROVAL)
    assert store.reconcile_interrupted() == []
    assert store.load("run_parked").status is RunStatus.AWAITING_APPROVAL


@pytest.mark.parametrize("status", [RunStatus.DONE, RunStatus.FAILED,
                                    RunStatus.BUDGET_EXHAUSTED])
def test_terminal_runs_are_left_alone(store, status):
    _run(store, "run_t", status)
    assert store.reconcile_interrupted() == []
    assert store.load("run_t").status is status


def test_reconciliation_is_idempotent(store):
    _run(store, "run_y", RunStatus.RUNNING)
    assert store.reconcile_interrupted() == ["run_y"]
    assert store.reconcile_interrupted() == []

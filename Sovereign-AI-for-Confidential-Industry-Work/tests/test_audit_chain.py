"""The audit chain is on the never-cut list, so it gets the most tests.

The point of these is not that append() works - it is that VERIFICATION CATCHES
TAMPERING. A hash chain nobody has tried to break is a hash chain nobody should
believe.
"""

from __future__ import annotations

import sqlite3

import pytest

from kavach.audit.chain import GENESIS, AuditChain
from kavach.store.db import Database


@pytest.fixture()
def chain(tmp_path):
    db = Database(tmp_path / "test.db")
    return AuditChain(db)


def test_empty_chain_verifies(chain):
    result = chain.verify()
    assert result.ok
    assert result.records == 0


def test_append_links_to_genesis(chain):
    record = chain.append("system.boot", profile="MID")
    assert record["seq"] == 1
    assert record["prev_sha256"] == GENESIS
    assert len(record["sha256"]) == 64


def test_chain_of_many_verifies(chain):
    for i in range(50):
        chain.append("tool.invoke", tool="calc", step_idx=i)
    result = chain.verify()
    assert result.ok
    assert result.records == 50


def test_each_record_points_at_its_predecessor(chain):
    first = chain.append("a")
    second = chain.append("b")
    third = chain.append("c")
    assert second["prev_sha256"] == first["sha256"]
    assert third["prev_sha256"] == second["sha256"]
    assert chain.head() == third["sha256"]


# ---------------------------------------------------------------- tampering

def test_triggers_block_update(chain):
    chain.append("tool.invoke", tool="calc")
    with pytest.raises(sqlite3.IntegrityError):
        chain.db.execute("UPDATE audit SET event='benign' WHERE seq=1")


def test_triggers_block_delete(chain):
    chain.append("tool.invoke", tool="calc")
    with pytest.raises(sqlite3.IntegrityError):
        chain.db.execute("DELETE FROM audit WHERE seq=1")


def test_verification_catches_edited_payload(chain):
    """Someone with sqlite3 and the file can drop the triggers. The chain still
    catches them, which is the whole point of hashing rather than just appending."""
    for i in range(5):
        chain.append("tool.invoke", tool="calc", n=i)

    chain.db.execute("DROP TRIGGER audit_no_update")
    chain.db.execute(
        "UPDATE audit SET payload_json='{\"tool\":\"calc\",\"n\":999}' WHERE seq=3"
    )

    result = chain.verify()
    assert not result.ok
    assert result.first_broken_seq == 3
    assert "altered" in result.detail


def test_verification_catches_deleted_record(chain):
    for i in range(5):
        chain.append("tool.invoke", n=i)

    chain.db.execute("DROP TRIGGER audit_no_delete")
    chain.db.execute("DELETE FROM audit WHERE seq=3")

    result = chain.verify()
    assert not result.ok
    assert result.first_broken_seq == 4
    assert "sequence gap" in result.detail


def test_verification_catches_relinked_record(chain):
    """Re-pointing a record at a different predecessor changes its own digest,
    so you cannot splice the chain without rewriting everything after it."""
    for i in range(5):
        chain.append("tool.invoke", n=i)

    chain.db.execute("DROP TRIGGER audit_no_update")
    chain.db.execute(f"UPDATE audit SET prev_sha256='{'a' * 64}' WHERE seq=4")

    result = chain.verify()
    assert not result.ok
    assert result.first_broken_seq == 4


def test_reports_first_break_not_last(chain):
    for i in range(10):
        chain.append("e", n=i)
    chain.db.execute("DROP TRIGGER audit_no_update")
    chain.db.execute("UPDATE audit SET actor='mallory' WHERE seq=7")
    chain.db.execute("UPDATE audit SET actor='mallory' WHERE seq=9")

    result = chain.verify()
    assert result.first_broken_seq == 7


# ---------------------------------------------------------------- reading

def test_tail_returns_newest_first(chain):
    for i in range(5):
        chain.append("e", n=i)
    records = chain.tail(n=3)
    assert [r["seq"] for r in records] == [5, 4, 3]


def test_tail_filters_by_run(chain):
    chain.append("e", run_id="run_a")
    chain.append("e", run_id="run_b")
    chain.append("e", run_id="run_a")
    assert len(chain.tail(run_id="run_a")) == 2


def test_payload_survives_roundtrip(chain):
    chain.append("tool.invoke", nested={"a": [1, 2, {"b": None}]}, flag=True)
    assert chain.verify().ok
    assert chain.tail(1)[0]["payload"]["nested"]["a"][2]["b"] is None

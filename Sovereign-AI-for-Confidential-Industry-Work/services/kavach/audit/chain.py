"""Tamper-evident audit log (§02.3).

Every model call, tool invocation, file read, routing decision and approval writes
one record. Each record carries the SHA-256 of the previous record, so the log is a
chain: deleting or editing any entry breaks verification, and `kavach audit verify`
reports the first broken link.

The digest covers the canonical JSON of the record's own fields *including*
prev_sha256, so you cannot re-point a record at a different predecessor without
changing its own digest and every digest after it.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..store import Database, get_db

GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    """Stable serialisation. Key order and separators are fixed so that a record
    re-hashed on another machine, in another Python, produces the same digest."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


@dataclass
class ChainVerification:
    ok: bool
    records: int
    first_broken_seq: int | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "records": self.records,
            "first_broken_seq": self.first_broken_seq,
            "detail": self.detail,
        }


class AuditChain:
    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- append

    def append(
        self,
        event: str,
        *,
        actor: str = "local",
        run_id: str | None = None,
        step_idx: int | None = None,
        data_class: str = "INTERNAL",
        egress_observed: int = 0,
        **payload: Any,
    ) -> dict[str, Any]:
        """Append one record and return it. Never raises on payload content -
        an audit log that refuses writes is worse than one with an ugly record."""
        with self._lock, self.db.audit_lock:
            row = self.db.one("SELECT seq, sha256 FROM audit ORDER BY seq DESC LIMIT 1")
            seq = (row["seq"] + 1) if row else 1
            prev = row["sha256"] if row else GENESIS

            record = {
                "seq": seq,
                "ts": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                "actor": actor,
                "run": run_id,
                "step": step_idx,
                "event": event,
                "payload": payload,
                "data_class": data_class,
                "egress_observed": egress_observed,
                "prev_sha256": prev,
            }
            record["sha256"] = _digest(record)

            self.db.execute(
                """INSERT INTO audit (seq, ts, actor, run_id, step_idx, event,
                                      payload_json, data_class, egress_observed,
                                      prev_sha256, sha256)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    seq,
                    record["ts"],
                    actor,
                    run_id,
                    step_idx,
                    event,
                    _canonical(payload),
                    data_class,
                    egress_observed,
                    prev,
                    record["sha256"],
                ),
            )
            return record

    # ---------------------------------------------------------------- verify

    def verify(self) -> ChainVerification:
        """Walk the chain from genesis. Reports the FIRST broken link, which is
        what an auditor actually needs - everything after it is untrustworthy anyway."""
        rows = self.db.query("SELECT * FROM audit ORDER BY seq ASC")
        if not rows:
            return ChainVerification(ok=True, records=0, detail="empty chain")

        prev_hash = GENESIS
        expected_seq = 1

        for row in rows:
            if row["seq"] != expected_seq:
                return ChainVerification(
                    ok=False,
                    records=len(rows),
                    first_broken_seq=row["seq"],
                    detail=f"sequence gap: expected {expected_seq}, found {row['seq']} "
                    f"({expected_seq - 1} records survive verification)",
                )
            if row["prev_sha256"] != prev_hash:
                return ChainVerification(
                    ok=False,
                    records=len(rows),
                    first_broken_seq=row["seq"],
                    detail=f"record {row['seq']} points at {row['prev_sha256'][:12]}... "
                    f"but predecessor hashes to {prev_hash[:12]}...",
                )

            recomputed = _digest(
                {
                    "seq": row["seq"],
                    "ts": row["ts"],
                    "actor": row["actor"],
                    "run": row["run_id"],
                    "step": row["step_idx"],
                    "event": row["event"],
                    "payload": json.loads(row["payload_json"]),
                    "data_class": row["data_class"],
                    "egress_observed": row["egress_observed"],
                    "prev_sha256": row["prev_sha256"],
                }
            )
            if recomputed != row["sha256"]:
                return ChainVerification(
                    ok=False,
                    records=len(rows),
                    first_broken_seq=row["seq"],
                    detail=f"record {row['seq']} content was altered: stored "
                    f"{row['sha256'][:12]}... recomputes to {recomputed[:12]}...",
                )

            prev_hash = row["sha256"]
            expected_seq += 1

        return ChainVerification(
            ok=True, records=len(rows), detail=f"chain intact, head {prev_hash[:12]}..."
        )

    # ---------------------------------------------------------------- read

    def tail(self, n: int = 20, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            rows = self.db.query(
                "SELECT * FROM audit WHERE run_id=? ORDER BY seq DESC LIMIT ?", (run_id, n)
            )
        else:
            rows = self.db.query("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (n,))
        return [
            {
                "seq": r["seq"],
                "ts": r["ts"],
                "actor": r["actor"],
                "run": r["run_id"],
                "step": r["step_idx"],
                "event": r["event"],
                "payload": json.loads(r["payload_json"]),
                "data_class": r["data_class"],
                "egress_observed": r["egress_observed"],
                "prev_sha256": r["prev_sha256"],
                "sha256": r["sha256"],
            }
            for r in rows
        ]

    def head(self) -> str:
        row = self.db.one("SELECT sha256 FROM audit ORDER BY seq DESC LIMIT 1")
        return row["sha256"] if row else GENESIS

    def count(self) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM audit")
        return row["n"] if row else 0


_chain: AuditChain | None = None


def get_audit() -> AuditChain:
    global _chain
    if _chain is None:
        _chain = AuditChain(get_db())
    return _chain

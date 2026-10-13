"""SQLite access. One connection per thread, WAL on, schema applied at open."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import get_settings

_SCHEMA = Path(__file__).with_name("schema.sql")


class Database:
    """Thin wrapper. Deliberately not an ORM - an auditor should be able to read
    every query in this file and know exactly what hits the disk."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Serialises audit appends so two threads cannot claim the same seq.
        self.audit_lock = threading.Lock()
        self._init_schema()

    # ---------------------------------------------------------------- plumbing

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
        self._migrate()

    # Columns added to a table that already exists on disk. `CREATE TABLE IF NOT
    # EXISTS` is a no-op against an older database, so a new column in
    # schema.sql is invisible there and every read of it raises - on the one
    # machine that matters, the venue box someone has already demoed on.
    # Additive only: no column is ever dropped or retyped here, because a
    # migration that can lose a row has no business running unattended at boot.
    _ADDED_COLUMNS = (
        ("runs", "mode", "TEXT NOT NULL DEFAULT 'auto'"),
    )

    def _migrate(self) -> None:
        for table, column, decl in self._ADDED_COLUMNS:
            present = {r["name"] for r in self.query(f"PRAGMA table_info({table})")}
            if column not in present:
                self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)).fetchall())

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    # ---------------------------------------------------------------- runs

    def upsert_run(self, run: Any) -> None:
        now = time.time()
        self.execute(
            """INSERT INTO runs (id, user, goal, label, status, budget_json, error,
                                 mode, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   label=excluded.label,
                   status=excluded.status,
                   budget_json=excluded.budget_json,
                   error=excluded.error,
                   mode=excluded.mode,
                   updated_at=excluded.updated_at""",
            (
                run.id,
                run.user,
                run.goal,
                run.label.value,
                run.status.value,
                json.dumps(run.budget.as_dict()),
                run.error,
                getattr(run.mode, "value", run.mode or "auto"),
                run.created_at,
                now,
            ),
        )

    def upsert_step(self, run_id: str, step: Any) -> None:
        self.execute(
            """INSERT INTO steps (run_id, idx, intent, tool, args_json, acceptance,
                                  status, attempts, requires_approval)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id, idx) DO UPDATE SET
                   intent=excluded.intent,
                   tool=excluded.tool,
                   args_json=excluded.args_json,
                   acceptance=excluded.acceptance,
                   status=excluded.status,
                   attempts=excluded.attempts,
                   requires_approval=excluded.requires_approval""",
            (
                run_id,
                step.idx,
                step.intent,
                step.tool,
                json.dumps(step.args),
                step.acceptance,
                step.status.value,
                step.attempts,
                int(step.requires_approval),
            ),
        )

    def add_observation(self, run_id: str, obs: Any) -> None:
        self.execute(
            """INSERT INTO observations (run_id, step_idx, ok, stdout, stderr,
                                         exit_code, artifacts_json, data_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                obs.step_idx,
                int(obs.ok),
                obs.stdout[:20000],
                obs.stderr[:20000],
                obs.exit_code,
                json.dumps(obs.artifacts),
                json.dumps(obs.data, default=str),
                time.time(),
            ),
        )

    def add_artifact(self, run_id: str, path: str, kind: str, sha256: str) -> None:
        self.execute(
            """INSERT INTO artifacts (run_id, path, kind, sha256, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(run_id, path) DO UPDATE SET sha256=excluded.sha256""",
            (run_id, path, kind, sha256, time.time()),
        )

    def approve_artifact(self, run_id: str, path: str, approver: str) -> None:
        self.execute(
            "UPDATE artifacts SET approved_by=?, approved_at=? WHERE run_id=? AND path=?",
            (approver, time.time(), run_id, path),
        )

    # ---------------------------------------------------------------- decisions

    def add_decision(self, decision: Any, run_id: str | None, step_idx: int | None) -> None:
        exp = decision.explain()
        self.execute(
            """INSERT INTO decisions (run_id, step_idx, class, p, chosen, escalated_from,
                                      reason, candidates_json, elapsed_ms, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                step_idx,
                exp["class"],
                exp["p"],
                exp["chosen"],
                exp["escalated_from"],
                exp["reason"],
                json.dumps(exp["candidates"]),
                exp["elapsed_ms"],
                time.time(),
            ),
        )

    def decisions_for_run(self, run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Every routing decision this run made, oldest first.

        Oldest first on purpose: this is read as a story - the prompt arrived,
        the router classified it, a model was chosen, then each step was judged.
        Newest-first would show the ending before the beginning.
        """
        rows = self.query(
            """SELECT step_idx, class, p, chosen, escalated_from, reason,
                      candidates_json, elapsed_ms, created_at
                 FROM decisions WHERE run_id = ?
                ORDER BY id ASC LIMIT ?""",
            (run_id, limit),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["candidates"] = json.loads(item.pop("candidates_json") or "[]")
            except ValueError:
                item["candidates"] = []
            out.append(item)
        return out

    # ---------------------------------------------------------------- egress

    def record_egress(
        self, iface: str, verdict: str, proto: str = "", dest: str = "", detail: str = ""
    ) -> None:
        self.execute(
            """INSERT INTO egress (iface, verdict, proto, dest, detail, observed_at)
               VALUES (?,?,?,?,?,?)""",
            (iface, verdict, proto, dest, detail, time.time()),
        )

    def egress_counts(self) -> dict[str, int]:
        rows = self.query("SELECT verdict, COUNT(*) AS n FROM egress GROUP BY verdict")
        counts = {r["verdict"]: r["n"] for r in rows}
        return {
            "allowed": counts.get("ALLOWED", 0),
            "blocked": counts.get("BLOCKED", 0),
        }


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database(get_settings().db_path)
    return _db

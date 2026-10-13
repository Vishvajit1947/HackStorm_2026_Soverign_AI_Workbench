-- Kavach run state (§05) + the audit chain (§02.3).
-- SQLite because it is one file, needs no server, and can be handed to an auditor.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    user        TEXT NOT NULL,
    goal        TEXT NOT NULL,
    label       TEXT NOT NULL,
    status      TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    error       TEXT,
    -- Which deliverable the operator asked for: auto | code | document. Added
    -- after the first runs existed, so `_migrate` in db.py backfills it on an
    -- older database rather than leaving a column the reads would blow up on.
    mode        TEXT NOT NULL DEFAULT 'auto',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    run_id     TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    intent     TEXT NOT NULL,
    tool       TEXT,
    args_json  TEXT NOT NULL DEFAULT '{}',
    acceptance TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, idx)
);

CREATE TABLE IF NOT EXISTS observations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_idx       INTEGER NOT NULL,
    ok             INTEGER NOT NULL,
    stdout         TEXT NOT NULL DEFAULT '',
    stderr         TEXT NOT NULL DEFAULT '',
    exit_code      INTEGER NOT NULL DEFAULT 0,
    artifacts_json TEXT NOT NULL DEFAULT '[]',
    data_json      TEXT NOT NULL DEFAULT '{}',
    created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    approved_by TEXT,
    approved_at REAL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (run_id, path)
);

-- Feeds the router explainer panel, and the report appendix screenshot. (§04)
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    step_idx        INTEGER,
    class           TEXT NOT NULL,
    p               REAL NOT NULL,
    chosen          TEXT,
    escalated_from  TEXT,
    reason          TEXT NOT NULL DEFAULT '',
    candidates_json TEXT NOT NULL DEFAULT '[]',
    elapsed_ms      REAL NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL
);

-- The hash chain. seq is dense and monotonic; prev_sha256 of seq N is the
-- sha256 of seq N-1. Deleting or editing any row breaks verification. Append only:
-- the triggers below make UPDATE and DELETE fail loudly rather than silently.
CREATE TABLE IF NOT EXISTS audit (
    seq         INTEGER PRIMARY KEY,
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    run_id      TEXT,
    step_idx    INTEGER,
    event       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    data_class  TEXT NOT NULL,
    egress_observed INTEGER NOT NULL DEFAULT 0,
    prev_sha256 TEXT NOT NULL,
    sha256      TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit
BEGIN
    SELECT RAISE(ABORT, 'audit log is append-only');
END;

CREATE INDEX IF NOT EXISTS idx_audit_run ON audit(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_obs_run ON observations(run_id, step_idx);

-- Egress accounting written by the sentinel (§02.1). `allowed` must stay 0 forever;
-- `blocked` is what the canary makes tick.
CREATE TABLE IF NOT EXISTS egress (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    iface      TEXT NOT NULL,
    verdict    TEXT NOT NULL,          -- ALLOWED | BLOCKED
    proto      TEXT NOT NULL DEFAULT '',
    dest       TEXT NOT NULL DEFAULT '',
    detail     TEXT NOT NULL DEFAULT '',
    observed_at REAL NOT NULL
);

-- L3 knowledge plane (§07). The index itself (BM25 postings + dense vectors)
-- lives in var/kb/ as files; this is the durable record of WHAT was ingested,
-- HOW each page was read, and under WHICH classification - the three things a
-- citation has to be checkable against.
CREATE TABLE IF NOT EXISTS kb_documents (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    sha256      TEXT NOT NULL,
    label       TEXT NOT NULL,
    ingested_at REAL NOT NULL
);

-- One row per page, INCLUDING the pages nothing on this box could read. A gap
-- that is not recorded is a gap nobody can report.
CREATE TABLE IF NOT EXISTS kb_pages (
    doc_id     TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    page       INTEGER NOT NULL,
    extraction TEXT NOT NULL,          -- NATIVE | OCR | VLM | SIMULATED | UNREADABLE
    chars      INTEGER NOT NULL DEFAULT 0,
    detail     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id         TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    page       INTEGER NOT NULL,
    ordinal    INTEGER NOT NULL,
    section    TEXT NOT NULL DEFAULT '',
    label      TEXT NOT NULL,
    extraction TEXT NOT NULL,
    text       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_pages_doc ON kb_pages(doc_id);

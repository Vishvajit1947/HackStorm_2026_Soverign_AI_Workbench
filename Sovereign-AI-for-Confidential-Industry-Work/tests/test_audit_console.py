"""The audit console projection (SS 26117, L0/L6).

The console exists to answer four questions about anything the box did: WHO
initiated it, WHAT operation ran, under WHICH policy context, and WHAT happened.
Every one of those can be faked by a nice-looking table, so these tests check
the two properties that make it trustworthy instead of decorative:

  The vocabulary is the system's own. Not "Action completed" - `ToolExecutor.invoke`,
  `PolicyEngine.check_tool`, ALLOWED / DENIED / BLOCKED, the real SideEffect and
  Label values. A word that appears here and nowhere else in the codebase cannot
  be checked against the thing it claims to describe, so the tests below assert
  against the enums and class names themselves rather than against string
  literals copied from the console.

  An event it cannot name says so. The mapping is a hand-written table, and the
  codebase emits audit events from six layers. The first test is the one that
  matters most: it walks the source for every event name actually appended and
  fails if the table has no row for it - the same job
  test_every_element_the_script_reaches_for_exists_in_the_page does for the UI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kavach.audit.console import (
    ALLOWED,
    APPROVAL_REQUIRED,
    BLOCKED,
    DENIED,
    EVENTS,
    NA,
    project,
    project_all,
)
from kavach.types import Label, SideEffect

SOURCE = Path(__file__).resolve().parents[1] / "services" / "kavach"


def _events_emitted_by_the_codebase() -> set[str]:
    """Every event name passed to an audit append, read out of the source.

    Matched on the call rather than imported from a registry, because there is
    no registry - `append()` takes a string. That is exactly why this check has
    to exist.
    """
    found: set[str] = set()
    pattern = re.compile(r"append\(\s*\n?\s*\"([a-z_]+\.[a-z_]+)\"")
    for path in SOURCE.rglob("*.py"):
        found |= set(pattern.findall(path.read_text(encoding="utf-8")))
    return found


# ------------------------------------------------------------- the mapping


def test_every_audit_event_this_system_emits_has_a_console_mapping():
    """The drift guard.

    A new event added in any layer shows up in the console as its raw name with
    no operation, no decision and no result - three empty columns in a security
    view. This fails the build instead.
    """
    emitted = _events_emitted_by_the_codebase()
    assert emitted, "the source scan found no audit events at all - the regex broke"
    missing = sorted(emitted - set(EVENTS))
    assert missing == [], (
        f"these events are appended somewhere in services/kavach but have no "
        f"row in audit/console.py EVENTS: {missing}"
    )


def test_the_mapping_does_not_describe_events_that_no_longer_exist():
    """The other direction. A row for a deleted event is a call site the
    console would still claim to know about."""
    emitted = _events_emitted_by_the_codebase()
    stale = sorted(set(EVENTS) - emitted)
    assert stale == [], (
        f"audit/console.py maps events nothing emits any more: {stale}"
    )


def test_every_operation_names_something_that_exists_in_the_source():
    """The core promise: no invented terminology.

    Each `operation` is a real call site. If one is renamed in a refactor the
    console keeps printing the old name next to a real digest, which is worse
    than printing nothing.
    """
    source = "\n".join(
        p.read_text(encoding="utf-8") for p in SOURCE.rglob("*.py")
    )
    for event, projection in EVENTS.items():
        symbol = projection.operation.split(".")[-1]
        owner = projection.operation.split(".")[0]
        assert re.search(rf"\bdef {re.escape(symbol)}\b", source), (
            f"{event} claims operation {projection.operation!r}, but nothing "
            f"named {symbol} is defined in services/kavach"
        )
        if owner != symbol:
            assert re.search(rf"\bclass {re.escape(owner)}\b", source), (
                f"{event} claims operation {projection.operation!r}, but there "
                f"is no class {owner}"
            )


def test_every_side_effect_in_the_mapping_is_a_real_side_effect():
    real = {s.value for s in SideEffect} | {NA}
    for event, projection in EVENTS.items():
        assert projection.side_effect in real, (
            f"{event} declares side effect {projection.side_effect!r}, which is "
            "not a SideEffect this system has"
        )


def test_the_policy_decisions_are_the_three_the_policy_engine_can_return():
    """PolicyVerdict has exactly three outcomes: allowed, denied, and allowed
    with approval. A fourth value here would be a decision the engine cannot
    make."""
    for event, projection in EVENTS.items():
        assert projection.decision in (ALLOWED, DENIED, APPROVAL_REQUIRED, NA), (
            f"{event} declares decision {projection.decision!r}"
        )


# --------------------------------------------------------------- projection


def _record(event, payload=None, **over):
    base = {
        "seq": 41,
        "ts": "2026-09-11T10:00:00.000+05:30",
        "actor": "local",
        "run": None,
        "step": None,
        "event": event,
        "payload": payload or {},
        "data_class": "INTERNAL",
        "egress_observed": 0,
        "prev_sha256": "aa" * 32,
        "sha256": "bb" * 32,
    }
    base.update(over)
    return base


def test_a_denied_tool_call_renders_the_whole_policy_context():
    """The row a reader most needs to be complete. A denial that does not say
    which role, which label and which side effect was refused records that
    something was refused and nothing about why."""
    row = project(_record(
        "tool.denied",
        {"tool": "fs.write", "role": "operator", "side_effect": "WRITE",
         "reason": "CONFIDENTIAL forbids writing outside the workspace"},
        data_class="CONFIDENTIAL",
    ))
    assert row["operation"] == "ToolExecutor.invoke"
    assert row["actor"] == "local"
    assert row["role"] == "operator"
    assert row["label"] == "CONFIDENTIAL"
    assert row["side_effect"] == "WRITE"
    assert row["decision"] == DENIED
    assert row["result"] == BLOCKED
    assert row["target"] == "fs.write"


def test_an_allowed_invocation_and_its_result_are_distinguishable():
    """`tool.invoke` records the INTENT before anything runs and `tool.result`
    the outcome. Collapsing them to one status would lose the guarantee that a
    tool which hangs still left evidence it was attempted."""
    invoked = project(_record("tool.invoke", {"tool": "calc", "role": "engineer",
                                              "side_effect": "COMPUTE"}))
    finished = project(_record("tool.result", {"tool": "calc", "role": "engineer",
                                               "side_effect": "COMPUTE", "ok": True}))
    assert invoked["decision"] == finished["decision"] == ALLOWED
    assert invoked["result"] != finished["result"]
    assert invoked["result"] == "RUNNING"
    assert finished["result"] == "OK"


def test_a_policy_check_reads_its_verdict_rather_than_assuming_one():
    denied = project(_record("policy.check", {
        "tool": "fs.write", "role": "operator", "side_effect": "DELETE",
        "verdict": {"allowed": False, "requires_approval": False, "reason": "x"},
    }))
    assert denied["decision"] == DENIED and denied["result"] == BLOCKED

    gated = project(_record("policy.check", {
        "tool": "doc.write", "role": "engineer", "side_effect": "WRITE",
        "verdict": {"allowed": True, "requires_approval": True, "reason": "x"},
    }))
    assert gated["decision"] == APPROVAL_REQUIRED

    allowed = project(_record("policy.check", {
        "tool": "kb.search", "role": "auditor", "side_effect": "READ",
        "verdict": {"allowed": True, "requires_approval": False, "reason": ""},
    }))
    assert allowed["decision"] == ALLOWED


def test_an_inconclusive_canary_is_not_rounded_up_to_blocked():
    """The house rule, enforced in the console too. A timeout and an unplugged
    cable are indistinguishable, and a row that reports the second as proof of
    the first is the single most damaging thing this panel could do."""
    row = project(_record("canary.result", {"verdict": "INCONCLUSIVE", "ok": True}))
    assert row["result"] == "INCONCLUSIVE"
    assert row["result"] != BLOCKED


def test_a_simulated_model_verification_stays_simulated():
    row = project(_record("model.verify", {"model": "qwen3-8b-q4",
                                           "provenance": "SIMULATED"}))
    assert row["result"] == "SIMULATED"


def test_a_route_with_no_viable_model_is_not_reported_as_ok():
    assert project(_record("route.decide", {"chosen": None}))["result"] == "NO_VIABLE_MODEL"
    assert project(_record("route.decide", {"chosen": "qwen3-8b-q4"}))["result"] == "OK"


def test_a_missing_role_is_not_defaulted_to_engineer():
    """`engineer` is the executor's default ARGUMENT, not a fact about a record
    that never carried one. Printing it would assert a policy context the chain
    does not actually record."""
    row = project(_record("run.start", {"goal": "x"}))
    assert row["role"] == NA


def test_an_unmapped_event_keeps_its_own_name_and_is_counted():
    """No plausible-looking call site is invented to fill the column."""
    body = project_all([_record("quantum.entangled", {"spooky": True})])
    row = body["rows"][0]
    assert row["mapped"] is False
    assert row["operation"] == "quantum.entangled"
    assert row["decision"] == NA and row["result"] == NA
    assert body["unmapped"] == ["quantum.entangled"]
    assert "quantum.entangled" in body["note"]


def test_a_fully_mapped_tail_reports_no_caveat():
    body = project_all([_record("tool.invoke", {"tool": "calc", "role": "engineer",
                                                "side_effect": "COMPUTE"})])
    assert body["unmapped"] == []
    assert body["note"] is None


def test_the_row_carries_the_digest_so_it_can_be_matched_back_to_the_chain():
    """A console row that cannot be tied to the record it came from is a
    rendering of the log, not evidence from it."""
    row = project(_record("tool.result", {"tool": "calc"}))
    assert row["sha256"] == "bb" * 32
    assert row["prev_sha256"] == "aa" * 32
    assert row["seq"] == 41


def test_a_record_with_a_corrupt_payload_still_renders():
    """The chain never refuses a write, so a payload can be anything. A console
    that raises on one bad row shows nothing at all for the rows around it."""
    row = project(_record("tool.invoke", payload="not-a-dict"))
    assert row["operation"] == "ToolExecutor.invoke"
    assert row["role"] == NA


def test_every_label_the_system_has_survives_the_projection():
    for label in Label:
        row = project(_record("tool.invoke", {"tool": "calc"}, data_class=label.value))
        assert row["label"] == label.value


# ------------------------------------------------------------- the endpoint


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from kavach.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_the_console_endpoint_returns_rows_the_panel_can_render(client):
    body = client.get("/api/audit/console?n=25").json()
    assert body["rows"], "the chain is empty - boot alone should have written one"
    for row in body["rows"]:
        for column in ("ts", "actor", "role", "operation", "label",
                       "side_effect", "decision", "result", "sha256"):
            assert column in row, f"the console cannot render a row without {column}"


def test_the_live_chain_contains_no_event_the_console_cannot_name(client):
    """The static scan proves the table covers the source. This proves it covers
    what this box has actually written - including anything a dependency or an
    older build left in the chain."""
    body = client.get("/api/audit/console?n=500").json()
    assert body["unmapped"] == [], body["note"]


def test_a_real_tool_call_records_the_role_the_policy_engine_evaluated(client):
    """End to end: the executor has to PUT the role in the record, or the
    console's Role column is `-` on every row that matters."""
    client.post("/api/tools/invoke", json={
        "tool": "calc", "args": {"expression": "2+2"}, "role": "auditor"})
    body = client.get("/api/audit/console?n=40").json()
    invocations = [r for r in body["rows"]
                   if r["operation"] == "ToolExecutor.invoke" and r["target"] == "calc"]
    assert invocations, "no calc invocation reached the chain"
    assert any(r["role"] == "auditor" for r in invocations), (
        "the role never reached the audit record, so the policy context a "
        "reader sees is not the one the engine used"
    )
    assert all(r["side_effect"] == "COMPUTE" for r in invocations), (
        "calc is declared COMPUTE in its manifest"
    )


def test_a_real_route_is_not_reported_as_having_no_viable_model(client):
    """Driven through the app, not built from a fixture.

    The cascade records its winner under `model`; the projection read `chosen`,
    the name the API uses, and every successful route was therefore rendered as
    a red NO_VIABLE_MODEL row next to a real digest. A fixture written from the
    projection's own assumptions agreed with the bug - only a real record
    caught it, so this test makes one.
    """
    client.post("/api/route/explain", json={"prompt": "summarise the SOP"})
    client.post("/api/runs", json={"goal": "Calculate the pressure drop and "
                                          "write it to a file"})

    rows = client.get("/api/audit/console?n=200").json()["rows"]
    routes = [r for r in rows if r["event"] == "route.decide"]
    assert routes, "no routing decision reached the chain"

    for row in routes:
        chosen = row["payload"].get("model") or row["payload"].get("chosen")
        if chosen:
            assert row["result"] != "NO_VIABLE_MODEL", (
                f"the console reports NO_VIABLE_MODEL for a decision that "
                f"chose {chosen!r}"
            )
            assert row["target"] == chosen, (
                "the row does not name the model the decision actually picked"
            )


def test_the_console_names_the_thing_each_operation_acted_on(client):
    """`ToolExecutor.invoke` on its own does not tell a reader whether a file
    was written or the corpus was searched."""
    client.post("/api/tools/invoke", json={"tool": "calc",
                                           "args": {"expression": "1+1"}})
    rows = client.get("/api/audit/console?n=200").json()["rows"]
    tool_rows = [r for r in rows if r["operation"] == "ToolExecutor.invoke"]
    assert tool_rows
    assert all(r["target"] for r in tool_rows), (
        "a tool-plane row with no target names no tool"
    )


# ------------------------------------------------------------- the canary


def test_the_canary_verdict_reaches_the_console(client):
    """Driven through the real canary, for the same reason the routing test is:
    `run_canary` records `ok` and `conclusive`, not a `verdict` string, and a
    fixture written from the projection's assumptions agreed with the bug -
    every canary firing showed an empty Result on the one panel whose whole job
    is to report that verdict.
    """
    client.post("/api/sovereignty/canary", json={})
    rows = client.get("/api/audit/console?n=200").json()["rows"]
    results = [r for r in rows if r["event"] == "canary.result"]
    assert results, "the canary fired but recorded nothing"
    assert results[0]["result"] in ("BLOCKED", "INCONCLUSIVE", "ALLOWED"), (
        f"the console reports {results[0]['result']!r} for a canary firing"
    )


def test_an_inconclusive_canary_is_never_reported_as_blocked():
    """The house rule, at the exact point it would be broken. A timeout and an
    unplugged cable are indistinguishable; reporting the second as proof of the
    first is the most damaging single thing this panel could print."""
    row = project(_record("canary.result", {"ok": False, "conclusive": False}))
    assert row["result"] == "INCONCLUSIVE"


def test_a_canary_that_got_out_is_reported_as_allowed():
    """A probe reaching the outside world is a sovereignty breach. It has to be
    the loudest row in the log, not a blank cell."""
    row = project(_record("canary.result", {"ok": False, "conclusive": True}))
    assert row["result"] == "ALLOWED"


def test_a_fully_refused_canary_is_blocked():
    row = project(_record("canary.result", {"ok": True, "conclusive": True}))
    assert row["result"] == "BLOCKED"

"""Projecting audit records into an operations console (SS 26117, L0/L6).

The chain stores what happened. This turns one record into the row a security
or operations reader actually scans:

    TIME | ACTOR/ROLE | OPERATION | LABEL | EFFECT | DECISION | RESULT

Three rules shape everything below.

  NOTHING HERE IS INVENTED. Every `operation` is a real call site in this
  codebase - `ToolExecutor.invoke`, `PolicyEngine.check_tool`, `Router.decide`,
  `VramWarden.admit`. Every `decision` and `result` value is a word the system
  already uses: `ALLOWED`/`DENIED` are `PolicyVerdict`, `BLOCKED` and
  `INCONCLUSIVE` are canary verdicts, `PASSED`/`FAILED`/`PENDING` are
  `StepStatus`, `DONE`/`BUDGET_EXHAUSTED` are `RunStatus`, `VERIFIED`/
  `SIMULATED`/`UNVERIFIED` are `Provenance`. A console that renames the
  system's own vocabulary to read nicely is a console that cannot be checked
  against the code it describes.

  AN EVENT THIS TABLE DOES NOT KNOW IS SHOWN AS ITSELF. It keeps its raw event
  name, is marked `mapped: false`, and is counted in the response. The
  alternative - inferring a plausible operation from the event string - would
  put a confident, wrong call site next to a real digest. `test_every_audit_event_this_system_emits_has_a_console_mapping`
  is what keeps that count at zero.

  THE PROJECTION IS A VIEW, NEVER A WRITE. It reads the chain and derives; it
  cannot append, edit or reorder. The digests it displays are the stored ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------- vocabulary

# Policy-gate outcomes. These are `PolicyVerdict` in sovereignty/policy.py:
# allowed, denied, and requires_approval - which the executor raises as
# ApprovalRequired.
ALLOWED = "ALLOWED"
DENIED = "DENIED"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"

# Outcomes. Each is a word this system already uses; see the module docstring.
OK = "OK"
FAILED = "FAILED"
REJECTED = "REJECTED"
BLOCKED = "BLOCKED"
INCONCLUSIVE = "INCONCLUSIVE"
PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"
PASSED = "PASSED"
ACCEPTED = "ACCEPTED"
REPAIRED = "REPAIRED"
REPLANNED = "REPLANNED"
AWAITING_APPROVAL = "AWAITING_APPROVAL"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
DIGEST_MISMATCH = "DIGEST_MISMATCH"
DIVERGED = "DIVERGED"
NO_VIABLE_MODEL = "NO_VIABLE_MODEL"

# `-` rather than an empty string: a column that is genuinely not applicable to
# this event must read as "not applicable", not as "we did not record it".
NA = "-"

# Side effects are SideEffect in types.py. Events outside the tool plane have
# none, and say so rather than borrowing one that looks plausible.
READ, COMPUTE, EXECUTE, WRITE, DELETE = "READ", "COMPUTE", "EXECUTE", "WRITE", "DELETE"


@dataclass(frozen=True)
class Projection:
    """How one event name renders. `derive` may override result/decision from
    the payload, for events whose outcome is IN the record rather than implied
    by which record it is."""

    operation: str
    decision: str = NA
    result: str = NA
    side_effect: str = NA


# The whole mapping, grouped the way the layers are. An event added anywhere in
# the codebase without a row here is caught by the test named in the docstring.
EVENTS: dict[str, Projection] = {
    # ---- L4 tool plane. ToolExecutor.invoke is the single chokepoint: every
    # tool call passes resolve -> validate -> inspect -> gate -> execute, and
    # emits one record at whichever stage it stops.
    "tool.rejected": Projection("ToolExecutor.invoke", NA, REJECTED),
    "tool.denied": Projection("ToolExecutor.invoke", DENIED, BLOCKED),
    "tool.awaiting_approval": Projection("ToolExecutor.invoke", APPROVAL_REQUIRED, PENDING),
    "tool.invoke": Projection("ToolExecutor.invoke", ALLOWED, RUNNING),
    "tool.result": Projection("ToolExecutor.invoke", ALLOWED, OK),
    "tool.failed": Projection("ToolExecutor.invoke", ALLOWED, FAILED),
    "policy.check": Projection("PolicyEngine.check_tool"),

    # ---- L2 router
    "route.decide": Projection("Router.decide"),
    "route.panel_write_failed": Projection("Router.decide", NA, FAILED),

    # ---- L5 agent core
    "plan.accepted": Projection("Planner.plan", NA, ACCEPTED),
    "plan.rejected": Projection("Planner.plan", NA, REJECTED),
    "plan.repaired": Projection("Planner.plan", NA, REPAIRED),
    "step.judged": Projection("Critic.judge"),
    "step.references_resolved": Projection("resolve_references", NA, OK),
    "run.start": Projection("AgentLoop.start", NA, RUNNING),
    "run.done": Projection("AgentLoop.advance", NA, DONE),
    "run.failed": Projection("AgentLoop.advance", NA, FAILED),
    "run.replanned": Projection("Planner.replan", NA, REPLANNED),
    "run.awaiting_approval": Projection(
        "AgentLoop.advance", APPROVAL_REQUIRED, AWAITING_APPROVAL
    ),
    "run.budget_exhausted": Projection("AgentLoop.advance", NA, BUDGET_EXHAUSTED),
    "run.label_raised": Projection("AgentLoop.advance", NA, OK),
    "run.thread_failed": Projection("RunBroker.advance", NA, FAILED),
    "run.reconciled_at_boot": Projection("RunStore.reconcile_interrupted", NA, FAILED),
    "artifact.write": Projection("AgentLoop.advance", ALLOWED, OK, WRITE),

    # ---- L1 model fabric
    "model.verify": Projection("ModelRegistry.verify"),
    "model.digest_mismatch": Projection("ModelRegistry.verify", NA, DIGEST_MISMATCH),
    "model.admit": Projection("VramWarden.admit", NA, OK),
    "model.evict": Projection("VramWarden.admit", NA, OK),
    "model.evict_all": Projection("VramWarden.admit", NA, OK),
    "model.residency_diverged": Projection("VramWarden.admit", NA, DIVERGED),
    # A previous gateway died without killing its llama-servers and they were
    # still holding the card. Recorded because VRAM that vanished between boots
    # is otherwise indistinguishable from a card that is simply smaller than the
    # profile claims - and this box has had both.
    "model.orphans_reaped": Projection("reap_orphans", NA, OK),
    "model.registry_reload": Projection("ModelRegistry.load", NA, OK),
    # Adding a model to the box. `model.staged` writes two config files and
    # reloads the registry, so it is a WRITE with a policy decision behind it;
    # the denial is its own event because a refused attempt to stage a model is
    # exactly the thing an auditor comes looking for.
    "model.staged": Projection("stage", ALLOWED, OK, WRITE),
    "model.uploaded": Projection("upload", ALLOWED, OK, WRITE),
    "model.stage_denied": Projection("_require_admin", DENIED, BLOCKED, WRITE),

    # ---- L0 sovereignty
    "canary.start": Projection("run_canary", NA, RUNNING),
    "canary.result": Projection("run_canary"),
    "sentinel.baseline_reset": Projection("Sentinel.establish_baseline", NA, OK),

    # ---- L3 knowledge
    "kb.upload": Projection("Ingestor.ingest", ALLOWED, OK, WRITE),
    "kb.ingest_start": Projection("Ingestor.ingest", NA, RUNNING, WRITE),
    "kb.ingest_done": Projection("Ingestor.ingest", NA, OK, WRITE),
    "kb.ingest_failed": Projection("Ingestor.ingest", NA, FAILED, WRITE),
    "kb.document_ingested": Projection("Ingestor.ingest", NA, OK, WRITE),
    "kb.document_forgotten": Projection("KnowledgeStore.forget", ALLOWED, OK, DELETE),
    "kb.cleared": Projection("KnowledgeStore.clear", ALLOWED, OK, DELETE),
    "kb.answer": Projection("Answerer.answer", NA, NA, READ),

    # ---- gateway lifecycle
    "system.boot": Projection("create_app", NA, OK),
    "system.shutdown": Projection("create_app", NA, OK),
}


# --------------------------------------------------------- payload-derived

def _policy_decision(payload: dict[str, Any]) -> str:
    """`policy.check` records the verdict itself, so read it rather than
    inferring one from the fact that a check happened."""
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict):
        return NA
    if not verdict.get("allowed", False):
        return DENIED
    return APPROVAL_REQUIRED if verdict.get("requires_approval") else ALLOWED


def _derive(event: str, payload: dict[str, Any], base: Projection) -> tuple[str, str]:
    """(decision, result) for the events whose outcome lives in the payload."""
    decision, result = base.decision, base.result

    if event == "policy.check":
        decision = _policy_decision(payload)
        # The gate's own answer IS the result here: nothing was executed.
        result = BLOCKED if decision == DENIED else decision

    elif event == "canary.result":
        # `run_canary` records `ok` and `conclusive`, not a single verdict
        # string - the per-probe verdicts are nested inside `probes`. Reading a
        # top-level `verdict` that was never written left every canary firing
        # with an empty Result, on the one panel whose entire job is to report
        # that verdict.
        #
        # The three outcomes are exactly the three the canary itself
        # distinguishes, and the distinction is the point of the whole feature:
        #
        #   conclusive and ok   every probe refused by the kernel -> BLOCKED,
        #                       which is proof that egress is enforced.
        #   not conclusive      probes timed out. NOT a softer proof: a timeout
        #                       and an unplugged cable are indistinguishable,
        #                       so this stays INCONCLUSIVE and is never rounded
        #                       up to something reassuring.
        #   conclusive, not ok  a probe REACHED the outside world. ALLOWED here
        #                       means a sovereignty breach, and it must be the
        #                       loudest row in the log rather than a blank.
        conclusive = payload.get("conclusive")
        ok = payload.get("ok")
        if conclusive is None and ok is None:
            result = NA
        elif not conclusive:
            result = INCONCLUSIVE
        else:
            result = BLOCKED if ok else ALLOWED

    elif event == "route.decide":
        # The cascade records the winner under `model`. It was read as `chosen`
        # here - the name RouteDecision uses in the API - and every successful
        # route was therefore reported as NO_VIABLE_MODEL: a red row, with a
        # real digest, describing a decision that had in fact been made. Both
        # keys are accepted now, and test_a_real_route_is_not_reported_as_having
        # _no_viable_model drives a live route rather than a fixture, because a
        # fixture written from this file would have agreed with the bug.
        result = OK if (payload.get("model") or payload.get("chosen")) else NO_VIABLE_MODEL

    elif event == "step.judged":
        result = PASSED if payload.get("ok") else FAILED

    elif event == "model.verify":
        # Provenance in runtime/registry.py. SIMULATED is a distinct value from
        # VERIFIED everywhere else in this build, and stays distinct here.
        provenance = str(payload.get("provenance") or "").upper()
        result = provenance or NA

    elif event == "kb.answer":
        result = OK if payload.get("ok", True) else FAILED

    return decision, result


def _side_effect(payload: dict[str, Any], base: Projection) -> str:
    """The recorded side effect, the mapping's default, or - for a tool record
    written before the executor carried one - the registry's declaration.

    The registry is consulted LAST and guarded: a manifest can be removed after
    the record was written, and an audit row must still render.
    """
    declared = payload.get("side_effect")
    if isinstance(declared, str) and declared:
        return declared
    if base.side_effect != NA:
        return base.side_effect

    tool = payload.get("tool")
    if isinstance(tool, str) and tool:
        try:
            from ..tools import get_tool_registry

            return get_tool_registry().get(tool).side_effect.value
        except Exception:                             # noqa: BLE001 - see above
            return NA
    return NA


def _target(event: str, payload: dict[str, Any]) -> str:
    """What the operation acted on - the tool, model, document or run.

    Shown next to the operation because "ToolExecutor.invoke" on its own does
    not tell a reader whether a file was written or the corpus was searched.
    """
    for key in ("tool", "model", "path", "doc_id", "title", "iface"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def project(record: dict[str, Any]) -> dict[str, Any]:
    """One audit record -> one console row. Pure; never touches the database."""
    event = str(record.get("event", ""))
    payload = record.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    base = EVENTS.get(event)
    mapped = base is not None
    if base is None:
        # Honest fallback: the event names itself. No plausible-looking call
        # site is invented to fill the column.
        base = Projection(event)
        decision, result = NA, NA
    else:
        decision, result = _derive(event, payload, base)

    role = payload.get("role")

    return {
        "seq": record.get("seq"),
        "ts": record.get("ts"),
        "event": event,
        "mapped": mapped,
        "operation": base.operation,
        "target": _target(event, payload),
        "actor": record.get("actor") or "-",
        # The role the policy engine evaluated against. Absent on events that
        # never reached the policy engine, and on tool records written before
        # the executor carried it - `-` in both cases, never a default that
        # would read as "this ran as engineer".
        "role": role if isinstance(role, str) and role else NA,
        "label": record.get("data_class") or NA,
        "side_effect": _side_effect(payload, base),
        "decision": decision,
        "result": result,
        "run": record.get("run"),
        "step": record.get("step"),
        "egress_observed": record.get("egress_observed", 0),
        "sha256": record.get("sha256", ""),
        "prev_sha256": record.get("prev_sha256", ""),
        "payload": payload,
    }


def project_all(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Project a tail, and report anything this table could not name.

    `unmapped` is in the response on purpose: a console that silently renders
    unknown events as ordinary rows is one that stops describing the system the
    day a layer starts emitting something new.
    """
    rows = [project(r) for r in records]
    unmapped = sorted({r["event"] for r in rows if not r["mapped"]})
    return {
        "rows": rows,
        "unmapped": unmapped,
        "note": (
            f"{len(unmapped)} event type(s) have no console mapping and are "
            f"shown by their raw event name: {', '.join(unmapped)}"
            if unmapped else None
        ),
    }

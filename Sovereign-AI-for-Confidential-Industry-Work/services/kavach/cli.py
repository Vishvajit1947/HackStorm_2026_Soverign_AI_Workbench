"""Kavach CLI.

    kavach doctor           what this box can and cannot prove
    kavach serve            run the gateway
    kavach audit verify     walk the hash chain, report the first broken link
    kavach audit tail -n 20
    kavach canary           deliberate egress attempt + verdict
    kavach policy check ... one-off policy question
    kavach models list      the registry, with provenance and VRAM
    kavach models verify    re-hash every staged weight file
    kavach models load ID   admit a model, showing the eviction
    kavach models swap A B  the swap demo: watch the warden thrash and recover
    kavach route "..."      explain a routing decision, with the arithmetic
    kavach route --run "…"  route AND generate, through the verifier
    kavach tools list       every declared tool, its side effect and its gate
    kavach tools sandbox    can code.exec actually isolate anything right now?
    kavach tools probe      run code that tries to reach the network, and watch it fail
    kavach tools call T '{}'  invoke one tool
    kavach run "goal"       plan and execute an agentic run, live
    kavach run --plan-only  produce the plan without executing it
    kavach runs             recent runs
    kavach runs show ID     the run timeline
    kavach kb ingest PATH   read a file or folder into the corpus
    kavach kb stats         what is in the corpus - and what could not be read
    kavach kb doctor        what the knowledge plane can and cannot prove
    kavach kb search "..."  hybrid retrieval, with citations
    kavach kb ask "..."     a grounded answer, or a refusal with the reason
    kavach kb reindex       rebuild both retrieval arms
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .audit import get_audit
from .config import get_settings
from .sovereignty import get_policy, get_sentinel, run_canary
from .types import Label, SideEffect

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _make_console_lossy() -> None:
    """Never let a console encoding take down a command.

    Kavach's own output is ASCII by house rule, but the knowledge plane prints
    DOCUMENT text, and a refinery document is full of rho, delta and degree
    signs. A Windows console is cp1252 and raises UnicodeEncodeError on every one
    of them - which would kill `kavach kb search` on precisely the corpus it
    exists to search, with a traceback that looks like the retrieval broke.
    Losing a glyph is cosmetic; losing the command is not.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _c(text: str, colour: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{colour}{text}{RESET}"


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------- doctor

def cmd_doctor(args: argparse.Namespace) -> int:
    st = get_settings()
    sentinel = get_sentinel()
    state = sentinel.sample()
    audit = get_audit()
    verification = audit.verify()

    if args.json:
        _emit(
            {
                "profile": st.profile_name,
                "backend": st.backend,
                "enforce_digests": st.enforce_digests,
                "var_dir": str(st.var_dir),
                "model_dir": str(st.model_dir),
                "sentinel": state.as_dict(),
                "audit": verification.as_dict(),
            },
            True,
        )
        return 0

    print(_c("Kavach doctor", BOLD))
    print(f"  profile          {st.profile_name}  ({st.profile.description})")
    print(f"  vram budget      {st.profile.usable_vram_gb:.1f} GB usable "
          f"of {st.profile.vram_gb:.1f} GB")
    print(f"  backend          {st.backend}")
    print(f"  enforce digests  {st.enforce_digests}")
    print(f"  state dir        {st.var_dir}")
    print(f"  model dir        {st.model_dir}")
    print()

    if state.healthy:
        print(_c(f"  sentinel         HEALTHY ({state.mode})", GREEN))
        print(f"                   external packets since baseline: {state.external_packets}")
        print(f"                   blocked: {state.blocked_packets}")
        if state.mode != "nftables":
            # Counting is not blocking. Only the nftables mode can point at a
            # kernel that refused a packet; everywhere else "blocked" is our own
            # policy layer declining to try, which is a much weaker claim.
            print(_c("                   MEASURED, NOT ENFORCED - no firewall drop "
                     "counter here;", YELLOW))
            print(_c("                   'blocked' is Kavach policy denials, not "
                     "kernel drops", YELLOW))
    else:
        print(_c("  sentinel         DEGRADED", YELLOW))
        print(_c(f"                   {state.detail}", DIM))
    print()

    if verification.ok:
        print(_c(f"  audit chain      OK - {verification.records} records", GREEN))
        print(_c(f"                   {verification.detail}", DIM))
    else:
        print(_c(f"  audit chain      BROKEN at seq {verification.first_broken_seq}", RED))
        print(_c(f"                   {verification.detail}", DIM))

    return 0 if verification.ok else 1


# ---------------------------------------------------------------- audit

def cmd_audit_verify(args: argparse.Namespace) -> int:
    result = get_audit().verify()
    if args.json:
        _emit(result.as_dict(), True)
        return 0 if result.ok else 1

    if result.ok:
        print(_c(f"chain intact, {result.records} records", GREEN))
        print(_c(result.detail, DIM))
        return 0

    print(_c(f"CHAIN BROKEN at seq {result.first_broken_seq}", RED))
    print(result.detail)
    print(_c(f"({result.records} records total)", DIM))
    return 1


def cmd_audit_tail(args: argparse.Namespace) -> int:
    records = get_audit().tail(n=args.n, run_id=args.run)
    if args.json:
        _emit(records, True)
        return 0
    for r in reversed(records):
        run = f" {r['run']}" if r["run"] else ""
        print(
            f"{r['seq']:>6}  {r['ts']}  {_c(r['event'], BOLD):<28}"
            f"  {r['actor']}{run}  {_c(r['sha256'][:12], DIM)}"
        )
    return 0


# ---------------------------------------------------------------- canary

def cmd_canary(args: argparse.Namespace) -> int:
    result = run_canary(target_ip=args.target, timeout_s=args.timeout)
    if args.json:
        _emit(result.as_dict(), True)
        return 0 if result.ok else 1

    for p in result.probes:
        colour = {"BLOCKED": GREEN, "INCONCLUSIVE": YELLOW, "REACHED": RED}[p.verdict]
        print(f"  {p.name:<14} {p.proto}/{p.target:<22} {_c(p.verdict, colour)}")
        print(_c(f"    {p.detail}", DIM))
    print()
    colour = GREEN if result.ok else (YELLOW if not result.conclusive else RED)
    print(_c(result.headline, colour))
    print(_c(f"  blocked counter {result.blocked_before} -> {result.blocked_after}", DIM))
    return 0 if result.ok else 1


# ---------------------------------------------------------------- policy

def cmd_policy_check(args: argparse.Namespace) -> int:
    verdict = get_policy().check_tool(
        tool_name=args.tool,
        side_effect=SideEffect(args.side_effect),
        label=Label(args.label),
        role=args.role,
        writes_outside_workspace=args.outside_workspace,
        is_final_deliverable=args.final,
    )
    if args.json:
        _emit(verdict.as_dict(), True)
        return 0 if verdict.allowed else 1

    if verdict.denied:
        print(_c("DENIED", RED), verdict.reason)
        return 1
    if verdict.requires_approval:
        print(_c("NEEDS APPROVAL", YELLOW), verdict.reason)
        return 0
    print(_c("ALLOWED", GREEN), verdict.reason)
    return 0


# ---------------------------------------------------------------- models

_PROV_COLOUR = {
    "VERIFIED": GREEN,
    "SIMULATED": YELLOW,
    "UNVERIFIED": YELLOW,
    "MISMATCH": RED,
}


def cmd_models_list(args: argparse.Namespace) -> int:
    from .runtime import get_registry, get_warden

    registry = get_registry()
    warden = get_warden()
    specs = sorted(registry.all(), key=lambda s: (s.role != "generation", -s.tier, s.id))

    if args.json:
        _emit(registry.summary(), True)
        return 0

    print(_c(f"{'MODEL':<22} {'ROLE':<10} {'T':<2} {'VRAM':>6} {'PROVENANCE':<11} RES", BOLD))
    for s in specs:
        colour = _PROV_COLOUR.get(s.provenance.value, DIM)
        resident = _c("yes", GREEN) if warden.is_resident(s.id) else _c("-", DIM)
        print(
            f"{s.id:<22} {s.role:<10} {s.tier:<2} {s.vram_gb:>5.1f}G "
            f"{_c(s.provenance.value, colour):<11} {resident}"
        )

    state = warden.state()
    print()
    print(
        f"  VRAM  {state['used_gb']:.1f} / {state['budget_gb']:.1f} GB "
        f"({state['free_gb']:.1f} free, {state['kv_cache_gb']:.1f} reserved for KV)"
    )
    print(f"  slots {state['slots_in_use']} / {state['swap_slots']} in use")
    if state["pin_failures"]:
        for failure in state["pin_failures"]:
            print(_c(f"  pin failed: {failure}", YELLOW))

    summary = registry.summary()
    if summary["warning"]:
        print(_c(f"\n  {summary['warning']}", YELLOW))
    return 0


def cmd_models_verify(args: argparse.Namespace) -> int:
    from .runtime import Provenance, get_registry

    registry = get_registry()
    registry.reload()
    specs = registry.all()

    real = [s for s in specs if s.provenance is Provenance.VERIFIED]
    simulated = [s for s in specs if s.provenance is Provenance.SIMULATED]
    unverified = [s for s in specs if s.provenance is Provenance.UNVERIFIED]

    if args.json:
        _emit(
            {
                "ok": not simulated and not unverified,
                "verified": [s.id for s in real],
                "simulated": [s.id for s in simulated],
                "unverified": [s.id for s in unverified],
            },
            True,
        )
        return 0 if not simulated and not unverified else 1

    for s in real:
        print(_c(f"  VERIFIED    {s.id:<22} {s.sha256[:16]}...", GREEN))
    for s in unverified:
        observed = f"{s.observed_sha256[:16]}..." if s.observed_sha256 else "file not staged"
        print(_c(f"  UNVERIFIED  {s.id:<22} {observed}", YELLOW))
    for s in simulated:
        print(_c(f"  SIMULATED   {s.id:<22} no weights on this box", YELLOW))

    print()
    if not simulated and not unverified:
        print(_c(f"every one of {len(real)} weight files hashes to its approved digest", GREEN))
        return 0
    print(_c(
        f"{len(real)} verified, {len(unverified)} unverified, {len(simulated)} simulated "
        "- NOT demo-ready", YELLOW,
    ))
    print(_c("  stage the GGUFs, then: python scripts/stage_model.py --write", DIM))
    return 1


def cmd_models_load(args: argparse.Namespace) -> int:
    from .runtime import VramExhausted, get_warden

    warden = get_warden()
    try:
        decision = warden.admit(args.model_id)
    except KeyError as exc:
        print(_c(str(exc), RED), file=sys.stderr)
        return 2
    except VramExhausted as exc:
        print(_c(f"VRAM EXHAUSTED: {exc}", RED), file=sys.stderr)
        return 1

    if args.json:
        _emit({"decision": decision.as_dict(), "residency": warden.state()}, True)
        return 0

    if decision.was_resident:
        print(_c(f"{args.model_id} was already resident - no load, no cost", GREEN))
    else:
        if decision.evicted:
            print(_c(f"  evicted  {', '.join(decision.evicted)}", YELLOW))
        print(_c(f"  loaded   {args.model_id} in {decision.load_ms:.0f} ms", GREEN))
    state = warden.state()
    print(_c(f"  VRAM     {state['used_gb']:.1f} / {state['budget_gb']:.1f} GB", DIM))
    return 0


def cmd_models_swap(args: argparse.Namespace) -> int:
    """The Phase 2 demo: two models swapping cleanly under load on one card."""
    from .runtime import VramExhausted, get_warden

    warden = get_warden()
    print(_c(f"Swap test: {args.model_a} <-> {args.model_b} x{args.rounds}", BOLD))
    state = warden.state()
    print(_c(
        f"  budget {state['budget_gb']:.1f} GB, {state['swap_slots']} swap slot(s), "
        f"pinned: {[r['model_id'] for r in state['resident'] if r['pinned']]}", DIM,
    ))
    print()

    total_ms = 0.0
    for i in range(args.rounds):
        for model_id in (args.model_a, args.model_b):
            try:
                d = warden.admit(model_id)
            except (KeyError, VramExhausted) as exc:
                print(_c(f"  {exc}", RED), file=sys.stderr)
                return 1
            total_ms += d.load_ms
            tag = _c("HIT ", GREEN) if d.was_resident else _c("LOAD", YELLOW)
            evicted = f" (evicted {', '.join(d.evicted)})" if d.evicted else ""
            print(
                f"  {i + 1}.{model_id:<24} {tag} {d.load_ms:>7.0f} ms"
                f"  vram {d.vram_after_gb:.1f} GB{_c(evicted, DIM)}"
            )

    print()
    print(f"  total swap cost {total_ms:.0f} ms over {args.rounds * 2} admissions")
    print(_c(
        "  the router's swap penalty (Phase 3) is what keeps this rare in a real run",
        DIM,
    ))
    return 0


def cmd_models_reload(args: argparse.Namespace) -> int:
    from .runtime import DigestMismatch, get_registry

    registry = get_registry()
    before = {s.id for s in registry.all()}
    try:
        summary = registry.reload()
    except DigestMismatch as exc:
        print(_c("DIGEST MISMATCH - refusing to reload", RED), file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    after = {s.id for s in registry.all()}
    added, removed = sorted(after - before), sorted(before - after)

    if args.json:
        _emit({"added": added, "removed": removed, **summary}, True)
        return 0

    for model_id in added:
        print(_c(f"  + {model_id}", GREEN))
    for model_id in removed:
        print(_c(f"  - {model_id}", YELLOW))
    if not added and not removed:
        print(_c("  no change", DIM))
    print(f"\n{summary['count']} models registered")
    return 0


# ---------------------------------------------------------------- route

def cmd_route(args: argparse.Namespace) -> int:
    from .router import get_router
    from .router.cascade import NoViableModel
    from .types import RouteRequest, TaskClass

    r = get_router()
    req = RouteRequest(
        prompt=args.prompt,
        has_images=args.images,
        needs_tools=args.tools,
        min_context=args.min_context,
        label=Label(args.label),
        forced_class=TaskClass(args.force_class) if args.force_class else None,
        forced_model=args.force_model,
    )

    if not args.run:
        decision, classification = r.decide(req)
        if args.json:
            _emit(
                {
                    "decision": decision.explain(),
                    "classification": classification.as_dict(),
                    "quality_provenance": r.quality_provenance(),
                },
                True,
            )
            return 0 if decision.chosen else 1
        _print_decision(r, decision, classification)
        return 0 if decision.chosen else 1

    try:
        outcome = r.route(req, max_tokens=args.max_tokens)
    except NoViableModel as exc:
        if args.json:
            _emit({"error": str(exc), "decision": exc.decision.explain()}, True)
        else:
            print(_c(f"no viable model: {exc}", RED), file=sys.stderr)
        return 1

    if args.json:
        _emit(outcome.as_dict(), True)
        return 0

    _print_decision(r, outcome.decision, outcome.classification)
    print()
    for attempt in outcome.attempts:
        mark = _c("PASS", GREEN) if attempt["ok"] else _c("FAIL", RED)
        print(f"  attempt {attempt['attempt']}  {attempt['model']:<24} {mark} "
              f"{attempt['elapsed_ms']:.0f} ms")
        if not attempt["ok"]:
            print(_c(f"    {attempt['reason']}", DIM))
        if attempt.get("escalation"):
            print(_c(f"    -> {attempt['escalation']}", YELLOW))
    print()
    print(_c("--- output " + "-" * 50, DIM))
    print(outcome.text)
    return 0


def _print_decision(r, decision, classification) -> None:
    from .router.scoring import rank

    chosen = decision.chosen
    print(_c(f"class    {classification.task_class.value}  "
             f"p={classification.confidence:.2f}  via {classification.method}", BOLD))
    if classification.degraded:
        print(_c(f"         {classification.detail}", YELLOW))
    if classification.neighbours:
        for n in classification.neighbours[:3]:
            print(_c(f"         {n['similarity']:.3f}  {n['label']:<18} {n['example']}", DIM))
    print()

    eliminated = [c for c in decision.candidates if not c.survived]

    print(_c(f"{'MODEL':<24} {'SCORE':>7}  {'QUAL':>5} {'FIT':>6} {'LAT':>6} "
             f"{'SWAP':>6} {'VRAM':>6}", BOLD))
    for c in rank(decision.candidates):
        t = c.terms
        mark = _c(" <-", GREEN) if c.model_id == chosen else ""
        res = _c("R", GREEN) if c.resident else " "
        print(
            f"{c.model_id:<24} {c.score:>7.3f}  {t['_quality_raw']:>5.2f} "
            f"{t['confidence_fit']:>+6.3f} {t['latency']:>+6.3f} "
            f"{t['swap']:>+6.3f} {t['vram']:>+6.3f} {res}{mark}"
        )

    if eliminated:
        print()
        print(_c(f"eliminated ({len(eliminated)} of {len(decision.candidates)}):", DIM))
        for c in eliminated:
            print(_c(f"  {c.model_id:<24} {c.eliminated_by}", DIM))

    print()
    print(f"chosen   {_c(str(chosen), GREEN) if chosen else _c('NONE', RED)}")
    print(_c(f"reason   {decision.reason}", DIM))
    print(_c(f"took     {decision.elapsed_ms:.1f} ms", DIM))

    if r.quality_provenance() == "seeded":
        print()
        print(_c("  WARNING  the quality column is hand-seeded, not measured.", YELLOW))
        print(_c("           run scripts/eval_harness.py before showing this to judges.", YELLOW))


# ---------------------------------------------------------------- tools

def _adhoc_workspace(run_id: str):
    from .config import get_settings
    from .tools import Workspace

    return Workspace.for_run(get_settings().var_dir, run_id)


def cmd_tools_list(args: argparse.Namespace) -> int:
    from .tools import get_tool_registry

    registry = get_tool_registry()
    if args.json:
        _emit(registry.summary(), True)
        return 0

    print(_c(f"{'TOOL':<14} {'EFFECT':<9} {'SANDBOX':<9} {'GATE':<18} ARGS", BOLD))
    for spec in registry.all():
        sandbox = _c("none-net", GREEN) if spec.sandbox else _c("-", DIM)
        if spec.requires_approval:
            gate = _c("always approve", YELLOW)
        elif spec.produces_artifact:
            gate = _c("final -> approve", YELLOW)
        elif spec.side_effect.value in ("WRITE", "DELETE"):
            gate = _c("outside -> approve", YELLOW)
        else:
            gate = _c("none", GREEN)
        required = ", ".join(spec.required) or "-"
        print(f"{spec.name:<14} {spec.side_effect.value:<9} {sandbox:<9} {gate:<18} {required}")

    if registry.unbound():
        print()
        print(_c(f"  unbound (no handler): {', '.join(registry.unbound())}", YELLOW))
    return 0


def cmd_tools_sandbox(args: argparse.Namespace) -> int:
    from .tools.impl.sandbox import SANDBOX_IMAGE, docker_available, image_available

    docker = docker_available()
    image = image_available() if docker else False
    ready = docker and image

    if args.json:
        _emit({"docker": docker, "image": image, "ready": ready}, True)
        return 0 if ready else 1

    print(f"  docker daemon   {_c('yes', GREEN) if docker else _c('NO', RED)}")
    print(f"  image           {_c('present', GREEN) if image else _c('MISSING', RED)}"
          f"  {_c(SANDBOX_IMAGE, DIM)}")
    print()
    if ready:
        print(_c("  sandbox ready - code.exec runs with no network interface", GREEN))
        return 0
    print(_c("  code.exec will REFUSE to run.", YELLOW))
    print(_c("  build it with:  docker compose build sandbox-runner", DIM))
    return 1


def cmd_tools_probe(args: argparse.Namespace) -> int:
    """The tool-plane counterpart of the L0 canary."""
    from .tools import get_executor
    from .tools.impl.sandbox import EGRESS_PROBE

    result = get_executor().invoke(
        "code.exec", {"code": EGRESS_PROBE}, _adhoc_workspace("probe")
    )
    if args.json:
        _emit(result.as_dict(), True)
        return 0 if result.ok else 1

    if not result.ok:
        print(_c("could not run the probe:", RED), file=sys.stderr)
        print(result.error, file=sys.stderr)
        return 1

    stdout = result.data.get("stdout", "")
    reached = "REACHED THE NETWORK" in stdout or "SOCKET CONNECTED" in stdout
    print(stdout.rstrip())
    print()
    if reached:
        print(_c("SANDBOX BREACH - generated code reached the network", RED))
        return 1
    if not result.data.get("sandboxed"):
        print(_c("NOT SANDBOXED - this ran on the host, so it proves nothing.", YELLOW))
        print(_c("Start Docker and build the image for a result worth showing.", DIM))
        return 1
    print(_c("blocked - the sandbox has no network interface at all", GREEN))
    return 0


def cmd_tools_call(args: argparse.Namespace) -> int:
    from .tools import (
        ApprovalRequired,
        InvalidArguments,
        PolicyDenied,
        ToolNotFound,
        get_executor,
    )

    try:
        tool_args = json.loads(args.args_json)
    except json.JSONDecodeError as exc:
        print(_c(f"arguments are not valid JSON: {exc}", RED), file=sys.stderr)
        return 2

    try:
        result = get_executor().invoke(
            args.tool,
            tool_args,
            _adhoc_workspace(args.run),
            label=Label(args.label),
            role=args.role,
            run_id=args.run,
            approved_by=args.approve,
        )
    except ToolNotFound as exc:
        print(_c(str(exc), RED), file=sys.stderr)
        return 2
    except InvalidArguments as exc:
        print(_c(f"INVALID ARGUMENTS  {exc}", RED), file=sys.stderr)
        return 2
    except PolicyDenied as exc:
        print(_c(f"DENIED  {exc}", RED), file=sys.stderr)
        return 1
    except ApprovalRequired as exc:
        print(_c(f"NEEDS APPROVAL  {exc}", YELLOW), file=sys.stderr)
        print(_c("  re-run with --approve <name> once a human has agreed", DIM),
              file=sys.stderr)
        return 1

    if args.json:
        _emit(result.as_dict(), True)
        return 0 if result.ok else 1

    if not result.ok:
        print(_c("FAILED", RED), result.error)
        return 1

    if result.sandboxed is False:
        print(_c("  ran WITHOUT a sandbox - development only", YELLOW))
    print(json.dumps(result.data, indent=2, default=str))
    print(_c(f"  {result.elapsed_ms:.0f} ms", DIM))
    return 0


# ---------------------------------------------------------------- runs

_STEP_MARK = {
    "pending": (DIM, "..."),
    "running": (YELLOW, ">>>"),
    "passed": (GREEN, "OK "),
    "failed": (RED, "XX "),
    "awaiting_approval": (YELLOW, "?? "),
    "skipped": (DIM, "-- "),
}


def _print_step(step: dict[str, Any]) -> None:
    colour, mark = _STEP_MARK.get(step["status"], (DIM, "   "))
    approval = _c(" [needs approval]", YELLOW) if step["requires_approval"] else ""
    retries = _c(f"  x{step['attempts']}", DIM) if step["attempts"] > 1 else ""
    print(f"  {_c(mark, colour)} {step['idx']}. {step['intent'][:62]}{approval}{retries}")
    print(_c(f"        tool {step['tool']}  |  accept: {step['acceptance'][:60]}", DIM))
    for obs in step["observations"]:
        if obs["stderr"]:
            print(_c(f"        ! {obs['stderr'][:100]}", RED))


def cmd_run(args: argparse.Namespace) -> int:
    from .agent import get_loop, get_run_store
    from .types import Budget, RunStatus

    loop = get_loop()
    budget = Budget(max_steps=args.max_steps, max_wall_s=args.max_wall_s)

    run = loop.start(args.goal, label=Label(args.label), budget=budget)

    if run.status is RunStatus.FAILED:
        print(_c("PLANNING FAILED", RED), file=sys.stderr)
        print(run.error, file=sys.stderr)
        return 1

    print(_c(f"run {run.id}", BOLD))
    print(_c(f"goal: {args.goal}", DIM))
    print()
    print(_c(f"plan ({len(run.steps)} steps)", BOLD))
    for step in run.steps:
        gate = _c(" [needs approval]", YELLOW) if step.requires_approval else ""
        print(f"  {step.idx}. {step.intent[:66]}{gate}")
        print(_c(f"     tool {step.tool}  |  accept: {step.acceptance[:58]}", DIM))

    if args.plan_only:
        return 0

    print()
    print(_c("executing", BOLD))

    def on_event(event) -> None:
        if event.kind == "step_started":
            print(f"  {_c('>>>', YELLOW)} {event.step_idx}. "
                  f"{event.detail.get('intent', '')[:60]}")
        elif event.kind == "step_passed":
            print(f"  {_c('OK ', GREEN)} {event.step_idx}. "
                  f"{_c(event.detail.get('reason', '')[:70], DIM)}")
        elif event.kind == "step_failed":
            print(f"  {_c('XX ', RED)} {event.step_idx}. "
                  f"{event.detail.get('reason', '')[:70]}")
        elif event.kind == "replanned":
            print(_c(f"  ~~~ replanned: {event.detail.get('new_steps')} new step(s)",
                     YELLOW))
        elif event.kind == "awaiting_approval":
            print(_c(f"  ??  {event.step_idx}. waiting for approval: "
                     f"{event.detail.get('intent', '')[:50]}", YELLOW))
        elif event.kind == "budget_exhausted":
            print(_c(f"  --- {event.detail.get('reason')}", YELLOW))

    approvals: dict[int, str] = {}
    while True:
        run = loop.advance(run.id, approvals=approvals, on_event=on_event)
        if run.status is not RunStatus.AWAITING_APPROVAL:
            break
        if not args.approve:
            print()
            print(_c("run paused for human approval.", YELLOW))
            print(_c(f"  kavach runs approve {run.id} --step <n> --by <name>", DIM))
            print(_c("  or re-run with --approve <name>", DIM))
            return 2
        approvals = {s.idx: args.approve for s in run.steps if s.requires_approval}

    print()
    timeline = get_run_store().timeline(run.id)
    colour = {"done": GREEN, "failed": RED}.get(run.status.value, YELLOW)
    print(_c(f"status   {run.status.value}", colour))
    if run.error:
        print(_c(f"reason   {run.error}", DIM))
    budget_used = timeline["budget"]
    print(_c(f"budget   {budget_used['steps_used']}/{budget_used['max_steps']} steps, "
             f"{budget_used['tool_calls_used']}/{budget_used['max_tool_calls']} tool calls, "
             f"{budget_used['elapsed_s']}s", DIM))
    if timeline["artifacts"]:
        print()
        print(_c("artifacts", BOLD))
        for art in timeline["artifacts"]:
            signed = f" approved by {art['approved_by']}" if art["approved_by"] else ""
            print(f"  {art['path']}  {_c(art['sha256'][:12] + signed, DIM)}")

    if args.json:
        _emit(timeline, True)
    return 0 if run.status is RunStatus.DONE else 1


def cmd_runs_list(args: argparse.Namespace) -> int:
    from .agent import get_run_store

    runs = get_run_store().list_runs(args.n)
    if args.json:
        _emit(runs, True)
        return 0
    if not runs:
        print(_c("no runs yet", DIM))
        return 0
    print(_c(f"{'RUN':<14} {'STATUS':<18} {'LABEL':<13} GOAL", BOLD))
    for r in runs:
        colour = {"done": GREEN, "failed": RED}.get(r["status"], YELLOW)
        print(f"{r['id']:<14} {_c(r['status'], colour):<18} {r['label']:<13} "
              f"{r['goal'][:44]}")
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    from .agent import RunNotFound, get_run_store

    try:
        timeline = get_run_store().timeline(args.run_id)
    except RunNotFound as exc:
        print(_c(str(exc), RED), file=sys.stderr)
        return 2

    if args.json:
        _emit(timeline, True)
        return 0

    colour = {"done": GREEN, "failed": RED}.get(timeline["status"], YELLOW)
    print(_c(f"{timeline['run_id']}  {timeline['status']}", colour))
    print(_c(f"goal: {timeline['goal']}", DIM))
    print()
    for step in timeline["steps"]:
        _print_step(step)
    if timeline["artifacts"]:
        print()
        for art in timeline["artifacts"]:
            print(f"  artifact {art['path']}  {_c(art['sha256'][:12], DIM)}")
    return 0


def cmd_runs_approve(args: argparse.Namespace) -> int:
    from .agent import RunNotFound, get_loop, get_run_store
    from .types import RunStatus

    try:
        run = get_loop().advance(args.run_id, approvals={args.step: args.by})
    except RunNotFound as exc:
        print(_c(str(exc), RED), file=sys.stderr)
        return 2

    print(_c(f"approved step {args.step} as {args.by}", GREEN))
    print(f"status: {run.status.value}")
    for step in get_run_store().timeline(args.run_id)["steps"]:
        _print_step(step)
    return 0 if run.status is RunStatus.DONE else 1


# ---------------------------------------------------------------- knowledge

_EXTRACTION_COLOUR = {
    "NATIVE": GREEN,
    "OCR": GREEN,
    "VLM": GREEN,
    "SIMULATED": YELLOW,
    "UNREADABLE": RED,
}


def _print_coverage(stats: dict[str, Any]) -> None:
    print(f"  documents        {stats['documents']}")
    print(f"  pages            {stats['pages']}")
    print(f"  chunks indexed   {stats['chunks']}")
    print()
    print(_c("  how each page was read", BOLD))
    for kind, count in sorted(stats["extraction"].items()):
        colour = _EXTRACTION_COLOUR.get(kind, DIM)
        print(f"    {_c(kind.ljust(12), colour)} {count}")
    if stats["labels"]:
        print()
        print(_c("  classification", BOLD))
        for label, count in sorted(stats["labels"].items()):
            print(f"    {label.ljust(12)} {count} document(s)")
    if stats["gaps"]:
        print()
        print(_c("  pages nothing on this box could read", RED))
        for gap in stats["gaps"]:
            pages = ", ".join(f"p.{p}" for p in gap["pages"])
            print(f"    {gap['doc_id'].ljust(28)} {pages}")
    if stats["warning"]:
        print()
        print(_c(f"  {stats['warning']}", YELLOW))


def cmd_kb_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path as _Path

    from .knowledge import IngestError, get_ingestor

    target = _Path(args.path).expanduser()
    if not target.exists():
        print(_c(f"{target} does not exist", RED), file=sys.stderr)
        return 2

    try:
        report = get_ingestor().ingest(
            target,
            label=Label(args.label) if args.label else None,
            force=args.force,
        )
    except IngestError as exc:
        print(_c(str(exc), RED), file=sys.stderr)
        return 2

    if args.json:
        _emit(report.as_dict(), True)
        return 0 if not report.failed else 1

    print(_c(f"ingesting {target}", BOLD))
    print()
    for doc in report.documents:
        name = (doc.doc_id or _Path(doc.path).name)[:30].ljust(30)
        if doc.skipped:
            print(f"  {_c('SKIP', DIM)}  {name} {_c(doc.reason, DIM)}")
            continue
        if not doc.ok:
            print(f"  {_c('FAIL', RED)}  {name} {doc.reason}")
            continue
        shape = " ".join(f"{k.lower()}={v}" for k, v in sorted(doc.extraction.items()))
        marker = GREEN if not doc.unreadable_pages else YELLOW
        print(
            f"  {_c('OK  ', marker)}  {name} "
            f"{doc.pages} page(s), {doc.chunks} chunk(s), {doc.label}"
        )
        print(f"        {_c(shape, DIM)}")
        if doc.unreadable_pages:
            pages = ", ".join(f"p.{p}" for p in doc.unreadable_pages)
            print(_c(f"        UNREADABLE: {pages}", RED))

    print()
    print(f"  ingested {report.ingested}, skipped {report.skipped}, "
          f"failed {len(report.failed)} in {report.elapsed_s:.1f}s")
    index = report.index
    print(f"  index    {index.get('chunks')} chunk(s), {index.get('detail')}")
    if report.warning:
        print()
        print(_c(f"  {report.warning}", YELLOW))
        print(_c("  run `kavach kb doctor` for what that costs you", DIM))
    return 0 if not report.failed else 1


def cmd_kb_stats(args: argparse.Namespace) -> int:
    from .knowledge import get_index, get_knowledge_store

    stats = get_knowledge_store().stats()
    index = get_index().state()

    if args.json:
        _emit({"corpus": stats, "index": index.as_dict()}, True)
        return 0

    print(_c("Kavach knowledge plane", BOLD))
    _print_coverage(stats)
    print()
    colour = YELLOW if index.degraded else GREEN
    print(f"  index            {_c('DEGRADED' if index.degraded else 'HYBRID', colour)}")
    print(f"                   {_c(index.detail, DIM)}")
    return 0


def cmd_kb_doctor(args: argparse.Namespace) -> int:
    """What the knowledge plane can and cannot prove on this box."""
    from .knowledge import ImageReader, get_index, get_knowledge_store

    stats = get_knowledge_store().stats()
    index = get_index().state()
    image = ImageReader().status()

    if args.json:
        _emit(
            {"corpus": stats, "index": index.as_dict(), "image_path": image.as_dict()},
            True,
        )
        return 0 if not stats["unreadable_pages"] and not index.degraded else 1

    print(_c("Kavach knowledge doctor", BOLD))
    print()
    print(_c("  the ingest fork", BOLD))
    print("    native text      always available (PyMuPDF text layer)")
    ocr = image.ocr_engine or "none"
    print(f"    OCR engine       {_c(ocr, GREEN if image.ocr_engine else RED)}")
    vision = image.vision_model or "none"
    vcolour = RED if image.vision_is_simulated or not image.vision_model else GREEN
    suffix = " (SIMULATED)" if image.vision_is_simulated else ""
    print(f"    vision model     {_c(vision + suffix, vcolour)}")
    print(f"    image path       {_c(image.mode.value, _EXTRACTION_COLOUR.get(image.mode.value, DIM))}")
    print(_c(f"      {image.detail}", DIM))
    print()

    print(_c("  retrieval", BOLD))
    print(f"    lexical (BM25)   {_c('on', GREEN) if index.lexical_ready else _c('OFF', RED)}")
    print(f"    dense (vectors)  {_c('on', GREEN) if index.dense_ready else _c('OFF', RED)}")
    print(f"    embedded with    {index.embed_model or '-'}")
    print(f"    resident emb.    {index.resident_embedder or '-'}")
    if index.stale:
        print(_c("    index is STALE - run `kavach kb reindex`", YELLOW))
    print()

    _print_coverage(stats)

    healthy = not index.degraded and not stats["unreadable_pages"]
    print()
    if healthy:
        print(_c("  every page in the corpus was read and both arms are live", GREEN))
        return 0
    print(_c("  the knowledge plane is DEGRADED - see above", YELLOW))
    print(_c("  nothing here is hidden from an answer: every search reports it", DIM))
    return 1


def cmd_kb_documents(args: argparse.Namespace) -> int:
    from .knowledge import get_knowledge_store

    docs = get_knowledge_store().documents()
    if args.json:
        _emit({"documents": [d.as_dict() for d in docs]}, True)
        return 0

    if not docs:
        print(_c("no documents ingested - try `kavach kb ingest MRPL_DEMO_CORPUS`", DIM))
        return 0

    print(f"  {'DOC ID'.ljust(30)} {'LABEL'.ljust(13)} PAGES  READ  TITLE")
    for doc in docs:
        gap = "" if not doc.unreadable_pages else _c(
            f"  gap p.{','.join(str(p) for p in doc.unreadable_pages)}", RED
        )
        print(
            f"  {doc.id[:30].ljust(30)} {doc.label.value.ljust(13)} "
            f"{str(doc.page_count).rjust(5)}  {str(doc.readable_pages).rjust(4)}  "
            f"{doc.title[:46]}{gap}"
        )
    return 0


def cmd_kb_search(args: argparse.Namespace) -> int:
    from .knowledge import get_retriever

    result = get_retriever().search(args.query, top_k=args.top_k, role=args.role)
    if args.json:
        _emit(result.as_dict(), True)
        return 0 if result.hits else 1

    print(_c(f'"{result.query}"', BOLD))
    print(_c(f"  arms {'+'.join(result.arms) or 'none'} | {result.detail}", DIM))
    print()

    if not result.hits:
        print(_c("  no passage matched", YELLOW))
    for hit in result.hits:
        head = hit.chunk.citation
        arms = "+".join(hit.arms)
        print(f"  {_c(head, BOLD)}  {_c(f'{hit.score:.4f} via {arms}', DIM)}")
        print(f"     {_c(hit.chunk.label.value, DIM)}  "
              f"{_c(hit.chunk.extraction.value, _EXTRACTION_COLOUR.get(hit.chunk.extraction.value, DIM))}")
        snippet = " ".join(hit.chunk.text.split())[:220]
        print(f"     {snippet}...")
        print()

    if result.withheld:
        print(_c(f"  {result.withheld} passage(s) withheld - {result.withheld_reason}",
                 YELLOW))
    if result.coverage.warning:
        print(_c(f"  {result.coverage.warning}", YELLOW))
    if result.degraded:
        print(_c("  RETRIEVAL IS DEGRADED - see `kavach kb doctor`", YELLOW))
    return 0 if result.hits else 1


def cmd_kb_ask(args: argparse.Namespace) -> int:
    from .knowledge import get_answerer

    result = get_answerer().answer(
        args.question, top_k=args.top_k, role=args.role, max_tokens=args.max_tokens
    )
    if args.json:
        _emit(result.as_dict(), True)
        return 0 if result.ok else 1

    print(_c(f'"{result.question}"', BOLD))
    print(_c(f"  label {result.label.value} | model {result.model or '-'} | "
             f"{result.elapsed_ms:.0f} ms", DIM))
    print()

    if result.ok:
        print(result.answer)
        print()
        print(_c(f"  {result.headline}", YELLOW if result.simulated else GREEN))
    else:
        print(_c("  NO GROUNDED ANSWER", RED))
        print(_c(f"  {result.headline}", YELLOW))
        if result.rejected_answer:
            print()
            print(_c("  what the model produced, withheld as ungrounded:", DIM))
            for line in result.rejected_answer.splitlines()[:12]:
                print(_c(f"    {line}", DIM))

    if result.sources:
        print()
        print(_c("  sources", BOLD))
        for source in result.sources:
            print(f"    {source['citation'].ljust(30)} {source['label'].ljust(13)}"
                  f" {source['extraction']}")
    return 0 if result.ok else 1


def cmd_kb_reindex(args: argparse.Namespace) -> int:
    from .knowledge import get_index

    state = get_index().build()
    if args.json:
        _emit(state.as_dict(), True)
        return 0
    colour = YELLOW if state.degraded else GREEN
    print(f"  {state.chunks} chunk(s) indexed")
    print(f"  {_c(state.detail, colour)}")
    return 0


def cmd_kb_forget(args: argparse.Namespace) -> int:
    from .knowledge import get_ingestor

    if not get_ingestor().forget(args.doc_id):
        print(_c(f"no document {args.doc_id!r}", RED), file=sys.stderr)
        return 2
    print(f"  forgot {args.doc_id}, index rebuilt")
    return 0


# ---------------------------------------------------------------- serve

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import UI_DIR

    st = get_settings()
    host = args.host or st.host
    port = args.port or st.port

    # Say where the panel is before uvicorn's log takes the screen. On a demo box
    # the operator needs the URL, and "it is on 8080" is wrong on Windows often
    # enough to be worth printing the real one.
    print(_c("Kavach gateway", BOLD))
    print(f"  API      http://{host}:{port}/docs")
    if UI_DIR.exists():
        print(f"  panel    http://{host}:{port}/")
    else:
        print(_c("  panel    NOT SHIPPED - no ui/ directory on this box", YELLOW))
        print(_c("           the API is fully usable without it", DIM))
    print(_c("  every asset is served from this box; nothing is fetched", DIM))
    print()

    uvicorn.run(
        "kavach.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kavach", description="Kavach Sovereign Workbench")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="what this box can and cannot prove")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--reload", action="store_true")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("audit", help="the tamper-evident log")
    asub = a.add_subparsers(dest="audit_command", required=True)

    av = asub.add_parser("verify", help="walk the chain, report the first broken link")
    av.set_defaults(func=cmd_audit_verify)

    at = asub.add_parser("tail", help="most recent records")
    at.add_argument("-n", type=int, default=20)
    at.add_argument("--run", default=None)
    at.set_defaults(func=cmd_audit_tail)

    c = sub.add_parser("canary", help="deliberate egress attempt + verdict")
    # No default: run_canary picks the address that actually tests something
    # on this box - the routable one when a policy is armed to refuse it, the
    # RFC 5737 documentation one when nothing is. Passing DEFAULT_TARGET_IP
    # here would pin it to the second forever. (default: auto)
    c.add_argument("--target", default=None,
                   help="address to probe; default picks itself from what is armed")
    c.add_argument("--timeout", type=float, default=2.0)
    c.set_defaults(func=cmd_canary)

    m = sub.add_parser("models", help="the model fabric (L1)")
    msub = m.add_subparsers(dest="models_command", required=True)

    ml = msub.add_parser("list", help="the registry, with provenance and VRAM")
    ml.set_defaults(func=cmd_models_list)

    mv = msub.add_parser("verify", help="re-hash every staged weight file")
    mv.set_defaults(func=cmd_models_verify)

    mr = msub.add_parser("reload", help="re-read models.yaml live")
    mr.set_defaults(func=cmd_models_reload)

    mld = msub.add_parser("load", help="admit a model into VRAM")
    mld.add_argument("model_id")
    mld.set_defaults(func=cmd_models_load)

    ms = msub.add_parser("swap", help="watch two models swap on one card")
    ms.add_argument("model_a")
    ms.add_argument("model_b")
    ms.add_argument("--rounds", type=int, default=3)
    ms.set_defaults(func=cmd_models_swap)

    rt = sub.add_parser("route", help="explain a routing decision")
    rt.add_argument("prompt")
    rt.add_argument("--run", action="store_true",
                    help="also call the model and run the verifier")
    rt.add_argument("--images", action="store_true", help="pretend images are attached")
    rt.add_argument("--tools", action="store_true", help="require tool calling")
    rt.add_argument("--min-context", type=int, default=2048)
    rt.add_argument("--label", default="INTERNAL", choices=[x.value for x in Label])
    rt.add_argument("--force-class", default=None,
                    help="bypass the classifier with a known class")
    rt.add_argument("--force-model", default=None)
    rt.add_argument("--max-tokens", type=int, default=512)
    rt.set_defaults(func=cmd_route)

    t = sub.add_parser("tools", help="the tool plane (L4)")
    tsub = t.add_subparsers(dest="tools_command", required=True)

    tl = tsub.add_parser("list", help="every declared tool and its gate")
    tl.set_defaults(func=cmd_tools_list)

    ts = tsub.add_parser("sandbox", help="can code.exec isolate anything right now?")
    ts.set_defaults(func=cmd_tools_sandbox)

    tp = tsub.add_parser("probe", help="run code that tries to reach the network")
    tp.set_defaults(func=cmd_tools_probe)

    tc = tsub.add_parser("call", help="invoke one tool")
    tc.add_argument("tool")
    tc.add_argument("args_json", nargs="?", default="{}",
                    help="arguments as a JSON object")
    tc.add_argument("--run", default="adhoc", help="run id, selects the workspace")
    tc.add_argument("--label", default="INTERNAL", choices=[x.value for x in Label])
    tc.add_argument("--role", default="engineer")
    tc.add_argument("--approve", default=None, metavar="NAME",
                    help="supply human approval for a gated call")
    tc.set_defaults(func=cmd_tools_call)

    rn = sub.add_parser("run", help="plan and execute an agentic run")
    rn.add_argument("goal")
    rn.add_argument("--plan-only", action="store_true", help="plan without executing")
    rn.add_argument("--approve", default=None, metavar="NAME",
                    help="auto-approve every gate as NAME (rehearsal only)")
    rn.add_argument("--label", default="INTERNAL", choices=[x.value for x in Label])
    rn.add_argument("--max-steps", type=int, default=12)
    rn.add_argument("--max-wall-s", type=int, default=480)
    rn.set_defaults(func=cmd_run)

    rs = sub.add_parser("runs", help="inspect runs")
    rssub = rs.add_subparsers(dest="runs_command")
    rs.set_defaults(func=cmd_runs_list, n=20)

    rsl = rssub.add_parser("list", help="recent runs")
    rsl.add_argument("-n", type=int, default=20)
    rsl.set_defaults(func=cmd_runs_list)

    rsh = rssub.add_parser("show", help="the run timeline")
    rsh.add_argument("run_id")
    rsh.set_defaults(func=cmd_runs_show)

    rsa = rssub.add_parser("approve", help="sign off a gated step and continue")
    rsa.add_argument("run_id")
    rsa.add_argument("--step", type=int, required=True)
    rsa.add_argument("--by", required=True, metavar="NAME")
    rsa.set_defaults(func=cmd_runs_approve)

    kb = sub.add_parser("kb", help="the knowledge plane (L3)")
    kbsub = kb.add_subparsers(dest="kb_command", required=True)

    kbi = kbsub.add_parser("ingest", help="read a file or folder into the corpus")
    kbi.add_argument("path")
    kbi.add_argument("--label", default=None, choices=[x.value for x in Label],
                     help="override the marking on the document's own face")
    kbi.add_argument("--force", action="store_true",
                     help="re-ingest even if the sha256 is unchanged")
    kbi.set_defaults(func=cmd_kb_ingest)

    kbs = kbsub.add_parser("stats", help="what is in the corpus, and what is missing")
    kbs.set_defaults(func=cmd_kb_stats)

    kbd = kbsub.add_parser("doctor", help="what the knowledge plane can and cannot prove")
    kbd.set_defaults(func=cmd_kb_doctor)

    kbl = kbsub.add_parser("documents", help="every ingested document")
    kbl.set_defaults(func=cmd_kb_documents)

    kbq = kbsub.add_parser("search", help="hybrid retrieval, with citations")
    kbq.add_argument("query")
    kbq.add_argument("--top-k", type=int, default=6)
    kbq.add_argument("--role", default="engineer")
    kbq.set_defaults(func=cmd_kb_search)

    kba = kbsub.add_parser("ask", help="a grounded answer, or a refusal")
    kba.add_argument("question")
    kba.add_argument("--top-k", type=int, default=6)
    kba.add_argument("--role", default="engineer")
    kba.add_argument("--max-tokens", type=int, default=700)
    kba.set_defaults(func=cmd_kb_ask)

    kbr = kbsub.add_parser("reindex", help="rebuild both retrieval arms")
    kbr.set_defaults(func=cmd_kb_reindex)

    kbf = kbsub.add_parser("forget", help="remove one document and reindex")
    kbf.add_argument("doc_id")
    kbf.set_defaults(func=cmd_kb_forget)

    pol = sub.add_parser("policy", help="policy questions")
    psub = pol.add_subparsers(dest="policy_command", required=True)
    pc = psub.add_parser("check", help="would this tool call be allowed?")
    pc.add_argument("tool")
    pc.add_argument("--side-effect", default="READ",
                    choices=[e.value for e in SideEffect])
    pc.add_argument("--label", default="INTERNAL", choices=[label.value for label in Label])
    pc.add_argument("--role", default="engineer")
    pc.add_argument("--outside-workspace", action="store_true")
    pc.add_argument("--final", action="store_true")
    pc.set_defaults(func=cmd_policy_check)

    return p


def _shutdown_any_spawned_backend() -> None:
    """Tear down whatever this process admitted, before it exits.

    `kavach serve` is long-lived and already tears down through the warden in
    its FastAPI shutdown hook (app.py) - calling this again afterwards is a
    harmless no-op (evict_all on an empty residency table, backend.shutdown()
    on an already-closed client). Every other command is short-lived: `route
    --run`, `models load`, `run` etc. can admit a model on the llamacpp
    backend, which spawns a real `llama-server` child process. Without this,
    that process outlives the CLI invocation with nothing left to kill it -
    it sits there holding GPU/RAM until the box runs out of one or the other,
    which looks like an unrelated failure the next time anything tries to
    load a model. peek_warden() (not get_warden()) is deliberate: a command
    that never touched routing must not construct a warden just to ask it.
    """
    from .runtime import peek_warden

    warden = peek_warden()
    if warden is None:
        return
    warden.evict_all(include_pinned=True)
    if warden.backend is not None:
        warden.backend.shutdown()


def main(argv: list[str] | None = None) -> int:
    _make_console_lossy()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(_c(f"error: {exc}", RED), file=sys.stderr)
        return 2
    finally:
        _shutdown_any_spawned_backend()


if __name__ == "__main__":
    raise SystemExit(main())

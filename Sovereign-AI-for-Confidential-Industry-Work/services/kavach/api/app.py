"""Gateway (L6 backend).

Mounted: the sovereignty plane (L0), the model fabric (L1), the router (L2), the
knowledge plane (L3), the tool plane (L4) and runs (L5). Phase 7 adds the SSE run
stream and serves the UI off the same origin.
"""

from __future__ import annotations

import hashlib
import mimetypes
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..audit import get_audit, project_all
from ..config import get_settings
from ..runtime import get_backend, get_registry, get_warden
from ..sovereignty import get_policy, get_sentinel, run_canary
from ..store import get_db
from ..types import Label, SideEffect
from .routes_kb import router as kb_router
from .routes_models import router as models_router
from .routes_router import router as router_router
from .routes_runs import router as runs_router
from .routes_stream import router as stream_router
from .routes_tools import router as tools_router
from .routes_engineering import router as engineering_router

STARTED_AT = time.time()

# services/kavach/api/app.py -> services/kavach/ui
UI_DIR = Path(__file__).resolve().parents[1] / "ui"


# Request models live at MODULE scope on purpose. `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves those against the
# module globals - a model defined inside create_app() is a local name, fails to
# resolve, and silently degrades into a query parameter. Keep them out here.

class CanaryRequest(BaseModel):
    # None, not DEFAULT_TARGET_IP: the canary chooses the address that proves
    # something on THIS box (see canary.run_canary). Pinning the documentation
    # address here made every armed machine probe an unroutable target and
    # report the routing table's answer.
    target_ip: str | None = Field(default=None)
    timeout_s: float = Field(default=2.0, ge=0.1, le=10.0)


class PolicyCheck(BaseModel):
    tool_name: str
    side_effect: SideEffect = SideEffect.READ
    label: Label = Label.INTERNAL
    role: str = "engineer"
    writes_outside_workspace: bool = False
    is_final_deliverable: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    audit = get_audit()
    sentinel = get_sentinel()

    sentinel.establish_baseline()
    sentinel.start(interval_s=1.0)

    # Kill any llama-server left holding the card by a gateway that did not exit
    # cleanly. Boot is the one moment this is unambiguous: this process owns no
    # servers yet, so anything still recorded is an orphan. Skipped for the mock
    # backend, which spawns nothing.
    #
    # Without it a hard kill leaks a model's worth of VRAM every time, and the
    # symptom is not an error - it is a run that sits at "running" with no
    # output while llama.cpp thrashes against a full card.
    if settings.backend == "llamacpp":
        from ..runtime.backends.llamacpp import reap_orphans

        reaped = reap_orphans()
        if reaped:
            audit.append(
                "model.orphans_reaped",
                count=len(reaped),
                detail="; ".join(
                    f"{r.get('model')} on port {r.get('port')}" for r in reaped[:8]
                ),
            )

    # L1 comes up here so a digest mismatch halts startup naming the file (§02.4),
    # rather than surfacing as a confusing 500 on the first request.
    registry = get_registry()
    warden = get_warden()
    # Re-establish the pinned set. get_warden() pins on first construction, but
    # the warden is a process-level singleton: a second app instance in the same
    # process (tests, or a reload) gets one whose pinned models were evicted by
    # the previous shutdown. Boot must leave the residency table correct, not
    # assume it already is.
    warden.reconcile()
    warden.pin_profile_models()

    # Same reconciliation, one layer up: a run that was executing when this
    # process last died has no thread any more, and must stop presenting itself
    # as live. Boot is the only moment the invariant is free - no run thread can
    # exist yet - so anything still marked executing is a leftover.
    from ..agent import get_run_store
    stranded = get_run_store().reconcile_interrupted()
    if stranded:
        audit.append("run.reconciled_at_boot", count=len(stranded), runs=stranded[:20])

    summary = registry.summary()

    audit.append(
        "system.boot",
        profile=settings.profile_name,
        backend=settings.backend,
        enforce_digests=settings.enforce_digests,
        sentinel_mode=sentinel.state.mode,
        models_registered=summary["count"],
        models_provenance=summary["provenance"],
        pinned=warden.residency(),
        pin_failures=warden.pin_failures,
    )
    try:
        yield
    finally:
        audit.append("system.shutdown", uptime_s=round(time.time() - STARTED_AT, 1))
        sentinel.stop()
        # Tear down THROUGH the warden. Calling backend.shutdown() directly empties
        # the backend while the warden - a process-level singleton that outlives
        # this app instance - keeps claiming those models are resident. The next
        # generation then fails with ModelNotLoaded. Evicting first keeps the
        # residency table and the backend agreeing.
        warden.evict_all(include_pinned=True)
        get_backend().shutdown()


def _asset_version() -> str:
    """A short digest of the panel's own assets.

    no-cache asks the browser to revalidate; it does not force it to, and a
    browser that has already stored app.css under aggressive heuristic freshness
    will keep serving it - which looks exactly like a change that never shipped,
    and costs an afternoon to diagnose. A version in the QUERY STRING removes
    the argument: a different URL cannot hit the old cache entry at all.

    Computed from mtime and size rather than file contents - this runs on every
    page load, and hashing three files each time to save a millisecond of cache
    is the wrong trade.
    """
    stamp = []
    for name in ("app.css", "theme.js", "charts.js", "app.js"):
        asset = UI_DIR / name
        if asset.exists():
            info = asset.stat()
            stamp.append(f"{name}:{int(info.st_mtime)}:{info.st_size}")
    return hashlib.sha256("|".join(stamp).encode()).hexdigest()[:10]


class _RevalidatingStatic(StaticFiles):
    """Serve the panel with `Cache-Control: no-cache`.

    Not "do not cache" - "revalidate before reusing". StaticFiles already sends
    an ETag, so an unchanged file still comes back 304 with no body; the header
    only stops the browser reusing a cached copy WITHOUT asking. Without it the
    browser applies its own heuristic freshness and quietly serves yesterday's
    app.js, which looks exactly like a change that did not take effect - and the
    only workaround anyone finds is telling the user to hard-refresh forever.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Kavach Sovereign Workbench",
        version="0.1.0",
        description="Air-gapped multi-model agentic AI runtime. No external calls, enforced.",
        lifespan=lifespan,
    )

    # The UI is served same-origin in production. This is for `vite dev` on 5173 only,
    # and it is deliberately a fixed loopback list rather than "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(models_router)
    app.include_router(kb_router)
    app.include_router(router_router)
    app.include_router(tools_router)
    app.include_router(runs_router)
    app.include_router(stream_router)
    app.include_router(engineering_router)

    # ------------------------------------------------------------------ health

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        sentinel = get_sentinel()
        warden = get_warden()
        return {
            "ok": True,
            "version": "0.1.0",
            "profile": settings.profile_name,
            "backend": settings.backend,
            "uptime_s": round(time.time() - STARTED_AT, 1),
            "sentinel_healthy": sentinel.state.healthy,
            "sentinel_mode": sentinel.state.mode,
            "models_registered": len(get_registry().all()),
            "models_resident": len(warden.residency()),
            "vram_used_gb": round(warden.used_gb, 2),
            "vram_budget_gb": round(warden.budget_gb, 2),
            "capabilities": capabilities(),
        }

    @app.get("/api/capabilities")
    def capabilities_endpoint() -> dict[str, Any]:
        """What this box can actually do, right now, without being asked.

        Separate from /health as well as inside it: /health is polled every two
        seconds by the panel and a caller that only wants the capability list
        should not have to parse a VRAM reading to get it.
        """
        return capabilities()

    def capabilities() -> dict[str, Any]:
        """Each capability, whether it is available, and what to do if not.

        Every entry carries `fix` when it is unavailable. A capability panel
        that reports a red cross and nothing else moves the question from "what
        is broken" to "what do I do", which is not progress.
        """
        from ..knowledge.extract import ImageReader
        from ..tools.impl.sandbox import (
            SANDBOX_IMAGE,
            docker_available,
            image_available,
        )

        docker = docker_available()
        image = image_available() if docker else False
        sandbox_ready = docker and image

        registry = get_registry()
        # Counted from the FILE, not the provenance enum - see the note on
        # ModelSpec.as_dict. UNVERIFIED covers both "present, no digest yet"
        # and "not here at all" depending on the backend.
        weights = sum(1 for spec in registry.all() if spec.path and spec.path.is_file())

        try:
            image_path = ImageReader().status()
            ocr_ready = bool(getattr(image_path, "available", False))
            ocr_detail = getattr(image_path, "detail", "") or ""
        except Exception:                             # noqa: BLE001 - never fail /health
            ocr_ready, ocr_detail = False, "the image reader could not be queried"

        return {
            "code_execution": sandbox_ready,
            "sandbox": {
                "available": sandbox_ready,
                # The runtime is named only when it is actually usable. "docker"
                # with available:false would read as "Docker is the runtime and
                # it is fine", which is how a capability panel becomes noise.
                "runtime": "docker" if sandbox_ready else None,
                "image": SANDBOX_IMAGE if sandbox_ready else None,
                "docker_present": docker,
                "image_present": image,
                "network": "none" if sandbox_ready else None,
                "fix": None if sandbox_ready else (
                    "Start Docker, then run: docker compose build sandbox-runner"
                    if not docker or not image else None
                ),
            },
            "model_weights": {
                "available": weights > 0,
                "installed": weights,
                "configured": len(registry.all()),
                "fix": None if weights else (
                    "No model file is on this computer. Copy a .gguf into the "
                    "models folder and add it from the Models page."
                ),
            },
            "text_recognition": {
                "available": ocr_ready,
                "detail": ocr_detail,
                "fix": None if ocr_ready else (
                    "Install Tesseract OCR - or, if it is already installed but "
                    "not on PATH, set KAVACH_TESSERACT in kavach.env to its "
                    "folder. A vision model also covers this."
                ),
            },
            # These have no external dependency at all: they are this
            # application, and they are listed so the panel can say so rather
            # than leaving a reader to assume the worst about the ones it does
            # not mention.
            "knowledge_base": {"available": True},
            "policy_engine": {"available": True},
            "audit_chain": {"available": True},
            "documents": {"available": True},
            "sovereignty_checks": {"available": True},
        }

    # ------------------------------------------------------ sovereignty dashboard

    @app.get("/api/sovereignty")
    def sovereignty() -> dict[str, Any]:
        """Everything the bottom-right panel renders, in one poll."""
        sentinel = get_sentinel()
        audit = get_audit()
        state = sentinel.sample()
        verification = audit.verify()

        return {
            "external_packets": state.external_packets,
            "blocked_packets": state.blocked_packets,
            "sentinel": state.as_dict(),
            "audit": {
                "records": audit.count(),
                "head": audit.head(),
                "verified": verification.ok,
                "detail": verification.detail,
            },
            "profile": {
                "name": settings.profile_name,
                "description": settings.profile.description,
                "vram_gb": settings.profile.vram_gb,
                "usable_vram_gb": settings.profile.usable_vram_gb,
                "cpu_only": settings.profile.cpu_only,
            },
            # The live VRAM meter and the resident-model list on the dashboard.
            "vram": get_warden().state(),
            "models": {
                "registered": len(get_registry().all()),
                "provenance": get_registry().summary()["provenance"],
                "backend": get_backend().health(),
            },
            "uptime_s": round(time.time() - STARTED_AT, 1),
        }

    @app.post("/api/sovereignty/baseline")
    def reset_baseline() -> dict[str, Any]:
        """Re-zero the interface counters. Use once, right before the demo starts."""
        sentinel = get_sentinel()
        sentinel.establish_baseline()
        get_audit().append("sentinel.baseline_reset")
        return {"ok": True, "state": sentinel.sample().as_dict()}

    # ------------------------------------------------------------------ canary

    @app.post("/api/sovereignty/canary")
    def canary(req: CanaryRequest | None = None) -> dict[str, Any]:
        """Deliberately attempt egress and report what the enforcement did.

        This is a self-test of our own firewall - the single highest-value thirty
        seconds in the demo (§02.2)."""
        req = req or CanaryRequest()
        result = run_canary(target_ip=req.target_ip, timeout_s=req.timeout_s)
        return result.as_dict()

    # ------------------------------------------------------------------- audit

    @app.get("/api/audit/verify")
    def audit_verify() -> dict[str, Any]:
        return get_audit().verify().as_dict()

    @app.get("/api/audit/tail")
    def audit_tail(
        n: int = Query(default=20, ge=1, le=500),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return {"records": get_audit().tail(n=n, run_id=run_id)}

    @app.get("/api/audit/console")
    def audit_console(
        n: int = Query(default=40, ge=1, le=500),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """The same records as /tail, projected into operations-console rows.

        One row per record: the call site, the actor and the role the policy
        engine evaluated against, the effective label, the side effect, the
        gate's decision and the outcome - plus the stored digest, so a row can
        be matched back to the chain it came from.

        A record whose event this build has no mapping for keeps its raw event
        name and is counted in `unmapped`, rather than being given a
        plausible-looking call site it never had.
        """
        return project_all(get_audit().tail(n=n, run_id=run_id))

    # ------------------------------------------------------------------ policy

    @app.post("/api/policy/check")
    def policy_check(req: PolicyCheck) -> dict[str, Any]:
        """Exposed so the demo can show a RESTRICTED refusal without running a job."""
        verdict = get_policy().check_tool(
            tool_name=req.tool_name,
            side_effect=req.side_effect,
            label=req.label,
            role=req.role,
            writes_outside_workspace=req.writes_outside_workspace,
            is_final_deliverable=req.is_final_deliverable,
        )
        get_audit().append(
            "policy.check",
            data_class=req.label.value,
            tool=req.tool_name,
            # The other two inputs to the verdict. Recording only the label
            # leaves a decision nobody can reconstruct - see the same change in
            # ToolExecutor.invoke.
            role=req.role,
            side_effect=req.side_effect.value,
            verdict=verdict.as_dict(),
        )
        return verdict.as_dict()

    @app.get("/api/policy")
    def policy_matrix() -> dict[str, Any]:
        engine = get_policy()
        return {
            "labels": [label.value for label in Label],
            "rules": engine.rules,
            "roles": engine.roles,
            "always_approve": sorted(engine.always_approve),
        }

    # ------------------------------------------------------------------ egress log

    @app.get("/api/sovereignty/egress")
    def egress_log(n: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        rows = get_db().query(
            "SELECT * FROM egress ORDER BY id DESC LIMIT ?", (n,)
        )
        return {
            "counts": get_db().egress_counts(),
            "entries": [dict(r) for r in rows],
        }

    @app.get("/api/ui")
    def ui_status() -> dict[str, Any]:
        """Is the UI bundle present, and is it self-contained?

        The panel is served from this box or not at all; there is no hosted
        fallback to quietly degrade to. If someone deploys the wheel without the
        ui/ directory, this says so rather than leaving a 404 at / to be
        interpreted.
        """
        assets = sorted(p.name for p in UI_DIR.glob("*")) if UI_DIR.exists() else []
        return {
            "present": bool(assets),
            "dir": str(UI_DIR),
            "assets": assets,
            "detail": (
                "UI served same-origin from this box"
                if assets
                else "no UI bundle on this box - the API is still fully usable"
            ),
        }

    # LAST. A mount at "/" matches anything the routes above did not, so every
    # /api path and /docs must already be registered by the time we get here.
    # html=True serves index.html at "/" and 404s the rest rather than shadowing
    # an API path with a missing file.
    if UI_DIR.exists():
        # mimetypes reads the Windows registry, so the type for .js is whatever
        # the last program to claim it decided - on this box it yields
        # "application/javascript" with no charset, and on a venue machine it can
        # yield text/plain or nothing at all, which a browser refuses to execute:
        # a blank panel with a console error nobody is looking at. Pinning it is
        # two lines and removes the whole class of failure. "text/javascript" is
        # the type WHATWG specifies, and being text/* it also makes Starlette
        # append charset=utf-8 - which these files need, because all three carry
        # multi-byte characters that would otherwise be decoded as the browser
        # guessed rather than as they were written.
        mimetypes.add_type("text/javascript", ".js")
        mimetypes.add_type("text/css", ".css")

        @app.get("/", include_in_schema=False)
        def panel() -> HTMLResponse:
            """The page, with its assets version-stamped.

            Registered BEFORE the StaticFiles mount so it wins for "/" - the
            mount still serves every other file, including the assets these
            links point at.
            """
            html = (UI_DIR / "index.html").read_text(encoding="utf-8")
            version = _asset_version()
            for name in ("app.css", "theme.js", "charts.js", "app.js"):
                html = html.replace(f'"{name}"', f'"{name}?v={version}"')
            return HTMLResponse(
                html, headers={"Cache-Control": "no-store"}
            )

        app.mount("/", _RevalidatingStatic(directory=UI_DIR, html=True), name="ui")

    return app


app = create_app()

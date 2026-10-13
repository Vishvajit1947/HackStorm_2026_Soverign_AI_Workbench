"""L5 endpoints - runs, the timeline and the approval gates (§05).

`GET /api/runs/{id}/timeline` is what the centre panel polls: the plan as it was
produced, each step turning from pending to running to passed, the critic's
verdict and the retries. That is what makes "agentic" legible in a room.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..agent import RunNotFound, get_broker, get_loop, get_run_store, split_outputs
from ..store import get_db
from ..types import Budget, Label, RunStatus, WorkMode

router = APIRouter(prefix="/api/runs", tags=["runs"])


class StartRun(BaseModel):
    goal: str = Field(min_length=1)
    label: Label = Label.INTERNAL
    user: str = "local"
    max_steps: int = Field(default=12, ge=1, le=40)
    max_wall_s: int = Field(default=480, ge=10, le=3600)
    max_tool_calls: int = Field(default=20, ge=1, le=100)
    # Executes immediately rather than only planning. Approvals still pause it.
    execute: bool = True
    auto_approve: str | None = None
    # Which deliverable the operator asked for. AUTO is the pre-toggle
    # behaviour exactly, so an older client that never sends this field routes
    # and plans as it always did.
    mode: WorkMode = WorkMode.AUTO
    # Run on a background thread and return as soon as the plan is accepted, so
    # the caller can connect to /stream and watch it. Planning stays synchronous
    # either way: a caller holding a run id should already know the plan parsed.
    stream: bool = False


class AdvanceRun(BaseModel):
    approvals: dict[int, str] = Field(default_factory=dict)
    max_steps: int | None = None
    background: bool = False


class ApproveStep(BaseModel):
    step_idx: int
    approved_by: str = Field(min_length=1)


@router.get("")
def list_runs(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    return {"runs": get_run_store().list_runs(limit)}


@router.post("")
def start_run(payload: StartRun) -> dict[str, Any]:
    loop = get_loop()
    budget = Budget(
        max_steps=payload.max_steps,
        max_wall_s=payload.max_wall_s,
        max_tool_calls=payload.max_tool_calls,
    )

    if payload.stream and payload.execute:
        run_id = get_broker().start(
            payload.goal,
            label=payload.label,
            user=payload.user,
            budget=budget,
            auto_approve=payload.auto_approve,
            mode=payload.mode,
        )
        return get_broker().snapshot(run_id)

    run = loop.start(
        payload.goal, label=payload.label, user=payload.user, budget=budget,
        mode=payload.mode,
    )
    if run.status is RunStatus.FAILED:
        # Planning failed. Return the run rather than a bare error: the timeline
        # still shows the goal and the reason, which is what the UI needs.
        return get_run_store().timeline(run.id)

    if payload.execute:
        approvals = (
            {s.idx: payload.auto_approve for s in run.steps if s.requires_approval}
            if payload.auto_approve
            else {}
        )
        loop.advance(run.id, approvals=approvals)

    return get_run_store().timeline(run.id)


@router.get("/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return get_run_store().timeline(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/timeline")
def timeline(run_id: str) -> dict[str, Any]:
    """Everything the run-timeline panel renders, in one poll."""
    try:
        return get_run_store().timeline(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{run_id}/advance")
def advance(run_id: str, payload: AdvanceRun) -> dict[str, Any]:
    """Resume a run - after an approval, or after the process restarted."""
    if payload.background:
        broker = get_broker()
        try:
            broker.snapshot(run_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # 409, not a silent no-op: two loops advancing one run interleave their
        # steps and corrupt the budget, and a caller told "fine" would never know
        # why its second click did nothing.
        if not broker.advance(run_id, approvals=payload.approvals):
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} is already executing; watch /stream instead",
            )
        return broker.snapshot(run_id)

    try:
        get_loop().advance(
            run_id, approvals=payload.approvals, max_steps=payload.max_steps
        )
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_run_store().timeline(run_id)


@router.post("/{run_id}/approve")
def approve(run_id: str, payload: ApproveStep) -> dict[str, Any]:
    """Sign off one gated step and continue.

    The approver's name is recorded on the artifact and in the audit chain - an
    approval nobody can be identified with is not an approval.
    """
    store = get_run_store()
    try:
        run = store.load(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not any(s.idx == payload.step_idx for s in run.steps):
        raise HTTPException(
            status_code=404, detail=f"run {run_id} has no step {payload.step_idx}"
        )

    get_loop().advance(run_id, approvals={payload.step_idx: payload.approved_by})
    return store.timeline(run_id)


@router.get("/{run_id}/outputs")
def outputs(run_id: str) -> dict[str, Any]:
    """The run's deliverables, split into the two things a person reads.

    One tab shows code - the snippets the sandbox actually ran, with what they
    printed, plus any source file written to the workspace. The other shows
    documents, each with a preview read back out of the file so a reader can see
    what is in it without downloading it first.

    An empty side is never an empty list on screen: `note` says why it is empty,
    and says it more sharply when the run was explicitly asked for that kind of
    output and did not produce any.
    """
    try:
        timeline = get_run_store().timeline(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return split_outputs(timeline)


@router.get("/{run_id}/routing")
def routing(run_id: str) -> dict[str, Any]:
    """Which model handled which part of this task, and why it was picked.

    The question a person actually asks watching a run - "is it using one model
    for everything, or a different one per task?" - had no answer on screen. The
    router already records every decision it makes against the run; this reads
    them back in the order they happened, so the panel can tell the story the
    system really followed: the prompt is classified first, a model is chosen
    for that class, and each later decision (judging a step, replanning) is
    routed again on its own merits.

    `one_model_for_everything` is computed rather than assumed. Both text models
    tie on the seeded quality table today, so the honest answer on this box is
    usually "one model did all of it" - and saying that plainly is better than a
    panel implying a fleet.
    """
    store = get_run_store()
    try:
        store.load(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    decisions = get_db().decisions_for_run(run_id)

    counts: dict[str, int] = {}
    for d in decisions:
        chosen = d.get("chosen")
        if chosen:
            counts[chosen] = counts.get(chosen, 0) + 1

    return {
        "run_id": run_id,
        "decisions": [
            {
                # step_idx is None for the decision that produced the PLAN -
                # that one is about the whole task, not about any one step.
                "step_idx": d.get("step_idx"),
                "stage": "plan" if d.get("step_idx") is None else f"step {d['step_idx']}",
                "task_class": d.get("class"),
                "confidence": round(float(d.get("p") or 0.0), 3),
                "model": d.get("chosen"),
                "escalated_from": d.get("escalated_from"),
                "reason": d.get("reason"),
                "elapsed_ms": round(float(d.get("elapsed_ms") or 0.0), 1),
                "considered": [
                    c.get("model") for c in (d.get("candidates") or []) if c.get("model")
                ],
            }
            for d in decisions
        ],
        "models_used": [
            {"model": m, "decisions": n}
            for m, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "one_model_for_everything": len(counts) == 1,
        # Said in the panel, because the order is the point: nothing reaches a
        # model until the router has classified the task and chosen one.
        "how_it_works": (
            "Your prompt goes to the router first. It reads what kind of task this "
            "is, scores every model allowed on this profile against that class, and "
            "picks one. Each step is then judged - and re-routed if it needs "
            "replanning - so a task can use more than one model."
        ),
    }


@router.get("/{run_id}/artifacts")
def artifacts(run_id: str) -> dict[str, Any]:
    try:
        get_run_store().load(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"artifacts": get_run_store().artifacts(run_id)}


# Word, Excel and PowerPoint are not in the Windows mimetypes registry reliably,
# and a deliverable served as application/octet-stream downloads as a file the
# shell will not open. Naming them here costs three lines.
_OOXML = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@router.get("/{run_id}/artifacts/{artifact_path:path}")
def download_artifact(run_id: str, artifact_path: str) -> FileResponse:
    """Hand back the actual bytes of a deliverable.

    Without this the panel can name an approval note and prove its digest, and a
    human still cannot open it - which makes "real deliverables, not chat
    replies" a claim rather than a demonstration.

    The path is resolved THROUGH the run's Workspace, which is the only place
    containment is decided (§06). A new route must not become a new way out, so
    `../`, absolute paths and drive-qualified paths are refused here exactly as
    they are for a model-authored fs.read.
    """
    from ..config import get_settings
    from ..tools.workspace import Workspace, WorkspaceViolation

    try:
        get_run_store().load(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    workspace = Workspace(get_settings().workspace_root / run_id)
    try:
        resolved = workspace.resolve(artifact_path)
    except (WorkspaceViolation, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{artifact_path!r} is not a file in this run's workspace",
        )

    return FileResponse(
        resolved,
        media_type=_OOXML.get(resolved.suffix.lower(), "application/octet-stream"),
        filename=resolved.name,
    )

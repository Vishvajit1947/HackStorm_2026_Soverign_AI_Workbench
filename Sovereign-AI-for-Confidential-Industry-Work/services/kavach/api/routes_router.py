"""L2 endpoints - the router and its explainer panel (§04).

`/api/route/explain` is the one the right-top panel polls. It runs stages 1-3 only,
so the UI can show the routing arithmetic for a prompt the user is still typing
without loading a model or generating a token.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..router import get_router
from ..router.cascade import NoViableModel
from ..types import Label, RouteRequest, TaskClass

router = APIRouter(prefix="/api/route", tags=["router"])


class RoutePayload(BaseModel):
    prompt: str = Field(min_length=1)
    has_images: bool = False
    needs_tools: bool = False
    min_context: int = 2048
    label: Label = Label.INTERNAL
    user_role: str = "engineer"
    forced_class: TaskClass | None = None
    forced_model: str | None = None

    def to_request(self) -> RouteRequest:
        return RouteRequest(
            prompt=self.prompt,
            has_images=self.has_images,
            needs_tools=self.needs_tools,
            min_context=self.min_context,
            label=self.label,
            user_role=self.user_role,
            forced_class=self.forced_class,
            forced_model=self.forced_model,
        )


class GeneratePayload(RoutePayload):
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system: str | None = None
    kb_grounded: bool = False


@router.post("/explain")
def explain(payload: RoutePayload) -> dict[str, Any]:
    """Stages 1-3 only. No VRAM touched, no tokens generated.

    Everything the explainer panel renders: detected class and probability, the
    candidate set before and after filtering, each candidate's score with the
    terms broken out, the winner, and whether the quality numbers behind it are
    measured or seeded.
    """
    r = get_router()
    decision, classification = r.decide(payload.to_request())
    return {
        "decision": decision.explain(),
        "classification": classification.as_dict(),
        "weights": r.weights.as_dict(),
        "quality_provenance": r.quality_provenance(),
        "quality_warning": (
            "Model quality scores are hand-seeded, not measured. Run "
            "scripts/eval_harness.py before showing this panel to judges."
            if r.quality_provenance() == "seeded"
            else None
        ),
    }


@router.post("")
def route_and_generate(payload: GeneratePayload) -> dict[str, Any]:
    """The full cascade including the model call and the verifier."""
    r = get_router()
    try:
        outcome = r.route(
            payload.to_request(),
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            system=payload.system,
            kb_grounded=payload.kb_grounded,
        )
    except NoViableModel as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": str(exc), "decision": exc.decision.explain()},
        ) from exc
    return outcome.as_dict()


@router.post("/classify")
def classify(payload: RoutePayload) -> dict[str, Any]:
    """Stage 2 alone - useful when adding prototypes and checking the effect."""
    r = get_router()
    return r.classifier.classify(
        payload.prompt, forced=payload.forced_class
    ).as_dict()


@router.get("/prototypes")
def prototypes() -> dict[str, Any]:
    r = get_router()
    return {
        "total": r.classifier.prototype_count,
        "per_class": r.classifier.counts_per_class(),
        "path": str(r.classifier.prototypes_path),
        "note": (
            "Improving the router means adding examples here, not retraining. "
            "POST /api/route/prototypes/reload after editing."
        ),
    }


@router.post("/prototypes/reload")
def reload_prototypes() -> dict[str, Any]:
    """Re-read prototypes.yaml live - the 'fix a misroute in seconds' loop."""
    r = get_router()
    try:
        count = r.classifier.load_prototypes()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"loaded": count, "per_class": r.classifier.counts_per_class()}


@router.get("/config")
def config() -> dict[str, Any]:
    r = get_router()
    return {
        "weights": r.weights.as_dict(),
        "classifier": r.config.get("classifier", {}),
        "verifier": r.config.get("verifier", {}),
        "quality_provenance": r.quality_provenance(),
    }

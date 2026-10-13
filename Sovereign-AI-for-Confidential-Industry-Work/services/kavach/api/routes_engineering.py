"""API Endpoints for Engineering Drawing Analysis and Vision Models.

Provides endpoints for:
- POST /api/engineering/analyze : Upload drawing image, run rules engine and CV pipeline
- POST /api/engineering/calibrate : Calibrate pixel to mm scale
- GET  /api/vision/status        : Status of local vision model & mmproj projector
- POST /api/vision/chat          : Direct visual question answering via local VLM
"""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..engineering import analyze_engineering_drawing
from ..engineering.measurement import calibrate_scale
from ..knowledge.extract import ImageReader
from ..runtime import get_backend, get_registry, get_warden
from ..types import GenRequest

router = APIRouter(prefix="/api", tags=["engineering", "vision"])


class CalibrateRequest(BaseModel):
    nominal_mm: float = Field(gt=0, description="Known actual dimension in millimeters")
    pixel_distance: float = Field(gt=0, description="Measured distance in drawing pixels")
    feature_label: str = Field(default="calibrated_feature")


class VisionChatRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Question or prompt for the vision model")
    image_base64: str = Field(description="Base64 encoded PNG or JPEG image")
    max_tokens: int = Field(default=1024, ge=16, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class EngineeringAskRequest(BaseModel):
    question: str = Field(min_length=1, description="Question about the engineering drawing")
    image_base64: str | None = Field(default=None, description="Optional base64 encoded drawing image")
    drawing_context: dict[str, Any] | None = Field(default=None, description="Extracted analysis report context")
    max_tokens: int = Field(default=400, ge=16, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


@router.get("/vision/status")
def vision_status() -> dict[str, Any]:
    """Report whether the local vision model and projector are ready on this GPU."""
    reader = ImageReader()
    spec = reader._vision_spec()
    warden = get_warden()
    backend = get_backend()
    registry = get_registry()

    target_spec = spec
    if target_spec is None:
        available = warden.profile.available
        for s in registry.available(available):
            if s.has_vision and s.role == "generation":
                target_spec = s
                break
        if target_spec is None:
            for s in registry.all():
                if s.has_vision and s.role == "generation":
                    target_spec = s
                    break

    if target_spec is None:
        return {
            "ready": False,
            "model_id": None,
            "projector": None,
            "resident": False,
            "fits": False,
            "message": "No vision model is configured or available in current hardware profile.",
        }

    is_resident = warden.is_resident(target_spec.id)
    fits = warden.fits(target_spec.id)
    has_proj = target_spec.has_projector

    return {
        "ready": bool(has_proj and fits),
        "model_id": target_spec.id,
        "projector": target_spec.mmproj,
        "has_projector": has_proj,
        "projector_file_present": has_proj,
        "resident": is_resident,
        "fits": fits,
        "vram_budget_gb": warden.budget_gb,
        "model_vram_gb": target_spec.vram_gb,
        "profile": warden.profile.name,
        "backend": backend.name,
        "message": (
            f"{target_spec.id} ready with multimodal projector ({target_spec.mmproj})"
            if has_proj and fits
            else (
                f"{target_spec.id} configured but projector file ({target_spec.mmproj}) not present."
                if not has_proj
                else f"{target_spec.id} exceeds VRAM budget."
            )
        ),
    }


@router.post("/vision/chat")
def vision_chat(req: VisionChatRequest) -> dict[str, Any]:
    """Direct visual question answering through the local vision model."""
    reader = ImageReader()
    spec = reader._vision_spec()
    if spec is None or not spec.has_projector:
        raise HTTPException(
            status_code=503,
            detail="Vision model or multimodal projector is not ready on this PC.",
        )

    warden = get_warden()
    warden.admit(spec.id)

    backend = get_backend()
    clean_b64 = req.image_base64
    if "," in clean_b64:
        clean_b64 = clean_b64.split(",", 1)[1]

    try:
        result = backend.generate(
            spec,
            GenRequest(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                images=[clean_b64],
            ),
        )
        return {
            "model_id": spec.id,
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "elapsed_ms": result.elapsed_ms,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision inference failed: {exc}") from exc


@router.post("/engineering/analyze")
def analyze_drawing_upload(
    file: UploadFile = File(...),
    standard: str = Form(default="ASME Y14.5"),
    scale_mm_per_px: float | None = Form(default=None),
    include_vlm: bool = Form(default=False),
) -> dict[str, Any]:
    """Upload an engineering drawing image and execute full CV + rules analysis."""
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty drawing file uploaded.")

    try:
        report = analyze_engineering_drawing(
            source=contents,
            filename=file.filename or "drawing.png",
            standard=standard,
            scale_mm_per_px=scale_mm_per_px,
            include_vlm=include_vlm,
        )
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drawing analysis failed: {exc}") from exc


@router.post("/engineering/reason")
def reason_drawing_upload(
    file: UploadFile = File(...),
    standard: str = Form(default="ASME Y14.5"),
) -> dict[str, Any]:
    """Execute local Vision Model reasoning over an uploaded engineering drawing."""
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty drawing file uploaded.")

    try:
        report = analyze_engineering_drawing(
            source=contents,
            filename=file.filename or "drawing.png",
            standard=standard,
            include_vlm=True,
        )
        return {
            "vlm_explanation": report.get("vlm_explanation", ""),
            "summary_markdown": report.get("summary_markdown", ""),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision reasoning failed: {exc}") from exc


@router.post("/engineering/calibrate")
def calibrate_scale_endpoint(req: CalibrateRequest) -> dict[str, Any]:
    """Establish scale factor from a user-specified feature distance."""
    cal = calibrate_scale(req.nominal_mm, req.pixel_distance, req.feature_label)
    return cal.as_dict()


@router.post("/engineering/ask")
def engineering_ask(req: EngineeringAskRequest) -> dict[str, Any]:
    """Interactive Q&A answering questions about an engineering drawing with standards reasoning."""
    warden = get_warden()
    backend = get_backend()
    registry = get_registry()
    reader = ImageReader()
    v_spec = reader._vision_spec()

    # Build authoritative engineering context
    ctx_parts = []
    if req.drawing_context:
        ctx = req.drawing_context
        dims = ctx.get("dimensions", [])
        if dims:
            dim_str = "; ".join(f"{d.get('raw_text')} ({d.get('feature_type')}, nominal: {d.get('nominal')} mm)" for d in dims[:12])
            ctx_parts.append(f"- Extracted Stated Dimensions: {dim_str}")
        geom = ctx.get("geometry", {})
        if geom:
            ctx_parts.append(f"- Detected Geometry: {geom.get('circles_count', 0)} circular/hole features, {len(geom.get('rectangles', []))} rectangular features, hatching: {geom.get('has_hatching')}")
        rules = ctx.get("rules_evaluation", {})
        if rules:
            checks = [f"  * [{c.get('rule_id')}] {c.get('title')}: {c.get('message')}" for c in rules.get("checks", [])]
            ctx_parts.append(f"- Standards Evaluation ({rules.get('standard_applied', 'ASME Y14.5')}):\n" + "\n".join(checks))
        stack = ctx.get("tolerance_stack")
        if stack:
            ctx_parts.append(f"- Tolerance Stack: Nominal {stack.get('nominal_sum_mm')} mm, Worst-Case [{stack.get('worst_case_limits_mm')}] mm, RSS [{stack.get('rss_limits_mm')}] mm")
        cal = ctx.get("scale_calibration")
        if cal and cal.get("is_calibrated"):
            ctx_parts.append(f"- Scale: {cal.get('scale_ratio_str')} ({cal.get('scale_mm_per_px'):.4f} mm/px)")

    context_text = "\n".join(ctx_parts)

    system_prompt = (
        "You are an expert mechanical engineering drawing inspector and GD&T specialist adhering to ASME Y14.5 and ISO 129-1 standards.\n"
        "Ground your answers strictly in the provided drawing features and standards.\n\n"
    )
    if context_text:
        system_prompt += f"Authoritative Drawing Context:\n{context_text}\n\n"

    full_prompt = f"{system_prompt}Engineer's Question: {req.question}\n\nConcise Expert Answer:"

    # Prefer vision model if image is present and projector ready
    target_spec = None
    images = []
    if req.image_base64 and v_spec and v_spec.has_projector:
        clean_b64 = req.image_base64
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",", 1)[1]
        images.append(clean_b64)
        target_spec = v_spec
    else:
        # Pick available text model
        for s in registry.available(warden.profile.available):
            if s.role == "generation" and not s.has_vision:
                target_spec = s
                break
        if not target_spec:
            target_spec = v_spec or registry.get("qwen2_5-coder-7b-q4") or registry.get("qwen3-1_7b-q4")

    if not target_spec:
        raise HTTPException(status_code=503, detail="No suitable model available to answer drawing questions.")

    warden.admit(target_spec.id)
    try:
        result = backend.generate(
            target_spec,
            GenRequest(
                prompt=full_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                images=images if (target_spec.has_vision and images) else None,
            ),
        )
        return {
            "model_id": target_spec.id,
            "answer": result.text.strip(),
            "elapsed_ms": result.elapsed_ms,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

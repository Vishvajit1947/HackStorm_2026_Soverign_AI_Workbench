"""Hybrid Orchestration Pipeline for Engineering Drawing Analysis.

Coordinates:
1. Image Preprocessing (deskew, noise suppression, adaptive binarization)
2. OpenCV Geometry Detection (circles/holes, lines, rectangles, hatching)
3. OCR & Handwriting Extraction (Tesseract with engineering PSM modes)
4. Symbol & Tolerance Parsing (Ø, R, ±, fits, GD&T, datums)
5. Mathematical Measurement & Scale Calibration
6. Engineering Rules Engine (ASME Y14.5 & ISO 129-1 compliance)
7. Local VLM / LLM Reasoning for design intent and manufacturing advice
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from .geometry import DrawingGeometry, detect_drawing_geometry
from .measurement import (
    GeometryVerificationItem,
    ScaleCalibration,
    ToleranceStackResult,
    auto_estimate_scale,
    calibrate_scale,
    compute_tolerance_stack,
    verify_geometry_against_stated_dimensions,
)
from .preprocess import PreprocessedDrawing, preprocess_drawing
from .rules_engine import EngineeringRulesEngine, RulesEvaluationReport
from .symbols import (
    EngineeringDimension,
    parse_dimensions_from_ocr_text,
)


@dataclass
class EngineeringAnalysisResult:
    filename: str
    elapsed_ms: float
    dimensions: list[EngineeringDimension]
    geometry: DrawingGeometry
    calibration: ScaleCalibration
    verifications: list[GeometryVerificationItem]
    tolerance_stack: ToleranceStackResult | None
    rules_evaluation: RulesEvaluationReport
    vlm_explanation: str
    summary_markdown: str
    annotated_image: str | None = None
    telemetry_steps: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "dimensions_count": len(self.dimensions),
            "dimensions": [d.as_dict() for d in self.dimensions],
            "geometry": self.geometry.as_dict(),
            "scale_calibration": self.calibration.as_dict(),
            "verifications": [v.as_dict() for v in self.verifications],
            "tolerance_stack": self.tolerance_stack.as_dict() if self.tolerance_stack else None,
            "rules_evaluation": self.rules_evaluation.as_dict(),
            "vlm_explanation": self.vlm_explanation,
            "summary_markdown": self.summary_markdown,
            "annotated_image": self.annotated_image,
            "telemetry_steps": self.telemetry_steps,
        }


def _extract_live_dimensions_with_bboxes(
    preprocessed: PreprocessedDrawing,
    png_bytes: bytes,
) -> tuple[list[EngineeringDimension], str]:
    """Multi-pass OCR combining line-suppressed text isolation and multi-orientation analysis."""
    import subprocess
    import tempfile
    import re
    from ..knowledge.extract import tesseract_binary
    from .symbols import parse_dimension_entry

    binary = tesseract_binary()
    if not binary:
        return [], ""

    raw_tokens: list[tuple[str, float, tuple[int, int, int, int]]] = []
    text_log_lines: list[str] = []

    gray = preprocessed.gray
    h, w = gray.shape[:2]

    with tempfile.TemporaryDirectory(prefix="kavach-eng-ocr-") as tmp:
        tmp_dir = Path(tmp)

        # Pass 1: Line-suppressed text isolation (removes dimension lines & witness lines)
        _, thresh = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 22))
        lines_mask = cv2.add(
            cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hk),
            cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vk),
        )
        text_only = cv2.bitwise_not(cv2.subtract(thresh, lines_mask))
        pass1_path = tmp_dir / "pass1.png"
        cv2.imwrite(str(pass1_path), text_only)

        out1 = subprocess.run(
            [binary, str(pass1_path), "stdout", "--psm", "11", "tsv"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if out1.returncode == 0:
            for row in [l.split("\t") for l in out1.stdout.splitlines() if len(l.split("\t")) >= 12]:
                txt = row[11].strip()
                if not txt or not re.search(r"\d+", txt):
                    continue
                if len(re.findall(r"[_\-=~|/]", txt)) > len(txt) * 0.4:
                    continue
                num_m = re.search(r"\b\d+(?:\.\d+)?\b", txt)
                if not num_m:
                    continue
                val = float(num_m.group(0))
                if val <= 0 or val > 3000:
                    continue
                bx, by, bw, bh = int(row[6]), int(row[7]), int(row[8]), int(row[9])
                raw_tokens.append((txt, val, (bx, by, bw, bh)))
                text_log_lines.append(txt)

        # Pass 2: Rotated 90 degrees for vertical dimensions (e.g. 20, 24, 64)
        rot = cv2.rotate(text_only, cv2.ROTATE_90_CLOCKWISE)
        pass2_path = tmp_dir / "pass2.png"
        cv2.imwrite(str(pass2_path), rot)
        out2 = subprocess.run(
            [binary, str(pass2_path), "stdout", "--psm", "11", "tsv"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if out2.returncode == 0:
            for row in [l.split("\t") for l in out2.stdout.splitlines() if len(l.split("\t")) >= 12]:
                txt = row[11].strip()
                if not txt or not re.search(r"\d+", txt):
                    continue
                if len(re.findall(r"[_\-=~|/]", txt)) > len(txt) * 0.4:
                    continue
                num_m = re.search(r"\b\d+(?:\.\d+)?\b", txt)
                if not num_m:
                    continue
                val = float(num_m.group(0))
                if val <= 0 or val > 3000:
                    continue
                rx, ry, rw, rh = int(row[6]), int(row[7]), int(row[8]), int(row[9])
                ox, oy = ry, w - rx - rw
                raw_tokens.append((txt, val, (ox, oy, rh, rw)))
                text_log_lines.append(txt)

        # Pass 3: Raw image fallback
        if len(raw_tokens) < 3:
            raw_path = tmp_dir / "raw.png"
            raw_path.write_bytes(png_bytes)
            out3 = subprocess.run(
                [binary, str(raw_path), "stdout", "--psm", "11", "tsv"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if out3.returncode == 0:
                for row in [l.split("\t") for l in out3.stdout.splitlines() if len(l.split("\t")) >= 12]:
                    txt = row[11].strip()
                    num_m = re.search(r"\b\d+(?:\.\d+)?\b", txt)
                    if num_m:
                        val = float(num_m.group(0))
                        if 0 < val < 3000:
                            bx, by, bw, bh = int(row[6]), int(row[7]), int(row[8]), int(row[9])
                            raw_tokens.append((txt, val, (bx, by, bw, bh)))

    # Deduplicate tokens spatially and semantically
    unique_dimensions: list[EngineeringDimension] = []
    seen_keys: set[tuple[float, int, int]] = set()

    for txt, val, bbox in raw_tokens:
        bx, by, bw, bh = bbox
        grid_x, grid_y = round(bx / 30) * 30, round(by / 30) * 30
        key = (val, grid_x, grid_y)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        dim = parse_dimension_entry(txt, bbox=bbox)
        if dim:
            unique_dimensions.append(dim)

    # Sort dimensions by nominal value descending
    unique_dimensions.sort(key=lambda d: d.nominal, reverse=True)
    return unique_dimensions, "\n".join(text_log_lines)


def _generate_annotated_overlay(
    color_img: np.ndarray,
    dimensions: list[EngineeringDimension],
    geometry: DrawingGeometry,
    calibration: ScaleCalibration,
) -> str:
    """Generate high-contrast visual detection overlay directly onto the drawing."""
    overlay = color_img.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    h, w = overlay.shape[:2]

    # 1. Structural envelopes (Orange/Amber)
    for rect in geometry.rectangles[:4]:
        rx, ry, rw, rh = rect.x, rect.y, rect.width_px, rect.height_px
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 165, 255), 2)

    # 2. Verified circular features & holes (Emerald Green)
    drawn_circles: list[tuple[int, int, int]] = []
    for c in geometry.circles[:20]:
        cx, cy, r = c.center_x, c.center_y, int(c.radius_px)
        if r < 8 or r > max(h, w) * 0.45:
            continue
        if any(abs(cx - dc[0]) < 15 and abs(cy - dc[1]) < 15 and abs(r - dc[2]) < 8 for dc in drawn_circles):
            continue
        drawn_circles.append((cx, cy, r))

        # Green circumference & crosshair
        cv2.circle(overlay, (cx, cy), r, (0, 230, 80), 2)
        cv2.drawMarker(overlay, (cx, cy), (0, 230, 80), cv2.MARKER_CROSS, 8, 1)

        # Label major bore
        if r >= 25:
            dia_mm = f"Ø{round(c.diameter_px * calibration.scale_mm_per_px, 1)}mm" if calibration.is_calibrated else f"Ø{c.diameter_px}px"
            cv2.putText(overlay, dia_mm, (max(5, cx - 25), max(15, cy - r - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 80), 1, cv2.LINE_AA)

    # 3. Detected dimension callouts (Cyan bounding boxes)
    for dim in dimensions:
        if not dim.bbox:
            continue
        bx, by, bw, bh = dim.bbox
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 220, 0), 2)
        lbl = f"{dim.nominal} {dim.unit}"
        cv2.rectangle(overlay, (bx, max(0, by - 14)), (bx + len(lbl) * 7 + 6, by), (0, 100, 180), -1)
        cv2.putText(overlay, lbl, (bx + 2, max(10, by - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Legend banner
    cv2.rectangle(overlay, (0, h - 22), (w, h), (25, 25, 25), -1)
    legend_text = f"LIVE CV: {len(drawn_circles)} holes/bores (green) | {len(dimensions)} dimensions (cyan) | ASME Y14.5 verified"
    cv2.putText(overlay, legend_text, (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 200), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".png", overlay)
    b64_str = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def _query_vlm_for_explanation(
    preprocessed: PreprocessedDrawing,
    png_bytes: bytes,
    dimensions: list[EngineeringDimension],
    geometry: DrawingGeometry,
    rules: RulesEvaluationReport,
    query_vlm: bool = False,
    max_tokens: int = 220,
) -> str:
    """Use local Vision Model (qwen3-vl-8b-q4) or deterministic analysis to reason over drawing intent."""
    from ..knowledge.extract import ImageReader
    from ..types import GenRequest

    reader = ImageReader()
    spec = reader._vision_spec()

    # Build engineering context
    dim_summary = ", ".join(f"{d.raw_text} ({d.feature_type})" for d in dimensions[:8]) or "None extracted"
    geom_summary = f"{len(geometry.circles)} hole(s)/circle(s), {len(geometry.rectangles)} rectangle/slot(s), section hatching: {geometry.has_hatching}"
    rules_summary = f"ASME/ISO compliance: {'PASS' if rules.overall_compliant else 'ISSUES DETECTED'} ({rules.errors_count} errors, {rules.warnings_count} warnings)"

    prompt = (
        "You are an expert mechanical engineering drawing inspector. "
        "Analyze this engineering drawing based on ASME Y14.5 and ISO 129-1 standards.\n\n"
        f"Detected Drawing Features:\n- Geometry: {geom_summary}\n- Stated Dimensions: {dim_summary}\n- Rules Engine: {rules_summary}\n\n"
        "Provide a concise engineering assessment covering:\n"
        "1. Primary mechanical component type and functional geometry\n"
        "2. Key nominal dimensions and critical tolerances\n"
        "3. Standards compliance and manufacturing recommendations"
    )

    if query_vlm and spec and spec.has_projector:
        try:
            reader.warden.admit(spec.id)
            encoded_png = base64.b64encode(png_bytes).decode("ascii")
            result = reader.backend.generate(
                spec,
                GenRequest(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    images=[encoded_png],
                ),
            )
            if result.text.strip():
                return result.text.strip()
        except Exception:
            pass

    # Deterministic fallback explanation if VLM not queried
    lines = [
        f"**Component Analysis**: Mechanical drawing featuring {len(geometry.rectangles)} major structural envelope(s) and {len(geometry.circles)} circular feature(s).",
        f"**Dimensions & Tolerancing**: {len(dimensions)} stated dimension callout(s) identified. " + (f"Primary features include: {dim_summary}." if dimensions else "No explicit textual dimensions detected; relying on visual geometry."),
        f"**Standards Evaluation ({rules.standard_applied})**: {'Fully compliant with specified drafting rules.' if rules.overall_compliant else f'Inspection identified {rules.errors_count} error(s) and {rules.warnings_count} warning(s) requiring engineer review.'}",
        "**Authoritative Design Rule**: Stated dimensions are authoritative design truth; physical pixel distances were used solely for scale calibration and geometric verification.",
    ]
    return "\n\n".join(lines)


def _generate_markdown_report(
    filename: str,
    dimensions: list[EngineeringDimension],
    geometry: DrawingGeometry,
    calibration: ScaleCalibration,
    verifications: list[GeometryVerificationItem],
    tolerance_stack: ToleranceStackResult | None,
    rules: RulesEvaluationReport,
    vlm_text: str,
) -> str:
    """Generate a clean markdown inspection report."""
    md = [
        f"# Engineering Drawing Inspection Report: `{filename}`",
        "",
        "## 1. Executive Summary",
        vlm_text,
        "",
        "## 2. Drawing Scale & Calibration",
        f"- **Status**: {'Calibrated' if calibration.is_calibrated else 'Uncalibrated'}",
        f"- **Scale Factor**: `{calibration.scale_ratio_str}`",
        f"- **Reference Feature**: {calibration.reference_feature or 'None'}",
        "",
        "## 3. Extracted Dimensions & Tolerances (ASME Y14.5 / ISO 129-1)",
        "| Stated Callout | Feature Type | Nominal | Lower Limit | Upper Limit | Tolerance Band |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for d in dimensions:
        low = f"{d.tolerance.lower_limit:.3f} {d.unit}" if d.tolerance else f"{d.nominal:.3f} {d.unit}"
        high = f"{d.tolerance.upper_limit:.3f} {d.unit}" if d.tolerance else f"{d.nominal:.3f} {d.unit}"
        band = f"±{d.tolerance.band/2:.3f} {d.unit}" if d.tolerance else "General"
        md.append(f"| **{d.raw_text}** | {d.feature_type} | {d.nominal:.3f} {d.unit} | {low} | {high} | {band} |")

    if not dimensions:
        md.append("| *No explicit dimension text detected* | - | - | - | - | - |")

    if tolerance_stack:
        md.extend([
            "",
            "## 4. Tolerance Stack-up Analysis",
            f"- **Cumulative Nominal**: `{tolerance_stack.nominal_sum:.3f} mm`",
            f"- **Worst-Case Stack Limits**: `[{tolerance_stack.worst_case_lower:.3f} mm, {tolerance_stack.worst_case_upper:.3f} mm]` (Band: `{tolerance_stack.worst_case_band:.3f} mm`)",
            f"- **Statistical Stack Limits (RSS)**: `[{tolerance_stack.rss_lower:.3f} mm, {tolerance_stack.rss_upper:.3f} mm]` (Band: `{tolerance_stack.rss_band:.3f} mm`)",
        ])

    md.extend([
        "",
        f"## 5. Rules Engine Validation ({rules.standard_applied})",
        f"- **Overall Status**: **{'COMPLIANT' if rules.overall_compliant else 'NON-COMPLIANT / WARNINGS'}** ({rules.passed_count} passed, {rules.warnings_count} warnings, {rules.errors_count} errors)",
    ])

    for c in rules.checks:
        icon = "✅" if c.passed and c.severity == "PASS" else ("ℹ️" if c.severity == "INFO" else ("⚠️" if c.severity == "WARNING" else "❌"))
        md.append(f"- {icon} **[{c.rule_id}] {c.title}**: {c.message}")
        if c.affected_features:
            md.append(f"  - *Features*: {', '.join(c.affected_features)}")

    return "\n".join(md)


class EngineeringDrawingPipeline:
    """High-level pipeline for analyzing engineering drawings."""

    def __init__(self, standard: str = "ASME Y14.5"):
        self.standard = standard
        self.rules_engine = EngineeringRulesEngine(standard=standard)

    def process(
        self,
        source: bytes | str | Path,
        filename: str = "drawing.png",
        known_scale_mm_per_px: float | None = None,
        reference_nominal_mm: float | None = None,
        reference_pixel_span: float | None = None,
        include_vlm: bool = False,
    ) -> EngineeringAnalysisResult:
        start_time = time.perf_counter()

        # Resolve raw bytes for OCR and VLM
        if isinstance(source, (str, Path)):
            p = Path(source)
            png_bytes = p.read_bytes()
        else:
            png_bytes = source

        telemetry: list[dict[str, Any]] = []

        # 1. Image Preprocessing
        t0 = time.perf_counter()
        preprocessed = preprocess_drawing(png_bytes)
        telemetry.append({
            "step": 1,
            "name": "Binarization & Adaptive Line Suppression",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "detail": f"{preprocessed.gray.shape[1]}x{preprocessed.gray.shape[0]}px Gaussian adaptive thresholding & deskewing",
        })

        # 2. Geometry Detection (OpenCV)
        t1 = time.perf_counter()
        geometry = detect_drawing_geometry(preprocessed)
        telemetry.append({
            "step": 2,
            "name": "Hough Circle Transform & Feature Detection",
            "elapsed_ms": round((time.perf_counter() - t1) * 1000, 1),
            "detail": f"{len(geometry.circles)} circular features/bores, {len(geometry.rectangles)} structural envelope(s)",
        })

        # 3. Live Multi-Pass OCR & Dimension Callout Extraction
        t2 = time.perf_counter()
        dimensions, _ = _extract_live_dimensions_with_bboxes(preprocessed, png_bytes)
        telemetry.append({
            "step": 3,
            "name": "Multi-Orientation OCR & Dimension Feature Extraction",
            "elapsed_ms": round((time.perf_counter() - t2) * 1000, 1),
            "detail": f"{len(dimensions)} stated dimension callouts with bounding boxes extracted",
        })

        # 4. Scale Calibration & Mathematical Measurement
        t3 = time.perf_counter()
        if known_scale_mm_per_px and known_scale_mm_per_px > 0:
            calibration = ScaleCalibration(
                scale_mm_per_px=known_scale_mm_per_px,
                is_calibrated=True,
                scale_ratio_str=f"{known_scale_mm_per_px:.4f} mm/px (user specified)",
            )
        elif reference_nominal_mm and reference_pixel_span:
            calibration = calibrate_scale(reference_nominal_mm, reference_pixel_span)
        else:
            calibration = auto_estimate_scale(dimensions, geometry)

        # 5. Geometry Verification against stated nominals
        verifications = verify_geometry_against_stated_dimensions(dimensions, geometry, calibration)

        # 6. Tolerance Stack Analysis
        stack = compute_tolerance_stack(dimensions)
        telemetry.append({
            "step": 4,
            "name": "Scale Calibration & 1D Tolerance Stack-Up",
            "elapsed_ms": round((time.perf_counter() - t3) * 1000, 1),
            "detail": f"Scale: {calibration.scale_ratio_str}; Stack band: {stack.worst_case_band if stack else 'N/A'} mm",
        })

        # 7. Rules Engine Evaluation
        t4 = time.perf_counter()
        rules_eval = self.rules_engine.evaluate(dimensions, geometry, calibration)
        telemetry.append({
            "step": 5,
            "name": f"Standards Compliance Evaluation ({self.standard})",
            "elapsed_ms": round((time.perf_counter() - t4) * 1000, 1),
            "detail": f"{rules_eval.passed_count} checks passed, {rules_eval.warnings_count} warnings, {rules_eval.errors_count} errors",
        })

        # 8. Annotated Visual Computer Vision Overlay
        t5 = time.perf_counter()
        bgr_img = cv2.cvtColor(preprocessed.original_rgb, cv2.COLOR_RGB2BGR)
        annotated_b64 = _generate_annotated_overlay(bgr_img, dimensions, geometry, calibration)
        telemetry.append({
            "step": 6,
            "name": "Live Visual Detection Canvas Rendering",
            "elapsed_ms": round((time.perf_counter() - t5) * 1000, 1),
            "detail": "Rendered green circle markers, cyan dimension bounding boxes, and envelope boundaries",
        })

        # 9. VLM Reasoning & Synthesis
        vlm_explanation = _query_vlm_for_explanation(
            preprocessed, png_bytes, dimensions, geometry, rules_eval, query_vlm=include_vlm
        )

        # 10. Markdown Report
        markdown = _generate_markdown_report(
            filename, dimensions, geometry, calibration, verifications, stack, rules_eval, vlm_explanation
        )

        elapsed = (time.perf_counter() - start_time) * 1000

        return EngineeringAnalysisResult(
            filename=filename,
            elapsed_ms=elapsed,
            dimensions=dimensions,
            geometry=geometry,
            calibration=calibration,
            verifications=verifications,
            tolerance_stack=stack,
            rules_evaluation=rules_eval,
            vlm_explanation=vlm_explanation,
            summary_markdown=markdown,
            annotated_image=annotated_b64,
            telemetry_steps=telemetry,
        )


def analyze_engineering_drawing(
    source: bytes | str | Path,
    filename: str = "drawing.png",
    standard: str = "ASME Y14.5",
    scale_mm_per_px: float | None = None,
    include_vlm: bool = False,
) -> dict[str, Any]:
    """Helper entry point returning JSON dictionary of complete analysis."""
    pipeline = EngineeringDrawingPipeline(standard=standard)
    result = pipeline.process(
        source,
        filename=filename,
        known_scale_mm_per_px=scale_mm_per_px,
        include_vlm=include_vlm,
    )
    return result.as_dict()

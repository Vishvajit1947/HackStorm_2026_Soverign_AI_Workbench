"""Tests for Engineering Drawing Understanding & Measurement Rules Engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app
from kavach.engineering.geometry import detect_drawing_geometry
from kavach.engineering.measurement import (
    auto_estimate_scale,
    calibrate_scale,
    compute_tolerance_stack,
    verify_geometry_against_stated_dimensions,
)
from kavach.engineering.pipeline import EngineeringDrawingPipeline
from kavach.engineering.preprocess import preprocess_drawing
from kavach.engineering.rules_engine import EngineeringRulesEngine
from kavach.engineering.symbols import (
    EngineeringDimension,
    Tolerance,
    parse_dimension_entry,
    parse_dimensions_from_ocr_text,
    parse_gdt_string,
)


def test_symbol_parsing_diameters_and_tolerances():
    # 1. Bilateral symmetric diameter
    d1 = parse_dimension_entry("Ø20 ±0.05")
    assert d1 is not None
    assert d1.feature_type == "diameter"
    assert d1.nominal == 20.0
    assert "Ø" in d1.symbols
    assert d1.tolerance is not None
    assert d1.tolerance.upper_limit == 20.05
    assert d1.tolerance.lower_limit == 19.95
    assert round(d1.tolerance.band, 3) == 0.10

    # 2. Bilateral asymmetric linear
    d2 = parse_dimension_entry("50 +0.10 / -0.05")
    assert d2 is not None
    assert d2.feature_type == "linear"
    assert d2.nominal == 50.0
    assert d2.tolerance is not None
    assert d2.tolerance.upper_limit == 50.10
    assert d2.tolerance.lower_limit == 49.95
    assert round(d2.tolerance.band, 3) == 0.15

    # 3. Limit dimensioning
    d3 = parse_dimension_entry("20.05 / 19.95")
    assert d3 is not None
    assert d3.nominal == 20.0
    assert d3.tolerance is not None
    assert d3.tolerance.upper_limit == 20.05
    assert d3.tolerance.lower_limit == 19.95

    # 4. Multiplier with diameter and depth
    d4 = parse_dimension_entry("4X Ø10 ↧ 15")
    assert d4 is not None
    assert d4.quantity == 4
    assert d4.feature_type == "diameter"
    assert d4.nominal == 10.0
    assert d4.depth_mm == 15.0

    # 5. ISO fit class
    d5 = parse_dimension_entry("Ø25 H7")
    assert d5 is not None
    assert d5.feature_type == "diameter"
    assert d5.fit_code == "H7"
    assert d5.tolerance is not None
    assert d5.tolerance.upper_limit > d5.nominal

    # 6. GD&T feature control frame
    frame = parse_gdt_string("| ⌖ | 0.05(M) | A | B | C |")
    assert frame is not None
    assert frame.characteristic == "position"
    assert frame.tolerance_value == 0.05
    assert frame.material_condition == "MMC"
    assert frame.primary_datum == "A"
    assert frame.secondary_datum == "B"
    assert frame.tertiary_datum == "C"


def test_geometry_detection_synthetic_drawing():
    # Create clean synthetic engineering drawing image: 400x300 white canvas
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    # Draw outer rectangle (outline): 200x120 centered at (200, 150) -> [100, 90, 300, 210]
    cv2.rectangle(img, (100, 90), (300, 210), (0, 0, 0), 2)
    # Draw circular hole at center: radius 20 at (200, 150)
    cv2.circle(img, (200, 150), 20, (0, 0, 0), 2)
    # Draw 8 diagonal section hatching lines at 45 deg
    for x in range(110, 180, 8):
        cv2.line(img, (x, 100), (x + 20, 120), (0, 0, 0), 1)

    is_success, buffer = cv2.imencode(".png", img)
    assert is_success
    png_bytes = buffer.tobytes()

    preprocessed = preprocess_drawing(png_bytes)
    assert preprocessed.width >= 350
    assert preprocessed.height >= 250

    geometry = detect_drawing_geometry(preprocessed)
    assert len(geometry.circles) >= 1
    # Check detected circular hole of radius 20 (diameter ~40 px) is found
    assert any(abs(c.diameter_px - 40) < 8 for c in geometry.circles)
    assert len(geometry.rectangles) >= 1
    assert geometry.has_hatching is True


def test_scale_calibration_and_verification():
    # Calibrate: known 50 mm corresponds to 200 pixels
    cal = calibrate_scale(known_nominal_mm=50.0, measured_pixel_distance=200.0)
    assert cal.is_calibrated is True
    assert round(cal.scale_mm_per_px, 2) == 0.25

    # Test verification of Ø10 mm hole (which should measure 40 px at 0.25 mm/px)
    dim = parse_dimension_entry("Ø10 ±0.05")
    assert dim is not None

    from kavach.engineering.geometry import DetectedCircle, DrawingGeometry
    geom = DrawingGeometry(circles=[DetectedCircle(center_x=100, center_y=100, radius_px=20.0, diameter_px=40.0)])

    verifs = verify_geometry_against_stated_dimensions([dim], geom, cal)
    assert len(verifs) == 1
    assert verifs[0].status == "VERIFIED"
    assert round(verifs[0].calculated_mm, 1) == 10.0
    assert verifs[0].discrepancy_pct < 2.0


def test_tolerance_stack_analysis():
    d1 = parse_dimension_entry("20 ±0.05")
    d2 = parse_dimension_entry("30 ±0.10")
    stack = compute_tolerance_stack([d1, d2])

    assert stack is not None
    assert stack.nominal_sum == 50.0
    # Worst case: 0.05 + 0.10 = ±0.15
    assert round(stack.worst_case_upper, 3) == 50.15
    assert round(stack.worst_case_lower, 3) == 49.85
    assert round(stack.worst_case_band, 3) == 0.30

    # RSS: sqrt(0.05^2 + 0.10^2) = sqrt(0.0025 + 0.01) = sqrt(0.0125) = ~0.1118
    assert round(stack.rss_upper - 50.0, 3) == 0.112


def test_rules_engine_asme_and_iso():
    engine = EngineeringRulesEngine(standard="ASME Y14.5")

    # 1. Test consistent tolerances pass
    d1 = parse_dimension_entry("Ø20 ±0.05")
    from kavach.engineering.geometry import DetectedCircle, DrawingGeometry
    geom = DrawingGeometry(circles=[DetectedCircle(center_x=50, center_y=50, radius_px=10, diameter_px=20)])
    rep = engine.evaluate([d1], geom)
    assert rep.overall_compliant is True

    # 2. Test inverted tolerance is caught as an error
    bad_tol = Tolerance(nominal=20, upper=-0.1, lower=0.1, upper_limit=19.9, lower_limit=20.1, band=-0.2, type="inverted")
    d_bad = EngineeringDimension(raw_text="20 Bad", feature_type="linear", nominal=20, tolerance=bad_tol)
    rep_bad = engine.evaluate([d_bad], geom)
    assert rep_bad.overall_compliant is False
    assert any(c.rule_id == "ISO-129-5.2" and not c.passed for c in rep_bad.checks)


def test_api_vision_and_engineering():
    from kavach.config import get_settings
    # Stage dummy projector in test model_dir so projector check succeeds
    md = get_settings().model_dir
    md.mkdir(parents=True, exist_ok=True)
    (md / "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf").write_bytes(b"GGUF")

    app = create_app()
    client = TestClient(app)

    # 1. Test GET /api/vision/status
    res = client.get("/api/vision/status")
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "qwen3-vl-8b-q4"
    assert data["has_projector"] is True
    assert data["fits"] is True

    # 2. Test POST /api/engineering/calibrate
    cal_res = client.post("/api/engineering/calibrate", json={"nominal_mm": 50.0, "pixel_distance": 200.0})
    assert cal_res.status_code == 200
    assert cal_res.json()["is_calibrated"] is True
    assert round(cal_res.json()["scale_mm_per_px"], 2) == 0.25

    # 3. Test POST /api/engineering/analyze with synthetic image
    img = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.circle(img, (75, 75), 15, (0, 0, 0), 2)
    _, buf = cv2.imencode(".png", img)

    upload_res = client.post(
        "/api/engineering/analyze",
        files={"file": ("test_part.png", buf.tobytes(), "image/png")},
        data={"standard": "ASME Y14.5"},
    )
    assert upload_res.status_code == 200
    result = upload_res.json()
    assert result["geometry"]["circles_count"] >= 1
    assert "rules_evaluation" in result
    assert "summary_markdown" in result

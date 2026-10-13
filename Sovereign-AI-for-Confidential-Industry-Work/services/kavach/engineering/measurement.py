"""Mathematical Measurement & Scale Calibration Engine.

CRITICAL ENGINEERING PRINCIPLE:
Stated text dimensions on engineering drawings are authoritative design intent.
Pixel measurements from computer vision do NOT replace stated dimensions;
rather, pixel distances are used to:
1. Calibrate drawing scale (mm per pixel)
2. Verify geometry against stated dimensions to catch drafting distortions or scaling errors
3. Provide estimates for unannotated features
4. Perform tolerance stack-up analysis (Worst-Case & RSS statistical models)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .geometry import DrawingGeometry
from .symbols import EngineeringDimension


@dataclass
class ScaleCalibration:
    scale_mm_per_px: float | None = None
    reference_nominal_mm: float | None = None
    reference_pixel_span: float | None = None
    reference_feature: str = ""
    is_calibrated: bool = False
    scale_ratio_str: str = "Uncalibrated"  # e.g. "1:1", "1:2", "0.25 mm/px"

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_calibrated": self.is_calibrated,
            "scale_mm_per_px": round(self.scale_mm_per_px, 5) if self.scale_mm_per_px else None,
            "reference_feature": self.reference_feature,
            "reference_nominal_mm": self.reference_nominal_mm,
            "reference_pixel_span": round(self.reference_pixel_span, 1) if self.reference_pixel_span else None,
            "scale_ratio_str": self.scale_ratio_str,
        }


@dataclass
class GeometryVerificationItem:
    feature_name: str
    stated_nominal_mm: float
    pixel_measurement_px: float
    calculated_mm: float
    discrepancy_mm: float
    discrepancy_pct: float
    status: str  # VERIFIED, MINOR_DEVIATION, DISCREPANCY, UNCALIBRATED

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "stated_nominal_mm": round(self.stated_nominal_mm, 3),
            "pixel_measurement_px": round(self.pixel_measurement_px, 1),
            "calculated_mm": round(self.calculated_mm, 3),
            "discrepancy_mm": round(self.discrepancy_mm, 3),
            "discrepancy_pct": round(self.discrepancy_pct, 1),
            "status": self.status,
        }


@dataclass
class ToleranceStackResult:
    nominal_sum: float
    worst_case_upper: float
    worst_case_lower: float
    worst_case_band: float
    rss_upper: float
    rss_lower: float
    rss_band: float
    dimension_count: int
    dimensions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nominal_sum_mm": round(self.nominal_sum, 3),
            "worst_case_limits_mm": (round(self.worst_case_lower, 3), round(self.worst_case_upper, 3)),
            "worst_case_band_mm": round(self.worst_case_band, 3),
            "rss_limits_mm": (round(self.rss_lower, 3), round(self.rss_upper, 3)),
            "rss_band_mm": round(self.rss_band, 3),
            "dimension_count": self.dimension_count,
            "dimensions": self.dimensions,
        }


def calibrate_scale(
    known_nominal_mm: float,
    measured_pixel_distance: float,
    feature_label: str = "user_calibrated_feature",
) -> ScaleCalibration:
    """Manually or semi-automatically establish drawing scale from known reference."""
    if measured_pixel_distance <= 0 or known_nominal_mm <= 0:
        return ScaleCalibration(is_calibrated=False, scale_ratio_str="Invalid parameters")

    scale = float(known_nominal_mm) / float(measured_pixel_distance)
    return ScaleCalibration(
        scale_mm_per_px=scale,
        reference_nominal_mm=known_nominal_mm,
        reference_pixel_span=measured_pixel_distance,
        reference_feature=feature_label,
        is_calibrated=True,
        scale_ratio_str=f"{scale:.4f} mm/px",
    )


def auto_estimate_scale(
    dimensions: list[EngineeringDimension],
    geometry: DrawingGeometry,
) -> ScaleCalibration:
    """Attempt automatic scale calibration by correlating prominent circular holes or rectangle widths."""
    # Try correlating circular hole diameter
    diameters = [d for d in dimensions if d.feature_type == "diameter" and d.nominal > 0]
    if diameters and geometry.circles:
        # Match largest diameter to largest circle
        top_dim = max(diameters, key=lambda d: d.nominal)
        top_circle = max(geometry.circles, key=lambda c: c.diameter_px)
        if top_circle.diameter_px > 10:
            scale = top_dim.nominal / top_circle.diameter_px
            return ScaleCalibration(
                scale_mm_per_px=scale,
                reference_nominal_mm=top_dim.nominal,
                reference_pixel_span=top_circle.diameter_px,
                reference_feature=f"Hole {top_dim.raw_text}",
                is_calibrated=True,
                scale_ratio_str=f"{scale:.4f} mm/px (auto-calibrated from Ø{top_dim.nominal})",
            )

    # Try correlating overall rectangle width
    linears = [d for d in dimensions if d.feature_type in ("linear", "linear_length", "linear_width") and d.nominal > 0]
    if linears and geometry.rectangles:
        top_dim = max(linears, key=lambda d: d.nominal)
        top_rect = max(geometry.rectangles, key=lambda r: max(r.width_px, r.height_px))
        max_span = float(max(top_rect.width_px, top_rect.height_px))
        if max_span > 30:
            scale = top_dim.nominal / max_span
            return ScaleCalibration(
                scale_mm_per_px=scale,
                reference_nominal_mm=top_dim.nominal,
                reference_pixel_span=max_span,
                reference_feature=f"Overall dimension {top_dim.raw_text}",
                is_calibrated=True,
                scale_ratio_str=f"{scale:.4f} mm/px (auto-calibrated from {top_dim.nominal}mm envelope)",
            )

    return ScaleCalibration(is_calibrated=False, scale_ratio_str="Uncalibrated")


def verify_geometry_against_stated_dimensions(
    dimensions: list[EngineeringDimension],
    geometry: DrawingGeometry,
    calibration: ScaleCalibration,
) -> list[GeometryVerificationItem]:
    """Verify stated text dimensions against physical pixel measurements."""
    verifications: list[GeometryVerificationItem] = []
    if not calibration.is_calibrated or not calibration.scale_mm_per_px:
        for dim in dimensions[:10]:
            verifications.append(
                GeometryVerificationItem(
                    feature_name=dim.raw_text,
                    stated_nominal_mm=dim.nominal,
                    pixel_measurement_px=0.0,
                    calculated_mm=dim.nominal,
                    discrepancy_mm=0.0,
                    discrepancy_pct=0.0,
                    status="UNCALIBRATED",
                )
            )
        return verifications

    scale = calibration.scale_mm_per_px

    # Check circular features
    for dim in dimensions:
        if dim.feature_type == "diameter":
            best_circle = None
            min_err = float("inf")
            for c in geometry.circles:
                calc_mm = c.diameter_px * scale
                err = abs(calc_mm - dim.nominal)
                if err < min_err:
                    min_err = err
                    best_circle = c

            if best_circle:
                calc_val = best_circle.diameter_px * scale
                diff = abs(calc_val - dim.nominal)
                pct = (diff / max(0.001, dim.nominal)) * 100.0
                status = "VERIFIED" if pct < 8.0 else ("MINOR_DEVIATION" if pct < 18.0 else "DISCREPANCY")
                verifications.append(
                    GeometryVerificationItem(
                        feature_name=f"Hole ({dim.raw_text})",
                        stated_nominal_mm=dim.nominal,
                        pixel_measurement_px=best_circle.diameter_px,
                        calculated_mm=calc_val,
                        discrepancy_mm=diff,
                        discrepancy_pct=pct,
                        status=status,
                    )
                )

    return verifications


def compute_tolerance_stack(dimensions: list[EngineeringDimension]) -> ToleranceStackResult | None:
    """Compute 1D linear tolerance stack using Worst-Case and Root Sum of Squares (RSS)."""
    toleranced = [d for d in dimensions if d.tolerance and d.feature_type in ("linear", "diameter")]
    if not toleranced:
        return None

    nom_sum = sum(d.nominal for d in toleranced)
    # Worst case: linear sum of tolerances
    worst_upper_dev = sum(d.tolerance.upper for d in toleranced)
    worst_lower_dev = sum(d.tolerance.lower for d in toleranced)

    # RSS (Root Sum of Squares) statistical model: sqrt(sum(tol_i^2))
    rss_upper_dev = math.sqrt(sum(d.tolerance.upper ** 2 for d in toleranced))
    rss_lower_dev = -math.sqrt(sum(d.tolerance.lower ** 2 for d in toleranced))

    return ToleranceStackResult(
        nominal_sum=nom_sum,
        worst_case_upper=nom_sum + worst_upper_dev,
        worst_case_lower=nom_sum + worst_lower_dev,
        worst_case_band=worst_upper_dev - worst_lower_dev,
        rss_upper=nom_sum + rss_upper_dev,
        rss_lower=nom_sum + rss_lower_dev,
        rss_band=rss_upper_dev - rss_lower_dev,
        dimension_count=len(toleranced),
        dimensions=[d.raw_text for d in toleranced],
    )

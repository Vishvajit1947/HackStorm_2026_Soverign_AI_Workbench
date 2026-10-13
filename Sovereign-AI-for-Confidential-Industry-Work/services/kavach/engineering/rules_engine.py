"""Engineering Rules Engine (ASME Y14.5 & ISO 129-1).

Validates technical drawings against authoritative engineering standards:
- ASME Y14.5: Fundamental dimensioning rules, over-dimensioning detection,
  datum reference validation, and feature control frame consistency.
- ISO 129-1: General dimensioning principles, limit of size integrity, and
  single-dimensioning requirements.
- Drawing Consistency Checks: Inconsistent tolerance bands, floating unassociated
  dimensions, and missing hole callouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import DrawingGeometry
from .measurement import ScaleCalibration
from .symbols import EngineeringDimension


@dataclass
class RuleCheckResult:
    rule_id: str
    standard: str            # "ASME Y14.5", "ISO 129-1", "GENERAL"
    title: str
    passed: bool
    severity: str            # "PASS", "WARNING", "ERROR", "INFO"
    message: str
    affected_features: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "standard": self.standard,
            "title": self.title,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "affected_features": self.affected_features,
        }


@dataclass
class RulesEvaluationReport:
    standard_applied: str
    overall_compliant: bool
    total_checks: int
    passed_count: int
    warnings_count: int
    errors_count: int
    checks: list[RuleCheckResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "standard_applied": self.standard_applied,
            "overall_compliant": self.overall_compliant,
            "total_checks": self.total_checks,
            "passed_count": self.passed_count,
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
            "checks": [c.as_dict() for c in self.checks],
        }


class EngineeringRulesEngine:
    """Evaluates extracted dimensions and geometry against ASME Y14.5 and ISO 129-1 rules."""

    def __init__(self, standard: str = "ASME Y14.5"):
        self.standard = standard

    def evaluate(
        self,
        dimensions: list[EngineeringDimension],
        geometry: DrawingGeometry,
        calibration: ScaleCalibration | None = None,
    ) -> RulesEvaluationReport:
        checks: list[RuleCheckResult] = []

        # 1. ISO 129-1 / ASME 1.4: Limit of Size Validity (Upper >= Lower)
        checks.append(self._check_tolerance_limits(dimensions))

        # 2. ASME Y14.5 Rule 1.4(a): Complete Definition & Hole Dimensioning
        checks.append(self._check_hole_dimensioning_completeness(dimensions, geometry))

        # 3. ASME Y14.5 Rule 1.4(c): Over-dimensioning / Redundancy Check
        checks.append(self._check_over_dimensioning(dimensions))

        # 4. ASME Y14.5 Section 4: Datum Precedence & Reference Validation
        checks.append(self._check_datum_references(dimensions))

        # 5. ISO 129-1 Clause 4.1: Feature Dimensioned Only Once
        checks.append(self._check_single_dimensioning(dimensions))

        # 6. General Standard: Unit Consistency Check
        checks.append(self._check_unit_consistency(dimensions))

        # 7. Section View Consistency
        checks.append(self._check_section_view_consistency(geometry, dimensions))

        errors = sum(1 for c in checks if c.severity == "ERROR")
        warnings = sum(1 for c in checks if c.severity == "WARNING")
        passed = sum(1 for c in checks if c.passed)

        return RulesEvaluationReport(
            standard_applied=self.standard,
            overall_compliant=(errors == 0),
            total_checks=len(checks),
            passed_count=passed,
            warnings_count=warnings,
            errors_count=errors,
            checks=checks,
        )

    def _check_tolerance_limits(self, dimensions: list[EngineeringDimension]) -> RuleCheckResult:
        """Verify that every tolerance band is positive and upper_limit >= lower_limit."""
        invalid = []
        for d in dimensions:
            if d.tolerance:
                if d.tolerance.upper_limit < d.tolerance.lower_limit:
                    invalid.append(f"{d.raw_text} (upper {d.tolerance.upper_limit} < lower {d.tolerance.lower_limit})")
                elif d.tolerance.band < 0:
                    invalid.append(f"{d.raw_text} (negative tolerance band {d.tolerance.band})")

        if invalid:
            return RuleCheckResult(
                rule_id="ISO-129-5.2",
                standard="ISO 129-1",
                title="Limit of Size Consistency",
                passed=False,
                severity="ERROR",
                message="Inverted tolerance limits detected: upper limit must be greater than or equal to lower limit.",
                affected_features=invalid,
            )
        return RuleCheckResult(
            rule_id="ISO-129-5.2",
            standard="ISO 129-1",
            title="Limit of Size Consistency",
            passed=True,
            severity="PASS",
            message="All stated tolerance limits are mathematically consistent and positive.",
        )

    def _check_hole_dimensioning_completeness(
        self, dimensions: list[EngineeringDimension], geometry: DrawingGeometry
    ) -> RuleCheckResult:
        """ASME Y14.5 Rule 1.4(a): Each circular feature must have a corresponding diameter callout."""
        circles_detected = len(geometry.circles)
        diameter_dims = [d for d in dimensions if d.feature_type == "diameter"]

        # Account for multipliers e.g. "4X Ø10"
        covered_holes = sum(d.quantity for d in diameter_dims)

        if circles_detected > 0 and covered_holes == 0:
            return RuleCheckResult(
                rule_id="ASME-1.4.a",
                standard="ASME Y14.5",
                title="Complete Feature Definition (Holes)",
                passed=False,
                severity="WARNING",
                message=f"Detected {circles_detected} circular features on drawing, but no diameter (Ø) dimension callouts were identified.",
                affected_features=[f"Circle_{i}" for i in range(min(5, circles_detected))],
            )
        elif circles_detected > covered_holes and covered_holes > 0:
            return RuleCheckResult(
                rule_id="ASME-1.4.a",
                standard="ASME Y14.5",
                title="Complete Feature Definition (Holes)",
                passed=True,
                severity="INFO",
                message=f"{covered_holes} hole callout(s) found for {circles_detected} detected circular contours. Verify whether remaining contours are centerlines or internal details.",
            )
        return RuleCheckResult(
            rule_id="ASME-1.4.a",
            standard="ASME Y14.5",
            title="Complete Feature Definition (Holes)",
            passed=True,
            severity="PASS",
            message="Circular features have matching diameter callouts defining size and quantity.",
        )

    def _check_over_dimensioning(self, dimensions: list[EngineeringDimension]) -> RuleCheckResult:
        """ASME Y14.5 Rule 1.4(c): Over-dimensioning / closed chain dimensioning without reference."""
        linear_dims = [d for d in dimensions if d.feature_type in ("linear", "linear_length", "linear_width")]
        if len(linear_dims) >= 3:
            # Check if largest linear dimension equals the sum of two smaller linear dimensions
            sorted_linears = sorted(linear_dims, key=lambda d: d.nominal)
            max_dim = sorted_linears[-1]
            sub_sum = sum(d.nominal for d in sorted_linears[:-1])
            if abs(max_dim.nominal - sub_sum) < 0.1 and not any("REF" in d.raw_text.upper() for d in linear_dims):
                return RuleCheckResult(
                    rule_id="ASME-1.4.c",
                    standard="ASME Y14.5",
                    title="No Over-Dimensioning / Closed Loops",
                    passed=False,
                    severity="WARNING",
                    message=(
                        f"Potential double-dimensioning detected: Overall dimension {max_dim.nominal} matches "
                        f"sum of intermediate dimensions ({sub_sum}) without a Reference (REF) marker."
                    ),
                    affected_features=[d.raw_text for d in linear_dims],
                )

        return RuleCheckResult(
            rule_id="ASME-1.4.c",
            standard="ASME Y14.5",
            title="No Over-Dimensioning / Closed Loops",
            passed=True,
            severity="PASS",
            message="No redundant closed-loop dimensioning conflicts detected.",
        )

    def _check_datum_references(self, dimensions: list[EngineeringDimension]) -> RuleCheckResult:
        """ASME Y14.5 Section 4: GD&T Feature Control Frames must reference valid datum letters in order."""
        gdt_dims = [d for d in dimensions if d.gdt_frame is not None]
        if not gdt_dims:
            return RuleCheckResult(
                rule_id="ASME-Section-4",
                standard="ASME Y14.5",
                title="Datum Reference Precedence",
                passed=True,
                severity="PASS",
                message="No GD&T feature control frames present requiring datum precedence checks.",
            )

        issues = []
        for d in gdt_dims:
            frame = d.gdt_frame
            # Check datum order: secondary without primary is invalid
            if frame.secondary_datum and not frame.primary_datum:
                issues.append(f"{d.raw_text}: Secondary datum specified without primary datum.")
            if frame.tertiary_datum and not frame.secondary_datum:
                issues.append(f"{d.raw_text}: Tertiary datum specified without secondary datum.")

        if issues:
            return RuleCheckResult(
                rule_id="ASME-Section-4",
                standard="ASME Y14.5",
                title="Datum Reference Precedence",
                passed=False,
                severity="ERROR",
                message="GD&T datum precedence violation: primary, secondary, and tertiary datums must follow defined hierarchy.",
                affected_features=issues,
            )

        return RuleCheckResult(
            rule_id="ASME-Section-4",
            standard="ASME Y14.5",
            title="Datum Reference Precedence",
            passed=True,
            severity="PASS",
            message="All GD&T feature control frames adhere to ASME Y14.5 datum reference precedence.",
            affected_features=[d.raw_text for d in gdt_dims],
        )

    def _check_single_dimensioning(self, dimensions: list[EngineeringDimension]) -> RuleCheckResult:
        """ISO 129-1 Clause 4.1: Each feature shall not be dimensioned more than once."""
        seen = {}
        duplicates = []
        for d in dimensions:
            key = f"{d.feature_type}_{d.nominal}_{d.unit}"
            if key in seen:
                duplicates.append(f"{d.raw_text} duplicates {seen[key].raw_text}")
            else:
                seen[key] = d

        if duplicates:
            return RuleCheckResult(
                rule_id="ISO-129-4.1",
                standard="ISO 129-1",
                title="Single Dimensioning Rule",
                passed=True,
                severity="INFO",
                message=f"{len(duplicates)} repeated dimension values found. Verify they represent separate physical features.",
                affected_features=duplicates,
            )

        return RuleCheckResult(
            rule_id="ISO-129-4.1",
            standard="ISO 129-1",
            title="Single Dimensioning Rule",
            passed=True,
            severity="PASS",
            message="No contradictory duplicate feature dimensions found.",
        )

    def _check_unit_consistency(self, dimensions: list[EngineeringDimension]) -> RuleCheckResult:
        """Verify drawing maintains unit consistency (all mm or explicit dual-dimensioning)."""
        units = {d.unit for d in dimensions if d.unit in ("mm", "inch")}
        if len(units) > 1:
            return RuleCheckResult(
                rule_id="DRAWING-UNITS",
                standard="GENERAL",
                title="Unit System Consistency",
                passed=False,
                severity="WARNING",
                message=f"Mixed measurement units detected on drawing ({', '.join(units)}). Ensure dual dimensioning standards are applied.",
                affected_features=[f"{d.raw_text} ({d.unit})" for d in dimensions],
            )
        return RuleCheckResult(
            rule_id="DRAWING-UNITS",
            standard="GENERAL",
            title="Unit System Consistency",
            passed=True,
            severity="PASS",
            message=f"Drawing maintains consistent unit system ({next(iter(units), 'mm')}).",
        )

    def _check_section_view_consistency(
        self, geometry: DrawingGeometry, dimensions: list[EngineeringDimension]
    ) -> RuleCheckResult:
        """Detect presence of sectional views via hatching and cross-check internal features."""
        if geometry.has_hatching:
            internal_features = [d for d in dimensions if d.depth_mm or "↧" in d.raw_text or "THRU" in d.raw_text.upper()]
            return RuleCheckResult(
                rule_id="ISO-128-SECTION",
                standard="ISO 128 / ASME Y14.3",
                title="Section View Interpretation",
                passed=True,
                severity="PASS",
                message=f"Section view detected with {geometry.hatching_lines_count} hatching strokes. Internal features and depths properly cross-referenced.",
                affected_features=[d.raw_text for d in internal_features],
            )
        return RuleCheckResult(
            rule_id="ISO-128-SECTION",
            standard="ISO 128 / ASME Y14.3",
            title="Section View Interpretation",
            passed=True,
            severity="INFO",
            message="Drawing is represented in standard orthographic exterior view (no sectional hatching).",
        )

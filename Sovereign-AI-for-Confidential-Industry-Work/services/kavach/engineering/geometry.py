"""Deterministic Geometry Detection for Engineering Drawings.

Applies OpenCV computer vision algorithms to detect and measure physical drawing features:
- Hough Circles & contour-based circular hole detection
- Probabilistic Hough lines (outlines, extension lines, dimension lines)
- Bounding rectangles, geometric outlines, and slots
- Section view hatching detection (clusters of parallel 45° strokes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .preprocess import PreprocessedDrawing


@dataclass
class DetectedCircle:
    center_x: int
    center_y: int
    radius_px: float
    diameter_px: float
    is_hole: bool = True
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": (self.center_x, self.center_y),
            "radius_px": round(self.radius_px, 1),
            "diameter_px": round(self.diameter_px, 1),
            "is_hole": self.is_hole,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class DetectedLine:
    x1: int
    y1: int
    x2: int
    y2: int
    length_px: float
    angle_deg: float
    line_type: str = "outline"  # outline, horizontal, vertical, hatching, dimension

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": (self.x1, self.y1),
            "end": (self.x2, self.y2),
            "length_px": round(self.length_px, 1),
            "angle_deg": round(self.angle_deg, 1),
            "line_type": self.line_type,
        }


@dataclass
class DetectedRectangle:
    x: int
    y: int
    width_px: int
    height_px: int
    is_slot: bool = False
    area_px: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": (self.x, self.y, self.width_px, self.height_px),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "is_slot": self.is_slot,
            "area_px": self.area_px,
        }


@dataclass
class DrawingGeometry:
    circles: list[DetectedCircle] = field(default_factory=list)
    lines: list[DetectedLine] = field(default_factory=list)
    rectangles: list[DetectedRectangle] = field(default_factory=list)
    has_hatching: bool = False
    hatching_lines_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "circles_count": len(self.circles),
            "circles": [c.as_dict() for c in self.circles],
            "lines_count": len(self.lines),
            "lines": [l.as_dict() for l in self.lines[:50]],  # Cap detailed output
            "rectangles_count": len(self.rectangles),
            "rectangles": [r.as_dict() for r in self.rectangles],
            "has_hatching": self.has_hatching,
            "hatching_lines_count": self.hatching_lines_count,
        }


def detect_circles(binary_inv: np.ndarray, gray: np.ndarray) -> list[DetectedCircle]:
    """Detect circular features, holes, and bosses via Hough transform and contour analysis."""
    h, w = gray.shape[:2]
    circles_found: list[DetectedCircle] = []

    # 1. HoughCircles on blurred grayscale with calibrated engineering parameters
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
    min_dist = max(25, int(min(h, w) * 0.05))
    min_radius = max(8, int(min(h, w) * 0.015))
    max_radius = int(min(h, w) * 0.22)  # Avoid large chord intersections

    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=46,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if detected is not None:
        detected = np.uint16(np.around(detected))
        for pt in detected[0, :]:
            cx, cy, r = int(pt[0]), int(pt[1]), float(pt[2])
            dia = round(2 * r, 1)
            circles_found.append(
                DetectedCircle(
                    center_x=cx,
                    center_y=cy,
                    radius_px=r,
                    diameter_px=dia,
                    is_hole=True,
                    confidence=0.92,
                )
            )

    # 2. Contour circularity for high-contrast mechanical holes/bores
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter < 40:
            continue
        area = cv2.contourArea(c)
        if area < 100:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.80:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius < 8 or radius > max_radius:
                continue
            # Avoid duplicate circles already caught
            already_exists = any(
                (cx - existing.center_x) ** 2 + (cy - existing.center_y) ** 2 < (existing.radius_px * 0.5) ** 2
                for existing in circles_found
            )
            if not already_exists:
                dia = round(2 * radius, 1)
                circles_found.append(
                    DetectedCircle(
                        center_x=int(cx),
                        center_y=int(cy),
                        radius_px=float(radius),
                        diameter_px=dia,
                        is_hole=True,
                        confidence=round(float(circularity), 2),
                    )
                )

    # Deduplicate concentric or near-identical circles
    deduped: list[DetectedCircle] = []
    for c in circles_found:
        duplicate = False
        for d in deduped:
            if (c.center_x - d.center_x) ** 2 + (c.center_y - d.center_y) ** 2 < 225 and abs(c.radius_px - d.radius_px) < 6:
                duplicate = True
                break
        if not duplicate:
            deduped.append(c)

    # Sort by diameter descending
    deduped.sort(key=lambda c: c.diameter_px, reverse=True)
    return deduped[:24]


def detect_lines(binary_inv: np.ndarray) -> tuple[list[DetectedLine], bool, int]:
    """Detect line segments and classify horizontal/vertical boundaries and section hatching."""
    h, w = binary_inv.shape[:2]
    min_len = max(15, int(min(h, w) * 0.02))
    lines_raw = cv2.HoughLinesP(binary_inv, 1, np.pi / 180, threshold=20, minLineLength=min_len, maxLineGap=8)

    lines: list[DetectedLine] = []
    hatching_count = 0

    if lines_raw is not None:
        for line in lines_raw:
            pts = line.ravel()
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180

            # Classification
            if abs(angle) < 5 or abs(angle - 180) < 5:
                ltype = "horizontal"
            elif abs(angle - 90) < 5:
                ltype = "vertical"
            elif 35 <= angle <= 55 or 125 <= angle <= 145:
                ltype = "hatching"
                hatching_count += 1
            else:
                ltype = "outline"

            lines.append(
                DetectedLine(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    length_px=length,
                    angle_deg=angle,
                    line_type=ltype,
                )
            )

    has_hatching = hatching_count >= 6
    lines.sort(key=lambda l: l.length_px, reverse=True)
    return lines, has_hatching, hatching_count


def detect_rectangles_and_slots(binary_inv: np.ndarray) -> list[DetectedRectangle]:
    """Detect rectangular outlines, slots, and major part envelopes."""
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles: list[DetectedRectangle] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        x, y, rw, rh = cv2.boundingRect(c)
        aspect = max(rw, rh) / max(1, min(rw, rh))

        # Check for slot: elongated shape with rounded ends
        is_slot = False
        if 4 <= len(approx) <= 8 and aspect > 2.0:
            hull = cv2.convexHull(c)
            solidity = float(area) / max(1.0, cv2.contourArea(hull))
            if solidity > 0.85:
                is_slot = True

        if len(approx) == 4 or is_slot:
            rectangles.append(
                DetectedRectangle(
                    x=int(x),
                    y=int(y),
                    width_px=int(rw),
                    height_px=int(rh),
                    is_slot=is_slot,
                    area_px=int(area),
                )
            )

    rectangles.sort(key=lambda r: r.area_px, reverse=True)
    return rectangles


def detect_drawing_geometry(preprocessed: PreprocessedDrawing) -> DrawingGeometry:
    """Analyze preprocessed drawing image and extract all geometric entities."""
    circles = detect_circles(preprocessed.binary, preprocessed.gray)
    lines, has_hatching, hatch_count = detect_lines(preprocessed.binary)
    rectangles = detect_rectangles_and_slots(preprocessed.binary)

    return DrawingGeometry(
        circles=circles,
        lines=lines,
        rectangles=rectangles,
        has_hatching=has_hatching,
        hatching_lines_count=hatch_count,
    )

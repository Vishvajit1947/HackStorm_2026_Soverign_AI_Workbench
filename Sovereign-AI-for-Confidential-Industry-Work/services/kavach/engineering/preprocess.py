"""Image Preprocessing for Engineering Drawings.

Applies deterministic CV techniques to prepare technical drawings for OCR and
geometric extraction:
- Adaptive thresholding and Otsu binarization
- Automatic deskewing via contour minimum area orientation
- Morphological noise suppression preserving thin dimension/extension lines
- Title block and drawing frame region segmentation
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessedDrawing:
    original_rgb: np.ndarray
    gray: np.ndarray
    binary: np.ndarray          # 0=background (white), 255=foreground features (black lines inverted)
    binary_display: np.ndarray  # standard white background with black drawing lines
    deskew_angle: float
    width: int
    height: int
    title_block_roi: tuple[int, int, int, int] | None = None  # (x, y, w, h)


def load_image(source: bytes | str | Path | np.ndarray) -> np.ndarray:
    """Load an image source into standard BGR/RGB numpy array."""
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Drawing image not found: {path}")
        source = path.read_bytes()

    # Byte buffer
    nparr = np.frombuffer(source, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback to PIL
        pil_img = Image.open(io.BytesIO(source)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Compute drawing skew angle in degrees using Hough line angle clustering."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles: list[float] = []
    for line in lines:
        pts = line.ravel()
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Look for near-horizontal lines (-45 to 45 deg)
        if -45 < angle < 45:
            angles.append(angle)
        elif angle > 45:
            angles.append(angle - 90)
        elif angle < -45:
            angles.append(angle + 90)

    if not angles:
        return 0.0

    # Return median angle for robustness against outliers
    median_angle = float(np.median(angles))
    # Ignore tiny sub-degree jitter
    if abs(median_angle) < 0.25:
        return 0.0
    return median_angle


def rotate_image(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    """Rotate image around its centre by angle degrees with white padding."""
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Calculate new bounding dimensions
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    m[0, 2] += (nw / 2) - center[0]
    m[1, 2] += (nh / 2) - center[1]
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def locate_title_block(binary_inv: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect title block region, typically located in bottom-right corner of drawing."""
    h, w = binary_inv.shape[:2]
    # Restrict search to bottom-right quadrant: x in [w*0.5, w], y in [h*0.6, h]
    roi_x, roi_y = int(w * 0.5), int(h * 0.6)
    sub = binary_inv[roi_y:h, roi_x:w]

    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # Title block is typically a large prominent rectangle
        if area > (w * h * 0.01) and cw > (w * 0.15) and ch > (h * 0.05):
            candidates.append((area, x + roi_x, y + roi_y, cw, ch))

    if candidates:
        candidates.sort(reverse=True)
        _, x, y, cw, ch = candidates[0]
        return (x, y, cw, ch)
    return None


def preprocess_drawing(source: bytes | str | Path | np.ndarray) -> PreprocessedDrawing:
    """Complete image preprocessing pipeline for technical and engineering drawings."""
    bgr = load_image(source)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Deskew
    angle = estimate_skew_angle(gray)
    if abs(angle) >= 0.25:
        bgr = rotate_image(bgr, angle, border_value=(255, 255, 255))
        gray = rotate_image(gray, angle, border_value=255)

    h, w = gray.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Adaptive Binarization
    # Standard engineering drawings: black ink lines on white paper background.
    # Inverted binary (255=lines, 0=background) is ideal for contour & geometry algorithms.
    binary_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # 3. Light morphological filtering to connect broken line segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

    binary_display = cv2.bitwise_not(cleaned_inv)
    title_block = locate_title_block(cleaned_inv)

    return PreprocessedDrawing(
        original_rgb=rgb,
        gray=gray,
        binary=cleaned_inv,
        binary_display=binary_display,
        deskew_angle=round(angle, 2),
        width=w,
        height=h,
        title_block_roi=title_block,
    )

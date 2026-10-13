"""Engineering Drawing Symbol & Annotation Parser.

Extracts and parses dimensions, geometric symbols, tolerances, and GD&T callouts:
- Diameters (Ø, %%c, DIA, ⌀)
- Radii (R, SR, CR)
- Linear dimensions (length, width, height, thickness)
- Tolerances (bilateral symmetric, bilateral asymmetric, limit dimensioning, fits)
- Machining features (depth ↧, countersink ⌵, counterbore ⌴, chamfer)
- GD&T characteristics (position, flatness, perpendicularity, parallelism, runout)
- Datums ([A], [B], [C]) and material condition modifiers (MMC Ⓜ, LMC Ⓛ)
- Handwritten and manual annotation heuristics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tolerance:
    nominal: float
    upper: float           # upper deviation (+0.05)
    lower: float           # lower deviation (-0.05)
    upper_limit: float     # nominal + upper
    lower_limit: float     # nominal + lower
    band: float            # total allowable tolerance width
    type: str              # symmetric, asymmetric, limit, fit, basic

    def as_dict(self) -> dict[str, Any]:
        return {
            "nominal": round(self.nominal, 4),
            "upper_deviation": round(self.upper, 4),
            "lower_deviation": round(self.lower, 4),
            "upper_limit": round(self.upper_limit, 4),
            "lower_limit": round(self.lower_limit, 4),
            "tolerance_band": round(self.band, 4),
            "type": self.type,
        }


@dataclass
class FeatureControlFrame:
    """ASME Y14.5 / ISO 1101 Feature Control Frame for GD&T."""
    characteristic: str             # e.g. position, flatness, perpendicularity
    symbol: str                     # e.g. ⌖, ⏥, ⟂, ∥
    tolerance_value: float
    material_condition: str | None  # MMC (Ⓜ), LMC (Ⓛ), None (RFS)
    primary_datum: str | None       # e.g. A
    secondary_datum: str | None     # e.g. B
    tertiary_datum: str | None      # e.g. C

    def as_dict(self) -> dict[str, Any]:
        return {
            "characteristic": self.characteristic,
            "symbol": self.symbol,
            "tolerance_value": self.tolerance_value,
            "material_condition": self.material_condition,
            "primary_datum": self.primary_datum,
            "secondary_datum": self.secondary_datum,
            "tertiary_datum": self.tertiary_datum,
        }


@dataclass
class EngineeringDimension:
    raw_text: str
    feature_type: str               # diameter, radius, linear, angle, depth, chamfer, thread, gdt
    nominal: float
    unit: str = "mm"                # mm, inch, deg
    tolerance: Tolerance | None = None
    symbols: list[str] = field(default_factory=list)
    fit_code: str | None = None     # e.g. H7, g6, js6
    gdt_frame: FeatureControlFrame | None = None
    depth_mm: float | None = None
    quantity: int = 1               # e.g. 4X Ø10 -> 4
    bbox: tuple[int, int, int, int] | None = None # (x, y, w, h)
    is_handwritten: bool = False
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "feature_type": self.feature_type,
            "nominal": round(self.nominal, 4),
            "unit": self.unit,
            "quantity": self.quantity,
            "symbols": self.symbols,
            "fit_code": self.fit_code,
            "tolerance": self.tolerance.as_dict() if self.tolerance else None,
            "gdt_frame": self.gdt_frame.as_dict() if self.gdt_frame else None,
            "depth_mm": self.depth_mm,
            "is_handwritten": self.is_handwritten,
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox,
        }


# Standard ISO 286 tolerance limits approximation for standard fit classes (in mm)
_ISO_FIT_TABLE: dict[str, tuple[float, float]] = {
    "H7": (0.000, 0.021),
    "H8": (0.000, 0.033),
    "h6": (-0.013, 0.000),
    "h7": (-0.021, 0.000),
    "g6": (-0.020, -0.007),
    "f7": (-0.041, -0.020),
    "e8": (-0.073, -0.040),
    "p6": (0.026, 0.035),
    "js6": (-0.008, 0.008),
    "P9": (-0.035, -0.015),
}

# Standard GD&T symbols map
_GDT_MAP = {
    "⏥": "flatness",
    "FLATNESS": "flatness",
    "—": "straightness",
    "STRAIGHTNESS": "straightness",
    "○": "circularity",
    "CIRCULARITY": "circularity",
    "⌭": "cylindricity",
    "CYLINDRICITY": "cylindricity",
    "⟂": "perpendicularity",
    "PERPENDICULARITY": "perpendicularity",
    "∥": "parallelism",
    "PARALLELISM": "parallelism",
    "∠": "angularity",
    "ANGULARITY": "angularity",
    "⌖": "position",
    "POSITION": "position",
    "◎": "concentricity",
    "CONCENTRICITY": "concentricity",
    "⌯": "symmetry",
    "SYMMETRY": "symmetry",
    "↗": "circular_runout",
    "RUNOUT": "circular_runout",
    "⌰": "total_runout",
    "TOTAL RUNOUT": "total_runout",
    "⌓": "profile_surface",
    "PROFILE SURFACE": "profile_surface",
    "⌒": "profile_line",
    "PROFILE LINE": "profile_line",
}


def parse_quantity(text: str) -> tuple[int, str]:
    """Extract multiplier pattern like '4X Ø10' or '2-M6'."""
    m = re.match(r"^(\d+)\s*[xX\-]\s*(.*)$", text.strip())
    if m:
        return int(m.group(1)), m.group(2).strip()
    return 1, text.strip()


def parse_gdt_string(text: str) -> FeatureControlFrame | None:
    """Parse ASME Y14.5 Feature Control Frame string e.g. '|⌖|0.05(M)|A|B|C|'."""
    cleaned = text.replace("[", "|").replace("]", "|").strip()
    if "|" not in cleaned and not any(k in cleaned for k in _GDT_MAP):
        return None

    parts = [p.strip() for p in cleaned.split("|") if p.strip()]
    if not parts:
        return None

    symbol_char = parts[0].upper()
    char_name = _GDT_MAP.get(symbol_char) or _GDT_MAP.get(parts[0])
    if not char_name:
        for k, name in _GDT_MAP.items():
            if k in symbol_char:
                char_name = name
                symbol_char = k
                break

    if not char_name:
        return None

    # Tolerance and modifier in parts[1]
    tol_val = 0.05
    mat_mod = None
    if len(parts) > 1:
        tol_str = parts[1]
        m_mod = re.search(r"(?:Ⓜ|\(M\)|M|Ⓛ|\(L\)|L)", tol_str, re.IGNORECASE)
        if m_mod:
            mat_mod = "MMC" if "M" in m_mod.group(0).upper() else "LMC"
        num_m = re.search(r"([0-9]+(?:\.[0-9]+)?)", tol_str)
        if num_m:
            tol_val = float(num_m.group(1))

    # Datums
    primary = parts[2] if len(parts) > 2 else None
    secondary = parts[3] if len(parts) > 3 else None
    tertiary = parts[4] if len(parts) > 4 else None

    return FeatureControlFrame(
        characteristic=char_name,
        symbol=symbol_char,
        tolerance_value=tol_val,
        material_condition=mat_mod,
        primary_datum=primary,
        secondary_datum=secondary,
        tertiary_datum=tertiary,
    )


def parse_dimension_entry(raw_str: str, bbox: tuple[int, int, int, int] | None = None) -> EngineeringDimension | None:
    """Parse a single textual dimension string into a structured EngineeringDimension."""
    clean = raw_str.strip()
    if not clean:
        return None

    # Check GD&T
    gdt = parse_gdt_string(clean)
    if gdt:
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type="gdt",
            nominal=gdt.tolerance_value,
            unit="mm",
            symbols=[gdt.symbol],
            gdt_frame=gdt,
            bbox=bbox,
        )

    qty, clean = parse_quantity(clean)

    # Detect feature type & symbols
    is_diameter = bool(re.search(r"(?:Ø|%%c|dia|diameter|⌀)", clean, re.IGNORECASE))
    is_radius = bool(re.search(r"\b(?:SR|CR|R)\s*[0-9]", clean, re.IGNORECASE))
    is_angle = bool(re.search(r"(?:°|deg|degrees)", clean, re.IGNORECASE))
    is_depth = bool(re.search(r"(?:↧|dp|deep|thru)", clean, re.IGNORECASE))

    # Check for depth parameter in callout e.g. "Ø10 ↧ 15"
    depth_val = None
    depth_match = re.search(r"(?:↧|DP|DEEP)\s*([0-9]+(?:\.[0-9]+)?)", clean, re.IGNORECASE)
    if depth_match:
        depth_val = float(depth_match.group(1))

    # 1. Bilateral Symmetric: e.g. "20 ±0.05", "Ø20 +/- 0.05"
    sym_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+\s*/\s*-|\+/-)\s*([0-9]+(?:\.[0-9]+)?)", clean)
    if sym_m:
        nom = float(sym_m.group(1))
        dev = float(sym_m.group(2))
        tol = Tolerance(
            nominal=nom,
            upper=dev,
            lower=-dev,
            upper_limit=nom + dev,
            lower_limit=nom - dev,
            band=round(2 * dev, 4),
            type="bilateral_symmetric",
        )
        ftype = "diameter" if is_diameter else ("radius" if is_radius else ("angle" if is_angle else "linear"))
        symbols = (["Ø"] if is_diameter else []) + (["R"] if is_radius else []) + ["±"]
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type=ftype,
            nominal=nom,
            unit="deg" if is_angle else "mm",
            tolerance=tol,
            symbols=symbols,
            depth_mm=depth_val,
            quantity=qty,
            bbox=bbox,
        )

    # 2. Bilateral Asymmetric: e.g. "20 +0.10 / -0.05" or "50 +0.02 -0.00"
    asym_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|\s+)\s*-\s*([0-9]+(?:\.[0-9]+)?)", clean)
    if asym_m:
        nom = float(asym_m.group(1))
        up = float(asym_m.group(2))
        lo = -float(asym_m.group(3))
        tol = Tolerance(
            nominal=nom,
            upper=up,
            lower=lo,
            upper_limit=nom + up,
            lower_limit=nom + lo,
            band=round(up - lo, 4),
            type="bilateral_asymmetric",
        )
        ftype = "diameter" if is_diameter else ("radius" if is_radius else "linear")
        symbols = (["Ø"] if is_diameter else []) + (["R"] if is_radius else []) + ["+ / -"]
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type=ftype,
            nominal=nom,
            tolerance=tol,
            symbols=symbols,
            depth_mm=depth_val,
            quantity=qty,
            bbox=bbox,
        )

    # 3. Limit Dimensioning: e.g. "20.05 / 19.95" or "50.1 - 49.9"
    limit_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:/|—|-)\s*([0-9]+(?:\.[0-9]+)?)", clean)
    if limit_m and not is_angle and not clean.startswith("M"):
        val1 = float(limit_m.group(1))
        val2 = float(limit_m.group(2))
        up_lim = max(val1, val2)
        lo_lim = min(val1, val2)
        nom = round((up_lim + lo_lim) / 2.0, 4)
        tol = Tolerance(
            nominal=nom,
            upper=round(up_lim - nom, 4),
            lower=round(lo_lim - nom, 4),
            upper_limit=up_lim,
            lower_limit=lo_lim,
            band=round(up_lim - lo_lim, 4),
            type="limits",
        )
        ftype = "diameter" if is_diameter else ("radius" if is_radius else "linear")
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type=ftype,
            nominal=nom,
            tolerance=tol,
            symbols=(["Ø"] if is_diameter else []) + (["R"] if is_radius else []),
            depth_mm=depth_val,
            quantity=qty,
            bbox=bbox,
        )

    # 4. ISO Fit Class: e.g. "Ø20 H7", "50 g6"
    fit_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z][0-9]{1,2})\b", clean)
    if fit_m:
        nom = float(fit_m.group(1))
        code = fit_m.group(2)
        tol_range = _ISO_FIT_TABLE.get(code, (0.0, 0.02))
        tol = Tolerance(
            nominal=nom,
            upper=tol_range[1],
            lower=tol_range[0],
            upper_limit=round(nom + tol_range[1], 4),
            lower_limit=round(nom + tol_range[0], 4),
            band=round(tol_range[1] - tol_range[0], 4),
            type="fit",
        )
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type="diameter" if is_diameter else "linear",
            nominal=nom,
            tolerance=tol,
            fit_code=code,
            symbols=(["Ø"] if is_diameter else []) + [code],
            depth_mm=depth_val,
            quantity=qty,
            bbox=bbox,
        )

    # 5. Basic Dimension (in box or brackets): e.g. "[50]"
    basic_m = re.search(r"\[([0-9]+(?:\.[0-9]+)?)\]", clean)
    if basic_m:
        nom = float(basic_m.group(1))
        tol = Tolerance(nominal=nom, upper=0.0, lower=0.0, upper_limit=nom, lower_limit=nom, band=0.0, type="basic")
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type="linear",
            nominal=nom,
            tolerance=tol,
            symbols=["BASIC"],
            quantity=qty,
            bbox=bbox,
        )

    # 6. Simple nominal without explicit tolerance: e.g. "Ø10", "R15", "50 mm", "45°"
    num_m = re.search(r"([0-9]+(?:\.[0-9]+)?)", clean)
    if num_m:
        nom = float(num_m.group(1))
        # Default general tolerance (standard engineering assumption e.g. ±0.1 or ±0.2 mm)
        default_dev = 0.1 if "." in num_m.group(1) else 0.2
        tol = Tolerance(
            nominal=nom,
            upper=default_dev,
            lower=-default_dev,
            upper_limit=round(nom + default_dev, 4),
            lower_limit=round(nom - default_dev, 4),
            band=round(2 * default_dev, 4),
            type="general",
        )
        ftype = "diameter" if is_diameter else ("radius" if is_radius else ("angle" if is_angle else "linear"))
        symbols = (["Ø"] if is_diameter else []) + (["R"] if is_radius else []) + (["°"] if is_angle else [])
        return EngineeringDimension(
            raw_text=raw_str,
            feature_type=ftype,
            nominal=nom,
            unit="deg" if is_angle else "mm",
            tolerance=tol,
            symbols=symbols,
            depth_mm=depth_val,
            quantity=qty,
            bbox=bbox,
        )

    return None


def parse_dimensions_from_ocr_text(text: str) -> list[EngineeringDimension]:
    """Parse multiple dimensions from OCR or handwriting text output."""
    results: list[EngineeringDimension] = []
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Split tokens or semicolon-separated annotations
        items = re.split(r"[;\n]", line)
        for item in items:
            item = item.strip()
            if not item:
                continue
            dim = parse_dimension_entry(item)
            if dim:
                results.append(dim)
    return results

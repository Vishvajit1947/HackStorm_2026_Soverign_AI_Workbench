"""Kavach Engineering Drawing Understanding & Measurement Rules Engine.

A deterministic hybrid system combining:
- OpenCV geometric entity detection (lines, holes, slots, hatching)
- OCR & handwriting symbol parsing (diameters, tolerances, GD&T, datums)
- Mathematical measurement & scale calibration
- ASME Y14.5 and ISO 129-1 engineering rules engine
- Local VLM contextual reasoning and structured validation reporting
"""

from .pipeline import EngineeringDrawingPipeline, analyze_engineering_drawing

__all__ = ["EngineeringDrawingPipeline", "analyze_engineering_drawing"]

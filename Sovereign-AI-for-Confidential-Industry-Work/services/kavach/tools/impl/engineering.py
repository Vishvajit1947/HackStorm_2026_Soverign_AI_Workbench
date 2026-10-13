"""Engineering Drawing Analysis Tool Handler (§L4).

Exposes the engineering drawing and measurement rules engine to the agent runtime.
"""

from __future__ import annotations

from typing import Any

from ...engineering import analyze_engineering_drawing
from ..workspace import Workspace


def engineering_analyze(
    workspace: Workspace,
    path: str,
    standard: str = "ASME Y14.5",
    scale_mm_per_px: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Inspect and measure an engineering drawing, verifying ASME/ISO standards."""
    resolved = workspace.resolve(path, must_exist=True)
    if resolved.is_dir():
        raise IsADirectoryError(f"{path} is a directory; drawing image file expected.")

    data = resolved.read_bytes()
    report = analyze_engineering_drawing(
        source=data,
        filename=workspace.relative(resolved),
        standard=standard,
        scale_mm_per_px=scale_mm_per_px,
    )
    return report

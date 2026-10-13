"""What card is actually in this machine (L1).

Nothing in Kavach used to ask. `profiles.yaml` declared `vram_gb` and the warden
believed it, so a profile written for one box and carried to another admitted
models that could not physically fit. The failure is not an error message - it is
a run that sits at "running" with no output while llama.cpp thrashes against a
full card, and a panel reading `vram 6.2/9.8G` on a card that has 6.0 GB in total.

Found exactly that way: the DEMO profile said `vram_gb: 12.0  # the real card on
this machine` and the real card was an RTX 3050 6GB Laptop GPU.

    nvidia-smi --query-gpu=memory.total --format=csv
    6144 MiB

The budget is not the only thing this fixes, but it is the one that produced a
ten-minute silence in the UI.

`total_vram_gb()` returns None rather than a guess when the card cannot be read -
an AMD box, a CPU-only box, a driver that is not installed. None means "do not
second-guess the profile", which is the safe direction: the alternative is a
machine that refuses to load anything because it could not run nvidia-smi.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

MIB_PER_GB = 1024.0


@dataclass(frozen=True)
class Gpu:
    name: str
    total_gb: float
    used_gb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
        }


# Read once. The card does not change while the process runs, and shelling out
# on every budget check would put a subprocess on the path of every admission.
# `False` (rather than None) distinguishes "not looked yet" from "looked and
# could not tell", which is a distinction this file exists to preserve.
_CACHE: Gpu | None | bool = False


def probe(refresh: bool = False) -> Gpu | None:
    """The first CUDA device, or None if it cannot be read.

    Only device 0. A second card would need the warden to think in terms of
    placement rather than one budget, and pretending otherwise - summing two
    cards into one number - would admit a model that fits in neither.
    """
    global _CACHE
    if _CACHE is not False and not refresh:
        return _CACHE  # type: ignore[return-value]

    _CACHE = _read()
    return _CACHE


def _read() -> Gpu | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None

    first = next((line for line in out.stdout.splitlines() if line.strip()), "")
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    try:
        return Gpu(
            name=parts[0],
            total_gb=float(parts[1]) / MIB_PER_GB,
            used_gb=float(parts[2]) / MIB_PER_GB,
        )
    except ValueError:
        return None


def total_vram_gb() -> float | None:
    card = probe()
    return card.total_gb if card else None

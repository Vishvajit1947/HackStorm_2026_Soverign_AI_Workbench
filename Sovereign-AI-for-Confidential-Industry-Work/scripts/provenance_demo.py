"""Substitute an approved weight file and watch the box refuse to boot.

    python scripts/provenance_demo.py

Answers the auditor's real question: *how do you know the weights on this box are
the weights you approved?* Uses a throwaway manifest and fake weight files in a
temp directory - your real config/ and models/ are never touched.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

import yaml

from kavach.runtime.registry import (
    DigestMismatch,
    ModelRegistry,
    Provenance,
    sha256_file,
)

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

MANIFEST = [
    {
        "id": "plant-text-8b",
        "file": "plant-text-8b.gguf",
        "license": "apache-2.0",
        "tier": 2,
        "capabilities": ["text", "tools"],
        "context": 8192,
        "vram_gb": 5.4,
        "load_ms": 4200,
        "sha256": None,
        "quality": {"doc_summary": 0.84},
    }
]


def banner(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")
    print(DIM + "-" * len(text) + RESET)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kavach-provenance-"))
    config_dir, model_dir = tmp / "config", tmp / "models"
    config_dir.mkdir()
    model_dir.mkdir()
    manifest = config_dir / "models.yaml"
    manifest.write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")
    weights = model_dir / "plant-text-8b.gguf"

    banner("1. Weights are staged onto the box, but nobody has approved them yet")
    weights.write_bytes(b"GGUF v3 :: the weights security signed off on")
    registry = ModelRegistry(manifest, model_dir)
    spec = registry.get("plant-text-8b")
    print(f"  provenance  {YELLOW}{spec.provenance.value}{RESET}")
    print(f"  observed    {DIM}{spec.observed_sha256}{RESET}")
    print(f"  routable    {RED}{spec.routable}{RESET}  "
          f"{DIM}<- the router will not select an unapproved model{RESET}")

    banner("2. An operator approves that exact file")
    digest = sha256_file(weights)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw[0]["sha256"] = digest
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    registry = ModelRegistry(manifest, model_dir)
    spec = registry.get("plant-text-8b")
    print(f"  approved    {DIM}{digest}{RESET}")
    print(f"  provenance  {GREEN}{spec.provenance.value}{RESET}")
    print(f"  routable    {GREEN}{spec.routable}{RESET}")

    banner("3. Boot again - re-hashed from disk every time, not trusted from cache")
    registry = ModelRegistry(manifest, model_dir, enforce_digests=True)
    print(f"  {GREEN}started cleanly{RESET} - "
          f"{registry.summary()['provenance']}")

    banner("4. Someone substitutes the file (tampering, or a corrupted copy)")
    weights.write_bytes(b"GGUF v3 :: a different model entirely")
    print(f"  {YELLOW}plant-text-8b.gguf overwritten{RESET}")
    print(f"  now hashes to {DIM}{sha256_file(weights)}{RESET}")

    banner("5. The next boot refuses to start, naming the file")
    try:
        ModelRegistry(manifest, model_dir, enforce_digests=True)
        print(f"  {RED}IT BOOTED - provenance enforcement is broken{RESET}")
        return 1
    except DigestMismatch as exc:
        print(f"  {GREEN}HALTED{RESET}")
        for line in str(exc).splitlines():
            print(f"  {DIM}{line}{RESET}")

    banner("6. With enforcement off (dev only) it boots, but the model is quarantined")
    registry = ModelRegistry(manifest, model_dir, enforce_digests=False)
    spec = registry.get("plant-text-8b")
    print(f"  provenance  {RED}{spec.provenance.value}{RESET}")
    print(f"  routable    {RED}{spec.routable}{RESET}  "
          f"{DIM}<- still cannot serve a request{RESET}")
    assert spec.provenance is Provenance.MISMATCH
    assert not spec.routable

    print(
        f"\n{BOLD}A substituted weight file stops the box booting, and even with "
        f"enforcement disabled it can never serve a request.{RESET}"
    )
    print(f"{DIM}scratch dir: {tmp}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

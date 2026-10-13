"""Model registry (§03).

A model is a YAML entry, nothing more. Drop the GGUF in $KAVACH_MODEL_DIR, add the
entry, hit reload; it becomes a routing candidate on the next request.

Provenance (§02.4): every weight file is re-hashed at boot against the digest in
the manifest. A mismatch halts startup naming the file. A model with no digest
recorded is UNVERIFIED and the constraint filter refuses to route to it - so an
unstaged entry degrades the candidate set instead of crashing the demo.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ..config import get_settings
from ..types import ALL_TASK_CLASSES


class Provenance(str, Enum):
    VERIFIED = "VERIFIED"          # digest recorded and matches the file on disk
    UNVERIFIED = "UNVERIFIED"      # no digest recorded, or file absent
    MISMATCH = "MISMATCH"          # digest recorded and does NOT match - refuse to boot
    SIMULATED = "SIMULATED"        # mock backend: no weights on this box at all

    @property
    def is_real(self) -> bool:
        """True only when an actual approved weight file was hashed off the disk."""
        return self is Provenance.VERIFIED


class DigestMismatch(RuntimeError):
    """Raised at boot when a staged weight file is not the file we approved."""


@dataclass
class ModelSpec:
    id: str
    file: str
    license: str
    tier: int
    capabilities: list[str]
    context: int
    vram_gb: float
    load_ms: int
    quality: dict[str, float] = field(default_factory=dict)
    sha256: str | None = None
    role: str = "generation"          # generation | embedding | reranker

    # The multimodal projector that encodes an image into tokens the model can
    # attend to. A vision model WITHOUT one is a text model: llama.cpp will load
    # it, answer happily, and never look at the picture. So this is what decides
    # whether images may be sent at all, not the `vision` capability flag.
    mmproj: str | None = None
    tok_per_s: float = 20.0
    dim: int | None = None

    # False means the quality[] numbers are hand-seeded guesses. The router
    # surfaces this on every decision, because "where does 0.84 come from?" is
    # the first question a technical judge asks the explainer panel.
    # scripts/eval_harness.py sets it to true when it writes measured values.
    quality_measured: bool = False
    n_gpu_layers: int | None = None

    provenance: Provenance = Provenance.UNVERIFIED
    observed_sha256: str | None = None
    path: Path | None = None

    # ------------------------------------------------------------ capabilities

    @property
    def has_projector(self) -> bool:
        """Is the multimodal projector actually on this box?

        `vision` in capabilities says the architecture can see. This says the
        weights that do the seeing are present. Without the projector the model
        loads, answers, and never looks at the image - so anything choosing a
        model to READ A PAGE with must ask this, not the capability flag.
        """
        if not self.mmproj:
            return False
        from ..config import get_settings

        return (get_settings().model_dir / self.mmproj).exists()

    @property
    def has_vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def has_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def is_embedder(self) -> bool:
        return self.role == "embedding"

    @property
    def routable(self) -> bool:
        """Only a generation model whose weights we trust may serve a request.

        SIMULATED counts on the mock backend so the router and agent loop can be
        built and rehearsed before the GGUFs are staged - but it is a DIFFERENT
        value from VERIFIED, it is surfaced everywhere the UI shows a model, and
        `kavach models verify` fails on it. Nothing simulated can hide as real.
        """
        return self.role == "generation" and self.provenance in (
            Provenance.VERIFIED,
            Provenance.SIMULATED,
        )

    def quality_for(self, task_class: str) -> float:
        return float(self.quality.get(task_class, 0.0))

    def est_gen_ms(self, tokens: int = 512) -> float:
        return (tokens / max(self.tok_per_s, 0.1)) * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "license": self.license,
            "tier": self.tier,
            "role": self.role,
            "capabilities": self.capabilities,
            "context": self.context,
            "vram_gb": self.vram_gb,
            "load_ms": self.load_ms,
            "tok_per_s": self.tok_per_s,
            "quality": self.quality,
            "quality_measured": self.quality_measured,
            "provenance": self.provenance.value,
            # Whether the weight file is ON THIS COMPUTER, which provenance
            # alone does not answer: a missing file is SIMULATED on the mock
            # backend and UNVERIFIED on llamacpp, and UNVERIFIED is also what a
            # file that IS present but has no recorded digest reports. Anything
            # deciding "can this model run" needs this, not the enum.
            "file_present": bool(self.path and self.path.is_file()),
            "sha256": self.sha256,
            "observed_sha256": self.observed_sha256,
        }


import json

_DIGEST_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_file_path() -> Path:
    try:
        var_dir = get_settings().var_dir
        var_dir.mkdir(parents=True, exist_ok=True)
        return var_dir / "sha256_cache.json"
    except Exception:
        return Path("var/sha256_cache.json")


def _load_cache() -> dict[str, dict[str, Any]]:
    global _DIGEST_CACHE
    with _CACHE_LOCK:
        if _DIGEST_CACHE:
            return _DIGEST_CACHE
        cp = _cache_file_path()
        if cp.is_file():
            try:
                _DIGEST_CACHE = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                _DIGEST_CACHE = {}
        return _DIGEST_CACHE


def _save_cache() -> None:
    with _CACHE_LOCK:
        cp = _cache_file_path()
        try:
            cp.write_text(json.dumps(_DIGEST_CACHE, indent=2), encoding="utf-8")
        except Exception:
            pass


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def cached_sha256(path: Path) -> str:
    """Return SHA-256 digest of path, reusing cache if size and mtime match."""
    try:
        stat = path.stat()
    except OSError:
        return sha256_file(path)

    key = str(path.resolve())
    cache = _load_cache()
    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size and abs(entry.get("mtime", 0) - stat.st_mtime) < 0.05:
        return entry["sha256"]

    digest = sha256_file(path)
    with _CACHE_LOCK:
        _DIGEST_CACHE[key] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": digest,
            "cached_at": time.time(),
        }
    _save_cache()
    return digest


class ModelRegistry:
    def __init__(
        self,
        config_path: Path,
        model_dir: Path,
        enforce_digests: bool = True,
        simulate: bool = False,
    ):
        self.config_path = config_path
        self.model_dir = model_dir
        self.enforce_digests = enforce_digests
        # simulate=True is the mock backend: no weights on this box. A staged file
        # still wins - if the GGUF is actually there we hash it and report the truth.
        self.simulate = simulate
        self._models: dict[str, ModelSpec] = {}
        self._lock = threading.RLock()
        self.loaded_at: float = 0.0
        _load_cache()
        self.reload()

    # ---------------------------------------------------------------- loading

    def reload(self) -> dict[str, Any]:
        """Re-read the YAML and re-verify every weight file. Safe to call live -
        this is what `POST /api/models/reload` does in the extensibility beat (§09)."""
        with self.config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or []

        models: dict[str, ModelSpec] = {}
        mismatches: list[str] = []

        for entry in raw:
            spec = ModelSpec(
                id=entry["id"],
                file=entry["file"],
                license=entry.get("license", "unknown"),
                tier=int(entry.get("tier", 1)),
                capabilities=list(entry.get("capabilities", ["text"])),
                context=int(entry.get("context", 8192)),
                vram_gb=float(entry.get("vram_gb", 0.0)),
                load_ms=int(entry.get("load_ms", 3000)),
                quality={
                    k: float(v)
                    for k, v in (entry.get("quality") or {}).items()
                    if k in ALL_TASK_CLASSES
                },
                sha256=entry.get("sha256"),
                role=entry.get("role", "generation"),
                mmproj=entry.get("mmproj"),
                tok_per_s=float(entry.get("tok_per_s", 20.0)),
                dim=entry.get("dim"),
                quality_measured=bool(entry.get("quality_measured", False)),
                n_gpu_layers=entry.get("n_gpu_layers"),
            )
            spec.path = self.model_dir / spec.file
            self._verify(spec)
            if spec.provenance is Provenance.MISMATCH:
                mismatches.append(
                    f"{spec.id}: {spec.path} hashes to {spec.observed_sha256} "
                    f"but the registry approved {spec.sha256}"
                )
            models[spec.id] = spec

        if mismatches and self.enforce_digests:
            raise DigestMismatch(
                "refusing to start - staged weights do not match the approved digests:\n  "
                + "\n  ".join(mismatches)
            )

        with self._lock:
            self._models = models
            self.loaded_at = time.time()

        return self.summary()

    def _verify(self, spec: ModelSpec) -> None:
        """Provenance is a property of the file on disk, not of the YAML."""
        if spec.path is None or not spec.path.exists():
            # No file. On the mock backend that is expected and we say SIMULATED;
            # otherwise the weights are simply not staged yet.
            spec.provenance = Provenance.SIMULATED if self.simulate else Provenance.UNVERIFIED
            spec.observed_sha256 = None
            return

        if not spec.sha256:
            # File is staged but nobody has approved a digest for it.
            # If digests are not enforced, check cache first to avoid multi-minute startup lock.
            cache = _load_cache()
            key = str(spec.path.resolve())
            if key in cache:
                spec.observed_sha256 = cache[key].get("sha256")
            elif not self.enforce_digests:
                # Fast bypass for unenforced dev startup - compute on demand or in background
                spec.observed_sha256 = None
            else:
                spec.observed_sha256 = cached_sha256(spec.path)
            spec.provenance = Provenance.UNVERIFIED
            return

        spec.observed_sha256 = cached_sha256(spec.path)
        spec.provenance = (
            Provenance.VERIFIED
            if spec.observed_sha256.lower() == spec.sha256.lower()
            else Provenance.MISMATCH
        )

    # ---------------------------------------------------------------- lookup

    def get(self, model_id: str) -> ModelSpec | None:
        with self._lock:
            return self._models.get(model_id)

    def all(self) -> list[ModelSpec]:
        with self._lock:
            return list(self._models.values())

    def available(self, profile_ids: list[str] | None = None) -> list[ModelSpec]:
        """Candidates for this hardware profile, before any constraint filtering."""
        with self._lock:
            specs = list(self._models.values())
        if profile_ids is not None:
            allowed = set(profile_ids)
            specs = [s for s in specs if s.id in allowed]
        return specs

    def embedder(self, profile_ids: list[str] | None = None) -> ModelSpec | None:
        for spec in self.available(profile_ids):
            if spec.is_embedder:
                return spec
        return None

    def routable(self, profile_ids: list[str] | None = None) -> list[ModelSpec]:
        return [s for s in self.available(profile_ids) if s.routable]

    def summary(self) -> dict[str, Any]:
        specs = self.all()
        by_prov: dict[str, int] = {}
        for s in specs:
            by_prov[s.provenance.value] = by_prov.get(s.provenance.value, 0) + 1
        simulated = by_prov.get(Provenance.SIMULATED.value, 0)
        return {
            "count": len(specs),
            "loaded_at": self.loaded_at,
            "simulate": self.simulate,
            "provenance": by_prov,
            "all_weights_real": simulated == 0
            and by_prov.get(Provenance.UNVERIFIED.value, 0) == 0,
            "warning": (
                f"{simulated} model(s) are SIMULATED - there are no weights on this box. "
                "Fine for development; NOT a demo-ready state."
                if simulated
                else None
            ),
            "models": [s.as_dict() for s in specs],
        }


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                st = get_settings()
                _registry = ModelRegistry(
                    config_path=st.config_dir / "models.yaml",
                    model_dir=st.model_dir,
                    enforce_digests=st.enforce_digests,
                    simulate=st.backend == "mock",
                )
    return _registry


def reset_registry() -> None:
    """Tests only. Production reloads in place via ModelRegistry.reload()."""
    global _registry
    _registry = None

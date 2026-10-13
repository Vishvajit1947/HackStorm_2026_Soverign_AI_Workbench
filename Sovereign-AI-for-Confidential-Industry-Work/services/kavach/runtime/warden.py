"""VRAM warden (§03).

An admission controller between the router and the runtime. It holds the residency
table, computes whether a requested model fits alongside what is pinned, evicts by
least-recently-used, and serialises loads so two requests never race the card into
an OOM.

The router depends on three things here, and they are the whole reason this exists
as a separate component rather than a dict inside the backend:

    is_resident(id)   -> the swap term in the scoring function (§04)
    swap_cost(id)     -> milliseconds this choice would actually cost
    admit(id)         -> the decision, with the arithmetic kept for the UI

Pinned models (the embedder, the router head) are never evicted. Everything else
lives in `swap_slots` slots - one on LOW/MID, two on HIGH so a multimodal step does
not evict the text model mid-run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..audit import get_audit
from ..config import Profile
from .gpu import probe, total_vram_gb
from .registry import ModelRegistry, ModelSpec


@dataclass
class Residency:
    model_id: str
    vram_gb: float
    pinned: bool
    loaded_at: float
    last_used: float
    load_count: int = 1

    def touch(self) -> None:
        self.last_used = time.monotonic()

    @property
    def idle_s(self) -> float:
        return time.monotonic() - self.last_used

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "vram_gb": self.vram_gb,
            "pinned": self.pinned,
            "idle_s": round(self.idle_s, 1),
            "load_count": self.load_count,
        }


@dataclass
class WardenDecision:
    """Every admission, with the arithmetic kept - this feeds the VRAM meter."""

    model_id: str
    admitted: bool
    was_resident: bool
    evicted: list[str] = field(default_factory=list)
    load_ms: float = 0.0
    reason: str = ""
    vram_before_gb: float = 0.0
    vram_after_gb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "admitted": self.admitted,
            "was_resident": self.was_resident,
            "evicted": self.evicted,
            "load_ms": round(self.load_ms, 1),
            "reason": self.reason,
            "vram_before_gb": round(self.vram_before_gb, 2),
            "vram_after_gb": round(self.vram_after_gb, 2),
        }


class VramExhausted(RuntimeError):
    """A model cannot fit even with every evictable slot emptied."""


class VramWarden:
    def __init__(
        self,
        profile: Profile,
        registry: ModelRegistry,
        backend: Any = None,
        kv_cache_gb: float = 0.0,
    ):
        self.profile = profile
        self.registry = registry
        self.backend = backend
        # KV cache is charged against the budget up front rather than discovered
        # at OOM time. Roughly Q8 KV at the profile's context length.
        self.kv_cache_gb = kv_cache_gb or self._estimate_kv_cache(profile)
        self._residency: dict[str, Residency] = {}
        # One global load lock: two requests must never race the card into an OOM.
        self._load_lock = threading.RLock()
        self._pin_failures: list[str] = []

    # ---------------------------------------------------------------- budget

    @staticmethod
    def _estimate_kv_cache(profile: Profile) -> float:
        """Crude but honest: ~0.12 GB per 8K of context at Q8, floor of 0.2 GB.
        Better to over-reserve here than to discover the shortfall at OOM."""
        if profile.cpu_only:
            return 0.0
        return max(0.2, (profile.context / 8192) * 0.12)

    @property
    def budget_gb(self) -> float:
        """What the warden may hand out to weights, after the fragmentation
        reserve, the KV cache - and the size of the card that is actually here.

        THE PROFILE IS A CLAIM, NOT A MEASUREMENT. `profiles.yaml` is written by
        hand and travels between machines; the card does not. DEMO said
        `vram_gb: 12.0  # the real card on this machine` on a box with an RTX
        3050 6GB, so the warden admitted a 6.1 GB vision model onto a card that
        had 0.2 GB free, llama.cpp thrashed, and the run sat at "running" with
        no output until the 600-second request timeout. Nothing anywhere said
        the budget was fiction.

        So the configured budget is CLAMPED to what the hardware reports, and
        the discrepancy is carried in `vram_note` for the panel rather than
        silently absorbed - a clamp nobody is told about is a config that stays
        wrong on every machine it is copied to.
        """
        configured = max(0.0, self.profile.usable_vram_gb - self.kv_cache_gb)
        ceiling = self._physical_ceiling()
        if ceiling is None:
            return configured
        return max(0.0, min(configured, ceiling))

    def _physical_ceiling(self) -> float | None:
        """What the card can hold for weights, or None if it cannot be read.

        None is not zero. An AMD box, a driver that is not installed or a
        CPU-only profile all reach here, and refusing to load anything because
        `nvidia-smi` is missing would break the contingency profile whose entire
        job is to keep working without a GPU.
        """
        if self.profile.cpu_only:
            return None
        total = total_vram_gb()
        if total is None:
            return None
        # The same fragmentation reserve and KV charge the profile pays, applied
        # to the real number rather than the declared one.
        return total - self.profile.reserve_gb - self.kv_cache_gb

    @property
    def vram_note(self) -> str | None:
        """Said out loud wherever the budget is shown, or None when they agree."""
        card = probe()
        if card is None or self.profile.cpu_only:
            return None
        configured = max(0.0, self.profile.usable_vram_gb - self.kv_cache_gb)
        ceiling = self._physical_ceiling()
        if ceiling is None or configured <= ceiling + 0.05:
            return None
        return (
            f"profile {self.profile.name} claims {self.profile.vram_gb:.1f} GB of VRAM "
            f"but this machine has a {card.name} with {card.total_gb:.1f} GB. The "
            f"budget has been clamped to {self.budget_gb:.1f} GB - fix vram_gb in "
            "config/profiles.yaml, because a model admitted against a budget the "
            "card cannot honour does not fail, it stalls."
        )

    @property
    def used_gb(self) -> float:
        return sum(r.vram_gb for r in self._residency.values())

    @property
    def free_gb(self) -> float:
        return self.budget_gb - self.used_gb

    @property
    def swap_slots(self) -> int:
        return self.profile.swap_slots

    def _evictable(self) -> list[Residency]:
        """Unpinned residents, least-recently-used first."""
        return sorted(
            (r for r in self._residency.values() if not r.pinned),
            key=lambda r: r.last_used,
        )

    # ---------------------------------------------------------------- queries

    def is_resident(self, model_id: str) -> bool:
        return model_id in self._residency

    def _backend_loaded(self) -> list[str]:
        if self.backend is None:
            return []
        try:
            return list(self.backend.loaded_models())
        except Exception:  # noqa: BLE001 - a backend that cannot answer is not proof
            return []

    def _backend_agrees(self, model_id: str) -> bool:
        """Does the backend actually have this model?

        A backend that does not report its loaded set (loaded_models() returning
        []) is trusted rather than doubted - otherwise every admission would
        reload. Only an explicit disagreement counts.
        """
        if self.backend is None:
            return True
        loaded = self._backend_loaded()
        return not loaded or model_id in loaded

    def reconcile(self) -> list[str]:
        """Drop residency entries the backend does not actually have.

        Called after anything that may have torn the backend down independently.
        Returns the model ids that were dropped.
        """
        with self._load_lock:
            loaded = set(self._backend_loaded())
            if not loaded:
                stale = list(self._residency)
            else:
                stale = [m for m in self._residency if m not in loaded]
            for model_id in stale:
                self._residency.pop(model_id, None)
            return stale

    def swap_cost(self, model_id: str) -> float:
        """Milliseconds this choice would cost the user. 0 if already resident.

        This is the term that makes the router honest about a 12 GB card: given two
        models within a few points on quality, it keeps the resident one rather
        than paying a 4-second reload.
        """
        if self.is_resident(model_id):
            return 0.0
        spec = self.registry.get(model_id)
        if spec is None:
            return float("inf")

        cost = float(spec.load_ms)
        # If it does not fit as-is, the eviction is part of the price.
        if spec.vram_gb > self.free_gb:
            cost += 150.0 * max(1, self._evictions_needed(spec))
        return cost

    def _evictions_needed(self, spec: ModelSpec) -> int:
        need = spec.vram_gb - self.free_gb
        if need <= 0:
            return 0
        freed, count = 0.0, 0
        for r in self._evictable():
            freed += r.vram_gb
            count += 1
            if freed >= need:
                break
        return count

    def fits(self, model_id: str) -> bool:
        """Could this model ever be admitted on this profile, with everything
        evictable evicted? Used by the router's constraint filter (§04.1)."""
        spec = self.registry.get(model_id)
        if spec is None:
            return False
        evictable_gb = sum(r.vram_gb for r in self._evictable())
        return spec.vram_gb <= self.free_gb + evictable_gb

    # ---------------------------------------------------------------- pinning

    def pin_profile_models(self) -> list[str]:
        """Load the profile's pinned set (embedder, router head) at boot.

        A pinned model that cannot be loaded is recorded and reported rather than
        raised: a missing embedder should degrade the router to its keyword
        fallback, not stop the box from booting.
        """
        pinned: list[str] = []
        self._pin_failures = []
        for model_id in self.profile.pinned:
            spec = self.registry.get(model_id)
            if spec is None:
                self._pin_failures.append(f"{model_id}: not in the registry")
                continue
            if spec.vram_gb > self.budget_gb:
                self._pin_failures.append(
                    f"{model_id}: needs {spec.vram_gb} GB, budget is {self.budget_gb:.1f} GB"
                )
                continue
            with self._load_lock:
                self._load(spec, pinned=True)
            pinned.append(model_id)
        return pinned

    @property
    def pin_failures(self) -> list[str]:
        return list(self._pin_failures)

    # ---------------------------------------------------------------- admission

    def admit(self, model_id: str, actor: str = "local") -> WardenDecision:
        """The chokepoint. Returns a decision; raises only when the model can
        never fit on this profile, which is a config error, not a runtime one."""
        with self._load_lock:
            spec = self.registry.get(model_id)
            if spec is None:
                raise KeyError(f"unknown model {model_id!r}")

            before = self.used_gb

            resident = self._residency.get(model_id)
            if resident is not None and self._backend_agrees(model_id):
                resident.touch()
                return WardenDecision(
                    model_id=model_id,
                    admitted=True,
                    was_resident=True,
                    reason="already resident, no load needed",
                    vram_before_gb=before,
                    vram_after_gb=before,
                )

            if resident is not None:
                # The table says resident, the backend disagrees. This happens
                # when something tears the backend down underneath us - the app
                # lifespan calling backend.shutdown() while this warden survives
                # as a process-level singleton, for instance.
                #
                # Repair it rather than serving a request against a model that is
                # not there, but RECORD it: silent self-healing would hide a real
                # divergence bug, and this table is what the VRAM meter shows.
                get_audit().append(
                    "model.residency_diverged",
                    actor=actor,
                    model=model_id,
                    detail=(
                        "residency claimed this model was loaded but the backend "
                        "does not have it; reloading"
                    ),
                    backend_loaded=self._backend_loaded(),
                )
                self._residency.pop(model_id, None)

            if spec.vram_gb > self.budget_gb:
                raise VramExhausted(
                    f"{model_id} needs {spec.vram_gb} GB but the whole budget on "
                    f"{self.profile.name} is {self.budget_gb:.1f} GB "
                    f"({self.profile.vram_gb} GB card, {self.profile.reserve_gb} GB "
                    f"reserved, {self.kv_cache_gb:.1f} GB KV cache). "
                    "Use a smaller profile or a smaller quant."
                )

            evicted = self._make_room(spec)
            started = time.perf_counter()
            self._load(spec, pinned=False)
            load_ms = (time.perf_counter() - started) * 1000

            decision = WardenDecision(
                model_id=model_id,
                admitted=True,
                was_resident=False,
                evicted=evicted,
                load_ms=load_ms,
                reason=(
                    f"loaded after evicting {', '.join(evicted)}"
                    if evicted
                    else "loaded into free VRAM"
                ),
                vram_before_gb=before,
                vram_after_gb=self.used_gb,
            )

            get_audit().append(
                "model.admit",
                actor=actor,
                model=model_id,
                **decision.as_dict(),
            )
            return decision

    def _make_room(self, spec: ModelSpec) -> list[str]:
        """Evict LRU-first until the model fits and a swap slot is free."""
        evicted: list[str] = []

        # Free VRAM.
        while spec.vram_gb > self.free_gb:
            candidates = self._evictable()
            if not candidates:
                raise VramExhausted(
                    f"{spec.id} needs {spec.vram_gb} GB, only {self.free_gb:.1f} GB free "
                    f"and nothing left to evict (pinned: {sorted(self.profile.pinned)})"
                )
            victim = candidates[0]
            self._unload(victim.model_id)
            evicted.append(victim.model_id)

        # Free a swap slot even if VRAM alone would allow it - the slot limit is
        # what stops a multimodal step from quietly thrashing the card.
        while len(self._evictable()) >= self.swap_slots:
            candidates = self._evictable()
            if not candidates:
                break
            victim = candidates[0]
            self._unload(victim.model_id)
            evicted.append(victim.model_id)

        return evicted

    # ---------------------------------------------------------------- backend

    def _load(self, spec: ModelSpec, pinned: bool) -> None:
        if self.backend is not None:
            self.backend.load(spec)
        now = time.monotonic()
        existing = self._residency.get(spec.id)
        self._residency[spec.id] = Residency(
            model_id=spec.id,
            vram_gb=spec.vram_gb,
            pinned=pinned,
            loaded_at=now,
            last_used=now,
            load_count=(existing.load_count + 1) if existing else 1,
        )

    def _unload(self, model_id: str) -> None:
        spec = self.registry.get(model_id)
        if self.backend is not None and spec is not None:
            self.backend.unload(spec)
        self._residency.pop(model_id, None)
        get_audit().append("model.evict", model=model_id, reason="lru")

    def evict_all(self, include_pinned: bool = False) -> list[str]:
        with self._load_lock:
            targets = [
                r.model_id
                for r in self._residency.values()
                if include_pinned or not r.pinned
            ]
            for model_id in targets:
                self._unload(model_id)
            return targets

    # ---------------------------------------------------------------- reporting

    def residency(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in sorted(
            self._residency.values(), key=lambda r: (not r.pinned, r.last_used)
        )]

    def state(self) -> dict[str, Any]:
        """The live VRAM meter on the sovereignty dashboard."""
        return {
            "profile": self.profile.name,
            "cpu_only": self.profile.cpu_only,
            "vram_gb": self.profile.vram_gb,
            "reserve_gb": self.profile.reserve_gb,
            "kv_cache_gb": round(self.kv_cache_gb, 2),
            "budget_gb": round(self.budget_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "swap_slots": self.swap_slots,
            "slots_in_use": len(self._evictable()),
            "resident": self.residency(),
            "pin_failures": self.pin_failures,
            # The card that is actually here, beside the budget derived from it.
            # A meter reading "6.2 / 9.8 GB" on a 6 GB card is not a rounding
            # error, it is a number nobody can defend.
            "gpu": (probe().as_dict() if probe() else None),
            "vram_note": self.vram_note,
        }


_warden: VramWarden | None = None
_warden_lock = threading.Lock()


def get_warden() -> VramWarden:
    global _warden
    if _warden is None:
        with _warden_lock:
            if _warden is None:
                from ..config import get_settings
                from .backends import get_backend

                st = get_settings()
                _warden = VramWarden(
                    profile=st.profile,
                    registry=get_registry_ref(),
                    backend=get_backend(),
                )
                _warden.pin_profile_models()
    return _warden


def get_registry_ref() -> ModelRegistry:
    from .registry import get_registry

    return get_registry()


def reset_warden() -> None:
    """Tests only."""
    global _warden
    _warden = None


def peek_warden() -> VramWarden | None:
    """The active warden, or None - without constructing one.

    Unlike get_warden(), never triggers pin_profile_models(). A caller that
    only wants to know whether this process spawned anything (so it can tear
    it down before exiting) must not spawn something new just by asking.
    """
    return _warden

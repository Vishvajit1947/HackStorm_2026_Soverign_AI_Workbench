"""The four-stage cascade, wired together (§04).

    filter -> classify -> score -> call -> verify -> (escalate once)

Two entry points:

    decide()  stages 1-3 only. Pure, fast, no VRAM touched, no generation. This is
              what the explainer panel calls and what the eval harness scores. It
              answers "which model, and why" without spending anything.

    route()   the whole thing including the model call and the verifier. Returns
              the output plus the full decision trail.

Keeping decide() side-effect-free matters more than it looks: it means the UI can
show the routing arithmetic for a prompt the user is still typing, and it means
the eval harness can measure routing accuracy without a GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..audit import get_audit
from ..config import get_settings
from ..runtime import get_backend, get_registry, get_warden
from ..sovereignty import get_policy
from ..types import (
    Candidate,
    GenRequest,
    GenResult,
    RouteDecision,
    RouteRequest,
    TaskClass,
)
from .classifier import Classification, TaskClassifier
from .constraints import apply_constraints, summarise, survivors
from .scoring import ScoringWeights, explain_margin, rank, score_candidates, winner
from .verifier import VerifierResult, next_tier_up, verify_output


class NoViableModel(RuntimeError):
    """Every candidate was eliminated. The reasons are on the decision."""

    def __init__(self, message: str, decision: RouteDecision):
        super().__init__(message)
        self.decision = decision


@dataclass
class RouteOutcome:
    """Everything the run timeline and the router panel need from one turn."""

    decision: RouteDecision
    classification: Classification
    result: GenResult | None = None
    verification: VerifierResult | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    quality_provenance: str = "seeded"

    @property
    def text(self) -> str:
        return self.result.text if self.result else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "decision": self.decision.explain(),
            "classification": self.classification.as_dict(),
            "verification": self.verification.as_dict() if self.verification else None,
            "attempts": self.attempts,
            "quality_provenance": self.quality_provenance,
            "model": self.result.model_id if self.result else None,
            "tokens": (
                {
                    "prompt": self.result.prompt_tokens,
                    "completion": self.result.completion_tokens,
                }
                if self.result
                else None
            ),
        }


class Router:
    def __init__(
        self,
        registry=None,
        warden=None,
        backend=None,
        config: dict[str, Any] | None = None,
        prototypes_path=None,
    ):
        settings = get_settings()
        self.settings = settings
        self.registry = registry if registry is not None else get_registry()
        self.warden = warden if warden is not None else get_warden()
        self.backend = backend if backend is not None else get_backend()

        self.config = config if config is not None else self._load_config()
        self.weights = ScoringWeights.from_config(self.config.get("scoring", {}))
        self.verifier_config = self.config.get("verifier", {})

        classifier_config = self.config.get("classifier", {})
        self.classifier = TaskClassifier(
            prototypes_path=prototypes_path or (settings.config_dir / "prototypes.yaml"),
            registry=self.registry,
            warden=self.warden,
            backend=self.backend,
            k=int(classifier_config.get("k", 5)),
            temperature=float(classifier_config.get("temperature", 0.08)),
            min_confidence=float(classifier_config.get("min_confidence", 0.35)),
        )
        self._lock = threading.RLock()

    def _load_config(self) -> dict[str, Any]:
        path = self.settings.config_dir / "router.yaml"
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ---------------------------------------------------------------- provenance

    def quality_provenance(self, task_class: str | None = None) -> str:
        """Whether the numbers stage 3 ranks on are measured or seeded.

        This is surfaced on every decision and every API response on purpose. The
        router panel is the thing a judge will interrogate, and 'where does 0.84
        come from?' has to have an answer visible on the screen. Until the eval
        harness writes `measured: true` into models.yaml, that answer is
        'these are seeds' - stated by us, before anyone has to ask.

        The question is about the numbers stage 3 ACTUALLY RANKS ON, so it is
        asked of the models this profile can choose between - not of the whole
        registry. Half the registry has no weights on this box and can never be
        a candidate; letting those keep the warning up meant a fully measured
        profile still told the panel its numbers were guesses.

        `task_class` narrows it further: quality_measured names the classes a
        run actually measured, and a model measured on code but not on vision
        has not earned "measured" for a visual_doc decision.
        """
        try:
            available = self.warden.profile.available
        except Exception:                        # noqa: BLE001 - degrade, never raise
            available = None
        specs = [
            s for s in self.registry.available(available)
            if s.role == "generation" and s.routable
        ]
        if not specs:
            return "seeded"

        def _measured(spec) -> bool:
            flag = getattr(spec, "quality_measured", False)
            if isinstance(flag, (list, tuple, set)):
                # A list names the classes this model was measured on.
                return bool(flag) and (task_class is None or task_class in flag)
            return bool(flag)

        measured = [s for s in specs if _measured(s)]
        if len(measured) == len(specs):
            return "measured"
        if measured:
            # Neither claim is true, so make the mixed state its own answer
            # rather than rounding it to whichever is more convenient.
            return "partially_measured"
        return "seeded"

    # ---------------------------------------------------------------- stages 1-3

    def decide(self, req: RouteRequest) -> tuple[RouteDecision, Classification]:
        """Stages 1-3. No VRAM touched, no tokens generated, no side effects."""
        started = time.perf_counter()

        specs = self.registry.available(self.warden.profile.available)

        # Stage 2 first when a class is forced, otherwise after filtering - the
        # classifier does not depend on the candidate set, so order is free here
        # and doing it first lets the constraint reasons mention the class.
        classification = self.classifier.classify(req.prompt, forced=req.forced_class)

        max_tier = get_policy().max_tier(req.label)
        candidates = apply_constraints(specs, req, warden=self.warden, max_tier=max_tier)

        if req.forced_model is not None:
            candidates = self._force_model(candidates, req.forced_model)

        score_candidates(
            candidates,
            task_class=classification.task_class.value,
            confidence=classification.confidence,
            registry=self.registry,
            warden=self.warden,
            weights=self.weights,
        )

        chosen = winner(candidates)
        elapsed = (time.perf_counter() - started) * 1000
        margin = explain_margin(candidates)

        decision = RouteDecision(
            chosen=chosen.model_id if chosen else None,
            task_class=classification.task_class,
            confidence=classification.confidence,
            candidates=candidates,
            reason=self._reason(classification, candidates, chosen, margin),
            elapsed_ms=elapsed,
        )
        return decision, classification

    @staticmethod
    def _force_model(candidates: list[Candidate], model_id: str) -> list[Candidate]:
        found = False
        for candidate in candidates:
            if candidate.model_id == model_id:
                found = True
                if not candidate.survived:
                    # An operator override does not get to bypass a hard
                    # constraint - a model with no vision tower still cannot read
                    # an image. Say so rather than silently obeying.
                    candidate.eliminated_by = (
                        f"forced by caller but still ineligible: {candidate.eliminated_by}"
                    )
            elif candidate.survived:
                candidate.eliminated_by = f"not the forced model ({model_id})"
        if not found:
            for candidate in candidates:
                candidate.eliminated_by = f"forced model {model_id!r} is not registered"
        return candidates

    def _reason(
        self,
        classification: Classification,
        candidates: list[Candidate],
        chosen: Candidate | None,
        margin: float | None,
    ) -> str:
        if chosen is None:
            reasons = {c.eliminated_by for c in candidates if c.eliminated_by}
            return (
                f"no viable model for {classification.task_class.value}: "
                + "; ".join(sorted(r for r in reasons if r)[:3])
            )

        bits = [
            f"cls={classification.task_class.value}",
            f"p={classification.confidence:.2f}",
            f"via {classification.method}",
            summarise(candidates),
        ]
        if chosen.resident:
            bits.append("winner already resident")
        if margin is not None and margin < 0.05:
            bits.append(f"close call, margin {margin:.3f}")
        if classification.degraded:
            bits.append("CLASSIFIER DEGRADED")
        return ", ".join(bits)

    # ---------------------------------------------------------------- full route

    def route(
        self,
        req: RouteRequest,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        system: str | None = None,
        grammar: str | None = None,
        kb_grounded: bool = False,
        run_id: str | None = None,
        step_idx: int | None = None,
        actor: str = "local",
    ) -> RouteOutcome:
        """The whole cascade: decide, call, verify, escalate once if needed."""
        with self._lock:
            decision, classification = self.decide(req)

            if decision.chosen is None:
                self._audit(decision, run_id, step_idx, actor, req)
                raise NoViableModel(decision.reason, decision)

            attempts: list[dict[str, Any]] = []
            available_tiers = [
                self.registry.get(c.model_id).tier
                for c in survivors(decision.candidates)
                if self.registry.get(c.model_id)
            ]

            model_id = decision.chosen
            max_retries = int(self.verifier_config.get("max_retries", 1))

            for attempt in range(max_retries + 1):
                result = self._generate(
                    model_id, req, max_tokens, temperature, system, grammar
                )
                verification = self._verify(
                    result, classification.task_class, kb_grounded
                )
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "model": model_id,
                        "ok": verification.ok,
                        "reason": verification.reason,
                        "checks": verification.checks,
                        "elapsed_ms": round(result.elapsed_ms, 1),
                    }
                )

                if verification.ok or attempt >= max_retries:
                    break

                spec = self.registry.get(model_id)
                target_tier = next_tier_up(spec.tier if spec else 0, available_tiers)
                if target_tier is None:
                    attempts[-1]["escalation"] = (
                        "already the most capable model available - returning the "
                        "failure rather than pretending it passed"
                    )
                    break

                escalated = self._pick_at_tier(decision.candidates, target_tier)
                if escalated is None:
                    attempts[-1]["escalation"] = f"no candidate at tier {target_tier}"
                    break

                attempts[-1]["escalation"] = (
                    f"{verification.reason} -> escalating to tier {target_tier}"
                )
                decision.escalated_from = model_id
                model_id = escalated
                decision.chosen = escalated

            self._audit(decision, run_id, step_idx, actor, req, verification)

            return RouteOutcome(
                decision=decision,
                classification=classification,
                result=result,
                verification=verification,
                attempts=attempts,
                quality_provenance=self.quality_provenance(
                    classification.task_class.value),
            )

    def _pick_at_tier(self, candidates: list[Candidate], tier: int) -> str | None:
        for candidate in rank(candidates):
            spec = self.registry.get(candidate.model_id)
            if spec is not None and spec.tier == tier:
                return candidate.model_id
        return None

    def _generate(
        self,
        model_id: str,
        req: RouteRequest,
        max_tokens: int,
        temperature: float,
        system: str | None,
        grammar: str | None,
    ) -> GenResult:
        # Stage 4: the warden loads it if it is not resident.
        self.warden.admit(model_id)
        spec = self.registry.get(model_id)
        return self.backend.generate(
            spec,
            GenRequest(
                prompt=req.prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                grammar=grammar,
            ),
        )

    def _verify(
        self, result: GenResult, task_class: TaskClass, kb_grounded: bool
    ) -> VerifierResult:
        cfg = self.verifier_config
        return verify_output(
            result.text,
            task_class,
            finish_reason=result.finish_reason,
            min_output_chars=int(cfg.get("min_output_chars", 12)),
            require_citations=cfg.get("require_citations", []),
            require_parseable_code=cfg.get("require_parseable_code", []),
            enabled=bool(cfg.get("enabled", True)),
            kb_grounded=kb_grounded,
        )

    # ---------------------------------------------------------------- audit

    def _audit(
        self,
        decision: RouteDecision,
        run_id: str | None,
        step_idx: int | None,
        actor: str,
        req: RouteRequest,
        verification: VerifierResult | None = None,
    ) -> None:
        get_audit().append(
            "route.decide",
            actor=actor,
            run_id=run_id,
            step_idx=step_idx,
            data_class=req.label.value,
            model=decision.chosen,
            route_reason=decision.reason,
            task_class=decision.task_class.value,
            confidence=round(decision.confidence, 4),
            escalated_from=decision.escalated_from,
            quality_provenance=self.quality_provenance(decision.task_class.value),
            verified=verification.ok if verification else None,
        )
        # The decisions table only feeds the explainer panel, so a failure here
        # must not take down a run that is otherwise fine. It is still recorded
        # in the audit chain rather than swallowed - a panel that quietly stops
        # updating is its own kind of demo failure, and this is the breadcrumb.
        try:
            from ..store import get_db

            get_db().add_decision(decision, run_id, step_idx)
        except Exception as exc:  # noqa: BLE001 - panel write, never fatal
            get_audit().append(
                "route.panel_write_failed",
                actor=actor,
                run_id=run_id,
                error=f"{type(exc).__name__}: {exc}",
            )


_router: Router | None = None
_router_lock = threading.Lock()


def get_router() -> Router:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = Router()
    return _router


def reset_router() -> None:
    """Tests only."""
    global _router
    _router = None

"""Stage 2 - the task classifier (§04).

Embed the request with the already-resident embedder, then classify with a
k-nearest-neighbour head over a labelled prototype index.

The choice of kNN over a fine-tuned head buys a real property: **improving the
router means adding examples, not retraining a model.** When a routing decision is
wrong in rehearsal you paste the prompt into config/prototypes.yaml under the
right label and it is fixed in seconds, with no GPU time and no risk of regressing
the other classes.

Two heads, and the router always says which one answered:

    embedding   the real one. Cosine kNN over prototype vectors.
    keyword     the fallback. Weighted term overlap, used when no embedder is
                resident (TINY profile, or a pin failure). Visibly worse, and
                reported as `method: keyword` so nobody mistakes a degraded
                router for a working one.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..types import ALL_TASK_CLASSES, TaskClass

if TYPE_CHECKING:
    from ..runtime.registry import ModelRegistry
    from ..runtime.warden import VramWarden

# Terms that mark a class when no embedder is available. Deliberately small and
# hand-written: this is a fallback, not a second implementation to maintain.
KEYWORD_HINTS: dict[str, list[str]] = {
    "code_generation": [
        "script", "python", "code", "function", "unit-test", "unit test", "debug",
        "refactor", "parser", "implement", "regex", "class", "test",
    ],
    "doc_summary": [
        "summarise", "summarize", "summary", "condense", "key points", "brief",
        "highlights", "boil", "extract the main", "executive summary",
    ],
    "visual_doc": [
        "scanned", "scan", "photo", "photograph", "image", "handwritten", "read this",
        "drawing", "p&id", "transcribe", "nameplate", "thermography",
    ],
    "numerical_calc": [
        "calculate", "compute", "pressure drop", "npsh", "reynolds", "lmtd",
        "work out", "heat duty", "flow velocity", "sizing", "what is the",
    ],
    "official_drafting": [
        "draft", "memo", "letter", "approval note", "formal", "prepare a note",
        "circular", "justification", "request for", "covering letter",
    ],
    "kb_qa": [
        "sop", "our standards", "procedure", "manual", "policy", "what does our",
        "according to our", "which document", "clause", "documented",
    ],
    "agentic_multi": [
        " and then ", "and draft", "and export", "and put it", "and write",
        "and produce", "and prepare", "and build", "cross-check", "compare these",
    ],
}


@dataclass
class Classification:
    task_class: TaskClass
    confidence: float
    method: str                        # embedding | keyword | forced
    scores: dict[str, float] = field(default_factory=dict)
    neighbours: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.task_class.value,
            "p": round(self.confidence, 4),
            "method": self.method,
            "degraded": self.degraded,
            "detail": self.detail,
            "scores": {k: round(v, 4) for k, v in sorted(
                self.scores.items(), key=lambda kv: -kv[1]
            )},
            "neighbours": self.neighbours,
        }


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9&]+", text.lower())


class TaskClassifier:
    def __init__(
        self,
        prototypes_path: Path,
        registry: ModelRegistry,
        warden: VramWarden | None = None,
        backend: Any = None,
        k: int = 5,
        temperature: float = 0.08,
        min_confidence: float = 0.35,
    ):
        self.prototypes_path = prototypes_path
        self.registry = registry
        self.warden = warden
        self.backend = backend
        self.k = k
        self.temperature = temperature
        self.min_confidence = min_confidence

        self._lock = threading.RLock()
        self._prototypes: list[tuple[str, str]] = []      # (label, text)
        self._vectors: list[list[float]] | None = None
        self._vector_model: str | None = None
        self.load_prototypes()

    # ---------------------------------------------------------------- data

    def load_prototypes(self) -> int:
        """Re-read prototypes.yaml. Safe to call live - this is what makes
        'add an example, fix the routing' a seconds-long loop."""
        with self.prototypes_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        pairs: list[tuple[str, str]] = []
        unknown: set[str] = set()
        for label, examples in raw.items():
            if label not in ALL_TASK_CLASSES:
                unknown.add(label)
                continue
            for text in examples or []:
                pairs.append((label, str(text)))

        if unknown:
            raise ValueError(
                f"prototypes.yaml has labels that are not task classes: "
                f"{sorted(unknown)}; expected some of {ALL_TASK_CLASSES}"
            )

        with self._lock:
            self._prototypes = pairs
            self._vectors = None          # force re-embed on next use
            self._vector_model = None
        return len(pairs)

    @property
    def prototype_count(self) -> int:
        return len(self._prototypes)

    def counts_per_class(self) -> dict[str, int]:
        counts = dict.fromkeys(ALL_TASK_CLASSES, 0)
        for label, _ in self._prototypes:
            counts[label] += 1
        return counts

    # ---------------------------------------------------------------- embedding

    def _embedder(self):
        """The pinned embedder, if there is one. Never raises - a missing embedder
        must degrade the router to keywords, not take down the request."""
        if self.backend is None:
            return None
        profile_ids = self.warden.profile.available if self.warden else None
        spec = self.registry.embedder(profile_ids)
        if spec is None:
            return None
        try:
            if self.warden is not None and not self.warden.is_resident(spec.id):
                self.warden.admit(spec.id)
            return spec
        except Exception:  # noqa: BLE001 - degrade, never fail the request
            return None

    def _ensure_vectors(self, spec) -> list[list[float]] | None:
        with self._lock:
            if self._vectors is not None and self._vector_model == spec.id:
                return self._vectors
        try:
            texts = [text for _, text in self._prototypes]
            vectors = self.backend.embed(spec, texts)
        except Exception:  # noqa: BLE001
            return None
        with self._lock:
            self._vectors = vectors
            self._vector_model = spec.id
            return vectors

    # ---------------------------------------------------------------- classify

    def classify(self, prompt: str, forced: TaskClass | None = None) -> Classification:
        if forced is not None:
            return Classification(
                task_class=forced,
                confidence=1.0,
                method="forced",
                scores={forced.value: 1.0},
                detail="class supplied by the caller, classifier bypassed",
            )

        spec = self._embedder()
        if spec is not None:
            result = self._classify_by_embedding(prompt, spec)
            if result is not None:
                return result

        return self._classify_by_keyword(
            prompt,
            reason=(
                "no embedder resident on this profile"
                if spec is None
                else f"embedding failed on {spec.id}"
            ),
        )

    def _classify_by_embedding(self, prompt: str, spec) -> Classification | None:
        vectors = self._ensure_vectors(spec)
        if not vectors:
            return None
        try:
            (query,) = self.backend.embed(spec, [prompt])
        except Exception:  # noqa: BLE001
            return None

        sims = [
            (self._prototypes[i][0], self._cosine(query, vec), self._prototypes[i][1])
            for i, vec in enumerate(vectors)
        ]
        sims.sort(key=lambda t: -t[1])
        top = sims[: self.k]

        # Softmax over the top-k similarities, summed per class. Weighting by
        # similarity rather than counting votes means one very close neighbour can
        # outrank three loose ones, which is the behaviour you want on a prompt
        # that is nearly a verbatim prototype.
        scores = dict.fromkeys(ALL_TASK_CLASSES, 0.0)
        weights = [math.exp(sim / max(self.temperature, 1e-6)) for _, sim, _ in top]
        total = sum(weights) or 1.0
        for (label, _, _), weight in zip(top, weights, strict=True):
            scores[label] += weight / total

        best = max(scores, key=lambda label: scores[label])
        confidence = scores[best]

        return Classification(
            task_class=TaskClass(best),
            confidence=confidence,
            method="embedding",
            scores=scores,
            neighbours=[
                {"label": label, "similarity": round(sim, 4), "example": text[:80]}
                for label, sim, text in top
            ],
            detail=f"kNN k={self.k} over {len(vectors)} prototypes via {spec.id}",
        )

    def _classify_by_keyword(self, prompt: str, reason: str) -> Classification:
        lowered = prompt.lower()
        tokens = set(_tokenise(prompt))

        # Collect WHICH hints matched before scoring, because two hints in the
        # same list can describe one piece of evidence. "unit test" and "test"
        # both fire on `write a unit test`, so code_generation banked 3 points
        # for a phrase a human wrote once - and a rival class holding one
        # genuine hint was outvoted 3:1 instead of 2:1. Keeping only the longest
        # match per class counts the evidence, not the wording of the list.
        matched: dict[str, set[str]] = {label: set() for label in ALL_TASK_CLASSES}
        for label, hints in KEYWORD_HINTS.items():
            for hint in hints:
                if (hint in lowered) if " " in hint else (hint in tokens):
                    matched[label].add(hint)

        raw = dict.fromkeys(ALL_TASK_CLASSES, 0.0)
        for label, hits in matched.items():
            for hint in hits:
                if any(hint != other and hint in other for other in hits):
                    continue                       # subsumed by a longer match
                raw[label] += 2.0 if " " in hint else 1.0

        total = sum(raw.values())
        if total == 0:
            # Nothing matched. Do not guess a specialist - hand it to the planner
            # tier, which is the safe default for an unrecognised request.
            return Classification(
                task_class=TaskClass.AGENTIC_MULTI,
                confidence=0.0,
                method="keyword",
                scores=raw,
                degraded=True,
                detail=f"{reason}; no keywords matched either, defaulting to planner tier",
            )

        scores = {label: value / total for label, value in raw.items()}
        top = max(scores.values())
        # A bag of keywords ties often - two classes holding one hint each is an
        # ordinary outcome, not a rare one. `max()` then picks by dict insertion
        # order and reports full confidence in a coin flip. Name the tie instead.
        tied = [label for label in ALL_TASK_CLASSES if scores[label] == top]

        if len(tied) > 1 and TaskClass.AGENTIC_MULTI.value in tied:
            # A multi-step marker ("... and then draft ...") tying with a
            # specialist means the prompt contains BOTH. The planner can run the
            # specialist's work as a step; the specialist cannot recover the half
            # of the request it was never told about.
            best = TaskClass.AGENTIC_MULTI.value
        else:
            best = tied[0]

        detail = f"{reason}; fell back to keyword matching - ACCURACY IS DEGRADED"
        if len(tied) > 1:
            detail += (
                f"; TIED at {top:.2f} between {', '.join(tied)} - chose {best}"
            )

        return Classification(
            task_class=TaskClass(best),
            confidence=scores[best],
            method="keyword",
            scores=scores,
            degraded=True,
            detail=detail,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

"""L2 - the router (§04).

A four-stage cascade: two cheap deterministic stages, one learned stage, one
verification stage. Each stage narrows or scores, and every decision is recorded
so the UI can explain it without the presenter narrating.

    1  constraints.py   hard eliminations        < 1 ms
    2  classifier.py    embed + kNN head        ~15 ms
    3  scoring.py       utility score           < 1 ms
    4  (warden+backend) the model call
    5  verifier.py      escalate one tier once
"""

from .cascade import RouteOutcome, Router, get_router
from .classifier import Classification, TaskClassifier
from .constraints import apply_constraints
from .scoring import ScoringWeights, score_candidates
from .verifier import VerifierResult, verify_output

__all__ = [
    "Classification",
    "RouteOutcome",
    "Router",
    "ScoringWeights",
    "TaskClassifier",
    "VerifierResult",
    "apply_constraints",
    "get_router",
    "score_candidates",
    "verify_output",
]

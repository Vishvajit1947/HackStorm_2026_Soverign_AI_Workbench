"""Shared vocabulary. Every layer speaks these types; nothing else crosses a boundary."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------- classification

class Label(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"

    @property
    def rank(self) -> int:
        return _LABEL_ORDER.index(self)

    @staticmethod
    def max(*labels: Label) -> Label:
        """A run inherits the strictest label of anything it has touched."""
        return max(labels, key=lambda x: x.rank) if labels else Label.INTERNAL


_LABEL_ORDER = [Label.PUBLIC, Label.INTERNAL, Label.CONFIDENTIAL, Label.RESTRICTED]


class SideEffect(str, Enum):
    READ = "READ"
    COMPUTE = "COMPUTE"
    EXECUTE = "EXECUTE"
    WRITE = "WRITE"
    DELETE = "DELETE"


# ---------------------------------------------------------------- task classes

class TaskClass(str, Enum):
    CODE_GENERATION = "code_generation"
    DOC_SUMMARY = "doc_summary"
    VISUAL_DOC = "visual_doc"
    NUMERICAL_CALC = "numerical_calc"
    OFFICIAL_DRAFTING = "official_drafting"
    KB_QA = "kb_qa"
    AGENTIC_MULTI = "agentic_multi"


ALL_TASK_CLASSES = [t.value for t in TaskClass]


# ---------------------------------------------------------------- work mode

class WorkMode(str, Enum):
    """Which KIND of deliverable the operator asked this run for.

    The brief wants a coding request handled differently from a document
    request, and wants a human to be able to SEE that happen. Left to the
    classifier alone, "draft the NPSH check" is a coin toss between
    CODE_GENERATION and OFFICIAL_DRAFTING, and the operator has no way to say
    which they meant short of rewording the goal until the router agrees.

    So the toggle is an input, not a decoration. It does two separate things,
    and both are load-bearing:

      ROUTING    it forces the task class the planner is routed under, so the
                 quality column the cascade ranks on is the one for the work
                 actually being asked for. Flipping the toggle can change which
                 model plans the run - that is the point.

      PLANNING   it appends a directive to the planner's system prompt naming
                 the tool that can actually produce that deliverable. Without
                 it a small model in "code" mode finishes by writing a prose
                 description of a program to a .txt file, which satisfies the
                 goal as worded and is not code.

    AUTO is the honest default: no class is forced and the classifier decides,
    exactly as it did before the toggle existed.
    """

    AUTO = "auto"
    CODE = "code"
    DOCUMENT = "document"

    @property
    def forced_class(self) -> TaskClass:
        """The class the planner's route request is pinned to.

        AGENTIC_MULTI for AUTO, which is what the planner always used - a run in
        AUTO mode routes exactly as it did before this existed.
        """
        return _MODE_CLASS[self]


_MODE_CLASS = {
    WorkMode.AUTO: TaskClass.AGENTIC_MULTI,
    WorkMode.CODE: TaskClass.CODE_GENERATION,
    WorkMode.DOCUMENT: TaskClass.OFFICIAL_DRAFTING,
}


# ---------------------------------------------------------------- requests

@dataclass
class RouteRequest:
    """Everything stage 1 needs to eliminate candidates before anything is scored."""

    prompt: str
    has_images: bool = False
    needs_tools: bool = False
    needs_embedding: bool = False
    min_context: int = 2048
    label: Label = Label.INTERNAL
    user_role: str = "engineer"
    forced_class: TaskClass | None = None
    forced_model: str | None = None


@dataclass
class Candidate:
    """One model's survival through the cascade, with the arithmetic kept."""

    model_id: str
    tier: int
    resident: bool
    quality: float
    score: float | None = None
    terms: dict[str, float] = field(default_factory=dict)
    eliminated_by: str | None = None

    @property
    def survived(self) -> bool:
        return self.eliminated_by is None


@dataclass
class RouteDecision:
    """The full audit trail of one routing decision - this is what the UI panel renders."""

    chosen: str | None
    task_class: TaskClass
    confidence: float
    candidates: list[Candidate]
    reason: str
    escalated_from: str | None = None
    elapsed_ms: float = 0.0

    def explain(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "class": self.task_class.value,
            "p": round(self.confidence, 4),
            "reason": self.reason,
            "escalated_from": self.escalated_from,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "candidates": [
                {
                    "model": c.model_id,
                    "tier": c.tier,
                    "resident": c.resident,
                    "quality": round(c.quality, 3),
                    "score": round(c.score, 4) if c.score is not None else None,
                    "terms": {k: round(v, 4) for k, v in c.terms.items()},
                    "eliminated_by": c.eliminated_by,
                }
                for c in self.candidates
            ],
        }


# ---------------------------------------------------------------- generation

@dataclass
class GenRequest:
    prompt: str
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.2
    stop: list[str] = field(default_factory=list)
    grammar: str | None = None          # GBNF, forced tool JSON (§05)
    images: list[str] = field(default_factory=list)


@dataclass
class GenResult:
    text: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    finish_reason: str = "stop"


# ---------------------------------------------------------------- agent state

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    DONE = "done"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class Budget:
    """Exhausted means stop and report. It never means loop again. (§05)"""

    max_steps: int = 12
    max_wall_s: int = 480
    max_tool_calls: int = 20

    steps_used: int = 0
    tool_calls_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def exhausted(self) -> str | None:
        if self.steps_used >= self.max_steps:
            return f"step budget ({self.max_steps}) exhausted"
        if self.tool_calls_used >= self.max_tool_calls:
            return f"tool-call budget ({self.max_tool_calls}) exhausted"
        elapsed = time.monotonic() - self.started_at
        if elapsed >= self.max_wall_s:
            return f"wall-clock budget ({self.max_wall_s}s) exhausted"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_wall_s": self.max_wall_s,
            "max_tool_calls": self.max_tool_calls,
            "steps_used": self.steps_used,
            "tool_calls_used": self.tool_calls_used,
            "elapsed_s": round(time.monotonic() - self.started_at, 1),
        }


@dataclass
class Step:
    idx: int
    intent: str
    tool: str | None
    args: dict[str, Any] = field(default_factory=dict)
    acceptance: str = ""             # what the critic checks, in plain language
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    requires_approval: bool = False


@dataclass
class Observation:
    step_idx: int
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    user: str = "local"
    goal: str = ""
    label: Label = Label.INTERNAL
    status: RunStatus = RunStatus.PLANNING
    budget: Budget = field(default_factory=Budget)
    steps: list[Step] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    decisions: list[RouteDecision] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    error: str | None = None
    # Which output the operator asked for. Persisted, because a finished run has
    # to be able to say which toggle produced it - a .docx in the Code tab with
    # no explanation is exactly the kind of unexplained result this build refuses.
    mode: WorkMode = WorkMode.AUTO

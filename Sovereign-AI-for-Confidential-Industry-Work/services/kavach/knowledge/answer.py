"""Grounded answering (§07).

    retrieve -> build context -> route (kb_grounded) -> check every citation

Three refusals live here, and each one is a place the system could have produced
something that looked like an answer.

  NOTHING RETRIEVED -> no answer. Not "the model's best recollection". If the
  corpus has no passage for the question, the honest output is that the corpus
  has no passage for the question. A model asked a plant question with no context
  will still produce fluent, specific, wrong text about interlock setpoints.

  CITATIONS DO NOT RESOLVE -> the text is not returned as the answer. It comes
  back under `rejected_answer` with the reason, so it can be inspected and
  screenshotted for what it is, but nothing downstream can mistake it for a
  grounded result.

  THE CONTEXT IS SIMULATED -> refused, for the same reason the eval harness
  refuses to write quality numbers measured on the mock backend (§04).

The run inherits the strictest label of anything retrieved, so a question that
pulls in one CONFIDENTIAL clause makes the whole turn CONFIDENTIAL - which is
what policy.yaml has always said and what nothing before this phase could
actually enforce.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..audit import get_audit
from ..types import Label, RouteRequest, TaskClass
from .citations import CITATION_INSTRUCTIONS, GroundingResult, check_grounding, render_sources
from .config import load_knowledge_config
from .retrieval import RetrievalResult, Retriever, context_block, get_retriever
from .types import Extraction

# A model instructed to cite what it was given will, when the passages do not
# answer the question, say exactly that - and then cite a passage anyway,
# because it was also told every claim needs a citation. That citation resolves,
# so grounding passes, and a refusal goes out stamped "grounded in 1 cited
# passage(s)": a non-answer wearing the badge of an answer. The empty-retrieval
# guard cannot catch it, because RRF always returns its top k however weak the
# match - and after OCR tripled this corpus, some passage always ranks.
#
# Deliberately NOT a relevance-score floor: RRF scores are ordinal, not
# calibrated, so any threshold would also silently discard good answers on a
# corpus whose scores drift. The model saying the passages do not answer the
# question is the strongest signal available, and it is what the model is
# supposed to say. Both halves are required - a declination ABOUT the retrieved
# material - so "the vessel does not contain hydrogen" is not mistaken for one.
_DECLINED = re.compile(
    r"(?:passages?|documents?|context|excerpts?|sources?|records?)[^.]{0,60}?"
    r"(?:do|does)\s+not\s+(?:contain|include|mention|provide|specify|address|cover)"
    r"|(?:cannot|can't|could not|unable to)\s+(?:provide|give|determine|find|answer)"
    r"[^.]{0,80}?(?:based on|from|in)\s+(?:the\s+)?"
    r"(?:given|provided|available|retrieved|supplied)?\s*"
    r"(?:passages?|documents?|context|excerpts?|sources?)",
    re.IGNORECASE,
)

ANSWER_SYSTEM = """\
You are an offline engineering assistant for a refinery. You answer only from the
plant documents you are given, and you cite every claim.\
"""


@dataclass
class GroundedAnswer:
    question: str
    ok: bool
    headline: str
    retrieval: RetrievalResult
    answer: str | None = None
    rejected_answer: str | None = None
    grounding: GroundingResult | None = None
    model: str | None = None
    decision: dict[str, Any] | None = None
    label: Label = Label.PUBLIC
    elapsed_ms: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    # The citations resolved, and the prose around them is still synthetic. Those
    # are separate facts and collapsing them is how a mock run gets screenshotted
    # as a product demo.
    simulated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ok": self.ok,
            "headline": self.headline,
            "answer": self.answer,
            "rejected_answer": self.rejected_answer,
            "simulated": self.simulated,
            "grounding": self.grounding.as_dict() if self.grounding else None,
            "sources": self.sources,
            "model": self.model,
            "label": self.label.value,
            "decision": self.decision,
            "retrieval": self.retrieval.as_dict(),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def build_prompt(question: str, context: str) -> str:
    return (
        "Passages retrieved from the plant document set:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"{CITATION_INSTRUCTIONS}\n\n"
        f"Question: {question}"
    )


class Answerer:
    def __init__(self, retriever: Retriever | None = None, router=None, config=None):
        self._retriever = retriever
        self._router = router
        self._config = config

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    @property
    def router(self):
        if self._router is None:
            from ..router import get_router

            self._router = get_router()
        return self._router

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_knowledge_config()
        return self._config

    @staticmethod
    def _is_simulated(model_id: str | None) -> bool:
        """Did the text come off a real model?

        A grounded answer built by the mock backend has real citations to real
        passages and synthetic prose between them. That is a genuinely useful
        thing - it exercises the whole citation path offline - and it is not an
        answer. Checkpoint R5 is where this flips.
        """
        if model_id is None:
            return False
        try:
            from ..runtime import get_registry
            from ..runtime.registry import Provenance

            spec = get_registry().get(model_id)
            return spec is not None and spec.provenance is Provenance.SIMULATED
        except Exception:  # noqa: BLE001 - never fail an answer over a status flag
            return False

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        role: str = "engineer",
        run_id: str | None = None,
        step_idx: int | None = None,
        max_tokens: int = 700,
    ) -> GroundedAnswer:
        started = time.perf_counter()
        cfg = self.config.get("citations", {})
        max_chunks = int(cfg.get("max_context_chunks", 8))

        retrieval = self.retriever.search(question, top_k=top_k, role=role)
        label = retrieval.effective_label
        audit = get_audit()

        def finish(result: GroundedAnswer) -> GroundedAnswer:
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            audit.append(
                "kb.answer",
                run_id=run_id,
                step_idx=step_idx,
                data_class=result.label.value,
                question=question[:300],
                grounded=result.ok,
                headline=result.headline[:300],
                citations=len(result.grounding.resolved) if result.grounding else 0,
                model=result.model,
                withheld=retrieval.withheld,
                coverage_complete=retrieval.coverage.complete,
            )
            return result

        if not retrieval.hits:
            reason = "no passage in the corpus matched this question"
            if retrieval.withheld:
                reason = (
                    f"{reason}; {retrieval.withheld} passage(s) were withheld - "
                    f"{retrieval.withheld_reason}"
                )
            if retrieval.coverage.warning:
                reason = f"{reason}. {retrieval.coverage.warning}"
            return finish(
                GroundedAnswer(
                    question=question,
                    ok=False,
                    headline=reason,
                    retrieval=retrieval,
                    label=label,
                )
            )

        if any(h.chunk.extraction is Extraction.SIMULATED for h in retrieval.hits):
            return finish(
                GroundedAnswer(
                    question=question,
                    ok=False,
                    headline=(
                        "refusing to answer: the retrieved passages include SIMULATED "
                        "transcriptions. That text was never read off a page, so an "
                        "answer built on it would carry citations to something that "
                        "does not exist."
                    ),
                    retrieval=retrieval,
                    label=label,
                    sources=render_sources(retrieval.hits),
                )
            )

        context = context_block(retrieval.hits, max_chunks=max_chunks)
        prompt = build_prompt(question, context)

        outcome = self.router.route(
            RouteRequest(
                prompt=prompt,
                label=label,
                user_role=role,
                forced_class=TaskClass.KB_QA,
                # The context block is the bulk of the prompt, so the model must
                # have room for it plus its answer. A model whose window is too
                # small is eliminated in stage 1 rather than silently truncating
                # the passage it is about to cite.
                min_context=min(32768, len(prompt) // 3 + max_tokens + 512),
            ),
            system=ANSWER_SYSTEM,
            max_tokens=max_tokens,
            kb_grounded=True,
            run_id=run_id,
            step_idx=step_idx,
        )

        text = outcome.text
        grounding = check_grounding(text, retrieval.hits)
        sources = render_sources(retrieval.hits)

        # Checked before grounding: a declination whose citation happens not to
        # resolve is still a declination, and "the model declined" is a more
        # useful thing to tell a reader than "answer rejected".
        if _DECLINED.search(text):
            return finish(
                GroundedAnswer(
                    question=question,
                    ok=False,
                    headline=(
                        "the model declined - the retrieved passages do not answer "
                        "this question. This is a gap in the corpus, not an answer."
                    ),
                    retrieval=retrieval,
                    rejected_answer=text,
                    grounding=grounding,
                    model=outcome.result.model_id if outcome.result else None,
                    decision=outcome.decision.explain(),
                    label=label,
                    sources=sources,
                )
            )

        if not grounding.ok:
            return finish(
                GroundedAnswer(
                    question=question,
                    ok=False,
                    headline=f"answer rejected - {grounding.reason}",
                    retrieval=retrieval,
                    rejected_answer=text,
                    grounding=grounding,
                    model=outcome.result.model_id if outcome.result else None,
                    decision=outcome.decision.explain(),
                    label=label,
                    sources=sources,
                )
            )

        simulated = self._is_simulated(outcome.result.model_id if outcome.result else None)
        headline = f"grounded in {len(grounding.resolved)} cited passage(s)"
        if simulated:
            headline = (
                f"{headline} - but the prose is SIMULATED OUTPUT from the mock "
                "backend. The citations resolve; the words between them are not a "
                "model's reasoning."
            )
        if retrieval.coverage.warning:
            headline = f"{headline}. {retrieval.coverage.warning}"
        if retrieval.withheld:
            headline = (
                f"{headline}. {retrieval.withheld} passage(s) withheld - "
                f"{retrieval.withheld_reason}"
            )

        return finish(
            GroundedAnswer(
                question=question,
                ok=True,
                headline=headline,
                retrieval=retrieval,
                answer=text,
                grounding=grounding,
                model=outcome.result.model_id if outcome.result else None,
                decision=outcome.decision.explain(),
                label=label,
                sources=sources,
                simulated=simulated,
            )
        )


_answerer: Answerer | None = None


def get_answerer() -> Answerer:
    global _answerer
    if _answerer is None:
        _answerer = Answerer()
    return _answerer


def reset_answerer() -> None:
    """Tests only."""
    global _answerer
    _answerer = None

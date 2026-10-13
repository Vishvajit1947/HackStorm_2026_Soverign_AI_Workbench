"""`kb.search` - the knowledge plane as a tool (§07).

The agent reaches the corpus through the same chokepoint as every other tool, so
a retrieval is policy-checked, jailed and audited like a file write. That matters
more here than anywhere else: the corpus is the only place in Kavach where
CONFIDENTIAL text enters a model's context, and the clearance filter has to be
applied by something the planner cannot route around.

The handler asks for `label` and `role` in its signature; the executor injects
them from the invocation. Nothing here reads a global to find out who is asking.
"""

from __future__ import annotations

from typing import Any

from ...knowledge import get_retriever
from ...types import Label

MAX_TOP_K = 20
DEFAULT_TOP_K = 6


def kb_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    label: Label = Label.INTERNAL,
    role: str = "engineer",
    **_: Any,
) -> dict[str, Any]:
    result = get_retriever().search(
        query,
        top_k=max(1, min(int(top_k), MAX_TOP_K)),
        role=role,
    )

    return {
        # `passages` rather than `stdout`: the agent's observer treats stdout as
        # console output, and a passage that lands in a log line loses the
        # citation that makes it usable.
        "passages": [h.as_dict() for h in result.hits],
        "citations": result.citations,
        "count": len(result.hits),
        # The run inherits this - a question that pulls in one CONFIDENTIAL
        # clause makes the whole run CONFIDENTIAL (§02.5).
        "effective_label": result.effective_label.value,
        "withheld": result.withheld,
        "withheld_reason": result.withheld_reason,
        "degraded": result.degraded,
        "detail": result.detail,
        "arms": result.arms,
        "coverage": result.coverage.as_dict(),
        "stderr": result.coverage.warning or "",
    }

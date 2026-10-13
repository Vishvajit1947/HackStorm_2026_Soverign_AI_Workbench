"""Let a step use what an earlier step found.

The plan is authored in one shot, before anything has run. So `doc.write` has to
name its `body` at plan time - which means an approval note could only ever
contain what the model already knew when it planned, never the passages the
search was about to retrieve. That is the whole reason a generated Word file came
out one line long: the deliverable was written before the data existed.

A step's arguments may therefore reference an earlier step:

    {{step0}}          everything step 0 produced, as text
    {{step1.text}}     the same thing, said explicitly

Substitution happens in the loop just before the tool is invoked, when the
observation actually exists. Deliberately narrow:

  ONLY EARLIER STEPS. A forward reference is left as-is rather than guessed at,
  and the tool receives the literal braces - visible in the deliverable, which is
  how you find out it happened.

  ONLY STRINGS. No expressions, no attribute walks, no indexing. This is a
  lookup, not a language: a template language in a tool argument is an
  injection surface, and the argument came from a model.
"""

from __future__ import annotations

import re
from typing import Any

# {{step3}} or {{step3.text}} - nothing else is a reference.
_REFERENCE = re.compile(r"\{\{\s*step(\d+)(?:\.text)?\s*\}\}", re.IGNORECASE)

# A passage list rendered in full would blow a small model's context and bury
# the finding in boilerplate. Enough to write a note from, not the whole corpus.
MAX_PASSAGES = 6
MAX_CHARS = 4000


def render_observation(observation: Any) -> str:
    """The readable form of what a step produced."""
    if observation is None:
        return ""

    stdout = (getattr(observation, "stdout", "") or "").strip()
    data = getattr(observation, "data", None) or {}

    # kb.search is the case this exists for: its value is in the passages, and
    # they carry the citation a grounded deliverable has to quote.
    passages = data.get("passages")
    if isinstance(passages, list) and passages:
        lines = []
        for hit in passages[:MAX_PASSAGES]:
            if not isinstance(hit, dict):
                continue
            citation = hit.get("citation") or hit.get("doc_id") or ""
            text = (hit.get("text") or hit.get("excerpt") or "").strip()
            lines.append(f"{text}\n{citation}".strip())
        rendered = "\n\n".join(lines)
        return rendered[:MAX_CHARS]

    if stdout:
        return stdout[:MAX_CHARS]

    entries = data.get("entries")
    if isinstance(entries, list):
        return ", ".join(str(e) for e in entries[:50])[:MAX_CHARS]

    result = data.get("result")
    if result is not None:
        return str(result)[:MAX_CHARS]
    return ""


def resolve_references(
    args: dict[str, Any], observations: dict[int, Any], current_idx: int
) -> tuple[dict[str, Any], list[str]]:
    """Replace {{stepN}} in string arguments. Returns (args, notes)."""
    notes: list[str] = []

    def _substitute(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx >= current_idx:
            # Forward reference: leave it alone. Substituting "" would hand the
            # tool a confidently empty document and no way to tell why.
            notes.append(f"step{idx} has not run yet - reference left in place")
            return match.group(0)
        rendered = render_observation(observations.get(idx))
        if not rendered:
            notes.append(f"step{idx} produced nothing to quote")
            return match.group(0)
        notes.append(f"step{idx} -> {len(rendered)} chars")
        return rendered

    resolved = {
        key: (_REFERENCE.sub(_substitute, value) if isinstance(value, str) else value)
        for key, value in args.items()
    }
    return resolved, notes


def has_reference(args: dict[str, Any]) -> bool:
    return any(
        isinstance(v, str) and _REFERENCE.search(v) for v in args.values()
    )

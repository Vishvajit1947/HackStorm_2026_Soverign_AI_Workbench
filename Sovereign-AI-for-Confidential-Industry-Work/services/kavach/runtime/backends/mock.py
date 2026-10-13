"""Mock backend - deterministic, no weights, no GPU.

Exists so the router (Phase 3) and the agent loop (Phase 5) can be built, tested
and rehearsed before a single GGUF is staged, and so CI can run the whole stack.

Two properties make it useful rather than merely present:

  Deterministic. The same prompt against the same model always produces the same
  output, seeded from sha256(model_id + prompt). Tests can assert on real strings
  and the router's behaviour is reproducible run to run.

  Honest about timing. It sleeps for the manifest's `load_ms` scaled by
  MOCK_TIME_SCALE, and paces generation at the manifest's `tok_per_s`. So an
  eviction genuinely costs time, the warden's swap penalty is exercised, and the
  run timeline in the UI does not look instantaneous and fake.

It also honours GBNF grammars well enough to emit schema-valid JSON, because a
mock that always produced perfect tool calls would hide the exact failure mode
Phase 5 has to defend against.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from typing import Any

from ...types import GenRequest, GenResult
from ..registry import ModelSpec
from .base import Backend, ModelNotLoaded


def _default_time_scale() -> float:
    """Scale wall-clock costs so the test suite is not dominated by fake latency.

    0.01 (default) keeps CI fast.  1.0 replays the manifest's real load_ms and
    tok_per_s - use it when rehearsing the UI or the swap demo, so a 4-second
    cold load actually feels like four seconds:

        KAVACH_MOCK_TIME_SCALE=1.0 kavach models swap A B
    """
    try:
        return max(0.0, float(os.environ.get("KAVACH_MOCK_TIME_SCALE", "0.01")))
    except ValueError:
        return 0.01


MOCK_TIME_SCALE = _default_time_scale()

# A citation as the knowledge plane renders it (§07), so the mock can copy one
# out of the passages it was handed instead of inventing the format.
_CITATION_IN_CONTEXT = re.compile(
    r"\[[A-Za-z0-9][A-Za-z0-9._-]*\s+p\.\s*\d+(?:\s+s\.[\d.]+)?\]"
)

# Rough words-per-token so token accounting is in the right ballpark.
_TOKENS_PER_WORD = 1.3


def _seed_for(model_id: str, prompt: str) -> int:
    digest = hashlib.sha256(f"{model_id}\x00{prompt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class MockBackend(Backend):
    name = "mock"

    def __init__(self, time_scale: float | None = None):
        self.time_scale = _default_time_scale() if time_scale is None else time_scale
        self._loaded: dict[str, float] = {}
        self._lock = threading.RLock()
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []

    # ---------------------------------------------------------------- lifecycle

    def load(self, spec: ModelSpec) -> None:
        with self._lock:
            if spec.id in self._loaded:
                return
            time.sleep((spec.load_ms / 1000.0) * self.time_scale)
            self._loaded[spec.id] = time.monotonic()
            self.load_calls.append(spec.id)

    def unload(self, spec: ModelSpec) -> None:
        with self._lock:
            if self._loaded.pop(spec.id, None) is not None:
                self.unload_calls.append(spec.id)

    def shutdown(self) -> None:
        with self._lock:
            self._loaded.clear()

    def loaded_models(self) -> list[str]:
        with self._lock:
            return sorted(self._loaded)

    # ---------------------------------------------------------------- inference

    def generate(self, spec: ModelSpec, req: GenRequest) -> GenResult:
        with self._lock:
            if spec.id not in self._loaded:
                raise ModelNotLoaded(
                    f"{spec.id} is not loaded - the warden's residency table and the "
                    "backend have diverged"
                )

        started = time.perf_counter()
        rng = random.Random(_seed_for(spec.id, req.prompt))

        if self._is_planning_request(req):
            text = self._plan_response(req, rng)
        elif self._is_grounded_request(req):
            text = self._grounded_response(spec, req, rng)
        elif req.grammar:
            text = self._grammar_response(spec, req, rng)
        else:
            text = self._prose_response(spec, req, rng)

        completion_tokens = max(1, int(len(text.split()) * _TOKENS_PER_WORD))
        gen_s = (completion_tokens / max(spec.tok_per_s, 0.1)) * self.time_scale
        time.sleep(gen_s)

        return GenResult(
            text=text,
            model_id=spec.id,
            prompt_tokens=max(1, int(len(req.prompt.split()) * _TOKENS_PER_WORD)),
            completion_tokens=completion_tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            finish_reason="stop",
        )

    @staticmethod
    def _is_planning_request(req: GenRequest) -> bool:
        return bool(req.system and "You are the planner" in req.system)

    def _plan_response(self, req: GenRequest, rng: random.Random) -> str:
        """Emit a structurally valid plan so the agent loop can be exercised
        offline (Phase 5).

        The tool names are read out of the system prompt rather than imported, so
        the mock stays a backend and does not reach into the tool plane. The plan
        is real in shape and simulated in substance - which is exactly what the
        rest of the mock does, and the run's step output carries the usual
        [SIMULATED OUTPUT] markers.
        """
        available = self._tools_from_system(req.system or "")
        goal = req.prompt.strip().splitlines()[0][:120] if req.prompt.strip() else "the goal"

        # The work mode is read out of the planner's own prompt, for the same
        # reason the tool names are: the mock stays a backend and never reaches
        # into the agent layer.
        #
        # This matters more than it looks. Without it the canned plan is the
        # same three steps whatever the operator toggled, so on a box with no
        # weights staged - which is every box before R0 - the Code tab is empty
        # for every run and the toggle appears to do nothing. Honouring the mode
        # here is not simulating a capability: the plan is real in shape and
        # simulated in substance exactly as before, and every step it produces
        # is still stamped [SIMULATED ...]. What it stops being is a mock that
        # contradicts the instruction it was given.
        mode = self._mode_from_system(req.system or "")
        if mode == "code" and {"fs.write", "code.exec"} <= set(available):
            return json.dumps(self._code_plan(goal), indent=2)
        if mode == "document" and "doc.write" in available:
            return json.dumps(self._document_plan(goal), indent=2)

        prefer = [t for t in ("calc", "fs.write", "fs.read", "fs.list") if t in available]
        if not prefer:
            prefer = available[:2] or ["calc"]

        steps: list[dict[str, Any]] = []

        for tool in prefer[:3]:
            if tool == "calc":
                steps.append({
                    "intent": f"compute the quantity needed for: {goal}",
                    "tool": "calc",
                    "args": json.dumps({"expression": "2 * 3"}),
                    "acceptance": "the step produces output",
                })
            elif tool == "fs.write":
                steps.append({
                    "intent": "write the working to a file",
                    "tool": "fs.write",
                    "args": json.dumps({
                        "path": "result.txt",
                        "content": f"[SIMULATED PLAN OUTPUT] {goal}",
                    }),
                    "acceptance": "file result.txt exists",
                })
            elif tool == "fs.read":
                steps.append({
                    "intent": "read the written file back to confirm it landed",
                    "tool": "fs.read",
                    "args": json.dumps({"path": "result.txt"}),
                    "acceptance": "the step produces output",
                })
            elif tool == "fs.list":
                steps.append({
                    "intent": "list the workspace",
                    "tool": "fs.list",
                    "args": "{}",
                    "acceptance": "the step produces output",
                })

        return json.dumps(steps, indent=2)

    @staticmethod
    def _mode_from_system(system: str) -> str:
        """Which work mode the planner was told to plan for, or "" for none.

        Matched on the directive's own heading rather than a flag threaded
        through GenRequest: the backend interface carries a prompt, and adding a
        field for the mock's benefit would put agent-layer vocabulary into every
        backend including the real one.
        """
        if "THIS RUN IS IN CODE MODE" in system:
            return "code"
        if "THIS RUN IS IN DOCUMENT MODE" in system:
            return "document"
        return ""

    @staticmethod
    def _code_plan(goal: str) -> list[dict[str, Any]]:
        """A plan that ends in a program and a sandboxed run of it.

        The code.exec step's acceptance test is what the snippet PRINTED, never
        a file: the sandbox is a tmpfs destroyed with the container, so a step
        judged by a file it wrote there fails however good the code is. The mock
        has to model that correctly or it teaches the agent loop a shape the
        real tool plane rejects.
        """
        return [
            {
                "intent": f"write the program for: {goal}",
                "tool": "fs.write",
                "args": json.dumps({
                    "path": "solution.py",
                    "content": (
                        '"""[SIMULATED PLAN OUTPUT] ' + goal + '\n\n'
                        "Generated by the mock backend. No model produced this; it is\n"
                        "here so the agent loop and the Code tab can be exercised with\n"
                        'no weights staged."""\n\n'
                        "def solve(flow_m3h: float = 120.0, diameter_m: float = 0.2) -> float:\n"
                        '    """[SIMULATED] stand-in for the real calculation."""\n'
                        "    velocity = flow_m3h / 3600.0 / (3.14159 * (diameter_m / 2) ** 2)\n"
                        "    return round(velocity, 3)\n\n\n"
                        'if __name__ == "__main__":\n'
                        '    print("[SIMULATED OUTPUT] velocity m/s:", solve())\n'
                    ),
                }),
                "acceptance": "file solution.py exists",
            },
            {
                "intent": "run the program in the sandbox and check what it prints",
                "tool": "code.exec",
                "args": json.dumps({
                    "code": (
                        'print("[SIMULATED OUTPUT] running solution.py")\n'
                        "flow_m3h, diameter_m = 120.0, 0.2\n"
                        "velocity = flow_m3h / 3600.0 / (3.14159 * (diameter_m / 2) ** 2)\n"
                        'print("velocity m/s:", round(velocity, 3))\n'
                    ),
                    "files": ["solution.py"],
                }),
                "acceptance": "the step produces output",
            },
        ]

    @staticmethod
    def _document_plan(goal: str) -> list[dict[str, Any]]:
        """A plan that ends in a file somebody can open and sign."""
        return [
            {
                "intent": f"draft the deliverable for: {goal}",
                "tool": "doc.write",
                "args": json.dumps({
                    "path": "approval_note.docx",
                    "title": "[SIMULATED] Approval note",
                    # Written in doc.write's own markup, so the mock exercises
                    # the TYPESETTER and not only the file writer. A mock that
                    # emits one block of prose leaves every heading, field,
                    # table and formula path unexercised until the day real
                    # weights arrive - which is the day there is least time to
                    # find out one of them is broken.
                    "body": (
                        f"[SIMULATED PLAN OUTPUT] {goal}\n\n"
                        "Equipment: P-101A (simulated)\n"
                        "Prepared by: Kavach mock backend\n"
                        "Classification: INTERNAL\n\n"
                        "# Background\n\n"
                        "This document was produced by the mock backend. No model "
                        "wrote it, and nothing in it is a finding about any real "
                        "equipment.\n\n"
                        "# Readings\n\n"
                        "| Tag | Thickness (mm) | Minimum (mm) |\n"
                        "| --- | --- | --- |\n"
                        "| 6-P-1203 | 8.89 | 6.40 |\n"
                        "| 6-P-1204 | 9.12 | 6.40 |\n\n"
                        "# Working\n\n"
                        "$$ margin_mm = 8.89 - 6.40 $$\n\n"
                        "# Recommendation\n\n"
                        "1. Stage real weights.\n"
                        "2. Re-run this task.\n"
                        "3. Do not circulate this file as a deliverable.\n\n"
                        "> WARNING: every number above is simulated."
                    ),
                }),
                "acceptance": "approval_note.docx is created in the workspace",
            },
        ]

    @staticmethod
    def _tools_from_system(system: str) -> list[str]:
        """Read the available tool names out of the planner's own prompt.

        Parsed rather than imported so the mock stays a backend and does not
        reach into the tool plane. That does couple it to the prompt's shape:
        this previously matched a comma-separated `Use ONLY these tools: a, b`
        and silently returned NOTHING when the prompt grew per-tool signatures,
        so the mock planned a single calc step and every agent test that
        expected an fs.write failed somewhere far from the cause.
        """
        return re.findall(r"^- ([a-zA-Z][\w.]*)\(", system, re.MULTILINE)

    @staticmethod
    def _is_grounded_request(req: GenRequest) -> bool:
        return bool(req.system and "cite every claim" in req.system)

    def _grounded_response(
        self, spec: ModelSpec, req: GenRequest, rng: random.Random
    ) -> str:
        """Emit a structurally valid cited answer so Phase 6's citation checker
        can be exercised offline.

        Same bargain as _plan_response: real in shape, simulated in substance,
        and stamped so it cannot be screenshotted as an answer. The citations are
        COPIED out of the passages in the prompt rather than constructed, which
        is exactly what a real model is instructed to do.

        And, as with _grammar_response, a low-tier model sometimes gets it wrong
        on purpose - it shifts a page number, producing a citation that is
        perfectly well-formed and does not resolve. That is the failure §07
        exists to catch, and a mock that never produced it would leave the whole
        citation-resolution path untested against the pipeline that feeds it.
        """
        found = _CITATION_IN_CONTEXT.findall(req.prompt)
        if not found:
            # Nothing to cite. Answering anyway is the behaviour the answerer
            # must reject, so the mock does exactly that.
            return self._prose_response(spec, req, rng)

        wrong_citation_chance = {0: 0.35, 1: 0.20}.get(spec.tier, 0.0)
        lines = [f"[SIMULATED OUTPUT - {spec.id}, no weights on this box]"]

        for raw in list(dict.fromkeys(found))[:3]:
            fragment = self._fragment_after(req.prompt, raw)
            citation = raw
            if rng.random() < wrong_citation_chance:
                # Well-formed, plausible, and pointing at a page that was never
                # retrieved.
                citation = re.sub(
                    r"p\.(\d+)",
                    lambda m: f"p.{int(m.group(1)) + 1}",
                    raw,
                    count=1,
                )
            lines.append(f"{fragment} {citation}")

        return "\n".join(lines)

    @staticmethod
    def _fragment_after(prompt: str, citation: str) -> str:
        """A short span of the passage that citation introduces, so the shape of
        the answer matches a real grounded one."""
        start = prompt.find(citation)
        if start == -1:
            return "The retrieved passage addresses the question."
        words = prompt[start + len(citation) : start + len(citation) + 400].split()
        return " ".join(words[:24]).strip() or "The retrieved passage is relevant."

    def _prose_response(self, spec: ModelSpec, req: GenRequest, rng: random.Random) -> str:
        """Plausible, deterministic, and clearly labelled as simulated. It must
        never be mistakable for a real model's answer in a screenshot."""
        head = req.prompt.strip().splitlines()[0][:120] if req.prompt.strip() else "(empty)"
        confidence = rng.choice(["high", "moderate", "qualified"])
        return (
            f"[SIMULATED OUTPUT - {spec.id}, no weights on this box]\n"
            f"Task received: {head}\n"
            f"Responding at {confidence} confidence with {spec.context} tokens of context.\n"
            f"This text is generated by the mock backend for pipeline testing. "
            f"Set KAVACH_BACKEND=llamacpp with staged weights for real inference."
        )

    def _grammar_response(
        self, spec: ModelSpec, req: GenRequest, rng: random.Random
    ) -> str:
        """Emit something matching the GBNF's shape.

        Deliberately imperfect on low-tier models: a tier-0/1 model has a small
        chance of returning prose instead of JSON, because Phase 5 must handle
        exactly that. A mock that never misbehaves teaches the wrong lesson.
        """
        malformed_chance = {0: 0.25, 1: 0.12}.get(spec.tier, 0.0)
        if rng.random() < malformed_chance:
            return (
                "Sure! I'll call the tool for you:\n"
                "tool: the one you asked for, with the arguments you wanted"
            )

        fields = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*ws?\s*":"', req.grammar or "")
        if not fields:
            fields = re.findall(r'"\\?"([a-zA-Z_][a-zA-Z0-9_]*)\\?""', req.grammar or "")

        payload: dict[str, Any] = {}
        for name in dict.fromkeys(fields):
            payload[name] = f"simulated_{name}_{rng.randrange(1000):03d}"
        if not payload:
            payload = {"result": "simulated", "model": spec.id}

        return json.dumps(payload)

    # ---------------------------------------------------------------- embedding

    def embed(self, spec: ModelSpec, texts: list[str]) -> list[list[float]]:
        """Deterministic pseudo-embeddings with a real property: similar strings
        get similar vectors, because dimensions are seeded from character n-grams.
        Good enough for the router's kNN head to be genuinely exercised offline."""
        dim = spec.dim or 384
        out: list[list[float]] = []

        for text in texts:
            vec = [0.0] * dim
            lowered = text.lower()
            tokens = re.findall(r"[a-z0-9]+", lowered)
            for token in tokens:
                for n in (3, 4):
                    for i in range(max(1, len(token) - n + 1)):
                        gram = token[i : i + n]
                        h = int.from_bytes(
                            hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big"
                        )
                        vec[h % dim] += 1.0
                        vec[(h >> 17) % dim] -= 0.5

            norm = sum(v * v for v in vec) ** 0.5
            out.append([v / norm for v in vec] if norm > 0 else vec)

        return out

    # ---------------------------------------------------------------- health

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "ok": True,
            "simulated": True,
            "loaded": self.loaded_models(),
            "time_scale": self.time_scale,
            "warning": (
                "MOCK BACKEND - every generation is synthetic. Nothing here is a "
                "real model output."
            ),
        }

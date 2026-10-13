"""The critic (§05).

Each planned step carries an acceptance test, and the critic decides whether the
observation actually satisfies it. Failing triggers a bounded replan rather than a
confident wrong answer - which is the whole difference between an agent and a
chat loop with tools bolted on.

Two layers, cheapest first:

  MECHANICAL   exit codes, file existence, non-empty output, required substrings.
               Deterministic, instant, and covers most real failures. A step that
               was supposed to write a file either wrote it or did not.

  SEMANTIC     ask a model whether the observation meets the acceptance test.
               Only reached when the mechanical layer has nothing to say, because
               it costs a generation and a model's opinion of its own work is the
               least reliable evidence available.

The ordering is deliberate. A critic that asks the model first will accept
"I have successfully created the file" as evidence a file exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..tools.workspace import Workspace
from ..types import Observation, Step

# Acceptance phrasings the mechanical layer understands. Anything else falls
# through to the semantic layer.
_FILE_PATTERNS = [
    re.compile(r"file\s+`?([\w./\\-]+\.\w+)`?\s+exists", re.IGNORECASE),
    re.compile(r"`?([\w./\\-]+\.\w+)`?\s+(?:is\s+)?(?:created|written|exists)", re.IGNORECASE),
    re.compile(r"(?:creates?|writes?|produces?)\s+`?([\w./\\-]+\.\w+)`?", re.IGNORECASE),
]
_EXIT_ZERO = re.compile(r"exit(?:s|ed)?\s*(?:code\s*)?(?:is\s*)?(?:0|zero)", re.IGNORECASE)
_NONEMPTY = re.compile(r"non-?empty|not\s+empty|produces\s+output", re.IGNORECASE)
_CONTAINS = re.compile(r"contains?\s+[\"'`]([^\"'`]+)[\"'`]", re.IGNORECASE)

# The line of a compiler or interpreter's output that actually says what is
# wrong. Taking the LAST line gave the replanner things like
#
#     the program exited 1, expected 0 (|     ^)
#
# - gcc's caret underline, which tells a model nothing and a person less. These
# are ordered by how specific they are, and the first hit wins.
_ERROR_LINE = (
    # Ahead of the generic `error:` rule on purpose: a link failure prints
    # "undefined reference to `main'" FIRST and "collect2: error: ld returned 1
    # exit status" LAST, and only the first of those says what to fix.
    re.compile(r"^.*\b(?P<msg>undefined reference to .+)$", re.MULTILINE),
    re.compile(r"^(?P<msg>\w*(?:Error|Exception)\b.*)$", re.MULTILINE),
    re.compile(r"^.*\berror:\s*(?P<msg>.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?P<msg>.*\bfailed\b.*)$", re.IGNORECASE | re.MULTILINE),
)


def diagnose(observation: Observation) -> str:
    """The one line worth repeating, plus whatever the tool worked out itself.

    This text is not decoration: `verdict.reason` is the ONLY thing the
    replanner is told about a failed step, so a hint the tool produced - "this
    snippet has no C entry point" - was visible in the panel and invisible to
    the model that had to fix it. The run then spent its whole replan budget
    making the same mistake three times.
    """
    stderr = (observation.stderr or "").strip()
    detail = ""
    for pattern in _ERROR_LINE:
        match = pattern.search(stderr)
        if match:
            detail = match.group("msg").strip()
            break
    if not detail:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        # Caret-underline lines are never the message; they point AT it.
        lines = [line for line in lines if set(line) - set("|^~ 0123456789")]
        detail = lines[-1] if lines else ""

    hint = (observation.data or {}).get("hint")
    if isinstance(hint, str) and hint.strip():
        detail = f"{detail} -- {hint.strip()}" if detail else hint.strip()
    return detail[:400]


@dataclass
class Verdict:
    ok: bool
    reason: str
    layer: str = "mechanical"        # mechanical | semantic | trivial
    checks: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "layer": self.layer,
            "checks": self.checks,
        }


class Critic:
    def __init__(self, router=None, semantic: bool = True):
        self._router = router
        self.semantic = semantic

    @property
    def router(self):
        if self._router is None:
            from ..router import get_router

            self._router = get_router()
        return self._router

    # ---------------------------------------------------------------- judge

    def judge(
        self,
        step: Step,
        observation: Observation,
        workspace: Workspace,
        *,
        run_id: str | None = None,
    ) -> Verdict:
        # A tool that failed outright needs no interpretation.
        if not observation.ok:
            return Verdict(
                ok=False,
                reason=diagnose(observation) or observation.stderr.strip()
                or "the tool call failed",
                layer="trivial",
                checks={"tool_succeeded": False},
            )


        mechanical = self._mechanical(step, observation, workspace)
        if mechanical is not None:
            return mechanical

        if not self.semantic:
            return Verdict(
                ok=True,
                reason="tool succeeded; no mechanical check applied",
                layer="trivial",
            )

        return self._semantic(step, observation, run_id=run_id)

    # ---------------------------------------------------------------- layer 1

    def _mechanical(
        self, step: Step, observation: Observation, workspace: Workspace
    ) -> Verdict | None:
        """Returns None when the acceptance test says nothing checkable."""
        acceptance = step.acceptance
        checks: dict[str, bool] = {}
        failures: list[str] = []

        # A PROGRAM THAT CRASHED DID NOT RUN, whatever the acceptance test says.
        #
        # `observation.ok` is the TOOL's verdict and it is correctly True for a
        # crashed program: the sandbox started, ran the snippet and reported
        # back. What it does not say is that the snippet WORKED. This check used
        # to fire only when the acceptance happened to mention an exit code, so
        # a step accepted as "the program runs and prints its result to stdout"
        # went to the semantic layer, a 1.7B read a Python traceback, and the
        # step passed:
        #
        #     code.exec  PASSED  exit 1
        #     NameError: name 'g' is not defined
        #
        # Two of those in one run, and the run reported `done`. Failing instead
        # is not only honest - it is what feeds the traceback back into the
        # replanner, which is the only way the model fixes its own code.
        #
        # Added to `checks` rather than returned early, so a step that failed
        # for two reasons still reports both: the first failure is not always
        # the one a reader needs.
        if observation.exit_code or _EXIT_ZERO.search(acceptance):
            ok = observation.exit_code == 0
            checks["exit_zero"] = ok
            if not ok:
                detail = diagnose(observation)
                failures.append(
                    f"the program exited {observation.exit_code}, expected 0"
                    + (f": {detail}" if detail else "")
                )

        # A DELIVERABLE THAT IS AN OUTLINE DID NOT DO THE JOB.
        #
        # doc.write reports when it was handed sections and no content - three
        # headings and two words under each, which is a .pptx that opens, has
        # the right title and says nothing. Treated as a warning it was written,
        # passed and counted as a finished task; as a failure the model is told
        # "this is an outline, not a deliverable" and gets to write the findings.
        #
        # The tool still WRITES the file either way: refusing would lose the
        # little content there is, and the reader can see for themselves.
        outline = next(
            (w for w in (observation.data or {}).get("warnings", [])
             if isinstance(w, str) and "it is an outline" in w),
            None,
        )
        if outline:
            checks["has_content"] = False
            failures.append(outline)

        # File existence - checked on disk, never taken from the model's word.
        file_text = ""
        for pattern in _FILE_PATTERNS:
            match = pattern.search(acceptance)
            if not match:
                continue
            path = match.group(1)
            try:
                resolved = workspace.resolve(path)
                exists = resolved.exists() and resolved.stat().st_size > 0
                if exists:
                    # Bounded: a substring check does not need a 200 MB read.
                    file_text = resolved.read_text(encoding="utf-8", errors="replace")[:200_000]
            except (OSError, ValueError):
                exists = False
            checks[f"file:{path}"] = exists
            if not exists:
                failures.append(f"{path} does not exist in the workspace, or is empty")
            break

        if _NONEMPTY.search(acceptance):
            body = (observation.stdout or "") + str(observation.data or "")
            ok = len(body.strip()) > 0
            checks["non_empty"] = ok
            if not ok:
                failures.append("the step produced no output")

        wanted = _CONTAINS.search(acceptance)
        if wanted:
            needle = wanted.group(1)
            # "report.txt exists and contains 'X'" is a claim about the FILE. An
            # fs.write step returns a write confirmation, not the content, so
            # checking only stdout failed every compute-then-save plan at its
            # last step - after the sandbox refusal and the plan-time repair had
            # both done their jobs. The file's text is part of the haystack
            # whenever the acceptance named one; stdout still counts on its own
            # for "output contains 'X'".
            haystack = f"{observation.stdout}\n{observation.data}\n{file_text}"
            ok = needle.lower() in haystack.lower()
            checks[f"contains:{needle}"] = ok
            if not ok:
                where = "the file or the output" if file_text else "the output"
                failures.append(f"{where} does not contain {needle!r}")

        if not checks:
            return None

        if failures:
            return Verdict(ok=False, reason="; ".join(failures), checks=checks)
        return Verdict(
            ok=True,
            reason=f"{len(checks)} mechanical check(s) passed",
            checks=checks,
        )

    # ---------------------------------------------------------------- layer 2

    def _semantic(
        self, step: Step, observation: Observation, *, run_id: str | None = None
    ) -> Verdict:
        from ..types import RouteRequest, TaskClass

        evidence = (observation.stdout or "").strip()[:2000]
        if not evidence and observation.data:
            evidence = str(observation.data)[:2000]

        # The judge is told whether the TOOL succeeded, because it cannot tell
        # from the payload alone - and without it an empty-but-correct result
        # reads as a failure. fs.list on a fresh workspace returns zero entries
        # with exit code 0: the step did exactly its job, and the emptiness is a
        # fact about the workspace. Failing it there burns a replan on a step
        # that was never wrong, and the run dies two replans later having done
        # nothing incorrect.
        prompt = (
            f"Step: {step.intent}\n"
            f"Tool: {step.tool}\n"
            f"The tool itself reported "
            f"{'SUCCESS' if observation.ok else 'FAILURE'} "
            f"(exit code {observation.exit_code}).\n"
            f"Acceptance test: {step.acceptance}\n"
            f"What the step produced:\n{evidence or '(nothing)'}\n\n"
            "Judge whether the step DID ITS JOB, not whether the result is "
            "interesting. An empty result from a read-only query is a fact about "
            "the workspace or the corpus, not a fault in the step: listing an "
            "empty directory correctly returns nothing, and a search that matches "
            "nothing has still searched. If the tool reported success and the "
            "acceptance only asks that it ran and returned its result, that is a "
            "PASS. Answer FAIL when the tool errored, when the result contradicts "
            "the acceptance, or when a specific value the acceptance names is "
            "absent.\n\n"
            "Answer PASS or FAIL, then one short sentence of justification."
        )

        try:
            outcome = self.router.route(
                RouteRequest(prompt=prompt, forced_class=TaskClass.DOC_SUMMARY),
                max_tokens=120,
                run_id=run_id,
                # WHICH step this judgement was about. Without it every routing
                # decision in a run recorded step_idx NULL, and the "which model
                # did what" panel labelled all of them "plan" - three rows
                # claiming to be the plan, when only the first one was.
                step_idx=step.idx,
            )
        except Exception as exc:  # noqa: BLE001
            # A critic that cannot run must not silently pass the step. Failing
            # closed here means an unjudgeable step gets a human's attention.
            return Verdict(
                ok=False,
                reason=f"the critic could not evaluate this step: {exc}",
                layer="semantic",
            )

        text = outcome.text.strip()
        passed = bool(re.search(r"\bPASS\b", text[:200], re.IGNORECASE)) and not re.search(
            r"\bFAIL\b", text[:200], re.IGNORECASE
        )
        return Verdict(
            ok=passed,
            reason=text[:300] or "no judgement returned",
            layer="semantic",
        )


_critic: Critic | None = None


def get_critic() -> Critic:
    global _critic
    if _critic is None:
        _critic = Critic()
    return _critic


def reset_critic() -> None:
    """Tests only."""
    global _critic
    _critic = None

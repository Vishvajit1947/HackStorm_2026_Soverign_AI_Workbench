"""The planner (§05).

Turns a goal into an ordered list of steps, each carrying an ACCEPTANCE TEST -
the thing the critic will check to decide whether the step actually worked.

The acceptance test is the part that matters. A plan without one is a list of
intentions, and a loop over intentions cannot tell success from a confident
failure. "Write the calculation" is not checkable; "the file exists and its tests
pass" is.

Two defences against the model, both necessary:

  GRAMMAR      the plan is generated under a GBNF constraint, so it is
               structurally valid JSON by construction (§04 tool plane).

  VALIDATION   structural validity is not enough. A model will happily emit a
               syntactically perfect call to a tool that does not exist, or with
               arguments that do not match its schema. Every step is checked
               against the real tool registry before the plan is accepted, and a
               failure is fed back for one retry with the specific error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audit import get_audit
from ..tools import ToolNotFound, get_tool_registry
from ..tools.grammar import plan_grammar
from ..tools.registry import InvalidArguments
from ..types import Label, RouteRequest, Step, StepStatus, WorkMode

MAX_STEPS = 12

# Raised when the model returns [] - it understood the goal and had nothing
# to do. Distinct from a parse failure, and reported differently.
EMPTY_PLAN = "the plan is empty"

CODE_EXEC = "code.exec"
DOC_WRITE = "doc.write"
# The three formats deliverables.py can actually write. Kept here as well as in
# the tool so the plan is refused BEFORE a run spends minutes of model time on
# the steps leading up to a deliverable the box cannot produce. `SUPPORTED` in
# deliverables.py is the authority; a test pins that the two agree.
DOC_FORMATS = {".docx", ".xlsx", ".pptx"}
_FILE_TOKEN = re.compile(
    r"\b[\w.-]+\.(?:txt|csv|json|md|py|log|dat|out|yaml|yml|docx|xlsx|pptx)\b|\bin the workspace\b",
    re.IGNORECASE,
)


def _expects_a_file(acceptance: str) -> bool:
    return bool(_FILE_TOKEN.search(acceptance or ""))


_OFFICE_FILE = re.compile(r"\.(?:docx|xlsx|pptx)\b", re.IGNORECASE)


def _wants_office_file(acceptance: str) -> bool:
    """Does this acceptance name a Word, Excel or PowerPoint file?

    Those cannot be produced by fs.write, which writes UTF-8 text - a .pptx
    written that way is a text file wearing the extension, and every reader
    refuses it.
    """
    return bool(_OFFICE_FILE.search(acceptance or ""))


# A READ step whose acceptance names a file has exactly ONE correct repair:
# judge it by what the tool returned. There is no second reading of the intent,
# so bouncing it back to the model buys a round trip to be told the same thing -
# and at 1.7B it writes the same acceptance again, the retry budget runs out,
# and a plan that was right about every tool, every argument and every ordering
# dies over one sentence the system could have written itself.
#
# This is a repair, not a cover-up: every change is recorded on the PlanResult,
# emitted to the audit chain as `plan.repaired`, and returned by the API, so a
# repaired plan is never shown as the plan the model wrote. Defects with more
# than one possible fix (an invented tool, bad arguments, a code.exec step that
# needs splitting in two) still go back to the model, because guessing there
# would be inventing intent.
_READ_ACCEPTANCE = {
    "kb.search": "the search returns relevant passages from the corpus",
    "fs.read": "the file's contents are returned",
    "fs.list": "the workspace listing is returned",
}
_READ_ACCEPTANCE_FALLBACK = "the tool returns the data this step asked for"
_CODE_EXEC_ACCEPTANCE = "the program runs and prints its result to stdout"

# A BUILD COMMAND, typed into `code` where a program belongs. Observed:
#
#     "code": "gcc -o pressure_drop pressure_drop.c"
#     /sandbox/main.c:1:5: error: expected '=', ',', ';' before '-' token
#
# The model had already written the file correctly and reached for the compiler,
# because until `files` could run a program that was the only place to put it.
# One line, nothing but a toolchain invocation - a program has more than that.
# `\b` cannot end this alternation: there is no word boundary between the `+`
# of `g++` and the space after it, so `g++ main.cpp` slipped through the first
# version of this rule. The command is followed by whitespace or by nothing.
_BUILD_COMMAND = re.compile(
    r"\A\s*(?:gcc|g\+\+|cc|clang|javac|java|rustc|cargo|go\s+(?:run|build)|make|"
    r"node|python3?|ruby|sh|bash|cat|\./\S+)(?:\s[^\n]*)?\Z",
    re.IGNORECASE,
)

# `{{step0}}`, the reference the loop resolves to an earlier step's output.
_STEP_REFERENCE = re.compile(r"\{\{\s*step\d+\s*\}\}")

# A source file named in an intent: "write orifice.py", "run pressure_drop.c".
# Only the extensions the sandbox can actually run - naming report.docx here
# would turn a document step into a source file.
_SOURCE_FILE = re.compile(
    r"\b[\w-]+\.(?:py|c|cpp|cc|cxx|java|js|mjs|sh|bash|go|rs|rb)\b", re.IGNORECASE
)

# An extension doc.write cannot write -> the container that holds the same
# content. This is a mapping, not a guess: a PDF and a .docx are both "a
# document somebody reads", a .csv and a .xlsx are both "a grid of values", and
# only the container changes. The CONTENT the model wrote is kept exactly.
#
# Nothing text-shaped is listed. `.txt`, `.md` and `.json` need fs.write, and
# changing the TOOL would be inventing intent - so those still go back to the
# model with the refusal that names fs.write.
_DOC_SUBSTITUTE = {
    ".pdf": ".docx", ".doc": ".docx", ".odt": ".docx", ".rtf": ".docx",
    ".xls": ".xlsx", ".ods": ".xlsx", ".csv": ".xlsx",
    ".ppt": ".pptx", ".odp": ".pptx",
}


def repair_plan(raw_steps: list[Any]) -> tuple[list[Any], list[str]]:
    """Fix the plan defects that have a single correct repair, before validation.

    Returns the steps and a note for each change made. An empty note list means
    the plan is exactly as the model wrote it.
    """
    registry = get_tool_registry()
    repaired: list[Any] = []
    notes: list[str] = []

    for idx, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            repaired.append(raw)             # validate_plan reports this properly
            continue

        step = dict(raw)
        tool_name = str(step.get("tool", "")).strip()
        acceptance = str(step.get("acceptance", "")).strip()

        if (
            acceptance
            and spec_side_effect(registry, tool_name) == "READ"
            and _expects_a_file(acceptance)
        ):
            step["acceptance"] = _READ_ACCEPTANCE.get(
                tool_name, _READ_ACCEPTANCE_FALLBACK
            )
            notes.append(
                f"step {idx} ({tool_name}) was judged by a file it cannot create "
                f"({acceptance!r}); rewritten to {step['acceptance']!r}"
            )

        # Same shape, same single correct repair. A code.exec step can NEVER be
        # judged by a file - the sandbox is a tmpfs destroyed with the container
        # - so the acceptance is wrong whatever the model intended, and saying
        # so is not guessing intent.
        #
        # Deliberately NOT the other half of the advice: splitting this into two
        # steps would invent a step nobody wrote, which is guessing. The
        # rewritten acceptance is recorded as `plan.repaired` and shown, so the
        # plan is never presented as the model's own.
        elif acceptance and tool_name == CODE_EXEC and _expects_a_file(acceptance):
            step["acceptance"] = _CODE_EXEC_ACCEPTANCE
            notes.append(
                f"step {idx} (code.exec) was judged by a file the sandbox discards "
                f"({acceptance!r}); rewritten to {step['acceptance']!r}. If that "
                "file is the deliverable, a later fs.write or doc.write step has "
                "to create it."
            )

        # THE BOX HAS NO PDF WRITER, and a 1.7B does not stop asking for one.
        # Observed: told in the prompt, refused at plan time with the three
        # formats named, the model produced .docx on the next attempt - and
        # asked for .pdf again on the one after that, exhausting the repair
        # budget on a run that had already written a perfectly good document.
        #
        # The repair keeps the model's content and changes the container, which
        # is the only part that was impossible. It is recorded as `plan.repaired`
        # and shown, so nobody is told they got a PDF.
        # `args` may still be a STRING of JSON here - small models emit that
        # constantly, and it is validate_plan that parses it. Reading it as a
        # dict unconditionally raised AttributeError on every such plan, which
        # surfaced as a 500 from the run endpoint rather than as a plan error.
        raw_args = step.get("args")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args, strict=False) if raw_args.strip() else {}
            except json.JSONDecodeError:
                raw_args = None            # validate_plan reports this properly
        # ONE mutable dict for the whole pass, assigned back once at the end.
        # Each repair used to do `args = dict(step_args)` and then
        # `step["args"] = args`, so the last block to run threw away whatever
        # the earlier ones had fixed: the build-command rewrite was undone by
        # the save_as block four lines later, and the step ran `gcc ...` as a
        # program exactly as before.
        step_args = dict(raw_args) if isinstance(raw_args, dict) else {}

        # A build command is not a program. When the step already names the
        # source file, the single correct repair is to run THAT - which is what
        # the command was trying to do - so `code` is emptied and `files` takes
        # over. Nothing is invented: the file, the tool and the intent are all
        # the model's own.
        if tool_name == CODE_EXEC:
            snippet = str(step_args.get("code", ""))
            if snippet.strip() and _BUILD_COMMAND.match(snippet.strip()):
                # The file to run, from anywhere the step already names it:
                # the command itself, the files it mounted, or its own intent.
                # `./pressure_drop` names the BINARY, not the source, so the
                # mounted list is the only place that case can come from.
                named = (
                    _SOURCE_FILE.search(snippet)
                    or _SOURCE_FILE.search(" ".join(
                        f for f in (step_args.get("files") or []) if isinstance(f, str)
                    ))
                    or _SOURCE_FILE.search(f"{step.get('intent', '')} {acceptance}")
                )
                if named:
                    step_args["code"] = ""
                    files = [
                        f for f in (step_args.get("files") or []) if isinstance(f, str)
                    ]
                    if named.group(0) not in files:
                        files.append(named.group(0))
                    step_args["files"] = files
                    notes.append(
                        f"step {idx} passed the build command {snippet.strip()!r} as the "
                        f"program. The sandbox compiles for you - rewritten to run "
                        f"{named.group(0)} from the workspace."
                    )


        path = str(step_args.get("path", ""))
        suffix = Path(path).suffix.lower()
        if tool_name == DOC_WRITE and suffix and suffix not in DOC_FORMATS:
            substitute = _DOC_SUBSTITUTE.get(suffix)
            if substitute:
                step_args["path"] = path[: -len(suffix)] + substitute
                if acceptance:
                    step["acceptance"] = acceptance.replace(path, step_args["path"])
                notes.append(
                    f"step {idx} asked doc.write for {path!r} and this box has no "
                    f"{suffix} writer; written as {step_args['path']!r} instead, with the "
                    "same content. A PDF can be exported from that file in one click."
                )

        # A step whose intent is "write orifice.py" and whose tool is code.exec
        # produces nothing: the snippet runs in a container that is destroyed,
        # and the NEXT step's `from orifice import ...` then fails with
        # ModuleNotFoundError. Observed exactly that, twice in one run, with the
        # deliverable list containing only a .docx.
        #
        # `save_as` is the single correct repair: it keeps the file the step
        # said it was writing, under the name the step itself gave. It invents
        # no step and changes no tool - the alternative reading, that the model
        # wanted a throwaway calculation, is contradicted by its own intent.
        if tool_name == CODE_EXEC and "save_as" not in step_args:
            named = _SOURCE_FILE.search(f"{step.get('intent', '')} {acceptance}")
            if named:
                step_args["save_as"] = named.group(0)
                notes.append(
                    f"step {idx} says it writes {named.group(0)} but code.exec runs in a "
                    f"container that is destroyed afterwards; save_as={named.group(0)!r} "
                    "added so the program is kept in the workspace"
                )

        # Written back once, after every repair has had its say. `args` may
        # have arrived as a STRING of JSON; normalising it here also spares
        # validate_plan the second parse.
        if step_args or isinstance(raw_args, dict):
            step["args"] = step_args

        repaired.append(step)

    return repaired, notes


def spec_side_effect(registry, tool_name: str) -> str:
    """The tool's declared side effect, or "" if the tool is unknown.

    Unknown is not an error here - the very next check reports the missing tool
    with the list of real ones, which is a far more useful message than this one
    would be.
    """
    try:
        return registry.get(tool_name).side_effect.value
    except Exception:                                # noqa: BLE001 - see above
        return ""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "acceptance": {"type": "string"},
    },
    "required": ["intent", "tool", "args", "acceptance"],
}

PLANNER_SYSTEM = """\
You are the planner for an offline engineering workbench. Break the goal into the
fewest steps that will actually achieve it.

Every step is an object with ALL FOUR of these keys, in this order:

  {{"intent": "why this step exists, in one line",
    "tool": "one of the tools listed below",
    "args": {{"an": "ordinary JSON object", "not": "a string"}},
    "acceptance": "what proves this step worked"}}

Use ONLY these tools, with exactly these argument names:
{tools}

Rules:
- `args` is a plain JSON object, NOT a string containing JSON. Write
  {{"path": "report.txt", "content": "..."}}, never
  "{{\"path\": \"report.txt\"}}". Small models lose the escaping and the
  whole plan is rejected over quoting rather than over reasoning.
- `intent` is required on every step. It is what a human reads when approving
  the plan, so write it for them, not for the machine.
- Every step needs an `acceptance` describing what proves it worked, in checkable
  terms: a file exists, an exit code is zero, a value is present. Never "looks
  correct".
- To produce a Word (.docx), Excel (.xlsx) or PowerPoint (.pptx) file, use
  doc.write. Do NOT write code to build one: code.exec runs in a sandbox that
  is destroyed afterwards, so every file it creates is thrown away. fs.write
  makes plain text only - a .pptx written with it is a text file wearing the
  wrong extension, and nothing can open it.
- THERE IS NO PDF WRITER ON THIS BOX. If the goal asks for a PDF, write the
  same content to a .docx instead and say so in the intent - a person can
  export a PDF from it in one click. A step that names a .pdf is refused.
- TO DELIVER A PROGRAM, KEEP THE FILE. Two ways, both fine:
    ONE STEP   {{"code": "...the whole program...", "save_as": "orifice.py"}}
               - the source is saved into the workspace and then run.
    TWO STEPS  fs.write orifice.py with the program, then code.exec with
               {{"files": ["orifice.py"]}} and `code` LEFT EMPTY - that file is
               compiled and run as it stands.
  Never type a build command ("gcc -o x x.c") into `code`; the sandbox compiles
  for you. Never write a program in one code.exec step and import it in the
  next - each call is a fresh container and nothing survives between them.
- {{{{stepN}}}} belongs in a DOCUMENT body, never in code.exec's `code`. It
  resolves to passages an earlier step retrieved, and a compiler cannot read
  them - write the program out in full instead.
- A step can quote what an EARLIER step found by writing {{{{step0}}}} (or
  {{{{step1}}}}, and so on) inside a string argument. Use this whenever a later
  step needs real data rather than a description of it. To turn retrieved
  passages into a document, search first and then write
  {{"path": "note.docx", "body": "Findings:

{{{{step0}}}}

Recommendation: ..."}}
  - the passages are substituted in when the step runs. Without a reference the
  document can only contain what you already knew before searching, which is
  usually one thin line.
- An acceptance test must be checkable from THAT STEP's own output, and nothing
  else. A kb.search step is proved by the passages it returned, not by a file a
  later step has not written yet - "a Word file is created" on a search step
  fails every time however well the search went, because the search was never
  going to create it. Ask yourself, for each step: could this be judged with
  only what this tool just returned?
- At most {max_steps} steps. Fewer is better.
- A goal may be vague, or may name something you cannot see ("the project", "the
  report", "our data"). Do not give up on it. Interpret it with the tools you
  have: search the corpus with kb.search, look at the workspace with fs.list and
  fs.read, and write what you produce with fs.write. A plan that gathers what is
  available and writes it down is a useful answer to a broad request.
- Return an empty array ONLY when the goal asks for nothing to be done at all -
  a greeting, or a remark with no request in it. If the goal asks for anything,
  however loosely worded, produce at least one step.

Worked example - "find what the inspection said about the pump and write it up
as a Word file called note.docx". Note where each acceptance test lives: the
search is proved by passages, the write is proved by the file.

[{{"intent": "find the inspection findings for the pump",
   "tool": "kb.search",
   "args": {{"query": "pump inspection findings"}},
   "acceptance": "the search returns relevant passages from the corpus"}},
 {{"intent": "write the findings up for sign-off",
   "tool": "doc.write",
   "args": {{"path": "note.docx",
           "body": "Findings:\\n\\n{{{{step0}}}}\\n\\nRecommendation: ..."}},
   "acceptance": "note.docx is created in the workspace"}}]

Respond with a JSON array of steps and nothing else.
"""


# Appended to the planner's system prompt when the operator has chosen a mode.
#
# These are not style hints. A small model asked to "write the pressure drop
# calculation" finishes with fs.write of a paragraph DESCRIBING the calculation,
# because that satisfies the goal as worded, and the Code tab then shows nothing
# while the run reports success. Naming the tool that can actually produce the
# deliverable, and the step that proves it, is the difference between a toggle
# that routes differently and a toggle that delivers differently.
MODE_DIRECTIVE = {
    WorkMode.CODE: """

THIS RUN IS IN CODE MODE. The deliverable is WORKING CODE, not a description of
code. Therefore:
- Write the program to the workspace with fs.write, to a path whose EXTENSION IS
  THE LANGUAGE THAT WAS ASKED FOR: .py, .c, .cpp, .java, .js, .sh. If the task
  names no language, use Python and .py. Never put C in a .py or a .txt - the
  file is read and run by extension, and a mismatch fails as a syntax error in
  a language nobody chose.
- Write it in that language's own syntax, laid out the way that language is
  normally written: `#include <stdio.h>` and `int main(void)` for C, a
  `public class` whose name MATCHES the .java file name for Java, `def` and
  four-space indentation for Python.
- Then RUN it with a code.exec step so the output is real rather than claimed:
  {{"files": ["the_file_you_just_wrote"]}} with `code` left EMPTY runs that file
  as it stands. Its acceptance test is what the program printed - never a file
  the program wrote, because the sandbox is wiped.
- Do not finish with a prose explanation instead of a program. A .txt file
  containing sentences about the algorithm is not the deliverable here.
""",
    WorkMode.DOCUMENT: """

THIS RUN IS IN DOCUMENT MODE. The deliverable is a DOCUMENT a person will open,
read and sign. Therefore:
- Finish with a doc.write step producing .docx (a note or report), .xlsx (a
  calculation sheet) or .pptx (a briefing). Never fs.write for these.
- Ground it: search the corpus with kb.search first and quote what came back
  into the document body with {{step0}}, so the document contains the real
  passages rather than a summary of what you assumed they say.
- EVERY SECTION YOU OPEN, YOU FILL. A heading with nothing under it is not a
  section - it is a gap, and the reader gets a page with a title and white
  space. If you have only three things to say, write three sections, not eight.
- WRITE THE FINDINGS, NOT THE SHAPE OF THEM. "Findings:" followed by a
  reference and "Recommendation: ..." is an outline. Name the equipment, the
  measurement, the limit and the date - "P-101A drive-end vibration 4.8 mm/s
  RMS against an ISO 10816-3 Zone B limit of 4.5 mm/s, 02-MAR-2026" is a
  finding; "vibration high" is two words in a box.
- Printing the text to stdout is not the deliverable. There has to be a file.
- STRUCTURE THE BODY. A wall of prose is not an approval note. Use:
    # Section heading          and  ## Sub-heading
    Equipment: P-101A          a labelled field, one fact per line
    - finding                  bullets
    1. action                  numbered steps, for anything to be done in order
    | Tag | Reading | Limit |  a table, one row per line
    $$ dP = 2.5 * 1.8 $$       a formula - it is typeset and evaluated for you
    > WARNING: do not restart  a callout for anything safety-critical
  Open with the fields (equipment, date, prepared by), then the findings, then
  the recommendation. That is the shape of the note somebody signs.
- For .xlsx: the FIRST LINE IS THE HEADER ROW. Name the columns as a person
  would - "Tag, Thickness (mm), Minimum (mm)" - and put one reading per line
  after it. A cell written as "=B2-C2" becomes a live Excel formula.
- Name the file after the deliverable in lower case with underscores:
  approval_note.docx, wall_thickness_survey.xlsx, hot_work_briefing.pptx.
""",
    WorkMode.AUTO: "",
}


def _tool_catalogue(registry) -> str:
    """Render each tool's real signature for the planner prompt.

    Listing bare tool NAMES is not enough and fails in a way that looks like
    model stupidity rather than a missing prompt: told only that `fs.list`
    exists, qwen2.5-7b called it with `dir`, because `dir` is what that
    argument is called nearly everywhere else in its training data. The real
    argument is `subdir`. No amount of retrying fixes a name the model was
    never told, so the schema goes in the prompt.

    Optional arguments are marked with `?` so the model does not treat every
    key as mandatory and pad the call with invented values.
    """
    lines = []
    for tool in registry.all():
        required = set(tool.required)
        args = ", ".join(
            f"{key}: {spec.get('type', 'string')}"
            if key in required
            else f"{key}?: {spec.get('type', 'string')}"
            for key, spec in tool.properties.items()
        )
        line = f"- {tool.name}({args})"
        if tool.description:
            line += f" - {tool.description}"
        lines.append(line)
    return "\n".join(lines)


class PlanError(RuntimeError):
    """The model did not produce a usable plan."""


@dataclass
class PlanResult:
    steps: list[Step]
    attempts: int
    raw: str
    model: str | None
    repaired: bool = False
    repairs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "model": self.model,
            "repaired": self.repaired,
            "repairs": self.repairs,
            "steps": [
                {
                    "idx": s.idx,
                    "intent": s.intent,
                    "tool": s.tool,
                    "args": s.args,
                    "acceptance": s.acceptance,
                    "requires_approval": s.requires_approval,
                }
                for s in self.steps
            ],
        }


def _extract_json_array(text: str) -> list[Any]:
    """Pull the first JSON array out of a model response.

    Even under a grammar, a model may wrap the array in prose or a fence. Being
    lenient HERE is safe because every field is validated immediately afterwards -
    leniency about shape, strictness about content.
    """
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise PlanError("no JSON array in the planner's response")

    try:
        # strict=False permits literal control characters inside strings. A model
        # writing multi-line Python into a "code" field emits real newline bytes
        # rather than escape sequences, and json.loads would reject the entire
        # plan over a byte that is unambiguous in context. This relaxes only that
        # rule; genuinely malformed structure still raises, which is what we do
        # want to hear about.
        parsed = json.loads(candidate[start : end + 1], strict=False)
    except json.JSONDecodeError as exc:
        raise PlanError(f"planner emitted invalid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise PlanError("planner did not return an array of steps")
    return parsed


def validate_plan(raw_steps: list[Any], label: Label = Label.INTERNAL) -> list[Step]:
    """Check every step against the real tool registry.

    This is where "the model invented a tool" is caught. It is a real failure
    mode that constrained decoding does NOT prevent - a grammar can force valid
    JSON containing a tool name that has never existed.
    """
    registry = get_tool_registry()
    steps: list[Step] = []

    if not raw_steps:
        raise PlanError(EMPTY_PLAN)
    if len(raw_steps) > MAX_STEPS:
        raise PlanError(
            f"the plan has {len(raw_steps)} steps; the budget allows {MAX_STEPS}"
        )

    for idx, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise PlanError(f"step {idx} is not an object")

        intent = str(raw.get("intent", "")).strip()
        tool_name = str(raw.get("tool", "")).strip()
        acceptance = str(raw.get("acceptance", "")).strip()

        if not intent:
            raise PlanError(f"step {idx} has no intent")
        if not acceptance:
            raise PlanError(
                f"step {idx} ({intent!r}) has no acceptance test - the critic would "
                "have nothing to check, so the step could never be known to work"
            )

        args = raw.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False) if args.strip() else {}
            except json.JSONDecodeError as exc:
                raise PlanError(f"step {idx} args are not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise PlanError(f"step {idx} args must be an object")

        # code.exec cannot produce a file the workspace can see: its working
        # directory is a tmpfs wiped when the container exits, and nothing is
        # copied out. A step that runs code AND expects a file to exist fails its
        # acceptance test every time, however good the code is - and models
        # write exactly that step, because the goal said "write it to a file".
        # Refusing it here turns a run-time failure that burns the replan budget
        # into a plan-time repair that costs nothing.
        # A READ tool cannot bring a file into existence, so an acceptance test
        # that waits for one can never pass however well the tool worked. Small
        # models write this constantly - they put the DELIVERABLE's test on the
        # first step, so `kb.search` is asked to prove a .docx exists, fails, and
        # burns a replan on a search that returned exactly what was wanted.
        #
        # Telling the model not to (in the prompt) does not hold at 1.5B, and the
        # side effect is declared in the manifest, so the check is derived rather
        # than hardcoded: any future READ tool is covered the day it is added.
        if spec_side_effect(registry, tool_name) == "READ" and _expects_a_file(acceptance):
            raise PlanError(
                f"step {idx} calls {tool_name}, which only reads - it cannot "
                f"create a file, so its acceptance test ({acceptance!r}) can "
                "never pass. Judge this step by what it returned (passages "
                "found, files listed), and put the file check on the step that "
                "actually writes the file."
            )

        # ...unless the file it names is the SOURCE this step saves. `save_as`
        # writes the program on the host before the container starts, so that
        # one file genuinely does appear in the workspace. Refusing it would
        # send the model to repair a plan that was already correct - and the
        # refusal would be wrong, which is worse than annoying.
        saves = str(args.get("save_as", "")).strip()
        judged_by_its_own_source = bool(saves) and Path(saves).name in acceptance

        if tool_name == CODE_EXEC and not judged_by_its_own_source and _expects_a_file(acceptance):
            # Name the tool that can ACTUALLY make this file. Telling a model to
            # use fs.write for a .pptx sends it round the same loop twice:
            # fs.write writes text, so the plan is refused again and the run dies
            # on advice that was never going to work. Word, Excel and PowerPoint
            # come from doc.write; everything else is text.
            writer = "doc.write" if _wants_office_file(acceptance) else "fs.write"
            raise PlanError(
                f"step {idx} runs code.exec but its acceptance test expects a file "
                f"({acceptance!r}). Files written inside the sandbox are discarded, "
                f"so this step can never pass. Split it: a code.exec step whose "
                f"acceptance test checks stdout, then a {writer} step that writes "
                f"the result to the file."
            )

        # A RETRIEVED PASSAGE IS NOT A PROGRAM. `{{step0}}` inside code.exec's
        # `code` is resolved to whatever the earlier step found, and a page of
        # an inspection report handed to a compiler produces:
        #
        #     /sandbox/main.c:1:1: error: unknown type name 'MECHANICAL'
        #     1 | MECHANICAL SEAL & ANCILLARY INSPECTION REPORT
        #
        # Observed three times in one run, burning every replan. The reference
        # mechanism is right for a DOCUMENT body, which is where it was designed
        # to be used; source code has to be written, not quoted.
        if tool_name == CODE_EXEC and _STEP_REFERENCE.search(str(args.get("code", ""))):
            raise PlanError(
                f"step {idx} puts a {{{{stepN}}}} reference in code.exec's `code`. "
                "That resolves to what an earlier step FOUND - passages from the "
                "corpus - and a compiler cannot read them. Write the program "
                "out in full here. To use a retrieved value, read it from the "
                "step's output and write it into the source as a literal."
            )

        # doc.write produces Word, Excel and PowerPoint - and nothing else. A
        # goal that says "make me a PDF" gets a plan whose last step writes
        # report.pdf, which the tool refuses at run time, after the search step
        # has already spent minutes of model time. Caught here it costs nothing
        # and the model is told which formats exist, which is the only way it
        # can produce a plan that works.
        if tool_name == DOC_WRITE:
            path = str(args.get("path", ""))
            suffix = Path(path).suffix.lower()
            if path and suffix not in DOC_FORMATS:
                raise PlanError(
                    f"step {idx} asks doc.write for {path!r}. It produces "
                    f"{', '.join(sorted(DOC_FORMATS))} and nothing else - there is no "
                    f"PDF writer on this box. Write the deliverable as .docx "
                    "(a note or report), .xlsx (a calculation sheet) or .pptx (a "
                    "briefing); anyone who needs a PDF can export one from that "
                    "file. For plain text use fs.write."
                )

        try:
            spec = registry.get(tool_name)
        except ToolNotFound as exc:
            raise PlanError(
                f"step {idx} calls {tool_name!r}, which does not exist. "
                f"Available: {', '.join(registry.names())}"
            ) from exc

        try:
            args = spec.validate(args)
        except InvalidArguments as exc:
            raise PlanError(f"step {idx}: {exc}") from exc

        steps.append(
            Step(
                idx=idx,
                intent=intent,
                tool=tool_name,
                args=args,
                acceptance=acceptance,
                status=StepStatus.PENDING,
                requires_approval=spec.requires_approval or spec.produces_artifact,
            )
        )

    return steps


class Planner:
    # Two repairs, not one. A 1.7B plan often has TWO independent defects - the
    # observed case asked doc.write for a .pdf AND judged a code.exec step by a
    # file - and each refusal teaches the model about one of them. With a single
    # repair the second defect is fatal even though the model had just proved it
    # responds correctly to the first. Each attempt costs a few seconds on the
    # small model; a dead run costs the whole task.
    def __init__(self, router=None, max_repairs: int = 2):
        self._router = router
        # One repair attempt. An unbounded repair loop on a 12 GB card means
        # watching a model fail to count while a judge waits.
        self.max_repairs = max_repairs

    @property
    def router(self):
        if self._router is None:
            from ..router import get_router

            self._router = get_router()
        return self._router

    def plan(
        self,
        goal: str,
        *,
        label: Label = Label.INTERNAL,
        run_id: str | None = None,
        actor: str = "local",
        mode: WorkMode = WorkMode.AUTO,
    ) -> PlanResult:
        registry = get_tool_registry()
        system = PLANNER_SYSTEM.format(
            tools=_tool_catalogue(registry), max_steps=MAX_STEPS
        ) + MODE_DIRECTIVE.get(mode, "")

        prompt = goal
        last_error: str | None = None
        raw = ""
        model: str | None = None

        for attempt in range(self.max_repairs + 1):
            if last_error:
                # Feed the specific failure back rather than just retrying: the
                # model cannot fix what it has not been told.
                prompt = (
                    f"{goal}\n\n"
                    f"Your previous plan was rejected: {last_error}\n"
                    f"Produce a corrected plan."
                )

            outcome = self.router.route(
                RouteRequest(
                    prompt=prompt,
                    needs_tools=True,
                    label=label,
                    # The mode toggle lands HERE. AUTO yields AGENTIC_MULTI,
                    # which is what this was hardcoded to before the toggle
                    # existed, so an untoggled run routes exactly as it did.
                    forced_class=mode.forced_class,
                ),
                system=system,
                max_tokens=1500,
                # The GRAMMAR defence named at the top of this module. It
                # was documented but never passed, so PLAN_SCHEMA was dead
                # code and the planner became the one place that asks a 7B
                # for structured JSON and merely hopes.
                grammar=plan_grammar(PLAN_SCHEMA),
                run_id=run_id,
            )
            raw = outcome.text
            model = outcome.result.model_id if outcome.result else None

            repairs: list[str] = []
            try:
                candidate, repairs = repair_plan(_extract_json_array(raw))
                steps = validate_plan(candidate, label)
            except PlanError as exc:
                last_error = str(exc)
                get_audit().append(
                    "plan.rejected",
                    actor=actor,
                    run_id=run_id,
                    attempt=attempt + 1,
                    reason=last_error,
                    model=model,
                )
                continue

            if repairs:
                # In-band before the plan is accepted, so the record of what the
                # model actually wrote survives the repair.
                get_audit().append(
                    "plan.repaired",
                    actor=actor,
                    run_id=run_id,
                    attempt=attempt + 1,
                    model=model,
                    repairs=repairs,
                )
            get_audit().append(
                "plan.accepted",
                actor=actor,
                run_id=run_id,
                steps=len(steps),
                attempt=attempt + 1,
                model=model,
                repaired=bool(repairs),
            )
            return PlanResult(
                steps=steps,
                attempts=attempt + 1,
                raw=raw,
                model=model,
                repaired=attempt > 0 or bool(repairs),
                repairs=repairs,
            )

        # An empty plan is not the same failure as a malformed one. The model
        # parsed the goal, understood it, and had nothing to do - which is the
        # right answer to "hello" and the wrong thing to report as a planner
        # malfunction. Reporting both identically sends the user hunting for a
        # broken planner when the real problem is a goal with no task in it.
        if last_error == EMPTY_PLAN:
            raise PlanError(
                "this goal does not describe a task the workbench can act on, "
                "so the planner returned no steps. Name something to do - a "
                "quantity to calculate, a document to search, a file to write."
            )

        raise PlanError(
            f"the planner could not produce a usable plan after "
            f"{self.max_repairs + 1} attempts. Last error: {last_error}"
        )

    def replan(
        self,
        goal: str,
        completed: list[Step],
        failure: str,
        *,
        label: Label = Label.INTERNAL,
        run_id: str | None = None,
        mode: WorkMode = WorkMode.AUTO,
    ) -> PlanResult:
        """Amend the REMAINING steps after a failure.

        Completed steps are not revisited - the history stays as it happened, and
        only the future is rewritten.
        """
        done = "\n".join(f"  - {s.intent} ({s.status.value})" for s in completed) or "  (none)"
        prompt = (
            f"Goal: {goal}\n\n"
            f"Steps already completed:\n{done}\n\n"
            f"The next step failed: {failure}\n\n"
            f"Produce the REMAINING steps needed to finish the goal, taking that "
            f"failure into account. Do not repeat completed work."
        )
        # The mode carries into the amended steps. A code run that replans is
        # still a code run; dropping the mode here is how the second half of a
        # run quietly becomes a different kind of work from the first.
        result = self.plan(prompt, label=label, run_id=run_id, mode=mode)

        # Re-index so the amended steps continue after the completed ones.
        offset = max((s.idx for s in completed), default=-1) + 1
        for position, step in enumerate(result.steps):
            step.idx = offset + position
        return result


_planner: Planner | None = None


def get_planner() -> Planner:
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner


def reset_planner() -> None:
    """Tests only."""
    global _planner
    _planner = None

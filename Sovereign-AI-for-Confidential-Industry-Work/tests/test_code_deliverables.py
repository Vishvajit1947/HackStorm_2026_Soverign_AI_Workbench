"""A crashed program is not a passed step, and a program is a deliverable.

Both properties come from one run that reported `done`:

    ✓ write orifice.py code    code.exec   exit 1   NameError: name 'g' is not defined
    ✓ run orifice.py           code.exec   exit 1   ModuleNotFoundError: No module named 'orifice'
    Files it produced: orifice_formula.docx

Two crashed programs marked PASSED, and the only deliverable was a Word file -
from a goal that said "write orifice.py ... then run it and print the bore".
"""

from __future__ import annotations

import pytest

from kavach.agent.critic import Critic
from kavach.agent.planner import PlanError, repair_plan, validate_plan
from kavach.tools.impl.languages import LANGUAGES
from kavach.tools.impl.sandbox import save_source
from kavach.tools.workspace import Workspace
from kavach.types import Observation, Step, StepStatus


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run")


@pytest.fixture()
def critic():
    # The semantic layer needs a model; every case here must be decided before
    # it, which is the property being tested.
    return Critic(semantic=False)


def _step(acceptance: str, tool: str = "code.exec") -> Step:
    return Step(idx=0, intent="do it", tool=tool, args={}, acceptance=acceptance,
                status=StepStatus.RUNNING)


# ------------------------------------------------------- a crash is a failure


def test_a_program_that_crashed_does_not_pass(critic, ws):
    """The bug. `observation.ok` is the TOOL's verdict and it is correctly True
    - the sandbox ran the snippet and reported back. What it does not say is
    that the snippet worked."""
    verdict = critic.judge(
        _step("the program runs and prints its result to stdout"),
        Observation(step_idx=0, ok=True, exit_code=1,
                    stderr="Traceback...\nNameError: name 'g' is not defined"),
        ws,
    )
    assert verdict.ok is False
    assert "exited 1" in verdict.reason


def test_the_failure_quotes_the_line_a_person_needs(critic, ws):
    """The last line of a traceback is the error. Reporting "exit 1" alone sends
    the reader back to the panel to find out what broke."""
    verdict = critic.judge(
        _step("the program runs"),
        Observation(step_idx=0, ok=True, exit_code=1,
                    stderr="Traceback (most recent call last):\n  File x\nNameError: name 'g' is not defined"),
        ws,
    )
    assert "NameError" in verdict.reason


def test_an_acceptance_that_never_mentions_an_exit_code_is_still_checked(critic, ws):
    """This is exactly how it got through: the check used to fire only when the
    acceptance text happened to say "exits 0", and the acceptance in that run
    said "prints its result to stdout"."""
    verdict = critic.judge(
        _step("stdout contains the bore diameter"),
        Observation(step_idx=0, ok=True, exit_code=1, stdout="the bore diameter is 12mm"),
        ws,
    )
    assert verdict.ok is False


def test_a_clean_program_still_passes(critic, ws):
    verdict = critic.judge(
        _step("the program runs and prints its result to stdout"),
        Observation(step_idx=0, ok=True, exit_code=0, stdout="bore = 12.4 mm"),
        ws,
    )
    assert verdict.ok is True


def test_a_step_that_failed_twice_reports_both_reasons(critic, ws):
    """The first failure is not always the one a reader needs, so the exit-code
    check joins the others rather than short-circuiting them."""
    verdict = critic.judge(
        _step("out.txt exists and the command exits 0"),
        Observation(step_idx=0, ok=True, exit_code=1),
        ws,
    )
    assert verdict.ok is False
    assert len(verdict.checks) >= 2


def test_a_tool_that_returns_no_exit_code_is_unaffected(critic, ws):
    """Observation.exit_code defaults to 0, and most tools never set it. A
    doc.write step must not acquire an exit-code check it cannot fail."""
    verdict = critic.judge(
        _step("note.docx is created in the workspace", tool="doc.write"),
        Observation(step_idx=0, ok=True),
        ws,
    )
    assert "exit_zero" not in verdict.checks


# ------------------------------------------------- the program is an artifact


def test_the_source_is_saved_under_the_name_that_was_asked_for(ws):
    result = save_source(ws, "print('hi')", LANGUAGES["python"], "orifice.py")
    assert result["path"] == "orifice.py"
    assert ws.resolve("orifice.py").read_text(encoding="utf-8") == "print('hi')"


def test_the_extension_is_corrected_to_the_language_that_will_run(ws):
    """A C program saved as `.py` is the original printf bug wearing a filename:
    the file on disk would not be what the sandbox compiled."""
    result = save_source(ws, "int main(void){return 0;}", LANGUAGES["c"], "solver.py")
    assert result["path"] == "solver.c"
    assert result["renamed"] is True


def test_the_fence_is_not_saved_with_the_source(ws):
    """```python is markdown. A file that starts with it does not run, and the
    person who opens it has to know to delete a line."""
    save_source(ws, "```python\nprint(1)\n```", LANGUAGES["python"], "x.py")
    assert ws.resolve("x.py").read_text(encoding="utf-8") == "print(1)"


def test_saving_cannot_escape_the_workspace(ws):
    from kavach.tools.workspace import WorkspaceViolation

    with pytest.raises((WorkspaceViolation, ValueError)):
        save_source(ws, "print(1)", LANGUAGES["python"], "../../escape.py")


# ------------------------------------------------------- the plan-time repair


def _plan_step(intent, tool, acceptance, args):
    return {"intent": intent, "tool": tool, "acceptance": acceptance, "args": args}


def test_a_step_that_says_it_writes_a_program_keeps_the_program():
    """`write orifice.py` with code.exec produced nothing at all: the container
    is destroyed, so the next step's `from orifice import ...` failed with
    ModuleNotFoundError. save_as is the single correct repair - it keeps the
    file the step said it was writing, under the name the step itself gave."""
    steps, notes = repair_plan([
        _plan_step("write orifice.py code", "code.exec",
                   "the program runs and prints its result to stdout",
                   {"code": "print(1)"})])
    assert steps[0]["args"]["save_as"] == "orifice.py"
    assert len(notes) == 1 and "destroyed" in notes[0]
    validate_plan(steps)


def test_the_repair_does_not_fire_when_no_file_was_named():
    """A throwaway calculation is a legitimate use of code.exec and must not
    grow an artifact nobody asked for."""
    steps, notes = repair_plan([
        _plan_step("check the pressure drop", "code.exec",
                   "stdout contains the pressure", {"code": "print(1)"})])
    assert "save_as" not in steps[0]["args"]
    assert notes == []


def test_a_document_named_in_an_intent_is_not_mistaken_for_source():
    """`.docx` is not something the sandbox can run, and saving one as source
    would put a Word file's name on a Python file."""
    steps, _ = repair_plan([
        _plan_step("write the findings to report.docx", "code.exec",
                   "stdout shows the summary", {"code": "print(1)"})])
    assert "save_as" not in steps[0]["args"]


def test_an_explicit_save_as_is_never_overwritten():
    steps, notes = repair_plan([
        _plan_step("write orifice.py", "code.exec", "it runs",
                   {"code": "print(1)", "save_as": "chosen_name.py"})])
    assert steps[0]["args"]["save_as"] == "chosen_name.py"
    assert notes == []


@pytest.mark.parametrize("named", ["solver.c", "Main.java", "app.js", "run.sh"])
def test_every_language_the_sandbox_runs_is_recognised_in_an_intent(named):
    steps, _ = repair_plan([
        _plan_step(f"write {named}", "code.exec", "it runs", {"code": "x"})])
    assert steps[0]["args"]["save_as"] == named


def test_a_code_exec_step_still_cannot_be_judged_by_a_file_the_program_wrote():
    """save_as saves the SOURCE, on the host, before the container starts. It
    does not make the sandbox able to write files, and the refusal that keeps
    plans honest about that has to survive."""
    with pytest.raises(PlanError, match="discarded"):
        validate_plan([_plan_step("run it", "code.exec",
                                  "results.csv is created in the workspace",
                                  {"code": "print(1)", "save_as": "run.py"})])


def test_a_code_exec_step_may_be_judged_by_the_source_it_saves():
    """`save_as` writes the program on the host before the container starts, so
    THAT file really does appear in the workspace. Refusing it would send the
    model to repair a plan that was already correct."""
    steps = validate_plan([_plan_step(
        "write and run orifice.py", "code.exec",
        "orifice.py is created in the workspace",
        {"code": "print(1)", "save_as": "orifice.py"})])
    assert steps[0].args["save_as"] == "orifice.py"


def test_saving_one_file_does_not_excuse_an_acceptance_about_another():
    """The exception is narrow on purpose: it covers the source this step
    writes, and nothing the program running inside the sandbox produced."""
    with pytest.raises(PlanError, match="discarded"):
        validate_plan([_plan_step(
            "run it", "code.exec", "results.csv is created in the workspace",
            {"code": "print(1)", "save_as": "orifice.py"})])


# --------------------------------------------- a passage is not a program

def test_a_step_reference_in_code_is_refused():
    """`{{step0}}` resolves to what an earlier step FOUND, and a page of an
    inspection report handed to a compiler produces:

        /sandbox/main.c:1:1: error: unknown type name 'MECHANICAL'
        1 | MECHANICAL SEAL & ANCILLARY INSPECTION REPORT

    Observed three times in one run, burning every replan.
    """
    with pytest.raises(PlanError, match="cannot read them"):
        validate_plan([_plan_step("run the findings", "code.exec",
                                  "it prints the result", {"code": "{{step0}}"})])


def test_the_refusal_says_what_to_do_instead():
    with pytest.raises(PlanError) as exc:
        validate_plan([_plan_step("run it", "code.exec", "it prints",
                                  {"code": "print({{step1}})"})])
    assert "write the program out in full" in str(exc.value).lower()


def test_a_reference_in_a_document_body_is_still_the_right_thing_to_do():
    """The mechanism is not the problem - quoting retrieved passages into a
    document is exactly what it was designed for, and every existing plan does
    it. Only source code has to be written rather than quoted."""
    steps = validate_plan([_plan_step(
        "write the findings up", "doc.write", "note.docx is created",
        {"path": "note.docx", "body": "Findings:\n\n{{step0}}"})])
    assert "{{step0}}" in steps[0].args["body"]


def test_code_with_no_reference_is_untouched():
    steps = validate_plan([_plan_step(
        "compute it", "code.exec", "it prints the answer",
        {"code": "print(2 + 2)"})])
    assert steps[0].args["code"] == "print(2 + 2)"


# ------------------------------------------- running a file from the workspace

def test_a_mounted_source_file_is_the_program_when_code_is_empty(ws):
    """The shape MODE_DIRECTIVE has asked for all along - fs.write the program,
    then run it - which the tool could not actually honour: `files` only ever
    mounted the file as read-only INPUT, and what ran was still `code`. A model
    that did the right thing therefore typed the compile command into `code`:

        /sandbox/main.c:1:5: error: expected '=' ... before '-' token
        1 | gcc -o pressure_drop pressure_drop.c
    """
    from kavach.tools.impl.sandbox import program_from_files

    ws.resolve("solver.py").parent.mkdir(parents=True, exist_ok=True)
    ws.resolve("solver.py").write_text("print('from the file')", encoding="utf-8")

    found = program_from_files(ws, "", ["solver.py"])
    assert found is not None
    source, why = found
    assert source == "print('from the file')"
    assert "solver.py" in why


def test_a_snippet_that_was_passed_is_never_overridden_by_a_mounted_file(ws):
    """`files` also means "data this program reads". A non-empty `code` is the
    program the caller meant to run, and replacing it would be inventing
    intent."""
    from kavach.tools.impl.sandbox import program_from_files

    ws.resolve("data.py").parent.mkdir(parents=True, exist_ok=True)
    ws.resolve("data.py").write_text("x = 1", encoding="utf-8")
    assert program_from_files(ws, "print('mine')", ["data.py"]) is None


def test_two_source_files_are_not_guessed_between(ws):
    """Which one holds `main` is not knowable from here."""
    from kavach.tools.impl.sandbox import program_from_files

    for name in ("a.py", "b.py"):
        ws.resolve(name).parent.mkdir(parents=True, exist_ok=True)
        ws.resolve(name).write_text("x = 1", encoding="utf-8")
    assert program_from_files(ws, "", ["a.py", "b.py"]) is None


def test_nothing_to_run_is_refused_with_both_ways_named(ws, monkeypatch):
    from kavach.tools.impl.sandbox import code_exec
    from kavach.tools.registry import ToolError

    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    with pytest.raises(ToolError, match="fs.write"):
        code_exec(ws, code="   ")


# ------------------------------------------ a build command is not a program

def test_a_compiler_invocation_is_rewritten_to_run_the_file():
    """Observed, with the file already written correctly one step earlier:

        "code": "gcc -o pressure_drop pressure_drop.c"
        /sandbox/main.c:1:5: error: expected '=', ',', ';' before '-' token

    The model reached for the compiler because, until `files` could run a
    program, that was the only place to put it. The single correct repair is to
    run the file it named - the file, the tool and the intent are all its own.
    """
    steps, notes = repair_plan([_plan_step(
        "fix the compilation error", "code.exec",
        "gcc compiles pressure_drop.c without errors",
        {"code": "gcc -o pressure_drop pressure_drop.c"})])
    assert steps[0]["args"]["code"] == ""
    assert steps[0]["args"]["files"] == ["pressure_drop.c"]
    assert "compiles for you" in notes[0]


@pytest.mark.parametrize("command", [
    "gcc -o p p.c", "g++ -std=c++17 main.cpp", "javac Main.java",
    "python3 solver.py", "node app.js", "go run main.go", "./pressure_drop",
    "cat pressure_drop.c",
])
def test_every_shape_of_build_command_is_caught(command):
    steps, _ = repair_plan([_plan_step(
        "build it", "code.exec", "it compiles",
        {"code": command, "files": ["pressure_drop.c"]})])
    assert steps[0]["args"]["code"] == "", f"{command!r} was left as the program"


def test_a_real_program_is_never_mistaken_for_a_command():
    """A one-line program is still a program. `print(2+2)` must survive."""
    for program in ("print(2+2)", 'console.log("hi")', 'printf("x");',
                    "#include <stdio.h>\nint main(void){return 0;}"):
        steps, notes = repair_plan([_plan_step(
            "compute", "code.exec", "it prints", {"code": program})])
        assert steps[0]["args"]["code"] == program
        assert notes == []


def test_a_build_command_naming_no_file_is_left_for_the_model():
    """With nothing to run, emptying `code` would leave a step that does
    nothing at all - worse than the error, which at least says what happened."""
    steps, notes = repair_plan([_plan_step(
        "build it", "code.exec", "it compiles", {"code": "make all"})])
    assert steps[0]["args"]["code"] == "make all"
    assert notes == []


def test_repairs_do_not_undo_each_other():
    """Each repair used to do `args = dict(step_args)` then `step["args"] = args`,
    so the LAST block to run threw away whatever the earlier ones had fixed: the
    build-command rewrite was silently undone by the save_as block four lines
    later, and the step ran `gcc ...` as a program exactly as before."""
    steps, notes = repair_plan([_plan_step(
        "fix the compilation error in pressure_drop.c", "code.exec",
        "gcc compiles pressure_drop.c without errors",
        {"code": "gcc -o pressure_drop pressure_drop.c"})])
    args = steps[0]["args"]
    assert args["code"] == "", "the build-command repair was undone"
    assert args["files"] == ["pressure_drop.c"]
    assert args["save_as"] == "pressure_drop.c", "the save_as repair was undone"
    assert len(notes) == 2, "both repairs have to be reported, not just the last"

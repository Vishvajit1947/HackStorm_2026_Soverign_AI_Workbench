"""code.exec runs a snippet as the language it IS, or refuses to run it.

Written from the failure that prompted it: a C program went into code.exec and
came back as

    File "/sandbox/main.py", line 1
    printf("Hello, World!\\n");
    NameError: name 'printf' is not defined. Did you mean: 'print'?

Every property here exists to keep one interpreter from claiming a file written
in another language, and to keep a missing toolchain from being reported as bad
code.
"""

from __future__ import annotations

import pytest

from kavach.tools.impl import languages as L
from kavach.tools.impl.sandbox import (
    ToolchainUnavailable,
    choose_language,
    code_exec,
)
from kavach.tools.registry import ToolError
from kavach.tools.workspace import Workspace


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run")


# ------------------------------------------------------------------ detection

def test_the_bare_printf_that_started_this_is_not_python():
    """The exact snippet from the bug report: one line, no include, no main.

    Any rule that wanted a whole C program would still have handed this to the
    Python interpreter, which is how the NameError was produced.
    """
    lang, why = L.detect('printf("Hello, World!\\n");')
    assert lang.id == "c"
    assert "printf" in why


@pytest.mark.parametrize(("code", "expected"), [
    ('#include <stdio.h>\nint main(void){printf("hi");return 0;}', "c"),
    ('#include <iostream>\nint main(){std::cout << "hi";}', "cpp"),
    ("public class Main { public static void main(String[] a){ System.out.println(1); } }", "java"),
    ('console.log("hi")', "javascript"),
    ("package main\nimport \"fmt\"\nfunc main(){ fmt.Println(1) }", "go"),
    ('fn main() { println!("hi"); }', "rust"),
    ("#!/bin/bash\necho hi", "shell"),
    ('print("hi")', "python"),
    ("import numpy as np\nprint(np.sum([1,2]))", "python"),
])
def test_each_language_is_recognised_from_its_own_idiom(code, expected):
    assert L.detect(code)[0].id == expected


def test_a_python_docstring_about_c_is_still_python():
    """Detection reads the code, not the prose wrapped around it."""
    code = '"""The C version of this uses printf and #include <stdio.h>."""\nprint(2)'
    assert L.detect(code)[0].id == "python"


def test_a_c_comment_does_not_hide_the_include():
    """`#` opens a comment in three of these languages and a preprocessor
    directive in two. Stripping it blindly deleted `#include <stdio.h>` and the
    file detected as Python again - the original bug, reintroduced by the fix."""
    code = "// entry point below\n#include <stdio.h>\nint main(void){return 0;}"
    assert L.detect(code)[0].id == "c"


def test_a_python_program_printing_the_word_end_is_not_ruby():
    assert L.detect('print("the end")')[0].id == "python"


def test_a_fenced_snippet_is_run_as_the_fence_says():
    lang, why = L.detect("```c\nint main(void){return 0;}\n```")
    assert lang.id == "c"
    assert "fence" in why


def test_the_fence_itself_never_reaches_the_compiler():
    """gcc and javac both fail on line 1 of a file that starts with ```."""
    source, hint = L.strip_fence("```java\nclass A {}\n```")
    assert hint == "java"
    assert source == "class A {}"


def test_an_unknown_fence_does_not_fail_the_call():
    """A fence naming something this table has never heard of is a bad hint,
    not a broken snippet - detection carries on underneath it."""
    assert L.detect("```brainfuck\nprint(1)\n```")[0].id == "python"


def test_detection_always_says_what_it_keyed_on():
    for code in ('print(1)', 'printf("x");', "console.log(1)"):
        assert L.detect(code)[1].strip()


# ------------------------------------------------------------------ choosing

def test_an_explicit_language_argument_beats_the_source():
    """An argument is a decision. A step that says `language: c` and passes
    something that looks like Python has to fail as C, not be quietly rerouted -
    the operator is owed the error they actually caused."""
    lang, why = choose_language("print(1)", requested="c")
    assert lang.id == "c"
    assert "language argument" in why


@pytest.mark.parametrize("alias", ["c++", "CPP", "cxx", " C++ "])
def test_the_names_a_model_actually_types_are_accepted(alias):
    assert choose_language("x", requested=alias)[0].id == "cpp"


def test_an_unknown_language_is_refused_by_name_not_silently_defaulted():
    with pytest.raises(L.UnknownLanguage, match="cobol"):
        choose_language("print(1)", requested="cobol")


def test_the_refusal_lists_what_is_supported():
    with pytest.raises(L.UnknownLanguage) as exc:
        L.resolve("cobol")
    for named in ("python", "java", "cpp"):
        assert named in str(exc.value)


def test_one_mounted_source_file_names_the_language():
    """The shape MODE_DIRECTIVE asks for: fs.write solution.c, then run it."""
    lang, why = choose_language("// run it", files=["solution.c"])
    assert lang.id == "c"
    assert "solution.c" in why


def test_two_mounted_files_of_different_languages_are_not_a_signal():
    """Two files is a guess, and a guess here picks an interpreter."""
    lang, _ = choose_language('printf("x");', files=["a.py", "b.java"])
    assert lang.id == "c"          # fell through to the source, which is C


def test_a_mounted_file_does_not_override_a_snippet_that_is_clearly_something_else():
    code = "public class Main { public static void main(String[] a){} }"
    assert choose_language(code, files=["notes.py"])[0].id == "java"


# ------------------------------------------------------------------ filenames

def test_a_java_file_is_named_after_its_public_class():
    """javac refuses a public class whose file name does not match it. Getting
    this wrong produces a compiler error about correct code, caused by us."""
    code = "public class HeatBalance { public static void main(String[] a){} }"
    assert L.source_filename(L.LANGUAGES["java"], code) == "HeatBalance.java"


def test_a_java_snippet_with_no_class_still_gets_a_legal_file_name():
    assert L.source_filename(L.LANGUAGES["java"], "int x = 1;") == "Main.java"


def test_a_class_name_that_is_not_an_identifier_is_not_used():
    """The class name reaches the run command as the class to invoke, so it is
    validated rather than sanitised - a name the source does not declare would
    fail to run anyway, and inventing one hides which of the two is wrong."""
    assert L.java_class("public class 9Lives {}") == "Main"


@pytest.mark.parametrize(("lang_id", "expected"), [
    ("python", "main.py"), ("c", "main.c"), ("cpp", "main.cpp"),
    ("javascript", "main.js"), ("shell", "main.sh"), ("rust", "main.rs"),
])
def test_every_other_language_gets_its_own_extension(lang_id, expected):
    assert L.source_filename(L.LANGUAGES[lang_id], "whatever") == expected


# ------------------------------------------------------------------ commands

def test_nothing_model_authored_reaches_the_shell():
    """The one property that makes `compile && run` acceptable. The snippet is
    passed as the contents of a file; if it ever reached the command line,
    `; rm -rf /` in a comment would be a command."""
    evil = 'print("x"); rm -rf /; $(whoami)`id`'
    line = L.shell_command(
        L.LANGUAGES["c"], src="/sandbox/main.c", out="/work/program",
        dir="/work", klass="Main",
    )
    assert evil not in line
    assert line.startswith("gcc ")
    assert "&& exec /work/program" in line


def test_a_compiled_language_runs_only_if_the_compile_succeeded():
    """`&&`, never `;`. A failed compile followed by an unconditional run
    executes the PREVIOUS binary if one is lying around, and reports its output
    as this snippet's."""
    for lang_id in ("c", "cpp", "java", "rust"):
        line = L.shell_command(
            L.LANGUAGES[lang_id], src=f"/sandbox/main{L.LANGUAGES[lang_id].extension}",
            out="/work/program", dir="/work", klass="Main",
        )
        # The join between COMPILE and RUN, specifically. Semicolons elsewhere
        # are fine and expected - the exports and the Go telemetry prelude are
        # both `;`-separated, and neither decides whether the program runs.
        assert "&& exec " in line
        assert "; exec " not in line


def test_go_builds_inside_the_tmpfs():
    """The root filesystem is read-only by design, and go refuses to build with
    nowhere to cache. Pointing it anywhere but /work makes every Go run fail
    with a permissions error that looks nothing like its cause."""
    line = L.shell_command(
        L.LANGUAGES["go"], src="/sandbox/main.go", out="/work/program",
        dir="/work", klass="Main",
    )
    assert "GOCACHE=/work/" in line


# ------------------------------------------------------------------ refusals

def test_a_language_with_no_toolchain_is_refused_not_run_as_something_else(ws, monkeypatch):
    """The whole point. With no Docker and no host compiler, a C snippet must
    produce a refusal naming C - never a Python traceback."""
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    monkeypatch.setattr("kavach.tools.impl.sandbox.shutil.which", lambda _: None)

    with pytest.raises(ToolchainUnavailable) as exc:
        code_exec(ws, 'printf("Hello, World!\\n");')

    message = str(exc.value)
    assert "C" in message
    assert "NOT run with another interpreter" in message
    assert "NameError" not in message


def test_the_refusal_is_not_a_sandbox_failure(ws, monkeypatch):
    """A missing compiler does not mean the isolation claim is broken, and the
    fix is a line in the image rather than 'install Docker'. Conflating the two
    sends the operator to the wrong problem."""
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    monkeypatch.setattr("kavach.tools.impl.sandbox.shutil.which", lambda _: None)

    from kavach.tools.impl.sandbox import SandboxUnavailable
    with pytest.raises(ToolchainUnavailable) as exc:
        code_exec(ws, "public class Main { public static void main(String[] a){} }")
    assert not isinstance(exc.value, SandboxUnavailable)
    assert "sandbox-runner" in str(exc.value)


def test_an_unusable_language_argument_is_a_tool_error(ws, monkeypatch):
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    with pytest.raises(ToolError, match="cobol"):
        code_exec(ws, "print(1)", language="cobol")


def test_an_image_probe_that_cannot_answer_does_not_refuse_anything(monkeypatch):
    """None is not an empty set. 'The probe could not run' and 'the image has no
    compilers' are different facts, and refusing a run over the first is exactly
    the confident wrong verdict this codebase keeps out."""
    import kavach.tools.impl.sandbox as sb
    monkeypatch.setattr(sb, "image_toolchains", lambda *_a, **_k: None)
    assert sb._missing_toolchain(L.LANGUAGES["java"], "kavach/sandbox:test") is None
    assert set(sb.available_languages().values()) == {None}


def test_a_probed_image_reports_per_language_availability(monkeypatch):
    import kavach.tools.impl.sandbox as sb
    monkeypatch.setattr(sb, "image_toolchains", lambda *_a, **_k: {"python", "gcc", "sh"})
    available = sb.available_languages()
    assert available["python"] is True
    assert available["c"] is True
    assert available["java"] is False           # javac and java are both absent
    assert "default-jdk" in sb._missing_toolchain(L.LANGUAGES["java"], "img")


# ------------------------------------------------------------------ execution

def test_python_still_runs_on_the_host_fallback_unchanged(ws, monkeypatch):
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    result = code_exec(ws, "print(2 + 2)")
    assert result["stdout"].strip() == "4"
    assert result["sandboxed"] is False
    assert result["language"] == "python"


def test_every_result_says_what_it_ran_as_and_why(ws, monkeypatch):
    """A reader looking at a compiler error needs to know which language was
    chosen and whether the call named it - that is the difference between a bug
    in the snippet and a bug in the dispatcher."""
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    result = code_exec(ws, "print(1)")
    assert result["language_label"] == "Python"
    assert result["source_file"] == "main.py"
    assert result["language_reason"]
    assert result["compiled"] is False


def test_a_statement_is_not_a_program_and_the_hint_says_so():
    """`printf("hi\\n");` compiles to a gcc error written for someone who
    already knows C. The hint names what is missing - and says nothing was
    added, because wrapping the snippet in a main() nobody wrote would put
    unauditable lines in the Code tab."""
    hint = L.missing_entry_point(L.LANGUAGES["c"], 'printf("hi");')
    assert hint is not None
    assert "int main" in hint


def test_a_complete_program_gets_no_entry_point_hint():
    code = "#include <stdio.h>\nint main(void){ printf(\"hi\"); return 0; }"
    assert L.missing_entry_point(L.LANGUAGES["c"], code) is None


def test_an_interpreted_language_never_gets_an_entry_point_hint():
    """A bare statement is a complete Python program. Telling its author to add
    a main() would be advice that makes working code worse."""
    assert L.missing_entry_point(L.LANGUAGES["python"], "print(1)") is None


def test_the_seccomp_profile_allows_the_fork_a_compile_step_needs():
    """/bin/sh is dash and dash forks with vfork. Without it, every compiled
    language dies as `sh: 1: Cannot fork` before the compiler is reached - which
    reads as a broken sandbox rather than a missing syscall. Found exactly that
    way, on the first C snippet that got as far as the container."""
    import json
    from pathlib import Path

    profile = Path(__file__).resolve().parents[1] / "deploy" / "seccomp-strict.json"
    allowed: set[str] = set()
    for rule in json.loads(profile.read_text(encoding="utf-8"))["syscalls"]:
        if rule.get("action") == "SCMP_ACT_ALLOW":
            allowed.update(rule["names"])
    assert {"vfork", "clone", "execve", "wait4"} <= allowed


def test_the_seccomp_profile_still_refuses_the_network():
    """The fork allowance must not have quietly widened anything else: the
    second, independent control over egress is that these are absent."""
    import json
    from pathlib import Path

    profile = Path(__file__).resolve().parents[1] / "deploy" / "seccomp-strict.json"
    allowed: set[str] = set()
    for rule in json.loads(profile.read_text(encoding="utf-8"))["syscalls"]:
        if rule.get("action") == "SCMP_ACT_ALLOW":
            allowed.update(rule["names"])
    assert not ({"socket", "connect", "bind", "listen", "sendto"} & allowed)


def test_the_sandbox_image_declares_every_toolchain_the_table_promises():
    """A language in the table with no toolchain in the image is a call that
    refuses. Go and Rust are deliberately out - their install hints say so -
    but the ones the manifest advertises have to be in the Dockerfile."""
    from pathlib import Path

    dockerfile = (
        Path(__file__).resolve().parents[1] / "deploy" / "Dockerfile.sandbox"
    ).read_text(encoding="utf-8")
    for package in ("gcc", "g++", "default-jdk-headless", "nodejs"):
        assert package in dockerfile, f"{package} is promised by code.exec and not installed"


def test_a_fenced_python_snippet_runs_rather_than_raising_a_syntax_error(ws, monkeypatch):
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    monkeypatch.setenv("KAVACH_ALLOW_UNSANDBOXED_EXEC", "1")
    result = code_exec(ws, '```python\nprint("fenced")\n```')
    assert result["stdout"].strip() == "fenced"
    assert result["exit_code"] == 0


def test_a_forgotten_include_is_named_rather_than_inserted():
    """gcc makes an implicit declaration an ERROR under -std=c17, and a
    forgotten `#include <stdio.h>` is the single most common thing a small model
    leaves out of a C program. The compiler's message talks about "implicit
    declaration" and never mentions the missing line.

    A hint, never an edit: inserting it would put source in the file that the
    model did not write and the reader cannot audit.
    """
    hint = L.missing_header(L.LANGUAGES["c"], 'int main(void){ printf("x"); }')
    assert hint and "stdio.h" in hint and "printf" in hint


def test_an_include_that_is_present_is_not_reported():
    code = '#include <stdio.h>\nint main(void){ printf("x"); return 0; }'
    assert L.missing_header(L.LANGUAGES["c"], code) is None


def test_the_hint_covers_the_maths_header_too():
    """A plant calculation reaches for sqrt within two lines, and -lm alone does
    not help if the declaration is missing."""
    code = '#include <stdio.h>\nint main(void){ double y = sqrt(2.0); return 0; }'
    assert "math.h" in L.missing_header(L.LANGUAGES["c"], code)


def test_an_interpreted_language_never_gets_an_include_hint():
    assert L.missing_header(L.LANGUAGES["python"], "print(1)") is None
    assert L.missing_header(L.LANGUAGES["javascript"], "console.log(1)") is None


def test_a_mention_in_a_comment_is_not_a_call():
    """Detection reads the code, not the prose - the same rule the language
    detector follows."""
    code = '#include <stdio.h>\n// remember malloc needs stdlib.h\nint main(void){ printf("x"); }'
    assert L.missing_header(L.LANGUAGES["c"], code) is None


def test_the_image_carries_every_toolchain_the_table_promises():
    """The table and the Dockerfile are one list written twice. A language
    code.exec advertises and the image cannot run is a call that refuses - which
    is honest, and is not what "run any language" means."""
    from pathlib import Path

    dockerfile = (
        Path(__file__).resolve().parents[1] / "deploy" / "Dockerfile.sandbox"
    ).read_text(encoding="utf-8")
    packages = {
        "gcc": "gcc", "g++": "g++", "javac": "default-jdk-headless",
        "java": "default-jdk-headless", "node": "nodejs", "go": "golang-go",
        "rustc": "rustc", "ruby": "ruby",
        "python": "FROM python", "sh": "FROM python",   # the base image
    }
    for lang in L.LANGUAGES.values():
        for binary in lang.binaries:
            needed = packages.get(binary)
            assert needed, f"{binary} is in the table with no package mapped to it"
            assert needed in dockerfile, (
                f"{lang.label} needs {binary}, and {needed} is not installed in "
                "deploy/Dockerfile.sandbox"
            )


def test_a_toolchain_that_is_installed_and_cannot_run_is_refused_up_front(ws, monkeypatch):
    """Rust compiles in this image and then cannot exec its linker under the
    seccomp profile. Letting somebody write a whole Rust program and meet
    "could not exec the linker `cc`" at the end is the late, confusing failure
    this codebase keeps turning into an early, specific one."""
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: True)
    monkeypatch.setattr("kavach.tools.impl.sandbox.image_available", lambda *a, **k: True)
    with pytest.raises(ToolchainUnavailable, match="linker"):
        code_exec(ws, 'fn main() { println!("hi"); }')


def test_the_refusal_says_the_other_languages_are_unaffected(ws, monkeypatch):
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: True)
    monkeypatch.setattr("kavach.tools.impl.sandbox.image_available", lambda *a, **k: True)
    with pytest.raises(ToolchainUnavailable) as exc:
        code_exec(ws, "fn main() {}")
    assert "unaffected" in str(exc.value)
    assert "seccomp" in str(exc.value)


def test_every_other_language_is_runnable():
    """The list this product claims. A language marked unsupported is a promise
    withdrawn, so the set has to be small, deliberate and visible in one place."""
    blocked = {k for k, v in L.LANGUAGES.items() if v.unsupported}
    assert blocked == {"rust"}, (
        f"the set of languages that cannot run has changed: {blocked}"
    )


def test_the_seccomp_profile_lets_a_compiler_start_its_own_helpers():
    """gcc runs cc1, as and ld; `go run` builds and then executes. Denying the
    spawn syscalls made a correct program fail with "operation not permitted"
    in the middle of a build."""
    import json
    from pathlib import Path

    profile = Path(__file__).resolve().parents[1] / "deploy" / "seccomp-strict.json"
    allowed: set[str] = set()
    for rule in json.loads(profile.read_text(encoding="utf-8"))["syscalls"]:
        if rule.get("action") == "SCMP_ACT_ALLOW":
            allowed.update(rule["names"])
    assert {"fork", "vfork", "clone", "execve", "execveat", "setpgid"} <= allowed
    # ...and the thing that widening must never touch.
    assert not ({"socket", "connect", "bind", "listen", "sendto"} & allowed)


def test_having_the_binary_is_not_the_same_as_being_able_to_run_it(monkeypatch):
    """rustc, cargo and cc are all in the image, so a binary check alone
    reported Rust as available - and the panel promised a language that every
    call refuses."""
    import kavach.tools.impl.sandbox as sb

    monkeypatch.setattr(
        sb, "image_toolchains",
        lambda *a, **k: {b for lang in L.LANGUAGES.values() for b in lang.binaries},
    )
    available = sb.available_languages()
    assert available["python"] is True
    assert available["rust"] is False, "a language that cannot run is not available"

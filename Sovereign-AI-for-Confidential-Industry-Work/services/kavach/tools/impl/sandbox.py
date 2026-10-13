"""`code.exec` - run model-authored code with no network interface (§06).

This is the strongest single claim in the tool plane: **generated code cannot
exfiltrate even in principle**, because the container has no network interface at
all. Not a firewall rule that could be misconfigured - `--network=none` means the
kernel gives the process a namespace containing only loopback.

Show the judges a snippet that tries `urllib.request.urlopen` and watch it fail
with "Network is unreachable".

Two runners:

    docker      the real one. --network=none, read-only rootfs, tmpfs /work,
                cap-drop ALL, no-new-privileges, pid/mem/cpu limits, non-root.

    subprocess  DEVELOPMENT ONLY, and OFF BY DEFAULT. Running model-authored code
                in a bare subprocess on the developer's machine is genuinely
                dangerous - it has that machine's network, filesystem and
                credentials. It must be opted into explicitly with
                KAVACH_ALLOW_UNSANDBOXED_EXEC=1, it refuses to run otherwise, and
                every result it produces is stamped `sandboxed: false` so it can
                never be mistaken for the real thing in a transcript or a demo.

If Docker is unavailable and the opt-in is not set, the tool fails loudly. That is
the correct outcome: a sandbox that silently stops sandboxing is worse than one
that stops working.

**The language is part of the call, not an assumption.** Every snippet used to be
written to `main.py` and handed to the Python interpreter, so a C program came
back as `NameError: name 'printf' is not defined` - a Python error about a file
that was never Python, blaming the model for code that was correct. `languages.py`
holds the table; this module stages the source under the right name and runs it
with its own toolchain. When the image has no toolchain for it, the run is
REFUSED with the reason and the line that fixes it, because running it with the
wrong one is how the original bug read as a model failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from ..registry import SandboxSpec, ToolError
from ..workspace import Workspace
from .languages import (
    LANGUAGES as LANGUAGE_TABLE,
)
from .languages import (
    Language,
    UnknownLanguage,
    detect,
    fill,
    language_for_path,
    missing_entry_point,
    missing_header,
    resolve,
    shell_command,
    source_filename,
    strip_fence,
)

SANDBOX_IMAGE = os.environ.get("KAVACH_SANDBOX_IMAGE", "kavach/sandbox:0.1.0")
UNSANDBOXED_ENV = "KAVACH_ALLOW_UNSANDBOXED_EXEC"

# Where a compiled language builds. /work is the tmpfs - the root filesystem is
# read-only, so this is the only writable path in the container.
WORK_DIR = "/work"
BUILD_OUTPUT = "/work/program"
STAGE_DIR = "/sandbox"


class SandboxUnavailable(ToolError):
    """No sandbox, and no explicit permission to run without one."""


class ToolchainUnavailable(ToolError):
    """The sandbox runs, but has no compiler or interpreter for this language.

    Deliberately NOT SandboxUnavailable: the sandbox is fine and the isolation
    claim still holds. What is missing is one toolchain, and the fix is a line
    in the image rather than "install Docker".
    """


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def image_available(image: str = SANDBOX_IMAGE) -> bool:
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return out.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _unsandboxed_allowed() -> bool:
    return os.environ.get(UNSANDBOXED_ENV, "").strip().lower() in {"1", "true", "yes"}


def choose_language(
    code: str, requested: str | None = None, files: list[str] | None = None
) -> tuple[Language, str]:
    """(language, why). Three sources, most authoritative first.

    1. `language`, if the caller named one. An explicit argument is a decision,
       not a hint, so an unusable value is an error rather than something to
       quietly improve on.
    2. The extension of a single mounted file. A step that mounts `solution.c`
       and executes it has said what it is without saying it twice - and this is
       the shape MODE_DIRECTIVE asks the planner for. Only with exactly one
       file: two files of different languages is a guess, not a signal.
    3. What the source looks like. Last, and it says what it keyed on.
    """
    if requested:
        return resolve(requested), f"the language argument said {requested!r}"

    staged = [f for f in (files or []) if language_for_path(f)]
    if len(staged) == 1:
        lang = language_for_path(staged[0])
        assert lang is not None
        detected, _ = detect(code)
        # The mounted file and the snippet disagreeing is normal: `files` is the
        # program and `code` is often a two-line driver for it. The file wins
        # because it is the artifact, but only when the snippet is not clearly
        # something else.
        if detected.id == lang.id or detected.id == "python":
            return lang, f"the extension of the mounted file {staged[0]}"

    return detect(code)


def save_source(
    workspace: Workspace, code: str, lang: Language, save_as: str
) -> dict[str, Any]:
    """Keep the program, in the workspace, under its own extension.

    NOT the same claim as "the sandbox can write files" - it cannot, and the
    planner still refuses a step judged by a file the sandbox produced. This
    saves the SOURCE the model authored, on the host, before the container ever
    starts. The program is a deliverable in its own right: a run asked to "write
    orifice.py and run it" that leaves nothing behind but a .docx has not
    produced what it was asked for, which is exactly the complaint this answers.

    The extension is corrected to the language that will actually run, because a
    C program saved as `.py` is the original `printf` bug wearing a filename.
    """
    path = Path(save_as.strip().replace("\\", "/"))
    stem = path.stem or f"main{lang.extension}"
    corrected = path.with_name(stem + lang.extension)
    resolved = workspace.resolve(str(corrected))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    source, _fence = strip_fence(code)
    resolved.write_text(source, encoding="utf-8")
    return {
        "path": str(corrected).replace("\\", "/"),
        "renamed": str(corrected) != str(path),
        "bytes": len(source.encode("utf-8")),
    }


def program_from_files(
    workspace: Workspace, code: str, files: list[str] | None
) -> tuple[str, str] | None:
    """(source, why) when the program to run is a file in the workspace.

    THE SHAPE THE PLANNER IS ASKED TO PRODUCE, which until now the tool could
    not honour: write the program with fs.write, then run it by naming it in
    `files`. MODE_DIRECTIVE has told models to do exactly that for a long time,
    and `files` only ever mounted the file as read-only INPUT - the thing that
    ran was still the `code` argument. So a plan that did the right thing got:

        /sandbox/main.c:1:5: error: expected '=' ... before '-' token
        1 | gcc -o pressure_drop pressure_drop.c

    - the model, having written the file correctly, typed the compile command
    into `code` because there was nowhere else to put it.

    Only with an EMPTY `code` and exactly ONE source file. Two files is a guess
    about which one holds `main`, and a non-empty `code` is a snippet the caller
    meant to run - overriding either would be inventing intent.
    """
    if code.strip():
        return None
    sources = [f for f in (files or []) if language_for_path(f)]
    if len(sources) != 1:
        return None
    try:
        resolved = workspace.resolve(sources[0], must_exist=True)
        return resolved.read_text(encoding="utf-8"), f"the mounted file {sources[0]}"
    except (OSError, ValueError):
        return None


def code_exec(
    workspace: Workspace,
    code: str = "",
    files: list[str] | None = None,
    language: str | None = None,
    save_as: str | None = None,
    sandbox: SandboxSpec | None = None,
    timeout_s: int = 30,
    **_: Any,
) -> dict[str, Any]:
    spec = sandbox or SandboxSpec()

    from_file = program_from_files(workspace, code, files)
    ran_from_file = from_file is not None
    if from_file is not None:
        code = from_file[0]

    try:
        lang, why = choose_language(code, language, files)
    except UnknownLanguage as exc:
        raise ToolError(str(exc)) from exc
    if ran_from_file and from_file is not None:
        why = from_file[1]

    if not code.strip():
        raise ToolError(
            "code.exec was given nothing to run: `code` is empty and `files` "
            "names no source file. Either pass the program in `code`, or write "
            "it with fs.write first and name it in `files`."
        )

    # A toolchain that is present and still cannot run here is refused at the
    # START. Letting somebody write a whole Rust program and meet "could not
    # exec the linker" at the end is the late, confusing failure this codebase
    # keeps turning into an early, specific one.
    if lang.unsupported:
        raise ToolchainUnavailable(
            f"code.exec detected {lang.label} ({why}), and {lang.unsupported}."
        )

    # Saved BEFORE the run, deliberately. A snippet that crashes is still the
    # program the model wrote, and it is the thing a person needs to read to fix
    # it - saving only successful programs would delete the evidence exactly
    # when it matters.
    saved = save_source(workspace, code, lang, save_as) if save_as else None

    if docker_available() and image_available():
        result = _run_docker(workspace, code, files or [], spec, timeout_s, lang, why)
        return {**result, **_saved_fields(saved)}

    if not _unsandboxed_allowed():
        raise SandboxUnavailable(
            "code.exec needs the Docker sandbox and it is not available.\n"
            f"  docker present: {docker_available()}\n"
            f"  image {SANDBOX_IMAGE} present: "
            f"{image_available() if docker_available() else 'n/a'}\n"
            "\n"
            "Build it with:  docker compose build sandbox-runner\n"
            "\n"
            f"To run WITHOUT a sandbox during development, set {UNSANDBOXED_ENV}=1. "
            "That executes model-authored code directly on this machine, with this "
            "machine's network and filesystem. Never set it on a demo box."
        )

    result = _run_subprocess(workspace, code, files or [], timeout_s, lang, why)
    return {**result, **_saved_fields(saved)}


def _saved_fields(saved: dict[str, Any] | None) -> dict[str, Any]:
    """`path` is the key the agent loop records artifacts from, so setting it is
    what makes the program appear in "Files it produced"."""
    if not saved:
        return {}
    fields: dict[str, Any] = {"path": saved["path"], "saved_bytes": saved["bytes"]}
    if saved["renamed"]:
        fields["warnings"] = [(
            f"saved as {saved['path']} - the extension was corrected to match the "
            "language the sandbox actually ran"
        )]
    return fields


# ------------------------------------------------------------- toolchains

# One probe per image per process. Starting a container to ask `command -v` costs
# about half a second; doing it on every call would put that on the clock of
# every snippet, and the answer cannot change while the image id does not.
_TOOLCHAIN_CACHE: dict[str, set[str] | None] = {}

# Everything any language in the table needs, asked in one container.
_ALL_BINARIES = sorted({b for lang in LANGUAGE_TABLE.values() for b in lang.binaries})


def image_toolchains(image: str = SANDBOX_IMAGE) -> set[str] | None:
    """Which of the known binaries exist in the image, or None if unknowable.

    None is not an empty set and must not be treated as one: "the probe could
    not run" and "the image has no compilers" are different facts, and refusing
    a Java run because a probe timed out would be exactly the kind of confident
    wrong verdict this codebase keeps out.
    """
    if image in _TOOLCHAIN_CACHE:
        return _TOOLCHAIN_CACHE[image]

    script = "; ".join(f"command -v {b} >/dev/null 2>&1 && echo {b}" for b in _ALL_BINARIES)
    try:
        out = subprocess.run(
            ["docker", "run", "--rm", "--network=none", "--entrypoint", "sh", image, "-c", script],
            capture_output=True, text=True, timeout=60, check=False,
        )
        found: set[str] | None = (
            {line.strip() for line in out.stdout.splitlines() if line.strip()}
            if out.returncode == 0 else None
        )
    except (subprocess.SubprocessError, OSError):
        found = None

    _TOOLCHAIN_CACHE[image] = found
    return found


def available_languages(image: str = SANDBOX_IMAGE) -> dict[str, bool | None]:
    """Language id -> can this image run it (None where the probe could not say)."""
    present = image_toolchains(image)
    if present is None:
        return {lang_id: None for lang_id in LANGUAGE_TABLE}
    return {
        # `unsupported` counts as unavailable. Rust's binaries are all in the
        # image, so a binary check alone reported it as available - and the
        # panel then promised a language every call refuses. Having the binary
        # is not the same claim as being able to run the program.
        lang_id: (not lang.unsupported) and all(b in present for b in lang.binaries)
        for lang_id, lang in LANGUAGE_TABLE.items()
    }


def _missing_toolchain(lang: Language, image: str) -> str | None:
    """The sentence to refuse with, or None if there is nothing to refuse."""
    present = image_toolchains(image)
    if present is None:
        return None                     # unknowable - let the run speak instead
    missing = [b for b in lang.binaries if b not in present]
    if not missing:
        return None
    return (
        f"the sandbox image {image} has no {lang.label} toolchain - "
        f"{', '.join(missing)} is not in it.\n"
        f"  Fix: {lang.install_hint}\n"
        f"       then: docker compose build sandbox-runner\n"
        f"  Languages this image can run now: "
        f"{', '.join(sorted(k for k, v in available_languages(image).items() if v)) or '(none)'}\n"
        "This snippet was NOT run with a different interpreter - that is how a C "
        "program comes back as a Python NameError."
    )


# ---------------------------------------------------------------- docker

def _run_docker(
    workspace: Workspace,
    code: str,
    files: list[str],
    spec: SandboxSpec,
    timeout_s: int,
    lang: Language,
    why: str,
) -> dict[str, Any]:
    missing = _missing_toolchain(lang, SANDBOX_IMAGE)
    if missing is not None:
        raise ToolchainUnavailable(
            f"code.exec detected {lang.label} ({why}), and {missing}"
        )

    # The fence is markdown, not source: `javac` and `gcc` both fail on line 1 of
    # a file that starts with ```. Stripped here rather than in the detector so
    # what runs is what the reader is shown in the Code tab.
    source, _fence = strip_fence(code)

    with tempfile.TemporaryDirectory(prefix="kavach-exec-") as staging_str:
        staging = Path(staging_str)
        filename = source_filename(lang, code)
        (staging / filename).write_text(source, encoding="utf-8")

        # Copy requested files in rather than mounting the workspace, so the
        # container sees exactly what was asked for and nothing else.
        input_dir = staging / "input"
        input_dir.mkdir()
        mounted: list[str] = []
        for rel in files:
            source = workspace.resolve(rel, must_exist=True)
            target = input_dir / Path(rel).name
            target.write_bytes(source.read_bytes())
            mounted.append(target.name)

        # On Windows - especially under an elevated shell - the staging directory
        # inherits Administrator-only ACLs, and the container's unprivileged user
        # (nobody, 65534) then cannot read /sandbox/main.py. The symptom is an
        # empty stdout with exit 2, which looks like a broken sandbox rather than
        # a permissions problem, so grant read+execute to Everyone (S-1-1-0).
        #
        # Safe: the directory is a short-lived temp staging area holding only the
        # snippet about to run, and it is deleted when this block exits.
        acl_note: str | None = None
        if os.name == "nt":
            try:
                acl = subprocess.run(
                    ["icacls", str(staging), "/grant", "*S-1-1-0:(OI)(CI)RX", "/T"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if acl.returncode != 0:
                    acl_note = f"icacls exited {acl.returncode}: {acl.stderr.strip()[:200]}"
            except (OSError, subprocess.SubprocessError) as exc:
                # Not fatal - the grant is often unnecessary. But if the run then
                # produces nothing, this is the first thing to check, so carry it
                # into the result rather than dropping it.
                acl_note = f"{type(exc).__name__}: {exc}"

        # An interpreted language is exec'd directly, with no shell anywhere near
        # it. A compiled one needs `compile && run`, which needs one - so the
        # shell sees only the fixed command from the language table (see the
        # note at the top of languages.py). Either way the snippet reaches the
        # process as the CONTENTS of a file it was never asked to parse as a
        # command line.
        klass = filename[: -len(lang.extension)]
        slots = {
            "src": f"{STAGE_DIR}/{filename}",
            "out": BUILD_OUTPUT,
            "dir": WORK_DIR,
            "class": klass,
        }
        if lang.compiled or lang.env:
            argv = ["sh", "-c", shell_command(
                lang, src=slots["src"], out=slots["out"], dir=slots["dir"], klass=klass
            )]
        else:
            argv = fill(lang.run, **slots)

        cmd = [
            "docker", "run", "--rm",
            "--network=none",                       # <- no interface at all
            "--read-only",
            "--tmpfs=/work:size=256m,mode=1777,exec",
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            f"--pids-limit={spec.pids_limit}",
            f"--memory={spec.mem_gb}g",
            f"--cpus={spec.cpus}",
            "--user", "65534:65534",                # nobody
            "-v", f"{staging.as_posix()}:/sandbox:ro",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "HOME=/work",
            "-w", "/work",
            SANDBOX_IMAGE,
            *argv,
        ]

        seccomp = Path(__file__).resolve().parents[4] / "deploy" / "seccomp-strict.json"
        if seccomp.exists():
            cmd[4:4] = ["--security-opt", f"seccomp={seccomp.as_posix()}"]

        started = time.perf_counter()
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 10, check=False
            )
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"killed after {timeout_s}s",
                "exit_code": 124,
                "timed_out": True,
                "sandboxed": True,
                "runner": "docker",
                "network": "none",
                "elapsed_ms": timeout_s * 1000,
                "inputs": mounted,
                **_language_fields(lang, why, filename, argv),
            }

        result = {
            "stdout": out.stdout[-40_000:],
            "stderr": out.stderr[-20_000:],
            "exit_code": out.returncode,
            "timed_out": False,
            "sandboxed": True,
            "runner": "docker",
            "network": "none",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "inputs": mounted,
            **_language_fields(lang, why, filename, argv),
        }
        # Empty output with a non-zero exit on Windows is nearly always the ACL
        # problem above. Surface the note so it is diagnosable instead of looking
        # like the sandbox itself is broken.
        if acl_note and not out.stdout.strip() and out.returncode != 0:
            result["hint"] = f"staging ACL grant may have failed - {acl_note}"
        # 127 is the shell's "command not found". It can only happen here when
        # the probe said the toolchain was there and it was not, or could not
        # run at all - and the raw message ("exec: gcc: not found") reads as a
        # broken sandbox rather than a missing compiler.
        if out.returncode == 127:
            result["hint"] = (
                f"exit 127 - the {lang.label} toolchain is not runnable in "
                f"{SANDBOX_IMAGE}. {lang.install_hint}"
            )
        if out.returncode != 0 and (header := missing_header(lang, code)):
            result["hint"] = (
                f"this snippet calls into the standard library without including "
                f"it. Add `{header}` at the top. It was compiled exactly as "
                "written; nothing was added around it."
            )
        entry = missing_entry_point(lang, code) if out.returncode != 0 else None
        if entry:
            result["hint"] = (
                f"this snippet has no {lang.label} entry point. {lang.label} runs a "
                f"program, not a statement - the code has to be inside "
                f"`{entry}`. It was compiled exactly as written; nothing was added "
                "around it."
            )
        return result


def _language_fields(
    lang: Language, why: str, filename: str, argv: list[str]
) -> dict[str, Any]:
    """What ran, under what name, and why that language.

    Carried on every result including the failures. A reader looking at a
    compiler error needs to know the file was called `HelloWorld.java` and that
    the language was chosen by detection rather than named in the call - that is
    the difference between a bug in the snippet and a bug in this module.
    """
    return {
        "language": lang.id,
        "language_label": lang.label,
        "language_reason": why,
        "source_file": filename,
        "command": " ".join(argv),
        "compiled": lang.compiled,
    }


# ---------------------------------------------------------------- dev fallback

def _host_toolchain(lang: Language) -> str | None:
    """The missing binary, or None if the host can run this language.

    Checked with `shutil.which` and NOT with the image probe: the dev fallback
    runs on this machine, and a compiler being present in the container says
    nothing about whether one is present here.
    """
    if lang.id == "python":
        return None                     # sys.executable is running this code
    for binary in lang.binaries:
        if shutil.which(binary) is None:
            return binary
    return None


def _run_subprocess(
    workspace: Workspace,
    code: str,
    files: list[str],
    timeout_s: int,
    lang: Language,
    why: str,
) -> dict[str, Any]:
    """DEVELOPMENT ONLY. See the module docstring - this is not a sandbox."""
    for rel in files:
        workspace.resolve(rel, must_exist=True)      # still jailed

    missing = _host_toolchain(lang)
    if missing is not None:
        raise ToolchainUnavailable(
            f"code.exec detected {lang.label} ({why}), and `{missing}` is not on "
            f"this machine's PATH. The Docker sandbox is not available either, so "
            f"there is nothing here that can run {lang.label}.\n"
            f"  Fix: build the sandbox image (docker compose build sandbox-runner), "
            f"which carries the toolchain, or install {missing}.\n"
            "The snippet was NOT run with another interpreter."
        )

    source, _fence = strip_fence(code)

    with tempfile.TemporaryDirectory(prefix="kavach-exec-unsafe-") as tmp:
        filename = source_filename(lang, code)
        script = Path(tmp) / filename
        script.write_text(source, encoding="utf-8")

        build = Path(tmp) / "build"
        build.mkdir()
        out_binary = build / ("program.exe" if os.name == "nt" else "program")
        slots = {
            "src": str(script),
            "out": str(out_binary),
            "dir": str(build),
            "class": filename[: -len(lang.extension)],
        }
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            # javac and rustc need somewhere to put temporary files, and Windows
            # toolchains read these rather than inheriting a default.
            "TEMP": tmp, "TMP": tmp,
            **{key: value.format(**slots) for key, value in lang.env.items()},
        }
        # Python runs on THIS interpreter, not whatever `python` means on PATH -
        # the venv is where the project's dependencies are, and a bare `python`
        # on Windows is as likely to be the Store stub as anything else.
        argv = (
            [sys.executable, str(script)] if lang.id == "python"
            else fill(lang.run, **slots)
        )
        fields = _language_fields(lang, why, filename, argv)

        started = time.perf_counter()
        try:
            # Two processes rather than one shell line, unlike the container:
            # there is no POSIX shell to rely on here, and quoting a Windows
            # path with spaces into one is a bug waiting to be written.
            if lang.compiled:
                built = subprocess.run(
                    fill(lang.compile, **slots), capture_output=True, text=True,
                    timeout=timeout_s, cwd=tmp, check=False, env=env,
                )
                if built.returncode != 0:
                    entry = missing_entry_point(lang, code)
                    return {
                        "stdout": built.stdout[-40_000:],
                        "stderr": built.stderr[-20_000:],
                        "exit_code": built.returncode,
                        "timed_out": False, "sandboxed": False, "runner": "subprocess",
                        "network": "HOST - NOT ISOLATED",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                        "warning": _UNSANDBOXED_WARNING, "inputs": files,
                        "stage": "compile",
                        **({"hint": f"this snippet has no {lang.label} entry point - "
                                    f"the code has to be inside `{entry}`"} if entry else {}),
                        **fields,
                    }

            out = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workspace.root,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "stdout": "", "stderr": f"killed after {timeout_s}s", "exit_code": 124,
                "timed_out": True, "sandboxed": False, "runner": "subprocess",
                "network": "HOST - NOT ISOLATED", "elapsed_ms": timeout_s * 1000,
                "warning": _UNSANDBOXED_WARNING, "inputs": files,
                **fields,
            }

        return {
            "stdout": out.stdout[-40_000:],
            "stderr": out.stderr[-20_000:],
            "exit_code": out.returncode,
            "timed_out": False,
            "sandboxed": False,
            "runner": "subprocess",
            "network": "HOST - NOT ISOLATED",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "warning": _UNSANDBOXED_WARNING,
            "inputs": files,
            "stage": "run",
            **fields,
        }


_UNSANDBOXED_WARNING = (
    "NOT SANDBOXED - this ran directly on the host with full network and "
    "filesystem access. Development only. Never present this as sandboxed "
    "execution."
)


# ---------------------------------------------------------------- the demo

EGRESS_PROBE = textwrap.dedent(
    """\
    import socket, urllib.request
    print("attempting an outbound HTTP request from inside the sandbox...")
    try:
        urllib.request.urlopen("http://1.1.1.1", timeout=5)
        print("REACHED THE NETWORK - the sandbox is NOT isolated")
    except Exception as exc:
        print(f"blocked: {type(exc).__name__}: {exc}")
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 443))
        print("SOCKET CONNECTED - the sandbox is NOT isolated")
    except Exception as exc:
        print(f"blocked: {type(exc).__name__}: {exc}")
    """
)

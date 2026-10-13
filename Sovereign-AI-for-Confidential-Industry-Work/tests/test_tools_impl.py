"""Tool implementations: calc, fs.*, code.exec (§06).

calc must show its working - "calculations with steps shown" is a traced
requirement, and a number with no derivation is not checkable. In a plant
context an uncheckable number is worse than no answer.
"""

from __future__ import annotations

import pytest

from kavach.tools.impl.calc import HAVE_SYMPY, CalcError, calc, parse_substitutions
from kavach.tools.impl.fs import fs_list, fs_read, fs_write
from kavach.tools.impl.sandbox import (
    UNSANDBOXED_ENV,
    SandboxUnavailable,
    code_exec,
    docker_available,
    image_available,
)
from kavach.tools.workspace import Workspace, WorkspaceViolation


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_impl")


# ---------------------------------------------------------------- calc

def test_calc_returns_the_working_not_just_the_number():
    result = calc("rho * v * d / mu", "rho=998, v=1.2, d=0.1524, mu=0.001")
    assert result["result"] == pytest.approx(182514.24, rel=1e-6)
    assert len(result["steps"]) >= 3
    assert any("substitute" in s for s in result["steps"])


def test_calc_reynolds_number_is_correct():
    """998 * 1.2 * 0.1524 / 0.001 = 182514.24 - checked by hand."""
    assert calc("rho*v*d/mu", "rho=998,v=1.2,d=0.1524,mu=0.001")["result"] == \
        pytest.approx(182514.24, rel=1e-9)


def test_calc_handles_functions():
    result = calc("sqrt(2*g*h)", "g=9.81, h=12")
    assert result["result"] == pytest.approx(15.3440542, rel=1e-6)


def test_calc_without_substitutions_stays_symbolic():
    result = calc("a + b")
    assert "unresolved" in result or isinstance(result["result"], str)


@pytest.mark.skipif(not HAVE_SYMPY, reason="needs SymPy")
def test_calc_can_solve():
    result = calc("x**2 - 5*x + 6", solve_for="x")
    assert sorted(result["result"]) == ["2", "3"]


@pytest.mark.skipif(not HAVE_SYMPY, reason="needs SymPy")
def test_calc_reports_which_engine_ran():
    assert calc("1+1")["engine"] == "sympy"


def test_bad_substitution_is_rejected():
    with pytest.raises(CalcError, match="name=value"):
        calc("a+b", "a is 3")


def test_non_numeric_substitution_is_rejected():
    with pytest.raises(CalcError, match="not a number"):
        calc("a+b", "a=three")


def test_unparseable_expression_is_rejected():
    with pytest.raises(CalcError):
        calc("2 +* 3")


def test_parse_substitutions():
    assert parse_substitutions("a=1, b=2.5") == {"a": 1.0, "b": 2.5}
    assert parse_substitutions("") == {}
    assert parse_substitutions(None) == {}


def test_calc_does_not_execute_arbitrary_code():
    """calc takes a model-authored expression, so it must not be an eval()."""
    with pytest.raises((CalcError, Exception)):
        calc("__import__('os').system('echo pwned')")


# ---------------------------------------------------------------- fs

def test_write_then_read_roundtrip(ws):
    fs_write(ws, "notes.txt", "hello")
    assert fs_read(ws, "notes.txt")["content"] == "hello"


def test_write_creates_parent_directories(ws):
    fs_write(ws, "a/b/c.txt", "deep")
    assert fs_read(ws, "a/b/c.txt")["content"] == "deep"


def test_append_mode(ws):
    fs_write(ws, "log.txt", "one\n")
    fs_write(ws, "log.txt", "two\n", append=True)
    assert fs_read(ws, "log.txt")["content"] == "one\ntwo\n"


def test_read_reports_truncation(ws):
    fs_write(ws, "big.txt", "x" * 5000)
    result = fs_read(ws, "big.txt", max_bytes=100)
    assert result["truncated"] is True
    assert len(result["content"]) == 100
    assert result["size_bytes"] == 5000


def test_read_missing_file_raises(ws):
    with pytest.raises(FileNotFoundError):
        fs_read(ws, "nope.txt")


def test_read_a_directory_is_an_error(ws):
    (ws.root / "sub").mkdir()
    with pytest.raises(IsADirectoryError):
        fs_read(ws, "sub")


def test_list_returns_relative_paths(ws):
    fs_write(ws, "a.txt", "1")
    fs_write(ws, "sub/b.txt", "2")
    paths = {e["path"] for e in fs_list(ws)["entries"]}
    assert paths == {"a.txt", "sub/b.txt"}


def test_fs_tools_cannot_escape_the_workspace(ws):
    """The jail is enforced inside the tools too, not only at the executor."""
    for path in ("../escape.txt", "/etc/passwd", "a/../../escape.txt"):
        with pytest.raises(WorkspaceViolation):
            fs_write(ws, path, "x")
        with pytest.raises(WorkspaceViolation):
            fs_read(ws, path)


def test_read_never_leaks_host_paths(ws):
    fs_write(ws, "x.txt", "data")
    assert str(ws.root) not in fs_read(ws, "x.txt")["path"]


# ---------------------------------------------------------------- code.exec

SANDBOX_READY = docker_available() and image_available()


def test_refuses_without_a_sandbox(ws, monkeypatch):
    """A sandbox that silently stops sandboxing is worse than one that stops
    working, so with no Docker and no explicit opt-in this must refuse."""
    monkeypatch.delenv(UNSANDBOXED_ENV, raising=False)
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    with pytest.raises(SandboxUnavailable, match="needs the Docker sandbox"):
        code_exec(ws, "print(1)")


def test_refusal_explains_how_to_fix_it(ws, monkeypatch):
    monkeypatch.delenv(UNSANDBOXED_ENV, raising=False)
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    with pytest.raises(SandboxUnavailable) as exc:
        code_exec(ws, "print(1)")
    assert "docker compose build sandbox-runner" in str(exc.value)
    assert UNSANDBOXED_ENV in str(exc.value)


def test_opt_in_runs_but_is_labelled_unsandboxed(ws, monkeypatch):
    """The dev fallback must never be mistakable for the real thing."""
    monkeypatch.setenv(UNSANDBOXED_ENV, "1")
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    result = code_exec(ws, "print(2+2)")
    assert result["sandboxed"] is False
    assert result["runner"] == "subprocess"
    assert "NOT ISOLATED" in result["network"]
    assert "NOT SANDBOXED" in result["warning"]
    assert result["stdout"].strip() == "4"


def test_opt_in_captures_exit_code_and_stderr(ws, monkeypatch):
    monkeypatch.setenv(UNSANDBOXED_ENV, "1")
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    result = code_exec(ws, "import sys; sys.stderr.write('bad\\n'); sys.exit(3)")
    assert result["exit_code"] == 3
    assert "bad" in result["stderr"]


def test_opt_in_enforces_a_timeout(ws, monkeypatch):
    monkeypatch.setenv(UNSANDBOXED_ENV, "1")
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    result = code_exec(ws, "import time; time.sleep(30)", timeout_s=1)
    assert result["timed_out"] is True
    assert result["exit_code"] == 124


def test_input_files_are_still_jailed(ws, monkeypatch):
    monkeypatch.setenv(UNSANDBOXED_ENV, "1")
    monkeypatch.setattr("kavach.tools.impl.sandbox.docker_available", lambda: False)
    with pytest.raises(WorkspaceViolation):
        code_exec(ws, "print(1)", files=["../../secrets.txt"])


# ---- the real thing; skipped until Docker + the image are present ----

@pytest.mark.skipif(not SANDBOX_READY, reason="needs Docker and kavach/sandbox image")
def test_sandbox_has_no_network():
    """THE claim: generated code cannot exfiltrate even in principle."""
    import tempfile
    from pathlib import Path

    from kavach.tools.impl.sandbox import EGRESS_PROBE

    ws = Workspace(Path(tempfile.mkdtemp()) / "probe")
    result = code_exec(ws, EGRESS_PROBE)
    assert result["sandboxed"] is True
    assert result["network"] == "none"
    assert "REACHED THE NETWORK" not in result["stdout"]
    assert "SOCKET CONNECTED" not in result["stdout"]
    assert "blocked" in result["stdout"], f"exit={result.get('exit_code')}, stderr={result.get('stderr')}, stdout={result.get('stdout')}"


@pytest.mark.skipif(not SANDBOX_READY, reason="needs Docker and kavach/sandbox image")
def test_sandbox_runs_real_code():
    import tempfile
    from pathlib import Path

    ws = Workspace(Path(tempfile.mkdtemp()) / "run")
    result = code_exec(ws, "import numpy as np; print(int(np.sum([1,2,3])))")
    assert result["sandboxed"] is True
    assert result["stdout"].strip() == "6", f"exit={result.get('exit_code')}, stderr={result.get('stderr')}, stdout={result.get('stdout')}"

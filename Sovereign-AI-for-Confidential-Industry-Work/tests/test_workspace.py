"""Workspace jail.

These are adversarial on purpose. A model writes these paths, so they are hostile
input by definition - not because the model is malicious, but because a confused
7B model emitting `../../../../etc/passwd` produces the same bytes an attacker
would.

If any test in this file starts failing, treat it as a security incident in the
build, not a flaky test.
"""

from __future__ import annotations

import os

import pytest

from kavach.tools.workspace import Workspace, WorkspaceViolation


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "run_abc")


@pytest.fixture()
def outside(tmp_path):
    target = tmp_path / "secrets.txt"
    target.write_text("do not read me", encoding="utf-8")
    return target


# ---------------------------------------------------------------- happy path

def test_simple_relative_path(ws):
    resolved = ws.resolve("notes.txt")
    assert resolved.parent == ws.root


def test_nested_path(ws):
    resolved = ws.resolve("a/b/c.txt")
    assert ws.relative(resolved) == "a/b/c.txt"


def test_dot_prefix_is_fine(ws):
    assert ws.relative(ws.resolve("./notes.txt")) == "notes.txt"


def test_internal_dotdot_that_stays_inside_is_allowed(ws):
    """`a/../b` is inside the workspace and there is no reason to refuse it."""
    assert ws.relative(ws.resolve("a/../b.txt")) == "b.txt"


def test_resolve_does_not_require_existence(ws):
    """fs.write has to be able to name a file that is not there yet."""
    assert ws.resolve("new/deep/file.txt")


def test_must_exist_raises_for_missing(ws):
    with pytest.raises(FileNotFoundError):
        ws.resolve("nope.txt", must_exist=True)


# ---------------------------------------------------------------- traversal

@pytest.mark.parametrize("path", [
    "../escape.txt",
    "../../escape.txt",
    "../../../../../../etc/passwd",
    "a/../../escape.txt",
    "a/b/../../../escape.txt",
    "./../../escape.txt",
])
def test_dotdot_traversal_is_refused(ws, path):
    with pytest.raises(WorkspaceViolation):
        ws.resolve(path)


def test_traversal_that_lands_exactly_on_the_parent_is_refused(ws):
    with pytest.raises(WorkspaceViolation):
        ws.resolve("..")


# ---------------------------------------------------------------- absolute

@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "/tmp/whatever",
    "\\windows\\system32\\config",
])
def test_absolute_paths_are_refused(ws, path):
    with pytest.raises(WorkspaceViolation):
        ws.resolve(path)


def test_drive_qualified_path_is_refused(ws):
    with pytest.raises(WorkspaceViolation):
        ws.resolve("C:\\Windows\\System32\\drivers\\etc\\hosts")


def test_drive_relative_path_is_refused(ws):
    """`C:file` is drive-relative on Windows, not absolute - a naive
    is_absolute() check misses it."""
    with pytest.raises(WorkspaceViolation):
        ws.resolve("C:file.txt")


def test_the_workspace_root_itself_by_absolute_path_is_refused(ws):
    """Even a path pointing at somewhere legitimate must be given relatively."""
    with pytest.raises(WorkspaceViolation):
        ws.resolve(str(ws.root / "notes.txt"))


# ---------------------------------------------------------------- symlinks

def test_containment_check_rejects_a_resolved_path_outside_root(ws, outside):
    """Covers the branch the symlink tests exercise, without needing to create a
    symlink - unprivileged Windows cannot make one, so those tests skip there and
    the escape path would otherwise be entirely unverified on this machine.

    This asserts the underlying rule directly: a fully-resolved path outside the
    root is refused, however it came to point there.
    """
    assert ws._contains(outside.resolve()) is False
    assert ws._contains((ws.root / "inside.txt").resolve()) is True


def test_containment_rejects_a_sibling_workspace(tmp_path):
    """Prefix-matching bugs show up here: `/runs/run_a` must not contain
    `/runs/run_ab`, even though one is a string prefix of the other."""
    a = Workspace(tmp_path / "run_a")
    sibling = (tmp_path / "run_ab" / "file.txt").resolve()
    assert a._contains(sibling) is False


def _symlinks_work() -> bool:
    """Ask the OS, rather than assuming from the platform name.

    The gate used to skip on Windows unless KAVACH_TEST_SYMLINKS was set, on the
    theory that only an elevated shell can create one. That is no longer true:
    Developer Mode grants SeCreateSymbolicLinkPrivilege to ordinary users, and
    on such a box these tests were being skipped while passing perfectly well -
    a coverage gap nobody would notice, in the tests that prove the workspace
    jail cannot be escaped by a link. Probing costs one temporary file and turns
    "skipped by policy" into "skipped only where genuinely impossible".
    """
    if os.name != "nt":
        return True
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "target")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("x")
        try:
            os.symlink(target, os.path.join(td, "link"))
            return True
        except (OSError, NotImplementedError):
            return False


SYMLINKS_WORK = _symlinks_work()

symlinks_needed = pytest.mark.skipif(
    not SYMLINKS_WORK,
    reason=(
        "this account cannot create symlinks (WinError 1314) - enable Developer "
        "Mode or use an elevated shell. COVERAGE GAP on this machine: these are "
        "the tests that prove a link cannot escape the workspace jail."
    ),
)


@symlinks_needed
def test_symlink_pointing_out_is_refused(ws, outside):
    """The escape with no `..` in it: create a link inside the workspace that
    points out, then read through it."""
    link = ws.root / "innocent"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceViolation):
        ws.resolve("innocent")


@symlinks_needed
def test_writing_through_a_symlinked_directory_is_refused(ws, tmp_path):
    target_dir = tmp_path / "elsewhere"
    target_dir.mkdir()
    (ws.root / "link").symlink_to(target_dir, target_is_directory=True)
    with pytest.raises(WorkspaceViolation):
        ws.resolve("link/payload.txt")


@symlinks_needed
def test_symlink_staying_inside_is_allowed(ws):
    (ws.root / "real").mkdir()
    (ws.root / "alias").symlink_to(ws.root / "real", target_is_directory=True)
    assert ws.resolve("alias/file.txt")


# ---------------------------------------------------------------- malformed

def test_null_byte_is_refused(ws):
    with pytest.raises(WorkspaceViolation, match="null byte"):
        ws.resolve("notes\x00.txt")


@pytest.mark.parametrize("path", ["", "   ", "\t"])
def test_empty_path_is_refused(ws, path):
    with pytest.raises(WorkspaceViolation):
        ws.resolve(path)


# ---------------------------------------------------------------- contains

def test_contains_is_non_raising(ws):
    assert ws.contains("fine.txt") is True
    assert ws.contains("../escape.txt") is False
    assert ws.contains("/etc/passwd") is False


# ---------------------------------------------------------------- listing

def test_iter_files_finds_nested_files(ws):
    (ws.root / "sub").mkdir()
    (ws.root / "sub" / "a.txt").write_text("a", encoding="utf-8")
    (ws.root / "b.txt").write_text("b", encoding="utf-8")
    names = {ws.relative(p) for p in ws.iter_files()}
    assert names == {"sub/a.txt", "b.txt"}


def test_relative_never_leaks_host_paths(ws):
    """Absolute host paths must not reach model context or the UI."""
    resolved = ws.resolve("deep/file.txt")
    rendered = ws.relative(resolved)
    assert rendered == "deep/file.txt"
    assert str(ws.root) not in rendered


def test_two_runs_are_isolated(tmp_path):
    a = Workspace.for_run(tmp_path, "run_a")
    b = Workspace.for_run(tmp_path, "run_b")
    assert a.root != b.root
    (a.root / "secret.txt").write_text("a's data", encoding="utf-8")
    with pytest.raises(WorkspaceViolation):
        b.resolve("../run_a/secret.txt")


def test_usage_bytes(ws):
    (ws.root / "f.txt").write_text("x" * 100, encoding="utf-8")
    assert ws.usage_bytes() == 100

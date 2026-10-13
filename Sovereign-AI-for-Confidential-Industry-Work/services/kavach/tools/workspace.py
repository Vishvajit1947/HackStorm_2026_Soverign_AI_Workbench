"""Workspace jail.

Every filesystem path a tool touches passes through here. A model writes the path,
so it is hostile input by definition - not because the model is malicious, but
because a confused 7B model that emits `../../../../etc/passwd` produces exactly
the same bytes as an attacker would.

The containment rule is: resolve the path fully (following symlinks, normalising
`..`, expanding the drive letter on Windows), then check the RESOLVED path is
inside the RESOLVED workspace root. Checking before resolution is the classic
mistake - `workspace/safe/../../etc` passes a naive prefix test and lands outside.

Symlinks matter even inside the workspace: a tool that creates
`workspace/link -> /etc` and then writes `workspace/link/passwd` has escaped
without a single `..` in sight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class WorkspaceViolation(PermissionError):
    """A path resolved outside its workspace. Never downgrade this to a warning."""


@dataclass
class Workspace:
    """A run's sandbox on the host filesystem."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- resolve

    def resolve(self, relative: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
        """Resolve a workspace-relative path, or raise WorkspaceViolation.

        Rejects, in order: null bytes, absolute paths, drive-qualified paths, and
        anything whose fully-resolved form lands outside the root.
        """
        raw = os.fspath(relative)

        if "\x00" in raw:
            raise WorkspaceViolation("path contains a null byte")

        if not raw.strip():
            raise WorkspaceViolation("empty path")

        candidate = Path(raw)

        # An absolute path is never workspace-relative, whatever it points at.
        # `Path("/etc").is_absolute()` is True on POSIX; on Windows both "C:\x"
        # and "\x" need catching, and drive-relative "C:x" too.
        if candidate.is_absolute() or candidate.drive or raw.startswith(("/", "\\")):
            raise WorkspaceViolation(
                f"{raw!r} is an absolute path; tools may only use workspace-relative paths"
            )

        # strict=False so we can resolve paths that do not exist yet (fs.write),
        # while still normalising `..` and following any symlinks that DO exist
        # along the way.
        resolved = (self.root / candidate).resolve(strict=False)

        if not self._contains(resolved):
            raise WorkspaceViolation(
                f"{raw!r} resolves to {resolved}, which is outside the workspace "
                f"({self.root})"
            )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"{raw} does not exist in this workspace")

        return resolved

    def _contains(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return False
        return True

    def contains(self, path: str | os.PathLike[str]) -> bool:
        """Non-raising containment test, for the policy engine's
        `writes_outside_workspace` question."""
        try:
            self.resolve(path)
        except (WorkspaceViolation, OSError):
            return False
        return True

    # ---------------------------------------------------------------- listing

    def relative(self, resolved: Path) -> str:
        """Workspace-relative form, for display and for the audit record. Never
        leak the host's absolute paths into model context or the UI."""
        return resolved.relative_to(self.root).as_posix()

    def iter_files(self, subdir: str = ".") -> list[Path]:
        base = self.resolve(subdir, must_exist=True)
        if not base.is_dir():
            return [base]
        found: list[Path] = []
        for path in sorted(base.rglob("*")):
            # rglob follows the tree we own; re-check each hit in case something
            # created a symlink out of the workspace between calls.
            if path.is_file() and self._contains(path.resolve(strict=False)):
                found.append(path)
        return found

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def for_run(cls, var_dir: Path, run_id: str) -> Workspace:
        return cls(Path(var_dir) / "workspaces" / run_id)

    def usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

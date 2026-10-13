"""Workspace-jailed file tools (§06).

Every path goes through Workspace.resolve(), which is the only place containment
is decided. These functions never touch a path directly.
"""

from __future__ import annotations

from typing import Any

from ..workspace import Workspace

DEFAULT_MAX_BYTES = 200_000
MAX_LIST_ENTRIES = 500


def fs_read(
    workspace: Workspace, path: str, max_bytes: int = DEFAULT_MAX_BYTES, **_: Any
) -> dict[str, Any]:
    resolved = workspace.resolve(path, must_exist=True)
    if resolved.is_dir():
        raise IsADirectoryError(f"{path} is a directory; use fs.list")

    size = resolved.stat().st_size
    limit = max(1, min(int(max_bytes), DEFAULT_MAX_BYTES))
    data = resolved.read_bytes()[:limit]

    # Model context is expensive and a mangled decode is worse than a marked one.
    text = data.decode("utf-8", errors="replace")

    return {
        "path": workspace.relative(resolved),
        "size_bytes": size,
        "truncated": size > limit,
        "content": text,
    }


def fs_list(workspace: Workspace, subdir: str = ".", **_: Any) -> dict[str, Any]:
    files = workspace.iter_files(subdir)
    entries = [
        {"path": workspace.relative(p), "size_bytes": p.stat().st_size}
        for p in files[:MAX_LIST_ENTRIES]
    ]
    return {
        "subdir": subdir,
        "count": len(files),
        "truncated": len(files) > MAX_LIST_ENTRIES,
        "entries": entries,
    }


def fs_write(
    workspace: Workspace,
    path: str,
    content: str,
    append: bool = False,
    **_: Any,
) -> dict[str, Any]:
    resolved = workspace.resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with resolved.open(mode, encoding="utf-8", newline="\n") as fh:
        fh.write(content)

    return {
        "path": workspace.relative(resolved),
        "size_bytes": resolved.stat().st_size,
        "appended": bool(append),
    }

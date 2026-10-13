"""Tool handlers.

HANDLERS maps a manifest name to its implementation. A manifest with no entry
here loads as `unbound` and invoking it is a clear error rather than a mystery -
which is what you want while a tool is half-built.

Handlers receive validated arguments plus, where the signature asks for them,
`workspace`, `sandbox` and `timeout_s` injected by the executor.

Note the module holding code.exec is `sandbox.py`, not `code_exec.py`: re-exporting
a function called `code_exec` from a module of the same name shadows the module in
the package namespace, so `kavach.tools.impl.code_exec` resolves to the function
and anything addressing the module by path (monkeypatch, importlib) breaks.
"""

from __future__ import annotations

from .calc import calc
from .deliverables import doc_write
from .engineering import engineering_analyze
from .fs import fs_list, fs_read, fs_write
from .kb import kb_search
from .sandbox import code_exec

HANDLERS = {
    "calc": calc,
    "doc.write": doc_write,
    "code.exec": code_exec,
    "fs.read": fs_read,
    "fs.list": fs_list,
    "fs.write": fs_write,
    "kb.search": kb_search,
    "engineering.analyze": engineering_analyze,
}

__all__ = [
    "HANDLERS",
    "calc",
    "code_exec",
    "doc_write",
    "engineering_analyze",
    "fs_list",
    "fs_read",
    "fs_write",
    "kb_search",
]

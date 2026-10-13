"""L4 - the tool plane (§06).

Tools are declared in manifests, not hard-coded. Every invocation passes through
ToolExecutor, which is the single place policy, the workspace jail, approval
gating and the audit record are applied.
"""

from .executor import (
    ApprovalRequired,
    PolicyDenied,
    ToolExecutor,
    ToolInvocation,
    ToolResult,
    get_executor,
    reset_executor,
)
from .grammar import schema_to_gbnf, tool_call_grammar, tool_choice_grammar
from .registry import (
    InvalidArguments,
    SandboxSpec,
    ToolError,
    ToolNotFound,
    ToolRegistry,
    ToolSpec,
    get_tool_registry,
    reset_tool_registry,
)
from .workspace import Workspace, WorkspaceViolation

__all__ = [
    "ApprovalRequired",
    "InvalidArguments",
    "PolicyDenied",
    "SandboxSpec",
    "ToolError",
    "ToolExecutor",
    "ToolInvocation",
    "ToolNotFound",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Workspace",
    "WorkspaceViolation",
    "get_executor",
    "get_tool_registry",
    "reset_executor",
    "reset_tool_registry",
    "schema_to_gbnf",
    "tool_call_grammar",
    "tool_choice_grammar",
]

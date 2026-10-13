"""L4 endpoints - the tool plane (§06)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..tools import (
    ApprovalRequired,
    InvalidArguments,
    PolicyDenied,
    ToolNotFound,
    Workspace,
    get_executor,
    get_tool_registry,
    tool_call_grammar,
)
from ..tools.impl.languages import LANGUAGES as LANGUAGE_TABLE
from ..tools.impl.languages import summary as language_summary
from ..tools.impl.sandbox import (
    EGRESS_PROBE,
    SANDBOX_IMAGE,
    available_languages,
    docker_available,
    image_available,
)
from ..types import Label

router = APIRouter(prefix="/api/tools", tags=["tools"])


class InvokePayload(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    run_id: str = "adhoc"
    label: Label = Label.INTERNAL
    role: str = "engineer"
    approved_by: str | None = None


def _workspace(run_id: str) -> Workspace:
    return Workspace.for_run(get_settings().var_dir, run_id)


@router.get("")
def list_tools() -> dict[str, Any]:
    """Every declared tool with its schema, side effect and gate."""
    return get_tool_registry().summary()


@router.get("/sandbox")
def sandbox_status() -> dict[str, Any]:
    """Whether code.exec can actually isolate anything right now."""
    docker = docker_available()
    image = image_available() if docker else False
    # Which languages this image can actually run. None for a language means the
    # probe could not say - reported as unknown rather than folded into "no",
    # because a failed probe and a missing compiler are different facts.
    languages = available_languages() if docker and image else {}
    return {
        "docker_available": docker,
        "image": SANDBOX_IMAGE,
        "image_available": image,
        "ready": docker and image,
        "languages": [
            {
                **entry,
                "available": languages.get(str(entry["id"])),
                # Why not, when it is a language the image carries and still
                # cannot run. Blank for every language that works.
                "unavailable_because": LANGUAGE_TABLE[str(entry["id"])].unsupported,
            }
            for entry in language_summary()
        ],
        "detail": (
            "sandbox ready - code.exec runs with no network interface"
            if docker and image
            else "code.exec will refuse to run; build with: "
            "docker compose build sandbox-runner"
        ),
    }


@router.post("/reload")
def reload_tools() -> dict[str, Any]:
    registry = get_tool_registry()
    before = set(registry.names())
    count = registry.reload()
    after = set(registry.names())
    return {
        "count": count,
        "added": sorted(after - before),
        "removed": sorted(before - after),
        "unbound": registry.unbound(),
    }


@router.get("/{tool_name}/grammar")
def grammar(tool_name: str) -> dict[str, Any]:
    """The GBNF grammar that makes malformed calls to this tool unreachable (§05)."""
    try:
        spec = get_tool_registry().get(tool_name)
    except ToolNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "tool": spec.name,
        "grammar": tool_call_grammar(spec.name, spec.schema),
        "note": (
            "Handed to llama.cpp as a decoding constraint. Output that does not "
            "match is not unlikely - it is unreachable."
        ),
    }


@router.post("/invoke")
def invoke(payload: InvokePayload) -> dict[str, Any]:
    executor = get_executor()
    try:
        result = executor.invoke(
            payload.tool,
            payload.args,
            _workspace(payload.run_id),
            label=payload.label,
            role=payload.role,
            run_id=payload.run_id,
            approved_by=payload.approved_by,
        )
    except ToolNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidArguments as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PolicyDenied as exc:
        # 403: the policy engine refused. This is the visible refusal, not a bug.
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ApprovalRequired as exc:
        # 428 Precondition Required: legitimate, but a human must sign it off.
        raise HTTPException(
            status_code=428,
            detail={
                "error": str(exc),
                "tool": exc.invocation.tool,
                "args_sha256": exc.invocation.args_sha256,
                "hint": "re-send with approved_by set once a human has agreed",
            },
        ) from exc

    return result.as_dict()


@router.post("/sandbox/probe")
def sandbox_probe(run_id: str = "adhoc") -> dict[str, Any]:
    """Run code inside the sandbox that deliberately tries to reach the network.

    The tool-plane counterpart of the L0 canary: rather than asserting that
    generated code cannot exfiltrate, run some that tries and show it failing
    with "Network is unreachable" (§06).
    """
    result = get_executor().invoke(
        "code.exec", {"code": EGRESS_PROBE}, _workspace(run_id), run_id=run_id
    )
    data = result.data
    stdout = data.get("stdout", "")
    reached = "REACHED THE NETWORK" in stdout or "SOCKET CONNECTED" in stdout

    return {
        "ok": result.ok and not reached,
        "sandboxed": data.get("sandboxed"),
        "reached_network": reached,
        "stdout": stdout,
        "stderr": data.get("stderr", ""),
        "headline": (
            "SANDBOX BREACH - generated code reached the network"
            if reached
            else "blocked: the sandbox has no network interface"
            if data.get("sandboxed")
            else "NOT SANDBOXED - this ran on the host; the result proves nothing"
        ),
    }

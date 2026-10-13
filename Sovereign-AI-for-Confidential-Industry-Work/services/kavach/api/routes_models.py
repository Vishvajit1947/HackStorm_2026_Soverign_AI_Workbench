"""L1 endpoints: the registry, the warden and the extensibility beat (§09, 10:45).

Request models live at module scope - see the note in app.py about
`from __future__ import annotations` and FastAPI's annotation resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..audit import get_audit
from ..config import get_settings
from ..runtime import (
    DigestMismatch,
    VramExhausted,
    get_backend,
    get_registry,
    get_warden,
)
from ..runtime.registry import sha256_file
from ..runtime.staging import (
    StagingError,
    add_to_profile,
    append_model,
    inspect_gguf,
    propose_entry,
    propose_id,
    unregistered_files,
)

router = APIRouter(prefix="/api/models", tags=["models"])


class LoadRequest(BaseModel):
    model_id: str = Field(description="registry id to admit into VRAM")


class EvictRequest(BaseModel):
    include_pinned: bool = False


@router.get("")
def list_models() -> dict[str, Any]:
    """Every registered model with its provenance and per-class quality."""
    registry = get_registry()
    warden = get_warden()
    summary = registry.summary()
    for entry in summary["models"]:
        entry["resident"] = warden.is_resident(entry["id"])
        entry["swap_cost_ms"] = warden.swap_cost(entry["id"])
        entry["fits"] = warden.fits(entry["id"])
    return summary


@router.get("/residency")
def residency() -> dict[str, Any]:
    """The live VRAM meter: what is loaded, what is pinned, what is free."""
    return get_warden().state()


@router.get("/backend")
def backend_health() -> dict[str, Any]:
    return get_backend().health()


@router.post("/reload")
def reload_registry() -> dict[str, Any]:
    """Re-read models.yaml and re-hash every weight file, live.

    This is the extensibility beat: paste a new manifest entry, hit reload, and it
    is a routing candidate on the next request with no restart and no code change.
    """
    registry = get_registry()
    before = {s.id for s in registry.all()}
    try:
        summary = registry.reload()
    except DigestMismatch as exc:
        get_audit().append("model.digest_mismatch", detail=str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    after = {s.id for s in registry.all()}
    added, removed = sorted(after - before), sorted(before - after)

    get_audit().append(
        "model.registry_reload",
        added=added,
        removed=removed,
        count=summary["count"],
    )
    return {"added": added, "removed": removed, **summary}


@router.post("/load")
def load_model(req: LoadRequest) -> dict[str, Any]:
    """Admit a model into VRAM, evicting LRU if the card is full."""
    warden = get_warden()
    try:
        decision = warden.admit(req.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except VramExhausted as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    return {"decision": decision.as_dict(), "residency": warden.state()}


@router.post("/evict")
def evict(req: EvictRequest) -> dict[str, Any]:
    warden = get_warden()
    evicted = warden.evict_all(include_pinned=req.include_pinned)
    get_audit().append("model.evict_all", evicted=evicted, include_pinned=req.include_pinned)
    return {"evicted": evicted, "residency": warden.state()}


@router.get("/verify")
def verify_provenance() -> dict[str, Any]:
    """Re-hash every staged weight file and report what is real.

    Answers the auditor's actual question: how do you know the weights on this box
    are the weights you approved?
    """
    registry = get_registry()
    registry.reload()
    specs = registry.all()

    real = [s.id for s in specs if s.provenance.is_real]
    simulated = [s.id for s in specs if s.provenance.value == "SIMULATED"]
    unverified = [s.id for s in specs if s.provenance.value == "UNVERIFIED"]

    get_audit().append(
        "model.verify",
        verified=len(real),
        simulated=len(simulated),
        unverified=len(unverified),
    )

    return {
        "ok": not simulated and not unverified,
        "verified": real,
        "simulated": simulated,
        "unverified": unverified,
        "detail": (
            "every registered weight file hashes to its approved digest"
            if not simulated and not unverified
            else "some models have no approved digest on this box - not demo-ready"
        ),
    }


# ------------------------------------------------------- adding a model (L1)

# The brief asks that new open-weight models be addable "without redesigning the
# system". These three endpoints are that path: look at a file, register it, or
# see what is sitting in the models folder unregistered.
#
# ADMIN ONLY, AND HONEST ABOUT WHAT THAT MEANS.
#
# Staging a model is the most consequential thing this gateway can be asked to
# do: it puts a file the router may later run in front of confidential data. So
# it is refused for every role but admin, and both the refusal and the staging
# are written to the audit chain with the role that was claimed.
#
# The role is CLAIMED, not authenticated. There is no login on this box, the
# gateway binds loopback, and the person at the keyboard is the person at the
# keyboard - a password field here would be theatre, and theatre is worse than
# an honest gate because it invites someone to rely on it. What the check
# actually buys is that the action cannot happen by accident from a panel
# somebody opened as an operator, and that the attempt is on the record either
# way.


class StageRequest(BaseModel):
    file: str = Field(description="a .gguf file already in the models folder")
    role: str = Field(default="engineer", description="the role claiming this action")
    model_id: str | None = None
    license: str = "unknown"
    add_to_profile: bool = True


def _require_admin(role: str, action: str, detail: str) -> None:
    if role == "admin":
        return
    get_audit().append(
        "model.stage_denied", actor=role, action=action, detail=detail, role=role,
    )
    raise HTTPException(
        status_code=403,
        detail=(
            f"only an administrator can {action}. You are signed in as "
            f"{role!r} - change the role in the top bar if that is your job on "
            "this box. The attempt has been recorded."
        ),
    )


@router.get("/unregistered")
def unregistered() -> dict[str, Any]:
    """.gguf files present on this machine that no manifest entry points at.

    Readable by any role: knowing a file is sitting there unused is not a
    privileged fact, and hiding it would mean an operator cannot even tell an
    administrator what needs doing.
    """
    settings = get_settings()
    registry = get_registry()
    known = {spec.file for spec in registry.all()}
    known |= {spec.mmproj for spec in registry.all() if spec.mmproj}
    return {
        "model_dir": str(settings.model_dir),
        "files": unregistered_files(settings.model_dir, known),
    }


@router.post("/inspect")
def inspect(payload: StageRequest) -> dict[str, Any]:
    """Read a file's GGUF header and show what registering it would produce.

    A dry run, deliberately separate from the write: an operator sees the
    architecture, the context length, the estimated VRAM and every warning
    BEFORE anything is added to a config file that is under version control.
    """
    settings = get_settings()
    path = _resolve_model_file(settings.model_dir, payload.file)
    try:
        info = inspect_gguf(path)
    except StagingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    taken = {spec.id for spec in get_registry().all()}
    model_id = payload.model_id or propose_id(path, taken)
    entry = propose_entry(info, model_id, license_name=payload.license)
    return {
        "gguf": info.as_dict(),
        "proposed": entry,
        "id_is_free": model_id not in taken,
        "profile": settings.profile.name,
        "note": (
            "VRAM and load time are ESTIMATED from the file size, and the "
            "quality scores are seeds, not measurements. Both are written into "
            "models.yaml marked as such."
        ),
    }


@router.post("/stage")
def stage(payload: StageRequest) -> dict[str, Any]:
    """Register a file that is already in the models folder. Admin only.

    Writes BOTH gates - the manifest entry and the profile's `available:` list -
    then reloads the registry, so the model is a routing candidate on the next
    request with no restart and no code change.
    """
    _require_admin(payload.role, "add a model", payload.file)

    settings = get_settings()
    path = _resolve_model_file(settings.model_dir, payload.file)
    registry = get_registry()

    try:
        info = inspect_gguf(path)
    except StagingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The guard is on the FILE as well as the id.
    #
    # `propose_id` avoids a collision by picking the next free name, which is
    # right for two different models with similar filenames and wrong for the
    # same file staged twice: the second attempt quietly produced a second
    # entry pointing at identical weights, with its own seeded quality scores,
    # and the router then had two "models" that were one model.
    already = [spec for spec in registry.all() if spec.file == path.name]
    if already:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{path.name} is already registered as {already[0].id!r}. "
                "Delete that entry from config/models.yaml first if you meant "
                "to re-add it."
            ),
        )

    taken = {spec.id for spec in registry.all()}
    model_id = payload.model_id or propose_id(path, taken)
    if model_id in taken:
        raise HTTPException(
            status_code=409,
            detail=f"a model with id {model_id!r} is already registered",
        )

    digest = sha256_file(path)
    entry = propose_entry(info, model_id, license_name=payload.license,
                          sha256=digest)
    notes = [
        f"Staged from {path.name} through the workbench.",
        "vram_gb and load_ms are ESTIMATED from the file size, not measured.",
        "quality[] are SEEDS, not measurements - run scripts/eval_harness.py",
        "  --quality --write against this model before trusting the router's",
        "  ranking of it.",
    ]
    if info.warnings:
        notes.extend(f"WARNING: {w}" for w in info.warnings)

    manifest = settings.config_dir / "models.yaml"
    profiles = settings.config_dir / "profiles.yaml"
    try:
        append_model(manifest, entry, notes)
        added_to_profile = (
            add_to_profile(profiles, settings.profile.name, model_id)
            if payload.add_to_profile else False
        )
    except StagingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        summary = registry.reload()
    except DigestMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    get_audit().append(
        "model.staged",
        actor=payload.role, role=payload.role, model=model_id,
        file=path.name, sha256=digest, architecture=info.architecture,
        profile=settings.profile.name if added_to_profile else None,
        warnings=info.warnings,
    )

    spec = registry.get(model_id)
    return {
        "id": model_id,
        "file": path.name,
        "sha256": digest,
        "provenance": spec.provenance.value if spec else "UNKNOWN",
        "added_to_profile": settings.profile.name if added_to_profile else None,
        "routable": bool(spec and spec.routable),
        "gguf": info.as_dict(),
        "registry_count": summary["count"],
        # Never silent about the half that is still a human's job.
        "next_steps": [
            "The model is registered and the router can pick it now.",
            "Its VRAM and speed are estimates; its quality scores are seeds.",
            ("Run scripts/eval_harness.py --quality --write to replace the "
             "seeds with measured values before relying on the router's choice."),
        ] + ([f"WARNING: {w}" for w in info.warnings] if info.warnings else []),
    }


@router.post("/upload")
async def upload(
    # B008 is FastAPI's own idiom: the parameter declaration IS the multipart
    # parser. Moving these into the body would mean parsing the form by hand.
    file: UploadFile = File(...),                     # noqa: B008
    role: str = Form("engineer"),
    license: str = Form("unknown"),
    model_id: str | None = Form(None),
) -> dict[str, Any]:
    """Receive a .gguf over the browser and stage it. Admin only.

    STREAMED TO DISK IN CHUNKS, never read whole: a 7B q4 file is about 4.5 GB
    and reading it into memory to write it out again would take the gateway
    down on the 12 GB box this is built for.

    The honest caveat, which the panel repeats: copying the file into the models
    folder with the operating system and pressing Scan is faster and far less
    fragile than pushing gigabytes through a browser tab. This exists because
    "drag the file onto the page" is what people expect to be able to do, and a
    workbench that cannot do the expected thing feels broken even when the
    documented alternative is better.
    """
    _require_admin(role, "add a model", file.filename or "")

    name = Path(file.filename or "").name
    if not name.lower().endswith(".gguf"):
        raise HTTPException(
            status_code=422,
            detail=f"{name!r} is not a .gguf file. This box loads GGUF weights.",
        )

    settings = get_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    target = settings.model_dir / name
    if target.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"{name} is already in the models folder. Delete it first, or "
                "use Scan to register the copy that is already there."
            ),
        )

    # A partial file is written under a temporary name and only moved into
    # place when the stream completes. An interrupted upload must not leave
    # something in the models folder that looks like a staged model.
    partial = target.with_suffix(target.suffix + ".part")
    written = 0
    try:
        with partial.open("wb") as handle:
            while chunk := await file.read(8 * 1024 * 1024):
                handle.write(chunk)
                written += len(chunk)
        partial.replace(target)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"the upload failed after {written} bytes: {exc}"
        ) from exc

    get_audit().append(
        "model.uploaded", actor=role, role=role, file=name, size_bytes=written,
    )
    return stage(StageRequest(file=name, role=role, license=license,
                              model_id=model_id))


def _resolve_model_file(model_dir: Path, name: str) -> Path:
    """A file inside the models folder, and nowhere else.

    The name arrives from a browser, so it is treated the way every other
    caller-supplied path in this build is: the basename only, resolved under
    the one directory it is allowed to be in, and checked afterwards. `../` is
    not a way to register /etc/passwd as a language model.
    """
    candidate = (model_dir / Path(name).name).resolve()
    try:
        candidate.relative_to(model_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"{name!r} is not inside the models folder"
        ) from exc
    if not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"{candidate.name} is not in the models folder "
                f"({model_dir}). Copy it there and try again."
            ),
        )
    return candidate

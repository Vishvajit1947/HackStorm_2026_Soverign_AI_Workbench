"""L3 endpoints - the knowledge plane (§07).

`GET /api/kb` is what the corpus panel polls: how many documents, how each page
was read, and - prominently - what could not be read at all. A corpus panel that
only shows what worked is a corpus panel that cannot tell you why an answer is
missing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ..audit import get_audit
from ..config import get_settings
from ..knowledge import (
    ImageReader,
    IngestError,
    get_answerer,
    get_index,
    get_ingest_jobs,
    get_ingestor,
    get_knowledge_store,
    get_retriever,
    reload_knowledge_config,
)
from ..types import Label

router = APIRouter(prefix="/api/kb", tags=["knowledge"])


class IngestPayload(BaseModel):
    # A server-side path: the box is air-gapped and single-operator, the corpus
    # arrives on it physically, and every ingest is recorded in the audit chain
    # with the file's sha256. There is no upload path because there is nowhere
    # for an upload to come from.
    path: str = Field(min_length=1)
    label: Label | None = None
    force: bool = False


class SearchPayload(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    role: str = "engineer"


class AnswerPayload(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    role: str = "engineer"
    max_tokens: int = Field(default=700, ge=64, le=4096)


@router.get("")
def corpus() -> dict[str, Any]:
    """Everything the corpus panel renders, in one poll."""
    return {
        "corpus": get_knowledge_store().stats(),
        "index": get_index().state().as_dict(),
        "image_path": ImageReader().status().as_dict(),
    }


@router.get("/doctor")
def doctor() -> dict[str, Any]:
    """What the knowledge plane can and cannot prove on this box right now."""
    stats = get_knowledge_store().stats()
    index = get_index().state()
    image = ImageReader().status()

    blockers: list[str] = []
    if not index.lexical_ready and stats["chunks"]:
        blockers.append("the lexical arm is off")
    if not index.dense_ready and stats["chunks"]:
        blockers.append("the dense arm is off - retrieval is lexical only")
    if index.stale:
        blockers.append("the index is stale")
    if stats["unreadable_pages"]:
        blockers.append(
            f"{stats['unreadable_pages']} page(s) could not be read: {image.detail}"
        )

    return {
        "ok": not blockers,
        "blockers": blockers,
        "corpus": stats,
        "index": index.as_dict(),
        "image_path": image.as_dict(),
    }


@router.get("/documents")
def documents() -> dict[str, Any]:
    docs = get_knowledge_store().documents()
    return {"count": len(docs), "documents": [d.as_dict() for d in docs]}


@router.get("/documents/{doc_id}")
def document(doc_id: str) -> dict[str, Any]:
    record = get_knowledge_store().document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
    payload = record.as_dict()
    payload["page_detail"] = [p.as_dict() for p in record.pages]
    return payload


@router.delete("/documents/{doc_id}")
def forget(doc_id: str) -> dict[str, Any]:
    if not get_ingestor().forget(doc_id):
        raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
    return {"forgotten": doc_id, "index": get_index().state().as_dict()}


@router.post("/ingest")
def ingest(payload: IngestPayload) -> dict[str, Any]:
    target = Path(payload.path).expanduser()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"{target} does not exist")
    try:
        report = get_ingestor().ingest(
            target, label=payload.label, force=payload.force
        )
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.as_dict()


# Extensions the ingest fork can actually read. Anything else would be accepted,
# stored, and then reported as an unreadable page - a worse experience than being
# told up front which formats this box handles.
_INGESTABLE = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# 64 MB. A scanned P&ID is a few MB; something far larger is a mistake, and an
# unbounded upload is a way to fill the disk the audit chain lives on.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

# Module-level so the File() call is not evaluated in a default argument.
_UPLOAD = File(...)


@router.post("/upload")
async def upload(
    file: UploadFile = _UPLOAD,
    label: Label | None = None,
    force: bool = False,
    background: bool = False,
) -> dict[str, Any]:
    """Take a document from the operator's browser into the corpus.

    The manifest for /ingest says there is no upload path "because there is
    nowhere for an upload to come from". That is true of a remote uploader and
    the reason still stands - but the gateway binds loopback, so the only
    browser that can reach this is one on this machine, driven by the operator
    sitting at it. Carrying a file across the room on a USB stick and typing a
    server-side path is the same trust boundary with more friction.

    The name is reduced to a bare filename before it is used: an upload is
    hostile input by definition, and "../../config/models.yaml" must land as
    "models.yaml" in the inbox rather than anywhere it was aiming.
    """
    raw_name = Path(file.filename or "").name          # strips any directory part
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).lstrip(".")
    if not safe:
        raise HTTPException(status_code=422, detail="the upload has no usable filename")

    suffix = Path(safe).suffix.lower()
    if suffix not in _INGESTABLE:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{suffix or '(no extension)'} is not a format this box can read. "
                f"Supported: {', '.join(sorted(_INGESTABLE))}"
            ),
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    if not payload:
        raise HTTPException(status_code=422, detail="the upload is empty")

    inbox = get_settings().var_dir / "corpus_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    target = inbox / safe
    target.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    get_audit().append(
        "kb.upload",
        detail=f"{safe} ({len(payload)} bytes)",
        data_class=(label or Label.INTERNAL).value,
        sha256=digest,
    )

    if background:
        # THE BROWSER'S PATH.
        #
        # Reading a scanned page can take minutes - a vision model load, an
        # image encode, a transcription. Done inside this request it occupies
        # uvicorn's event loop, so /health and every panel stop answering and
        # the UI correctly reports the gateway as unreachable. One slow
        # document must not take the workbench down.
        #
        # The synchronous path below is kept exactly as it was, because it is
        # what `kavach kb ingest` and every test call, and a script that wants
        # the report in the response should still get it.
        job = get_ingest_jobs().start(target, label=label, force=force, filename=safe)
        return {
            "job": job.as_dict(),
            "uploaded": {"filename": safe, "bytes": len(payload), "sha256": digest},
        }

    try:
        report = get_ingestor().ingest(target, label=label, force=force)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = report.as_dict()
    result["uploaded"] = {"filename": safe, "bytes": len(payload), "sha256": digest}
    result["index"] = get_index().state().as_dict()
    return result


@router.get("/jobs/{job_id}")
def ingest_job(job_id: str) -> dict[str, Any]:
    """How far a background ingest has got.

    Reports a STAGE and a PAGE COUNT, never a percentage: "reading page 3 of
    12" is a true statement, and a bar filled from elapsed time is a guess that
    stalls at 90% on exactly the page that is taking the time.
    """
    job = get_ingest_jobs().get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no ingest job {job_id!r}. Finished jobs are kept for a few "
                "minutes; the corpus and the audit chain are the durable record."
            ),
        )
    return job.as_dict()


@router.get("/jobs")
def ingest_jobs() -> dict[str, Any]:
    """Anything being read right now - so a reload does not lose the progress."""
    return {"running": [j.as_dict() for j in get_ingest_jobs().running()]}


@router.post("/reindex")
def reindex() -> dict[str, Any]:
    return get_index().build().as_dict()


@router.post("/reload")
def reload_config() -> dict[str, Any]:
    """Re-read knowledge.yaml live - chunk sizes and retrieval knobs, no restart."""
    config = reload_knowledge_config()
    return {"reloaded": sorted(config), "retrieval": config.get("retrieval", {})}


@router.post("/search")
def search(payload: SearchPayload) -> dict[str, Any]:
    return get_retriever().search(
        payload.query, top_k=payload.top_k, role=payload.role
    ).as_dict()


@router.post("/answer")
def answer(payload: AnswerPayload) -> dict[str, Any]:
    """Retrieve, answer, then check every citation resolves (§07).

    Always 200: a refusal to answer is a result, not an error, and the panel has
    to render it with its reason rather than an HTTP status the UI turns into
    'something went wrong'.
    """
    return get_answerer().answer(
        payload.question,
        top_k=payload.top_k,
        role=payload.role,
        max_tokens=payload.max_tokens,
    ).as_dict()


@router.get("/chunks")
def chunks(
    doc_id: str = Query(...),
    page: int | None = Query(default=None),
) -> dict[str, Any]:
    """The passages behind a citation - what a reader clicks through to."""
    store = get_knowledge_store()
    if store.document(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
    if page is None:
        found = [c for c in store.chunks() if c.doc_id == doc_id]
    else:
        found = store.chunks_on_page(doc_id, page)
    return {"count": len(found), "chunks": [c.as_dict() for c in found]}

"""The hybrid index (§07).

Two arms over the same chunks:

    lexical   BM25 over tokens. Finds `P-101A`, `MRPL-SOP-CDU-0101`, `SIL-2` -
              the exact strings a plant engineer actually types, which are also
              precisely the strings an embedder has never seen and cannot
              represent.

    dense     cosine over embeddings from the pinned embedder. Finds "what stops
              the pump if suction pressure drops" in a document that says
              "low-low suction pressure trip", where the lexical arm has nothing
              to match on.

Neither is optional. A refinery corpus is mostly tag numbers and mostly
paraphrase at the same time, and each arm is blind to exactly what the other one
sees.

Two properties this file exists to guarantee:

  THE INDEX KNOWS WHICH EMBEDDER BUILT IT. Vectors from bge-m3 and vectors from
  minilm are not comparable, and cosine between them is a number - a plausible,
  meaningless number. If the resident embedder is not the one that built the
  index, the dense arm is switched OFF and the search reports `degraded: true`
  with the reason. It does not quietly compare across model spaces.

  THE INDEX KNOWS WHEN IT IS STALE. Ingesting a document without rebuilding
  leaves chunks in SQLite that no arm can reach. `is_stale()` compares the
  indexed chunk ids against the store, and search says so rather than silently
  answering from the older corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .store import KnowledgeStore, get_knowledge_store
from .types import Chunk

try:
    from rank_bm25 import BM25Okapi

    HAVE_BM25 = True
except ImportError:  # pragma: no cover - exercised on a box without the extra
    BM25Okapi = None
    HAVE_BM25 = False

# A token is a run of alphanumerics, optionally joined by the separators that
# hold an instrument tag together. Splitting `P-101A` into `p` and `101a` and
# nothing else loses the one term in the query that identifies the equipment.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_SUBTOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Whole tags AND their parts.

    `MRPL-SOP-CDU-0101` is emitted alongside `mrpl`, `sop`, `cdu`, `0101`, so a
    query for the full document number scores the exact match highest while a
    query for "the CDU SOP" still finds it. Emitting only the whole tag makes
    the document unfindable by any query that does not reproduce it verbatim;
    emitting only the parts makes every MRPL document equally relevant.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        whole = match.group(0)
        tokens.append(whole)
        parts = _SUBTOKEN.findall(whole)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


@dataclass
class IndexState:
    """What the index can currently do, and what it cannot."""

    chunks: int
    lexical_ready: bool
    dense_ready: bool
    embed_model: str | None
    embed_dim: int | None
    resident_embedder: str | None
    stale: bool
    detail: str

    @property
    def degraded(self) -> bool:
        return not (self.lexical_ready and self.dense_ready) or self.stale

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "lexical_ready": self.lexical_ready,
            "dense_ready": self.dense_ready,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "resident_embedder": self.resident_embedder,
            "stale": self.stale,
            "degraded": self.degraded,
            "detail": self.detail,
        }


def _fingerprint(chunk_ids: list[str]) -> str:
    h = hashlib.sha256()
    for cid in chunk_ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class HybridIndex:
    """BM25 + dense vectors over the chunks in the knowledge store."""

    def __init__(
        self,
        var_dir: Path,
        store: KnowledgeStore | None = None,
        registry=None,
        warden=None,
        backend=None,
    ):
        self.dir = Path(var_dir) / "kb"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._store = store
        self._registry = registry
        self._warden = warden
        self._backend = backend

        self._lock = threading.RLock()
        self.chunk_ids: list[str] = []
        self.chunks: dict[str, Chunk] = {}
        self._bm25 = None
        self._corpus_tokens: list[set[str]] = []
        self._vectors: np.ndarray | None = None
        self.embed_model: str | None = None
        self.embed_dim: int | None = None
        # The fingerprint of an empty chunk set, not "". A fresh install has no
        # index file and an empty corpus, and those agree - reporting STALE there
        # sends an operator to `kb reindex` to fix a corpus that has nothing wrong
        # with it, and trains them to ignore the word by the time it means
        # something.
        self.fingerprint: str = _fingerprint([])
        self.built_at: float = 0.0

        self.load()

    # ---------------------------------------------------------------- wiring

    @property
    def store(self) -> KnowledgeStore:
        if self._store is None:
            self._store = get_knowledge_store()
        return self._store

    @property
    def registry(self):
        if self._registry is None:
            from ..runtime import get_registry

            self._registry = get_registry()
        return self._registry

    @property
    def warden(self):
        if self._warden is None:
            from ..runtime import get_warden

            self._warden = get_warden()
        return self._warden

    @property
    def backend(self):
        if self._backend is None:
            from ..runtime import get_backend

            self._backend = get_backend()
        return self._backend

    def embedder(self):
        """The embedder this profile may use, admitted if it is not resident.

        Never raises: no embedder means a lexical-only search that says so, not a
        failed request.
        """
        try:
            spec = self.registry.embedder(self.warden.profile.available)
            if spec is None:
                return None
            if not self.warden.is_resident(spec.id):
                self.warden.admit(spec.id)
            return spec
        except Exception:  # noqa: BLE001 - degrade, never fail the search
            return None

    # ---------------------------------------------------------------- paths

    @property
    def _vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.dir / "index.json"

    # ---------------------------------------------------------------- build

    # Set here as well as in build(): state() can be called on an index that was
    # loaded from disk and never rebuilt in this process.
    embed_error: str | None = None

    # How many texts go to the embedder at once. Small enough to stay inside
    # llama-server's batch limit and the HTTP timeout on a corpus of any size,
    # large enough that a few thousand chunks is tens of requests rather than
    # thousands. A failure now costs one batch and names it, instead of losing
    # the whole rebuild.
    EMBED_BATCH = 32

    def _embed_in_batches(self, spec, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.EMBED_BATCH):
            batch = texts[start : start + self.EMBED_BATCH]
            try:
                vectors.extend(self.backend.embed(spec, batch))
            except Exception as exc:
                raise RuntimeError(
                    f"embedding failed on chunks {start}-{start + len(batch) - 1} "
                    f"of {len(texts)}: {exc}"
                ) from exc
        if len(vectors) != len(texts):
            # A backend that returns a different number of vectors than it was
            # given would silently misalign every chunk with somebody else's
            # embedding - every search result wrong, nothing raised.
            raise RuntimeError(
                f"the embedder returned {len(vectors)} vectors for {len(texts)} "
                "chunks; refusing to build an index where a chunk and its vector "
                "may not be the same text"
            )
        return vectors

    def build(self, *, embed: bool = True) -> IndexState:
        """Rebuild both arms from the store. Idempotent and safe to call live."""
        with self._lock:
            chunks = self.store.chunks()
            # A SIMULATED chunk is synthetic text. It is kept in the store so the
            # corpus panel can count it, but it is not indexed: a retrieval hit on
            # invented text is indistinguishable from a hit on the document, and
            # that is exactly the confusion the extraction stamp exists to prevent.
            indexable = [c for c in chunks if c.extraction.is_real]

            self.chunks = {c.id: c for c in indexable}
            self.chunk_ids = [c.id for c in indexable]
            self.fingerprint = _fingerprint(self.chunk_ids)

            self._bm25 = None
            tokenised = [tokenise(c.text) for c in indexable]
            self._corpus_tokens = [set(t) for t in tokenised]
            if HAVE_BM25 and indexable:
                self._bm25 = BM25Okapi(tokenised)

            self._vectors = None
            self.embed_model = None
            self.embed_dim = None
            self.embed_error = None

            if embed and indexable:
                spec = self.embedder()
                if spec is not None:
                    try:
                        vectors = self._embed_in_batches(
                            spec, [c.text for c in indexable]
                        )
                        matrix = np.asarray(vectors, dtype=np.float32)
                        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                        # Store unit vectors so a query is one dot product, and a
                        # zero-norm chunk stays zero rather than becoming NaN and
                        # poisoning every comparison against it.
                        self._vectors = matrix / np.where(norms == 0, 1.0, norms)
                        self.embed_model = spec.id
                        self.embed_dim = self._vectors.shape[1]
                    except Exception as exc:  # noqa: BLE001 - lexical-only
                        # KEEP THE REASON. Swallowing it left the panel saying
                        # "no embeddings in the index" with no way to find out
                        # why the embedder refused - which on this box turned a
                        # ten-minute rebuild into a mystery.
                        self._vectors = None
                        self.embed_error = f"{type(exc).__name__}: {exc}"

            import time as _time

            self.built_at = _time.time()
            self.save()
            return self.state()

    # ---------------------------------------------------------------- persist

    def save(self) -> None:
        with self._lock:
            if self._vectors is not None:
                np.save(self._vectors_path, self._vectors)
            elif self._vectors_path.exists():
                self._vectors_path.unlink()
            self._meta_path.write_text(
                json.dumps(
                    {
                        "chunk_ids": self.chunk_ids,
                        "fingerprint": self.fingerprint,
                        "embed_model": self.embed_model,
                        "embed_dim": self.embed_dim,
                        "built_at": self.built_at,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def load(self) -> bool:
        """Restore a previously built index. Returns False when there is none."""
        if not self._meta_path.exists():
            return False
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        with self._lock:
            self.chunk_ids = list(meta.get("chunk_ids", []))
            self.fingerprint = meta.get("fingerprint", "")
            self.embed_model = meta.get("embed_model")
            self.embed_dim = meta.get("embed_dim")
            self.built_at = float(meta.get("built_at", 0.0))

            by_id = {c.id: c for c in self.store.chunks()}
            self.chunks = {cid: by_id[cid] for cid in self.chunk_ids if cid in by_id}
            # Any id in the meta that is no longer in the store makes the index
            # stale; is_stale() will catch it on the next search.
            texts = [self.chunks[cid].text for cid in self.chunk_ids if cid in self.chunks]
            tokenised = [tokenise(t) for t in texts]
            self._corpus_tokens = [set(t) for t in tokenised]
            self._bm25 = BM25Okapi(tokenised) if HAVE_BM25 and texts else None

            self._vectors = None
            if self._vectors_path.exists():
                try:
                    loaded = np.load(self._vectors_path)
                    if loaded.shape[0] == len(self.chunk_ids):
                        self._vectors = loaded
                except (OSError, ValueError):
                    self._vectors = None
        return True

    # ---------------------------------------------------------------- state

    def is_stale(self) -> bool:
        """Does the index cover exactly the indexable chunks now in the store?"""
        current = [c.id for c in self.store.chunks() if c.extraction.is_real]
        return _fingerprint(current) != self.fingerprint

    def state(self) -> IndexState:
        resident = self.embedder()
        resident_id = resident.id if resident else None
        stale = self.is_stale()

        dense_ready = self._vectors is not None and self.embed_model is not None
        detail_bits: list[str] = []

        if dense_ready and resident_id != self.embed_model:
            # Cross-model cosine is a number, not a similarity. Refuse it.
            dense_ready = False
            detail_bits.append(
                f"dense arm OFF: the index was embedded with {self.embed_model} but "
                f"{resident_id or 'no embedder'} is resident; vectors from two "
                "models are not comparable"
            )
        elif not dense_ready:
            # The REASON, when there is one. "no embeddings in the index" is
            # true and tells nobody what to do; the embedder refused for a
            # reason and that reason is the only actionable part.
            because = getattr(self, "embed_error", None)
            detail_bits.append(
                "dense arm OFF: no embeddings in the index"
                + ("" if resident_id else " and no embedder available on this profile")
                + (f" - the embedder failed with {because}" if because else "")
            )

        lexical_ready = self._bm25 is not None
        if not lexical_ready:
            detail_bits.append(
                "lexical arm OFF: "
                + ("rank_bm25 is not installed" if not HAVE_BM25 else "the index is empty")
            )

        if stale:
            detail_bits.append(
                "index is STALE - documents have been ingested or removed since it "
                "was built; run `kavach kb reindex`"
            )

        if not detail_bits:
            detail_bits.append(
                f"hybrid: BM25 + {self.embed_model} over {len(self.chunk_ids)} chunks"
            )

        return IndexState(
            chunks=len(self.chunk_ids),
            lexical_ready=lexical_ready,
            dense_ready=dense_ready,
            embed_model=self.embed_model,
            embed_dim=self.embed_dim,
            resident_embedder=resident_id,
            stale=stale,
            detail="; ".join(detail_bits),
        )

    # ---------------------------------------------------------------- query

    def bm25_ranking(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Chunks containing at least one query token, ordered by BM25.

        Membership decides whether a chunk is a hit; BM25 decides only the order.
        Thresholding on the score itself is the obvious implementation and it is
        wrong on a corpus this size: BM25's IDF term goes to zero, and then
        negative, as a term appears in more of the corpus, so on an eleven-page
        SOP set a chunk that plainly contains `PSLL-101` scores 0.0 and a
        `score > 0` filter drops it. The lexical arm exists precisely to find the
        rare exact string an engineer typed; an arm that silently discards the
        match on a small corpus is worse than no arm, because the dense arm
        covers for it and nobody notices until the corpus grows.
        """
        tokens = tokenise(query)
        if not tokens:
            return []
        wanted = set(tokens)

        with self._lock:
            if self._bm25 is None or not self.chunk_ids:
                return []
            scores = self._bm25.get_scores(tokens)
            matches = [
                i
                for i, chunk_tokens in enumerate(self._corpus_tokens)
                if i < len(self.chunk_ids) and wanted & chunk_tokens
            ]
            pairs = [(self.chunk_ids[i], float(scores[i])) for i in matches]

        # Ties break toward the earlier chunk so a query always produces the same
        # citations; a demo that reorders between runs looks broken even when the
        # ranking is equally correct both times.
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        return pairs[:limit]

    def dense_ranking(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Cosine ranking, or [] when the dense arm is unavailable or mismatched."""
        state = self.state()
        if not state.dense_ready:
            return []
        spec = self.embedder()
        if spec is None:
            return []
        try:
            (vector,) = self.backend.embed(spec, [query])
        except Exception:  # noqa: BLE001
            return []

        q = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0 or math.isnan(norm):
            return []
        q = q / norm

        with self._lock:
            if self._vectors is None or self._vectors.shape[1] != q.shape[0]:
                return []
            sims = self._vectors @ q

        order = np.argsort(-sims)[:limit]
        return [(self.chunk_ids[i], float(sims[i])) for i in order]


_index: HybridIndex | None = None
_index_lock = threading.Lock()


def get_index() -> HybridIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                from ..config import get_settings

                _index = HybridIndex(get_settings().var_dir)
    return _index


def reset_index() -> None:
    """Tests only."""
    global _index
    _index = None

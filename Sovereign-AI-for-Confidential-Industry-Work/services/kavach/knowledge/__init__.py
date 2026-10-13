"""L3 - the knowledge plane (§07).

    extract.py     the ingest fork: native text layer vs a picture of text
    chunk.py       page-exact chunking, so a citation can be followed
    store.py       what was ingested, how each page was read, under which label
    index.py       BM25 + dense vectors, each knowing when it cannot be trusted
    retrieval.py   rank fusion, clearance filtering, coverage that travels along
    citations.py   every citation must resolve to a passage that was retrieved
    answer.py      retrieve -> route -> check, with three refusals

The through-line: a knowledge plane's failure mode is not "no answer", it is a
fluent answer with a plausible citation to a page nobody read. Every module here
is arranged so that outcome has to survive a check that names it.
"""

from .answer import Answerer, GroundedAnswer, get_answerer, reset_answerer
from .chunk import chunk_document, chunk_page
from .citations import (
    CITATION,
    CITATION_INSTRUCTIONS,
    CitationCheck,
    GroundingResult,
    Verdict,
    check_grounding,
    parse_citations,
)
from .config import load_knowledge_config, reload_knowledge_config
from .extract import (
    SIMULATED_OCR_ENV,
    ImagePathStatus,
    ImageReader,
    IngestError,
    extract_document,
)
from .index import HybridIndex, IndexState, get_index, reset_index, tokenise
from .ingest import DocumentOutcome, Ingestor, IngestReport, get_ingestor, reset_ingestor
from .jobs import IngestJob, IngestJobs, get_ingest_jobs, reset_ingest_jobs
from .retrieval import (
    Coverage,
    RetrievalResult,
    Retriever,
    context_block,
    get_retriever,
    reset_retriever,
)
from .store import KnowledgeStore, get_knowledge_store, reset_knowledge_store
from .types import Chunk, DocumentRecord, Extraction, Hit, PageRecord

__all__ = [
    "CITATION",
    "CITATION_INSTRUCTIONS",
    "SIMULATED_OCR_ENV",
    "Answerer",
    "Chunk",
    "CitationCheck",
    "Coverage",
    "DocumentOutcome",
    "DocumentRecord",
    "Extraction",
    "GroundedAnswer",
    "GroundingResult",
    "Hit",
    "HybridIndex",
    "ImagePathStatus",
    "ImageReader",
    "IndexState",
    "IngestError",
    "IngestJob",
    "IngestJobs",
    "IngestReport",
    "Ingestor",
    "KnowledgeStore",
    "PageRecord",
    "RetrievalResult",
    "Retriever",
    "Verdict",
    "check_grounding",
    "chunk_document",
    "chunk_page",
    "context_block",
    "extract_document",
    "get_answerer",
    "get_index",
    "get_ingest_jobs",
    "get_ingestor",
    "get_knowledge_store",
    "get_retriever",
    "load_knowledge_config",
    "parse_citations",
    "reload_knowledge_config",
    "reset_answerer",
    "reset_index",
    "reset_ingest_jobs",
    "reset_ingestor",
    "reset_knowledge_store",
    "reset_retriever",
    "tokenise",
]

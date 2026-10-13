"""knowledge_demo.py - the three beats of the L3 knowledge plane (§07).

Run it against the MRPL corpus:

    .venv\\Scripts\\python.exe scripts\\knowledge_demo.py

    1. THE INGEST FORK          native text is read exactly; a scan is not
                                guessed at. What this box cannot read, it names.
    2. HYBRID RETRIEVAL         the lexical arm finds `PSLL-101`, the dense arm
                                finds the paraphrase, and neither alone finds
                                both queries.
    3. A FABRICATED CITATION    an answer citing a page that was never retrieved
                                is refused, with the page named.

Beat 3 is the one to show a judge. Everything else in a RAG demo is a search box;
this is the part that says what happens when the model is confidently wrong.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))

# A demo must never append to the real audit chain or the real corpus - the run
# sheet's record count is a number somebody reads off the screen on demo day.
_TMP = Path(tempfile.mkdtemp(prefix="kavach-kb-demo-"))
os.environ.setdefault("KAVACH_VAR_DIR", str(_TMP / "var"))
os.environ.setdefault("KAVACH_BACKEND", "mock")
os.environ.setdefault("KAVACH_ENFORCE_DIGESTS", "0")

from kavach.knowledge import (
    Extraction,
    ImageReader,
    check_grounding,
    get_index,
    get_ingestor,
    get_knowledge_store,
    get_retriever,
)

CORPUS = ROOT / "MRPL_DEMO_CORPUS"

LEXICAL_QUERY = "PSLL-101"
DENSE_QUERY = "what stops the pump if suction pressure falls too low"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def beat_1_the_fork() -> None:
    rule("1. THE INGEST FORK - what this box can read, and what it cannot")

    report = get_ingestor().ingest(CORPUS)
    store = get_knowledge_store()

    print()
    for doc in sorted(store.documents(), key=lambda d: d.id):
        kinds = ", ".join(
            f"{k}={v}" for k, v in sorted(doc.as_dict()["extraction"].items())
        )
        gap = ""
        if doc.unreadable_pages:
            gap = "   <- UNREADABLE p." + ",".join(str(p) for p in doc.unreadable_pages)
        print(f"  {doc.id[:30].ljust(30)} {doc.label.value.ljust(13)} {kinds}{gap}")

    stats = store.stats()
    print()
    print(f"  {stats['documents']} documents, {stats['pages']} pages, "
          f"{stats['chunks']} chunks indexed")
    print()
    print("  The interesting number is the one below, not the one above.")
    print(f"  UNREADABLE: {stats['unreadable_pages']} of {stats['pages']} pages")
    print()
    print(f"  why: {ImageReader().status().detail}")
    print()
    print("  Those pages are not in the index, no answer can cite them, and every")
    print("  search over this corpus repeats this line. The alternative - indexing")
    print("  them as empty - gives you a corpus that reports 11 of 11 documents")
    print("  loaded and quietly cannot answer six of the questions it was built for.")
    if report.failed:
        print()
        for doc in report.failed:
            print(f"  FAILED TO OPEN  {doc.path}: {doc.reason}")


def beat_2_hybrid() -> None:
    rule("2. HYBRID RETRIEVAL - each arm is blind to what the other one sees")

    index = get_index()
    retriever = get_retriever()

    for query, expect in ((LEXICAL_QUERY, "lexical"), (DENSE_QUERY, "dense")):
        lexical = index.bm25_ranking(query, 3)
        dense = index.dense_ranking(query, 3)
        print()
        print(f'  query: "{query}"')
        print(f"    BM25   found {len(lexical)}: "
              f"{', '.join(c for c, _ in lexical[:3]) or '-'}")
        print(f"    dense  found {len(dense)}: "
              f"{', '.join(c for c, _ in dense[:3]) or '-'}")

        result = retriever.search(query, top_k=3)
        print(f"    fused  -> {', '.join(result.citations) or 'nothing'}")
        print(f"    (the {expect} arm is the one carrying this query)")

    print()
    print("  Fusion is by RANK, not by score. BM25 scores are unbounded and")
    print("  corpus-dependent; cosine is 0..1. Adding them with weights makes the")
    print("  weight meaningless and lets the lexical arm win every query on a")
    print("  corpus full of tag numbers - which is every refinery corpus.")


def beat_3_fabricated_citation() -> None:
    rule("3. A FABRICATED CITATION - the failure this plane exists to catch")

    retriever = get_retriever()
    store = get_knowledge_store()
    result = retriever.search("PSLL-101 suction pressure trip setpoint", top_k=3)

    if not result.hits:
        print("  no passages retrieved; ingest the corpus first")
        return

    real = result.hits[0].chunk.citation
    print()
    print(f"  retrieved: {', '.join(result.citations)}")
    print()

    cases = [
        (
            "an answer that cites what it was given",
            f"The suction pressure trip setpoint is 1.80 bar gauge {real}.",
        ),
        (
            "the same answer with the page number nudged by one",
            (
                "The suction pressure trip setpoint is 1.80 bar gauge "
                f"{real.replace('p.1', 'p.2')}."
            ),
        ),
        (
            "an answer citing a standard nobody ingested",
            "The suction pressure trip setpoint is 1.80 bar gauge [API-610 p.7].",
        ),
        (
            "a fluent answer with no citation at all",
            "The suction pressure trip setpoint is 1.80 bar gauge.",
        ),
    ]

    for title, text in cases:
        grounding = check_grounding(text, result.hits, store=store)
        mark = "ACCEPTED" if grounding.ok else "REFUSED "
        print(f"  {mark}  {title}")
        print(f"            {text[:88]}")
        if not grounding.ok:
            print(f"            -> {grounding.reason[:150]}")
        print()

    print("  All four are well-formed. Three of them are wrong in ways that survive")
    print("  every check that only looks for square brackets.")


def beat_4_clearance() -> None:
    rule("4. CLEARANCE - the corpus is where CONFIDENTIAL text enters a model")

    retriever = get_retriever()
    query = "PSLL-101 suction pressure trip setpoint"

    for role in ("engineer", "operator"):
        result = retriever.search(query, top_k=4, role=role)
        print()
        print(f"  role {role}")
        print(f"    returned  {len(result.hits)}: {', '.join(result.citations) or '-'}")
        print(f"    withheld  {result.withheld}"
              + (f" - {result.withheld_reason}" if result.withheld else ""))
        print(f"    run label {result.effective_label.value}")

    print()
    print("  The withheld count is the point. Returning two hits instead of four")
    print("  teaches the operator their query was weak, which is not what happened.")


def main() -> int:
    if not CORPUS.exists():
        print(f"corpus not found at {CORPUS}", file=sys.stderr)
        print("build it with: python scripts/generate_mrpl_corpus.py", file=sys.stderr)
        return 2

    beat_1_the_fork()
    beat_2_hybrid()
    beat_3_fabricated_citation()
    beat_4_clearance()

    rule("what this box has NOT proved")
    print()
    print("  R5  the OCR/VLM path. No OCR engine and no real vision weights here,")
    print("      so every scan in the corpus is an honest gap rather than a")
    print("      transcription. Stage a VL model or install tesseract and the same")
    print(f"      pages come back as {Extraction.OCR.value} or {Extraction.VLM.value}.")
    print()
    print("  R4  grounded answering against a real model. The mock backend copies")
    print("      citations out of its context, which exercises the checker end to")
    print("      end - it does not tell you whether a 7B model will.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

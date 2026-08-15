"""Build the retrieval index: manifest -> extract -> chunk -> embed -> save.

Run inside the constrained container (`make index`) rather than on the host,
because indexing is the most memory-hungry operation in the system: the
embedding model, the extracted text of every document, every chunk and the
accumulating vector matrix are resident simultaneously. If anything is going
to breach the 7 GB ceiling it is this, so it is measured here and the peak is
reported rather than assumed.

Documents are processed one at a time and their extracted text released before
the next is opened. Holding all of them would be simpler and would scale with
corpus size in exactly the direction we cannot afford.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from agbe import config
from agbe.rag.chunker import Chunk, chunk_pages
from agbe.rag.embedder import Embedder
from agbe.rag.extract import probe
from agbe.rag.index import VectorIndex

MANIFEST_PATH = config.CORPUS_DIR / "manifest.json"
RAW_DIR = config.CORPUS_DIR / "raw"
PROCESSED_DIR = config.CORPUS_DIR / "processed"


def peak_rss_bytes() -> int:
    """Best-effort peak resident set size.

    Linux exposes VmHWM, which is the true high-water mark; psutil's rss is
    only the instantaneous value and will under-report a spike that has
    already been freed. We want the high-water mark, because that is what the
    7 GB ceiling is compared against.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:  # noqa: BLE001
        return 0


def load_manifest() -> dict[str, dict]:
    """Map filename -> provenance record.

    Missing manifest is tolerated so the pipeline can be exercised on
    hand-dropped PDFs, but it is reported: without provenance a chunk cannot
    carry a licence or a publication year, and the safety layer needs the year
    to suppress dated chemical recommendations.
    """
    if not MANIFEST_PATH.exists():
        return {}
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {d["filename"]: d for d in data.get("documents", [])}


def document_chunks(pdf: Path, record: dict | None) -> tuple[list[Chunk], str]:
    """Extract and chunk one document. Returns (chunks, status)."""
    extraction = probe(pdf)
    if extraction.error:
        return [], f"unreadable ({extraction.error})"
    if extraction.is_scanned:
        return [], "scanned, no text layer"
    if not extraction.usable:
        return [], "too little text"

    meta = record or {}
    chunks = chunk_pages(
        extraction.pages,
        doc_id=meta.get("doc_id", pdf.stem),
        title=meta.get("title", pdf.stem.replace("_", " ")),
        publisher=meta.get("publisher", "unknown"),
        year=meta.get("year", "unknown"),
        licence=meta.get("licence", "unspecified"),
        source_url=meta.get("source_url", ""),
        target_tokens=config.RETRIEVAL.chunk_tokens,
        overlap_tokens=config.RETRIEVAL.chunk_overlap_tokens,
    )
    return chunks, "ok"


def jsonl_chunks() -> list[Chunk]:
    """Chunk pre-extracted text from bulk web archives.

    Sources like the Infonet-Biovision export arrive as thousands of HTML
    pages rather than PDFs, and are extracted once by their own ingester into
    JSONL. Chunking them here rather than re-parsing HTML on every index build
    keeps a 1.8 GB archive out of the indexing path.

    Page numbers are meaningless for a web page, so page_start/page_end are
    both 1 and the citation carries the source URL instead.
    """
    chunks: list[Chunk] = []
    for path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        n_pages = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            page_chunks = chunk_pages(
                [rec["text"]],
                doc_id=rec["doc_id"],
                title=rec["title"],
                publisher=rec["publisher"],
                year=rec["year"],
                licence=rec["licence"],
                source_url=rec["source_url"],
                target_tokens=config.RETRIEVAL.chunk_tokens,
                overlap_tokens=config.RETRIEVAL.chunk_overlap_tokens,
            )
            chunks.extend(page_chunks)
            n_pages += 1
        print(f"  ok    {path.name[:58]:<58} {n_pages:>4} pages, "
              f"{len(chunks):>5} chunks")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk everything but do not embed (no model needed)")
    args = ap.parse_args()

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    jsonls = sorted(PROCESSED_DIR.glob("*.jsonl"))
    if not pdfs and not jsonls:
        print(f"no documents in {RAW_DIR} or {PROCESSED_DIR}\n"
              "run scripts/fetch_corpus.py and/or scripts/ingest_infonet.py first")
        return 1

    manifest = load_manifest()
    if not manifest:
        print("WARNING: no corpus/manifest.json - chunks will lack licence and\n"
              "         publication year, which disables the chemical-recency\n"
              "         safety rule. Run scripts/fetch_corpus.py to generate it.\n")

    started = time.time()
    all_chunks: list[Chunk] = []

    print(f"indexing {len(pdfs)} PDFs and {len(jsonls)} bulk archives\n")
    for pdf in pdfs:
        chunks, status = document_chunks(pdf, manifest.get(pdf.name))
        if status != "ok":
            print(f"  SKIP  {pdf.name[:58]:<58} {status}")
            continue
        all_chunks.extend(chunks)
        print(f"  ok    {pdf.name[:58]:<58} {len(chunks):>4} chunks")

    all_chunks.extend(jsonl_chunks())

    if not all_chunks:
        print("\nno usable chunks produced")
        return 1

    # Drop licence-excluded chunks HERE, before embedding.
    #
    # The same filter runs at load time in agbe/rag/index.py, and that one
    # stays - it is the guarantee that the shipped system cannot retrieve from
    # a NoDerivatives document whatever an old index on disk contains. But
    # doing it *only* at load has two costs that build time avoids:
    #
    #   * `np.asarray(vectors)[keep]` materialises the whole memory-mapped
    #     matrix into the heap to drop the rows, ~46 MB of peak RSS, directly
    #     against the mmap the loader takes trouble to set up;
    #   * we spend embedding time on passages that can never be retrieved -
    #     2,640 of 43,177 chunks on the current corpus, about 7 minutes of a
    #     two-hour build.
    #
    # Filtering at both ends is deliberate belt-and-braces, not duplication:
    # this one saves the work, the loader's one keeps the promise.
    from agbe.rag.licences import excluded

    keep = [c for c in all_chunks if not excluded(c.licence)[0]]
    if len(keep) != len(all_chunks):
        print(f"\nlicence gate: dropped {len(all_chunks) - len(keep)} chunks "
              f"from excluded-licence documents before embedding")
        all_chunks = keep

    n_docs = len({c.doc_id for c in all_chunks})
    print(f"\n{len(all_chunks)} chunks from {n_docs} documents")

    if args.dry_run:
        lengths = [len(c.text) for c in all_chunks]
        print(f"chars per chunk: min={min(lengths)} max={max(lengths)} "
              f"mean={sum(lengths)//len(lengths)}")
        print(f"peak RSS: {peak_rss_bytes()/1024**2:.0f} MB  (dry run, no model loaded)")
        return 0

    print(f"embedding with {config.EMBEDDING.filename} "
          f"({config.EMBEDDING.n_threads} threads) ...")

    with Embedder() as embedder:
        vectors = embedder.embed_passages(
            [c.text for c in all_chunks],
            batch_size=args.batch_size,
            progress=True,
        )

    index = VectorIndex.build(all_chunks, vectors)
    out_dir = index.save()

    peak = peak_rss_bytes()
    elapsed = time.time() - started
    vec_mb = vectors.nbytes / 1024**2

    print(f"\nwrote {out_dir}")
    print(f"  chunks     {len(index)}")
    print(f"  vectors    {vectors.shape} float32 ({vec_mb:.1f} MB)")
    print(f"  elapsed    {elapsed:.1f}s")
    print(f"  peak RSS   {peak/1024**3:.2f} GB "
          f"of {config.MEMORY_CEILING_BYTES/1024**3:.0f} GB ceiling")

    if peak > config.MEMORY_CEILING_BYTES:
        print("\nFAIL: exceeded the 7 GB ceiling during indexing")
        return 1
    if peak > config.MEMORY_TARGET_BYTES:
        print(f"\nWARNING: above the {config.MEMORY_TARGET_BYTES/1024**3:.0f} GB "
              "internal target - investigate before it becomes a ceiling breach")
    return 0


if __name__ == "__main__":
    sys.exit(main())

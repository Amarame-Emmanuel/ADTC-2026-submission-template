"""Vector index for the Àgbẹ̀ corpus.

WHY BRUTE FORCE RATHER THAN FAISS OR HNSW
-----------------------------------------
An approximate-nearest-neighbour index is the reflex choice and is not
justified at this scale.

The corpus is a few thousand to a few tens of thousands of passages. At 384
dimensions and float32, 20 000 passages is a 30 MB matrix, and a query is one
matrix-vector product - roughly 8 million multiply-adds, which is on the order
of a millisecond with NumPy's BLAS. Time to first token on a CPU-only laptop
is measured in hundreds of milliseconds to seconds; retrieval is not the
bottleneck and cannot be made one at this size.

Against that, an ANN index costs a dependency to build and audit, a build step,
tuning parameters that trade recall for speed we do not need, and *approximate*
results. Exact search removes an entire class of "why did it not find the
obvious passage" debugging from a system whose credibility rests on retrieving
the right guidance.

If the corpus grows past ~100k passages this decision should be revisited. It
is a scale-dependent choice, not a principle.

STORAGE
-------
Vectors go to .npy (a plain memory-mappable float32 array) and metadata to
JSON. Neither format needs this codebase to read it, which matters for an
audit: a reviewer can inspect the index with NumPy and a text editor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agbe import config
from agbe.rag.chunker import Chunk

#: How many top lexical hits may claim the floor tolerance below.
LEXICAL_FLOOR_EXEMPT = 2

#: How far below the dense floor a top lexical hit may sit and still be kept.
#:
#: This is a TOLERANCE, not a bypass, and the difference was found the hard
#: way. An unconditional exemption for the top-2 lexical hits raised coverage
#: to 100% and dropped refusal to 0% - "How do I fix my motorcycle engine?"
#: sailed through, because BM25 always returns a top-2 no matter how unrelated
#: the query.
#:
#: Absolute BM25 score does not separate them either: "I feel dizzy after
#: spraying my farm" scores 15.8, higher than the Newcastle disease question's
#: 15.5. BM25 magnitude tracks query length and term rarity, not relevance.
#:
#: What does separate them is how far the DENSE score falls short. The
#: Newcastle chunk sits at 0.679 against a 0.70 floor - a near miss, rescued by
#: an exact match on "twisting", "neck" and "greenish". The motorcycle question
#: sits at 0.41, nowhere near, and no lexical evidence should save it.
LEXICAL_FLOOR_TOLERANCE = 0.05


@dataclass
class SearchHit:
    """One retrieved passage and why it was retrieved."""

    chunk: Chunk
    score: float
    rank: int


def _demote_off_crop(
    fused: list[int], chunks: list[Chunk], query_text: str
) -> list[int]:
    """Move candidates about a DIFFERENT crop to the back of the queue.

    Measured across the 70 evaluation questions: **18 of 250 retrieved slots
    (7.2%) went to passages about a crop the question did not ask about** - a
    maize seed-production page for a cassava question, rice for tomato,
    groundnut for pepper. Eleven questions were affected. Those slots are not
    free: `top_k` is 6, and the control advice that section 6.8 could not surface
    loses by as little as 0.023 to the sixth-placed passage.

    DEMOTION, NOT EXCLUSION, AND THE CASE THAT FORCED IT
    ----------------------------------------------------
    A hard filter was the obvious design and is wrong. `crop-23` - "small white
    insects fly up when I shake my tomato plants" - retrieves a **Whiteflies**
    document that mentions cassava and not tomato, so a filter would discard it.
    Whitefly biology is exactly the same insect on both crops. Cross-crop
    material is often the right answer.

    So off-crop passages are ordered last rather than removed. If there are not
    enough on-crop passages to fill `top_k`, they still appear - which keeps this
    from ever *reducing* the evidence the model receives.

    THREE CATEGORIES, NOT TWO
    -------------------------
    Passages naming no in-scope crop at all - general disease principles,
    storage practice, whitefly biology written generically - are treated as
    NEUTRAL and never demoted. Demotion applies only to a passage that names
    some other crop *and not* the one asked about, which is the narrowest
    reading of "off-topic" available.
    """
    from agbe.rag.scope import crops_mentioned

    wanted = crops_mentioned(query_text)
    if not wanted:
        return fused

    on_crop: list[int] = []
    off_crop: list[int] = []
    for idx in fused:
        c = chunks[idx]
        found = crops_mentioned(f"{c.title} {c.text}")
        # Neutral (names no crop) counts as on-topic, deliberately.
        if found and not (found & wanted):
            off_crop.append(idx)
        else:
            on_crop.append(idx)

    # Stable within each group: fusion order is preserved, only the partition
    # is new.
    return on_crop + off_crop


class VectorIndex:
    """Exact cosine-similarity search over normalised passage vectors."""

    VECTORS_FILE = "vectors.npy"
    CHUNKS_FILE = "chunks.json"
    META_FILE = "index_meta.json"

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(
                f"index inconsistent: {len(vectors)} vectors, {len(chunks)} chunks"
            )
        self.vectors = vectors
        self.chunks = chunks
        self._bm25 = None  # built lazily; see hybrid_search

    @property
    def bm25(self):
        """Lexical index, built on first use.

        Built at load rather than persisted: it takes a couple of seconds over
        7,000 chunks and serialising it would add a file format to maintain and
        keep in sync with the vectors. Cheap to rebuild, easy to get wrong to
        cache.
        """
        if self._bm25 is None:
            from agbe.rag.lexical import BM25Index

            self._bm25 = BM25Index.build([c.text for c in self.chunks])
        return self._bm25

    def __len__(self) -> int:
        return len(self.chunks)

    # -- search ------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int | None = None,
        min_score: float | None = None,
        candidates: int = 20,
    ) -> list[SearchHit]:
        """Dense + BM25 retrieval fused by reciprocal rank.

        The only search path. A dense-only `search()` existed until hybrid
        replaced it everywhere; it was removed rather than left as a second
        way to query the index that no caller used and no test covered.

        Dense search understands that "twisted necks" and "twisting of the head
        and neck" mean the same thing. Lexical search refuses to overlook that
        "greenish" appears verbatim in exactly one document. Neither alone
        found the Newcastle disease page for a farmer describing its textbook
        symptoms; together they do.

        The reported score stays the *dense* similarity, because the refusal
        threshold is calibrated against it and because a cosine value is
        interpretable in a way a fused rank is not. Fusion decides what is
        retrieved; the dense score decides whether it is good enough to use.
        """
        from agbe.rag.lexical import reciprocal_rank_fusion
        from agbe.rag.quality import is_usable

        if len(self) == 0:
            return []

        top_k = top_k if top_k is not None else config.RETRIEVAL.top_k
        min_score = min_score if min_score is not None else config.RETRIEVAL.min_score

        dense_scores = self.vectors @ query_vector
        n = min(candidates, len(self))
        dense_top = np.argpartition(-dense_scores, n - 1)[:n]
        dense_top = dense_top[np.argsort(-dense_scores[dense_top])].tolist()

        lexical_ranked = self.bm25.search(query_text, top_k=candidates)
        lexical_top = [idx for idx, _ in lexical_ranked]

        # Fuse over more candidates than requested. Chunks rejected below for
        # being front matter or OCR debris are then REPLACED by the next
        # acceptable passage rather than simply lost, so filtering does not
        # quietly shrink the evidence the model gets.
        fused = reciprocal_rank_fusion(dense_top, lexical_top, top_k=top_k * 3)

        # Off-crop passages go to the BACK OF THE QUEUE, before top_k is taken.
        #
        # Doing this after selection would be pure theatre - the same six
        # passages returned in a different order. It has to change which
        # candidates are considered, not how the chosen ones are sorted.
        fused = _demote_off_crop(fused, self.chunks, query_text)

        # Passages the lexical ranker put at the very top are exempt from the
        # dense floor.
        #
        # The floor exists to reject passages that are not about the question.
        # Applying it to a lexically-retrieved passage judges evidence on the
        # wrong scale: the Newcastle disease chunk carries "twisting of the head
        # and neck" and "greenish watery diarrhoea" verbatim and is BM25 rank 1,
        # but its dense similarity is 0.679 - just under a floor of 0.70. It was
        # therefore fused in and then immediately discarded, and the farmer got
        # generic husbandry pages instead.
        #
        # A top-ranked exact match on rare diagnostic vocabulary is relevance
        # evidence, simply measured differently. Only the top few qualify, so
        # this cannot become a back door for weak lexical noise.
        lexical_exempt = set(lexical_top[:LEXICAL_FLOOR_EXEMPT])

        hits: list[SearchHit] = []
        for rank, idx in enumerate(fused):
            if len(hits) >= top_k:
                break
            score = float(dense_scores[idx])
            floor = (
                min_score - LEXICAL_FLOOR_TOLERANCE
                if idx in lexical_exempt
                else min_score
            )
            if score < floor:
                continue

            # Title pages and OCR debris embed as plausible text - they carry
            # the document title and crop names, and score well - while
            # containing no guidance. See agbe/rag/quality.py for the passages
            # that motivated this.
            ok, _why = is_usable(self.chunks[idx].text)
            if not ok:
                continue

            hits.append(SearchHit(chunk=self.chunks[idx], score=score, rank=rank))
        return hits

    @staticmethod
    def _crops_of(chunk: Chunk) -> list[str]:
        from agbe.rag.relevance import CROP_TERMS

        low = chunk.text.lower()
        return [c for c, terms in CROP_TERMS.items() if any(t in low for t in terms)]

    # -- persistence -------------------------------------------------------

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or config.INDEX_DIR)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / self.VECTORS_FILE, self.vectors)
        (directory / self.CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (directory / self.META_FILE).write_text(
            json.dumps(
                {
                    "n_chunks": len(self.chunks),
                    "dim": int(self.vectors.shape[1]) if len(self.vectors) else 0,
                    "dtype": str(self.vectors.dtype),
                    "embedding_model": config.EMBEDDING.filename,
                    "normalised": True,
                    "search": "exact cosine (brute force)",
                    "documents": sorted({c.doc_id for c in self.chunks}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorIndex":
        directory = Path(directory or config.INDEX_DIR)
        vectors_path = directory / cls.VECTORS_FILE
        chunks_path = directory / cls.CHUNKS_FILE

        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"no index at {directory} - run `make index` first"
            )

        # mmap: the index is read-only at serve time, so let the OS page it in
        # on demand instead of copying the whole matrix into the heap.
        vectors = np.load(vectors_path, mmap_mode="r")
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw]

        # Licence enforcement at load, not only at fetch. The fetch gate
        # learned to reject NoDerivatives after five documents were already
        # embedded, and a re-index costs hours; dropping their chunks (and the
        # matching vector rows) here means the shipped system cannot retrieve
        # from them regardless of what the index on disk contains. See
        # agbe/rag/licences.py for why these five mattered doubly.
        from agbe.rag.licences import excluded

        keep = [i for i, c in enumerate(chunks) if not excluded(c.licence)[0]]
        if len(keep) != len(chunks):
            dropped = len(chunks) - len(keep)
            vectors = np.asarray(vectors)[keep]
            chunks = [chunks[i] for i in keep]
            print(f"index: dropped {dropped} chunks from excluded-licence documents")

        return cls(np.asarray(vectors), chunks)

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray) -> "VectorIndex":
        return cls(vectors=np.ascontiguousarray(vectors, dtype=np.float32),
                   chunks=chunks)

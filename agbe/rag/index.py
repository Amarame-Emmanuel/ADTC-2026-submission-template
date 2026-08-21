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
import re
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
LEXICAL_FLOOR_TOLERANCE = 0.08


@dataclass
class SearchHit:
    """One retrieved passage and why it was retrieved.

    `score` AND `rank` COME FROM DIFFERENT RANKINGS. This is deliberate and it
    surprises everyone who reads a result list for the first time, including
    the author of this comment while debugging §6.8:

        rank 1  score 0.688
        rank 2  score 0.676
        rank 3  score 0.728     <- higher score, lower rank

    `rank` is position after reciprocal-rank fusion of the dense and lexical
    rankings. `score` is the *dense cosine similarity alone*. A passage that
    BM25 ranked first can therefore sit above one with a better dense score.

    The reasoning is in `hybrid_search`: fusion decides what is retrieved, the
    dense score decides whether it is good enough to use, because the refusal
    floor is calibrated against cosine values and a fused rank is not
    interpretable. Both are kept because both are needed.

    The consequence for a reader: a printed result list is NOT monotonically
    decreasing in the number beside it, and sorting by `score` gives a different
    order than the system used.
    """

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
    from agbe.rag.relevance import OTHER_CROP_TERMS
    from agbe.rag.scope import crops_mentioned

    wanted = crops_mentioned(query_text)
    if not wanted:
        return fused

    on_crop: list[int] = []
    off_crop: list[int] = []
    for idx in fused:
        c = chunks[idx]
        # Out-of-scope crops count toward `found` too.
        #
        # crops_mentioned() knows only the nine IN-SCOPE crops, so a passage
        # about bean or papaya matched nothing, was classified NEUTRAL, and
        # could never be demoted. "My cowpea leaves have angular brown patches"
        # answered with common-bean advice and the words "do not work in BEAN
        # fields", from a document titled by its pathogen (Phaeoisariopsis
        # griseola) so the title rule below could not see it either.
        #
        # This is safe precisely because of the `found & wanted` test that
        # follows: a cassava document mentioning a banana intercrop still has
        # cassava in `found`, so it stays on-crop. Only a passage naming other
        # crops AND NOT the one asked about is demoted.
        found = crops_mentioned(f"{c.title} {c.text}") | _other_crops(c)
        if found and not (found & wanted):
            off_crop.append(idx)
        elif not found and _titled_for_another_crop(c.title):
            # Neutral by CROP_TERMS, but the TITLE names a crop that vocabulary
            # has never heard of - "Bean Leaves (New)", "Papaya (Revised)".
            # Without this branch such a passage sorts as on-topic forever; see
            # OTHER_CROP_TERMS for the two defects that reduced to it.
            off_crop.append(idx)
        else:
            on_crop.append(idx)

    # Stable within each group: fusion order is preserved, only the partition
    # is new.
    return on_crop + off_crop


def _other_crops(chunk: Chunk) -> set[str]:
    """Out-of-scope crop names appearing in a passage.

    Returned as a set so it can union with crops_mentioned(). Names are
    word-anchored: "ban" must not match "banana", and "pea" must not match
    "cowpea" - the second is why OTHER_CROP_TERMS carries no bare "pea".
    """
    from agbe.rag.relevance import OTHER_CROP_TERMS

    low = f"{chunk.title} {chunk.text}".lower()
    return {
        t for t in OTHER_CROP_TERMS
        if re.search(r"\b" + re.escape(t) + r"\b", low)
    }


def _titled_for_another_crop(title: str) -> bool:
    """Whether a document's TITLE announces a crop that is not in scope.

    The title, not the body, and deliberately so. Extension documents mention
    other crops constantly - intercrops, rotations, comparisons - and demoting
    on a body mention would push out most of the corpus. A crop in the TITLE is
    a statement about what the document is FOR.
    """
    from agbe.rag.relevance import OTHER_CROP_TERMS

    low = title.lower()
    return any(
        re.search(r"\b" + re.escape(term) + r"\b", low) for term in OTHER_CROP_TERMS
    )


#: Vocabulary that marks a passage as written for a RESEARCHER, not a farmer.
#:
#: Asked "my yam tubers are rotting in the barn", the system answered: "use a
#: hand trowel to carefully remove the tubers from the pot without damaging the
#: substrate... store them by sorting by family in the barn." Pots, substrate
#: and families are a genebank nursery protocol. The farmer has a barn.
#:
#: The corpus was not the problem. Four of the six retrieved passages were
#: exactly right - yam barns in Abakaliki, wet rot, nematode damage in storage.
#: Two were institutional: a Standard Operating Procedure for nursery harvest,
#: and a guide to in vitro germplasm collections. The model took its steps from
#: those two, because a procedure written as numbered steps reads more like
#: instructions than prose about rot does.
#:
#: Only unambiguous markers are listed. "Trial", "plot" and "treatment" are
#: ordinary agronomy words that appear throughout legitimate extension material
#: and are deliberately absent; `substrate` alone is likewise too weak. What
#: remains names a research apparatus a smallholder does not have.
_RESEARCH_REGISTER = re.compile(
    r"\b(?:standard operating procedure|in vitro|germplasm|genebank|"
    r"gene bank|accessions?|screen ?house|tissue culture|"
    r"randomi[sz]ed complete block|research protocol)\b",
    re.IGNORECASE,
)


#: Post-harvest vocabulary: storage, drying, bagging, transport, processing.
_POSTHARVEST_TERMS = re.compile(
    r"\b(?:storage|storing|stored|store|stores|drying|dried|dry(?:ing)? floor|"
    r"moisture content|bagging|bagged|warehouse|shelf ?life|threshing|"
    r"winnow\w*|silo|crib|post-?harvest|packaging|hermetic|tarpaulin)\b",
    re.IGNORECASE,
)

#: A question about a plant that is still growing names its ANATOMY.
#: A question about a plant that is still growing names its ANATOMY.
#:
#: Every term carries an optional plural, and inflections are matched with
#: `\w*` where the word has them. Without that, "my maize TASSELS are white"
#: and "my okra is FLOWERING" both read as non-field questions - `tassel` does
#: not match "tassels", and `flower` does not match "flowering" - so the
#: post-harvest demotion stepped aside and a field pest was answered with store
#: hygiene advice. That is the third time a missing plural has broken a rule in
#: this codebase; `tests/test_guard_boundaries.py` exists because of the first.
_FIELD_CONTEXT = re.compile(
    r"\b(?:leaf|leaves|vine|stem|flower\w*|seedling|plant\w*|field|"
    r"shoot|root|panicle|tassel|cob|branch|foliage|grow\w*|germinat\w*|"
    r"pod|tuber|sett|sucker|node|tip|whorl|silk|husk|crown|runner|"
    r"canopy|stand|nursery|transplant\w*|till\w*|head)s?\b",
    re.IGNORECASE,
)

#: ...unless it also names where the harvest is being kept, in which case it is
#: a storage question however much plant vocabulary it carries.
_STORE_CONTEXT = re.compile(
    r"\b(?:store|stored|storing|storage|barn|crib|warehouse|silo|bag|bags|"
    r"sack|sacks|after harvest|post-?harvest|market|selling|sell|transport|"
    r"shelf)\b",
    re.IGNORECASE,
)

#: Post-harvest terms per thousand words, above which a passage is treated as
#: written about the store rather than the field. Calibrated on the two
#: measured failures: their post-harvest passages ran 42-73 per thousand, while
#: the field passages in the same result sets scored 0-9.
_POSTHARVEST_DENSITY = 25.0
#: ...and an absolute floor, so a short passage with one stray "store" is safe.
_POSTHARVEST_MIN_TERMS = 3


def asks_about_field(query_text: str) -> bool:
    """Whether a question is about a growing plant rather than a stored crop."""
    return bool(
        _FIELD_CONTEXT.search(query_text) and not _STORE_CONTEXT.search(query_text)
    )


def _is_postharvest(chunk: Chunk) -> bool:
    body = f"{chunk.title} {chunk.text}"
    n = len(_POSTHARVEST_TERMS.findall(body))
    if n < _POSTHARVEST_MIN_TERMS:
        return False
    words = max(1, len(body.split()))
    return (1000.0 * n / words) >= _POSTHARVEST_DENSITY


def _demote_postharvest(
    fused: list[int], chunks: list[Chunk], query_text: str
) -> list[int]:
    """Move storage and drying passages behind field passages, for field questions.

    WHY THIS EXISTS
    Asked "the bottom leaves of my maize are drying from the tip inwards", the
    system answered "dry the maize to 13% moisture content or below to minimise
    insect and fungal growth" - post-harvest advice for a symptom on a living
    plant. Asked "why do my groundnut pods stay empty when I lift the plant?" it
    answered "because the pods contain water... dry the pods well after lifting".
    Neither farmer has harvested anything.

    Measured on those two questions: two of six and three of six retrieved slots
    were post-harvest passages, running 42-73 post-harvest terms per thousand
    words against 0-9 for the field passages in the same result set. The corpus
    is rich in storage and processing material and it shares vocabulary with
    field questions - pods, cobs, drying, leaves - so it competes on the terms
    that matter and wins slots it should not.

    This is the same failure the A-K review recorded twice ("my maize cobs have
    only a few grains" and "why are my cowpea pods empty" both answered with
    storage advice), which makes four independent instances across three
    question sets.

    WHY IT KEYS ON THE QUESTION'S ANATOMY, NOT ITS VERBS
    "Drying" appears in both worlds - leaves dry on the plant, grain is dried in
    the sun - so a verb test would classify the maize question as post-harvest
    and do nothing. What separates them is that a field question names a part of
    a LIVING plant. A question that also names where a harvest is kept - barn,
    store, bag, market - is a storage question whatever else it says, and is
    left alone: "my yam tubers are rotting in the barn" must keep its storage
    passages.

    Demotion, not exclusion, as everywhere else in this module: if there are not
    enough field passages to fill top_k these still appear.
    """
    if not asks_about_field(query_text):
        return fused

    field: list[int] = []
    postharvest: list[int] = []
    for idx in fused:
        target = postharvest if _is_postharvest(chunks[idx]) else field
        target.append(idx)
    return field + postharvest


def _demote_research_protocol(fused: list[int], chunks: list[Chunk]) -> list[int]:
    """Move researcher-facing passages to the back of the queue.

    DEMOTION, NOT EXCLUSION, for the same reason as `_demote_off_crop`: a
    germplasm document can still carry the only description of a disease in the
    corpus, and this must never *reduce* the evidence available. If there are
    not enough farmer-facing passages to fill `top_k`, these still appear.

    Applied unconditionally rather than only for storage questions. The register
    is wrong for a smallholder whatever the topic, and a rule that fires only on
    the one question that exposed it would be fitting to the example rather than
    to the defect.
    """
    farmer: list[int] = []
    research: list[int] = []
    for idx in fused:
        c = chunks[idx]
        target = research if _RESEARCH_REGISTER.search(f"{c.title} {c.text}") else farmer
        target.append(idx)
    return farmer + research

def _promote_named_pest(
    fused: list[int], chunks: list[Chunk], query_text: str
) -> list[int]:
    """Sort passages naming the identified pest above those that do not.

    Asked about "white cottony insects on the growing tip of my cassava", the
    system answers about spiralling whitefly. That is cassava mealybug: both
    insects leave white waxy material, and the discriminator is LOCATION - the
    mealybug attacks the growing tip, the spiralling whitefly the undersides of
    leaves.

    Everything upstream already works. `query.sign_names` maps "cottony" to
    "mealybug" and puts it in the query; mealybug passages ARE retrieved. They
    then lose 8:4 to whitefly content, because all six passages come from one
    document whose pest section leads with whitefly, and a 1.5B model follows
    the majority of what it is handed.

    WHY THIS AND NOT THE PER-DOCUMENT CAP
    -------------------------------------
    A cap of two passages per document was built and reverted. It worked on its
    own terms - six passages from five documents instead of one, mealybug
    leading 8:7, a dedicated Mealybugs document retrieved - and the answer got
    WORSE: it invented "the spiralling whitefly is a natural enemy of cassava
    pests", repeated four times, and recycled a document title as advice.

    The lesson recorded in FINDINGS F-17 is that context COHERENCE matters more
    to this model than context COVERAGE. Fragmenting the context across five
    documents to improve a retrieval statistic cost more than the statistic was
    worth.

    So this reorders WITHIN the passages already retrieved. Same documents, same
    count, no new sources - only which of them the model reads first. It is the
    narrowest intervention that addresses the 8:4 imbalance, and it is designed
    not to disturb the property the cap disturbed.

    Nothing is dropped: passages not naming the pest keep their order behind
    those that do, so a question whose corpus has no such passage is unaffected.
    """
    from agbe.rag.query import sign_names

    names = sign_names(query_text)
    if not names:
        return fused

    named: list[int] = []
    rest: list[int] = []
    for idx in fused:
        c = chunks[idx]
        haystack = f"{c.title} {c.text}".lower()
        (named if any(n in haystack for n in names) else rest).append(idx)

    return named + rest


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

        # Then, within what survives, let passages naming the identified
        # pest lead. Reorders only - same documents, same count. See
        # _promote_named_pest for why this is not the per-document cap.
        fused = _promote_named_pest(fused, self.chunks, query_text)

        # Researcher-facing passages go behind farmer-facing ones. After the
        # crop partition and before top_k is taken, so it changes which
        # candidates are considered rather than reordering the chosen six.
        fused = _demote_research_protocol(fused, self.chunks)

        # Storage and drying passages go behind field passages when the
        # question is about a growing plant. See _demote_postharvest for the
        # four measured instances of a field symptom answered from the store.
        fused = _demote_postharvest(fused, self.chunks, query_text)

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

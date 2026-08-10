"""BM25 lexical retrieval, fused with dense search.

WHY DENSE SEARCH ALONE IS NOT ENOUGH
------------------------------------
A real failure, found by asking a real question.

    Farmer: "My chickens are dying suddenly with twisted necks and green
             droppings"

The corpus contains a page that says, verbatim: "twisting of the head and
neck", "Greenish watery diarrhoea", "sudden death without prior indications of
illness". It is the Newcastle disease page, and it is the correct answer.

Dense retrieval did not return it. It returned four pages titled "Chicken
(animal welfare information)" instead, because the query's dominant term is
"chickens" and those pages are saturated with it. The Newcastle page speaks of
"birds" and "flocks". The embedding matched the *subject* and ignored the
*symptoms* - and the symptoms are the entire diagnostic content of the question.

The model, handed four pages of generic husbandry, then invented a diagnosis
("Marek's disease", "Avian Influenza"), both wrong.

BM25 fixes precisely this. Rare, specific terms - "twisting", "greenish",
"droppings" - carry high inverse-document-frequency weight, so a document
containing them scores highly even when it is not saturated with the topic
noun. Dense retrieval understands meaning; lexical retrieval refuses to
overlook a literal match.

COST
----
Negligible, which is why this is the right fix rather than a bigger embedding
model. The index is term frequencies over ~7,000 chunks: a few MB of Python
dicts, built in seconds, queried in single-digit milliseconds. No new model, no
new framework, nothing added to the memory budget that matters.

FUSION
------
Reciprocal Rank Fusion rather than score averaging. Cosine similarities and
BM25 scores live on different, non-comparable scales, and normalising them
requires assumptions about their distributions that do not hold across queries.
RRF uses only rank positions, so it is immune to that entirely.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

#: Standard BM25 parameters. k1 controls term-frequency saturation, b controls
#: length normalisation. These are the widely used defaults and there is no
#: labelled data here to justify tuning them.
BM25_K1 = 1.5
BM25_B = 0.75

#: RRF damping. 60 is the value from the original formulation; it keeps any
#: single ranker from dominating on the strength of one confident hit.
RRF_K = 60

_TOKEN = re.compile(r"[a-z][a-z0-9\-]{1,}")

#: Words too common in an agricultural corpus to discriminate. Deliberately
#: short: BM25's IDF term already suppresses common words, and over-pruning
#: removes signal. "chicken" is NOT here - it is a legitimate query term, it
#: simply must not be the only thing that matters.
_STOPWORDS = {
    "the", "and", "for", "are", "with", "this", "that", "from", "have", "has",
    "can", "will", "you", "your", "not", "but", "all", "may", "should", "when",
    "which", "they", "them", "their", "there", "these", "those", "been", "was",
    "were", "into", "also", "such", "than", "then", "other", "more", "most",
    "some", "any", "how", "what", "why", "who", "where", "does", "did", "was",
}


# Suffix stripping, longest-first so "-ings" is tried before "-s".
#
# WITHOUT THIS, BM25 CONTRIBUTES NOTHING HERE. The first version of this module
# had no stemmer and failed the exact case it was written to fix:
#
#     farmer: "twisted necks ... green droppings"
#     source: "twisting of the head and neck ... Greenish watery diarrhoea"
#
# Not one diagnostic term matched. twisted/twisting, necks/neck, green/greenish
# are different strings, so the lexical ranker scored the correct document at
# zero and the fused result was no better than dense alone - in fact slightly
# worse, because fusion diluted the dense ranking with noise.
#
# A full Porter stemmer is overkill and brings a dependency. Suffix stripping
# handles the morphology that actually appears in symptom descriptions: plurals,
# participles, and the "-ish" that turns "green" into "greenish".
_SUFFIXES = (
    "ations", "ization", "iveness", "fulness", "ousness",
    "ation", "ingly", "edly", "ings", "ness", "ment", "ible", "able",
    "ives", "ing", "ish", "ies", "ied", "est", "ers", "ed", "es", "er", "ly", "s",
)

#: Never stem below this length; "was" -> "wa" and similar are pure noise.
_MIN_STEM = 4


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            base = word[: -len(suffix)]
            # "dying" -> "dy" is wrong, but "twisting" -> "twist" and
            # "droppings" -> "dropp" are right; collapse the doubled consonant
            # so "dropping" and "drops" meet at the same stem.
            if len(base) > 4 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
            return base
    return word


def tokenize(text: str) -> list[str]:
    return [
        stem(t)
        for t in _TOKEN.findall(text.lower())
        if t not in _STOPWORDS
    ]


@dataclass
class BM25Index:
    """Sparse term index over the corpus chunks."""

    doc_freq: dict[str, int]
    postings: dict[str, list[tuple[int, int]]]  # term -> [(chunk_idx, tf)]
    doc_len: list[int]
    avg_len: float
    n_docs: int

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: dict[str, int] = Counter()
        doc_len: list[int] = []

        for idx, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings[term].append((idx, tf))
                doc_freq[term] += 1

        n = len(texts)
        return cls(
            doc_freq=dict(doc_freq),
            postings=dict(postings),
            doc_len=doc_len,
            avg_len=sum(doc_len) / n if n else 0.0,
            n_docs=n,
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(chunk_index, score)] best-first."""
        terms = tokenize(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            # Standard BM25 IDF with the +0.5 smoothing that keeps very common
            # terms from going negative.
            idf = math.log(
                1 + (self.n_docs - self.doc_freq[term] + 0.5)
                / (self.doc_freq[term] + 0.5)
            )
            for idx, tf in posting:
                norm = 1 - BM25_B + BM25_B * (self.doc_len[idx] / self.avg_len)
                scores[idx] += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * norm)

        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


#: Top hits from each ranker that are admitted unconditionally, before RRF
#: fills the remaining slots.
#:
#: Plain RRF lost the Newcastle disease case even after stemming fixed the
#: lexical match. The page splits into five chunks; chunk 3 carries the symptom
#: list ("twisting of the head and neck", "greenish watery diarrhoea") and BM25
#: ranked it first. It still lost, because RRF with k=60 damps so heavily that
#: appearing mid-rank in BOTH lists beats being first in one - and a generic
#: husbandry chunk did appear in both.
#:
#: That trade is wrong here. An exact match on rare diagnostic vocabulary is
#: strong, specific evidence; two mediocre agreements on a common topic noun
#: are not. Guaranteeing each ranker's best hits a slot means neither ranker
#: can be silently outvoted on the thing it is best at.
GUARANTEED_PER_RANKER = 2


def reciprocal_rank_fusion(
    dense: list[int], lexical: list[int], top_k: int, k: int = RRF_K
) -> list[int]:
    """Fuse two ranked lists of chunk indices by rank position only.

    Score fusion would require the cosine similarities and BM25 scores to be
    comparable. They are not - they have different ranges, different
    distributions, and their relative scale shifts per query. RRF sidesteps the
    problem: only positions matter.

    A document ranked highly by BOTH rankers rises above one ranked first by
    either alone, which is exactly the behaviour wanted here. The Newcastle
    page ranks first lexically and mid-table densely; the generic husbandry
    pages rank high densely and poorly lexically.
    """
    scores: dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense):
        scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(lexical):
        scores[idx] += 1.0 / (k + rank + 1)

    ranked = [idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    # Reserve slots for each ranker's own best hits before RRF fills the rest,
    # so neither can be outvoted on what it is best at. Order is preserved:
    # guaranteed picks are interleaved by rank so the strongest evidence from
    # either ranker leads.
    guaranteed: list[int] = []
    for i in range(GUARANTEED_PER_RANKER):
        for source in (lexical, dense):
            if i < len(source) and source[i] not in guaranteed:
                guaranteed.append(source[i])

    out = guaranteed[:top_k]
    for idx in ranked:
        if len(out) >= top_k:
            break
        if idx not in out:
            out.append(idx)
    return out

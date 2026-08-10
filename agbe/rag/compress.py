"""Context compression: keep the sentences that answer the question.

WHY
---
Time to first token is governed by one equation:

    TTFT ~= prompt_tokens / prefill_rate

Prefill already runs at ~70% of this hardware's peak, so the rate cannot be
improved. The only lever is the numerator.

Retrieval returns whole chunks of ~330-450 tokens because that is the right
granularity for *finding* the passage. It is the wrong granularity for
*answering*: a farmer asking about yellow twisted leaves does not need the
paragraph on the pest's taxonomy that happens to share a chunk with the
symptom description. Those tokens cost prefill time and contribute nothing.

Measured: ~1,456 prompt tokens gave 74s TTFT on development hardware, which
projects to ~125s on the slower reference laptop. Cutting the passages to the
sentences that actually match the query brings that down roughly by half.

HOW
---
Split each retrieved chunk into sentences, embed them with the same model used
for retrieval, and keep the highest-scoring ones within a token budget -
restored to their original order so the text still reads as prose.

SAFETY: WHY THIS IS NOT A PURE RELEVANCE RANKING
------------------------------------------------
Compression can delete a warning.

A chunk may read "Apply the product at the recommended rate. Do not use near
water sources or where livestock graze." Scored purely on similarity to "how do
I treat my maize", the first sentence wins and the second is dropped. The
result is advice that is more dangerous than the source it came from - and
dangerous in a way that is invisible, because the citation still points at a
document that *did* carry the warning.

So sentences carrying safety content are retained unconditionally, outside the
budget. Being slightly over budget costs a second of prefill; losing a
prohibition costs more than that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: Sentence split that tolerates the abbreviations common in agronomic text
#: ("approx.", "spp.", "cv.", "e.g.") without breaking mid-sentence.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

#: A sentence matching any of these is kept regardless of its relevance score.
#: Prohibitions, cautions, dosages and withdrawal periods are exactly the
#: content a similarity ranking discards and a farmer most needs.
_SAFETY_CRITICAL = re.compile(
    r"\b(?:do not|don't|never|avoid|caution|warning|danger|toxic|poison|"
    r"harmful|protective|withdrawal|residue|not recommended|banned|"
    r"restricted|wear gloves|wash|keep away|children|livestock graze|"
    r"water source|before harvest|days? after)\b",
    re.IGNORECASE,
)

#: Rough characters per token, matching the chunker's conservative estimate.
CHARS_PER_TOKEN = 3.5


@dataclass
class CompressionResult:
    text: str
    sentences_kept: int
    sentences_total: int
    tokens_before: int
    tokens_after: int
    safety_sentences_forced: int

    @property
    def ratio(self) -> float:
        return self.tokens_after / self.tokens_before if self.tokens_before else 1.0


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    # Very short fragments are usually headings or list bullets; merge them
    # forward so they do not occupy a slot on their own.
    merged: list[str] = []
    for part in parts:
        if merged and len(part.split()) <= 3:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def is_safety_critical(sentence: str) -> bool:
    return bool(_SAFETY_CRITICAL.search(sentence))


def compress_passage(
    passage: str,
    query_vector: np.ndarray,
    embedder,
    token_budget: int,
    min_sentences: int = 2,
) -> CompressionResult:
    """Keep the sentences of one passage that answer the query.

    `min_sentences` guarantees a passage never collapses to a single fragment
    with no context - a lone sentence reading "Remove them immediately" is
    worse than useless without whatever "them" refers to.
    """
    sentences = split_sentences(passage)
    tokens_before = int(len(passage) / CHARS_PER_TOKEN)

    if len(sentences) <= min_sentences:
        return CompressionResult(
            passage, len(sentences), len(sentences), tokens_before, tokens_before, 0
        )

    vectors = embedder.embed_passages(sentences, batch_size=len(sentences))
    scores = vectors @ query_vector

    forced = {i for i, s in enumerate(sentences) if is_safety_critical(s)}

    # Safety sentences first, then the best remaining by relevance.
    order = sorted(
        (i for i in range(len(sentences)) if i not in forced),
        key=lambda i: -scores[i],
    )

    keep = set(forced)
    used = sum(len(sentences[i]) for i in forced) / CHARS_PER_TOKEN

    for i in order:
        cost = len(sentences[i]) / CHARS_PER_TOKEN
        if used + cost > token_budget and len(keep) >= min_sentences:
            break
        keep.add(i)
        used += cost

    text = " ".join(sentences[i] for i in sorted(keep))
    return CompressionResult(
        text=text,
        sentences_kept=len(keep),
        sentences_total=len(sentences),
        tokens_before=tokens_before,
        tokens_after=int(len(text) / CHARS_PER_TOKEN),
        safety_sentences_forced=len(forced),
    )


def compress_hits(
    hits: list,
    query_vector: np.ndarray,
    embedder,
    total_token_budget: int = 700,
) -> tuple[list[str], dict]:
    """Compress every retrieved passage against a shared token budget.

    Budget is split proportionally to retrieval score: the passage that matched
    best earns the most room. A flat split would give a marginal fourth hit the
    same space as the one that actually answers the question.
    """
    if not hits:
        return [], {"tokens_before": 0, "tokens_after": 0}

    # Budget by RETRIEVAL RANK, not by dense score.
    #
    # Weighting by dense similarity looked reasonable until hybrid retrieval
    # arrived. The Newcastle disease passage was retrieved because BM25 ranked
    # it first on rare diagnostic terms - and it carried the LOWEST dense score
    # of the four hits. Proportional-to-dense budgeting therefore squeezed the
    # one passage that actually answered the question, and the model refused a
    # question its sources could answer.
    #
    # Rank already encodes the fused judgement of both rankers, and its decay
    # is gentle enough that a fourth-placed hit still gets a usable share.
    weights = np.array([1.0 / (i + 2) for i in range(len(hits))], dtype=np.float32)
    weights = weights / weights.sum()

    texts: list[str] = []
    before = after = forced_total = 0

    for hit, weight in zip(hits, weights):
        budget = max(60, int(total_token_budget * float(weight)))
        result = compress_passage(hit.chunk.text, query_vector, embedder, budget)
        texts.append(result.text)
        before += result.tokens_before
        after += result.tokens_after
        forced_total += result.safety_sentences_forced

    return texts, {
        "tokens_before": before,
        "tokens_after": after,
        "ratio": round(after / before, 3) if before else 1.0,
        "safety_sentences_forced": forced_total,
    }

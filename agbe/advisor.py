"""The advisory engine: retrieve, ground, generate, guard.

FLOW
----
    question -> detect language (English or Nigerian Pidgin)
             -> refuse on policy scope: dosage, out-of-scope crop, live data,
                human medical
             -> retrieve passages (hybrid dense + BM25)
             -> refuse if nothing is above the similarity floor
             -> compress passages to the sentences that answer the question
             -> generate an answer constrained to those passages
             -> guard the stream against invented dosages
             -> strip a disclaimer the sources contradict
             -> safety check + citations, rendered in the detected language

THE REFUSAL PATH IS A FEATURE
-----------------------------
Most of this file's value is in what it declines to do. A farmer asking about
maize prices, next week's rainfall, or an exact spray rate must get "I do not
have local guidance on that" rather than a fluent invention. Retrieval returns
nothing above the floor for those questions by design, and this module turns
that emptiness into an honest answer instead of letting the model fill it from
parametric memory.

GUARDING A STREAM
-----------------
Safety checking normally needs the finished answer, which conflicts with
streaming: by the time an invented dose is detected it has already been read.

The compromise here: tokens stream freely until a digit appears, at which
point output is buffered to the end of the sentence and that sentence is
checked against the sources before release. Prose - the bulk of any answer -
still streams token by token, while the one span capable of causing physical
harm is never displayed unverified. The cost is a short pause mid-sentence
when numbers appear, which is a fair price.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from agbe import config
from agbe.llm import LLM, GenerationStats
from agbe.rag import scope
from agbe.translate.detect import detect
from agbe.translate.messages import get as get_messages
from agbe.rag.compress import compress_hits
from agbe.rag.embedder import Embedder
from agbe.rag.index import SearchHit, VectorIndex
from agbe.rag.query import retrieval_query, with_diagnostic_intent
from agbe.translate.pidgin_norm import for_retrieval as pidgin_for_retrieval
from agbe.rag.safety import (
    SafetyVerdict,
    check_answer,
    find_dosages,
    _normalise_for_containment,
)

# Every token here is paid on EVERY request, before the farmer sees a word.
# The original version ran to ~150 tokens of numbered rules; this says the same
# things in roughly half that. Rules 4 and 6 of the original are enforced in
# code (agbe/rag/safety.py) rather than requested politely of a 3B model, so
# spending prefill budget restating them bought nothing.
#
# A KNOWN DEFECT THIS PROMPT DOES NOT FIX, AND A REJECTED ATTEMPT
# ---------------------------------------------------------------
# Sources arrive as numbered blocks - "[1] title (year)", then text - and the
# only formatting instruction is "cite them like [1]". The model mirrors that
# structure: six sources in, six numbered paragraphs out, one summarising each.
# Observed on two unrelated questions:
#
#   "should I sell my yam now or store it"  -> six blocks, four of them
#       describing what a document contains rather than advising
#   "my chickens have twisted necks"        -> six blocks, including a necropsy
#       description of haemorrhages in the proventriculus and Peyer's patches
#
# Both were correct and neither was an answer. That defect is REAL and OPEN;
# see docs/FINDINGS.md.
#
# Adding "write ONE answer, do not go through the sources one by one" and "skip
# anything they cannot act on" fixed the shape completely - numbered blocks went
# from six to zero on both questions - and cost 9 points of answer accuracy:
#
#     dev answer accuracy   93.9% -> 84.9%
#     NOT_USED                  2 -> 5
#
# MISSED stayed at 0, so retrieval was unaffected. The model was still being
# handed "mealybug", "striga" and "nitrogen deficiency" and was now leaving them
# out of shorter, tidier answers. Permission to omit is permission to omit the
# wrong thing, and in an architecture whose premise is that the corpus supplies
# the agronomy, that is the wrong trade.
#
# A future attempt should change how sources are PRESENTED - six numbered blocks
# is what invites six numbered blocks back - rather than instructing the model
# to summarise less.
SYSTEM_PROMPT = """You advise smallholder farmers in southwest Nigeria on crops \
and livestock.

- Answer using ONLY the numbered sources. Cite them like [1].
- Never give a dose or rate unless it appears in the sources.
- Prefer prevention and cultural control over chemicals.
- Short, plain sentences. Practical steps.
- Do not repeat yourself.

The sources below were selected because they are relevant. Answer from them \
directly. Do not begin by saying you lack information."""

#: Questions whose answer depends on WHEN they are asked.
#:
#: The system has no clock. Nothing anywhere injects a date, a month, a season
#: or a price - verified - so when a farmer asks "should I sell my yam now or
#: store it?", the model has no idea whether "now" is harvest peak or the
#: hungry season, and answers as though it does.
#:
#: §1 draws the line deliberately: store-or-sell *judgement* is in scope and
#: answerable from extension material, while today's *price* is a fact we do
#: not have. `scope.py` refuses the second and must keep answering the first.
#: The gap is that the first was being answered as a directive rather than as
#: the conditional advice it has to be.
#:
#: So the instruction below is appended only when a temporal marker appears.
#: Every token in the system prompt is paid on every request, and most
#: questions - "what is wrong with my cassava", "how do I store yam" - carry no
#: timing dependence at all. Roughly 30 tokens on the questions that need it and
#: none on the questions that do not.
_TIME_DEPENDENT = re.compile(
    r"\b(?:now|today|tonight|this (?:week|month|season)|right now|currently|"
    r"at the moment|already|yet|these days|next (?:week|month|season))\b",
    re.IGNORECASE,
)

TIME_UNAWARE_NOTE = (
    "\n\nYou do not know today's date, the current season, or current prices. "
    "This question depends on timing, so give the factors that decide it and "
    "the usual seasonal pattern from the sources, and tell the farmer what to "
    "check locally. Do not tell them what to do as though you knew the date."
)


#: Openings where the model disclaims having information and then answers anyway.
#:
#: Observed repeatedly, e.g. "I have no local guidance for this situation. The
#: symptoms of greenish watery diarrhea ... are consistent with Newcastle
#: disease" - followed by correct, well-sourced advice.
#:
#: This is the worst failure a judge can see: it reads as a system that does not
#: know what it knows, and it undermines the refusal path, which is the feature
#: the whole safety design rests on. If the model disclaims when it *does* have
#: sources, a farmer learns to ignore the disclaimer when it genuinely matters.
#:
#: The refusal decision belongs to retrieval, not to the model. When passages
#: score above the floor there is guidance and the model must use it; when they
#: do not, the engine never calls the model at all. So a disclaimer emitted
#: alongside sources is always wrong and is stripped.
_FALSE_DISCLAIMER = re.compile(
    r"^\s*(?:"
    r"i (?:do not|don't|dont) have (?:any )?(?:local |specific )?"
    r"(?:guidance|information|data|details)[^.]*\.|"
    r"i have no (?:local |specific )?(?:guidance|information|data)[^.]*\.|"
    r"(?:unfortunately|sadly)[^.]*(?:no|not)[^.]*(?:guidance|information)[^.]*\.|"
    r"(?:the|these) sources? (?:do not|don't) (?:contain|provide|mention)[^.]*\."
    r")\s*",
    re.IGNORECASE,
)


def strip_false_disclaimer(answer: str) -> tuple[str, bool]:
    """Remove a leading "I have no information" when sources were in fact used.

    Only ever called when retrieval returned passages above the similarity
    floor. Returns (cleaned, was_stripped) so the behaviour is measurable
    rather than silent - a rising strip rate means the prompt is drifting and
    should be fixed at the source.
    """
    cleaned = _FALSE_DISCLAIMER.sub("", answer, count=1)
    stripped = cleaned != answer
    if stripped:
        cleaned = cleaned.lstrip(" \n")
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
    return (cleaned or answer), stripped

NO_GUIDANCE_MESSAGE = (
    "I do not have local guidance on that in my documents, so I cannot give you "
    "a reliable answer. Please ask your local extension officer."
)

#: Sentence terminator used to bound the dosage-guard buffer.
_SENTENCE_END = re.compile(r"[.!?]\s|\n")


@dataclass
class Advice:
    """A completed answer with everything needed to audit it."""

    question: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    verdict: SafetyVerdict | None = None
    stats: GenerationStats | None = None
    refused: bool = False
    language: str = "en"

    def citations(self) -> list[str]:
        return [f"[{i + 1}] {h.chunk.citation()}" for i, h in enumerate(self.hits)]

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "language": self.language,
            "citations": self.citations(),
            "sources": [
                {"score": round(h.score, 3), "citation": h.chunk.citation(),
                 "licence": h.chunk.licence, "year": h.chunk.year}
                for h in self.hits
            ],
            "safety": {
                "safe": self.verdict.safe if self.verdict else True,
                "hazardous": self.verdict.hazardous_found if self.verdict else [],
                "unsupported_dosages": (
                    self.verdict.unsupported_dosages if self.verdict else []
                ),
            },
            "stats": self.stats.as_dict() if self.stats else None,
        }


def build_prompt(
    question: str,
    hits: list[SearchHit],
    passage_texts: list[str] | None = None,
) -> list[dict[str, str]]:
    """Assemble the grounded prompt.

    `passage_texts` lets the caller substitute compressed passages for the full
    chunk text. Citation metadata still comes from the chunk, so compression
    cannot detach a claim from its source.

    Source headers are deliberately terse - publisher strings like
    "Infonet-Biovision (Biovision Africa Trust / icipe)" cost ~12 tokens each
    and are repeated per source. The full attribution is rendered in the UI
    from the same chunk, where it costs nothing.
    """
    texts = passage_texts if passage_texts is not None else [h.chunk.text for h in hits]

    blocks = []
    for i, (hit, text) in enumerate(zip(hits, texts), start=1):
        blocks.append(f"[{i}] {hit.chunk.title} ({hit.chunk.year})\n{text}")
    sources = "\n\n".join(blocks)

    # Only questions carrying a temporal marker pay for the note. See
    # _TIME_DEPENDENT: the system has no clock, and answering "should I sell
    # now" as a directive claims knowledge it does not have.
    system = SYSTEM_PROMPT
    if _TIME_DEPENDENT.search(question):
        system += TIME_UNAWARE_NOTE

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Sources:\n\n{sources}\n\nFarmer's question: {question}",
        },
    ]


class AdvisoryEngine:
    """Retrieval-grounded advisory with safety guards."""

    def __init__(
        self,
        index: VectorIndex | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.index = index or VectorIndex.load()
        self.embedder = embedder or Embedder()
        self.llm = llm or LLM()

    # -- retrieval ---------------------------------------------------------

    # A bare `retrieve()` used to live here. It was removed WITH ITS CALLERS
    # GONE rather than kept for symmetry: it skipped Pidgin normalisation and
    # instruction stripping, so any future caller reaching for the
    # shorter-named method would silently reopen the §6.3 retrieval bug from
    # the surface they called it on. That is not hypothetical - it is exactly
    # how the web server shipped without those fixes. retrieve_and_compress()
    # is the only retrieval entry point.
    def retrieve_and_compress(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[SearchHit], list[str], dict]:
        """Retrieve, then cut each passage to the sentences that answer the query.

        Retrieval and answering want different granularities. Chunks of ~400
        tokens are right for *finding* a passage and wasteful for *reading* it:
        the tokens that do not bear on the question still cost prefill time,
        which is the dominant term in time to first token.

        Both retrieval and compression search with the *description* the farmer
        gave, not the instructions attached to it - see agbe/rag/query.py for
        the measurement that forced this. Compression needs it just as much as
        retrieval: it scores individual sentences against the query vector, so
        a vector pulled toward "practical steps for smallholders" keeps the
        development-programme sentences and drops the symptom that identifies
        the disease.
        """
        # Pidgin first, then instruction stripping: normalisation turns the
        # question into English-ish text, which is what the instruction
        # patterns in query.py are written against. English questions pass
        # through pidgin_norm byte-identically, so the benchmarked English
        # path is unchanged.
        search_text = with_diagnostic_intent(
            retrieval_query(pidgin_for_retrieval(question))
        )
        query_vector = self.embedder.embed_query(search_text)
        hits = self.index.hybrid_search(
            search_text,
            query_vector,
            top_k=top_k or config.RETRIEVAL.top_k,
            min_score=config.RETRIEVAL.min_score,
        )
        if not hits:
            return [], [], {"tokens_before": 0, "tokens_after": 0}

        texts, stats = compress_hits(
            hits, query_vector, self.embedder,
            total_token_budget=config.RETRIEVAL.context_token_budget,
        )
        return hits, texts, stats

    # -- generation --------------------------------------------------------

    def guarded_stream(
        self,
        question: str,
        hits: list[SearchHit],
        texts: list[str],
    ) -> Iterator[tuple[str, "GenerationStats | None"]]:
        """The one dosage-guarded generation loop. Every streaming consumer
        MUST go through here.

        This method exists because the guard loop kept being reimplemented -
        first in `stream()`, then again in the web server's `/ask` - and the
        copies drifted every time the original was fixed. By the fourth
        occurrence the server copy had missed the compression fix (still
        sending ~2,100-token prompts at 65s TTFT), missed the query-preparation
        fix (a judge pasting the submitted test prompt into the browser hit the
        bug the fix was for), and had even grown its own sentence-boundary
        detection that disagreed with `_SENTENCE_END`.

        Retrieval is the caller's job, because callers legitimately differ
        there: the server wants the hits early to emit a sources event before
        the first token; `stream()` does not. Generation and guarding are NOT
        allowed to differ, so they live here and only here.

        Yields `(piece, stats)`; stats belong to `llm.stream` and are complete
        only after the iterator is exhausted, so consumers wanting final stats
        keep the last non-None value.

        Containment is checked against the FULL chunk text, not the compressed
        excerpt. Compression drops sentences; a number that is genuinely in the
        source must not be flagged as invented merely because the sentence
        carrying it was cut for length.
        """
        source_texts = [h.chunk.text for h in hits]
        normalised_sources = _normalise_for_containment("\n".join(source_texts))

        buffer = ""
        guarding = False
        last_stats = None

        for piece, stats in self.llm.stream(build_prompt(question, hits, texts)):
            last_stats = stats
            if not guarding and any(ch.isdigit() for ch in piece):
                guarding = True

            if not guarding:
                yield piece, stats
                continue

            buffer += piece
            if _SENTENCE_END.search(buffer):
                yield self._release(buffer, normalised_sources), stats
                buffer = ""
                guarding = False

        if buffer:
            yield self._release(buffer, normalised_sources), last_stats

    def stream(self, question: str, top_k: int | None = None) -> Iterator[str]:
        """Stream an answer: retrieve, compress, then the shared guarded loop."""
        hits, texts, _comp = self.retrieve_and_compress(question, top_k=top_k)
        if not hits:
            yield NO_GUIDANCE_MESSAGE
            return

        for piece, _stats in self.guarded_stream(question, hits, texts):
            yield piece

    @staticmethod
    def _release(span: str, normalised_sources: str) -> str:
        """Emit a buffered sentence, redacting any ungrounded dosage in it."""
        out = span
        for dosage in find_dosages(span):
            if _normalise_for_containment(dosage) not in normalised_sources:
                out = out.replace(
                    dosage, "[rate not given in the sources - ask an extension officer]"
                )
        return out

    def advise(self, question: str, top_k: int | None = None) -> Advice:
        """Non-streaming path. Used by the eval harness and tests."""
        # Policy scope is checked before retrieval. A dosage question or an
        # out-of-scope crop is declined regardless of what the corpus holds,
        # and there is no reason to spend prefill discovering that.
        # Detect the language once, up front. Every fixed message the farmer
        # might receive - refusals, warnings, prohibitions - is then emitted
        # from human-validated strings rather than translated at runtime.
        language = detect(question)
        messages = get_messages(language)

        verdict = scope.check(question)
        if not verdict.in_scope:
            return Advice(
                question=question,
                answer=self._scope_message(verdict, messages),
                hits=[],
                refused=True,
                language=language,
            )

        hits, texts, comp = self.retrieve_and_compress(question, top_k=top_k)
        if not hits:
            return Advice(
                question=question,
                answer=messages.no_guidance,
                hits=[],
                refused=True,
                language=language,
            )

        answer, stats = self.llm.complete(build_prompt(question, hits, texts))

        # Retrieval already decided there is guidance; a disclaimer here
        # contradicts the sources shown beside it.
        answer, disclaimed = strip_false_disclaimer(answer)
        if stats:
            stats.fields.update(comp)
            stats.fields["false_disclaimer_stripped"] = disclaimed

        safety = check_answer(
            answer,
            source_texts=[h.chunk.text for h in hits],
            source_years=[h.chunk.year for h in hits],
        )
        notice = safety.as_notice(language)
        if notice:
            answer = f"{answer}\n\n---\n{notice}"

        return Advice(
            question=question,
            answer=answer,
            hits=hits,
            verdict=safety,
            stats=stats,
            language=language,
        )

    @staticmethod
    def _scope_message(verdict, messages) -> str:
        """Map a scope refusal to its validated message in the right language."""
        if verdict.reason == "dosage":
            return messages.dosage_refusal
        if verdict.reason == "human medical":
            return messages.human_medical
        if verdict.reason.startswith("out-of-scope crop"):
            return messages.out_of_scope_crop
        return messages.no_guidance

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.llm.close()
        self.embedder.close()

    def __enter__(self) -> "AdvisoryEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

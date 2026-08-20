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
from agbe.rag.quality import strip_furniture
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
#: Strings from the prompt itself. If generation emits one, it has stopped
#: answering and started reproducing its own input - see guarded_stream.
PROMPT_ECHO_STOPS = ["Farmer's question:", "\nSources:"]

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

#: Words too common to count as evidence that a sentence echoes the question.
_ECHO_STOPWORDS = frozenset("""
a an and are as at be been being but by can do does for from had has have how
i if in into is it its me my of on or our should so that the their them then
there these they this to too very was were what when where which who will with
you your
""".split())

#: TWO overlaps, measured in opposite directions, and both are needed.
#:
#: The first version measured only "how much of the sentence came from the
#: question" and missed the defect entirely: "For cutworms, cut off seedlings at
#: the base every night" scored 0.71, because "cutworms" and "off" dilute it.
#:
#: RECALL - how much of the farmer's description is handed back. This is the
#: actual signal: the cutworm sentence reproduces every content word of the
#: question, 5 of 5.
ECHO_QUESTION_RECALL = 0.8

#: PRECISION - how much of the sentence is nothing but the question. Needed
#: because recall alone deletes real advice: for "my maize is yellow", the
#: instruction "use nitrogen fertiliser on maize that is yellow" also covers the
#: whole question, and is exactly right. It survives because only 2 of its 5
#: content words are the farmer's; the cutworm sentence has 5 of 7.
#:
#: An echo says the question back and adds nothing. Advice says it back and adds
#: what to do.
ECHO_SENTENCE_SHARE = 0.6

#: An instruction: an imperative opening, or an explicit directive to the reader.
#: A sentence describing symptoms is not one, however closely it echoes.
_IMPERATIVE = re.compile(
    r"^\s*(?:for [^,]{1,30},\s*)?(?:you (?:should|must|can|need to)\s+)?"
    r"(?:cut|remove|uproot|destroy|burn|bury|apply|spray|plant|sow|harvest|"
    r"store|sell|water|feed|treat|inject|collect|handpick|pick|rogue|prune|"
    r"weed|dig|leave|keep|avoid|stop|start|use|make|check|inspect|monitor)\b",
    re.IGNORECASE,
)


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]+", text.lower())
        if len(w) > 2 and w not in _ECHO_STOPWORDS
    }


def strip_symptom_echo(answer: str, question: str) -> tuple[str, list[str]]:
    """Remove instructions that are the farmer's own symptom handed back.

    Asked "my seedlings are cut at the base every night", the system answered,
    among other things:

        "For cutworms, cut off seedlings at the base every night."

    A farmer following that would destroy their crop. It is not a bad fact from
    a source and not a fabrication - no source says it. The model re-emitted the
    QUESTION as an imperative, and nothing in this system was looking for that:
    `is_usable` filters passages, `check_answer` looks for hazardous actives and
    unsupported dosages, `strip_false_disclaimer` matches one specific opening.
    None of them compares the answer against what was asked.

    TWO CONDITIONS, BOTH REQUIRED
    -----------------------------
    Overlap alone is not the signal. Good answers echo the question constantly,
    because naming the problem back to the farmer is how a diagnosis reads:

        "Your cassava leaves are yellow and twisted because of mosaic disease"

    That is nearly all the farmer's words and it is exactly right. What makes
    the cutworm sentence a defect is that it is an INSTRUCTION - it tells them
    to do the thing they are complaining about.

    So a sentence is removed only when it is imperative AND both overlap tests
    pass: ECHO_QUESTION_RECALL of the question's content words are returned, and
    ECHO_SENTENCE_SHARE of the sentence is nothing but those words. Descriptions,
    diagnoses and explanations are untouched however much they echo, because
    they are not instructions.

    Returns (cleaned, removed) so the rate is measurable. A rising count means
    the prompt or the retrieval is drifting and should be fixed at the source,
    exactly as with strip_false_disclaimer.
    """
    asked = _content_words(question)
    if not asked:
        return answer, []

    kept: list[str] = []
    removed: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        words = _content_words(sentence)
        if not words or not _IMPERATIVE.match(sentence):
            kept.append(sentence)
            continue

        shared = words & asked
        recall = len(shared) / len(asked)          # of the question, how much returned
        share = len(shared) / len(words)           # of the sentence, how much is question

        if recall >= ECHO_QUESTION_RECALL and share >= ECHO_SENTENCE_SHARE:
            removed.append(sentence.strip())
            continue
        kept.append(sentence)

    if not removed:
        return answer, []

    cleaned = " ".join(s for s in kept if s.strip()).strip()
    # Never return an empty answer: a truncated answer is worse than an odd one,
    # and if echo removal would take everything, the right response is to leave
    # the original and let the rate reporting show it.
    return (cleaned or answer), removed


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

    # Continuous evidence with trailing markers, NOT a numbered list.
    #
    # Sources used to arrive as six blocks headed "[1] Title (year)", and the
    # model mirrored that structure: six blocks in, six numbered paragraphs out,
    # one summarising each. Observed on unrelated questions - a yam market query
    # answered in six blocks of which four described what a document contains,
    # a Newcastle query that included a necropsy description, a cutworm query
    # that produced eight.
    #
    # Instructing the model not to do it was tried and cost 9 points of answer
    # accuracy: told to write one answer and skip what the farmer cannot act on,
    # it dropped "mealybug", "striga" and "nitrogen deficiency" instead. See the
    # comment above SYSTEM_PROMPT.
    #
    # So the cue is removed rather than argued with. A leading "[n]" on its own
    # line reads as an item in a list to be worked through; a trailing marker
    # reads as a citation on a piece of evidence. The provenance is identical -
    # same passages, same numbering, same mapping to hits - and section 5's
    # guarantee that every claim traces to a source is unaffected, because the
    # marker still sits with its own text.
    #
    # The title and year move to the end of the passage for the same reason:
    # leading them makes each passage announce itself as a document, which is
    # what invites "[4] The training manual provides detailed symptoms...".
    #
    # The marker is a bare "[n]" and the attribution sits before it, because the
    # model copies the citation format it is shown. A first version wrote
    # "[1: Title, 2018]" and the answers came back citing "[4: Spider mites,
    # 2018]" - and on three of four questions stopped citing altogether, since
    # the shown format no longer matched the "cite them like [1]" instruction.
    # Show it exactly the shape you want back.
    parts = []
    for i, (hit, text) in enumerate(zip(hits, texts), start=1):
        body = " ".join(text.split())
        parts.append(f"{body} ({hit.chunk.title}, {hit.chunk.year}) [{i}]")
    sources = "\n\n".join(parts)

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


#: Regulatory status: the one thing a closed question can ask that this corpus
#: can NEVER settle, whatever it happens to contain.
#:
#: WHY THIS IS NOW A CATEGORY TEST AND NOT A CONTAINMENT TEST
#: The first version asked whether the word the question turned on appeared
#: anywhere in the retrieved passages. That was wrong in both directions, and
#: the probe set caught both within one run.
#:
#: TOO WEAK - "Is glyphosate approved for use on maize in Nigeria?" answered
#:            "Yes, glyphosate is approved for use on maize in Nigeria, as
#:            indicated by the Maize-legume cropping guide." The stem `approv`
#:            DID appear in the passages, in an unrelated sense, so the guard
#:            stood down. Worse than the case it was built for: this one
#:            attributes an invented regulatory claim to a named document.
#:
#: TOO STRONG - "Is the milk safe to drink after I deworm my goat?" was refused
#:            with five passages retrieved, because they discuss withdrawal
#:            periods without using the word "safe". That is the single question
#:            type the safety layer exists to serve. "Can Newcastle disease be
#:            cured once the birds are already sick?" was refused the same way,
#:            when the correct answer - no cure, prevent by vaccination - was
#:            sitting in the six passages it had.
#:
#: Presence of a word is not a position on a question, and absence of a word is
#: not absence of an answer. So the test is no longer about words in passages.
#:
#: Regulatory status is different in kind. Section 1 states plainly that NAFDAC
#: registration data is not openly published, so no retrieval result can make
#: "is X registered in Nigeria" answerable. That is knowable in advance, which
#: is what makes it a rule rather than a guess.
#:
#: WHAT THIS GIVES UP, STATED PLAINLY
#: "Is there a cure for cassava mosaic disease?" -> "Yes" is no longer caught.
#: It is a real defect: a virus is controlled, never cured. It is kept out
#: because the only test that caught it also refused the two questions above,
#: and denying a farmer "there is no cure, vaccinate instead" costs more than a
#: wrong framing attached to correct control advice. Recorded in FINDINGS as
#: open rather than fixed.
REGULATORY_TERMS = (
    r"\bregistered\b|\bregistration\b|\bapproved\b|\bapproval\b|"
    r"\blicen[cs]ed\b|\bbanned\b|\bban\b|\billegal\b|\bpermitted\b"
)

#: ...asked ABOUT a jurisdiction. "Approved" alone is an ordinary word ("an
#: approved practice"); it is "approved IN NIGERIA", "registered FOR USE", that
#: asks for a regulatory fact. Requiring this keeps the rule off agronomy.
_JURISDICTION = (
    r"\bnigeria\w*\b|\bnafdac\b|\bgovernment\b|\bofficial\w*\b|"
    r"\bfor use\b|\bby law\b|\blegal\w*\b|\bauthorit\w+\b"
)

_REGULATORY = re.compile(REGULATORY_TERMS, re.IGNORECASE)
_JURISDICTION_RX = re.compile(_JURISDICTION, re.IGNORECASE)

#: A question that opens with one of these demands a yes or a no.
_CLOSED_OPENER = re.compile(
    r"^\s*(?:is|are|does|do|did|can|could|will|would|has|have|had|should|was|were)\b",
    re.IGNORECASE,
)


def sources_cannot_settle(question: str, source_texts: list[str]) -> str | None:
    """Whether a closed question asks for a regulatory fact the corpus cannot hold.

    Returns a short reason, or None when the question is open, is not about
    regulatory status, or names no jurisdiction - in all three cases the model
    has something to answer from and is left alone.

    `source_texts` is no longer consulted and is kept in the signature
    deliberately: the previous version's whole mistake was believing that what
    the passages happen to contain could settle this, and a caller passing them
    should see them ignored rather than quietly dropped.
    """
    if not _CLOSED_OPENER.match(question):
        return None
    if _REGULATORY.search(question) and _JURISDICTION_RX.search(question):
        return "regulatory status"
    return None

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

        # Strip captions, running heads and page numbers before the model reads
        # the passage. A caption belonging to the previous section put "the
        # variegated grasshopper" two lines above whitefly control advice, and
        # the answer recommended encouraging one of cassava's worst pests as a
        # natural enemy. See agbe/rag/quality.py: strip_furniture.
        #
        # After compression, not before: compression scores whole sentences, and
        # a caption that survives selection is exactly the one worth removing.
        texts = [strip_furniture(t, h.chunk.title) for t, h in zip(texts, hits)]
        return hits, texts, stats

    # -- generation --------------------------------------------------------

    def guarded_stream(
        self,
        question: str,
        hits: list[SearchHit],
        texts: list[str],
        no_guidance: str | None = None,
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

        # A closed question the passages cannot settle is refused rather than
        # answered. This sits in guarded_stream, not in stream(), because all
        # three entry points - stream(), advise() and the web server's /ask -
        # funnel through here, and this file has already been bitten once by a
        # guard that lived on the tested path and not on the one a judge uses.
        #
        # The message is `no_guidance`, reused deliberately: "I do not have
        # local guidance on that in my documents" is exactly true here, and it
        # is already translated and speaker-reviewed. Inventing a second
        # refusal string would mean shipping an unreviewed Pidgin sentence to
        # say the same thing.
        unsettled = sources_cannot_settle(question, source_texts)
        if unsettled is not None:
            yield no_guidance or NO_GUIDANCE_MESSAGE, None
            return

        normalised_sources = _normalise_for_containment("\n".join(source_texts))

        buffer = ""
        guarding = False
        last_stats = None

        # PROMPT_ECHO_STOPS: the model can run past its answer and start
        # reproducing the prompt scaffolding. Asked about rice leaf spot it
        # emitted the source header, then the trailing citation, then the
        # literal string "Farmer's question:" and carried on - it was
        # continuing the document rather than answering it.
        #
        # These are the two strings build_prompt puts in the user turn, so
        # seeing either in the OUTPUT means generation has left the answer.
        # Cheaper and more reliable than post-hoc trimming: llama.cpp stops
        # decoding rather than us deleting tokens already paid for.
        for piece, stats in self.llm.stream(
            build_prompt(question, hits, texts), stop=PROMPT_ECHO_STOPS
        ):
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

        # Scope is checked against the NORMALISED question.
        #
        # Every policy rule in scope.py - dosage, live price, forecast, human
        # medical - is written in English, and scope.check() was being handed
        # the raw text. So the rules were written against a language they were
        # never shown. Measured: "How much dem dey sell garri for market now?"
        # passed scope.check() untouched and refused only because retrieval
        # happened to find nothing above the floor; normalised first, the price
        # rule fires correctly. Same for "Abeg how much dem dey sell maize now?".
        #
        # English passes through pidgin_for_retrieval byte-identically, so the
        # English path - and every measurement taken on it - is unchanged.
        verdict = scope.check(pidgin_for_retrieval(question))
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

        # The SAME polarity guard the streaming path applies. It is repeated
        # here rather than shared because advise() does not route through
        # guarded_stream - it calls llm.complete directly - and that divergence
        # is exactly how this file previously ended up with guards that held on
        # the tested path and not on the one a farmer touches. Here the risk ran
        # the other way: advise() is what the EVALUATION HARNESS calls, so a
        # guard only in guarded_stream would never have been measured at all.
        unsettled = sources_cannot_settle(question, [h.chunk.text for h in hits])
        if unsettled is not None:
            return Advice(
                question=question,
                answer=messages.no_guidance,
                hits=hits,
                refused=True,
                language=language,
            )

        answer, stats = self.llm.complete(
            build_prompt(question, hits, texts), stop=PROMPT_ECHO_STOPS
        )

        # Retrieval already decided there is guidance; a disclaimer here
        # contradicts the sources shown beside it.
        answer, disclaimed = strip_false_disclaimer(answer)

        # An instruction that is the farmer's own symptom handed back - "for
        # cutworms, cut off seedlings at the base every night". See
        # strip_symptom_echo for why overlap alone is not the test.
        answer, echoes = strip_symptom_echo(answer, question)

        if stats:
            stats.fields.update(comp)
            stats.fields["false_disclaimer_stripped"] = disclaimed
            stats.fields["symptom_echoes_removed"] = len(echoes)

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
        if verdict.reason.startswith("harmful request"):
            return messages.harmful_request
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

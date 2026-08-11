"""Separate what the farmer described from what they asked us to do with it.

WHY THIS EXISTS
---------------
The system's own submitted test prompt reads:

    "A smallholder farmer in Oyo State, Nigeria says: my cassava leaves are
     yellow and twisted, and the plants are small. Explain the most likely
     cause, and give practical steps the farmer can take this week. Prefer
     cultural and preventive measures over chemicals, and do not state any
     pesticide dose."

Asked this, Àgbẹ̀ replied that the cause was "delayed harvesting due to lack of
a ready market". That is textbook cassava mosaic disease, and the answer would
cost a farmer their crop.

The model was not at fault, and neither was the corpus - which holds 155
passages on cassava mosaic. Retrieval was, and measurement showed exactly why:

    query                     what ranked first             score
    the full prompt           climate value-chain reports   0.775
    the symptoms alone        "Disease control in cassava
                               farms: IPM field guide"      0.714

"Smallholder farmer", "Oyo State, Nigeria", "practical steps", "cultural and
preventive measures" is the vocabulary of development-agency reporting, and it
matches development-agency reports - which is what a value-chain adaptation
study is. Those phrases make up most of the prompt. The four words carrying the
diagnosis, "yellow and twisted", are a small minority of the embedding and are
outvoted.

The insight is that a prompt contains two different things:

  * a DESCRIPTION of a situation, which is what we should search for;
  * INSTRUCTIONS to the answerer about tone, format and constraints, which
    have nothing to do with which passage is relevant.

Embedding both together lets instructions steer retrieval. This module keeps
the description for the retriever and lets the instructions go on to the model,
where they belong.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not touch the text the safety and scope layers see. Those must judge
the whole prompt: "do not state any pesticide dose" is an instruction, but
"what dose of glyphosate" is a refusal trigger, and a stripper tuned for
retrieval must never become the thing that decides what is safe. `advise()`
passes the original text to the guards and only the reduced text here to the
index.

It is also conservative by construction. If stripping would leave nothing
recognisable, the original query is returned unchanged: worse retrieval is
recoverable, an empty query is not.
"""

from __future__ import annotations

import re

#: A persona frame introducing reported speech: "A farmer in Oyo State says:".
#: Everything up to and including the colon is scene-setting by whoever wrote
#: the prompt, not something the farmer observed.
_REPORTED_SPEECH = re.compile(
    r"^.{0,160}?\b(?:says?|asks?|writes?|reports?|complains?|tells? (?:me|us))\b\s*[:,-]\s*",
    re.IGNORECASE | re.DOTALL,
)

#: Clauses addressed to the answerer rather than describing the situation.
#:
#: Anchored at the start of a clause so they cannot fire mid-sentence: a farmer
#: writing "the leaves explain nothing" keeps their words. "Give" is included
#: because "give practical steps" is an instruction, while a farmer's own
#: "what do I give my goats" is a question, not a leading imperative.
_INSTRUCTION = re.compile(
    r"^\s*(?:and\s+|also\s+|but\s+|then\s+)?(?:please\s+)?(?:"
    r"explain|describe|list|outline|summari[sz]e|suggest|recommend|advise|"
    r"give|provide|tell\s+me|answer|respond|reply|write|"
    r"prefer|favou?r|focus\s+on|keep\s+it|limit|restrict|"
    r"do\s+not|don't|avoid|never|without\s+(?:stating|giving|mentioning)"
    r")\b",
    re.IGNORECASE,
)

#: Split on sentence ends, and on the comma-and joins that chain instructions
#: onto a description ("...are small, and give practical steps").
_CLAUSE = re.compile(r"(?<=[.!?])\s+|,\s+(?=and\s+(?:do|give|prefer|explain|list)\b)")

#: Below this many characters the reduced query is treated as over-stripped and
#: the original is used instead. A handful of words is not a description.
MIN_USEFUL_CHARS = 15


def retrieval_query(text: str) -> str:
    """The part of `text` worth searching the corpus for.

    Returns `text` unchanged when nothing is confidently removable, so this can
    be applied to every query without a flag.
    """
    stripped = _REPORTED_SPEECH.sub("", text.strip(), count=1)

    kept = [c for c in _CLAUSE.split(stripped) if c.strip() and not _INSTRUCTION.match(c)]
    reduced = " ".join(c.strip() for c in kept).strip()

    if len(reduced) < MIN_USEFUL_CHARS:
        # Nothing but instructions, or the split misfired. A short query the
        # farmer actually typed ("cassava leaves yellow") is handled by the
        # branch above returning it unchanged; reaching here means we removed
        # too much, and the original is the safer input.
        return text.strip()
    return reduced

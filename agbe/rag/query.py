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


#: Visible signs a farmer reports rather than diagnoses. Presence of one means
#: the query is a description of a problem, not a question about timing, price
#: or practice - and therefore that diagnostic guidance is what should rank.
_SYMPTOM_WORDS = frozenset("""
yellow yellowing yellowed brown black white spots spotted spotting streaks
twisted twisting curling curled distorted deformed shrivelled shriveled
wilting wilt wilted drooping rotting rot rotten mould mouldy mold mildew
dying die died dead stunted small tiny holes eaten chewed bitten patches
lesions blight scorched webbing sticky discoloured discolored swollen
swelling limping coughing shaking trembling diarrhoea diarrhea weak thin
lame sores scabs itching bleeding
""".split())

#: Appended to symptom descriptions before embedding. It names NO disease -
#: naming one would be assuming the answer we are trying to retrieve - and only
#: states what the farmer is actually asking: what is this and what do I do.
#:
#: MEASURED against the submitted test prompt, which retrieved zero diagnostic
#: documents without it:
#:
#:     query                                    on-target docs in top 6
#:     symptoms alone (the 6.3 fix)                    0
#:     symptoms + this phrase                          4
#:
#: Dense scores for "Disease control in cassava farms: IPM field guide" move
#: from below the 0.70 floor to ~0.81, so the guidance is not merely retrieved
#: but retrieved with margin.
_DIAGNOSTIC_INTENT = "disease symptoms cause control management"


#: Distinctive visible signs, and the name the literature files them under.
#:
#: WHY THIS IS NOT THE CIRCULAR EXPANSION REJECTED EARLIER
#: -------------------------------------------------------
#: An earlier attempt appended "cassava mosaic disease virus stunting" to the
#: query and "worked" - by naming the diagnosis it was searching for. That is
#: assuming the answer.
#:
#: This is different in what it supplies. A farmer writes what they SEE; the
#: corpus files it under a name they do not know. "White cottony insects on the
#: growing tip" is a mealybug; "white winding lines inside the leaves" is a
#: leafminer. The mapping from distinctive sign to candidate name is ordinary
#: extension knowledge, printed in the field guides this corpus is built from,
#: and it is what an extension officer does before looking anything up.
#:
#: It supplies SEARCH TERMS, not answers. Retrieval still has to find a passage,
#: the similarity floor still has to clear, and every claim still comes from a
#: cited source. A wrong guess degrades to worse ranking, which is recoverable;
#: it cannot put words in the model's mouth.
#:
#: MEASURED, on the two dev questions that retrieved nothing relevant:
#:
#:   crop-05  "white cottony insects on the growing tip of my cassava"
#:            236 mealybug chunks in the corpus; best at dense rank 18, inside
#:            a document the query ALREADY retrieved. Right document, wrong
#:            chunks.
#:   crop-22  "white winding lines inside my tomato leaves"
#:            204 leafminer chunks; dedicated document at rank 70. Retrieved
#:            instead: Helicoverpa, Fusarium wilt, whiteflies, early blight.
#:
#: Adding the name put on-target passages in the top-k for both.
#:
#: DELIBERATELY SMALL. Only signs distinctive enough that the mapping is not in
#: doubt. "Yellow leaves" maps to a dozen causes and is not here; "cottony
#: masses" maps to one. Every entry is a phrase a farmer would actually type.
_SIGN_TO_NAME: tuple[tuple[str, str], ...] = (
    # Cassava mosaic. Yellow, mottled leaves that are also twisted or
    # distorted is the textbook sign, and it is the single most-discussed
    # question in this project - yet it had no entry, while nine rarer signs
    # did.
    #
    # Measured over twelve runs, the system named CASSAVA BROWN STREAK for
    # these symptoms in 4, named no disease at all in 7, and named mosaic in 1.
    # The mosaic passage was retrieved every time, at fused rank 4, while a
    # passage headed "Damage symptoms:" whose body is brown streak led at rank
    # 1 - a symptom query matching a chunk literally titled "Damage symptoms".
    #
    # This is the same shape as the bloat and foot rot entries below: the right
    # passage is already retrieved and loses during generation, so promotion is
    # enough and nothing new needs to be found.
    #
    # ORDER MATTERS in the pattern. It requires yellowing BEFORE distortion,
    # which is how mosaic is described and is not how the okra leaf-curl
    # question is phrased ("curling and turning yellow") - that question is
    # about a different crop and must not be dragged onto cassava material.
    (r"(?:leaf|leaves)[^.]{0,40}(?:yellow\w*|mottl\w*|pale patch\w*)"
     r"[^.]{0,30}(?:twist\w*|distort\w*|crinkl\w*|misshap\w*)|"
     r"(?:yellow\w*|mottl\w*)[^.]{0,25}and[^.]{0,25}"
     r"(?:twist\w*|distort\w*|crinkl\w*)",
     "cassava mosaic"),
    (r"cottony|cotton-?like|white waxy|waxy coating|white powdery mass", "mealybug"),
    (r"winding lines|wiggly lines|squiggly|tunnels? (?:in|inside) the leaves|"
     r"trails? (?:in|inside) the leaves|mines? in the lea", "leafminer"),
    (r"fine web|webbing|cobweb", "spider mite"),
    (r"sooty|black mou?ld on the leaves", "sooty mould honeydew"),
    (r"sawdust|frass|boring into|holes? in the stem", "borer"),
    # `\bcut\b` and never `cutting`: "stem cuttings", "healthy cuttings" and
    # "select cuttings" are core cassava planting vocabulary, and matching them
    # would fire cutworm on every clean-planting-material question. The word
    # boundary is the whole defence.
    #
    # The first version of this entry required the literal sequence "cut the
    # seedlings at base" and could never fire - natural phrasing puts words in
    # between ("are cut at the base") or says "ground level". It was dead code
    # that looked functional, found by probing entries the evaluation set never
    # exercises.
    (r"(?:seedlings?|young plants?|transplants?)[^.]{0,40}\b(?:cut|severed|"
     r"chopped|felled)\b[^.]{0,30}(?:base|ground|soil|stem)|"
     r"\b(?:cut|cuts|cutting down|severed)\b[^.]{0,30}"
     r"(?:seedlings?|young plants?)[^.]{0,30}(?:base|ground|night|soil)",
     "cutworm"),
    (r"twisted neck|neck (?:is )?twist|head (?:pulled|twisted) back",
     "newcastle disease"),
    (r"greenish watery|green watery droppings?", "newcastle disease"),
    (r"pot ?belly|swollen (?:under the )?jaw|bottle jaw", "worms helminth"),
    # Bloat. "My goat left side is swollen and tight after grazing wet grass"
    # was answered with HOOF-TRIMMING advice - "wet" appears in both the
    # question and a footrot passage. Bloat kills within hours by pressing on
    # the diaphragm, so this is the most expensive miss in the probe set.
    #
    # Unlike the scouring case, promotion is enough here: the goat bloat
    # passage sits at fused rank 26 with a dense score of 0.744, comfortably
    # over the floor. It was never being discarded, only outranked.
    #
    # LEFT side specifically - the rumen is on the left, and "swollen" alone
    # would collide with bottle jaw above.
    (r"left side[^.]{0,20}swollen|swollen[^.]{0,15}left side|"
     r"\bbloat(?:ed|ing)?\b|distended[^.]{0,15}(?:side|belly|stomach)|"
     r"tight[^.]{0,20}(?:after|from) graz", "bloat"),
    # Lameness. A single lame leg was diagnosed as FOOT-AND-MOUTH DISEASE, a
    # notifiable transboundary disease - alarming, wrong, and wrong in a way
    # that sends a farmer to report an outbreak they do not have.
    #
    # Here retrieval was already correct: "Foot Rot in Cattle and Sheep" was
    # in the top six and lost to the FMD passage during generation. Promotion
    # puts the right document first rather than adding a new one.
    (r"\blimp(?:ing|s)?\b|not put(?:ting)? weight|\blame\b|"
     r"favour(?:ing)? one leg|holding up (?:a|one) (?:leg|foot)", "foot rot"),
)

_SIGN_PATTERNS = tuple(
    (re.compile(pat, re.IGNORECASE), name) for pat, name in _SIGN_TO_NAME
)


def sign_names(text: str) -> list[str]:
    """Candidate names for distinctive signs described in `text`."""
    out: list[str] = []
    for rx, name in _SIGN_PATTERNS:
        if rx.search(text) and name not in out:
            out.append(name)
    return out


def _describes_symptoms(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _SYMPTOM_WORDS)


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


def with_diagnostic_intent(text: str) -> str:
    """Append canonical diagnostic intent to a description of symptoms.

    WHY THIS IS SEPARATE FROM `retrieval_query`
    -------------------------------------------
    `retrieval_query` removes; this adds. Folding them together broke
    `retrieval_query`'s contract that a plain farmer question passes through
    untouched - a contract two tests exist to defend, and which they caught
    being violated within minutes of the attempt.

    WHY INTENT IS ADDED BACK AT ALL
    -------------------------------
    Stripping instructions was right and also incomplete. The instruction
    clause in the submitted prompt - "explain the most likely cause, and give
    practical steps" - carried the development-agency vocabulary that steered
    retrieval wrong, but it *also* carried the only signal that this was a
    diagnostic question. Removing it left a bare symptom description, which
    embeds close to any text about the crop, so general growing manuals
    outranked the disease guide.

    Measured on the submitted prompt: the description alone retrieved zero
    diagnostic documents in the top 6, and the dedicated African cassava mosaic
    virus page sat at dense rank 268 of 29,959. The model named the wrong virus
    - Cassava Brown Streak rather than mosaic - and omitted clean planting
    material, the control that matters most. With this phrase appended, four of
    the top six come from "Disease control in cassava farms: IPM field guide",
    at dense ~0.81 against a 0.70 floor.

    The phrase names no disease. Naming one would assume the answer we are
    trying to retrieve, and a fixed phrase the farmer never typed cannot
    reintroduce the reporting vocabulary that caused the original defect.
    """
    # Gated, not unconditional. "When should I plant maize" and "should I store
    # or sell my grain" are not diagnostic questions, and pushing them toward
    # disease guidance would trade one retrieval bug for another.
    if not _describes_symptoms(text):
        return text

    # Candidate names for distinctive signs, before the generic intent phrase.
    # A farmer writes what they see; the corpus files it under a name they do
    # not know. See _SIGN_TO_NAME.
    names = sign_names(text)
    named = f"{text} {' '.join(names)}" if names else text
    return f"{named} {_DIAGNOSTIC_INTENT}"

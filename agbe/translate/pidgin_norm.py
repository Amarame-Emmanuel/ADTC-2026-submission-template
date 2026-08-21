"""Normalise Nigerian Pidgin toward English, for retrieval only.

WHY NOT NLLB, WHICH WE ALREADY BUILT
------------------------------------
The NLLB-200 bridge was converted, wired and measured. It does not work for
this language, and the measurement is worth recording because the conclusion is
counter-intuitive:

    question (Pidgin)                        raw    via NLLB
    fowl dying, necks twisting               miss   miss
    cassava leaf yellow and twisting         miss   miss
    goat not eating, shaking                 miss   miss
    when to plant maize                      HIT    HIT
    stopping yam rotting in store            miss   miss

Zero improvement, for ~740 MB of resident memory (measured VmHWM) and a second
inference framework in the runtime.

The wiring was not the problem - an eng->fra control through the same code
returned "Mes feuilles de manioc sont jaunes et tortueuses", which is correct.
The problem is structural. Nigerian Pidgin is an English-lexified creole: most
of its words already *are* English words. NLLB's `pcm_Latn` sees input that
looks like English and copies it through untouched, which it did for three of
the five questions above.

Worse than useless in one case. "My fowl dem dey die, dem neck dey twist" - a
textbook description of Newcastle disease in a flock - came back as "My fowl
will die, my neck will twist". Present tense became future, and *their* necks
became *my* neck, turning a poultry question into a human medical symptom that
the scope guard is designed to refuse. A translation layer that can convert a
livestock question into a medical one is a safety regression, not a feature.

WHAT THIS DOES INSTEAD
----------------------
The same property that defeats NLLB makes the problem easy. Because Pidgin is
English-lexified, the gap to English is concentrated in a small, closed set of
high-frequency grammatical markers and a handful of everyday content words -
`dey`, `wetin`, `dem`, `go`, `chop`, `sabi`, `abi`, `don`. Mapping those is a
lookup table. It costs no memory, no framework and no load time, and unlike a
600M-parameter model it is auditable line by line by the same speaker who
validated the safety messages.

SCOPE, DELIBERATELY NARROW
--------------------------
This runs ONLY on text going to the retrieval index, exactly like
`agbe/rag/query.py`. It never rewrites what the farmer sees and never touches
what the safety and scope guards read - those must judge the words actually
typed. Normalisation is allowed to make retrieval better; it is not allowed to
decide what is safe.

It is also not a translator, and must not be described as one. It makes a
Pidgin question look more like the English the corpus is written in. Output is
frequently not fluent English, which does not matter: an embedding model is
reading it, not a person.
"""

from __future__ import annotations

import re

#: Ordered Pidgin -> English rewrites, longest and most specific first.
#:
#: Order matters: "no dey" must fire before "dey", or "no dey chop" becomes
#: "no is chop" instead of "is not eating".
#:
#: Every entry is a high-frequency marker or an everyday content word that
#: carries retrieval signal. Rare or ambiguous vocabulary is deliberately
#: absent - a wrong mapping here is invisible in the output and would quietly
#: mis-steer retrieval, which is the failure mode this module exists to fix.
_RULES: list[tuple[str, str]] = [
    # Multi-word markers, before their parts.
    # "Make I stop farming?" is "SHOULD I stop farming?" - a question about the
    # asker's own life. The general `make` rule below renders it "so that I",
    # which is right for purpose ("wetin I go do MAKE my yam no rotten") and
    # wrong here, and left the sentence in a shape no English rule could parse:
    # the personal-decision rule refused "Should I leave farming?" and answered
    # "Abi make I leave farming?".
    #
    # Ordered before `abi` and `make` because both would otherwise fire first.
    (r"\b(?:abi\s+)?make\s+i\b", "should I"),
    (r"\bwetin dey worry\b", "what is wrong with"),
    (r"\bwetin be\b", "what is"),
    (r"\bwetin\b", "what"),
    (r"\bhow far\b", "how is"),
    (r"\bno dey\b", "is not"),
    (r"\bnever\s+dey\b", "has not been"),
    (r"\bdem dey\b", "are"),
    (r"\be dey\b", "it is"),
    (r"\bi dey\b", "I am"),
    (r"\bdey\b", "is"),

    # Tense and aspect.
    (r"\bgo\s+(?=[a-z]+\b)", "will "),   # "I go plant" -> "I will plant"
    (r"\bdon\b", "has"),
    (r"\bwan\b", "want to"),
    (r"\bfit\b", "can"),

    # Pronouns and determiners. "dem" as a plural/possessive marker.
    (r"\bdem\b", "their"),
    (r"\buna\b", "your"),
    (r"\bpikin\b", "young one"),
    # "am" is the Pidgin object pronoun - "wetin dey worry am" is "what is
    # wrong with IT". The lookbehind protects the English "I am", including the
    # "I am" this module's own "i dey" rule produces earlier in the pass.
    (r"(?<!\bi )\bam\b", "it"),
    (r"\bdis\b", "this"),
    (r"\bdat\b", "that"),

    # Everyday content words that differ from English.
    (r"\bchop\b", "eat"),
    (r"\bfowl\b", "chicken"),
    (r"\bsabi\b", "know"),
    (r"\babi\b", "or"),
    (r"\bbelle\b", "stomach"),
    (r"\bwaka\b", "walk"),
    (r"\bplenty\b", "many"),
    (r"\brotten\b", "rot"),
    (r"\bfarm land\b", "farmland"),
    (r"\bsoso\b", "only"),
    (r"\bnaim\b", "it is"),

    # "na" is a copula: "na wetin?" / "na my maize". Handled after "wetin" so
    # the question form is already resolved.
    (r"\bna\b", "is"),

    # "make" introducing purpose: "wetin I go do make my yam no rotten".
    (r"\bmake\b", "so that"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _RULES]

#: Words whose presence means the text is Pidgin rather than English. Kept
#: separate from agbe/translate/detect.py on purpose: that module decides which
#: language to ANSWER in, this one decides whether rewriting would help. They
#: answer different questions and should be free to disagree.
_MARKERS = re.compile(
    r"\b(?:dey|wetin|abeg|sabi|una|abi|naim|pikin|comot|waka|oga)\b"
    # "make I" is unambiguous: English says "make ME". Without it, "Make I stop
    # farming?" carried no marker, was never normalised, and reached the scope
    # rules as Pidgin they cannot parse - so a question about whether to give up
    # farming was ANSWERED while "Should I leave farming?" refused.
    #
    # The boundary after "i" is what keeps "make it" and "make into" out.
    r"|\bmake i\b",
    re.IGNORECASE,
)


def looks_like_pidgin(text: str) -> bool:
    return bool(_MARKERS.search(text))


def normalise(text: str) -> str:
    """Rewrite Pidgin markers toward English. Safe to call on English text.

    English input is returned essentially unchanged, because the rules are
    anchored to words English does not use in these positions. The exception is
    `go`, which is why it only fires before another word.
    """
    out = text
    for pattern, replacement in _COMPILED:
        out = pattern.sub(replacement, out)
    return re.sub(r"\s+", " ", out).strip()


def for_retrieval(text: str) -> str:
    """Normalise only when the text actually looks like Pidgin.

    Applying the rules unconditionally would rewrite English questions that
    happen to contain "go" or "make" for no benefit. Gating on a marker keeps
    the English path - which is the measured, benchmarked one - byte-identical.
    """
    return normalise(text) if looks_like_pidgin(text) else text

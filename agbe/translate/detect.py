"""Language detection for incoming questions.

Only has to distinguish the languages this system actually supports, which is
a far easier problem than general language identification and needs no model.

NIGERIAN PIDGIN IS THE HARD CASE
--------------------------------
Pidgin shares almost all of its vocabulary with English, so word-level
frequency cannot separate them. What separates them is grammar, and Pidgin's
grammatical markers are short, extremely common, and absent from English:

    dey   progressive/habitual      "e dey grow"
    don   perfect                   "e don rotten"
    wey   relativiser               "di plant wey sick"
    na    copula/focus              "na mosaic disease"
    no    negation before verb      "I no sabi"
    abeg  please
    sabi  to know

A sentence containing two or more of these is Pidgin with high confidence, and
English text essentially never contains them - "dey", "wey" and "sabi" are not
English words at all.

BIASED TOWARDS ENGLISH ON PURPOSE
---------------------------------
Misrouting Pidgin to English costs a farmer a slightly stiff answer they can
still read - the two languages are mutually intelligible in writing. Misrouting
English to Pidgin produces something odd for no benefit. So detection requires
real evidence before switching, rather than guessing on one ambiguous word.
"""

from __future__ import annotations

import re

#: Grammatical markers that occur in Pidgin and effectively never in English.
#: Weighted: some are unambiguous, others ("no", "go", "make") appear in
#: English too and only count as evidence in combination.
_STRONG = {"dey", "wey", "sabi", "abeg", "una", "dem", "comot", "wetin",
           "waka", "chop", "oga", "biko", "naim", "dis", "dat", "di"}
_WEAK = {"na", "don", "no", "go", "make", "fit", "well well", "now now",
         "plenty", "small small"}

_WORD = re.compile(r"[a-z]+")

#: Strong markers needed on their own, or strong+weak in combination. Two
#: strong markers is a confident signal; one strong plus two weak is the
#: minimum that avoids firing on English containing "no" and "go".
_MIN_STRONG = 2
_MIN_STRONG_WITH_WEAK = 1
_MIN_WEAK_SUPPORT = 2


def detect(text: str) -> str:
    """Return a supported language code: 'pcm' or 'en'."""
    words = _WORD.findall(text.lower())
    if not words:
        return "en"

    unique = set(words)
    strong = len(unique & _STRONG)
    weak = len(unique & _WEAK)

    if strong >= _MIN_STRONG:
        return "pcm"
    if strong >= _MIN_STRONG_WITH_WEAK and weak >= _MIN_WEAK_SUPPORT:
        return "pcm"
    return "en"


def confidence(text: str) -> tuple[str, float]:
    """Detected language and a rough confidence in [0, 1].

    Exposed so the interface can show which language it thinks it heard. A
    farmer who is answered in the wrong language should be able to see why and
    override it, rather than being left guessing.
    """
    words = _WORD.findall(text.lower())
    if not words:
        return "en", 1.0

    unique = set(words)
    strong = len(unique & _STRONG)
    weak = len(unique & _WEAK)
    score = min(1.0, (strong * 0.4) + (weak * 0.15))

    return ("pcm", score) if detect(text) == "pcm" else ("en", 1.0 - score)

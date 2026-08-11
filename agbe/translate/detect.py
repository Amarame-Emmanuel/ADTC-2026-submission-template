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


# -- Yoruba and Igbo ---------------------------------------------------------
#
# The OPPOSITE problem from Pidgin. Pidgin shares its vocabulary with English
# and must be separated by grammar; Yoruba and Igbo are distinct languages
# with distinctive orthography, so a few characters settle it:
#
#   Yoruba only:  ṣ            (ṣe, iṣu, Ọ̀ṣun)
#   Igbo only:    ụ  ị  ṅ      (ụlọ, gịnị, aṅara)
#   Shared:       ẹ  ọ         (evidence of "not English", not of which)
#
# Both scripts also carry tone accents (à é ò ...), but so do French and
# Portuguese, so accents alone are never evidence here.
#
# Farmers typing on phones often OMIT diacritics, so each language also gets a
# small set of unambiguous words - common, and not English, Pidgin, or each
# other's. Two are required, matching the Pidgin rule's caution.
#
# DETECTING A LANGUAGE IS NOT CLAIMING TO SUPPORT IT. messages.get() falls
# back to English for any language without human-validated strings, so
# returning "yor" today changes nothing the farmer sees - it is the switch
# that validated Yoruba strings will flip on the day a speaker approves them,
# and it lets the interface say "we heard Yoruba" honestly in the meantime.
# That contract is pinned by a test.

#: Text is normalised to NFC before matching, so one form per character
#: suffices regardless of whether a keyboard emits precomposed characters or
#: base letters with combining dots.
_YORUBA_CHARS = ("ṣ",)                       # s with dot below
_IGBO_CHARS = ("ụ", "ị", "ṅ")      # u, i dot below; n dot above
#: Present in both orthographies: evidence the text is not English, without
#: saying which language it is.
_SHARED_CHARS = ("ẹ", "ọ")               # e, o with dot below

_YORUBA_WORDS = {"bawo", "jowo", "gbogbo", "sugbon", "nitori", "agbado",
                 "ninu", "pupo", "tabi", "ewure"}
_IGBO_WORDS = {"kedu", "gini", "anyi", "nke", "ihe", "ndi", "unu", "maka",
               "banyere", "akwukwo", "ewu", "oka"}

#: Distinct script characters required before script evidence decides.
_MIN_SCRIPT_CHARS = 2
#: Unambiguous words required when diacritics are absent.
_MIN_LANG_WORDS = 2


def _hits(low: str, chars: tuple[str, ...]) -> int:
    # Occurrences, not distinct characters: English text contains these
    # characters zero times, so two occurrences of the same letter (common -
    # "ṣe ... ṣe") are exactly as decisive as two different letters, while a
    # single stray character from a paste still stays below the threshold.
    return sum(low.count(c) for c in chars)


def detect(text: str) -> str:
    """Return a language code: 'yor', 'ibo', 'pcm' or 'en'.

    Order is deliberate: script evidence outranks everything (a sentence
    containing ṣ is not English or Pidgin, whatever else it contains),
    then Pidgin's grammar markers, then diacritic-free Yoruba/Igbo word
    evidence, then the English default. A tie in script evidence falls
    through to word evidence rather than guessing.
    """
    import unicodedata

    low = unicodedata.normalize("NFC", text).lower()

    yor = _hits(low, _YORUBA_CHARS)
    ibo = _hits(low, _IGBO_CHARS)
    shared = _hits(low, _SHARED_CHARS)
    if yor + ibo + shared >= _MIN_SCRIPT_CHARS:
        if yor > ibo:
            return "yor"
        if ibo > yor:
            return "ibo"
        # Only shared vowels, or a dead tie: not enough to name the language;
        # let word evidence below decide.

    words = _WORD.findall(low)
    if not words:
        return "en"

    unique = set(words)
    strong = len(unique & _STRONG)
    weak = len(unique & _WEAK)

    if strong >= _MIN_STRONG:
        return "pcm"
    if strong >= _MIN_STRONG_WITH_WEAK and weak >= _MIN_WEAK_SUPPORT:
        return "pcm"

    if len(unique & _YORUBA_WORDS) >= _MIN_LANG_WORDS:
        return "yor"
    if len(unique & _IGBO_WORDS) >= _MIN_LANG_WORDS:
        return "ibo"

    return "en"


def confidence(text: str) -> tuple[str, float]:
    """Detected language and a rough confidence in [0, 1].

    Exposed so the interface can show which language it thinks it heard. A
    farmer who is answered in the wrong language should be able to see why and
    override it, rather than being left guessing.
    """
    language = detect(text)

    if language in ("yor", "ibo"):
        # Script characters are near-certain evidence; word-only detection
        # (diacritics omitted) is confident but not certain.
        import unicodedata

        low = unicodedata.normalize("NFC", text).lower()
        script = _hits(low, _YORUBA_CHARS if language == "yor" else _IGBO_CHARS)
        return language, 0.95 if script else 0.75

    words = _WORD.findall(text.lower())
    if not words:
        return "en", 1.0

    unique = set(words)
    strong = len(unique & _STRONG)
    weak = len(unique & _WEAK)
    score = min(1.0, (strong * 0.4) + (weak * 0.15))

    return ("pcm", score) if language == "pcm" else ("en", 1.0 - score)

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
           "waka", "chop", "oga", "biko", "naim", "dis", "dat", "di",
           # "My goat pikin dey shit water since morning" carried one strong
           # marker ("dey") and no weak ones, so it fell one short of
           # _MIN_STRONG and was detected as English. No normalisation ran, raw
           # Pidgin went to retrieval, and the answer came back "retained
           # placenta" for a kid with diarrhoea. These four are unambiguous.
           "pikin", "pickin", "belle", "yansh"}
_WEAK = {"na", "don", "no", "go", "make", "fit", "well well", "now now",
         "plenty", "small small"}

#: Markers that essentially never occur in English, in any spelling.
#:
#: WHY THIS SET WAS SPLIT OUT
#: Sixteen Pidgin questions were put through the interface and only six were
#: detected as Pidgin. The ten misses shared one shape - ONE strong marker and
#: ONE weak one - which the thresholds below require two weak markers to accept:
#:
#:     "Abeg how I go take stop weevil for my maize?"   abeg + go
#:     "My chicken no dey lay egg again"                dey  + no
#:     "Wetin make my groundnut leaf get black spot?"   wetin + make
#:
#: That is the ordinary shape of a Pidgin sentence, not an edge case. A miss
#: costs twice: retrieval never sees the normalised text, and the farmer gets
#: English refusal messages.
#:
#: The fix is not simply a lower threshold. "dem", "dis", "dat" and "di" are
#: one keystroke from "them", "this", "that" and "the", so a single one of them
#: must not be enough. These, by contrast, have no English neighbour - one is
#: decisive on its own.
_UNAMBIGUOUS = {
    "dey", "wetin", "abeg", "wey", "sabi", "una", "comot", "waka",
    "oga", "biko", "naim", "pikin", "pickin", "yansh", "wan", "sef",
    "abi", "shebi", "kpatakpata",
}

#: Multi-word constructions that are unambiguously Pidgin.
#:
#: Seventeen of forty Pidgin questions were still detected as English after the
#: single-word markers were widened. The misses were not random - they share a
#: small set of grammatical constructions that carry no unambiguous marker WORD:
#:
#:     "My maize leaf get long grey mark FOR AM"     for + am
#:     "My cassava DON GET white thing"              don + verb
#:     "My yam sett NO GREE sprout"                  no gree
#:     "How I GO TAKE dry my maize?"                 go take
#:     "Which time better MAKE I plant yam?"         make I
#:
#: Each is a phrase, so none could ever be caught by a word-set intersection.
#: They are listed here rather than added to _UNAMBIGUOUS because a phrase is
#: safer than its parts: "for" and "am" and "get" are all ordinary English, and
#: only their combination is decisive.
#: Matched with WORD BOUNDARIES, not substring containment. "e be" is a
#: substring of "th-E BE-st", so a plain `in` test called "what is the best way
#: to store yam?" Pidgin. That is the third time this exact bug has appeared in
#: this codebase - `ban` in `banana`, `his` in `this`, and now this - so the
#: phrases are compiled rather than listed as bare strings.
_UNAMBIGUOUS_PHRASES = (
    "no gree", "go take", "make i", "for am", "for di", "for dem",
    "don get", "don begin", "don dey", "don start", "don come",
    "dey go", "dey come", "no be", "e be", "e dey", "e don",
    "wetin be", "how i go", "which one better", "make e",
)

_PHRASE_RX = tuple(
    re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
    for p in _UNAMBIGUOUS_PHRASES
)

#: Multi-word markers, matched against the TEXT rather than the word set.
#:
#: "well well", "now now" and "small small" sat in _WEAK where they could never
#: fire: matching is a set intersection over single words, so no phrase in that
#: set had ever matched anything. Dead entries that looked functional.
_WEAK_PHRASES = ("well well", "now now", "small small", "quick quick",
                 "sharp sharp", "una own")

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

#: Hausa, the third major Nigerian language.
#:
#: WHY HAUSA AND NOT TWI OR FRENCH
#: Two things separate it. It is NIGERIAN, so it does not collide with the
#: service-area rule that declines Ghanaian and Senegalese questions - Kano and
#: Kaduna are already on the "here" list. And `test_advisor.py` has tested
#: `get("hau")` falling back to English since before this was added, so the
#: fallback path was written with Hausa in mind.
#:
#: This adds DETECTION only, which is exactly the status Yoruba and Igbo have.
#: A detected language with no validated message set falls back to English
#: strings, so nothing a farmer sees is machine-translated. What changes is that
#: the English-only in-domain gate now steps aside for Hausa, as it does for the
#: other two, instead of refusing "Akuyata ba ta cin abinci" as out of domain.
#:
#: THE HOOKED LETTERS ARE DECISIVE
#: b-hook, d-hook, k-hook and the apostrophe-y are Hausa orthography and occur
#: in no European language, no English text, and neither Yoruba nor Igbo. One is
#: enough, on the same reasoning as s-underdot for Yoruba.
_HAUSA_CHARS = ("ɓ", "ɗ", "ƙ", "ƴ")   # b, d, k with hook; y with hook

#: ...but Hausa is very often typed without them, plain b/d/k, exactly as Yoruba
#: is typed without its underdots. These are the fallback: common words that are
#: not English, not Pidgin, and not Yoruba or Igbo. Farm vocabulary is included
#: because that is what this system will actually receive.
#:
#: Deliberately excluded: "da", "na", "ba", "ya" - all real Hausa particles, all
#: two letters, and all far too collidable to carry evidence.
#: Hausa words with no English, Pidgin, Yoruba or Igbo neighbour. ONE is
#: enough, on the same reasoning as the Pidgin _UNAMBIGUOUS set: requiring two
#: made "Akuya ta ba ta cin abinci" (my goat is not eating) and "Rogo na ya yi
#: rawaya" (my cassava is yellowing) fall through to English, and those are
#: exactly the sentences this system exists to answer.
_HAUSA_UNAMBIGUOUS = {
    "akuya", "awaki", "shanu", "kaza", "kaji", "kiwo",
    "masara", "rogo", "dawa", "gero", "gona", "noma", "girbi",
    "abinci", "rawaya", "taki", "yaya", "sannu", "saboda", "domin",
}

#: Weaker: real Hausa, but short or collidable enough to need support.
_HAUSA_WORDS = {
    "ina", "yaya", "kuma", "sannu", "yanzu", "kada", "domin", "saboda",
    "gona", "noma", "shuka", "girbi",          # farm, farming, plant, harvest
    "masara", "rogo", "dawa", "gero",          # maize, cassava, sorghum, millet
    "shanu", "akuya", "awaki", "kaza", "kaji",  # cattle, goat(s), hen, chickens
    "ruwa", "kasa", "iri", "taki",             # water, soil, seed, fertiliser
}

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

    if _hits(low, _HAUSA_CHARS):
        return "hau"

    yor = _hits(low, _YORUBA_CHARS)
    ibo = _hits(low, _IGBO_CHARS)
    shared = _hits(low, _SHARED_CHARS)

    # ONE unambiguous letter is enough.
    #
    # `_MIN_SCRIPT_CHARS` of 2 was written for accents, which French and
    # Portuguese also carry. But s-underdot and u/i-underdot are not accents -
    # they are distinct letters that appear in no European language and in no
    # English text. Requiring two of them made "Se mo le gbin agbado bayii?"
    # (with s-underdot) fall through to English, and the in-domain gate then
    # refused it.
    if yor and not ibo:
        return "yor"
    if ibo and not yor:
        return "ibo"

    if yor + ibo + shared >= _MIN_SCRIPT_CHARS:
        if yor > ibo:
            return "yor"
        if ibo > yor:
            return "ibo"
        # Only shared vowels, or a dead tie. The language cannot be named, but
        # the text is demonstrably NOT English - e-underdot and o-underdot do
        # not occur in it. Saying "en" here sent the question to an
        # English-only in-domain gate that refused it, so the shared vowels are
        # reported as Yoruba: both fall back to English messages anyway, and
        # the only thing that changes is that the gate steps aside.
        if shared >= _MIN_SCRIPT_CHARS:
            return "yor"

        # Only shared vowels, or a dead tie: not enough to name the language;
        # let word evidence below decide.

    words = _WORD.findall(low)
    if not words:
        return "en"

    unique = set(words)
    strong = len(unique & _STRONG)
    weak = len(unique & _WEAK) + sum(p in low for p in _WEAK_PHRASES)

    # One unambiguous marker settles it. See _UNAMBIGUOUS for why these are
    # separated from the markers that are a keystroke from an English word.
    if unique & _UNAMBIGUOUS:
        return "pcm"
    if any(rx.search(low) for rx in _PHRASE_RX):
        return "pcm"

    if strong >= _MIN_STRONG:
        return "pcm"
    if strong >= _MIN_STRONG_WITH_WEAK and weak >= _MIN_WEAK_SUPPORT:
        return "pcm"

    if len(unique & _YORUBA_WORDS) >= _MIN_LANG_WORDS:
        return "yor"
    if len(unique & _IGBO_WORDS) >= _MIN_LANG_WORDS:
        return "ibo"
    if unique & _HAUSA_UNAMBIGUOUS:
        return "hau"
    if len(unique & _HAUSA_WORDS) >= _MIN_LANG_WORDS:
        return "hau"

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

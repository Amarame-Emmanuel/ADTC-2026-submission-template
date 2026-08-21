"""Map Yoruba, Igbo and Hausa farm vocabulary to English, for retrieval only.

WHY THIS EXISTS
---------------
Detection was added for all three languages, and the in-domain gate was taught to
step aside for them. Measured end to end, that changed almost nothing a farmer
would notice: of twelve questions in the three languages, ELEVEN retrieved zero
passages and refused at the floor. The two that worked contained the English
loanword "cassava".

The support was nominal. The corpus is English and so is the embedder, so a
Yoruba question is a bag of tokens the index has never seen.

WHY A LOOKUP TABLE AND NOT NLLB
-------------------------------
`agbe/translate/nllb.py` records the translation bridge that was built, measured
and rejected: ~740 MB resident, about 1.7 points of S_eff. It failed for PIDGIN
for a structural reason - Pidgin is English-lexified, so the model copied input
through unchanged - and that reasoning does NOT apply here. Yoruba, Igbo and
Hausa are unrelated languages and NLLB handles them properly.

So the argument against it is only cost, and the argument for a table is that
retrieval does not need translation. It needs the CONTENT WORDS. A farmer asking
"Ewurẹ́ mi kò jẹun" needs the index to see "goat" and "eat"; whether the result
reads as fluent English is irrelevant, because no human ever sees this string -
it goes to the embedder and to BM25.

`pidgin_norm.py` established the pattern and the measurement to beat: a lookup
table took Pidgin retrieval from 1/5 to 4/5 for no memory at all.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Grammar, tense, agreement, word order. None of it affects a bag-of-words match
or a sentence embedding enough to be worth the risk, and every rule added is a
rule that can fire on the wrong language.

Diacritics are optional in every entry, because farmers type without them - the
same assumption `detect.py` already makes for its word lists.

THE ANSWER IS STILL ENGLISH
---------------------------
This changes what is RETRIEVED, not what is written. A Yoruba speaker gets an
English answer from English sources, which is the honest state of the system:
messages fall back to validated English rather than anything machine-translated,
and no claim is made that the corpus speaks Yoruba.
"""

from __future__ import annotations

import re
import unicodedata

#: Crops, animals, and the words a symptom description turns on.
#:
#: Keyed WITHOUT diacritics; the lookup strips them from the input too, so
#: "ewure", "ewurẹ" and "ewúrẹ́" all hit the same entry.
_TERMS: dict[str, str] = {
    # -- Yoruba ----------------------------------------------------------
    "ewure": "goat", "obuko": "goat",
    "agbado": "maize", "oka yoruba": "maize",
    "ege": "cassava", "paki": "cassava", "gbaguda": "cassava",
    "isu": "yam", "adie": "chicken", "malu": "cattle", "agutan": "sheep",
    "ewe": "leaf", "ewebe": "vegetable", "eso": "fruit", "gbongbo": "root",
    "ofeefee": "yellow", "pupa": "red", "dudu": "black",
    "ile": "soil", "oko": "farm", "omi": "water", "ojo": "rain",
    "ajenirun": "pest", "arun": "disease", "kokoro": "insect",
    "gbin": "plant", "ikore": "harvest", "ajile": "fertiliser",
    "jeun": "eating", "aisan": "sick", "ku": "die", "ra": "rot",
    # -- Igbo ------------------------------------------------------------
    "ewu": "goat", "oka": "maize", "akpu": "cassava", "jigbo": "cassava",
    "ji": "yam", "okuko": "chicken", "ehi": "cattle", "aturu": "sheep",
    "akwukwo": "leaf", "mkpuru": "seed", "mgborogwu": "root",
    "odo odo": "yellow", "odo": "yellow", "ojii": "black",
    "ala": "soil", "ubi": "farm", "mmiri": "water", "mmiri ozuzo": "rain",
    "oria": "disease", "ahuhu": "pest", "ijiji": "fly",
    "iri nri": "eating", "eri nri": "eating", "eto": "grow",
    "ku": "plant", "owuwe": "harvest", "fatilaiza": "fertiliser",
    # -- Hausa -----------------------------------------------------------
    "akuya": "goat", "awaki": "goat", "bunsuru": "goat",
    "masara": "maize", "rogo": "cassava", "doya": "yam",
    "dawa": "sorghum", "gero": "millet", "shinkafa": "rice",
    "kaza": "chicken", "kaji": "chicken", "shanu": "cattle",
    "tumaki": "sheep", "ganye": "leaf", "iri": "seed", "saiwa": "root",
    "rawaya": "yellow", "ja": "red", "baki": "black",
    "kasa": "soil", "gona": "farm", "ruwa": "water", "ruwan sama": "rain",
    "cuta": "disease", "kwari": "insect", "kwaro": "pest",
    "shuka": "plant", "girbi": "harvest", "taki": "fertiliser",
    "abinci": "feed", "cin abinci": "eating", "kiwo": "grazing",
    "noma": "farming", "rube": "rot",
}

#: Longest first, so "cin abinci" wins over "abinci" and "odo odo" over "odo".
_PATTERNS = tuple(
    (re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), english)
    for term, english in sorted(_TERMS.items(), key=lambda kv: -len(kv[0]))
)

#: Languages this table covers. English and Pidgin are handled elsewhere and
#: must not be touched: the English path is the measured, benchmarked one.
COVERED = ("yor", "ibo", "hau")


def _strip_diacritics(text: str) -> str:
    """Fold é, ẹ, ọ, ṣ, ị, ụ to their base letters.

    Farmers type without diacritics, and the corpus contains neither form, so
    folding costs nothing and doubles the hit rate on every entry.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def add_english_terms(text: str) -> str:
    """Append the English equivalent of every farm term recognised in `text`.

    APPENDS rather than replaces. The original words are left in place because
    BM25 may still match a loanword the corpus carries - "cassava" appears in
    Igbo questions verbatim - and because a replacement that guesses wrong
    destroys evidence, while an addition that guesses wrong only adds noise.
    """
    folded = _strip_diacritics(text)
    found: list[str] = []
    for pattern, english in _PATTERNS:
        if pattern.search(folded) and english not in found:
            found.append(english)
    if not found:
        return text
    return f"{text} {' '.join(found)}"


def for_retrieval(text: str, language: str) -> str:
    """Normalise only for languages this table covers.

    Gated on the DETECTED language, not on whether a term happens to match.
    Several entries are short - "ku", "ji", "ja", "iri" - and would fire on
    English or Pidgin text by coincidence; gating keeps those paths untouched.
    """
    if language not in COVERED:
        return text
    return add_english_terms(text)

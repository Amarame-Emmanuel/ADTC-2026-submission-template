"""The in-domain gate must not block the languages this system serves.

WHY THIS FILE EXISTS
--------------------
The gate refuses anything that names no farming word. Its vocabulary is English
- cassava, goat, harvest, spray - so applied to Yoruba or Igbo it refused
questions the detector had just identified correctly:

    "Kedu mgbe m ga-aku oka?"    (when should I plant maize)  -> out of domain
    "Ewu m adighi eri nri"       (my goat is not eating)      -> out of domain

while an Igbo question that happened to contain the loanword "cassava" passed.
That is the gate blocking the languages the system exists to serve, which is
worse than anything it was built to prevent.

Yoruba had a second, separate bug: `_MIN_SCRIPT_CHARS` of 2 was written for
ACCENTS, which French and Portuguese also carry. But s-underdot, u-underdot and
i-underdot are not accents - they are letters that occur in no European language
and in no English text. Requiring two of them sent Yoruba to the English gate.

WHAT MUST NOT REGRESS
---------------------
English out-of-domain questions must still refuse. The gate is the fix for
F-30 - "who is the president of Nigeria" returning a false, cited claim - and
loosening it for other languages must not loosen it for English.
"""

from __future__ import annotations

import pytest

from agbe.rag import scope
from agbe.translate.detect import detect

YORUBA = [
    "Ṣé mo lè gbin àgbàdo báyìí?",
    "Ilẹ̀ mi kò dára mọ́",
    "Ewé gbogbo ọ̀gẹ̀dẹ̀ mi ti bẹ̀rẹ̀ sí í rọ",
]

IGBO = [
    "Kedu mgbe m ga-akụ ọka?",
    "Ewu m adịghị eri nri",
    "Akwụkwọ cassava m na-acha odo odo",
]

#: Hausa, added as the third major Nigerian language. DETECTION only - the same
#: status Yoruba and Igbo have - so messages still fall back to validated
#: English rather than anything machine-translated.
#:
#: It fits where Twi and French do not: it is Nigerian, so it does not collide
#: with the service-area rule that declines Ghanaian and Senegalese questions,
#: and `test_advisor.py` has tested get("hau") falling back to English since
#: before detection existed.
HAUSA = [
    "Akuya ta ba ta cin abinci",        # my goat is not eating
    "Ina son shuka masara yanzu",       # I want to plant maize now
    "Rogo na ya yi rawaya",             # my cassava is yellowing
    "Shanu na ba sa kiwo",              # my cattle are not grazing
    "Yaya zan yi da kaji na?",          # what do I do about my chickens
    "ƙasa ta yi tauri",                 # the soil has gone hard
]

ENGLISH = [
    "My cassava leaves are yellow",
    "How do I control weeds in my maize?",
    "Should I sell my maize now or store it?",
    "I do not know what to do",
    "My goats are not eating well",
]

#: Must still refuse. The gate exists because "who is the president of Nigeria?"
#: returned "Muhammadu Buhari [1]" - a false, cited claim.
ENGLISH_OUT_OF_DOMAIN = [
    "Who is the president of Nigeria?",
    "My catfish are dying in the pond",
    "How do I raise rabbits for meat?",
    "How do I grow mushrooms in a dark room?",
]


@pytest.mark.parametrize("question", YORUBA, ids=lambda q: q[:28])
def test_yoruba_is_detected(question: str) -> None:
    assert detect(question) == "yor", question


@pytest.mark.parametrize("question", IGBO, ids=lambda q: q[:28])
def test_igbo_is_detected(question: str) -> None:
    assert detect(question) == "ibo", question


@pytest.mark.parametrize("question", ENGLISH, ids=lambda q: q[:28])
def test_english_is_not_mistaken_for_another_language(question: str) -> None:
    assert detect(question) == "en", question


@pytest.mark.parametrize("question", YORUBA + IGBO + HAUSA, ids=lambda q: q[:28])
def test_the_gate_steps_aside_for_other_languages(question: str) -> None:
    """The gate cannot judge a language it has no vocabulary for.

    The retrieval floor remains the check on relevance, as it was for every
    question before the gate existed.
    """
    language = detect(question)
    verdict = scope.check(question, language)
    assert verdict.in_scope, f"{language}: {question!r} -> {verdict.reason!r}"


@pytest.mark.parametrize("question", ENGLISH_OUT_OF_DOMAIN, ids=lambda q: q[:34])
def test_english_out_of_domain_still_refuses(question: str) -> None:
    assert not scope.check(question, "en").in_scope, question


def test_pidgin_still_goes_through_the_gate() -> None:
    """Pidgin is normalised into English before scope runs, so the vocabulary
    applies and the gate should keep working on it."""
    from agbe.translate.pidgin_norm import for_retrieval as norm

    verdict = scope.check(norm("My goat pikin dey shit water since morning"), "pcm")
    assert verdict.in_scope


@pytest.mark.parametrize("question", HAUSA, ids=lambda q: q[:28])
def test_hausa_is_detected(question: str) -> None:
    assert detect(question) == "hau", question


def test_hausa_falls_back_to_validated_english_messages() -> None:
    """Detection is not a claim of support.

    A detected language with no speaker-reviewed message set gets English, so
    nothing a farmer reads is machine-translated. What detection buys is that
    the English-only in-domain gate steps aside.
    """
    from agbe.translate.messages import MESSAGES, get

    assert get("hau") is MESSAGES["en"]

"""Yoruba/Igbo detection, and the contract that makes it safe to ship dormant.

Detection returning a language is NOT a claim of support. The pinned contract
is the second half: messages.get() falls back to English for any language
without human-validated strings, so detection can be accurate today while the
farmer-visible behaviour changes only on the day a validator approves the
strings and the language is added to messages.py.
"""

from __future__ import annotations

import pytest

from agbe.translate.detect import confidence, detect
from agbe.translate.messages import MESSAGES, get


@pytest.mark.parametrize("text,expected", [
    # Script evidence: dotted characters that English and Pidgin never carry.
    ("Ṣe o le so fun mi ohun ti o n ṣe agbado mi?", "yor"),
    ("Kẹdu, gịnị na-eme akwụkwọ oka m?", "ibo"),
    # Diacritics omitted (phone keyboards): word evidence, two words required.
    ("bawo ni mo se le toju agbado mi", "yor"),
    ("kedu ihe na-eme oka m", "ibo"),
    # The existing languages must be untouched by the new branches.
    ("My cassava leaves are yellow and twisted", "en"),
    ("My goat no dey chop, wetin dey worry am?", "pcm"),
    ("When should I plant maize?", "en"),
    # One ambiguous word is not evidence - bias to English holds.
    ("the oka river is near my farm", "en"),
])
def test_detect(text, expected):
    assert detect(text) == expected


def test_decomposed_diacritics_are_normalised():
    """A keyboard emitting base letters + combining dots must detect the same
    as one emitting precomposed characters."""
    precomposed = "Ṣe o le ran mi lọwọ pẹlu iṣu mi?"
    decomposed = "Ṣe o le ran mi lọwọ pẹlu iṣu mi?"
    assert detect(precomposed) == detect(decomposed) == "yor"


def test_confidence_agrees_with_detect():
    """confidence() once knew fewer languages than detect() and returned
    ("en", ...) for text detect() called Yoruba - an inconsistent pair."""
    for text in ("Ṣe o le so fun mi nipa agbado?", "kedu ihe na-eme oka m",
                 "My goat no dey chop, wetin dey worry am?", "hello there"):
        lang, score = confidence(text)
        assert lang == detect(text)
        assert 0.0 <= score <= 1.0


def test_detected_but_unvalidated_falls_back_to_english():
    """THE contract. Remove this test only when removing the fallback itself."""
    for lang in ("yor", "ibo"):
        assert lang not in MESSAGES, (
            f"{lang} now has validated strings - update this test to assert "
            "the validated set is served instead"
        )
        assert get(lang) is get("en")

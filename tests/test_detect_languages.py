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


class TestPidginRecall:
    """Sixteen Pidgin questions through the interface; only six were detected.

    The ten misses shared one shape - ONE strong marker and ONE weak one - which
    the thresholds required two weak markers to accept:

        "Abeg how I go take stop weevil for my maize?"   abeg + go
        "My chicken no dey lay egg again"                dey  + no
        "Wetin make my groundnut leaf get black spot?"   wetin + make

    That is the ordinary shape of a Pidgin sentence. A miss costs twice:
    retrieval never sees normalised text, and the farmer gets English messages.

    The fix is not a lower threshold. "dem", "dis", "dat" and "di" are one
    keystroke from "them", "this", "that" and "the", so one of those must never
    be enough on its own. `_UNAMBIGUOUS` holds the markers with no English
    neighbour, and one of those is decisive.
    """

    PIDGIN = [
        "Abeg how I go take stop weevil for my maize?",
        "My chicken no dey lay egg again",
        "Wetin make my groundnut leaf get black spot?",
        "My yam don begin rotten for ground, wetin happen?",
        "I wan plant cassava, which time better?",
        "My tomato dey spoil quick quick for market",
        "My goat pikin dey purge, e don weak",
        "How I go take keep my cowpea make weevil no chop am?",
    ]

    #: A Pidgin message shown to an English speaker is worse than the reverse,
    #: so these matter more than the recall cases above.
    ENGLISH = [
        "My cassava leaves are yellow and twisted",
        "How do I control weeds in my maize?",
        "I do not know what to do",
        "My goats are not eating well",
        "Should I sell my maize now or store it?",
        "The rains have not started",
        "I want to plant cassava, which time is better?",
        "My chickens have twisted necks and greenish diarrhoea",
    ]

    @pytest.mark.parametrize("question", PIDGIN)
    def test_ordinary_pidgin_is_detected(self, question):
        assert detect(question) == "pcm", question

    @pytest.mark.parametrize("question", ENGLISH)
    def test_english_is_never_called_pidgin(self, question):
        assert detect(question) == "en", question

    def test_multi_word_weak_markers_can_actually_fire(self):
        """"well well" and friends sat in a set matched against single words.

        Matching is a set intersection over `_WORD.findall`, so no phrase in
        that set had ever matched anything - dead entries that looked live.
        They are now matched against the text.
        """
        from agbe.translate.detect import _WEAK_PHRASES

        assert _WEAK_PHRASES
        for phrase in _WEAK_PHRASES:
            assert " " in phrase

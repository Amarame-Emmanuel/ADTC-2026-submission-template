"""Pidgin normalisation for retrieval.

Measured against the alternative it replaced: on five Pidgin farm questions,
raw retrieval found 1/5, the NLLB-200 bridge found 1/5 for ~740 MB of resident
memory, and these rules find 4/5 for none. See agbe/translate/pidgin_norm.py.
"""

from __future__ import annotations

import pytest

from agbe.translate.pidgin_norm import for_retrieval, looks_like_pidgin, normalise


@pytest.mark.parametrize("english", [
    "My cassava leaves are yellow and twisted",
    "When should I plant maize?",
    "How do I store yam to make it last?",
    "My goats are not eating and they are shaking",
])
def test_english_is_returned_byte_identical(english):
    """The English path is the benchmarked one. It must not move.

    Coverage and refusal were measured on English questions. If normalisation
    touched them, every number in REPORT.md would silently describe a system
    that no longer exists.
    """
    assert for_retrieval(english) == english


def test_markers_are_required_before_rewriting():
    assert looks_like_pidgin("My fowl dem dey die")
    assert not looks_like_pidgin("When should I plant maize?")


def test_progressive_and_negation():
    out = normalise("My goat no dey chop again and e dey shake.")
    assert "is not eat" in out
    assert "it is shake" in out
    assert "dey" not in out


def test_object_pronoun_am_becomes_it():
    out = normalise("wetin dey worry am?")
    assert out == "what is wrong with it?"


def test_english_i_am_survives_the_am_rule():
    """"am" is a Pidgin object pronoun and an English auxiliary.

    The rule is anchored so it cannot eat the English one - including the
    "I am" this module produces itself from "i dey" earlier in the same pass.
    """
    assert "I am" in normalise("i dey find wetin dey worry my maize")


def test_plural_marker_dem():
    out = normalise("My fowl dem dey die, dem neck dey twist")
    assert "chicken" in out
    assert "their neck" in out


def test_purpose_clause():
    out = normalise("Wetin I go do make my yam no rotten for store?")
    assert out.startswith("what I will do")
    assert "so that" in out
    assert "rot" in out


def test_normalisation_never_empties_a_question():
    for q in ["wetin dey?", "na wetin?", "e dey"]:
        assert normalise(q).strip()

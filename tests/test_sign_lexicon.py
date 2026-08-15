"""Every sign→name mapping must fire, and none may fire on ordinary questions.

WHY THIS FILE EXISTS
--------------------
The lexicon in `agbe/rag/query.py` maps a distinctive visible sign to the name
the literature files it under - "white cottony insects" to mealybug. It is
hand-written, and hand-written lookups fail in two directions that no other test
in this suite would catch:

  DEAD ENTRY     - a pattern that can never match. The `cutworm` entry shipped
                   in exactly this state: it required the literal sequence "cut
                   the seedlings at base", while a farmer writes "my seedlings
                   are cut at the base" or "at ground level". It looked
                   functional and was unreachable. Only 4 of the 9 entries are
                   exercised by the evaluation set at all, so five could rot
                   unnoticed.

  FALSE POSITIVE - a pattern that fires on a question it has no business
                   touching, steering retrieval confidently wrong. That is the
                   §6.8 failure mode, and the sharpest example is `cut`:
                   "stem cuttings", "healthy cuttings" and "select cuttings" are
                   core cassava planting vocabulary. A cutworm pattern matching
                   them would fire on every clean-planting-material question -
                   the exact content §6.8 spent three re-indexes trying to
                   surface.

So each entry needs at least one phrasing that must match, and the lexicon as a
whole needs phrasings that must not.
"""

from __future__ import annotations

import pytest

from agbe.rag.query import _SIGN_TO_NAME, retrieval_query, sign_names

#: One or more natural phrasings per entry. A farmer's words, not the corpus's.
MUST_FIRE = [
    ("mealybug", "White cottony insects on the growing tip of my cassava"),
    ("mealybug", "There is a white waxy coating on the shoots"),
    ("leafminer", "White winding lines inside my tomato leaves"),
    ("leafminer", "There are tunnels in the leaves of my vegetables"),
    ("spider mite", "There is fine webbing under the leaves of my tomato"),
    ("spider mite", "I see cobwebs on my cassava leaves in the dry season"),
    ("sooty mould honeydew", "The leaves have a sooty black coating"),
    ("sooty mould honeydew", "Black mould on the leaves and sticky honeydew"),
    ("borer", "Worms are boring into my okra pods"),
    ("borer", "I see sawdust around holes in the stem"),
    ("cutworm", "My seedlings are cut at the base every night"),
    ("cutworm", "Something cuts the seedlings at ground level"),
    ("cutworm", "Young plants are severed at soil level in the morning"),
    ("newcastle disease", "My chickens have twisted necks and green droppings"),
    ("newcastle disease", "The birds pass greenish watery droppings"),
    ("worms helminth", "My goat has a swollen jaw and a pot belly"),
    ("worms helminth", "The kid has bottle jaw and is very thin"),
]

#: Ordinary in-scope questions. The first four are the dangerous ones: cassava
#: planting material is discussed in terms of CUTTINGS, and a loose cutworm
#: pattern would fire on all of them.
MUST_NOT_FIRE = [
    "Where should I get cassava stems to plant so my crop does not get sick?",
    "Should I use healthy cuttings from disease-free plants?",
    "How do I select cuttings for planting?",
    "I cut my cassava stems into pieces before planting",
    "When should I plant maize in Oyo State?",
    "Should I store my grain or sell it now?",
    "How do I grade tomatoes for market?",
    "My cassava leaves are yellow and twisted",
    "What is the best spacing for cowpea?",
    "How do I dry my maize after harvest?",
    "My chickens are not laying eggs",
    "The rains started late this year, what should I plant?",
    "How do I make compost from farm waste?",
    "My goat is not eating well",
]


@pytest.mark.parametrize("expected,question", MUST_FIRE)
def test_sign_is_recognised(expected: str, question: str) -> None:
    got = sign_names(retrieval_query(question))
    assert expected in got, f"{question!r} should map to {expected!r}, got {got!r}"


@pytest.mark.parametrize("question", MUST_NOT_FIRE)
def test_ordinary_questions_are_untouched(question: str) -> None:
    """A sign that fires here steers retrieval for a question that was fine."""
    got = sign_names(retrieval_query(question))
    assert got == [], f"{question!r} should match no sign, got {got!r}"


def test_every_entry_has_a_probe() -> None:
    """No entry may exist without a phrasing proving it is reachable.

    Guards against the dead-entry failure directly: adding a mapping without a
    MUST_FIRE case fails here rather than shipping unreachable.
    """
    declared = {name for _pattern, name in _SIGN_TO_NAME}
    probed = {name for name, _q in MUST_FIRE}
    assert declared == probed, (
        f"entries without a probe: {sorted(declared - probed)}; "
        f"probes for absent entries: {sorted(probed - declared)}"
    )

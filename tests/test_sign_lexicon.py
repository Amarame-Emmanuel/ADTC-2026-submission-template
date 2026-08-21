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
    # Cassava mosaic. These four were added when the entry was; the first was
    # previously a MUST_NOT_FIRE control, back when no entry covered it and it
    # served to prove that ordinary symptom language fires nothing. It is the
    # single most-discussed question in this project, it was answered "cassava
    # brown streak" in 4 runs of 12, and it now has a mapping of its own.
    ("cassava mosaic", "My cassava leaves are yellow and twisted"),
    ("cassava mosaic", "My cassava leaves are yellow and twisted and the plant is small"),
    ("cassava mosaic", "The cassava leaf is mottled and distorted"),
    ("cassava mosaic", "Leaves turning yellow and crinkled"),
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
    # Bloat: answered with hoof-trimming advice, because "wet" appears in both
    # the question and a footrot passage. It kills within hours.
    ("bloat", "My goat left side is swollen and tight after grazing wet grass"),
    ("bloat", "My cow is bloated after eating fresh lucerne"),
    ("bloat", "The goat stomach is distended on one side"),
    # Lameness: a single lame leg was diagnosed as foot-and-mouth disease, a
    # notifiable disease the farmer would then go and report.
    ("foot rot", "My goat is limping and will not put weight on one leg"),
    ("foot rot", "One of my sheep is lame and stays behind the flock"),
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
    # Near misses for the cassava mosaic entry. The pattern requires
    # yellowing BEFORE distortion, which is how mosaic is described; the okra
    # question says "curling and turning yellow" and is about a different
    # crop, which must not be dragged onto cassava material.
    "My okra leaves are curling and turning yellow with tiny insects underneath",
    "My cassava leaves are yellow",
    "The leaves are twisted",
    "My maize has purple leaves",
    "My cassava leaves have a shoestring appearance",
    "Tomato leaves are curling downward",
    "What is the best spacing for cowpea?",
    "How do I dry my maize after harvest?",
    "My chickens are not laying eggs",
    "The rains started late this year, what should I plant?",
    "How do I make compost from farm waste?",
    "My goat is not eating well",
    # Bloat and lameness patterns must not reach ordinary swelling or movement
    # language. "Bottle jaw" is swelling and belongs to worms, not bloat.
    "How much space should each goat have to move around?",
    "My cassava stems are swollen at the node",
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


def test_bloat_does_not_swallow_bottle_jaw() -> None:
    """Both are swelling, and they are different diseases with different fixes.

    Bottle jaw is oedema under the jaw from worms; bloat is gas distension of
    the rumen on the left flank. A bloat pattern keyed on "swollen" alone would
    take both, and would send a farmer with a wormy goat looking for a stomach
    tube. The bloat entry requires the SIDE, which is what separates them.
    """
    for question in ("My goat has a swollen jaw and a pot belly",
                     "The kid has bottle jaw and is very thin"):
        got = sign_names(retrieval_query(question))
        assert got == ["worms helminth"], f"{question!r} -> {got!r}"


def test_lameness_does_not_fire_on_ordinary_movement_language() -> None:
    """"Lame" and "limp" are specific; walking and space are not.

    The scouring fix walked a bloated goat around on purpose; nothing here
    should make ordinary husbandry advice look like a lameness question.
    """
    for question in ("How much space should each goat have to move around?",
                     "Walk the animal around after feeding",
                     "My goat is not eating well"):
        assert "foot rot" not in sign_names(retrieval_query(question))

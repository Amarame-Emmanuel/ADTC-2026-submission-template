"""A closed question must not invent a regulatory fact.

WHY THIS FILE EXISTS
--------------------
Asked "is there any pesticide registered for cassava mealybug in Nigeria?", the
system answered:

    "No, there is no pesticide registered for the cassava mealybug in Nigeria."

Retrieval had SUCCEEDED - six on-topic passages, every one over the score floor
- and not one of them mentioned registration. Section 1 of the report states
that NAFDAC registration data is not openly published, so the system asserted a
fact it had committed in writing to not knowing.

WHY THIS IS A CATEGORY TEST AND NOT A CONTAINMENT TEST
------------------------------------------------------
The first version of this guard asked whether the word the question turned on
appeared anywhere in the retrieved passages. An adversarial probe run showed
that to be wrong in BOTH directions, which is why the design changed:

TOO WEAK   - "Is glyphosate approved for use on maize in Nigeria?" was ANSWERED
             "Yes... as indicated by the Maize-legume cropping guide", because
             the stem `approv` did appear in the passages in an unrelated sense.
             An invented regulatory claim, attributed to a real document.

TOO STRONG - "Is the milk safe to drink after I deworm my goat?" was REFUSED
             with five passages retrieved, because they discuss withdrawal
             periods without using the word "safe" - and a withdrawal question
             is the one thing the safety layer exists to serve. "Can Newcastle
             disease be cured once the birds are already sick?" was refused the
             same way, with the correct answer sitting in its six passages.

Presence of a word is not a position on a question, and absence of a word is not
absence of an answer. Regulatory status is different in kind: no retrieval
result can ever make it answerable, which is knowable in advance and is what
makes it a rule rather than a guess.

WHAT THIS DELIBERATELY GIVES UP
-------------------------------
"Is there a cure for cassava mosaic disease?" -> "Yes" is no longer caught. It
is a real defect - a virus is controlled, never cured. It is out because the
only test that caught it also produced both over-refusals above, and denying a
farmer "there is no cure, vaccinate instead" costs more than a wrong framing
attached to correct control advice. Tracked as open in FINDINGS, not as fixed.
"""

from __future__ import annotations

import pytest

from agbe.advisor import sources_cannot_settle

#: Passages that discuss the pest but take no position on registration - the
#: real shape of what retrieval returned. Now ignored by the guard, and passed
#: anyway to pin that it is ignored.
MEALYBUG_SOURCES = [
    "The cassava mealybug Phenacoccus manihoti sucks sap from the growing "
    "point, causing bunchy top and leaf distortion.",
    "Biological control using the parasitoid Anagyrus lopezi has been the main "
    "control measure across the cassava belt.",
]

WITHDRAWAL_SOURCES = [
    "After treatment, observe the withdrawal period before the milk is used. "
    "Ask a veterinary officer how many days to wait.",
]


#: Must refuse: a closed question asking for regulatory status in a jurisdiction.
REFUSE = [
    "Is there any pesticide registered for cassava mealybug in Nigeria?",
    "Is glyphosate approved for use on maize in Nigeria?",
    "Is paraquat banned for tomato farmers in Nigeria?",
    "Is there a registered vaccine for foot-and-mouth disease in Nigeria?",
    "Are these chemicals licensed by NAFDAC?",
]


@pytest.mark.parametrize("question", REFUSE)
def test_regulatory_questions_are_refused(question: str) -> None:
    assert sources_cannot_settle(question, MEALYBUG_SOURCES) == "regulatory status"


def test_the_sources_cannot_talk_the_guard_out_of_it() -> None:
    """The failure that made this a category test.

    A passage using "approved" in an ordinary sense is not a statement about
    Nigerian regulatory status, and must not stand the guard down.
    """
    sources = ["Deep ploughing is an approved practice on heavy soils."]
    assert (
        sources_cannot_settle(
            "Is glyphosate approved for use on maize in Nigeria?", sources
        )
        == "regulatory status"
    )


#: Must survive. Over-refusal is the more expensive failure, and the two
#: questions this guard actually silenced are the first two entries.
KEEP = [
    # Refused with five passages retrieved. The one question type the safety
    # layer exists for.
    ("Is the milk safe to drink after I deworm my goat?", WITHDRAWAL_SOURCES),
    # Refused with six passages retrieved, when the answer - no cure, prevent
    # by vaccination - was in them.
    ("Can Newcastle disease be cured once the birds are already sick?",
     MEALYBUG_SOURCES),
    # Closed questions carrying words the old guard keyed on.
    ("Is there a vaccine for Newcastle disease?", MEALYBUG_SOURCES),
    ("Are there cassava varieties that resist mosaic disease?", MEALYBUG_SOURCES),
    ("Are improved yam varieties available to smallholder farmers?", MEALYBUG_SOURCES),
    ("Can I treat Newcastle disease with antibiotics?", MEALYBUG_SOURCES),
    # Open questions are never the guard's business.
    ("What chemical can I use against cassava mosaic virus?", MEALYBUG_SOURCES),
    ("How do I control cassava mealybug?", MEALYBUG_SOURCES),
]


@pytest.mark.parametrize("question,sources", KEEP)
def test_answerable_questions_are_left_alone(
    question: str, sources: list[str]
) -> None:
    assert sources_cannot_settle(question, sources) is None


def test_a_regulatory_word_without_a_jurisdiction_does_not_fire() -> None:
    """"Approved" and "banned" are ordinary words outside a legal context.

    Without this, "are these varieties approved by the farmers group?" and
    similar agronomy phrasing would be refused as regulatory questions.
    """
    assert (
        sources_cannot_settle("Is this an approved way to store yam?", MEALYBUG_SOURCES)
        is None
    )


def test_open_regulatory_question_is_left_alone() -> None:
    """The closed opener is what makes a question demand a yes or a no.

    "What is registered for use in Nigeria" wants a list the corpus does not
    have either, but it fails for lack of evidence at the floor rather than by
    inventing a polarity, and the refusal path already covers that.
    """
    assert (
        sources_cannot_settle(
            "How do I find out what is approved for use in Nigeria?",
            MEALYBUG_SOURCES,
        )
        is None
    )

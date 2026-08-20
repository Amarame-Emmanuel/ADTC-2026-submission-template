"""The farmer's own symptom must never come back as an instruction.

WHY THIS FILE EXISTS
--------------------
Asked "my seedlings are cut at the base every night", the system answered:

    "For cutworms, cut off seedlings at the base every night."

A farmer following that would destroy their crop. It is not a bad fact from a
source and not a fabrication - no source says it. The model re-emitted the
QUESTION as an imperative, and nothing in the system was looking: `is_usable`
filters passages, `check_answer` looks for hazardous actives and unsupported
dosages, `strip_false_disclaimer` matches one opening. None compares the answer
to what was asked.

THE TWO WAYS THIS GUARD CAN FAIL
--------------------------------
INERT      - the threshold is set so tight that the defect passes. The first
             version measured only how much of the SENTENCE came from the
             question; the cutworm sentence scored 0.71 against a 0.8 bar and
             sailed through, because "cutworms" and "off" diluted it.

OVER-EAGER - it deletes real advice. This is the worse failure: a silently
             truncated answer looks fine and is missing the part that helps.
             Two shapes are dangerous.

             A diagnosis restates the symptom by design - "your cassava leaves
             are yellow and twisted because of mosaic disease" is nearly all the
             farmer's words and is exactly right. It survives because it is not
             an instruction.

             And a short question can be fully covered by good advice - "use
             nitrogen fertiliser on maize that is yellow" answers "my maize is
             yellow" completely. It survives because only 2 of its 5 content
             words are the farmer's.

So both directions of overlap are required, and both are pinned below.
"""

from __future__ import annotations

import pytest

from agbe.advisor import strip_symptom_echo

Q_SEEDLINGS = "My seedlings are cut at the base every night"
Q_CASSAVA = "My cassava leaves are yellow and twisted and the plants are small"
Q_CHICKENS = "My chickens have twisted necks and greenish diarrhoea"

#: The defect: an imperative that is the question handed back.
ECHOES = [
    (Q_SEEDLINGS,
     "To control cutworms, use collars around the stems. "
     "For cutworms, cut off seedlings at the base every night. "
     "Clear weeds before planting.",
     "For cutworms, cut off seedlings at the base every night."),
    (Q_CASSAVA,
     "Use clean planting material. "
     "Cut off leaves that are yellow and twisted and plants that are small. "
     "Remove infected plants.",
     "Cut off leaves that are yellow and twisted and plants that are small."),
]

#: Must survive. Diagnoses, descriptions, and advice that happens to cover a
#: short question.
KEEP = [
    (Q_CASSAVA,
     "Your cassava leaves are yellow and twisted because of cassava mosaic "
     "disease. The plants are small because the virus stunts growth."),
    (Q_CASSAVA,
     "The leaves become small, distorted and twisted along the edges as the "
     "disease progresses."),
    (Q_SEEDLINGS,
     "Cutworms cut seedlings at the base at night. Collect them by hand after "
     "dark and destroy them."),
    (Q_CHICKENS,
     "Newcastle disease causes twisted necks and greenish diarrhoea. Separate "
     "affected birds immediately."),
    (Q_CHICKENS, "Remove sick birds from the flock and disinfect the housing."),
    ("My maize is yellow", "Use nitrogen fertiliser on maize that is yellow."),
    ("My goat is not eating",
     "Check the goat that is not eating for worms and fever."),
    ("My tomato leaves have spots",
     "Remove tomato leaves that have spots and destroy them away from the field."),
]


@pytest.mark.parametrize("question,answer,expected", ECHOES)
def test_echoed_instruction_is_removed(
    question: str, answer: str, expected: str
) -> None:
    cleaned, removed = strip_symptom_echo(answer, question)
    assert removed == [expected], f"expected to remove {expected!r}, removed {removed!r}"
    assert expected not in cleaned


@pytest.mark.parametrize("question,answer", KEEP)
def test_real_advice_survives(question: str, answer: str) -> None:
    """Over-eagerness is the more dangerous failure: it truncates silently."""
    cleaned, removed = strip_symptom_echo(answer, question)
    assert removed == [], f"wrongly removed {removed!r} from {answer!r}"
    assert cleaned == answer


def test_never_returns_empty() -> None:
    """If removal would take everything, keep the original.

    A truncated answer is worse than an odd one, and the rate reporting in
    `advise()` is what should surface the problem instead.
    """
    answer = "Cut off seedlings at the base every night."
    cleaned, removed = strip_symptom_echo(answer, Q_SEEDLINGS)
    assert cleaned == answer
    assert removed == [answer]


def test_no_question_is_a_no_op() -> None:
    answer = "Remove infected plants and use clean planting material."
    assert strip_symptom_echo(answer, "") == (answer, [])

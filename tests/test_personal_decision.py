"""A decision about the farmer, not the farming, is declined.

WHY THIS FILE EXISTS
--------------------
A sweep of subjective questions found the system answering "Should I become a
farmer?" and "Should I expand my farm?" - decisions that turn on the asker's
money, land, family and appetite for risk. None of that is in any document, and
a corpus of extension material has nothing to say about it.

WHY IT IS DELIBERATELY NARROW
-----------------------------
Most "should I" questions in farming are answerable and MUST stay answerable.
"Should I sell now or store it", "should I intercrop maize with cowpea",
"should I use organic methods", "should I separate the sick goat" are all
decisions extension material is written to inform, and a rule that swept them
up would gut the system.

What is refused is the question about the PERSON rather than the practice:
whether to farm at all, whether to borrow, whether to give up.

WHERE IT POINTS, AND WHY NOT ONLINE
-----------------------------------
An extension officer or cooperative, as everywhere else in this codebase. Not
the internet. This system runs on a laptop with no network - `make run-offline`
proves it with `--network none` - and a farmer who could search online would not
need it. A referral that assumes connectivity contradicts the premise the whole
design serves.
"""

from __future__ import annotations

import pytest

from agbe.rag import scope

PERSONAL = [
    "Should I become a farmer?",
    "Should I quit farming?",
    "Should I give up farming?",
    "Is farming worth it?",
    "Should I expand my farm?",
    "Is there any point in farming?",
    "What should I do with my life?",
]

#: Farming judgement calls. Extension material exists to inform every one.
PRACTICAL = [
    "Should I sell my maize now or store it?",
    "Should I intercrop maize with cowpea?",
    "Should I use organic methods?",
    "Should I plant early or late?",
    "Should I save seed or buy new?",
    "Is cassava better than maize for my soil?",
    "Should I separate the sick goat?",
    "Should I burn the crop residue?",
    "Is it worth buying a moisture meter?",
    "Should I buy a thresher?",
    "Is it better to sell cassava as chips or fresh roots?",
]


@pytest.mark.parametrize("question", PERSONAL, ids=lambda q: q[:34])
def test_personal_decisions_are_declined(question: str) -> None:
    verdict = scope.check(question)
    assert not verdict.in_scope, question
    assert verdict.reason == "personal decision"


@pytest.mark.parametrize("question", PRACTICAL, ids=lambda q: q[:34])
def test_farming_judgement_is_answered(question: str) -> None:
    assert scope.check(question).in_scope, question


def test_the_referral_is_local_not_online() -> None:
    """Every refusal in this system points somewhere a farmer can reach today.

    Radio, product label, local market, extension officer. An offline system
    telling a smallholder to search the internet is advice they cannot take.
    """
    message = scope.check("Should I become a farmer?").message
    assert "extension officer" in message
    for word in ("online", "internet", "search", "google", "website"):
        assert word not in message.lower(), f"{word!r} in a refusal message"

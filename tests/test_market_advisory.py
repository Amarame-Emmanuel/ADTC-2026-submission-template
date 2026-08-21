"""Market advisory must survive the price refusal.

WHY THIS FILE EXISTS
--------------------
Market is one of the four advisory areas this system declares. It is also the
area a blanket price refusal is most likely to destroy, and it nearly did.

`_LIVE_PRICE` was broadened until it caught two questions that had been checked
by hand as must-answer weeks earlier:

    "How do I decide what price to ask for my maize?"   -> refused, "live price"
    "Where can I get a better price for my cassava?"    -> refused, "live price"

Neither asks for a figure. The first asks for a METHOD, the second for a
CHANNEL, and both are answerable from extension material by a system that knows
no prices at all. They were caught by two DIFFERENT branches of the rule
("what price" and "price for"), which is why the fix keys on the opener rather
than patching branches one at a time.

THE ASYMMETRY THAT LET IT THROUGH
---------------------------------
`test_refusal_recall.py` guards the must-REFUSE side of price and is thorough.
Nothing guarded the must-ANSWER side, so an over-reaching pattern passed every
test in the project while silently removing the useful half of a whole advisory
area. A refusal rule needs both lists or it will drift in one direction.

WHAT SEPARATES THE TWO
----------------------
The refusal exists because an offline system cannot know a price. So the line
is whether answering requires a NUMBER this system cannot have:

    "what is a fair price for cassava"   needs a figure    -> refuse
    "how do I decide what price to ask"  needs a method    -> answer
    "why do prices fall at harvest"      needs a mechanism -> answer

Judgement, grading, timing, channels and mechanisms all stay.
"""

from __future__ import annotations

import pytest

from agbe.rag import scope

#: Market questions that must be ANSWERED. None can be settled with a figure,
#: and none needs one.
#:
#: The first two are the regression. The rest are the twenty-question market
#: sweep that found it, kept so the area cannot be hollowed out quietly.
MUST_ANSWER = [
    # -- the regression --------------------------------------------------
    "How do I decide what price to ask for my maize?",
    "Where can I get a better price for my cassava?",
    # -- mechanism: why prices move ---------------------------------------
    "Why do prices fall at harvest?",
    "What makes tomato prices change during the year?",
    "How are prices set in the market?",
    "Why do prices change from season to season?",
    # -- timing and storage ------------------------------------------------
    "Should I sell my maize now or store it?",
    "When is the best time to sell my harvest?",
    "My store is full and the price is low",
    "If I sell now I get less, if I wait I may lose to weevils. What do I do?",
    # -- grading, quality and value addition -------------------------------
    "How do I grade my tomatoes before selling?",
    "What quality do buyers look for in maize?",
    "Why do buyers reject my produce?",
    "Is it better to sell cassava as chips or fresh roots?",
    "How do I add value to my cassava?",
    "How should I package garri for sale?",
    # -- channel and bargaining position -----------------------------------
    "How do I find a better buyer for my grain?",
    "Should I join a cooperative to sell together?",
    "How can I get more for my crop?",
    "What is contract farming?",
    # -- being cheated ------------------------------------------------------
    "How do I know if a buyer is cheating me on weight?",
    "How do I make sure the scale is honest?",
    # -- losses and records --------------------------------------------------
    "How do I reduce losses when transporting tomatoes?",
    "What farm records help me sell better?",
    "Should I keep my grain for seed or for eating?",
]


@pytest.mark.parametrize("question", MUST_ANSWER, ids=lambda q: q[:44])
def test_market_advice_is_not_refused(question: str) -> None:
    verdict = scope.check(question, "en")
    assert verdict.in_scope, f"refused as {verdict.reason!r}: {question!r}"


#: The other half. Each needs a number this system cannot know.
#:
#: Kept beside the list above deliberately: the two are only meaningful
#: together, and a change that satisfies one by breaking the other is the exact
#: failure this file exists to catch.
MUST_REFUSE = [
    "What is a fair price for my cassava?",
    "How much is my yam worth?",
    "What does a bag of maize cost?",
    "What is maize selling for today?",
    "How much per kilo for tomatoes?",
    "What should I charge for my yam?",
    "How much should I expect to get per bag of maize?",
    "What are cowpeas fetching at Bodija market?",
    "Give me a ballpark figure for a tonne of cassava",
    "How much do traders pay for garri?",
    "Roughly what am I looking at per bag?",
    "Is two hundred thousand naira reasonable for my harvest?",
    "What are people paying nowadays?",
    "How much would you value my herd at?",
    "What is the current market rate for rice?",
]


@pytest.mark.parametrize("question", MUST_REFUSE, ids=lambda q: q[:44])
def test_a_question_needing_a_figure_is_refused(question: str) -> None:
    verdict = scope.check(question, "en")
    assert not verdict.in_scope, f"answered a price question: {question!r}"
    assert verdict.reason == "live price", verdict.reason

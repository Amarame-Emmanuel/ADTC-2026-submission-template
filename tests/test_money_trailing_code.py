"""A currency code can sit on either side of the number.

WHY THIS FILE EXISTS
--------------------
The money redaction had three years of currency codes listed and matched them
only BEFORE the amount. A real market answer leaked:

    "the premium for aflatoxin-safe maize is around 100 KSH per kg,
     which is approximately [this system has no price data - ...]"

A Kenyan figure, shown to a Nigerian farmer, in a sentence where the redaction
fired correctly on a LATER amount. That is the diagnostic detail: the guard was
awake and working and still let the number through, because the writer put the
code after the digits and only the other order was listed.

This is F-33 again in a new place - a guard that passes its own tests and is
weaker against real output than against the strings it was written with.

THE OTHER HALF: WHAT MUST NOT BE REDACTED
-----------------------------------------
Extending a money pattern rightwards is dangerous in a way that extending it
leftwards is not, because agronomy puts UNITS after numbers. "60 N per hectare"
is nitrogen, not naira. So bare "N" is accepted before a number and never after
one, and the quantity list below is the guard on that.

TWO MORE, FOUND ONLY BY RE-RUNNING THE STREAM
---------------------------------------------
Fixing the trailing code was not enough, and the unit tests could not say so.
Re-running the same question five times through `guarded_stream` produced two
further leaks in three of the five answers:

    "the local market price per kilogram is ZMK 8"
    "approximately 8.00 US$ per kg"

ZMK is the OLD Zambian code. The list carried ZMW, the current one - so a guard
whose currency list covers only currencies still in use will always lag a corpus
of documents written in the past.

"8.00 US$" defeated both orders at once: the leading branch needs digits after
the symbol, and the trailing branch closed on `\b`, which never fires after "$"
because "$" and a following space are two non-word characters with no boundary
between them.

AND: THE PATTERN MUST NOT MATCH THE EMPTY STRING
------------------------------------------------
The first version of this fix introduced a doubled `|`, giving the alternation
an empty branch. `find_money("250 kg per hectare")` then returned twenty-odd
empty matches and redaction would have replaced every character boundary in
every answer with the no-price notice. It raises nothing and passes any test
that only checks amounts ARE caught, so it is asserted directly.
"""

from __future__ import annotations

import pytest

from agbe.rag.safety import MONEY_REDACTION, _MONEY, find_money, redact_money


def test_the_pattern_never_matches_the_empty_string() -> None:
    """An empty alternative would redact every answer into the notice."""
    assert _MONEY.search("") is None
    assert _MONEY.search("no amounts here at all") is None


#: Amounts that must be caught, code on either side of the number.
AMOUNTS = [
    # the leaks, all found in REAL streamed answers
    "the local market price per kilogram is ZMK 8",
    "an average price of 8.00 ZMK per kg",
    "which translates to approximately 8.00 US$ per kg",
    # trailing code
    "around 100 KSH per kg",
    "a premium of 100 KSh per kg",
    "450 NGN per kg",
    "1,200 UGX a litre",
    "traders offered 2500 NGN",
    "about 75 KES a kilo",
    "roughly 3000 TZS",
    "some 40 ZAR per crate",
    "close to 12 GHS",
    "priced at 15 USD",
    # leading code and symbol - must still work
    "KSh 100 per kg",
    "UShs 426,377 per hectare",
    "US$204 per hectare",
    "N45,000 a bag",
    "₦12,500 per basket",
    # spelled-out names - must still work
    "sold at 2500 naira per bag",
    "4,500 Kenyan shillings",
]


@pytest.mark.parametrize("text", AMOUNTS, ids=lambda t: t[:34])
def test_money_is_found(text: str) -> None:
    assert find_money(text), f"leaked: {text!r}"


@pytest.mark.parametrize("text", AMOUNTS, ids=lambda t: t[:34])
def test_money_is_actually_removed(text: str) -> None:
    """Finding it is not the guarantee. The digits must leave the answer."""
    cleaned, removed = redact_money(text)
    assert removed
    assert MONEY_REDACTION in cleaned
    for amount in removed:
        assert amount not in cleaned, f"{amount!r} survived in {cleaned!r}"


#: Quantities that follow a number the same way a currency code does. Every one
#: is ordinary advice, and redacting any of them would be worse than the leak.
NOT_MONEY = [
    "apply 60 N per hectare",
    "apply 120 kg N per hectare",
    "top dress with 30 N at knee height",
    "250 kg per hectare",
    "dry to 13% moisture content",
    "wait 18 days before slaughter",
    "plant 25 cm apart",
    "use 6 bags of fertiliser",
    "2 tonnes of maize",
    "give 250 ml of cooking oil",
    "a 20 kg goat",
    "20mm of rain over four consecutive days",
    "deworm every 3 months",
    "the pH should be 6.0 to 7.0",
    "at least 35 cm of feeding space per goat",
    "count 12 shoots per stand",
    "store for 6 months",
    "a 50 kg bag",
    "space rows 75 cm apart",
    "3 seeds per hole",
]


@pytest.mark.parametrize("text", NOT_MONEY, ids=lambda t: t[:34])
def test_quantities_are_left_alone(text: str) -> None:
    assert find_money(text) == [], text
    cleaned, removed = redact_money(text)
    assert removed == []
    assert cleaned == text

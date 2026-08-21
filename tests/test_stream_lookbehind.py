"""A currency token split across pieces must still be redacted.

WHY THIS FILE EXISTS
--------------------
F-33 recorded that a guard can be weaker in the interface than in its tests,
and fixed the case where the model emitted "$" and "204" separately. This is the
same defect one level deeper, and the earlier fix did not reach it.

Asked "How do I decide what price to ask for my maize?", the interface produced:

    "The local market price per kilogram is ZMK 8."

`find_money` run over that exact finished answer FINDS the amount. The guard was
correct; the stream defeated it. The model emitted the code as "Z", "MK", " 8",
so no single piece ever looked like money: the currency mark never matched, the
guard woke on the bare digit, and by then "ZMK" had already been sent.

WHY LOOKING AT THE PIECE CANNOT WORK
------------------------------------
The previous fix widened what STARTS buffering. That helps only when the trigger
and the amount are in the same piece. Here the trigger is in an EARLIER piece
that is already gone, and no test applied to the current piece can recover it.

So the stream now holds a short tail unsent, and folds that tail into the buffer
when buffering starts. The guard gets to look backwards across the split.

THE RISK THE TAIL INTRODUCES
----------------------------
Text that is held back can be lost. An answer silently missing its last few
characters is a worse defect than the leak, and would show up in no metric - so
every test here checks the FULL text survives, not only that money is gone.

Pieces are streamed one character at a time, which is the worst case for any
guard that matches across a boundary and the case real tokenisers approximate.
"""

from __future__ import annotations

import pytest

from agbe.advisor import AdvisoryEngine
from agbe.rag.safety import MONEY_REDACTION, find_money


class _Chunk:
    text = "Maize marketing depends on local demand."
    title = "Maize marketing"
    year = "2018"

    def citation(self) -> str:
        return "Maize marketing (2018)"


class _Hit:
    chunk = _Chunk()
    score = 0.8
    rank = 0


class _CharLLM:
    """Streams one CHARACTER per piece - the worst case for a split token."""

    def __init__(self, text: str) -> None:
        self.text = text

    def stream(self, messages, **kwargs):
        for ch in self.text:
            yield ch, None


def _run(text: str) -> str:
    engine = AdvisoryEngine.__new__(AdvisoryEngine)
    engine.llm = _CharLLM(text)
    return "".join(
        piece for piece, _stats in engine.guarded_stream("q", [_Hit()], ["text"])
    )


#: Every one of these was produced by the real model, or is the same shape.
LEAKY_ANSWERS = [
    "The local market price per kilogram is ZMK 8. Consider your costs.",
    "For example, the bags are priced at ZMK 8 per kg in that district.",
    "An average price of 8.00 ZMK per kg is reported. Grade before selling.",
    "This translates to approximately 8.00 US$ per kg for good grain.",
    "Traders generally pay KSh 100 per kg for aflatoxin-safe maize.",
    "The premium is around 100 KSH per kg. Ask your buyer what they offer.",
    "Budget about UShs 426,377 per hectare for the season.",
    "Costs reached US$204 per hectare on that trial site.",
    "Some farmers were offered 450 NGN per kg last season.",
    "A bag went for 2500 naira at the depot.",
]


@pytest.mark.parametrize("answer", LEAKY_ANSWERS, ids=lambda a: a[:40])
def test_a_split_currency_token_is_still_redacted(answer: str) -> None:
    streamed = _run(answer)
    assert find_money(streamed) == [], f"leaked through the stream: {streamed!r}"
    assert MONEY_REDACTION in streamed


#: Answers with no money in them at all. Not one character may be lost.
#:
#: The last entry is deliberately shorter than the look-behind window: an
#: implementation that only flushes the tail when it overflows would return an
#: empty answer here, and that is a whole answer lost rather than a few
#: characters.
CLEAN_ANSWERS = [
    "Grade your tomatoes by size and colour before selling. Uniform fruit sells better.",
    "Store maize in hermetic bags to keep the larger grain borer out.",
    "Sell cassava as chips if you cannot move fresh roots within two days.",
    "Apply 60 N per hectare at knee height, then top dress at tasselling.",
    "Space rows 75 cm apart and plant 3 seeds per hole, thinning to 2.",
    "Dry the grain to 13% moisture content before you put it in the store.",
    "Join a cooperative.",
    "Yes.",
]


@pytest.mark.parametrize("answer", CLEAN_ANSWERS, ids=lambda a: a[:40])
def test_no_text_is_lost_to_the_held_back_tail(answer: str) -> None:
    streamed = _run(answer)
    assert streamed == answer, f"expected {answer!r}, got {streamed!r}"


@pytest.mark.parametrize("answer", LEAKY_ANSWERS, ids=lambda a: a[:40])
def test_everything_except_the_amount_survives(answer: str) -> None:
    """Redaction replaces the figure. It must not eat the advice around it."""
    streamed = _run(answer)
    last_words = answer.rsplit(" ", 3)[-1].strip(".")
    assert last_words in streamed, f"tail lost: {streamed!r}"

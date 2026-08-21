"""Generation that produces nothing is retried once before giving up.

WHY THIS FILE EXISTS
--------------------
"My goat left side is swollen and tight after grazing wet grass" produced an
EMPTY answer in 2 of 5 runs - six good passages retrieved, the bloat guidance
sitting in them, and nothing generated. Bloat kills within hours by pressing on
the diaphragm, so telling that farmer "I have no local guidance" when the
guidance is right there is the worst version of this failure.

WHY A RETRY IS SAFE HERE AND NOT IN GENERAL
-------------------------------------------
Nothing has been sent to the caller while `head_done` is False: the head buffer
holds back everything until the first LETTER arrives, so a second generation
cannot duplicate or contradict a first one. That is the whole reason this is a
retry rather than a redesign - the buffering that exists to strip citation dumps
also makes the first attempt discardable.

The retry is bounded at one. An unbounded retry would turn a model that has
stopped producing into a hang, and the farmer is better served by an honest
refusal than by a spinner.
"""

from __future__ import annotations

from agbe.advisor import AdvisoryEngine, NO_GUIDANCE_MESSAGE


class _Chunk:
    text = "Bloat is gas in the rumen. Give cooking oil and walk the animal."
    title = "Goats"
    year = "2018"


class _Hit:
    chunk = _Chunk()
    score = 0.8
    rank = 0


class _ScriptedLLM:
    """Yields whatever each scripted attempt says, in order."""

    def __init__(self, attempts):
        self.attempts = list(attempts)
        self.calls = 0

    def stream(self, messages, **kwargs):
        pieces = self.attempts[min(self.calls, len(self.attempts) - 1)]
        self.calls += 1
        for piece in pieces:
            yield piece, None


def _engine(llm):
    engine = AdvisoryEngine.__new__(AdvisoryEngine)
    engine.llm = llm
    return engine


def _run(llm) -> str:
    engine = _engine(llm)
    return "".join(
        piece for piece, _stats in engine.guarded_stream("q", [_Hit()], ["text"])
    )


def test_an_empty_first_attempt_is_retried() -> None:
    llm = _ScriptedLLM([[], ["It sounds like ", "bloat. Give cooking oil."]])
    answer = _run(llm)
    assert llm.calls == 2, "should have retried"
    assert "bloat" in answer.lower()


def test_a_marker_only_first_attempt_is_retried() -> None:
    """A block of citations is as empty as no tokens at all."""
    llm = _ScriptedLLM([["[1]", " [2]", " [3]"], ["Give cooking oil for bloat."]])
    answer = _run(llm)
    assert llm.calls == 2
    assert "cooking oil" in answer


def test_a_good_first_attempt_is_not_retried() -> None:
    """The retry must not double the cost of the normal path."""
    llm = _ScriptedLLM([["Give cooking oil for bloat."], ["SHOULD NOT BE USED"]])
    answer = _run(llm)
    assert llm.calls == 1
    assert "SHOULD NOT" not in answer


def test_two_empty_attempts_fall_back_to_no_guidance() -> None:
    """Bounded at one retry: an honest refusal beats a spinner."""
    llm = _ScriptedLLM([[], []])
    answer = _run(llm)
    assert llm.calls == 2
    assert answer == NO_GUIDANCE_MESSAGE


def test_the_retry_does_not_duplicate_the_first_attempt() -> None:
    """Nothing was emitted, so the answer is the SECOND attempt alone."""
    llm = _ScriptedLLM([["[1] [2]"], ["Clean advice only."]])
    answer = _run(llm)
    assert answer.strip() == "Clean advice only."

"""An answer must not be citation markers with no advice in it.

WHY THIS FILE EXISTS
--------------------
Asked "my goat has a swollen udder that feels hot" four times, the system
answered correctly twice, once with "[1] [2] [3] [4] [5] [6]" followed by
correct mastitis advice, and once with:

    [1] [2] [3] [4] [5] [6]

and nothing else. Twenty-three characters, six citations, no advice. Sources are
presented as "<passage> (Title, year) [n]" with the marker trailing each
passage, and the model sometimes reproduces the markers as a block before
writing - or instead of writing.

An empty answer is worse than a refusal, because six citations sitting under a
question read as though the system said something.

THE TWO WAYS THIS GUARD CAN FAIL
--------------------------------
INERT      - the block passes through and the farmer sees only markers.

OVER-EAGER - it strips a marker the sentence needed. The model also writes
             "[1] suggests that the swelling is likely an infection", using a
             single marker as the SUBJECT. The first version of this guard
             stripped that and produced "suggests that the swelling..." - an
             answer starting mid-sentence. That is why a RUN of two or more is
             required, and it is pinned below.
"""

from __future__ import annotations

import pytest

from agbe.advisor import strip_citation_dump


def test_a_pure_marker_block_is_reported_empty() -> None:
    cleaned, only_citations = strip_citation_dump("[1] [2] [3] [4] [5] [6]")
    assert only_citations is True


def test_a_leading_block_is_stripped_from_real_advice() -> None:
    cleaned, only_citations = strip_citation_dump(
        "[1] [2] [3] If the goat has a swollen udder it is likely mastitis."
    )
    assert only_citations is False
    assert cleaned.startswith("If the goat")


#: Must survive untouched.
KEEP = [
    # A single marker used as the sentence subject. Stripping it leaves the
    # answer beginning mid-sentence.
    "[1] suggests that the swelling is likely an infection.",
    "[1] Goats (new) recommends that you squeeze out the abscess.",
    # Trailing markers are the intended presentation, not a defect.
    "Mastitis is likely. Isolate the goat. [1] [2]",
    "Use clean planting material. [3]",
    # Ordinary prose.
    "Remove infected plants and destroy them away from the field.",
]


@pytest.mark.parametrize("answer", KEEP)
def test_real_answers_are_untouched(answer: str) -> None:
    cleaned, only_citations = strip_citation_dump(answer)
    assert only_citations is False
    assert cleaned == answer


def test_a_bracketed_number_mid_sentence_is_not_a_leading_run() -> None:
    """The pattern is anchored, so numbers inside the answer are safe."""
    answer = "Apply [2] bags per hectare as the sources describe."
    assert strip_citation_dump(answer) == (answer, False)

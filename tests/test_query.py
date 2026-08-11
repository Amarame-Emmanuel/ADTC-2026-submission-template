"""Query preparation: keep the description, drop the instructions.

The regression these guard against is recorded in agbe/rag/query.py: the
project's own submitted test prompt retrieved development-agency value-chain
reports instead of cassava disease guidance, because the framing vocabulary
outweighed the four words describing the symptom.
"""

from __future__ import annotations

from agbe.rag.query import retrieval_query

FLAGSHIP = (
    "A smallholder farmer in Oyo State, Nigeria says: my cassava leaves are "
    "yellow and twisted, and the plants are small. Explain the most likely "
    "cause, and give practical steps the farmer can take this week. Prefer "
    "cultural and preventive measures over chemicals, and do not state any "
    "pesticide dose."
)


def test_flagship_prompt_keeps_only_the_symptoms():
    out = retrieval_query(FLAGSHIP)
    # The symptoms survive.
    assert "cassava leaves are yellow and twisted" in out
    assert "plants are small" in out
    # The framing that pulled retrieval toward value-chain reports does not.
    for noise in ("smallholder", "Oyo State", "practical steps",
                  "cultural and preventive", "pesticide dose"):
        assert noise not in out, f"{noise!r} should not steer retrieval"


def test_plain_question_is_untouched():
    """A farmer typing into the UI writes no instructions. Nothing to strip."""
    q = "My goats have twisted necks and greenish diarrhoea. What is wrong?"
    assert retrieval_query(q) == q


def test_short_query_survives():
    assert retrieval_query("cassava leaves yellow") == "cassava leaves yellow"


def test_instruction_only_prompt_falls_back_to_original():
    """Never hand the index an empty string.

    Stripping is a retrieval optimisation; if it would remove everything, the
    original query is the safer input. Worse ranking is recoverable, an empty
    query is not.
    """
    q = "Explain the causes. Give practical steps. Do not state any dose."
    assert retrieval_query(q) == q


def test_farmer_imperatives_are_not_mistaken_for_instructions():
    """"Do not" as reported speech, not as an instruction to us.

    The instruction pattern is anchored at clause start precisely so a farmer's
    own words are not shredded. This is the case that anchoring protects.
    """
    q = "The extension officer told me not to burn the plantation at harvest."
    assert retrieval_query(q) == q


def test_reported_speech_frame_is_removed_but_content_kept():
    q = "A farmer in Kano says: my maize has streaks on the leaves."
    out = retrieval_query(q)
    assert out.startswith("my maize has streaks")
    assert "Kano" not in out


def test_dose_wording_still_reaches_the_safety_layer():
    """Stripping must not become the thing that decides what is safe.

    `retrieval_query` is applied ONLY to the text sent to the index. The scope
    and safety guards read the original question. This test states that
    contract at the seam: a dose request is not something query preparation is
    allowed to launder away.
    """
    from agbe.rag import scope

    q = "How many ml of glyphosate per litre should I spray on my maize?"
    # Whatever retrieval does with it, the guard sees the original.
    assert not scope.check(q).in_scope

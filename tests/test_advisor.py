"""Tests for advisory-engine behaviour that is not about retrieval or safety.

Every case here comes from output the system actually produced.
"""

from __future__ import annotations

from agbe.advisor import strip_false_disclaimer
from agbe.rag import scope
from agbe.translate.detect import detect
from agbe.translate.messages import MESSAGES, VALIDATED_LANGUAGES, get


class TestFalseDisclaimer:
    """The model claiming ignorance and then answering anyway.

    Observed verbatim: "I have no local guidance for this situation. The
    symptoms of greenish watery diarrhea ... are consistent with Newcastle
    disease" - followed by correct, sourced advice.

    This is worse than a wrong answer. It teaches a reader that the
    disclaimer is noise, which disarms the refusal path exactly when it
    matters. Retrieval, not the model, decides whether guidance exists.
    """

    def test_strips_disclaimer_followed_by_real_answer(self):
        text = ("I have no local guidance for this situation. The symptoms are "
                "consistent with Newcastle disease.")
        out, stripped = strip_false_disclaimer(text)
        assert stripped
        assert out.startswith("The symptoms")
        assert "no local guidance" not in out

    def test_strips_however_construction(self):
        out, stripped = strip_false_disclaimer(
            "I do not have local guidance on that. However, whiteflies carry the virus."
        )
        assert stripped
        assert "whiteflies" in out

    def test_strips_sources_do_not_contain(self):
        out, stripped = strip_false_disclaimer(
            "The sources do not contain specific details. Remove infected plants."
        )
        assert stripped
        assert out.startswith("Remove infected plants")

    def test_leaves_clean_answer_untouched(self):
        text = "Your cassava has mosaic disease. Remove infected plants."
        out, stripped = strip_false_disclaimer(text)
        assert not stripped
        assert out == text

    def test_leaves_genuine_refusal_untouched(self):
        """A real refusal must survive - it is not a false disclaimer."""
        text = "I cannot tell you how much to use."
        out, stripped = strip_false_disclaimer(text)
        assert not stripped
        assert out == text

    def test_capitalises_after_stripping(self):
        out, _ = strip_false_disclaimer(
            "I have no local guidance on that. remove the infected plants."
        )
        assert out[0].isupper()


class TestScopeGuards:
    """Policy refusals that similarity scoring cannot make.

    Both cases below scored 0.72-0.75 on the held-out evaluation - retrieval
    was working correctly, the questions are simply out of policy scope.
    """

    def test_dosage_question_refused(self):
        v = scope.check("How many millilitres of antibiotic should I inject into my goat?")
        assert not v.in_scope
        assert v.reason == "dosage"

    def test_out_of_scope_crop_refused(self):
        v = scope.check("What is killing my cocoa pods and turning them black?")
        assert not v.in_scope
        assert "cocoa" in v.reason

    def test_human_medical_refused(self):
        v = scope.check("I feel dizzy after spraying my farm. What tablet should I take?")
        assert not v.in_scope
        assert v.reason == "human medical"

    def test_newly_scoped_crops_accepted(self):
        """Crops added after the corpus coverage audit must not be refused."""
        for q in (
            "My rice leaves have grey eye-shaped spots",
            "My groundnut plants are stunted with yellow bunched leaves",
            "Insects are boring holes in my stored cowpea",
            "My pepper plants wilt suddenly",
            "Worms are boring into my okra pods",
        ):
            assert scope.check(q).in_scope, q

    def test_in_scope_crop_beats_incidental_mention(self):
        """A cassava question mentioning cocoa in passing is still in scope."""
        v = scope.check("My cassava is dying. My neighbour grows cocoa.")
        assert v.in_scope

    def test_ordinary_question_allowed(self):
        assert scope.check("My cassava leaves are yellow and twisted").in_scope


class TestLanguageRouting:
    def test_detects_pidgin(self):
        assert detect("Wetin dey worry my cassava? Di leaf dey yellow") == "pcm"

    def test_detects_english(self):
        assert detect("My cassava leaves are yellow and twisted") == "en"

    def test_biased_towards_english(self):
        """One ambiguous word must not flip the language.

        Misrouting Pidgin to English costs a slightly stiff answer the farmer
        can still read; the reverse is odd for no benefit.
        """
        assert detect("Should I sell my maize now or later") == "en"
        assert detect("I do not know what to do") == "en"


class TestValidatedMessages:
    def test_every_validated_language_has_all_messages(self):
        english = MESSAGES["en"]
        for lang in VALIDATED_LANGUAGES:
            msgs = MESSAGES[lang]
            for field in english.__dataclass_fields__:
                value = getattr(msgs, field)
                assert value and value.strip(), f"{lang}.{field} is empty"

    def test_pidgin_differs_from_english(self):
        """Guards against a language silently falling back to copied English."""
        en, pcm = MESSAGES["en"], MESSAGES["pcm"]
        for field in en.__dataclass_fields__:
            assert getattr(en, field) != getattr(pcm, field), field

    def test_unknown_language_falls_back_to_english(self):
        """Falling back is safe; emitting unverified output is not."""
        assert get("hau") is MESSAGES["en"]
        assert get("zz") is MESSAGES["en"]

"""Tests for advisory-engine behaviour that is not about retrieval or safety.

Every case here comes from output the system actually produced.
"""

from __future__ import annotations

from agbe.advisor import strip_false_disclaimer
import pytest

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

    #: Fields deliberately serving English to every language, because no
    #: speaker has reviewed a translation yet. Listed explicitly rather than
    #: skipped, so the debt is visible in the test file and has to be actively
    #: deleted from this set when a speaker approves a string - the failure
    #: mode this guards against is a gap nobody remembers.
    PENDING_SPEAKER_REVIEW = {"harmful_request"}

    def test_pidgin_differs_from_english(self):
        """Guards against a language silently falling back to copied English."""
        en, pcm = MESSAGES["en"], MESSAGES["pcm"]
        for field in en.__dataclass_fields__:
            if field in self.PENDING_SPEAKER_REVIEW:
                continue
            assert getattr(en, field) != getattr(pcm, field), field

    def test_pending_review_fields_are_genuinely_identical(self):
        """An exemption must be an exemption, not a place to hide a change.

        If someone writes a Pidgin string for a listed field but forgets to
        remove it from the set, the guard above stays switched off for a field
        that no longer needs it. This fails in that case.
        """
        en, pcm = MESSAGES["en"], MESSAGES["pcm"]
        for field in self.PENDING_SPEAKER_REVIEW:
            assert getattr(en, field) == getattr(pcm, field), (
                f"{field} now differs - remove it from PENDING_SPEAKER_REVIEW"
            )

    def test_unknown_language_falls_back_to_english(self):
        """Falling back is safe; emitting unverified output is not."""
        assert get("hau") is MESSAGES["en"]
        assert get("zz") is MESSAGES["en"]


class TestLivePriceGap:
    """A price question must be refused however the farmer phrases it.

    `_LIVE_PRICE` refused "what is the price of a bag of maize in Ibadan market
    today?" - the evaluation set's wording - and ANSWERED "what is maize selling
    for in Ibadan today?", because the alternation required `today` to follow
    `selling for` immediately and a place name sat between them. It also
    answered "what are tomatoes selling for in the market?", which carries no
    time word at all.

    The second is the instructive one: the present tense is itself the live
    marker. No price was invented in either case, but section 1 says today's
    price is a fact the system does not have, and answering the question at all
    contradicts that.
    """

    REFUSE = [
        "What is maize selling for in Ibadan today?",
        "What are tomatoes selling for in the market?",
        "what is the price of a bag of maize in Ibadan market today?",
        "How much is a basket of tomatoes going for?",
        "What is the current price of yam?",
    ]

    #: Judgement, not fact. These are answerable from extension material and
    #: were the reason `sell` alone could never be the trigger.
    ANSWER = [
        "When is the best time to sell my cassava?",
        "Should I sell my maize now or store it?",
        "Should I sell my groundnut as seed or for eating?",
        "How do I grade my tomatoes before selling?",
        "How can I add value to my cassava instead of selling it raw?",
        "Is there an advantage to selling together with other farmers?",
    ]

    @pytest.mark.parametrize("question", REFUSE)
    def test_price_lookups_are_refused(self, question):
        verdict = scope.check(question)
        assert not verdict.in_scope
        assert verdict.reason == "live price"

    @pytest.mark.parametrize("question", ANSWER)
    def test_selling_judgement_still_answered(self, question):
        assert scope.check(question).in_scope

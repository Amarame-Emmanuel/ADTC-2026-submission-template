"""Tests for the agricultural safety rules.

These guard the failure modes with real-world consequences: recommending a
banned compound, inventing an application rate, or presenting decades-old
chemical guidance as current.
"""

from __future__ import annotations

import pytest

from agbe.rag.safety import (
    CHEMICAL_RECENCY_YEARS,
    check_answer,
    contains_chemical_context,
    find_dosages,
    find_hazardous_actives,
    is_source_stale,
    redact_unsupported_dosages,
    find_foreign_money,
)


class TestHazardousActives:
    def test_detects_banned_organochlorine(self):
        text = "Older guidance recommended spraying DDT against the pest."
        assert "ddt" in find_hazardous_actives(text)

    def test_detects_who_class_ib_carbamate(self):
        assert "carbofuran" in find_hazardous_actives("Apply carbofuran granules.")

    def test_ignores_unlisted_compound(self):
        assert find_hazardous_actives("Neem oil is effective against aphids.") == []

    def test_does_not_match_substrings(self):
        """'nicotine' must not fire on 'nicotinoid'-style words mid-token."""
        assert find_hazardous_actives("Use a neonicotinoidal product") == []

    def test_answer_mentioning_banned_compound_is_unsafe(self):
        verdict = check_answer(
            "Spray endosulfan on the affected plants.",
            source_texts=["Endosulfan was formerly recommended for this pest."],
        )
        assert not verdict.safe
        assert "endosulfan" in verdict.hazardous_found
        assert "banned or severely restricted" in verdict.as_notice()


class TestDosageDetection:
    def test_detects_rate_per_hectare(self):
        assert find_dosages("Apply 2 litres per hectare of the product.")

    def test_detects_mixing_rate(self):
        assert find_dosages("Mix 20 ml in 20 litres of water.")

    def test_detects_percentage_solution(self):
        assert find_dosages("Use a 0.5% solution for the drench.")

    def test_ignores_non_dosage_numbers(self):
        assert find_dosages("Plant at 1 metre spacing and harvest after 12 months.") == []


class TestUnsupportedDosages:
    def test_invented_dosage_is_flagged(self):
        verdict = check_answer(
            "Apply 2.5 litres per hectare of the recommended fungicide.",
            source_texts=["Apply the recommended fungicide when symptoms appear."],
        )
        assert not verdict.safe
        assert verdict.unsupported_dosages

    def test_grounded_dosage_is_allowed(self):
        source = "Apply mancozeb at 2 kg per hectare when symptoms first appear."
        verdict = check_answer(
            "Apply mancozeb at 2 kg per hectare when you first see symptoms.",
            source_texts=[source],
        )
        assert verdict.unsupported_dosages == []

    def test_grounded_dosage_tolerates_whitespace_differences(self):
        verdict = check_answer(
            "Use 500 ml/ha.",
            source_texts=["The label rate is 500 ml / ha for this crop."],
        )
        assert verdict.unsupported_dosages == []

    def test_redaction_removes_invented_rate(self):
        answer = "Apply 2.5 litres per hectare of the fungicide."
        verdict = check_answer(answer, source_texts=["Apply the fungicide."])
        redacted = redact_unsupported_dosages(answer, verdict)
        assert "2.5 litres per hectare" not in redacted
        assert "extension officer" in redacted


class TestChemicalRecency:
    def test_old_document_is_stale(self):
        assert is_source_stale("1995", now=2026)

    def test_recent_document_is_not_stale(self):
        assert not is_source_stale(str(2026 - CHEMICAL_RECENCY_YEARS + 1), now=2026)

    def test_unknown_year_is_treated_as_stale(self):
        """Missing provenance must fail safe, not fail open."""
        assert is_source_stale("unknown", now=2026)
        assert is_source_stale("", now=2026)
        assert is_source_stale(None, now=2026)  # type: ignore[arg-type]

    def test_chemical_answer_from_only_old_sources_is_flagged(self):
        verdict = check_answer(
            "Spray the recommended insecticide when the pest appears.",
            source_texts=["Chemical control: spray at first sign of infestation."],
            source_years=["1992", "1988"],
            now=2026,
        )
        assert not verdict.safe
        assert verdict.stale_chemical_sources

    def test_chemical_answer_with_one_recent_source_is_allowed(self):
        verdict = check_answer(
            "Spray the recommended insecticide when the pest appears.",
            source_texts=["Chemical control: spray at first sign of infestation."],
            source_years=["1992", "2024"],
            now=2026,
        )
        assert verdict.stale_chemical_sources == []

    def test_non_chemical_answer_from_old_sources_is_fine(self):
        """Old documents remain good on biology and cultural control."""
        verdict = check_answer(
            "Remove and destroy infected plants, and plant clean stem cuttings.",
            source_texts=["Rogue infected plants and use clean planting material."],
            source_years=["1990"],
            now=2026,
        )
        assert verdict.safe


class TestChemicalContext:
    def test_recognises_chemical_discussion(self):
        assert contains_chemical_context("Apply the fungicide at the rate of...")
        assert contains_chemical_context("Chemical control is an option.")

    def test_ignores_cultural_control_only(self):
        assert not contains_chemical_context(
            "Rogue infected plants, rotate crops and weed early."
        )


class TestVeterinarySafety:
    """Livestock advisory carries a hazard the crop side does not.

    A treatment named without its withdrawal period endangers whoever drinks
    the milk or eats the meat - a person who never saw the advice and cannot
    judge it. That asymmetry is why this is enforced rather than left to the
    prompt.
    """

    def test_treatment_without_withdrawal_is_flagged(self):
        verdict = check_answer(
            "Treat the animal with oxytetracycline injection for three days.",
            source_texts=["Oxytetracycline is used for this condition."],
        )
        assert not verdict.safe
        assert verdict.missing_withdrawal_warning
        assert "withdrawal period" in verdict.as_notice()

    def test_treatment_with_withdrawal_is_accepted(self):
        verdict = check_answer(
            "Treat with oxytetracycline. Observe the withdrawal period before "
            "selling milk from the animal.",
            source_texts=["Oxytetracycline is used for this condition."],
        )
        assert not verdict.missing_withdrawal_warning

    def test_deworming_counts_as_treatment(self):
        verdict = check_answer(
            "Give albendazole to the goats every three months.",
            source_texts=["Albendazole is an effective anthelmintic."],
        )
        assert verdict.missing_withdrawal_warning

    def test_restricted_veterinary_drug_is_flagged(self):
        verdict = check_answer(
            "Chloramphenicol will clear the infection. Observe the withdrawal period.",
            source_texts=["Chloramphenicol was formerly used for this."],
        )
        assert not verdict.safe
        assert "chloramphenicol" in verdict.restricted_veterinary
        assert "banned or restricted in food-producing animals" in verdict.as_notice()

    def test_non_treatment_livestock_advice_is_clean(self):
        """Husbandry advice must not trip the veterinary guards."""
        verdict = check_answer(
            "House your chickens off the ground and keep the litter dry. "
            "Separate sick birds from the flock immediately.",
            source_texts=["Keep poultry housing dry and separate sick birds."],
            source_years=["2019"],
            now=2026,
        )
        assert verdict.safe
        assert not verdict.missing_withdrawal_warning


class TestSafeAnswers:
    def test_clean_cultural_advice_passes(self):
        verdict = check_answer(
            "Remove infected plants and use certified cuttings from a clean field.",
            source_texts=["Use clean planting material and rogue infected plants."],
            source_years=["2020"],
            now=2026,
        )
        assert verdict.safe
        assert verdict.as_notice() == ""


class TestForeignCurrency:
    """Prices from another country must be labelled as such.

    Asked "what do traders pay for a bag of maize these days?", the system
    answered "Traders generally pay farmers between KSh$81.21 and KSh$517 per
    50 kg bag... In Bungoma, farmers are willing to pay an average of KSh$87."
    Those are Kenyan shillings, from a CGIAR baseline study in Uasin Gishu and
    Bungoma, given without qualification to a farmer in southwest Nigeria.

    The scope rule now refuses that question, but the exposure is broader:
    AFRICA_TERMS admits Kenyan, Ugandan and Malawian material deliberately,
    because it is good agronomy. Agronomy travels across borders; prices do not.

    A number is required beside the token. Without it "Randomised plots" and
    "the brand of fertiliser" are prices, and the check becomes noise.
    """

    FOREIGN = [
        "Traders pay between KSh 81 and KSh 517 per 50 kg bag",
        "farmers earned 4,500 Kenyan shillings",
        "average of UGX 3000 per sack",
        "about 120 000 TZS for the season",
        "roughly 25 Ghanaian cedis",
    ]

    NOT_FOREIGN = [
        "Sell at N45,000 per bag in Ibadan",
        "the price rose to 1200 naira",
        "Grand total of the harvest",
        "Randomised plots gave higher yields",
        "The brand of fertiliser matters less than the rate",
        "Apply 250 kg per hectare",
        # A currency name with no figure beside it is prose.
        "Kenyan shilling notes were redesigned",
    ]

    @pytest.mark.parametrize("answer", FOREIGN)
    def test_foreign_money_is_found(self, answer):
        assert find_foreign_money(answer)

    @pytest.mark.parametrize("answer", NOT_FOREIGN)
    def test_naira_and_prose_are_left_alone(self, answer):
        assert find_foreign_money(answer) == []

    def test_the_notice_names_the_currency(self):
        v = check_answer(
            "Traders pay between KSh 81 and KSh 517 per bag.", source_texts=[]
        )
        notice = v.as_notice("en")
        assert "KSh" in notice
        assert "not Nigerian prices" in notice

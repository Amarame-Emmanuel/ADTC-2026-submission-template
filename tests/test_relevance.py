"""Regression tests for the corpus content gates.

Every case here came from a real failure during corpus harvesting, not from
imagination. The corpus filter is the component with the least formal grounding
in the system - hand-tuned thresholds, no learned model - so its behaviour is
pinned by examples that actually fooled it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agbe.rag.relevance import (
    ACCEPT_THRESHOLD,
    detect_language,
    score_text,
)

FIXTURES = Path(__file__).parent / "fixtures"


GUIDANCE_TEXT = """
Cassava mosaic disease is the most important viral disease of cassava in
Nigeria and across West Africa. Infected plants show yellowing and mottling of
leaves, curling and distortion of the leaf blade, and severe stunting. Yield
losses in susceptible varieties can be total.

The whitefly Bemisia tabaci transmits the virus between fields, and infected
stem cuttings carry it from season to season. Farmers should rogue infected
plants as soon as symptoms appear and destroy them away from the field.

Control depends on clean planting material. Take stem cuttings only from
plants with no symptoms, or obtain certified cuttings from an extension
officer. Resistant varieties are available and should be planted where
mosaic pressure is high. Spacing of one metre by one metre allows scouting
and reduces whitefly movement between plants. Weeding early keeps alternative
hosts out of the field.
"""

INSTITUTIONAL_TEXT = """
This progress report covers the reporting period and summarises deliverables
against the workplan agreed with the donor. Stakeholders convened for a
two-day workshop; workshop participants reviewed the theory of change and the
impact pathway.

The steering committee approved the revised logframe. Monitoring and
evaluation indicators were updated and the implementing partner confirmed
disbursement against the grant agreement. Focus group discussion findings and
key informant interviews were analysed; respondents were selected from the
sampling frame. Regression results were significant at the five per cent
level. Gender mainstreaming remains a cross-cutting priority for the project
team, and scaling readiness will be assessed in the next reporting period.
"""

# Condensed from the document that actually broke the filter: FAO's
# "Crop production manual - a guide to fruit and vegetable production in the
# Federated States of Micronesia" (2020), which scored 55.8 - the highest of
# any candidate - on agronomic vocabulary alone.
WRONG_CONTINENT_TEXT = """
Starting your own garden means spending less time at the grocery store and
the supermarket. Picking a good spot for your backyard garden is the first
step. A garden near the patio can create ground shade and help with cooling.

Planting and spacing matter for yield. Prepare the seedbed, apply compost and
manure, and follow the recommended seed rate. Transplanting seedlings from
the nursery should follow the planting calendar. Harvest and storage practices
determine quality. Irrigation and drainage must suit local conditions in the
Federated States of Micronesia and across the Pacific island region, where
conditions differ from Queensland and from the United States mainland.
Fertilizer, mulching and weeding remain essential throughout.

Watch for aphids, whiteflies, thrips and caterpillars on the leaves. Fungal
and bacterial infections cause wilting, yellowing, leaf spots and rot. Remove
infected plants, spray only when scouting shows the threshold is exceeded,
and use resistant varieties with crop rotation where mildew or blight occurs.
"""

FRENCH_TEXT = """
Les cartes sont utilisees dans la planification des fermes experimentales.
Cette methode permet aux chercheurs de mieux comprendre les sols et les
cultures dans une region donnee, avec des resultats plus precis pour leur
travail. Les techniques des cartes sont appliquees dans les stations de
recherche pour ameliorer la production agricole des petits producteurs.
"""


class TestGuidanceVersusInstitutionalProse:
    def test_extension_guidance_is_accepted(self):
        result = score_text(GUIDANCE_TEXT * 4)
        assert result.accepted, result.reason
        assert result.score > ACCEPT_THRESHOLD
        assert "cassava" in result.crops

    def test_donor_progress_report_is_rejected(self):
        result = score_text(INSTITUTIONAL_TEXT * 4)
        assert not result.accepted
        assert result.institutional_hits > result.positive_hits

    def test_guidance_outscores_institutional_prose(self):
        assert score_text(GUIDANCE_TEXT * 4).score > score_text(INSTITUTIONAL_TEXT * 4).score


class TestGeographyGate:
    """The Micronesia regression.

    Vocabulary density cannot distinguish audience. Without a geography gate
    this document would have been the single highest-ranked item in the corpus.
    """

    def test_wrong_continent_is_rejected_despite_high_vocabulary_score(self):
        result = score_text(WRONG_CONTINENT_TEXT * 4)
        assert not result.accepted
        assert "out of scope" in result.reason or "no African context" in result.reason

    def test_wrong_continent_still_scores_high_on_vocabulary(self):
        """Documents the exact blind spot: the lexical score alone is fooled."""
        result = score_text(WRONG_CONTINENT_TEXT * 4)
        assert result.score > ACCEPT_THRESHOLD
        assert result.elsewhere_hits > result.africa_hits

    def test_african_guidance_passes_geography_gate(self):
        result = score_text(GUIDANCE_TEXT * 4)
        assert result.africa_hits > 0
        assert result.accepted

    def test_global_reference_with_substantial_african_content_is_kept(self):
        """The false-negative regression.

        "Integrated pest and disease management in major agroecosystems" (2005)
        was rejected at elsewhere=150 vs africa=135 while being the richest
        guidance document harvested. A comprehensive reference that covers
        Africa among other regions must survive the geography gate.
        """
        # Proportioned to mirror the real document: substantial African
        # content (>= the absolute floor) alongside heavier coverage of
        # other regions.
        global_reference = (GUIDANCE_TEXT * 15) + (WRONG_CONTINENT_TEXT * 6)
        result = score_text(global_reference)
        assert result.africa_hits >= 40, "test text must clear the absolute floor"
        assert result.elsewhere_hits > 0, "test text must also reference elsewhere"
        assert result.accepted, (
            f"global reference with substantial African content was rejected: "
            f"{result.summary()}"
        )


class TestLanguageGate:
    def test_french_document_is_rejected(self):
        result = score_text(FRENCH_TEXT * 12)
        assert not result.accepted
        assert "not English" in result.reason

    def test_english_is_detected(self):
        lang, share = detect_language(GUIDANCE_TEXT * 4)
        assert lang == "en"
        assert share > 0.55

    def test_short_samples_do_not_guess(self):
        """Below the evidence floor the detector must abstain, not guess."""
        lang, _ = detect_language("cassava leaves")
        assert lang == "unknown"


class TestNarrowContent:
    def test_glossary_style_listing_is_rejected(self):
        """High organism count, no guidance: a pest name list helps nobody."""
        listing = " ".join(["aphid whitefly mealybug weevil borer nematode thrips"] * 60)
        result = score_text(listing)
        assert not result.accepted


@pytest.mark.skipif(
    not (FIXTURES / "NEGATIVE_micronesia_manual.pdf").exists(),
    reason="fixture PDF not present (not redistributed; fetch via scripts/fetch_corpus.py)",
)
class TestAgainstRealDocument:
    def test_real_micronesia_pdf_is_rejected(self):
        from agbe.rag.extract import probe

        extraction = probe(FIXTURES / "NEGATIVE_micronesia_manual.pdf")
        assert extraction.usable, "fixture should have a text layer"
        result = score_text(extraction.text)
        assert not result.accepted, (
            f"the Micronesia manual must stay rejected; got {result.summary()}"
        )

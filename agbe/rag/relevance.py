"""Content-level relevance scoring for corpus candidates.

WHY METADATA GATES ARE NOT ENOUGH
---------------------------------
The repository metadata gates in `scripts/fetch_corpus.py` accepted 40 of 98
candidates, but inspection showed a large fraction were institutional prose:
donor progress reports, workshop write-ups and productivity-growth analyses.
CGSpace assigns `dcterms.type = Report` to a farmer training manual and to a
programme's technical progress report alike, so type alone cannot separate
them.

The distinguishing signal is in the writing, not the metadata. Extension
guidance talks about wilting leaves, larvae, spacing and roguing. Project prose
talks about stakeholders, deliverables, work packages and disbursement. This
module measures that difference.

DESIGN CHOICES
--------------
Lexical, not model-based. A scorer that needed the LLM would cost minutes per
document and burn budget we have reserved for serving farmers, to answer a
question a word list answers in milliseconds. It also has to be *auditable*:
a reviewer can read this file and see exactly why a document was accepted,
which is not true of an embedding threshold.

Every score carries its matched terms, because the output of this module is a
shortlist a human reviews - and "rejected, score 0.31" is not reviewable
while "rejected: 4 guidance terms, 37 institutional terms" is.

Ambiguous vocabulary ("capacity building", "value chain", "smallholder") is
deliberately in neither list. Those words appear in good and bad documents
alike and would only add noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Positive signal: the vocabulary of practical crop-health guidance
# --------------------------------------------------------------------------

SYMPTOM_TERMS = {
    "wilting", "wilt", "yellowing", "chlorosis", "curling", "mottling",
    "mosaic", "streak", "lesion", "lesions", "spots", "blight", "rot",
    "rotting", "stunted", "stunting", "necrosis", "necrotic", "galls",
    "dieback", "canker", "mildew", "rust", "scorch", "defoliation",
    "discoloured", "discolored", "malformed", "shrivelled",
}

ORGANISM_TERMS = {
    "larva", "larvae", "caterpillar", "caterpillars", "aphid", "aphids",
    "mite", "mites", "whitefly", "whiteflies", "mealybug", "mealybugs",
    "weevil", "weevils", "borer", "borers", "nematode", "nematodes",
    "thrips", "beetle", "beetles", "grub", "grubs", "armyworm", "moth",
    "fungus", "fungal", "bacterial", "bacterium", "virus", "viral",
    "pathogen", "infestation", "infested", "vector", "grasshopper",
    "termite", "termites", "scale insect",
}

CONTROL_TERMS = {
    "spray", "spraying", "roguing", "rogue", "uproot", "prune", "pruning",
    "rotation", "intercrop", "intercropping", "resistant variety",
    "resistant varieties", "tolerant variety", "clean planting material",
    "certified seed", "weeding", "mulch", "mulching", "trap", "trapping",
    "biological control", "natural enemies", "biopesticide", "neem",
    "sanitation", "solarisation", "fallow", "destroy infected",
    "remove infected", "burn residues", "handpick", "handpicking",
    "threshold", "scouting", "monitor the crop",
}

AGRONOMY_TERMS = {
    "spacing", "planting", "sowing", "transplanting", "harvest",
    "harvesting", "storage", "cuttings", "stem cuttings", "seed rate",
    "ridges", "mounds", "manure", "compost", "fertiliser", "fertilizer",
    "topdressing", "irrigation", "drainage", "soil fertility", "yield",
    "germination", "nursery", "seedbed", "field", "farmers should",
}

# The four advisory domains are crop, livestock, weather and market. The
# lexicons below carry the last three; without them the content gate scores
# a livestock health manual or a grain-storage marketing guide as irrelevant,
# because every positive term it knows about is botanical.

LIVESTOCK_TERMS = {
    "cattle", "cow", "cows", "calf", "calves", "goat", "goats", "sheep",
    "lamb", "poultry", "chicken", "chickens", "hen", "hens", "cockerel",
    "broiler", "layer", "pig", "pigs", "swine", "livestock", "herd",
    "flock", "grazing", "browse", "fodder", "forage", "silage",
    "deworming", "dewormer", "vaccination", "vaccinate", "newcastle",
    "foot and mouth", "mastitis", "brucellosis", "trypanosomiasis",
    "tick", "ticks", "tsetse", "anthelmintic", "withdrawal period",
    "calving", "kidding", "farrowing", "weaning", "stocking rate",
    "veterinary", "animal health", "lameness", "diarrhoea", "diarrhea",
}

WEATHER_TERMS = {
    "rainfall", "rains", "rainy season", "dry season", "harmattan",
    "onset of rains", "cessation", "planting calendar", "planting date",
    "growing season", "drought", "dry spell", "flooding", "waterlogging",
    "humidity", "temperature", "agro-ecological zone", "agroecological",
    "climate", "seasonal forecast", "early planting", "late planting",
    "soil moisture", "evapotranspiration", "mulch to conserve moisture",
}

MARKET_TERMS = {
    "market", "markets", "price", "prices", "marketing", "buyer",
    "buyers", "trader", "traders", "middleman", "aggregation",
    "bulking", "grading", "quality standard", "value addition",
    "processing", "gari", "flour", "storage loss", "post-harvest loss",
    "postharvest", "shelf life", "packaging", "transport cost",
    "cooperative", "farmer group", "contract", "off-taker",
    "gross margin", "profit", "cost of production", "sell",
}

# --------------------------------------------------------------------------
# Negative signal: the vocabulary of institutional and donor reporting
# --------------------------------------------------------------------------

INSTITUTIONAL_TERMS = {
    "stakeholder", "stakeholders", "deliverable", "deliverables",
    "work package", "workplan", "milestone", "milestones", "donor",
    "donors", "consortium", "theory of change", "logframe", "log frame",
    "monitoring and evaluation", "inception report", "steering committee",
    "disbursement", "budget allocation", "co-funding", "grant agreement",
    "annual report", "progress report", "workshop participants",
    "focus group discussion", "key informant", "questionnaire",
    "respondents", "sampling frame", "regression", "econometric",
    "randomised controlled", "randomized controlled", "p-value",
    "significant at", "gender mainstreaming", "empowerment index",
    "terms of reference", "project team", "implementing partner",
    "scaling readiness", "impact pathway", "reporting period",
}

POSITIVE_GROUPS = {
    "symptom": SYMPTOM_TERMS,
    "organism": ORGANISM_TERMS,
    "control": CONTROL_TERMS,
    "agronomy": AGRONOMY_TERMS,
    "livestock": LIVESTOCK_TERMS,
    "weather": WEATHER_TERMS,
    "market": MARKET_TERMS,
}

#: Which advisory area a document serves. A document usually serves one; the
#: tag drives corpus-coverage reporting per area, so a gap in livestock or
#: market coverage is visible rather than hidden inside an average.
ADVISORY_AREAS = ("crop", "livestock", "weather", "market")

#: Scope: Nigerian smallholder staples and common vegetables. Widened after a
#: coverage audit showed several of these were already well represented in the
#: corpus while being refused - see agbe/rag/scope.py.
CROP_TERMS = {
    "cassava": {"cassava", "manihot"},
    "maize": {"maize", "zea mays"},
    "yam": {"yam", "yams", "dioscorea"},
    "tomato": {"tomato", "tomatoes", "lycopersicon"},
    "rice": {"rice", "oryza"},
    "cowpea": {"cowpea", "cowpeas", "vigna unguiculata"},
    "groundnut": {"groundnut", "groundnuts", "arachis"},
    "pepper": {"pepper", "peppers", "capsicum"},
    "okra": {"okra", "abelmoschus"},
}

#: Crops that are neither in scope nor recognised by CROP_TERMS above.
#:
#: WHY THIS EXISTS SEPARATELY FROM CROP_TERMS
#: `_demote_off_crop` partitions candidates using CROP_TERMS, which lists the
#: NINE IN-SCOPE crops. A passage about a crop that is neither asked about nor
#: in scope matches nothing, is classified NEUTRAL, and is therefore never
#: demoted - the docstring's "neutral counts as on-topic, deliberately" was
#: written about passages naming NO crop and silently also covered passages
#: naming a crop the vocabulary had never heard of.
#:
#: Two defects reduced to that one gap:
#:   - "my rice leaves have orange-brown spots" retrieved "Bean Leaves (New)"
#:     at 0.719 and answered with angular leaf spot caused by Phaeoisariopsis
#:     griseola, a BEAN disease. Brown leaf spot (Bipolaris oryzae) was in the
#:     corpus and had been retrieved.
#:   - crop-22, the last remaining coverage gap, retrieved "Papaya (Revised)
#:     (Aphis gossypii)" for a tomato leafminer question.
#:
#: Neither `bean` nor `papaya` appears in scope.OUT_OF_SCOPE_CROPS, which exists
#: to refuse QUESTIONS about other crops and is a different job.
OTHER_CROP_TERMS = {
    "bean", "beans", "phaseolus", "papaya", "pawpaw", "carica",
    "mango", "citrus", "avocado", "pineapple", "watermelon",
    "cocoa", "cacao", "coffee", "cashew", "oil palm",
    "banana", "plantain", "sugarcane", "cotton",
    "sorghum", "millet", "wheat", "sesame", "soybean", "soya", "soyabean",
    "sweet potato", "sweetpotato", "cocoyam", "taro",
    "onion", "cabbage", "carrot", "lettuce", "cucumber",
}

# --------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------
# Added after a live failure. "Crop production manual: A guide to fruit and
# vegetable production in the Federated States of Micronesia" (FAO, 2020)
# scored 55.8 - the highest of any candidate - because it is dense with
# planting, spacing, compost and harvest vocabulary. It is entirely legitimate
# extension material for the wrong ocean.
#
# Vocabulary density says nothing about who a document was written for. This
# is the cheapest available correction, and it is still only a filter: it
# cannot tell a Kenyan highland manual from a Nigerian lowland one, which is
# why human review of the shortlist remains mandatory rather than advisory.

AFRICA_TERMS = {
    "africa", "african", "nigeria", "nigerian", "west africa", "sub-saharan",
    "subsaharan", "ghana", "benin", "togo", "cameroon", "côte d'ivoire",
    "cote d'ivoire", "ivory coast", "kenya", "tanzania", "uganda", "ethiopia",
    "malawi", "zambia", "mozambique", "burkina", "mali", "senegal",
    "sahel", "savanna", "savannah", "ibadan", "lagos", "oyo state",
    "iita", "icipe", "cgiar",
}

ELSEWHERE_TERMS = {
    "micronesia", "pacific island", "caribbean", "latin america",
    "south america", "central america", "philippines", "indonesia",
    "vietnam", "thailand", "bangladesh", "sri lanka", "nepal",
    "australia", "new zealand", "european union", "united kingdom",
    "united states", "u.s. department", "usda", "canada", "california",
    "florida", "queensland", "samoa", "fiji", "papua new guinea",
    "supermarket", "grocery store", "backyard garden", "patio", "lawn",
}

#: A document must not be dominated by non-African regional context. Some
#: mention of elsewhere is normal - pest biology is global, and a Nigerian
#: manual may cite research from India - so this is a ratio, not a veto.
MAX_ELSEWHERE_RATIO = 1.0

#: Ratio alone was too blunt and produced a costly false negative.
#:
#: "Integrated pest and disease management in major agroecosystems" (2005) was
#: rejected at elsewhere=150 vs africa=135, despite being the richest guidance
#: document harvested (organism 1232, symptom 239, control 117). It is a global
#: IPM reference that covers African agroecosystems substantially - it simply
#: covers other continents too.
#:
#: The question worth asking is not "is this document exclusively African?" but
#: "does it contain substantial African material?". A document clearing this
#: absolute floor is admitted regardless of ratio. The Micronesia manual, the
#: case the geography gate exists for, scores 3 - an order of magnitude below
#: the floor - so it stays rejected.
#:
#: This deliberately trades a little precision for recall on comprehensive
#: references, because retrieval is per-passage: an irrelevant chapter on
#: Queensland is simply never retrieved for a cassava query, whereas a rejected
#: document is unavailable in its entirety.
MIN_AFRICA_ABSOLUTE = 40

# --------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------
# CGSpace carries substantial francophone West African material (Benin, Togo,
# Côte d'Ivoire, Cameroon). A live harvest pulled
# "Utilisation des cartes dans la planification des fermes..." (1993).
#
# That content is regionally relevant but unusable by this pipeline: the
# embedder is bge-small-EN, so French text embeds into the wrong part of the
# space and retrieves poorly, and the retrieved passage would then reach an
# English-prompted model. Half-supporting a language is worse than not
# supporting it - the failure is silent and looks like bad retrieval.
#
# Rejected here and recorded in REPORT.md as a known limitation. Adding French
# would mean a multilingual embedder (larger) or a second translation
# direction, and neither is affordable before Gate 1.

STOPWORDS = {
    "en": {"the", "and", "of", "to", "is", "are", "with", "for", "that",
           "this", "from", "be", "have", "which", "can", "should"},
    "fr": {"les", "des", "une", "dans", "pour", "sur", "avec", "est",
           "sont", "cette", "aux", "par", "plus", "leur", "être", "nous"},
    "es": {"los", "las", "una", "para", "con", "por", "que", "del",
           "como", "más", "este", "son", "ser", "está"},
    "pt": {"dos", "das", "uma", "para", "com", "por", "que", "não",
           "mais", "como", "são", "está", "pelo"},
}

#: English must be the clear plurality. A genuine English document with a
#: French abstract or a Spanish reference list still passes comfortably.
MIN_ENGLISH_RATIO = 0.55

#: Institutional prose is penalised at 2x. A document can legitimately mention
#: a workshop in passing; one that is *mostly* project reporting must not
#: survive on the strength of a few incidental agronomic words.
INSTITUTIONAL_PENALTY = 2.0

#: Per 1000 words. A genuine extension manual sits well above this; the
#: progress reports we sampled sit far below it.
ACCEPT_THRESHOLD = 6.0

#: A document must show real guidance vocabulary, not merely a low penalty.
#: Without this, a document mentioning nothing at all would pass on a
#: technicality.
#:
#: SCALED BY LENGTH, not absolute. This threshold was calibrated on
#: hundred-page PDFs; applied unchanged to the Infonet web archive it rejected
#: a genuine 2,300-character page on calf coccidiosis that scored 36.6 against
#: an acceptance threshold of 6.0. A short page cannot accumulate 25 term
#: occurrences however good it is, so an absolute floor penalises documents for
#: being concise rather than for being irrelevant.
MAX_POSITIVE_HITS_REQUIRED = 25
MIN_POSITIVE_HITS_FLOOR = 6
#: One required hit per this many words, between the floor and the maximum.
WORDS_PER_REQUIRED_HIT = 40


def required_positive_hits(n_words: int) -> int:
    """Length-scaled minimum for guidance vocabulary."""
    return max(
        MIN_POSITIVE_HITS_FLOOR,
        min(MAX_POSITIVE_HITS_REQUIRED, n_words // WORDS_PER_REQUIRED_HIT),
    )

#: Guidance must be varied. A glossary listing 200 pest names scores high on a
#: single group while telling a farmer nothing about what to do.
MIN_GROUPS_PRESENT = 2

#: ...but breadth alone is the wrong discriminator, and a real page proved it.
#:
#: Infonet's "Calf white scours / diarrhoea" page scores 53 livestock terms and
#: zero in every other group. It is excellent, focused livestock guidance, and
#: the breadth rule rejected it at a relevance score of 75.8.
#:
#: What actually separates a glossary from a disease page is not how many
#: vocabularies it touches but whether it is *prose*. A word list is almost
#: entirely search terms; real guidance scatters those terms through sentences.
#: A document concentrated in one vocabulary is therefore admitted provided its
#: term density looks like writing rather than an index.
MAX_PROSE_TERM_DENSITY = 0.30


@dataclass
class RelevanceScore:
    """An explainable verdict on one document."""

    score: float
    accepted: bool
    reason: str
    n_words: int
    positive_hits: int
    institutional_hits: int
    group_counts: dict[str, int] = field(default_factory=dict)
    top_terms: list[tuple[str, int]] = field(default_factory=list)
    crops: list[str] = field(default_factory=list)
    africa_hits: int = 0
    elsewhere_hits: int = 0
    language: str = "unknown"
    areas: list[str] = field(default_factory=list)

    def summary(self) -> str:
        groups = ", ".join(f"{g}={n}" for g, n in sorted(self.group_counts.items()) if n)
        crops = ",".join(self.crops) if self.crops else "general"
        return (
            f"score={self.score:6.2f} [{'ACCEPT' if self.accepted else 'REJECT'}] "
            f"lang={self.language} areas={'/'.join(self.areas) or '-'} "
            f"crops={crops} | {groups} "
            f"| institutional={self.institutional_hits} "
            f"| africa={self.africa_hits}/elsewhere={self.elsewhere_hits} "
            f"| {self.reason}"
        )


def _count(text: str, terms: set[str]) -> tuple[int, dict[str, int]]:
    """Count whole-word/phrase occurrences, case-insensitively."""
    hits: dict[str, int] = {}
    total = 0
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"(?:s\b|\b)" if " " not in term else re.escape(term)
        n = len(re.findall(pattern, text))
        if n:
            hits[term] = n
            total += n
    return total, hits


def detect_language(text: str) -> tuple[str, float]:
    """Cheap stopword-frequency language guess.

    Returns (best_language, english_share). A full language-ID model would be
    more accurate, but this runs in milliseconds with no dependency and only
    has to answer one question: is this document mostly English?
    """
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœáíóúü]+", text.lower())
    if not words:
        return "unknown", 0.0

    sample = words[:20000]
    counts = {
        lang: sum(1 for w in sample if w in stops)
        for lang, stops in STOPWORDS.items()
    }
    total = sum(counts.values())
    if total < 20:
        return "unknown", 0.0

    best = max(counts, key=lambda k: counts[k])
    return best, counts["en"] / total


def score_text(text: str) -> RelevanceScore:
    """Score a document's extracted text for farmer-facing relevance."""
    lowered = text.lower()
    n_words = max(len(lowered.split()), 1)
    per_1k = 1000.0 / n_words

    group_counts: dict[str, int] = {}
    all_hits: dict[str, int] = {}
    positive_hits = 0

    for group, terms in POSITIVE_GROUPS.items():
        n, hits = _count(lowered, terms)
        group_counts[group] = n
        positive_hits += n
        all_hits.update(hits)

    institutional_hits, _ = _count(lowered, INSTITUTIONAL_TERMS)
    africa_hits, _ = _count(lowered, AFRICA_TERMS)
    elsewhere_hits, _ = _count(lowered, ELSEWHERE_TERMS)

    crops = sorted(
        crop for crop, terms in CROP_TERMS.items()
        if any(t in lowered for t in terms)
    )

    # Which advisory areas this document actually serves. Thresholded rather
    # than binary: a crop manual that mentions "market" once in a preface is
    # not market advisory material.
    #
    # Scaled by length for the same reason as the guidance floor - a 350-word
    # page on one pest is real crop material and will never show ten symptom
    # terms.
    area_floor = max(3, min(10, n_words // 80))
    areas: list[str] = []
    if (group_counts.get("symptom", 0) + group_counts.get("agronomy", 0)
            + group_counts.get("organism", 0) + group_counts.get("control", 0)) >= area_floor:
        areas.append("crop")
    if group_counts.get("livestock", 0) >= area_floor:
        areas.append("livestock")
    if group_counts.get("weather", 0) >= area_floor:
        areas.append("weather")
    if group_counts.get("market", 0) >= area_floor:
        areas.append("market")

    score = (positive_hits * per_1k) - (INSTITUTIONAL_PENALTY * institutional_hits * per_1k)
    groups_present = sum(1 for n in group_counts.values() if n > 0)

    # Fraction of words that are vocabulary hits. Prose sits well under 0.3;
    # a glossary or tag list approaches 1.0.
    term_density = positive_hits / n_words

    language, english_share = detect_language(text)

    if language not in ("en", "unknown") and english_share < MIN_ENGLISH_RATIO:
        accepted, reason = False, (
            f"not English (detected {language}, English share {english_share:.0%})"
        )
    elif positive_hits < required_positive_hits(n_words):
        accepted, reason = False, (
            f"too little guidance vocabulary ({positive_hits} hits, "
            f"need {required_positive_hits(n_words)} for {n_words} words)"
        )
    elif groups_present < MIN_GROUPS_PRESENT and term_density > MAX_PROSE_TERM_DENSITY:
        accepted, reason = False, (
            f"term listing, not guidance ({groups_present} vocabulary, "
            f"density {term_density:.0%})"
        )
    elif score < ACCEPT_THRESHOLD:
        accepted, reason = False, f"institutional prose dominates (score {score:.2f})"
    elif africa_hits >= MIN_AFRICA_ABSOLUTE:
        # Substantial African content: admitted even if the document also
        # covers other regions. Retrieval filters at passage level anyway.
        accepted, reason = True, f"practical guidance (African content {africa_hits})"
    elif africa_hits == 0 and elsewhere_hits > 0:
        accepted, reason = False, (
            f"no African context, {elsewhere_hits} references elsewhere"
        )
    elif elsewhere_hits > MAX_ELSEWHERE_RATIO * max(africa_hits, 1):
        accepted, reason = False, (
            f"regionally out of scope (elsewhere {elsewhere_hits} vs Africa {africa_hits})"
        )
    else:
        accepted, reason = True, "practical guidance"

    top_terms = sorted(all_hits.items(), key=lambda kv: -kv[1])[:12]

    return RelevanceScore(
        score=score,
        accepted=accepted,
        reason=reason,
        n_words=n_words,
        positive_hits=positive_hits,
        institutional_hits=institutional_hits,
        group_counts=group_counts,
        top_terms=top_terms,
        crops=crops,
        africa_hits=africa_hits,
        elsewhere_hits=elsewhere_hits,
        language=language,
        areas=areas,
    )

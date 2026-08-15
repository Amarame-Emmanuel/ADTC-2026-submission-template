"""Safety rules for agricultural advice.

This system tells farmers what to do about diseased crops, and some of that
advice concerns pesticides. The failure modes are not embarrassing, they are
harmful: a wrong active ingredient, an invented dose, or a control measure
copied faithfully out of a 1990 manual recommending a compound that has since
been banned in most of the world.

Three rules, each guarding a different failure:

1. CHEMICAL RECENCY
   Old documents remain excellent on botany, symptoms and pest biology - none
   of which changes - but pesticide registration changes constantly. A passage
   older than the threshold may inform identification and cultural control and
   must not be the basis of a chemical recommendation.

2. HAZARDOUS ACTIVE INGREDIENTS
   Some compounds appearing in older extension literature are WHO Class Ia/Ib
   or listed under the Rotterdam Convention. Retrieval will surface them
   because they are genuinely in the corpus. They must be flagged, never
   recommended.

3. NO INVENTED DOSAGES
   A dose or a mixing rate that does not appear in the retrieved passages must
   not appear in the answer. This is the highest-risk hallucination available
   to the system and the cheapest to detect, because a dosage has a
   recognisable numeric shape.

WHAT THIS IS NOT
----------------
Not a substitute for national pesticide registration. Nigeria's registration
data (NAFDAC) is not openly published, so this system cannot tell a farmer
whether a product is legally registered for a crop in their state. The correct
answer in that situation is to defer to an extension officer, and the advisory
layer does exactly that rather than guessing.

The hazardous-ingredient list below is curated, not exhaustive. It is weighted
towards compounds that plausibly appear in older African extension material.
A compound's absence from this list is not evidence that it is safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

#: A chemical recommendation sourced from a document older than this is
#: suppressed. Ten years is a judgement call: long enough to retain most
#: useful extension literature, short enough to exclude the era when several
#: now-banned organochlorines and organophosphates were routine advice.
CHEMICAL_RECENCY_YEARS = 10

#: WHO Class Ia/Ib and Rotterdam Convention Annex III compounds, plus actives
#: widely withdrawn from use, that realistically appear in older extension
#: literature for African crops. Curated, not exhaustive - see module docstring.
HAZARDOUS_ACTIVES = {
    # Organochlorines - largely banned, common in pre-2000 manuals
    "ddt", "aldrin", "dieldrin", "endrin", "chlordane", "heptachlor",
    "lindane", "hch", "bhc", "toxaphene", "endosulfan",
    "hexachlorobenzene", "pentachlorophenol",
    # Organophosphates - WHO Ia/Ib
    "parathion", "methyl parathion", "parathion-methyl", "monocrotophos",
    "methamidophos", "phosphamidon", "dichlorvos", "ddvp", "disulfoton",
    "phorate", "terbufos", "fenamiphos", "ethoprophos", "sulfotep",
    "azinphos-methyl", "methidathion", "triazophos", "omethoate",
    "dicrotophos", "coumaphos", "edifenphos", "mecarbam", "thiometon",
    "vamidothion", "isoxathion", "isazofos", "famphur", "cadusafos",
    "chlorethoxyfos", "chlormephos", "heptenophos", "propetamphos",
    "oxydemeton-methyl", "tepp", "epn",
    # Carbamates - WHO Ia/Ib
    "aldicarb", "carbofuran", "methomyl", "methiocarb", "oxamyl",
    "formetanate", "furathiocarb", "thiofanox", "carbaryl",
    # Other high-hazard / heavily restricted
    "paraquat", "captafol", "binapacryl", "dinoseb", "dinoterb", "dnoc",
    "fluoroacetamide", "sodium fluoroacetate", "sodium arsenite",
    "sodium cyanide", "lead arsenate", "paris green", "zinc phosphide",
    "mercuric chloride", "mercuric oxide", "phenylmercury acetate",
    "methyl bromide", "ethylene dibromide", "alachlor", "tributyltin",
    "brodifacoum", "warfarin", "nicotine", "flucythrinate",
    # Restricted for CHRONIC harm, not acute toxicity.
    #
    # The list above screens on WHO Ia/Ib and Rotterdam Annex III - that is, on
    # how quickly a compound kills an adult. That criterion has a hole, and the
    # system fell into it: asked about cutworms it recommended "carbaryl or
    # chlorpyrifos". Carbaryl was caught and the banned-pesticide warning fired
    # correctly. Chlorpyrifos was not, because it is WHO Class II - moderately
    # hazardous acutely - and by the stated criterion it belonged outside.
    #
    # Chlorpyrifos is banned across the EU and severely restricted in the US.
    # The reason is developmental neurotoxicity in children, which acute
    # toxicity class does not measure at all. Mancozeb is the same shape: WHO
    # Class III, banned in the EU in 2021 as a suspected endocrine disruptor and
    # reproductive toxicant.
    #
    # A screen that asks only "how fast does this kill" will keep missing the
    # compounds withdrawn for what they do to children slowly. These are the
    # ones a smallholder is most likely to be sold, precisely because their
    # acute hazard class kept them on the shelf longest.
    "chlorpyrifos", "chlorpyrifos-methyl", "mancozeb", "maneb", "zineb",
    "thiram", "ziram", "propineb", "atrazine", "diuron", "linuron",
    "chlorothalonil", "iprodione", "procymidone", "vinclozolin",
    "carbendazim", "benomyl", "thiophanate-methyl", "fipronil",
    "imidacloprid", "clothianidin", "thiamethoxam", "acetamiprid",
    "dimethoate", "malathion", "profenofos", "quinalphos", "fenthion",
}

#: Dosage-shaped patterns. Deliberately broad: a false positive costs one
#: unnecessary deferral, a false negative can cost a farmer their crop or
#: their health.
# Units are listed longest-first so the alternation cannot settle on a short
# prefix ("l" inside "litres"), and every unit takes an optional plural - the
# original patterns missed "2 litres per hectare", which is how the rate is
# actually written in extension literature.
_UNIT = (
    r"(?:millilitres?|milliliters?|litres?|liters?|kilograms?|kilogrammes?|"
    r"grammes?|grams?|ml|kg|g|l|oz|lb)"
)
_PER_UNIT = (
    r"(?:hectares?|acres?|plants?|trees?|stands?|square\s+metres?|"
    r"litres?|liters?|ha|m2|l)"
)

_DOSAGE_PATTERNS = [
    # 2 litres per hectare, 500 ml/ha, 1.5 kg per acre
    re.compile(
        rf"\b\d+(?:[.,]\d+)?\s*{_UNIT}\s*(?:per|/)\s*{_PER_UNIT}\b",
        re.IGNORECASE,
    ),
    # 20 ml in 20 litres of water
    re.compile(
        rf"\b\d+(?:[.,]\d+)?\s*{_UNIT}\b[^.]{{0,30}}?\bin\b"
        rf"[^.]{{0,20}}?\b\d+(?:[.,]\d+)?\s*(?:litres?|liters?|gallons?|l)\b",
        re.IGNORECASE,
    ),
    # dilution ratios: 1:200
    re.compile(r"\b(?:dilut|ratio|mix)\w*[^.]{0,25}?\b\d+\s*:\s*\d+\b", re.IGNORECASE),
    # concentration: 0.5% solution
    re.compile(r"\b\d+(?:[.,]\d+)?\s*%\s*(?:solution|concentration|a\.i\.|active)",
               re.IGNORECASE),
]

#: Words that signal the surrounding text is about chemical control at all.
#: Used to decide whether the recency rule is even relevant to a passage.
_CHEMICAL_CONTEXT = re.compile(
    r"\b(?:pesticide|insecticide|fungicide|herbicide|nematicide|acaricide|"
    r"spray|spraying|chemical control|active ingredient|formulation|"
    r"dose|dosage|apply\w*\s+(?:at|the rate)|rate of application)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Veterinary safety
# ---------------------------------------------------------------------------
# Livestock advisory carries a hazard the crop side does not: residues.
#
# An animal treated with an antibiotic or anthelmintic must not have its milk
# or meat consumed until the withdrawal period has passed. Advice that names a
# treatment and omits the withdrawal period is not incomplete - it is unsafe,
# and the harm lands on whoever drinks the milk rather than on the farmer who
# asked. That person never saw the advice and cannot judge it.
#
# So: any answer recommending a veterinary treatment must mention withdrawal,
# or one is appended.

VETERINARY_TREATMENT = re.compile(
    r"\b(?:antibiotic|antibiotics|oxytetracycline|tetracycline|penicillin|"
    r"streptomycin|sulphonamide|sulfonamide|tylosin|enrofloxacin|"
    r"anthelmintic|dewormer|deworming|albendazole|levamisole|ivermectin|"
    r"fenbendazole|acaricide|dip|drench|injectable|vaccinate|vaccination|"
    r"trypanocide|diminazene|isometamidium)\b",
    re.IGNORECASE,
)

WITHDRAWAL_MENTIONED = re.compile(
    r"\b(?:withdrawal period|withdrawal time|withholding period|"
    r"do not (?:drink|consume|sell) (?:the )?milk|"
    r"milk withdrawal|meat withdrawal|residue)\b",
    re.IGNORECASE,
)

#: Veterinary drugs banned or heavily restricted for food-producing animals in
#: many jurisdictions, and still present in older extension literature.
#: Curated, not exhaustive - the same caveat as the pesticide list.
RESTRICTED_VETERINARY = {
    "chloramphenicol",      # banned in food animals in most jurisdictions
    "nitrofuran", "nitrofurans", "furazolidone",
    "dimetridazole", "metronidazole", "ronidazole",
    "clenbuterol",
    "diethylstilbestrol", "stilbenes",
    "phenylbutazone",
}


@dataclass
class SafetyVerdict:
    """Outcome of checking one candidate answer against its sources."""

    safe: bool
    warnings: list[str] = field(default_factory=list)
    hazardous_found: list[str] = field(default_factory=list)
    unsupported_dosages: list[str] = field(default_factory=list)
    stale_chemical_sources: list[str] = field(default_factory=list)
    restricted_veterinary: list[str] = field(default_factory=list)
    missing_withdrawal_warning: bool = False

    def as_notice(self, language: str = "en") -> str:
        """Farmer-facing text appended to an answer when something was caught.

        Rendered from human-validated fixed strings rather than translated at
        runtime. These are the highest-consequence sentences the system emits;
        a machine translation of "do not use this pesticide" that reads as a
        suggestion has failed in the way that matters most, and failed
        invisibly. See agbe/translate/messages.py.
        """
        from agbe.translate.messages import get as get_messages

        if self.safe and not self.warnings:
            return ""

        msg = get_messages(language)
        lines = []
        if self.hazardous_found:
            lines.append(
                msg.hazardous_pesticide
                + " ("
                + ", ".join(sorted(self.hazardous_found))
                + ")"
            )
        if self.unsupported_dosages:
            lines.append(msg.unsupported_dosage)
        if self.stale_chemical_sources:
            lines.append(
                msg.stale_chemical
                + " ("
                + ", ".join(sorted(self.stale_chemical_sources))
                + ")"
            )
        if self.restricted_veterinary:
            lines.append(
                "Some sources mention veterinary medicines that are banned or "
                "restricted in food-producing animals ("
                + ", ".join(sorted(self.restricted_veterinary))
                + "). Do not use them. Ask a veterinary officer."
            )
        if self.missing_withdrawal_warning:
            lines.append(msg.withdrawal_period)
        return "\n\n".join(lines)


def contains_chemical_context(text: str) -> bool:
    return bool(_CHEMICAL_CONTEXT.search(text))


def find_hazardous_actives(text: str) -> list[str]:
    low = text.lower()
    return sorted(
        {
            active
            for active in HAZARDOUS_ACTIVES
            if re.search(r"\b" + re.escape(active) + r"\b", low)
        }
    )


def find_dosages(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _DOSAGE_PATTERNS:
        found.extend(match.group(0).strip() for match in pattern.finditer(text))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique = []
    for item in found:
        key = re.sub(r"\s+", " ", item.lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _normalise_for_containment(text: str) -> str:
    return re.sub(r"[\s,]+", "", text.lower())


def is_source_stale(year: str, now: int | None = None) -> bool:
    """True when a document is too old to support chemical recommendations."""
    try:
        published = int(str(year)[:4])
    except (TypeError, ValueError):
        # Unknown provenance is treated as stale. The alternative - assuming a
        # document is recent because we failed to record its year - fails in
        # the dangerous direction.
        return True
    current = now or datetime.now().year
    return (current - published) > CHEMICAL_RECENCY_YEARS


def check_answer(
    answer: str,
    source_texts: list[str],
    source_years: list[str] | None = None,
    now: int | None = None,
) -> SafetyVerdict:
    """Check a generated answer against the passages it was grounded in.

    `source_texts` are the retrieved passages. A dosage in the answer is
    acceptable only if it also appears in a source; anything else was invented
    by the model, however plausible it sounds.
    """
    verdict = SafetyVerdict(safe=True)
    combined_sources = "\n".join(source_texts)
    normalised_sources = _normalise_for_containment(combined_sources)

    hazardous = find_hazardous_actives(answer) or find_hazardous_actives(combined_sources)
    if hazardous:
        verdict.hazardous_found = hazardous
        verdict.safe = False

    for dosage in find_dosages(answer):
        if _normalise_for_containment(dosage) not in normalised_sources:
            verdict.unsupported_dosages.append(dosage)
    if verdict.unsupported_dosages:
        verdict.safe = False

    restricted_vet = sorted(
        {
            drug
            for drug in RESTRICTED_VETERINARY
            if re.search(r"\b" + re.escape(drug) + r"\b", answer.lower())
            or re.search(r"\b" + re.escape(drug) + r"\b", combined_sources.lower())
        }
    )
    if restricted_vet:
        verdict.restricted_veterinary = restricted_vet
        verdict.safe = False

    # A treatment named without a withdrawal period is unsafe for whoever
    # consumes the milk or meat - a person who never saw this advice.
    if VETERINARY_TREATMENT.search(answer) and not WITHDRAWAL_MENTIONED.search(answer):
        verdict.missing_withdrawal_warning = True
        verdict.safe = False

    if source_years and contains_chemical_context(answer):
        stale = [y for y in source_years if is_source_stale(y, now=now)]
        if stale and len(stale) == len(source_years):
            # Every supporting source is stale: the chemical guidance in this
            # answer has no current backing at all.
            verdict.stale_chemical_sources = sorted(set(stale))
            verdict.safe = False

    return verdict


def redact_unsupported_dosages(answer: str, verdict: SafetyVerdict) -> str:
    """Remove invented application rates from an answer.

    Redaction rather than regeneration: asking the model again costs seconds of
    CPU inference and offers no guarantee the second attempt is grounded.
    """
    out = answer
    for dosage in verdict.unsupported_dosages:
        out = out.replace(dosage, "[rate not stated in sources - ask an extension officer]")
    return out

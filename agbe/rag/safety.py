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

#: A question ASKING about withdrawal, whatever the answer turns out to say.
#:
#: WHY THE ANSWER-SIDE TRIGGER IS NOT ENOUGH
#: VETERINARY_TREATMENT keys on drug names in the ANSWER, which works when the
#: model names a drug. It missed the two most dangerous cases in a probe run,
#: because both answers avoided naming one:
#:
#:   "After treating my goat, when can I sell the milk?"
#:   -> "wait until it has produced milk... around 8-14 weeks after conception"
#:      Lactation onset, not drug withdrawal. A farmer following that sells
#:      contaminated milk.
#:
#:   "How long before I can eat a treated chicken?"
#:   -> "The CDC recommends waiting at least 24 hours after treatment"
#:      Fabricated, with an invented authority. Real withdrawal periods are
#:      drug-specific and run to days or weeks.
#:
#: Neither answer contained a drug name, so neither carried the warning. But the
#: QUESTION is unmistakable, and it is precisely the question where a wrong
#: answer causes harm to someone who never saw it - whoever drinks the milk.
#:
#: So the warning also fires on the question. The system cannot know which drug
#: was used and therefore cannot know the period; "ask a veterinary officer" is
#: not a hedge here, it is the only correct answer.
WITHDRAWAL_QUESTION = re.compile(
    r"\b(?:"
    r"(?:when|how long|how soon)[^?]{0,60}"
    r"(?:sell|drink|eat|consume|slaughter|market)[^?]{0,30}"
    r"(?:milk|meat|egg|bird|chicken|goat|cow|animal)|"
    r"(?:milk|meat|eggs?)[^?]{0,40}(?:after|following)[^?]{0,30}"
    r"(?:treat\w*|inject\w*|dos\w*|medicine|drug)|"
    r"(?:can|should|may) (?:i|we)[^?]{0,20}"
    r"(?:drink|eat|consume|sell)[^?]{0,30}(?:milk|meat|eggs?)|"
    r"(?:safe|alright|ok)[^?]{0,30}(?:to )?(?:drink|eat|consume|sell)"
    r"[^?]{0,30}(?:milk|meat|egg)|"
    r"(?:treated|medicated)[^?]{0,20}(?:animal|goat|cow|chicken|bird|milk|meat)"
    r")\b",
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
    advises_container_reuse: bool = False
    foreign_currency: list[str] = field(default_factory=list)

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
        if self.foreign_currency:
            lines.append(
                "Some figures above are in a currency from another country ("
                + ", ".join(sorted(self.foreign_currency))
                + "). They are not Nigerian prices and should not be used to "
                + "decide what to sell for. Ask at your local market."
            )
        if self.advises_container_reuse:
            lines.append(CONTAINER_REUSE_WARNING)
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


#: Money in a currency a Nigerian farmer does not sell in.
#:
#: Asked "what do traders pay for a bag of maize these days?", the system
#: answered "Traders generally pay farmers between KSh$81.21 and KSh$517 per
#: 50 kg bag... In Bungoma, farmers are willing to pay an average of KSh$87."
#: Those are Kenyan shillings from a CGIAR baseline study in Uasin Gishu and
#: Bungoma, presented without qualification to a farmer in southwest Nigeria.
#:
#: The scope rule now refuses that particular question, but the underlying
#: exposure is broader: AFRICA_TERMS admits Kenyan, Ugandan and Malawian
#: material on purpose - it is good agronomy - and any of it can carry local
#: money. Agronomy travels across borders; prices do not.
#:
#: A notice rather than a redaction. The figure may be the only quantitative
#: anchor in the answer, and deleting it silently would leave a sentence that
#: reads as though it said something. The farmer is told whose money it is.
_FOREIGN_MONEY = re.compile(
    r"(?<![A-Za-z])("
    # Currency SYMBOLS, not only codes. A long framed question about whether
    # to sell maize now produced "the market price for maize in your region is
    # around $0.40 per kilogram" - a fabricated price, in US dollars, claiming
    # regional knowledge. `USD` was in the list; the bare `$` was not, so
    # nothing fired. The naira symbol and "N45,000" are deliberately absent.
    r"[$£€](?=\s*\d)|"
    # Shilling codes vary by country and by writer: KSh, UShs, TShs, Ush, Tsh.
    # "UShs 426,377 per hectare" reached the interface because only UGX and
    # KSh were listed. The optional leading letter and plural cover the set.
    r"[KUT]?\.?Sh(?:s|illing)?\.?(?=\s*\d)|"
    r"KES|TZS|UGX|ZMW|MWK|GH\.?S|XOF|XAF|ZAR|USD|"
    r"(?:kenyan|tanzanian|ugandan|zambian|malawian|ghanaian|south african)\s+"
    r"(?:shillings?|kwacha|cedis?|rand)|"
    r"shillings?|kwacha|cedis?"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)

#: How far from the currency token a number may sit. "120 000 TZS" puts the
#: amount before the code; "KSh 87" puts it after. Requiring a number at all is
#: what keeps "Randomised plots" and "the brand of fertiliser" out - a currency
#: name with no figure beside it is prose, not a price.
_MONEY_WINDOW = 14


def find_foreign_money(text: str) -> list[str]:
    """Non-Naira currency tokens that appear next to a number."""
    found: set[str] = set()
    for m in _FOREIGN_MONEY.finditer(text):
        lo = max(0, m.start() - _MONEY_WINDOW)
        window = text[lo:m.end() + _MONEY_WINDOW]
        if any(ch.isdigit() for ch in window):
            found.add(m.group(1).strip())
    return sorted(found)

#: Any amount of money, in any currency.
#:
#: WHY THIS IS ABSOLUTE AND NOT A GROUNDING CHECK
#: The dosage guard above redacts a dose only when it is ABSENT from the
#: sources, because a dose that appears in an extension manual is a real
#: recommendation. Money is different in kind: this system runs offline against
#: a fixed corpus, so it cannot know what anything costs today. A figure the
#: model invented is wrong, and a figure it read out of a 2019 CGIAR baseline is
#: also wrong - it is a Kenyan shilling price from six years ago handed to a
#: Nigerian farmer as though it were current. Neither is usable, so containment
#: is not the test. Presence is.
#:
#: WHAT FORCED IT
#: A long framed question about whether to sell maize now produced "the market
#: price for maize in your region is around $0.40 per kilogram" - fabricated, in
#: dollars, and claiming regional knowledge. The scope rules never saw it: they
#: pattern-match short questions and this was a paragraph. An answer-side rule
#: does not care how the question was phrased, which is the point.
#: Every currency token this system may meet, symbol or code, in ONE place.
#:
#: This used to be three hand-maintained lists - symbols, codes before a
#: number, names after one - and they drifted apart exactly as three copies of
#: a list do. Codes were added to the leading list and not the trailing one, so
#: "KSh 100" was caught and "100 KSh" was not.
#:
#: ZMK is the specific lesson. The list carried ZMW, the CURRENT Zambian code;
#: the corpus was written when it was ZMK, so a real answer read "the local
#: market price per kilogram is ZMK 8" and the guard did not know the word. A
#: currency list covering only currencies still in use will always lag a corpus
#: of documents written in the past.
#: Symbols, with the country prefix writers put in front of them. "8.00 US$"
#: defeated both old branches at once: the leading one needs digits AFTER the
#: symbol, and the trailing one wanted the symbol immediately after the digits,
#: with no room for the "US".
CURRENCY_SYMBOL = r"(?:US|U\.S\.|AU|CA|NZ|HK|SG|NT)?[$£€₦]"

#: Codes. Africa first, current AND the ones the documents still use.
#:
#: ZMK is the lesson. The list carried ZMW, the CURRENT Zambian code; the corpus
#: was written when it was ZMK, so a real streamed answer read "the local market
#: price per kilogram is ZMK 8" and no guard knew the word. A currency list
#: covering only currencies still in use will always lag a corpus of documents
#: written in the past.
#:
#: MUST be used inside \b...\b. "[KUT]?\.?Shs?" matches a bare "sh", which
#: without boundaries fires inside "should", "sheep" and "harvesting".
CURRENCY_CODE = (
    r"(?:[KUT]?\.?Shs?"
    r"|KES|KSH|TZS|UGX|ZMW|ZMK|MWK|RWF|ETB|BIF|SOS|SDG|SSP|CDF|AOA"
    r"|GHS|GHC|XOF|XAF|GNF|GMD|LRD|SLL|SLE|CVE|STN|MZN|BWP|NAD|ZAR|LSL|SZL"
    r"|EGP|MAD|DZD|TND|LYD|MUR|SCR|MGA|KMF|DJF|ERN|NGN"
    r"|USD|EUR|GBP|CNY|JPY|INR|CHF|CAD|AUD)"
)

#: Spelled-out names, which only ever follow the number.
CURRENCY_WORD = (
    r"(?:naira|kobo|dollars?|cents?|pounds?|euros?|shillings?|kwacha|cedis?|"
    r"rand|francs?|birr|dalasis?|leones?|kwanzas?|meticals?|pulas?|dirhams?)"
)

#: Everything that marks an amount. ONE list, because there were four - three
#: in this file and a fourth in `advisor._MONEY_MARK` - and they drifted exactly
#: as four copies of a list do. Codes were added to some and not others, so
#: "KSh 100" was caught, "100 KSh" was not, and "ZMK 8" was caught by neither.
_CURRENCY = CURRENCY_SYMBOL + r"|" + CURRENCY_CODE

#: A name may carry a nationality adjective ("4,500 Kenyan shillings"); a code
#: may not.
_CURRENCY_NAME = (
    r"(?:kenyan|ghanaian|nigerian|ugandan|tanzanian|zambian|malawian|rwandan|"
    r"ethiopian|south african|american|us|west african|central african)?\s*"
    + CURRENCY_WORD
)
#: A number preceded OR followed by a currency token.
#:
#: Bare "N" is accepted only BEFORE a number. After one it is nitrogen - "apply
#: 60 N per hectare" is ordinary fertiliser advice, and redacting it would be
#: the price rule eating agronomy.
_MONEY = re.compile(
    # `(?!\w)` rather than `\b` to close the token. A word boundary after a
    # SYMBOL never fires - "8.00 US$ per kg" ends the match on "$", and "$"
    # followed by a space is two non-word characters with no boundary between
    # them - so that amount leaked while every code-ending one was caught.
    r"(?:"
    r"(?:" + _CURRENCY + r"|\bN)\s*\d[\d,.]*|"
    r"\d[\d,.]*\s*(?:" + _CURRENCY + r")(?!\w)|"
    r"\d[\d,.]*\s*(?:" + _CURRENCY_NAME + r")\b"
    r")",
    re.IGNORECASE,
)

#: What replaces a redacted amount. Says what the system cannot do, rather than
#: leaving a gap the reader fills in themselves.
MONEY_REDACTION = "[this system has no price data - ask at your local market]"


def find_money(text: str) -> list[str]:
    """Monetary amounts appearing in `text`, in any currency."""
    return [m.group(0) for m in _MONEY.finditer(text)]


def redact_money(text: str) -> tuple[str, list[str]]:
    """Replace every monetary amount. Returns `(cleaned, what_was_removed)`."""
    found = find_money(text)
    if not found:
        return text, []
    out = text
    for amount in found:
        out = out.replace(amount, MONEY_REDACTION)
    return out, found

#: Practices that are dangerous however confidently they are phrased.
#:
#: WHY THIS EXISTS
#: Asked "can I reuse a pesticide container for water?" the system answered:
#:
#:     "No, reuse is not recommended. Use empty pesticide containers for other
#:      uses. [1]"
#:
#: The first sentence is right and the second contradicts it. Reproduced in one
#: run out of two. The source says not to reuse them; the model dropped the
#: negation and emitted the instruction on its own.
#:
#: Container reuse is a documented cause of poisoning in exactly this setting -
#: an empty drum becomes a water container, and the residue goes into whoever
#: drinks from it. That harm lands on a household, not on the farmer who asked.
#:
#: A negation-dropping error is not something the containment guard can see: the
#: words ARE in the sources. Only the polarity is wrong, and polarity is what
#: F-23 already showed this model getting wrong on closed questions.
_REUSE_CONTAINER = re.compile(
    r"\b(?:re-?use|reusing|use|using|keep|store|fill)\w*\s+"
    r"(?:the\s+|an?\s+|empty\s+|old\s+){0,3}"
    r"(?:pesticide|chemical|agrochemical|herbicide|insecticide)\s+"
    r"(?:container|containers|drum|drums|can|cans|bottle|bottles|jerr?y ?can)"
    r"|"
    # Words may sit between the container and its intended new use: "empty
    # herbicide bottles can be USED TO HOLD drinking water".
    r"\b(?:container|containers|drum|drums|bottle|bottles|jerr?y ?can)s?"
    r"[^.!?]{0,40}?\b(?:hold|holding|store|storing|carry|carrying|keep|"
    r"keeping|fetch|fetching|for|to)\s+(?:\w+\s+){0,2}(?:water|drinking|food|grain|milk|palm ?oil)",
    re.IGNORECASE,
)

#: A negation anywhere in the same sentence makes it correct advice, not a
#: dangerous instruction. "Do NOT reuse pesticide containers" must pass.
_NEGATED = re.compile(
    r"\b(?:not|never|don'?t|do not|avoid|should not|must not|no)\b",
    re.IGNORECASE,
)

CONTAINER_REUSE_WARNING = (
    "NEVER reuse a pesticide container for water, food or drink. Residue stays "
    "in the container and has poisoned whole households. Puncture empty "
    "containers and bury them away from water, or return them to the dealer."
)


def advises_container_reuse(answer: str) -> bool:
    """Whether any SENTENCE tells the reader to reuse a chemical container.

    Checked sentence by sentence, because the failure was a correct sentence
    followed by a contradicting one - looking at the answer as a whole would
    find the negation in the first sentence and clear the second.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if _REUSE_CONTAINER.search(sentence) and not _NEGATED.search(sentence):
            return True
    return False

def check_answer(
    answer: str,
    source_texts: list[str],
    question: str = "",
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

    foreign = find_foreign_money(answer)
    if foreign:
        verdict.foreign_currency = foreign
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
    if advises_container_reuse(answer):
        verdict.advises_container_reuse = True
        verdict.safe = False

    asks_about_withdrawal = bool(question) and bool(
        WITHDRAWAL_QUESTION.search(question)
    )
    names_a_treatment = bool(VETERINARY_TREATMENT.search(answer))
    if (names_a_treatment or asks_about_withdrawal) and not WITHDRAWAL_MENTIONED.search(
        answer
    ):
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

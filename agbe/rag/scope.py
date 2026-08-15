"""Policy scope guards: questions we decline on principle, not on similarity.

WHY A SEPARATE MECHANISM FROM min_score
---------------------------------------
The similarity floor answers "does the corpus contain anything about this?".
It is the right instrument for a question like "how do I fix my motorcycle
engine", which scores 0.41 and is correctly refused.

It is the wrong instrument for two failures found on the held-out evaluation:

    "How many millilitres of antibiotic should I inject into my goat?"   0.75
    "What is killing my cocoa pods and turning them black?"              0.72

Retrieval is behaving correctly in both cases. The corpus genuinely holds
livestock material and genuinely holds crop-disease material, so similarity is
legitimately high. These questions are out of *policy* scope, not out of topic,
and no similarity threshold can separate them - raising the floor far enough to
catch them would refuse most legitimate questions too.

So scope is enforced by explicit rules, stated once, testable, and visible in
the answer the farmer receives.

WHAT IS REFUSED AND WHY
-----------------------
DOSAGE      A dose depends on the product, its concentration, the animal's
            weight and the local registration. None of that is in the corpus,
            and Nigeria's registration data (NAFDAC) is not openly published.
            The safety layer already redacts invented doses from answers; this
            catches the question earlier and explains rather than deletes.

OUT-OF-SCOPE CROP
            Cocoa, rice, cowpea and the rest are excluded from the declared
            scope. The corpus contains crop-disease text that will happily
            retrieve for them, producing an answer that looks grounded and is
            about the wrong plant.

HUMAN MEDICAL
            Pesticide exposure symptoms are a medical emergency. The correct
            response is a clinic, not an advisory app.

Each rule returns a reason, and the reason is shown. "I cannot answer that" is
far less useful to a farmer than "I cannot tell you a dose, because it depends
on the product - ask your extension officer".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Crops inside the declared scope. Everything else is refused even though the
#: corpus may contain material about it.
#:
#: Widened after auditing what the corpus actually covers. The original four
#: were chosen from regional reasoning about southwest Nigeria, but the bulk
#: source (Infonet) reflects its Kenyan origin, so the declared scope and the
#: available material did not line up: yam, an in-scope crop, had 18 pages -
#: thinner than thirteen crops we were refusing. Rice (41), groundnut (34),
#: pepper (23) and cowpea (19) are Nigerian smallholder staples that were
#: already in the corpus and being turned away.
#:
#: Each addition costs evaluation questions, not architecture: retrieval always
#: takes a fixed top_k, so corpus breadth is free at query time in a way model
#: size is not.
IN_SCOPE_CROPS = {
    "cassava", "maize", "yam", "tomato",
    "rice", "cowpea", "groundnut", "pepper", "okra",
}

#: Still refused. Cocoa is the clearest case: 8 pages in the corpus, the
#: thinnest coverage of any crop measured, so answering about it would mean
#: generating from material that is not really there.
OUT_OF_SCOPE_CROPS = {
    "cocoa": "cocoa", "cacao": "cocoa",
    "sorghum": "sorghum", "millet": "millet", "cotton": "cotton",
    "oil palm": "oil palm", "plantain": "plantain", "banana": "banana",
    "sugarcane": "sugarcane", "coffee": "coffee", "cashew": "cashew",
    "soya": "soyabean", "soybean": "soyabean", "sweet potato": "sweet potato",
    "cocoyam": "cocoyam", "wheat": "wheat", "sesame": "sesame",
}

#: Asking *how much* of a chemical or medicine to use. Deliberately broad on
#: the quantity side and narrow on the substance side, so "how much water" or
#: "how much space" are not caught.
_DOSAGE_QUESTION = re.compile(
    r"\b(?:how (?:much|many)|what (?:dose|dosage|rate|amount|quantity|"
    r"concentration)|exactly how|how many (?:ml|millilitres|milliliters|"
    r"grams?|litres?|liters?|cc|spoons?|caps?))\b",
    re.IGNORECASE,
)

#: NAMED ACTIVES, not just categories.
#:
#: The category list below catches "how much antibiotic" and misses "how many ml
#: of ivermectin does a 200 kg cow need?" - a farmer who knows the product name
#: asks for it by name, which is the more likely phrasing, not the less. That
#: question reached retrieval, cleared the lexical floor tolerance at 0.65 and
#: was answered, until an evaluation question was written for it.
#:
#: These are the actives a smallholder in this region is most likely to name.
#: The list is not exhaustive and cannot be - which is why the category terms
#: stay as the backstop, and why the dosage refusal message tells the farmer to
#: read the label and ask an extension officer rather than implying we know
#: every product.
_NAMED_ACTIVES = (
    r"ivermectin|albendazole|levamisole|oxytetracycline|tetracycline|"
    r"penicillin|amprolium|tylosin|closantel|praziquantel|"
    r"glyphosate|paraquat|atrazine|mancozeb|chlorpyrifos|cypermethrin|"
    r"lambda[- ]?cyhalothrin|imidacloprid|carbofuran|metalaxyl|"
    r"emamectin|abamectin|deltamethrin|dimethoate|profenofos"
)

_SUBSTANCE = re.compile(
    r"\b(?:pesticide|insecticide|fungicide|herbicide|acaricide|nematicide|"
    r"chemical|spray|antibiotic|antibiotics|drug|medicine|dewormer|"
    r"anthelmintic|vaccine|injection|inject|dose|dosage|fertiliser|fertilizer|"
    r"npk|urea|" + _NAMED_ACTIVES + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Live data
# ---------------------------------------------------------------------------
# Questions needing a fact that changes daily. No static corpus holds them, and
# the answer would be specific, confident and wrong.
#
# These only became a problem once the corpus grew. Before it held real market
# and weather material, "what is the price of maize today?" scored below the
# similarity floor and was refused for free. With 214 documents it scores 0.69
# and retrieves plausible-looking passages about grain marketing - so better
# coverage bought worse refusal, and the floor can no longer separate them.
#
# The distinction is between judgement and fact. "Should I sell now or store?"
# is judgement, is answerable from extension material, and must keep working.
# "What is it selling for today?" is a fact we do not have. So the patterns
# below key on the markers of a live lookup - today, current, right now - and
# not on the topic.
_LIVE_PRICE = re.compile(
    r"\b(?:what(?:'s| is) the (?:current |today'?s? )?price|"
    r"how much (?:is|does|are|do)[^?]*(?:cost|sell|selling|go(?:ing)? for)|"
    r"price (?:of|for)[^?]*(?:today|now|currently|this week)|"
    r"(?:current|today'?s?|latest|market) price|"
    r"selling (?:for|at) (?:today|now)|"
    r"what (?:are|is) [^?]*going for)\b",
    re.IGNORECASE,
)

#: Asking us to predict the weather, in any phrasing.
#:
#: This used to enumerate ways of asking whether it WILL rain, and missed the
#: ways of asking whether it will STOP. "Is the rain going to stop before I
#: harvest next month?" scored 0.67, cleared the lexical floor tolerance and was
#: answered - a farmer planning a harvest asks about the end of the rains at
#: least as often as their start.
#:
#: The shape is now "prediction verb + weather noun + change verb" rather than a
#: list of remembered sentences, which is what the category actually is.
_LIVE_FORECAST = re.compile(
    r"\b(?:"
    r"(?:will|is|are|going to|gonna)[^?]{0,40}"
    r"(?:rain|rains|raining|dry spell|harmattan|flood|drought)"
    r"[^?]{0,40}(?:stop|start|come|end|continue|begin)|"
    r"will it rain|is it going to rain|"
    r"when will (?:it|the rains?) (?:rain|come|start|stop|end)|"
    r"(?:weather )?forecast|"
    r"rain (?:next week|tomorrow|this week|today|next month)|"
    r"(?:next week|tomorrow|next month)[^?]*rain"
    r")\b",
    re.IGNORECASE,
)

_HUMAN_MEDICAL = re.compile(
    r"\b(?:i feel|i am feeling|my head|my eyes|my skin|dizzy|vomit|nausea|"
    r"headache|breathing|poisoned|swallowed|what tablet|what drug should i|"
    r"my child|my wife|my husband)\b",
    re.IGNORECASE,
)


#: Financial, legal and administrative services. Outside all four advisory
#: areas this system covers - crop, livestock, weather and market guidance -
#: and outside what an agronomic corpus can answer.
#:
#: WHY THIS RULE EXISTS AND WHY IT DID NOT BEFORE
#: ----------------------------------------------
#: The corpus legitimately contains CGIAR and IFPRI material on rural credit,
#: land tenure and farmer organisations, because those documents also carry
#: agronomy. Nothing declined these questions on policy; they were refused only
#: because no chunk happened to score above the similarity floor.
#:
#: That was never a guarantee, and it stopped being true. After re-chunking on
#: section headings the corpus went from 31,682 to 43,177 smaller, more
#: precisely labelled chunks, and passages headed "Access to credit" and "Land
#: tenure" began clearing the floor. Test-split refusal fell from 100% to 66.7%
#: - "Which bank gives the best loan to farmers in Oyo State?" and "How do I
#: register my farmland title?" were both answered, from policy papers, about
#: someone's money and someone's land.
#:
#: A threshold was masking a missing rule. Retrieval quality improved and the
#: mask came off. The rule states the reason instead.
#: Word boundaries are per-alternative, not global. A trailing \b on the whole
#: group silently broke every prefix in it - "credit facilit" would not match
#: "credit facility", and "register my farm" would not match "register my
#: farmland", because the next character is a letter rather than a boundary.
#: Both target questions slipped through the first version for that reason.
#:
#: The boundaries that remain are load-bearing in the other direction: `tax\b`
#: is what stops this rule firing on "taxonomy", which appears throughout
#: agronomic text.
_FINANCIAL_LEGAL = re.compile(
    r"\b(?:"
    r"banks?\b|loans?\b|microfinance\b|collateral\b|mortgages?\b|"
    r"credit facilit\w*|interest rates?\b|insurance premium\w*|"
    r"land titles?\b|title deeds?\b|land registr\w*|farmland title\w*|"
    r"deed of assignment\b|certificate of occupancy\b|"
    r"register (?:my |the )?(?:farm|land)\w*|"
    r"taxe?s?\b|levy\b|levies\b|"
    r"subsidy applicat\w*|grant applicat\w*|"
    r"cooperative registrat\w*|business registrat\w*|licence applicat\w*"
    r")",
    re.IGNORECASE,
)

FINANCIAL_LEGAL_MESSAGE = (
    "I cannot advise on loans, land titles or registration. My documents are "
    "agricultural guidance - crops, livestock, weather and markets - and "
    "answering from them would mean guessing about your money or your land. "
    "Please ask your bank, your local land registry, or your extension officer, "
    "who can refer you."
)


#: Mechanical repair. Two patterns, and BOTH must match.
#:
#: The domain is crops, livestock, weather and markets. Repairing a machine is
#: none of them, and the corpus holds no repair manuals - so an answer would be
#: invented. That applies whether or not the machine is agricultural: a water
#: pump is farm equipment and fixing its engine is still not agronomy.
#:
#: WHY TWO PATTERNS AND NOT ONE
#: ----------------------------
#: Equipment alone must not trigger this. Sprayer calibration, knapsack
#: maintenance and equipment storage are genuinely in extension literature and
#: genuinely in scope - "how do I calibrate my sprayer" is a question this
#: system should answer. What is out of scope is a MALFUNCTION: a thing that
#: will not start, run or work.
#:
#: So a machine noun and a fault verb are both required, which is the narrowest
#: rule that still catches the case.
#:
#: WHY A RULE RATHER THAN THE FLOOR
#: --------------------------------
#: Both evaluation questions in this category were refused only because nothing
#: in an agriculture corpus resembles engine repair. That is luck, not policy,
#: and it is the same shape as the two failures in 6.8 - a threshold masking a
#: missing rule, holding until the corpus changed and then not.
_MACHINE = re.compile(
    r"\b(?:engine|motor|motorcycle|motorbike|okada|generator|genset|pump|"
    r"tractor|tiller|mill|grinder|machine|machinery|gearbox|carburettor|"
    r"carburetor|battery|alternator|spark plug)\b",
    re.IGNORECASE,
)

_MECHANICAL_FAULT = re.compile(
    r"\b(?:fix|repair|repairing|mend|service|servicing|overhaul|"
    r"(?:will|would|does|do|is|are)\s*n[o']?t\s*(?:start|starting|run|running|"
    r"work|working|turn|move)|broken|broke down|breaking down|stopped working|"
    r"not starting|won'?t start|fault|faulty|leaking oil|smoking)\b",
    re.IGNORECASE,
)

MECHANICAL_MESSAGE = (
    "I cannot help with repairing machinery. My documents are agricultural "
    "guidance - crops, livestock, weather and markets - and they contain no "
    "repair instructions, so anything I said about your engine would be "
    "guesswork. Please ask a mechanic. If your question is about using or "
    "calibrating farm equipment rather than repairing it, ask me again that way."
)


@dataclass
class ScopeVerdict:
    in_scope: bool
    reason: str = ""
    message: str = ""


DOSAGE_MESSAGE = (
    "I cannot tell you how much to use. The correct amount depends on the "
    "specific product, its concentration, and what is registered for your crop "
    "or animal - none of which is in my documents. Read the product label, or "
    "ask your extension officer or veterinary officer. Guessing a dose can harm "
    "your crop, your animals, or you."
)

HUMAN_MEDICAL_MESSAGE = (
    "This sounds like it may affect your health, and I only give advice about "
    "crops and livestock. If you feel unwell after handling farm chemicals, "
    "stop work, wash thoroughly, and go to a clinic or hospital now. Take the "
    "product container with you so they know what you were exposed to."
)


def crops_mentioned(text: str) -> set[str]:
    """In-scope crops named in `text`.

    Used by retrieval to tell three cases apart, which matters more than it
    sounds: a passage about *another* crop, a passage about *this* crop, and a
    passage about *no particular* crop. The third group - whitefly biology,
    general disease principles, storage practice - is cross-cutting and must
    not be treated as off-topic. See VectorIndex.hybrid_search.
    """
    low = text.lower()
    return {c for c in IN_SCOPE_CROPS if re.search(rf"\b{re.escape(c)}s?\b", low)}


def out_of_scope_crop(question: str) -> str | None:
    low = question.lower()
    if any(crop in low for crop in IN_SCOPE_CROPS):
        return None
    for term, name in OUT_OF_SCOPE_CROPS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            return name
    return None


LIVE_PRICE_MESSAGE = (
    "I cannot tell you today's price - my documents are guides, not a market "
    "report, and prices change every day. I can help you decide WHEN to sell, "
    "how to store your crop so it keeps its value, and how grading affects "
    "what you are offered. For today's price, ask at the market or call a "
    "trader you trust."
)

LIVE_FORECAST_MESSAGE = (
    "I cannot tell you what the weather will do - I have no forecast, only "
    "farming guides. I can help you decide when the rains have truly "
    "established, what to do if a dry spell comes, and how to prepare a field "
    "that floods. For a forecast, listen to the radio or check with your "
    "extension officer."
)


def check(question: str) -> ScopeVerdict:
    """Decide whether a question is answerable as a matter of policy.

    Runs before retrieval: there is no point spending prefill on a question we
    will decline regardless of what comes back.
    """
    if _HUMAN_MEDICAL.search(question):
        return ScopeVerdict(False, "human medical", HUMAN_MEDICAL_MESSAGE)

    if _DOSAGE_QUESTION.search(question) and _SUBSTANCE.search(question):
        return ScopeVerdict(False, "dosage", DOSAGE_MESSAGE)

    if _LIVE_PRICE.search(question):
        return ScopeVerdict(False, "live price", LIVE_PRICE_MESSAGE)

    if _LIVE_FORECAST.search(question):
        return ScopeVerdict(False, "live forecast", LIVE_FORECAST_MESSAGE)

    if _FINANCIAL_LEGAL.search(question):
        return ScopeVerdict(False, "financial or legal", FINANCIAL_LEGAL_MESSAGE)

    if _MACHINE.search(question) and _MECHANICAL_FAULT.search(question):
        return ScopeVerdict(False, "mechanical repair", MECHANICAL_MESSAGE)

    crop = out_of_scope_crop(question)
    if crop:
        return ScopeVerdict(
            False,
            f"out-of-scope crop: {crop}",
            f"I do not cover {crop}. My documents are about cassava, maize, yam "
            "and tomato, and about livestock. Answering from material about a "
            "different crop would risk giving you the wrong advice. Please ask "
            "your extension officer about your " + crop + ".",
        )

    return ScopeVerdict(True)

"""The dosage guard must cover the shape a veterinary dose is actually written in.

WHY THIS FILE EXISTS
--------------------
`REPORT.md` promises that "How many ml of antibiotic for my goat?" is refused,
and the scope rule does refuse the QUESTION. This is the backstop: the guard
that redacts a dose appearing in an ANSWER to a question that was not
dosage-shaped. It was measured and it did not hold.

Streamed through the interface, four of five ordinary dose sentences passed
through untouched:

    "Give 5 ml of ivermectin per 50 kg body weight"     -> not found
    "Inject 2.5 mg/kg of oxytetracycline once daily"    -> not found
    "Apply 250 ml of the chemical per hectare"          -> not found
    "Drench with 10 ml per animal"                      -> not found
    "Use 1.5 litres per hectare"                        -> found

The cause was not the token boundary this was suspected of. Every rate pattern
required a per-AREA unit - hectare, acre, m2 - so the guard covered crop spray
rates and left veterinary dosing, the higher-harm half, entirely open. `mg` was
not a unit it knew. Neither were tablets, sachets, drops or cc.

THREE SHAPES, NOT ONE
---------------------
    rate      "2 litres per hectare", "5 ml per 50 kg body weight"
    ratio     "2.5 mg/kg", "1 ml/10 kg"
    BARE      "Give 5 ml of ivermectin."   <- no "per" anywhere

The bare shape is the dangerous one and the hardest to match safely, because a
number and a unit on their own are also how ordinary advice is written. It is
keyed on an administration VERB near the quantity, and refuses to fire when the
unit is followed by a noun that makes the quantity descriptive - "a 20 kg goat",
"a 50 kg bag".

WHAT REDACTION ACTUALLY DEPENDS ON
----------------------------------
Matching here does not mean redacting. `find_dosages` proposes; containment
against the retrieved sources disposes. A dose that IS in the sources is kept.
That is what makes it safe to match generously - but only if the match is the
quantity and not the sentence around it. Capturing the whole span made
containment compare "Use 500 ml" against a source reading "500 ml / ha", so a
correctly grounded rate was reported as invented. The quantity is captured on
its own for exactly that reason.
"""

from __future__ import annotations

import pytest

from agbe.rag.safety import check_answer, find_dosages

#: Doses. Every one of these must be visible to the guard.
DOSES = [
    # rate, per area
    "Apply 250 ml of the chemical per hectare of maize.",
    "Use 1.5 litres per hectare mixed in 200 litres of water.",
    "Spray at 400 ml/ha.",
    "Apply 2 kg per acre before flowering.",
    # rate, per body weight - the shape that was entirely invisible
    "Give 5 ml of ivermectin per 50 kg body weight to the goat.",
    "Dose at 7.5 mg per kg body weight.",
    "Administer 1 ml per 10 kg.",
    # rate, per animal
    "Drench with 10 ml per animal, repeated after 14 days.",
    "Treat with 1 bolus per animal.",
    "Give 2 tablets per bird.",
    # ratio
    "Inject 2.5 mg/kg of oxytetracycline once daily.",
    "The rate is 1 ml/10 kg.",
    # in a volume of water
    "Mix 20 ml in 20 litres of water.",
    "Add 40 g to 15 litres of water.",
    # bare - no "per" at all
    "Give 2 tablets to each bird.",
    "Give 3 drops in the eye.",
    "Inject 5 cc into the muscle.",
    "Give 1 teaspoon twice a day.",
    "Apply 1 sachet per knapsack.",
    "Use 2 caps per drum.",
    "Give 250 ml of cooking oil.",
    # dilution and concentration
    "Dilute at a ratio of 1:200.",
    "Use a 0.5% solution.",
]


@pytest.mark.parametrize("sentence", DOSES, ids=lambda s: s[:44])
def test_a_dose_is_found(sentence: str) -> None:
    assert find_dosages(sentence), f"dose invisible to the guard: {sentence!r}"


#: Quantities that are NOT doses. Redacting any of these would replace correct
#: advice with "[rate not given in the sources]", which is worse than the leak:
#: it destroys something true rather than withholding something invented.
NOT_DOSES = [
    "A 20 kg goat needs more feed.",
    "Store the grain in a 50 kg bag.",
    "Give the 20 kg goat clean water.",
    "Dry to 13% moisture content.",
    "Plant 25 cm apart.",
    "Space rows 75 cm apart.",
    "Use 6 bags of fertiliser.",
    "Use 6 bags of fertiliser per hectare.",
    "Harvest after 9 months.",
    "The pH should be 6.0 to 7.0.",
    "Allow 35 cm of feeding space per goat.",
    "Wait 18 days before slaughter.",
    "Deworm every 3 months.",
    "Sell 1670 kilograms of maize.",
    "Sell 2 tonnes of maize.",
    "Use clean planting material.",
    "Add compost to the soil.",
    "Use a 20 litre bucket to carry water.",
    "Apply mulch around the plants.",
    "Give the birds clean water daily.",
    "Separate the sick goat from the herd.",
    "Plant 3 seeds per hole.",
]


@pytest.mark.parametrize("sentence", NOT_DOSES, ids=lambda s: s[:44])
def test_ordinary_advice_is_not_a_dose(sentence: str) -> None:
    assert find_dosages(sentence) == [], sentence


#: A dose that IS in the sources must survive, whatever the whitespace.
#:
#: This is the half a generous matcher endangers, and the reason the quantity is
#: captured without the words around it.
GROUNDED = [
    ("Use 500 ml/ha.", "The label rate is 500 ml / ha for this crop."),
    ("Apply mancozeb at 2 kg per hectare.", "Apply mancozeb at 2 kg per hectare."),
    ("Give 250 ml of cooking oil.", "Give 250 ml of cooking oil and walk the animal."),
    ("Mix 20 ml in 20 litres of water.", "Mix 20 ml in 20 litres of water."),
]


@pytest.mark.parametrize("answer,source", GROUNDED, ids=lambda x: str(x)[:40])
def test_a_grounded_dose_is_not_reported_as_invented(answer: str, source: str) -> None:
    verdict = check_answer(answer, source_texts=[source])
    assert verdict.unsupported_dosages == [], verdict.unsupported_dosages


#: The same doses with sources that do NOT carry them. Every one must be caught.
UNGROUNDED = [
    ("Give 5 ml of ivermectin to the goat.", "Ivermectin treats worms in goats."),
    ("Inject 2.5 mg/kg of oxytetracycline.", "Oxytetracycline treats infections."),
    ("Apply 250 ml per hectare.", "Apply the fungicide when symptoms appear."),
    ("Drench with 10 ml per animal.", "Drench the animals at the start of the rains."),
]


@pytest.mark.parametrize("answer,source", UNGROUNDED, ids=lambda x: str(x)[:40])
def test_an_invented_dose_is_caught(answer: str, source: str) -> None:
    verdict = check_answer(answer, source_texts=[source])
    assert verdict.unsupported_dosages, f"invented dose passed: {answer!r}"

"""No guard may fire on ordinary farming language.

WHY THIS FILE EXISTS
--------------------
The same defect has appeared three times, in three different rules, written on
three different days:

    `\\bban\\b`   without a leading boundary matched "BANana"
    `his`        without a leading boundary matched "tHIS"
    `"e be"`     tested with `in` matched "thE BEst"

Each was found by hand, patched, and the lesson was not carried across - the
guard against it existed as a one-off assertion in one test file while the next
rule was written with the same hole. The third instance was found by an audit
script that was then deleted as a temporary file.

So this is that audit, made permanent. It fires ordinary farming sentences at
every rule that can REFUSE or ALTER an answer, and fails if any of them react.
The sentences are chosen to contain the substrings that bite: this, these,
their, weather, other, harvest, banana, plantain, mother, rather, gather.

WHAT THIS DOES NOT COVER
------------------------
It cannot prove a pattern is correct, only that it is not obviously greedy. A
rule that fails to fire when it should is invisible here; that is what the probe
sets and the per-rule tests are for.
"""

from __future__ import annotations

import pytest

from agbe.advisor import strip_citation_dump, strip_symptom_echo
from agbe.rag import scope
from agbe.rag.safety import find_dosages, find_money
from agbe.translate.detect import detect

#: Ordinary in-scope questions. None may be refused by any rule.
#:
#: Every entry carries at least one substring that has broken a guard before, or
#: is the kind of word that will: "this", "their", "weather", "other",
#: "harvest", "banana", "rather", "mother", "gather", "these".
IN_SCOPE = [
    "My maize is lodging after heavy rain",
    "Is there rain damage on my maize?",
    "Is the soil too wet to plant after rain?",
    "The other farmer plants banana beside his cassava, is that a problem?",
    "My goats damaged their pen, how do I fix it?",
    "These seedlings need more manure",
    "The mother hen is sitting on her eggs",
    "Rather than spray, can I rogue the infected plants?",
    "Weather permitting, when should I harvest these yams?",
    "My neighbour helped me weed this plot",
    "Gather the harvest and store it in the barn - is that right?",
    "How do I kill ticks on my goats?",
    "Destroy this infected plant and burn the residue - is that right?",
    "Kill the weeds in this field before planting",
    "How much space does each goat need?",
    "How much water should I give my chickens?",
    "What is the stocking rate for goats?",
    "What is the seeding rate for maize?",
    "How do I grade my tomatoes before selling?",
    "Should I sell my maize now or store it?",
    "Is it better to sell cassava as chips or fresh roots?",
    "My cow swallowed a plastic bag",
    "My sheep is coughing and breathing fast",
    "My goats look poisoned after grazing",
    "How do I decide between organic and chemical control?",
    "Should I use organic methods?",
    "What is the best way to store yam?",
    "The best time to plant is early, is that so?",
    "How do I know the rains have truly started?",
    "When do the rains usually start?",
    "What can I plant if the rains come late?",
    "How do I make compost from farm waste?",
    "Should I intercrop maize with cowpea?",
    "How do I know if my goat is pregnant?",
    # The in-domain gate refused all of these. Named pests and weeds are the
    # corpus's subject, and safety equipment is the topic a refusal is least
    # excusable on.
    "How do I identify striga early?",
    "What protective clothing do I need?",
    "How do I dispose of an empty pesticide container?",
    "How do I control armyworm in maize?",
    "My cassava has mealybug",
    "How do I calibrate my knapsack sprayer?",
    "How do I store my sprayer after spraying?",
    "How do I clean my sprayer nozzle?",
    "What records should I keep for my farm?",
    # Refused as OUT OF DOMAIN by the gate before a sweep widened it.
    "What is the withdrawal period?",
    "How do I dispose of empty containers?",
    "How do I protect myself when spraying?",
    "When do the rains normally start here?",
    "What causes empty maize cobs?",
    "How do I stop birds eating my rice?",
]


@pytest.mark.parametrize("question", IN_SCOPE, ids=lambda q: q[:40])
def test_no_rule_refuses_an_ordinary_question(question: str) -> None:
    verdict = scope.check(question)
    assert verdict.in_scope, f"refused as {verdict.reason!r}: {question!r}"


#: Quantities a farmer is given all the time. None is money, none is a dose
#: that should be redacted for being money.
QUANTITIES = [
    "Apply 250 kg per hectare",
    "dry to 13% moisture content",
    "wait 18 days before slaughter",
    "plant 25 cm apart",
    "use 6 bags of fertiliser",
    "2 tonnes of maize",
    "give 250 ml of cooking oil",
    "a 20 kg goat",
    "20mm of rain over four consecutive days",
    "deworm every 3 months",
    "the pH should be 6.0 to 7.0",
    "at least 35 cm of feeding space per goat",
]


@pytest.mark.parametrize("text", QUANTITIES, ids=lambda t: t[:30])
def test_money_redaction_leaves_quantities_alone(text: str) -> None:
    assert find_money(text) == [], text


#: A diagnosis restates the symptom by design. The echo guard must not eat it.
ECHO_SAFE = [
    ("My maize is yellow", "Use nitrogen fertiliser on maize that is yellow."),
    ("My cassava leaves are yellow and twisted",
     "Your cassava leaves are yellow and twisted because of mosaic disease."),
    ("My chickens have twisted necks",
     "Newcastle disease causes twisted necks. Separate affected birds."),
]


@pytest.mark.parametrize("question,answer", ECHO_SAFE, ids=lambda x: str(x)[:30])
def test_symptom_echo_never_truncates_to_nothing(question: str, answer: str) -> None:
    """Removal is allowed; returning less than the farmer needs is not."""
    cleaned, _removed = strip_symptom_echo(answer, question)
    assert cleaned.strip(), f"emptied {answer!r}"


#: Real answers, none of which opens with a run of citation markers.
CITATION_SAFE = [
    "[1] suggests that the swelling is likely an infection.",
    "Mastitis is likely. Isolate the goat. [1] [2]",
    "Use clean planting material. [3]",
    "Remove infected plants and destroy them away from the field.",
    "Apply [2] bags per hectare as the sources describe.",
]


@pytest.mark.parametrize("answer", CITATION_SAFE, ids=lambda a: a[:30])
def test_citation_guard_leaves_real_answers_alone(answer: str) -> None:
    cleaned, only_citations = strip_citation_dump(answer)
    assert only_citations is False
    assert cleaned == answer


#: English that contains a Pidgin marker as a SUBSTRING. Detecting any of
#: these as Pidgin means an English speaker is shown Pidgin messages, which is
#: worse than the reverse.
#:
#: "the best" contains "e be". "gather" contains "the". "banana" contains
#: "ban". This list exists because the phrase matcher was written with `in`
#: rather than word boundaries and called "what is the best way to store yam?"
#: Pidgin - the third appearance of that bug, and the one this file failed to
#: catch on its first version.
ENGLISH_SUBSTRING_TRAPS = [
    "What is the best way to store yam?",
    "The best time to plant is early",
    "These beans are the best I have grown",
    "Gather the harvest before the rain",
    "I planted banana beside the cassava",
    "The mother hen is sitting on her eggs",
    "Rather than spray, I rogue the plants",
    "My goats damaged their pen",
    "Is the weather going to change?",
    "How do I make compost from farm waste?",
    "Which container is best for storing cowpea?",
    "I do not know what to do",
]


@pytest.mark.parametrize("sentence", ENGLISH_SUBSTRING_TRAPS, ids=lambda s: s[:34])
def test_english_is_never_detected_as_pidgin(sentence: str) -> None:
    assert detect(sentence) == "en", sentence

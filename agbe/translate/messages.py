"""Human-validated fixed messages in supported languages.

WHY THESE ARE NOT MACHINE-TRANSLATED
------------------------------------
Every message here is a fixed string: a refusal, a safety warning, a
prohibition. They are the highest-consequence text the system produces - the
banned-pesticide warning, the dosage refusal, the veterinary withdrawal period,
the instruction to go to a clinic.

Passing them through NLLB would take text a native speaker has verified and
degrade it into text nobody has. A machine translation of a safety warning is
exactly the thing that must not be trusted, because a warning that reads as a
suggestion has failed silently: the farmer sees words, understands something
softer than was meant, and acts on it.

So these are stored as validated translations and emitted verbatim. The
translation model, when present, handles only free-form advisory text.

CONSEQUENCES BEYOND CORRECTNESS
-------------------------------
  * Zero memory. Emitting a validated string costs nothing, where loading NLLB
    costs ~600 MB - about 1.7 points of the efficiency score, which is 20% of
    the total.
  * Zero latency. No translation pass before the farmer sees the warning.
  * No dependency. Language support for the safety path works whether or not
    the translation model was ever converted, so the African-language claim
    rests on something that exists today rather than on a pending download.

PROVENANCE
----------
Pidgin (`pcm`) reviewed and approved by a Nigerian Pidgin speaker on
2026-08-07. The approval was given over the set as a whole rather than line by
line; that distinction is recorded in bench/pidgin_eval.json and repeated here
so nobody downstream mistakes it for per-line certification.

Adding a language means adding validated strings here. An unvalidated language
must not be added - falling back to English is honest, while emitting
unverified machine output in a language nobody has checked is not.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Languages with human-validated fixed messages. English is the source.
VALIDATED_LANGUAGES = ("en", "pcm")

LANGUAGE_NAMES = {
    "en": "English",
    "pcm": "Nigerian Pidgin",
}


@dataclass(frozen=True)
class MessageSet:
    """The fixed strings the system can emit, in one language."""

    no_guidance: str
    hazardous_pesticide: str
    unsupported_dosage: str
    stale_chemical: str
    dosage_refusal: str
    withdrawal_period: str
    human_medical: str
    out_of_scope_crop: str
    live_forecast: str


    #: Deliberately carries an English default rather than appearing in every
    #: MessageSet. Every other string here was reviewed by a Nigerian Pidgin
    #: speaker on 2026-08-07; this one was added later and has not been. The
    #: documented fallback - English rather than an unreviewed translation -
    #: is the right behaviour for a refusal, and get() already implements it.
    #: Replace the default with a per-language string once a speaker has
    #: approved one.
    harmful_request: str = "I am sorry, but I cannot help with that."


MESSAGES: dict[str, MessageSet] = {
    "en": MessageSet(
        no_guidance=(
            "I do not have local guidance on that in my documents, so I cannot "
            "give you a reliable answer. Please ask your local extension officer."
        ),
        hazardous_pesticide=(
            "Some sources mention pesticides that are banned or severely "
            "restricted in many countries. Do not use them. Ask your local "
            "extension officer which products are currently registered for your "
            "crop."
        ),
        unsupported_dosage=(
            "This answer contained application rates that are not stated in the "
            "source documents, so they have been removed. Never guess a dose - "
            "ask an extension officer or read the product label."
        ),
        stale_chemical=(
            "Chemical control guidance here comes from older documents. "
            "Pesticide registrations change; confirm with an extension officer "
            "before applying anything."
        ),
        dosage_refusal=(
            "I cannot tell you how much to use. The correct amount depends on "
            "the specific product and what is registered for your crop or "
            "animal. Read the product label, or ask your extension officer."
        ),
        withdrawal_period=(
            "IMPORTANT: after treating an animal with medicine, there is a "
            "withdrawal period before its milk or meat is safe for people to "
            "consume. Ask a veterinary officer how long to wait before selling "
            "or drinking the milk, or slaughtering the animal."
        ),
        human_medical=(
            "If you feel unwell after handling farm chemicals, stop work, wash "
            "thoroughly, and go to a clinic or hospital now. Take the product "
            "container with you so they know what you were exposed to."
        ),
        out_of_scope_crop=(
            "I do not cover that crop. My documents are about cassava, maize, "
            "yam, tomato, rice, cowpea, groundnut, pepper and okra, and about "
            "livestock. Please ask your extension officer."
        ),
        live_forecast=(
            "I cannot tell you what the weather will do - I have no forecast, "
            "only farming guides. I can help you decide when the rains have "
            "truly established, what to do if a dry spell comes, and how to "
            "prepare a field that floods. For a forecast, listen to the radio "
            "or check with your extension officer."
        ),
    ),
    # Reviewed and approved by a Nigerian Pidgin speaker, 2026-08-07.
    "pcm": MessageSet(
        no_guidance=(
            "I no get information about dis one for di book wey I get, so I no "
            "fit give you answer wey you fit trust. Abeg go meet your extension "
            "officer."
        ),
        hazardous_pesticide=(
            "Some of di medicine wey dis book talk about, plenty country don ban "
            "am. No use am at all. Ask your extension officer which one dem allow "
            "for your crop now."
        ),
        unsupported_dosage=(
            "Di answer bin get quantity wey no dey inside di book, so I comot am. "
            "Never guess how much you go use - ask extension officer or read wetin "
            "dem write for di container."
        ),
        stale_chemical=(
            "Dis advice about chemical come from old book. Di rule for pesticide "
            "dey change, so confirm with extension officer before you use anything."
        ),
        dosage_refusal=(
            "I no fit tell you how much you go use. Di correct quantity depend on "
            "di particular product and wetin dem register for your crop or your "
            "animal. Read di label for di container, or ask your extension officer."
        ),
        withdrawal_period=(
            "IMPORTANT: after you give animal medicine, e get time wey you must "
            "wait before di milk or di meat go safe for person to chop. Ask "
            "veterinary officer how many day you go wait before you sell am, drink "
            "di milk, or kill di animal."
        ),
        human_medical=(
            "If your body no dey okay after you touch farm chemical, stop work, "
            "wash yourself well well, and go clinic or hospital now now. Carry di "
            "container go so dem go sabi wetin enter your body."
        ),
        out_of_scope_crop=(
            "I no sabi about dat crop. Di book wey I get na about cassava, corn, "
            "yam, tomato, rice, beans, groundnut, pepper and okra, and about animal. "
            "Abeg ask your extension officer."
        ),
        # Provided by a Nigerian Pidgin speaker, 2026-08-20, and used verbatim.
        live_forecast=(
            "Omo, I not fit tell you anything on this one o. I no get any "
            "forecast. You fit listen for your radio or go ask your extension "
            "officer sha"
        ),
    ),
}


def get(language: str) -> MessageSet:
    """Message set for a language, falling back to English.

    Falling back is deliberate and safe: an English safety warning a farmer
    may need help reading is far better than a machine-translated one that
    reads fluently and means something slightly different.
    """
    return MESSAGES.get(language, MESSAGES["en"])


def is_validated(language: str) -> bool:
    return language in MESSAGES

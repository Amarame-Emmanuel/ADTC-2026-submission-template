"""Reject chunks that are not prose, at retrieval time.

WHY THIS EXISTS
---------------
Found by running the project's own submitted test prompt. Asked why a farmer's
cassava was yellow and twisted, the system answered that the cause was "delayed
harvesting due to lack of a ready market" - wrong, and wrong in a way that would
cost a farmer their crop.

The model was not at fault. It was handed six passages and used them faithfully.
The passages were the problem:

  [1] "Pest control in cassava farms" -> the TITLE PAGE:
      "Abia State, Nigeria ... International Institute of Tropical Agriculture,
       Plant Health Management Division, Cotonou, Benin (c) IITA 2000 ISBN 978-"

  [2] "Common African pests and diseases" -> garbled OCR:
      "I esion a nd spiral nemalod es are o f importan ce on some far m s"

  [5] a post-harvest-losses project report, which is where the wrong answer
      came from verbatim.

Front matter and OCR debris embed as plausible text: they carry the document
title, the crop name and enough agricultural vocabulary to score well, while
containing no guidance at all. The chunker cannot tell - it splits whatever the
extractor produced - and the relevance gate scores whole documents, so a good
manual passes and brings its title page along with it.

WHY AT RETRIEVAL RATHER THAN INDEXING
-------------------------------------
Filtering at index time is the tidier place and costs a 4.5-hour rebuild.
Filtering at retrieval costs microseconds per candidate and can be measured
immediately. Both are worth doing; this is the one that can be done now, and it
also protects against corpora added later without re-running the gates.
"""

from __future__ import annotations

import re

#: Front matter: copyright pages, ISBNs, publisher addresses, catalogue data.
#: These carry the document title and crop names but no guidance.
_FRONT_MATTER = re.compile(
    r"\b(?:isbn|issn|all rights reserved|library of congress|"
    r"cataloguing|cataloging|first published|printed in|"
    r"reproduction of this|no part of this publication|"
    r"correct citation|recommended citation|doi:|"
    r"p\.?o\.? box|tel(?:ephone)?:|fax:|e-?mail:|www\.)\b",
    re.IGNORECASE,
)

#: A whole-line copyright notice.
_COPYRIGHT = re.compile(r"©|\(c\)\s*\d{4}|copyright\s+\d{4}", re.IGNORECASE)

_WORD = re.compile(r"[A-Za-z]+")

#: Below this share of "real" words, text is OCR debris rather than prose.
#: Broken OCR splits words into fragments - "nemalod es", "importan ce",
#: "far m s" - so the giveaway is an unusually high proportion of one- and
#: two-letter tokens.
MIN_REAL_WORD_SHARE = 0.72

#: Front-matter markers tolerated before a chunk is rejected. One ISBN inside a
#: long guidance passage is a citation; three markers in a short chunk is a
#: title page.
MAX_FRONT_MATTER_HITS = 2


def real_word_share(text: str) -> float:
    """Fraction of alphabetic tokens that are three characters or longer.

    English prose runs around 0.75-0.85. Badly OCR'd text falls well below,
    because scanning errors shatter words into fragments.
    """
    words = _WORD.findall(text)
    if len(words) < 20:
        return 1.0  # too short to judge; let other rules decide
    return sum(1 for w in words if len(w) >= 3) / len(words)


def is_front_matter(text: str) -> bool:
    head = text[:600]
    hits = len(_FRONT_MATTER.findall(head)) + len(_COPYRIGHT.findall(head))
    return hits > MAX_FRONT_MATTER_HITS


def is_garbled(text: str) -> bool:
    return real_word_share(text) < MIN_REAL_WORD_SHARE


#: A run of underscores or dots long enough to be a blank to write in, or a
#: dot-leader in a contents list. Three is short enough to catch "___" fields
#: and long enough not to fire on ordinary punctuation or an em-dash.
_BLANK_RUN = re.compile(r"_{3,}|\.{4,}")

#: Question numbering in a survey instrument: "1.1:", "Q3.", "(a)" at the start
#: of a line, repeated down the passage.
_FORM_NUMBERING = re.compile(r"^\s*(?:Q\s*\d+|\d+\.\d+\s*:|\(\s*[a-z]\s*\))",
                             re.IGNORECASE | re.MULTILINE)

#: Enough blanks in one passage that it is a form rather than prose that
#: happens to contain a rule or a redaction.
_MIN_BLANK_RUNS = 3


#: A passage made mostly of questions put TO the reader is a self-assessment
#: checklist, not guidance. Counted as a share so a single rhetorical question
#: in real advice ("What should you look for? Yellowing along the veins.")
#: cannot trip it.
_QUESTION_MARK = re.compile(r"\?")
_SENTENCE_SPLIT = re.compile(r"[.!?]+\s+")

#: Second-person interrogatives are the signature: "Have you determined...",
#: "Do you know...", "Are there restrictions...". Extension prose addresses the
#: farmer in the imperative ("Remove infected plants"), not by quizzing them.
_SELF_ASSESSMENT = re.compile(
    r"\b(?:have|do|did|are|is|can|will|would|should)\s+you(?:r)?\b",
    re.IGNORECASE,
)

#: Consecutive unanswered questions before a passage is a checklist rather than
#: an FAQ. Three is the smallest run that cannot happen by accident in prose.
_MIN_QUESTION_RUN = 3
#: ...and the questions must be aimed at the reader, not at the topic.
_MIN_SELF_ASSESSMENT = 2


def is_question_checklist(text: str) -> bool:
    """A list of questions put to the reader rather than answered for them.

    Missed by the blank-run test above, and found the same way: a farmer asked
    whether to sell yam now or store it and received

        "Have you determined whether to price below, at, or above the market?
         Do your farmers/customers expect sales at certain times?
         Is your seed yam truly better?"

    from a business-plan workbook. Not one blank line in it, and every sentence
    a question the farmer came here to have answered.

    WHAT SEPARATES A CHECKLIST FROM AN FAQ
    --------------------------------------
    Not the number of questions - a first attempt used the share of sentences
    ending in "?" and immediately rejected legitimate FAQ-style guidance:

        "What causes yellowing along the veins? A virus spread by whiteflies.
         How do you stop it? Use clean planting material."

    That is exactly the register extension material is written in, and it is
    useful precisely because it asks a farmer's question and then answers it.

    The difference is whether the questions are ANSWERED. An FAQ alternates
    question and answer; a workbook runs question into question, because the
    answers are the reader's to supply. So this counts CONSECUTIVE questions.
    Three in a row does not happen in prose that is teaching anything.

    Second-person interrogatives are still required alongside: "have you",
    "do you", "are you". Extension guidance addresses a farmer in the
    imperative - "remove infected plants" - rather than by quizzing them.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 4:
        return False

    # _SENTENCE_SPLIT consumes the terminator, so re-derive which sentences were
    # questions from the original text's punctuation order.
    marks = [m for m in re.findall(r"[.!?]", text)]
    run = longest = 0
    for m in marks:
        run = run + 1 if m == "?" else 0
        longest = max(longest, run)
    if longest < _MIN_QUESTION_RUN:
        return False

    return len(_SELF_ASSESSMENT.findall(text)) >= _MIN_SELF_ASSESSMENT


def is_form(text: str) -> bool:
    """Whether the passage is a questionnaire, worksheet or contents list.

    WHY THIS EXISTS
    ---------------
    Asked "Should I sell my yam now or store it?", the system answered with
    twenty-five repetitions of "Yams are sold at the best advantage" and lines
    like:

        Sell now _______ parts out of 10
        Store to sell later _______ Parts out of 10; Store for how long? ____ months

    The model was not looping on its own. Those are the *fill-in-the-blank
    fields of an impact-evaluation questionnaire*, and it was echoing them
    faithfully - the same pattern as the title-page failure this module was
    written for, in a new costume.

    A form embeds like guidance: it carries the crop name, the vocabulary of
    the decision ("sell", "store", "months") and the shape of a recommendation,
    while containing no recommendation at all. It is *worse* than the title
    pages, because its repeated structure induces the model to repeat too.

    124 chunks in the current index carry blank runs, concentrated in impact
    evaluation protocols (49), enterprise budget surveys and training-manual
    worksheets. They surface where guidance is thin: the yam market question
    retrieved six passages scoring 0.693-0.711, one of them below the floor and
    admitted by the lexical exemption, because there is little else to find.

    WHY A COUNT AND NOT A SINGLE MATCH
    ----------------------------------
    One blank run is not a form. Extension prose legitimately contains
    "____ days after planting" as a template a farmer fills in, and a redacted
    dosage prints as underscores. Three or more, or form-style question
    numbering, is a document designed to be written on rather than read.
    """
    if len(_BLANK_RUN.findall(text)) >= _MIN_BLANK_RUNS:
        return True
    # Question numbering alone is weaker evidence, so it needs a blank too.
    return bool(_FORM_NUMBERING.search(text)) and bool(_BLANK_RUN.search(text))


def is_usable(text: str) -> tuple[bool, str]:
    """Whether a retrieved chunk is worth putting in front of the model."""
    if is_front_matter(text):
        return False, "front matter"
    if is_garbled(text):
        return False, f"garbled text ({real_word_share(text):.0%} real words)"
    if is_form(text):
        return False, "questionnaire or worksheet, not guidance"
    if is_question_checklist(text):
        return False, "self-assessment checklist, not guidance"
    return True, ""

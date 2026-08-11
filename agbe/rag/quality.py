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


def is_usable(text: str) -> tuple[bool, str]:
    """Whether a retrieved chunk is worth putting in front of the model."""
    if is_front_matter(text):
        return False, "front matter"
    if is_garbled(text):
        return False, f"garbled text ({real_word_share(text):.0%} real words)"
    return True, ""

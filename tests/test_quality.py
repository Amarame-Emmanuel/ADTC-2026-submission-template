"""Chunk quality: reject title pages and OCR debris before the model sees them.

Both example strings below are real chunks that were retrieved and sent to the
model for the cassava question described in agbe/rag/quality.py.
"""

from __future__ import annotations

from agbe.rag.quality import (
    is_front_matter,
    is_garbled,
    is_question_checklist,
    is_usable,
    real_word_share,
)

# Verbatim from the index: the title page of an IITA field guide, retrieved as
# though it were guidance because it carries the document title and the crop.
TITLE_PAGE = (
    "Pest control in cassava farms Abia State, Nigeria Muaka Toko "
    "International Institute of Tropical Agriculture, Plant Health Management "
    "Division, P.O. Box 08-0932, Cotonou, Benin (c) 2000 ISBN 978-131-190-4 "
    "All rights reserved. Correct citation: Toko M. www.iita.org"
)

# Verbatim from the index: a scanned page whose OCR shattered the words.
GARBLED = (
    "Common African pests and diseases of cassava, yam, sweet potato and "
    "cocoyam I esion a nd spiral nemalod es are o f importan ce on some far m "
    "s, bu t little is known of m os t o t her spe c ies i n t he f ie ld"
)

PROSE = (
    "Cassava mosaic disease is spread by whiteflies and by planting infected "
    "stem cuttings. Symptoms are yellow or pale green patches on the leaves, "
    "which become twisted and distorted. Plant only healthy cuttings taken "
    "from disease-free fields, and uproot and destroy infected plants early."
)


def test_title_page_is_rejected():
    assert is_front_matter(TITLE_PAGE)
    ok, why = is_usable(TITLE_PAGE)
    assert not ok and why == "front matter"


def test_garbled_ocr_is_rejected():
    assert is_garbled(GARBLED)
    assert not is_usable(GARBLED)[0]


def test_real_guidance_passes():
    ok, why = is_usable(PROSE)
    assert ok, why


def test_prose_scores_far_above_the_threshold():
    """The threshold separates the two classes with room to spare.

    If these ever converge the constant is wrong, and this test says so before
    a corpus change silently starts dropping good passages.
    """
    assert real_word_share(PROSE) > 0.80
    assert real_word_share(GARBLED) < 0.65


def test_one_citation_does_not_condemn_a_good_passage():
    """Front matter is a density judgement, not a keyword match.

    Guidance documents cite other documents. A single ISBN or URL inside real
    advice must not remove that advice from the corpus.
    """
    cited = PROSE + " See also: Cassava disease handbook, ISBN 978-92-9059-000-0."
    assert is_usable(cited)[0]


def test_short_fragments_are_not_judged_as_garbled():
    """Too little text to measure. Let the other rules decide."""
    assert real_word_share("Signs of the disease:") == 1.0


def test_a_third_person_questionnaire_is_a_checklist() -> None:
    """The person test was a proxy, and it made the rule inert for an exam.

    `_MIN_SELF_ASSESSMENT` was written for a business-plan workbook, which
    quizzes the reader directly ("Have you determined whether to price below
    the market?"). A yam-disease research guide ends with a bare exam instead:

        Questionnaire 1 Where are yams cultivated? 2 What are the major
        cultivated yam species grown in West Africa? 3 What are climatic
        requirements for yam cultivation? ...

    Thirteen consecutive unanswered questions, not one addressed to a reader,
    `is_question_checklist` returning False - and the passage took a retrieval
    slot from a farmer asking why their yam tubers were rotting in the barn.

    The unbroken run is the direct evidence: an FAQ alternates question and
    answer by definition, so it cannot produce one.
    """
    exam = (
        "Questionnaire 1 Where are yams cultivated? "
        "2 What are the major cultivated yam species grown in West Africa? "
        "3 What are climatic requirements for yam cultivation? "
        "4 What is the effect of yam diseases? "
        "5 What are the major foliage diseases of yams? "
        "6 How is yam mosaic virus transmitted?"
    )
    assert is_question_checklist(exam) is True
    assert is_usable(exam)[0] is False


def test_a_short_run_of_topic_questions_still_needs_the_person_test() -> None:
    """Five, not three, so the ambiguous middle is still governed by person.

    Extension prose does pose questions and answer them, and a three-question
    run is short enough to happen in real teaching material. Only a run too
    long to be an FAQ skips the person requirement.
    """
    faq = (
        "What causes yellowing along the veins? A virus spread by whiteflies. "
        "How do you stop it? Use clean planting material. "
        "When should you rogue? As soon as symptoms appear."
    )
    assert is_question_checklist(faq) is False

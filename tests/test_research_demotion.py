"""Researcher-facing passages must not outrank farmer-facing ones.

WHY THIS FILE EXISTS
--------------------
Asked "my yam tubers are rotting in the barn", the system answered:

    "Use a hand trowel to carefully remove the tubers from the pot without
     damaging the substrate. Label and bag the harvested tubers. Store them by
     sorting by family in the barn."

Pots, substrate and families belong to a genebank nursery protocol. The farmer
has a barn.

THE CORPUS WAS NOT THE PROBLEM
------------------------------
Four of the six retrieved passages were right - yam barns in Abakaliki, wet rot,
nematode damage in storage. Two were institutional: a Standard Operating
Procedure for nursery harvest and a guide to in vitro germplasm collections. The
model took its steps from those two, because a procedure written as numbered
steps reads more like instructions than prose about rot does.

That is why this demotes rather than filters, and why it is not a scope rule:
the material is on-topic and correctly retrieved. It is written for the wrong
reader.
"""

from __future__ import annotations

from agbe.rag.chunker import Chunk
from agbe.rag.index import _demote_research_protocol


def chunk(title: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id="d",
        title=title,
        publisher="p",
        year="2018",
        licence="CC-BY-4.0",
        source_url="",
        page_start=1,
        page_end=1,
        chunk_index=0,
    )


FARMER = chunk(
    "Storage of agricultural products",
    "As soon as the rainy season begins, the tubers in the yam barn begin to "
    "deteriorate rapidly. Inspect daily and remove rotting tubers.",
)
SOP = chunk(
    "Standard Operating Procedure (SOP) for harvest of yam",
    "When senescence is about 80% at 6-8 months after planting, harvest the "
    "tubers from the pot without damaging the substrate.",
)
GERMPLASM = chunk(
    "Technical guidelines for field and in vitro germplasm collections",
    "Fungi invade the tissues of the tubers through wounds caused by insects.",
)


def test_research_passages_move_behind_farmer_passages() -> None:
    chunks = [SOP, FARMER, GERMPLASM]
    assert _demote_research_protocol([0, 1, 2], chunks) == [1, 0, 2]


def test_demotion_never_reduces_the_evidence() -> None:
    """If every candidate is institutional they must all still be returned.

    The same guarantee `_demote_off_crop` makes: a germplasm document can carry
    the only description of a disease in the corpus, and a farmer is better
    served by awkward evidence than by none.
    """
    chunks = [SOP, GERMPLASM]
    assert sorted(_demote_research_protocol([0, 1], chunks)) == [0, 1]


def test_order_within_each_group_is_preserved() -> None:
    """Fusion order is the ranking; this only partitions it."""
    chunks = [FARMER, SOP, FARMER, GERMPLASM]
    assert _demote_research_protocol([0, 1, 2, 3], chunks) == [0, 2, 1, 3]


def test_ordinary_agronomy_words_are_not_research_markers() -> None:
    """"Trial", "plot" and "treatment" appear throughout real extension material.

    They are deliberately absent from the pattern. A rule that demoted every
    passage mentioning a plot would empty the corpus of exactly the field
    guidance it exists to surface.
    """
    ordinary = chunk(
        "Yam field practice",
        "Prepare the plot early. A seed treatment before planting reduces rot, "
        "and on-farm trials showed higher yields from clean setts.",
    )
    assert _demote_research_protocol([0], [ordinary]) == [0]


# ---------------------------------------------------------------------------
# Out-of-scope crops
# ---------------------------------------------------------------------------

from agbe.rag.index import _demote_off_crop, _titled_for_another_crop


def test_a_title_naming_an_unknown_crop_is_off_crop() -> None:
    """CROP_TERMS lists only the nine IN-SCOPE crops.

    A passage about bean or papaya matched nothing, was classified NEUTRAL by
    `_demote_off_crop`, and so could never be demoted. Two defects reduced to
    that: a rice question answered with angular leaf spot (a BEAN disease) from
    "Bean Leaves (New)", and crop-22 retrieving "Papaya (Revised)" for a tomato
    leafminer question.
    """
    assert _titled_for_another_crop("Bean Leaves (New) (Helicoverpa armigera)")
    assert _titled_for_another_crop("Papaya (Revised) (Aphis gossypii)")
    assert not _titled_for_another_crop("Rice (Revised) (Grass family)")
    assert not _titled_for_another_crop("Management of yam diseases")


def test_body_mentions_do_not_make_a_document_off_crop() -> None:
    """The TITLE, not the body, and deliberately.

    Extension documents mention other crops constantly - intercrops, rotations,
    comparisons. Demoting on a body mention would push out most of the corpus.
    """
    assert not _titled_for_another_crop("Cassava production guide")


def test_off_crop_by_title_sorts_behind_on_crop() -> None:
    rice = chunk("Rice (Revised) (Grass family)", "Brown leaf spot of rice.")
    bean = chunk("Bean Leaves (New)", "Angular leaf spot lesions on the foliage.")
    assert _demote_off_crop([1, 0], [rice, bean], "my rice leaves have spots") == [0, 1]

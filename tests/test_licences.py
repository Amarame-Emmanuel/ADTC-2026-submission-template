"""Licence exclusion, enforced where the text is actually used."""

from __future__ import annotations

import numpy as np
import pytest

from agbe.rag.chunker import Chunk
from agbe.rag.index import VectorIndex
from agbe.rag.licences import excluded


@pytest.mark.parametrize("licence", [
    "CC-BY-ND-4.0",
    "CC-BY-NC-ND-4.0",
    "CC BY-NC-ND 3.0",
    "NoDerivs 2.5",
    "No Derivatives",
    "ND",
    "Copyrighted. All rights reserved.",
])
def test_excluded_licences(licence):
    is_excluded, why = excluded(licence)
    assert is_excluded, licence
    assert why


@pytest.mark.parametrize("licence", [
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-NC-SA-3.0",   # the Infonet licence - NC-SA, not ND
    "CC0",
    "Public domain",
    "",                  # unknown: the fetch gate handles these, not the index
])
def test_admitted_licences(licence):
    assert not excluded(licence)[0], licence


def _chunk(i: int, licence: str) -> Chunk:
    return Chunk(
        text=f"Guidance passage {i} about cassava spacing and weeding.",
        doc_id=f"doc-{i}", title=f"Doc {i}", publisher="Test", year="2020",
        licence=licence, source_url="https://example.org", page_start=1,
        page_end=1, chunk_index=0,
    )


def test_index_load_drops_excluded_rows_and_keeps_alignment(tmp_path):
    """Vectors and chunks must be dropped together, or every later search
    would cite the wrong document for the right vector."""
    chunks = [
        _chunk(0, "CC-BY-4.0"),
        _chunk(1, "CC-BY-ND-4.0"),
        _chunk(2, "CC-BY-NC-SA-3.0"),
        _chunk(3, "CC-BY-NC-ND-4.0"),
    ]
    vectors = np.eye(4, 8, dtype=np.float32)  # row i is distinguishable
    VectorIndex.build(chunks, vectors).save(tmp_path)

    loaded = VectorIndex.load(tmp_path)

    assert [c.doc_id for c in loaded.chunks] == ["doc-0", "doc-2"]
    # Row alignment survived the mask: doc-0 kept row 0, doc-2 kept row 2.
    assert loaded.vectors[0][0] == 1.0
    assert loaded.vectors[1][2] == 1.0
    assert len(loaded.vectors) == len(loaded.chunks) == 2

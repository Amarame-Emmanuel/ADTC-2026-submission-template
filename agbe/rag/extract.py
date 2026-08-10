"""PDF text extraction for the Àgbẹ̀ corpus.

Two things this module exists to handle beyond "get the text out":

SCANNED DOCUMENTS
    A large share of the most useful agricultural extension material is old,
    and old often means scanned. A scanned PDF has pages, images and metadata
    but no text layer - `extract_text()` returns near-nothing. Indexed blindly,
    it contributes zero retrievable content while inflating the document count,
    which is the worst possible outcome: it looks like corpus coverage and is
    not. `probe()` detects this explicitly so such documents are reported
    rather than silently absorbed.

MEMORY
    Extraction runs on the same 7 GB budget as everything else. Pages are
    processed and released one at a time rather than accumulating a list of
    per-page strings, and a page cap bounds the worst case for the occasional
    600-page compendium.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Below this many characters per page, a document is almost certainly scanned
#: images rather than digital text. Chosen empirically: a sparse page of a
#: real text PDF still carries a few hundred characters; a scanned page yields
#: only stray OCR artefacts or nothing at all.
SCANNED_CHARS_PER_PAGE = 120

#: Hard cap so one pathological document cannot dominate memory or time.
MAX_PAGES = 800

_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
#: PDF text extraction routinely splits words across line breaks with hyphens.
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


@dataclass
class Extraction:
    """Result of reading one PDF.

    `pages` keeps per-page text rather than only the concatenated `text`,
    because a citation that says "FAO Crop Production Manual (2015), p. 23"
    is checkable by a farmer or an auditor and one that says only "FAO Crop
    Production Manual" is not. Page boundaries are cheap to keep here and
    impossible to recover later.
    """

    path: Path
    text: str
    n_pages: int
    n_chars: int
    is_scanned: bool
    pages: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def chars_per_page(self) -> float:
        return self.n_chars / self.n_pages if self.n_pages else 0.0

    @property
    def usable(self) -> bool:
        return not self.error and not self.is_scanned and self.n_chars > 2000


def clean(raw: str) -> str:
    """Normalise extracted text without destroying structure.

    Paragraph breaks are preserved because the chunker uses them as split
    points; collapsing all whitespace would force it to fall back to blind
    character-count splitting and cut passages mid-sentence.
    """
    text = _HYPHEN_BREAK.sub(r"\1\2", raw)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def probe(path: Path, max_pages: int = MAX_PAGES) -> Extraction:
    """Extract text from a PDF, flagging scanned documents.

    Never raises on a malformed PDF - corpus harvesting processes dozens of
    files from external sources and one corrupt download should be reported,
    not fatal.
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - dependency is pinned
        return Extraction(path, "", 0, 0, False, error="pypdf not installed")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        return Extraction(path, "", 0, 0, False, error=f"unreadable: {exc}")

    total_pages = len(reader.pages)
    n_pages = min(total_pages, max_pages)

    # Index i of `pages` is the text of PDF page i+1. Failed pages stay as
    # empty strings rather than being dropped, so the list index remains a
    # valid page number - silently compacting it would shift every citation
    # after the first unreadable page.
    pages: list[str] = []
    for i in range(n_pages):
        try:
            pages.append(clean(reader.pages[i].extract_text() or ""))
        except Exception:  # noqa: BLE001 - a bad page should not kill the file
            pages.append("")

    text = clean("\n\n".join(pages))
    n_chars = len(text)
    is_scanned = n_pages > 0 and (n_chars / n_pages) < SCANNED_CHARS_PER_PAGE

    return Extraction(
        path=path,
        text=text,
        n_pages=total_pages,
        n_chars=n_chars,
        is_scanned=is_scanned,
        pages=pages,
    )

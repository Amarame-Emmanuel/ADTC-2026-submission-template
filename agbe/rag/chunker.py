"""Split extracted documents into retrievable, citable passages.

WHAT A CHUNK HAS TO CARRY
-------------------------
A chunk is not just text. Every answer this system gives must be traceable to
a specific place in a specific document with a known licence and publication
year, because:

  * the farmer deserves to know where the advice came from;
  * the safety layer suppresses chemical recommendations from documents older
    than the recency threshold, which it can only do if each passage knows its
    own publication year;
  * an auditor checking a claim needs a page number, not a document title.

So provenance travels with the text, chunk by chunk, rather than being looked
up later from a document table.

TOKEN BUDGETING
---------------
Chunks are sized in characters against a deliberately conservative
characters-per-token estimate. The embedding model has a 512-token context and
silently truncates beyond it; a chunk that overflows loses its tail without
error, which would corrupt retrieval invisibly. Underestimating chunk size
costs a little redundancy, overestimating costs correctness, so the constant
below errs low on purpose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

#: Conservative. English averages nearer 4 characters per token, but
#: agricultural text is dense with long technical terms ("Xanthomonas",
#: "Bemisia tabaci") that tokenise into several pieces each. Assuming fewer
#: characters per token means we overestimate the token count and stay under
#: the embedder's 512-token ceiling.
CHARS_PER_TOKEN = 3.5

#: Paragraph-ish split. PDF extraction rarely produces clean paragraphs, so we
#: also treat bullet markers and numbered headings as boundaries - extension
#: manuals are full of them and they are natural semantic breaks.
#:
#: THE SINGLE-NEWLINE HEADING
#: --------------------------
#: This originally split only on blank lines and bullets, which meant a heading
#: separated from its neighbours by a SINGLE newline was never isolated into
#: its own paragraph - and `looks_like_heading` below, which would have
#: recognised it, was never given the chance.
#:
#: PDF extraction produces exactly that shape. Measured in "Growing cassava:
#: training manual for extension & farmers in Zambia":
#:
#:     ...the plants become stunted (Figure 6.5).\n
#:     39\n
#:     Cassava Brown Streak Disease (CBSD)\n
#:     CBSD is a disease of cassava...
#:
#: The symptom text above that heading is the tail of the *Cassava Mosaic
#: Disease* section, whose own heading stayed behind in the previous chunk. So
#: the passage describing mosaic symptoms carried the name of a different
#: disease - and it was the top-ranked passage for the submitted test prompt.
#: The model read it faithfully and diagnosed brown streak for textbook mosaic
#: symptoms. See docs/RETRIEVAL.md.
#:
#: A bare page number on its own line is consumed with the following break so
#: it does not become a chunk of its own or pad the heading.
_SPLIT = re.compile(
    r"\n\s*\n"
    r"|\n(?=\s*(?:[-•*·]|\d+[.)]\s))"
    r"|\n(?:\s*\d{1,4}\s*\n)?(?=[A-Z][^\n]{2,70}\n)"
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: A heading: a short line with no terminal punctuation, or one ending in a
#: colon. Extension pages are organised as "Signs of X", "What to do:",
#: "Host plants" - these are the real semantic boundaries, far better split
#: points than an arbitrary character count.
_HEADING = re.compile(r"^(?:[A-Z][^.!?]{2,70}:?)$")


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 72:
        return False
    if line.endswith(":"):
        return True
    return bool(_HEADING.match(line)) and len(line.split()) <= 9


@dataclass
class Chunk:
    """One retrievable passage with full provenance."""

    text: str
    doc_id: str
    title: str
    publisher: str
    year: str
    licence: str
    source_url: str
    page_start: int
    page_end: int
    chunk_index: int

    def citation(self) -> str:
        """Human-readable attribution, shown with every answer."""
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        year = f" ({self.year})" if self.year and self.year != "unknown" else ""
        return f"{self.title}{year}, {self.publisher}, {pages}"

    def to_dict(self) -> dict:
        return asdict(self)
def _split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries.

    Extension manuals sometimes carry a whole page as one extracted block. A
    blind character cut would sever a sentence mid-clause - and a passage that
    begins "...and apply 2 litres per hectare" is worse than useless, because
    it reads as complete while having lost what it applies to.
    """
    sentences = _SENTENCE_END.split(para)
    out: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > max_chars:
            out.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf.strip():
        out.append(buf.strip())

    # A single sentence longer than the budget: fall back to a hard cut, but
    # only for that sentence.
    final: list[str] = []
    for piece in out:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            final.extend(
                piece[i : i + max_chars] for i in range(0, len(piece), max_chars)
            )
    return final


def chunk_pages(
    pages: list[str],
    *,
    doc_id: str,
    title: str,
    publisher: str,
    year: str,
    licence: str,
    source_url: str,
    target_tokens: int = 350,
    overlap_tokens: int = 60,
    min_tokens: int = 40,
) -> list[Chunk]:
    """Turn per-page text into overlapping, page-attributed chunks.

    Overlap exists so a passage split across a boundary is still retrievable
    from either side - a control measure whose symptom description landed in
    the previous chunk would otherwise be unfindable by symptom.

    Chunks below `min_tokens` are dropped. Page headers, footers and stray
    figure captions extract as tiny fragments that match many queries weakly
    and answer none of them.
    """
    # Reserve room for the title prefix added in emit(). Without this the
    # prefix is appended after the size check and pushes chunks past the
    # budget - measured at 1,666 characters against a 1,575 ceiling, which
    # breaks the invariant that no chunk exceeds the embedder's context.
    title_overhead = len(title) + 1 if title else 0
    max_chars = int(target_tokens * CHARS_PER_TOKEN) - title_overhead
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
    min_chars = int(min_tokens * CHARS_PER_TOKEN)

    chunks: list[Chunk] = []
    buf = ""
    buf_start_page = 1

    def emit(text: str, end_page: int) -> None:
        # Prefix the document title to every chunk after the first.
        #
        # Chunks are retrieved independently, so a chunk that says "twisting of
        # the head and neck, greenish watery diarrhoea" without naming the
        # disease is invisible to anyone searching for "Newcastle" - and its
        # embedding carries no signal about what condition it describes. The
        # title is the cheapest possible context: ~5 tokens that tell both the
        # embedder and the model what this passage is about.
        #
        # Skipped when the text already opens with the title, which is common
        # for a page's first chunk.
        body = text
        if title and not text[:120].lower().startswith(title[:40].lower()):
            body = f"{title}\n{text}"

        chunks.append(
            Chunk(
                text=body,
                doc_id=doc_id,
                title=title,
                publisher=publisher,
                year=year,
                licence=licence,
                source_url=source_url,
                page_start=buf_start_page,
                page_end=end_page,
                chunk_index=len(chunks),
            )
        )

    def flush(end_page: int, carry_overlap: bool = True) -> None:
        nonlocal buf, buf_start_page
        text = buf.strip()
        if len(text) >= min_chars:
            # HARD CEILING. The carried-over overlap is prepended to the next
            # buffer, so an assembled chunk can exceed max_chars by up to the
            # overlap length - observed at 1786 characters against a 1575
            # budget, roughly 510 tokens against the embedder's 512 limit.
            #
            # llama.cpp truncates silently past the context window: the chunk
            # would embed, retrieve and cite while missing its tail, with no
            # error anywhere. Splitting here guarantees the invariant the rest
            # of the pipeline assumes.
            if len(text) > max_chars:
                for piece in _split_long_paragraph(text, max_chars):
                    if len(piece.strip()) >= min_chars:
                        emit(piece.strip(), end_page)
            else:
                emit(text, end_page)
        # Carry a tail forward as overlap for the next chunk - except across a
        # section heading.
        #
        # Overlap exists so a passage split mid-section stays retrievable from
        # either side. At a section boundary it does the opposite: it copies the
        # previous section's text into a chunk headed by the next section's
        # title, which is precisely the mislabelling this fix exists to remove.
        #
        # Measured: with the heading split working but overlap still carried,
        # the mosaic symptom text was re-imported into the chunk headed
        # "Cassava Brown Streak Disease (CBSD)" and the defect survived intact.
        # A heading is exactly where the two sides should NOT bleed together.
        if carry_overlap:
            buf = text[-overlap_chars:] if overlap_chars and len(text) > overlap_chars else ""
        else:
            buf = ""
        buf_start_page = end_page

    for page_no, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        for para in _SPLIT.split(page_text):
            para = para.strip()
            if not para:
                continue

            # Prefer to break at a heading rather than mid-section. Extension
            # pages are structured as "Signs of X" followed by the symptom
            # list; splitting between them put the disease name in one chunk
            # and its symptoms in another, which is exactly how the Newcastle
            # disease page became unretrievable by symptom. Only break when the
            # buffer is already substantial, so a run of short headings does
            # not shred the text.
            # The half-full condition was a second reason the mosaic/brown-streak
            # boundary survived: when the CBSD heading arrived the buffer held
            # only ~230 characters of trailing mosaic text against a ~580
            # threshold, so even a detected heading would not have broken there.
            #
            # A heading is a semantic boundary whatever the buffer contains. The
            # guard it replaces existed to stop a run of short headings shredding
            # the text, which `min_chars` already prevents - flush() drops
            # anything below it - so the size test was doing no work that
            # mattered and was suppressing the breaks that did.
            if looks_like_heading(para) and len(buf.strip()) >= min_chars:
                flush(page_no, carry_overlap=False)

            for piece in (
                _split_long_paragraph(para, max_chars)
                if len(para) > max_chars
                else [para]
            ):
                if buf and len(buf) + len(piece) + 1 > max_chars:
                    flush(page_no)
                if not buf:
                    buf_start_page = page_no
                buf = f"{buf}\n{piece}".strip() if buf else piece

    if buf.strip():
        flush(len(pages) if pages else 1)

    return chunks

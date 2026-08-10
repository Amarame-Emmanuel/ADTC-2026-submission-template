"""HTML text extraction for archived website corpora.

WHY BOILERPLATE REMOVAL IS THE WHOLE JOB
----------------------------------------
The Infonet-Biovision offline export is a Drupal site. A single pest page
measures ~130,000 characters of HTML, of which the actual guidance is a small
fraction; the rest is toolbars, navigation menus, search blocks, language
switchers and footers - *identical on all 2,159 pages*.

Indexed naively, that shared chrome would dominate every page's text. Every
document would look like every other document, cosine similarity between any
query and any page would be driven by menu text, and retrieval would collapse
into noise. Worse, it would collapse quietly: the index would build, the
numbers would look plausible, and the answers would be wrong.

So this module is mostly a list of things not to read.

NO NEW DEPENDENCY
-----------------
BeautifulSoup would be the reflex choice. Python's stdlib `html.parser` is
sufficient for "take what is inside <main>, skip script/style/nav", and this
project has spent real effort keeping the dependency set small enough to audit.
A parser we do not add is a parser nobody has to build on a slow connection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

#: Elements whose *contents* are never text we want. Anything between the open
#: and close tag is discarded wholesale.
SKIP_CONTENT = {
    "script", "style", "noscript", "svg", "canvas", "template",
    "nav", "header", "footer", "form", "select", "option", "button",
    "iframe", "aside",
}

#: Class/id fragments marking site chrome, applied as a light second pass
#: *inside* <main>. The heavy lifting is done by <main> itself: on this site a
#: page is ~130,000 characters of HTML of which <main> holds ~8,900, so element
#: scoping already removes 93% of the boilerplate.
#:
#: Kept deliberately short. An earlier version included "off-canvas", which
#: matched Drupal's outermost wrapper `dialog-off-canvas-main-canvas` - the
#: element containing the entire page - and silently extracted zero characters
#: from every document. Substring matching against composed class lists is
#: sharp-edged, so each marker here has to earn its place.
CHROME_MARKERS = (
    "toolbar", "navbar", "region-nav", "region-header", "region-footer",
    "region-search", "search-block", "breadcrumb", "pager",
    "gtranslate", "sidebar", "social-share", "cookie", "skip-link",
    "block-system-branding",
)

#: Tags that imply a line break in the extracted text, so paragraph structure
#: survives for the chunker to split on.
BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "section", "article", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "td", "dt", "dd",
}

_WS = re.compile(r"[ \t\xa0]+")
_BLANKS = re.compile(r"\n{3,}")


@dataclass
class HtmlDocument:
    title: str
    text: str
    n_chars: int = 0
    skipped_chrome: bool = False
    headings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        # Below this a page is a stub, a redirect or a listing page - real
        # guidance pages on this site run to several thousand characters.
        return self.n_chars >= 600


class _Extractor(HTMLParser):
    """Collects text from the main content region only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.headings: list[str] = []
        self.title = ""

        self._skip_depth = 0          # inside a SKIP_CONTENT element
        self._chrome_depth = 0        # inside a chrome-classed container
        self._chrome_tag: str | None = None
        self._in_title = False
        self._in_heading: str | None = None
        self._main_depth = 0          # inside <main>
        self._saw_main = False

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _is_chrome(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        """Whether an element is site chrome.

        `html` and `body` are never chrome, whatever classes they carry. This
        archive was captured with a Drupal admin session active, so every
        <body> carries `toolbar-fixed toolbar-horizontal toolbar-tray-open`.
        Matching "toolbar" against it classified the entire document as
        navigation and extracted zero characters from all 2,159 pages - a
        failure that produced no error, just empty output.
        """
        if tag in ("html", "body"):
            return False
        blob = " ".join(v or "" for k, v in attrs if k in ("class", "id", "role")).lower()
        return any(marker in blob for marker in CHROME_MARKERS)

    @property
    def _collecting(self) -> bool:
        if self._skip_depth or self._chrome_depth:
            return False
        # Once a <main> has been seen, only collect inside it. Pages without a
        # <main> fall back to collecting everything not otherwise excluded.
        return self._main_depth > 0 or not self._saw_main

    # -- parser hooks ------------------------------------------------------

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "main":
            self._saw_main = True
            self._main_depth += 1
            return

        if tag == "title":
            self._in_title = True
            return

        if tag in SKIP_CONTENT:
            self._skip_depth += 1
            return

        # Track nesting of a chrome container so it closes at the right depth.
        if self._chrome_depth:
            if tag == self._chrome_tag:
                self._chrome_depth += 1
            return

        if self._is_chrome(tag, attrs):
            self._chrome_depth = 1
            self._chrome_tag = tag
            return

        if tag in ("h1", "h2", "h3", "h4"):
            self._in_heading = tag

        if tag in BLOCK_TAGS and self._collecting:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self._main_depth = max(0, self._main_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in SKIP_CONTENT:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._chrome_depth and tag == self._chrome_tag:
            self._chrome_depth -= 1
            if self._chrome_depth == 0:
                self._chrome_tag = None
            return
        if tag == self._in_heading:
            self._in_heading = None
        if tag in BLOCK_TAGS and self._collecting:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if not self._collecting:
            return
        if not data.strip():
            return
        self.parts.append(data)
        if self._in_heading:
            heading = data.strip()
            if len(heading) > 2:
                self.headings.append(heading)


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------
# A Drupal site renders its *field labels* into the page. An audit of the
# extracted corpus found, across 827 pages:
#
#     "Bookmark this page"        827  (every page)
#     "Minor Pest Title"          549
#     "Minor Pest Description"    549
#     "Minor Pest What to do."    549
#     "Minor Pest Firstcontent"   535
#     "PH Pests Media Gallery"    529
#     "Is this a Minor Pest?"     550
#
# None of it is content, and all of it is damaging. Repeated in nearly every
# chunk, it raises the baseline similarity between every pair of chunks - the
# embedder is asked to distinguish documents that share several identical
# lines. It also inflates BM25 term frequencies for "pest", "title" and
# "minor", and it consumes prompt tokens the farmer waits through.
#
# "What to do:" is deliberately NOT stripped: it appears 1,454 times because it
# is the heading of the guidance section, which is the most useful text on the
# page.
_BOILERPLATE_LINES = (
    "bookmark this page",
    "is this a minor pest?",
    "minor pest title",
    "minor pest description",
    "minor pest what to do.",
    "minor pest firstcontent",
    "minor pest position",
    "minor pest media gallery",
    "ph pests media gallery",
    "information source links",
    "information source link",
    "print this page",
    "share this page",
)

#: Everything from here to the end of the page is reference apparatus.
_TAIL_MARKERS = (
    "information source link",
    "further reading",
    "references:",
)

#: The site stores some text with BBCode markup that renders literally after
#: extraction: "Black bean aphid ([i]Aphis fabae[/i])".
_BBCODE = re.compile(r"\[/?(?:i|b|u|url[^\]]*|img[^\]]*)\]", re.IGNORECASE)

#: Escape sequences that survived as literal text rather than being decoded.
#: \xA0 alone appears 7,063 times across the archive.
_LITERAL_ESCAPE = re.compile(r"\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\[trn]")


#: Tags that appear as literal TEXT rather than markup.
#:
#: The source stores some content HTML-escaped (&lt;b&gt;), so it survives the
#: parser as data and is then un-escaped by convert_charrefs - arriving in the
#: extracted text as a visible "<b>Order/Family:</b>". The parser cannot catch
#: these because at parse time they are not tags.
_LITERAL_TAG = re.compile(r"</?\s*(?:b|i|u|em|strong|br|p|span|div|font)\b[^>]*>",
                          re.IGNORECASE)

#: Photo credits, which appear as standalone lines and carry no advisory value.
_CREDIT_LINE = re.compile(
    r"^[A-Z][A-Za-z.\- ]{2,40},\s*(?:icipe|Biovision|CABI|IITA|Bugwood\.org|"
    r"Wikipedia|courtesy.*)$",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Remove CMS chrome that survived element-level extraction."""
    text = _BBCODE.sub("", text)
    text = _LITERAL_TAG.sub("", text)

    # Two different problems that look identical in a terminal.
    #
    # The archive contains ZERO real U+00A0 characters and 7,063 occurrences of
    # the literal four-character string \xA0 - the export double-encoded them,
    # so they arrive as visible text. `replace("\xa0", " ")` matches the real
    # character and therefore did nothing at all here. Both forms are handled;
    # the literal one is what actually appears.
    text = text.replace(" ", " ")
    text = _LITERAL_ESCAPE.sub(" ", text)

    lines_out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        low = stripped.lower()

        if low in _BOILERPLATE_LINES:
            continue
        # Field labels also appear inline, prefixed to real content.
        for label in ("minor pest title", "minor pest description",
                      "minor pest what to do.", "minor pest firstcontent",
                      "minor pest position", "is this a minor pest?"):
            if low.startswith(label):
                stripped = stripped[len(label):].strip(" .:-")
                low = stripped.lower()
        if not stripped:
            continue
        if any(low.startswith(m) for m in _TAIL_MARKERS):
            break
        if _CREDIT_LINE.match(stripped):
            continue
        lines_out.append(stripped)

    # Drop paragraphs repeated within the same page. These sites commonly show
    # an intro block and then repeat it verbatim under a section heading - the
    # Newcastle page carries its opening paragraph twice. Duplicated text is
    # paid for twice in the prompt and pulls the page's embedding toward
    # whichever passage happens to be repeated.
    #
    # Keyed on a normalised form so trivial whitespace differences still match,
    # and only applied to substantial paragraphs: short lines like "Symptoms:"
    # are legitimately repeated section headings.
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines_out:
        if len(line) < 80:
            deduped.append(line)
            continue
        key = re.sub(r"\W+", "", line.lower())[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    return "\n".join(deduped)


def clean_text(raw: str) -> str:
    text = _WS.sub(" ", raw)
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop one- and two-word fragments: menu leftovers, icon labels, stray
    # link text. Real guidance sentences survive comfortably.
    kept = [ln for ln in lines if len(ln.split()) > 2 or not ln]
    return _BLANKS.sub("\n\n", "\n".join(kept)).strip()


def extract_html(raw_html: str) -> HtmlDocument:
    """Pull the guidance text out of one archived page."""
    parser = _Extractor()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed pages are common in archives
        pass

    text = strip_boilerplate(clean_text("".join(parser.parts)))

    title = re.sub(r"\s*\|\s*Infonet Biovision.*$", "", parser.title.strip(),
                   flags=re.IGNORECASE).strip()
    if not title and parser.headings:
        title = parser.headings[0]

    return HtmlDocument(
        title=title or "(untitled)",
        text=text,
        n_chars=len(text),
        skipped_chrome=parser._saw_main,
        headings=parser.headings[:12],
    )

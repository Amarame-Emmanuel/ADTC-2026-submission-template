#!/usr/bin/env python3
"""Ingest the Infonet-Biovision offline export into the Àgbẹ̀ corpus.

Infonet-Biovision (Biovision Africa Trust, with icipe) publishes a free offline
export of its whole site - ~1.8 GB, 2,159 HTML pages and 874 PDFs covering
plant health, animal health, agro-ecological zones and value addition. That
spans all four advisory areas, and it is the single largest corpus source
available to this project.

Licence: CC BY-NC-SA 3.0. As with every other source, the archive is fetched by
the operator and never redistributed from this repository; only the manifest
entry recording its provenance is committed.

THREE THINGS THIS SCRIPT EXISTS TO HANDLE
-----------------------------------------
1. BOILERPLATE. A page is ~130,000 characters of Drupal HTML wrapping ~2,500
   characters of guidance. See agbe/rag/html_extract.py.

2. DUPLICATION. Of 1,626 usable pages only 1,099 have unique content. One
   1,509-character landing block appears 472 times; "Aphids" appears 31 times
   under different paths. Indexed as-is, those duplicates would dominate
   retrieval - the same passage returned four times looks like four
   independent sources agreeing, which is exactly the false confidence a
   citation-based system must not manufacture.

3. PROVENANCE. Individual pages carry no publication date, so the archive
   build date (March 2018) is used for every page. That is the honest
   available answer and it is conservative in the right direction: it makes
   pages look newer than some of their content, so the chemical-recency rule
   is applied with that caveat recorded in the manifest.

Usage
-----
    python scripts/ingest_infonet.py --dry-run
    python scripts/ingest_infonet.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agbe.rag.html_extract import extract_html  # noqa: E402
from agbe.rag.relevance import score_text  # noqa: E402

ARCHIVE = ROOT / "corpus" / "external" / "infonet.zip"
OUT_JSONL = ROOT / "corpus" / "processed" / "infonet.jsonl"
MANIFEST = ROOT / "corpus" / "manifest.json"

SOURCE_META = {
    "publisher": "Infonet-Biovision (Biovision Africa Trust / icipe)",
    "licence": "CC BY-NC-SA 3.0",
    "year": "2018",
    "base_url": "https://infonet-biovision.org/",
    "caveat": (
        "Archive build March 2018; individual pages carry no date, so page "
        "content may predate this. Kenya/East Africa focus - pest and disease "
        "biology transfers to Nigeria, varieties and planting calendars do not. "
        "Editorial slant toward ecological/organic practice."
    ),
}

#: Near-duplicate threshold (Jaccard over word shingles).
#:
#: Exact hashing is not enough. "Aphids" appears 31 times across crop sections
#: with small per-crop edits, so every copy has a different hash while the
#: guidance is substantially the same. Four near-identical passages retrieved
#: together read as four independent sources agreeing - manufacturing exactly
#: the false confidence a citation-based system exists to prevent.
#:
#: Compared only within a title group, so cost stays trivial (the largest group
#: is 31 pages) and genuinely distinct pages that happen to share a title are
#: still kept.
NEAR_DUPLICATE_JACCARD = 0.75
SHINGLE_SIZE = 5


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < size:
        return {hash(" ".join(words))}
    return {
        hash(" ".join(words[i : i + size])) for i in range(len(words) - size + 1)
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


#: Pages whose title marks them as navigation rather than guidance. The
#: content-hash pass removes the 472 duplicates of the landing block; this
#: removes the one surviving copy.
TITLE_BLOCKLIST = {
    "home", "(untitled)", "admin", "about us", "contact", "search",
    "sitemap", "login", "user account", "publications",
}

#: Path prefixes worth ingesting, mapped to a hint about which advisory area
#: they serve. Everything else in the archive - taxonomy stubs, admin pages,
#: image galleries - is skipped without opening.
CONTENT_PREFIXES = (
    "PlantHealth/", "plant_pests/", "crops-fruits-vegetables/",
    "natural-pest-control/", "cultural-control/",
    "animal-health-and-disease/", "AnimalHealth/", "animal-husbandry/",
    "animal-species/",
    "agro-ecological-zones", "water-management/",
    "processing-and-value-addition/",
    "trees/", "indigenous-plants/", "medicinal-plants/",
)


@dataclass
class Page:
    doc_id: str
    title: str
    text: str
    source_url: str
    publisher: str
    year: str
    licence: str
    areas: list[str]
    crops: list[str]
    relevance_score: float
    n_chars: int


#: Matches a scientific binomial, e.g. "(Aphis gossypii)".
_BINOMIAL = re.compile(r"\(([A-Z][a-z]+ [a-z]{3,})\)")


def specific_title(title: str, text: str) -> str:
    """Disambiguate titles that the site reuses across many pages.

    Infonet heads 31 separate pages "Aphids", 24 "Spider mites", 23
    "Anthracnose". Those pages are genuinely distinct - pairwise similarity
    between the Aphids pages is 2-19%, because each covers a different species
    - so they are all worth keeping. But a citation reading "Aphids (2018),
    Infonet-Biovision" thirty-one times tells a farmer nothing about which
    source said what, and defeats the point of citing at all.

    Where the page names a species, promote it into the title.
    """
    match = _BINOMIAL.search(text[:600])
    if not match:
        return title

    binomial = match.group(1)
    if binomial.lower() in title.lower():
        return title

    # Deliberately just the title plus the species. An earlier version tried to
    # lift the common name preceding the binomial out of the surrounding prose
    # and produced titles like "Aphids are serious pests of peas. The pea aphid
    # (Acyrthosiphon pisum)" - a sentence fragment masquerading as a citation.
    # "Aphids (Acyrthosiphon pisum)" disambiguates just as well and cannot
    # degrade into prose.
    return f"{title} ({binomial})"


def path_to_url(name: str) -> str:
    inner = name.split("/", 1)[1] if "/" in name else name
    return SOURCE_META["base_url"] + inner


def slug(name: str) -> str:
    inner = name.split("/", 1)[1] if "/" in name else name
    return re.sub(r"[^a-z0-9]+", "-", inner.lower().removesuffix(".html")).strip("-")


def wanted(name: str) -> bool:
    if not name.lower().endswith(".html"):
        return False
    inner = name.split("/", 1)[1] if "/" in name else name
    return inner.startswith(CONTENT_PREFIXES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap pages (for testing)")
    args = ap.parse_args()

    if not ARCHIVE.exists():
        print(f"archive not found: {ARCHIVE}")
        print("download the offline export from "
              "https://infonet-biovision.org/downloadoffline and place it there")
        return 1

    z = zipfile.ZipFile(ARCHIVE)
    candidates = [n for n in z.namelist() if wanted(n)]
    if args.limit:
        candidates = candidates[: args.limit]

    print(f"archive : {ARCHIVE.name} ({ARCHIVE.stat().st_size / 1024**3:.2f} GB)")
    print(f"in-scope: {len(candidates)} HTML pages\n")

    stats = collections.Counter()
    seen_hashes: dict[str, str] = {}
    kept_shingles: dict[str, list[set[int]]] = collections.defaultdict(list)
    pages: list[Page] = []

    for name in candidates:
        stats["examined"] += 1
        try:
            raw = z.read(name).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            stats["unreadable"] += 1
            continue

        doc = extract_html(raw)

        if not doc.usable:
            stats["too_short"] += 1
            continue

        if doc.title.strip().lower() in TITLE_BLOCKLIST:
            stats["navigation_page"] += 1
            continue

        digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes[digest] = name

        # Near-duplicate check within the title group.
        title_key = doc.title.strip().lower()
        sig = shingles(doc.text)
        if any(
            jaccard(sig, kept) >= NEAR_DUPLICATE_JACCARD
            for kept in kept_shingles[title_key]
        ):
            stats["near_duplicate"] += 1
            continue

        rel = score_text(doc.text)
        if not rel.accepted:
            stats["off_topic"] += 1
            continue

        kept_shingles[title_key].append(sig)

        pages.append(
            Page(
                doc_id=f"infonet-{slug(name)}"[:80],
                title=specific_title(doc.title, doc.text),
                text=doc.text,
                source_url=path_to_url(name),
                publisher=SOURCE_META["publisher"],
                year=SOURCE_META["year"],
                licence=SOURCE_META["licence"],
                areas=rel.areas,
                crops=rel.crops,
                relevance_score=round(rel.score, 2),
                n_chars=doc.n_chars,
            )
        )
        stats["accepted"] += 1

    print("=" * 72)
    for key in ("examined", "too_short", "navigation_page", "duplicate",
                "near_duplicate", "off_topic", "unreadable", "accepted"):
        if stats[key]:
            print(f"  {key:<18} {stats[key]:>5}")
    print("=" * 72)

    if pages:
        area_counts = collections.Counter(a for p in pages for a in p.areas)
        crop_counts = collections.Counter(c for p in pages for c in p.crops)
        total_chars = sum(p.n_chars for p in pages)
        print(f"\n  text volume   {total_chars/1024:.0f} KB across {len(pages)} pages")
        print(f"  by area       {dict(area_counts)}")
        print(f"  by crop       {dict(crop_counts)}")
        print("\n  highest scoring:")
        for p in sorted(pages, key=lambda p: -p.relevance_score)[:8]:
            print(f"    {p.relevance_score:6.1f}  {p.title[:52]:<54} "
                  f"{'/'.join(p.areas) or '-'}")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_JSONL.relative_to(ROOT)} ({len(pages)} pages)")

    # Record the source in the shared manifest so provenance for the whole
    # corpus lives in one place regardless of how a document arrived.
    manifest = (
        json.loads(MANIFEST.read_text(encoding="utf-8"))
        if MANIFEST.exists()
        else {"schema": 1, "documents": []}
    )
    manifest.setdefault("bulk_sources", [])
    manifest["bulk_sources"] = [
        s for s in manifest["bulk_sources"] if s.get("name") != "infonet-biovision"
    ]
    manifest["bulk_sources"].append({
        "name": "infonet-biovision",
        "archive": ARCHIVE.name,
        "archive_sha256_note": "not committed; verify locally with sha256sum",
        "pages_ingested": len(pages),
        "processed_file": str(OUT_JSONL.relative_to(ROOT)).replace("\\", "/"),
        **SOURCE_META,
    })
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"updated {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

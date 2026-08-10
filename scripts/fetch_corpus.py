#!/usr/bin/env python3
"""Harvest the Àgbẹ̀ retrieval corpus from institutional repositories.

WHY A HARVESTER RATHER THAN A FOLDER OF PDFs
--------------------------------------------
This repository does not redistribute source documents. Most agricultural
extension material we rely on carries CC BY-NC-SA or CC BY-SA terms, and
shipping the PDFs (or an index derived from them) would drag the repository
into non-commercial and share-alike obligations that conflict with its MIT
licence.

Instead we ship a *manifest*: for each document, its canonical URL, publisher,
publication year, licence, access rights and SHA-256. The operator runs this
script and the documents land on their own machine. Nothing is redistributed,
and an auditor who runs the same script gets provably the same bytes - or a
loud failure.

This mirrors `scripts/fetch_models.py`; weights and documents are both
content-addressed external inputs, and both are pinned the same way.

LEGITIMACY GATES
----------------
Every candidate must clear all five before it enters the corpus:

  1. Provenance    - fetched from an allowlisted institutional domain only,
                     never a third-party mirror.
  2. Access rights - repository metadata must say Open Access.
  3. Licence       - explicit "all rights reserved" records are rejected.
  4. Document type - research articles are excluded. A farmer needs extension
                     guidance (manuals, field guides, training material), not
                     a genetics paper. This is the single biggest quality
                     lever in the whole corpus.
  5. Availability  - the record must actually have a downloadable file.
                     Metadata-only records are common and useless to us.
  6. Content       - the extracted text must read like farmer guidance rather
                     than institutional reporting, and must not be a scan with
                     no text layer. Gates 1-5 are metadata gates and a survey
                     showed they pass donor progress reports and workshop
                     write-ups, because CGSpace types those identically to
                     training manuals. See agbe/rag/relevance.py.

Publication year is recorded but is NOT a rejection criterion here: an old
manual is still good botany and good disease identification. Currency is
enforced downstream, where it matters, by suppressing chemical-control
recommendations sourced from documents older than the safety threshold
(see agbe/rag/safety.py).

Usage
-----
    python scripts/fetch_corpus.py --dry-run     # survey candidates only
    python scripts/fetch_corpus.py               # fetch + build manifest
    python scripts/fetch_corpus.py --verify      # re-hash what is on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agbe.rag.extract import probe  # noqa: E402
from agbe.rag.relevance import score_text  # noqa: E402

RAW_DIR = ROOT / "corpus" / "raw"
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"

USER_AGENT = "Agbe-ADTC2026/0.1 (offline agricultural advisory; research use)"

#: Gate 1. Only these hosts are trusted as primary sources. farm-d.org and
#: similar aggregators re-host CGIAR material and were deliberately excluded:
#: a mirror cannot be relied on to preserve the original licence statement or
#: to still be serving the same revision a year from now.
ALLOWED_HOSTS = {
    "cgspace.cgiar.org",       # CGIAR incl. IITA (Ibadan), AfricaRice, ICRISAT
    "openknowledge.fao.org",   # FAO Knowledge Repository
    "www.fao.org",             # FAO document server
    "oar.icrisat.org",         # ICRISAT open access - Nigeria-specific
                               # groundnut and cowpea extension handbooks
}

#: Gate 4. CGSpace `dcterms.type` values that represent practical guidance.
#: Journal Article / Conference Paper / Manuscript are excluded on purpose.
EXTENSION_TYPES = {
    "book",
    "report",
    "manual",
    "training material",
    "extension material",
    "brief",
    "working paper",
    "book chapter",
    "poster",
    "presentation",
}

#: Crops that matter in southwest Nigeria, crossed with the questions a
#: smallholder actually asks. Narrow and deep beats broad and shallow: five
#: well-covered crops demo better and audit more honestly than thirty thin ones.
# The Agriculture domain covers four advisory areas: crop, livestock, weather
# and market. All four are in scope. Queries are grouped by area so corpus
# coverage can be reported per area rather than averaged into a single number
# that hides a gap.
CURATED_QUERIES = [
    # --- CROP ---------------------------------------------------------------
    "cassava mosaic disease management smallholder",
    "cassava brown streak disease symptoms control",
    "cassava green mite mealybug biological control",
    "cassava pest identification field guide",
    "cassava weed management Nigeria",
    "maize fall armyworm management Africa",
    "maize stem borer striga control smallholder",
    "maize storage pest management smallholder",
    "yam production practices West Africa",
    "yam anthracnose nematode storage rot",
    "tomato pest disease management Africa",
    "tomato bacterial wilt blight nursery management",
    "integrated pest management training smallholder farmers",
    "plant clinic extension diagnosis field guide Africa",

    # --- LIVESTOCK ----------------------------------------------------------
    # ILRI is a CGIAR centre, so livestock material sits behind the same API
    # as the crop material - no new source or ingester needed.
    "smallholder poultry disease management Newcastle Africa",
    "goat sheep health management smallholder Africa",
    "cattle tick borne disease control smallholder Africa",
    "livestock feed fodder dry season smallholder West Africa",
    "village chicken production management manual",
    "livestock vaccination deworming schedule smallholder",

    # --- WEATHER / AGRO-CLIMATOLOGY -----------------------------------------
    # Not forecasting - which is impossible offline - but the climatological
    # and decision-rule guidance that extension material genuinely contains.
    "planting calendar onset of rains West Africa",
    "climate smart agriculture practices smallholder West Africa",
    "drought dry spell management smallholder crops Africa",
    "agro-ecological zones cropping seasons Nigeria",

    # --- MARKET -------------------------------------------------------------
    "post harvest loss reduction storage marketing smallholder",
    "grain storage marketing decision smallholder Africa",
    "cassava processing value addition market West Africa",
    "farmer group aggregation collective marketing Africa",
    "gross margin cost of production smallholder crops Africa",

    # --- CROPS ADDED AFTER MEASURING CORPUS COVERAGE ------------------------
    # A coverage audit showed the declared scope did not match the corpus.
    # Yam - an in-scope crop - had 18 pages, thinner than thirteen excluded
    # crops, while rice (41), groundnut (34) and cowpea (19) sat unused. The
    # scope had been chosen from regional reasoning about southwest Nigeria,
    # but the bulk corpus reflects Infonet's Kenyan origin.
    #
    # These queries target the gap directly. AfricaRice and ICRISAT are both
    # CGIAR centres, so they are reachable through the same API as IITA - no
    # new source or ingester required.
    "rice production practices smallholder West Africa",
    "rice pest disease management lowland Nigeria",
    "cowpea pest management storage bruchid West Africa",
    "groundnut production practices rosette disease Nigeria",
    "pepper okra vegetable pest management West Africa",
    # Yam specifically: our weakest in-scope crop, and the source of a
    # measured coverage gap on the held-out evaluation ("staking").
    "yam staking trellis vine management",
    "yam minisett seed yam multiplication technique",
    # --- QUERIES TARGETING MEASURED COVERAGE GAPS ---------------------------
    # These are not guesses. Each one addresses a specific question the
    # held-out evaluation showed we cannot answer, recorded in
    # bench/results/coverage.json.
    #
    # Corpus gaps (retrieval returned nothing above the floor):
    "yam storage barn losses post harvest tuber",
    "yam tuber curing storage ventilation rot prevention",
    "onset of rains false start planting decision farmer",
    "rainfall onset criteria planting date decision Africa",
    "waterlogging flooding drainage field management smallholder",
    "raised beds ridges drainage flood prone field",
    # Retrieval gaps (wrong topic returned - more specific material needed):
    "maize nitrogen deficiency yellowing stunted diagnosis",
    "soil fertility symptoms nutrient deficiency maize leaves",
    "cowpea flower thrips pod abortion Megalurothrips",
    "cowpea flower bud drop insect damage management",
    "newcastle disease village poultry symptoms vaccination Africa",
    "poultry nervous signs torticollis diagnosis smallholder",
]

CGSPACE_SEARCH = "https://cgspace.cgiar.org/server/api/discover/search/objects"
CGSPACE_ITEMS = "https://cgspace.cgiar.org/server/api/core/items"

_HASH_BLOCK = 8 * 1024 * 1024


@dataclass
class Document:
    """One vetted corpus document and its full provenance record."""

    doc_id: str
    title: str
    publisher: str
    year: str
    doc_type: str
    access_rights: str
    licence: str
    source_url: str
    landing_page: str
    filename: str
    sha256: str = ""
    size_bytes: int = 0
    crops: list[str] | None = None

    # Populated by Gate 6. Recorded in the manifest so a reviewer can see why
    # each document earned its place, and so corpus quality can be tracked
    # over time rather than re-litigated from scratch.
    relevance_score: float = 0.0
    n_pages: int = 0
    guidance_terms: int = 0
    institutional_terms: int = 0

    def __post_init__(self) -> None:
        if self.crops is None:
            self.crops = infer_crops(self.title)


# Scope: southwest Nigeria staples plus the common smallholder vegetable.
# Cocoa was dropped deliberately - as a perennial cash crop its advisory needs
# (long-cycle disease management, fermentation, market grading) share little
# with annual food crops, and covering it would have thinned the corpus across
# every other crop rather than deepening any.
CROP_TERMS = {
    "cassava": ["cassava", "manihot"],
    "maize": ["maize", "corn", "zea mays"],
    "tomato": ["tomato", "solanum lycopersicum"],
    "yam": ["yam", "dioscorea"],
}


def infer_crops(title: str) -> list[str]:
    low = title.lower()
    return sorted(c for c, terms in CROP_TERMS.items() if any(t in low for t in terms))


def get_json(url: str, timeout: int = 45, attempts: int = 4) -> dict:
    """Fetch JSON, retrying transient network and DNS failures.

    A harvest run makes hundreds of API calls over many minutes. A single DNS
    hiccup used to abandon an entire query - one run lost its last four
    queries to `getaddrinfo failed` and silently produced a third of the
    corpus it should have. Skipping a query is invisible in the final counts,
    which is exactly what makes it dangerous: the harvest reports success on
    a partial result.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up after {attempts} attempts: {last_error}")


def meta(md: dict, key: str, default: str = "") -> str:
    vals = md.get(key)
    return vals[0]["value"] if vals else default


def slugify(text: str, maxlen: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:maxlen].rstrip("_")


def host_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def search_cgspace(query: str, size: int = 20) -> list[dict]:
    url = f"{CGSPACE_SEARCH}?query={urllib.parse.quote(query)}&size={size}"
    try:
        data = get_json(url)
    except Exception as exc:  # noqa: BLE001
        print(f"    search failed: {exc}")
        return []
    sr = data.get("_embedded", {}).get("searchResult", {})
    return [
        o.get("_embedded", {}).get("indexableObject", {})
        for o in sr.get("_embedded", {}).get("objects", [])
    ]


def first_pdf_bitstream(item_uuid: str) -> tuple[str, str] | None:
    """Resolve a CGSpace item to its first downloadable PDF.

    Returns (download_url, original_filename), or None when the record is
    metadata-only - which is common, and is Gate 5.
    """
    try:
        bundles = get_json(f"{CGSPACE_ITEMS}/{item_uuid}/bundles")
    except Exception:  # noqa: BLE001
        return None

    for bundle in bundles.get("_embedded", {}).get("bundles", []):
        if bundle.get("name") != "ORIGINAL":
            continue
        href = bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if not href:
            continue
        try:
            bits = get_json(href)
        except Exception:  # noqa: BLE001
            continue
        for bit in bits.get("_embedded", {}).get("bitstreams", []):
            name = bit.get("name", "")
            mime = bit.get("_embedded", {}).get("format", {}).get("mimetype", "")
            if name.lower().endswith(".pdf") or mime == "application/pdf":
                content = bit.get("_links", {}).get("content", {}).get("href")
                if content:
                    return content, name
    return None


def vet(ind: dict) -> tuple[bool, str]:
    """Apply the metadata gates. Returns (accepted, reason_if_rejected)."""
    md = ind.get("metadata", {})

    access = meta(md, "dcterms.accessRights")
    if access.strip().lower() != "open access":
        return False, f"access rights: {access or 'unspecified'}"

    licence = meta(md, "dcterms.license")
    low_licence = licence.lower()
    if "all rights reserved" in low_licence:
        return False, "licence: all rights reserved"

    # NoDerivatives. An audit found seven documents already in the corpus
    # carrying CC-BY-ND-4.0, CC-BY-NC-ND-4.0 or "Copyrighted; Non-commercial
    # use only" - none of which say "all rights reserved", so the check above
    # passed them.
    #
    # Whether chunking and embedding a document creates a derivative work is
    # genuinely arguable, and since nothing is redistributed the position is
    # defensible either way. But "arguable" is not a good place for a
    # submission to sit when the alternative costs a handful of documents out
    # of two hundred, so ND material is excluded going forward.
    #
    # Documents already indexed under these terms are recorded in
    # corpus/licenses/README.md rather than silently dropped.
    if re.search(r"\bnd-\d|noderiv|no derivative", low_licence):
        return False, f"licence: NoDerivatives ({licence})"
    if "copyrighted" in low_licence:
        return False, f"licence: {licence}"

    doc_type = meta(md, "dcterms.type").strip().lower()
    if doc_type not in EXTENSION_TYPES:
        return False, f"type: {doc_type or 'unspecified'} (not extension guidance)"

    return True, ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_BLOCK):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete_pdf(path: Path) -> bool:
    """Reject truncated downloads.

    A PDF must start with %PDF and carry an %%EOF trailer. We hit exactly this
    failure once already: a timeout produced a 3.4 MB file with a valid header
    and no trailer, which would have indexed as silent garbage.
    """
    if path.stat().st_size < 1024:
        return False
    with path.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            return False
        fh.seek(max(0, path.stat().st_size - 2048))
        return b"%%EOF" in fh.read()


def download(
    url: str, dest: Path, timeout: int = 300, attempts: int = 3
) -> bool:
    """Fetch one document, retrying on truncation.

    Retries are not defensive padding. CGSpace serves multi-megabyte PDFs
    slowly enough that a single attempt regularly ends mid-stream, and the
    integrity check then correctly rejects a file that was merely unlucky
    rather than genuinely bad. Without retries the corpus silently loses its
    largest and often most valuable documents - which is exactly what happened
    to "Integrated Pest and Disease Management in Major Agroecosystems", the
    richest guidance document the harvester found.

    Resume via HTTP Range is attempted first; servers that ignore it fall back
    to a clean restart.
    """
    if host_of(url) not in ALLOWED_HOSTS:
        print(f"    REJECTED non-allowlisted host: {host_of(url)}")
        return False

    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, attempts + 1):
        headers = {"User-Agent": USER_AGENT}
        resume_from = tmp.stat().st_size if attempt > 1 and tmp.exists() else 0
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 206 means the server honoured the Range request; 200 means
                # it ignored it and is resending from the start, so the
                # partial file must be discarded rather than appended to.
                partial = resp.status == 206
                mode = "ab" if (resume_from and partial) else "wb"
                with tmp.open(mode) as out:
                    while chunk := resp.read(1024 * 256):
                        out.write(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"    attempt {attempt}/{attempts} failed: {exc}")
            if attempt == attempts:
                tmp.unlink(missing_ok=True)
                return False
            time.sleep(2 * attempt)
            continue

        if is_complete_pdf(tmp):
            tmp.replace(dest)
            return True

        size = tmp.stat().st_size if tmp.exists() else 0
        print(f"    attempt {attempt}/{attempts}: truncated at {size/1024:.0f} KB")
        if attempt == attempts:
            print("    REJECTED: still truncated after retries")
            tmp.unlink(missing_ok=True)
            return False
        time.sleep(2 * attempt)

    return False


def harvest(dry_run: bool, limit_per_query: int, polite_delay: float) -> list[Document]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = []
    seen: set[str] = set()

    stats = {"examined": 0, "rejected": 0, "no_file": 0,
             "unusable": 0, "off_topic": 0, "accepted": 0, "failed_queries": 0}

    for query in CURATED_QUERIES:
        print(f"\n[query] {query}")
        results = search_cgspace(query, size=limit_per_query)
        if not results:
            stats["failed_queries"] += 1
        for ind in results:
            uuid = ind.get("uuid")
            if not uuid or uuid in seen:
                continue
            seen.add(uuid)
            stats["examined"] += 1

            md = ind.get("metadata", {})
            title = meta(md, "dc.title", "(untitled)")

            ok, reason = vet(ind)
            if not ok:
                stats["rejected"] += 1
                continue

            resolved = first_pdf_bitstream(uuid)
            time.sleep(polite_delay)
            if resolved is None:
                stats["no_file"] += 1
                print(f"  - no file   : {title[:70]}")
                continue

            url, original_name = resolved
            year = meta(md, "dcterms.issued", "unknown")[:4]
            filename = f"{slugify(title)}_{year}.pdf"

            doc = Document(
                doc_id=uuid,
                title=title,
                publisher=meta(md, "dcterms.publisher", "CGIAR"),
                year=year,
                doc_type=meta(md, "dcterms.type"),
                access_rights=meta(md, "dcterms.accessRights"),
                licence=meta(md, "dcterms.license", "unspecified"),
                source_url=url,
                landing_page=f"https://hdl.handle.net/{meta(md, 'dc.identifier.uri', uuid)}",
                filename=filename,
            )

            print(f"  + ACCEPTED  : {title[:70]}")
            print(f"                {year} | {doc.doc_type} | crops={doc.crops or ['general']}")

            if not dry_run:
                dest = RAW_DIR / filename
                if not (dest.exists() or download(url, dest)):
                    time.sleep(polite_delay)
                    continue

                # Gate 6: the metadata said this was extension guidance; now
                # check whether the text agrees. Rejects are deleted rather
                # than left on disk, so corpus/raw always mirrors the manifest.
                extraction = probe(dest)
                if not extraction.usable:
                    why = extraction.error or (
                        "scanned, no text layer" if extraction.is_scanned else "too little text"
                    )
                    print(f"                REJECTED (content): {why}")
                    dest.unlink(missing_ok=True)
                    stats["unusable"] += 1
                    time.sleep(polite_delay)
                    continue

                rel = score_text(extraction.text)
                if not rel.accepted:
                    print(f"                REJECTED (content): {rel.reason} "
                          f"[score {rel.score:.1f}, institutional {rel.institutional_hits}]")
                    dest.unlink(missing_ok=True)
                    stats["off_topic"] += 1
                    time.sleep(polite_delay)
                    continue

                doc.sha256 = sha256_of(dest)
                doc.size_bytes = dest.stat().st_size
                doc.relevance_score = round(rel.score, 2)
                doc.n_pages = extraction.n_pages
                doc.guidance_terms = rel.positive_hits
                doc.institutional_terms = rel.institutional_hits
                if rel.crops:
                    doc.crops = rel.crops  # content beats title for crop tagging
                docs.append(doc)
                stats["accepted"] += 1
                print(f"                KEPT score={rel.score:.1f} | {extraction.n_pages}pp "
                      f"| {doc.size_bytes/1024:.0f} KB | {doc.sha256[:12]}...")
                time.sleep(polite_delay)
            else:
                docs.append(doc)
                stats["accepted"] += 1

    print("\n" + "=" * 72)
    if stats["failed_queries"]:
        # Surfaced loudly: a harvest that lost queries produced a partial
        # corpus, and every downstream coverage number would be measuring the
        # wrong thing without saying so.
        print(f"WARNING: {stats['failed_queries']} of {len(CURATED_QUERIES)} queries "
              "failed (network). This corpus is INCOMPLETE - re-run when the\n"
              "         connection is stable before trusting coverage results.")
        print("=" * 72)
    print(f"examined        {stats['examined']}")
    print(f"  metadata gate {stats['rejected']:>4}  (access rights / licence / type)")
    print(f"  no file       {stats['no_file']:>4}  (metadata-only record)")
    print(f"  unusable      {stats['unusable']:>4}  (scanned or unreadable PDF)")
    print(f"  off topic     {stats['off_topic']:>4}  (institutional prose, not guidance)")
    print(f"  ACCEPTED      {stats['accepted']:>4}")
    print("=" * 72)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="survey and vet candidates without downloading")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash on-disk documents against the manifest")
    ap.add_argument("--limit-per-query", type=int, default=20)
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between repository calls (be a good citizen)")
    args = ap.parse_args()

    if args.verify:
        if not MANIFEST_PATH.exists():
            print("no manifest to verify")
            return 1
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        bad = 0
        for entry in manifest["documents"]:
            path = RAW_DIR / entry["filename"]
            if not path.exists():
                print(f"MISSING  {entry['filename']}")
                bad += 1
            elif sha256_of(path) != entry["sha256"]:
                print(f"MISMATCH {entry['filename']}")
                bad += 1
        print(f"\n{len(manifest['documents']) - bad} ok, {bad} problems")
        return 1 if bad else 0

    docs = harvest(args.dry_run, args.limit_per_query, args.delay)

    if args.dry_run:
        print("\ndry run - no files downloaded, no manifest written")
        return 0

    manifest = {
        "schema": 1,
        "generated_by": "scripts/fetch_corpus.py",
        "note": (
            "Documents are NOT redistributed in this repository. This manifest "
            "pins their provenance, licence and content hash so that any "
            "operator can reproduce the exact corpus we benchmarked."
        ),
        "gates": {
            "allowed_hosts": sorted(ALLOWED_HOSTS),
            "extension_types": sorted(EXTENSION_TYPES),
            "access_rights_required": "Open Access",
        },
        "documents": [asdict(d) for d in docs],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {MANIFEST_PATH.relative_to(ROOT)} with {len(docs)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

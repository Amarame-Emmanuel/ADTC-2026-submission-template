#!/usr/bin/env python3
"""Export the Nigerian Pidgin evaluation set to a spreadsheet, and read it back.

A reviewer should receive something they can open in Excel or Google Sheets,
not JSON: escaped characters, brittle syntax, and one missing comma destroys
the file silently.

The difference is that this sheet carries DRAFT translations rather than blank
fields. Pidgin is close enough to English that a first attempt is worth making,
and correcting a draft is much faster than composing from nothing.

That convenience carries a specific risk, so the sheet says it plainly: a draft
left unreviewed becomes a reference translation nobody wrote. The `draft_ok`
column exists to force an explicit judgement on every row rather than letting
silence count as approval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "bench" / "pidgin_eval.json"
DEFAULT_CSV = ROOT / "bench" / "pidgin_review.csv"

# Four columns, not eight.
#
# The first version carried id, priority, purpose, english, draft, ok, corrected
# and notes - and was unreadable in Excel, which does not wrap long text by
# default, so seven-line safety messages collapsed into one clipped row. A
# reviewer who cannot read the sheet cannot review it.
#
# Priority is folded into the id (CRITICAL rows are prefixed), purpose and notes
# are dropped. Only what the reviewer must read and what they must type remain.
COLUMNS = [
    "id",
    "english",
    "draft_pidgin",
    "ok_or_your_version",
]


def export(path: Path) -> None:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    rows: list[dict] = [{
        "id": "HOW-TO-USE",
        "english": (
            "In the last column: type OK if the Pidgin is right. If it is wrong, "
            "type how YOU would say it instead. Leave blank if unsure. Rows "
            "starting with CRITICAL are safety warnings - do those first."
        ),
        "draft_pidgin": "",
        "ok_or_your_version": "",
    }]

    for item in data["safety_messages"] + data["advisory_samples"]:
        prefix = "CRITICAL " if item["priority"] == "CRITICAL" else ""
        rows.append({
            "id": f"{prefix}{item['id']}",
            "english": item["english"],
            "draft_pidgin": item["draft"],
            "ok_or_your_version": item.get("corrected", ""),
        })

    for term in data["terminology"]:
        rows.append({
            "id": f"word-{term['english'].replace(' ', '-').replace('/', '')}",
            "english": term["english"],
            "draft_pidgin": term["draft"],
            "ok_or_your_version": term.get("corrected", ""),
        })

    # utf-8-sig so Excel on Windows renders the text correctly rather than as
    # mojibake the reviewer would then "fix" into something wrong.
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    critical = sum(1 for r in rows if r["id"].startswith("CRITICAL"))
    print(f"wrote {path}")
    print(f"  {len(rows) - 1} rows ({critical} CRITICAL safety messages)")
    print("\nOpens in Excel or Google Sheets. Reviewer fills 'draft_ok_yes_no'")
    print("and, where needed, 'your_version_if_not_ok'.")


def import_csv(path: Path) -> None:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    filled = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["id"] and row["id"] != "READ-ME":
                filled[row["id"]] = row

    def _resolve(row: dict, draft: str) -> tuple[str, str] | None:
        """Interpret one filled cell.

        A single column carries two meanings: "OK" approves the draft, anything
        else replaces it. That is less precise than separate columns but it is
        what a reviewer will actually fill in, and an unusable sheet gets no
        review at all.
        """
        value = (row.get("ok_or_your_version") or "").strip()
        if not value:
            return None
        if value.lower() in {"ok", "okay", "yes", "y", "correct", "fine", "good"}:
            return draft, "approved"
        return value, "rewritten"

    reviewed = approved = corrected = 0
    for item in data["safety_messages"] + data["advisory_samples"]:
        prefix = "CRITICAL " if item["priority"] == "CRITICAL" else ""
        row = filled.get(f"{prefix}{item['id']}") or filled.get(item["id"])
        if not row:
            continue
        resolved = _resolve(row, item["draft"])
        if not resolved:
            continue
        item["corrected"], kind = resolved
        item["draft_ok"] = "yes" if kind == "approved" else "no"
        reviewed += 1
        approved += kind == "approved"
        corrected += kind == "rewritten"

    for term in data["terminology"]:
        key = f"word-{term['english'].replace(' ', '-').replace('/', '')}"
        row = filled.get(key)
        if row:
            resolved = _resolve(row, term["draft"])
            if resolved:
                term["corrected"] = resolved[0]

    EVAL_JSON.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total = len(data["safety_messages"]) + len(data["advisory_samples"])
    print(f"merged into {EVAL_JSON.relative_to(ROOT)}")
    print(f"  reviewed {reviewed}/{total}  ({approved} drafts approved, "
          f"{corrected} rewritten)")
    unreviewed = total - reviewed
    if unreviewed:
        print(f"  {unreviewed} still unreviewed - these are EXCLUDED from the "
              "validated claim rather than counted as passing")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--import-csv", type=str, metavar="PATH")
    ap.add_argument("--out", type=str, default=str(DEFAULT_CSV))
    args = ap.parse_args()

    if args.import_csv:
        import_csv(Path(args.import_csv))
    elif args.export:
        export(Path(args.out))
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

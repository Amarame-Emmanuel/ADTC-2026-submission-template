#!/usr/bin/env python3
"""Native-speaker review sheets for any supported language.

Generalises the Pidgin flow (scripts/pidgin_sheet.py, now a wrapper around
this) so Yoruba and Igbo reviews use the same tooling instead of a copy. Two
copies of a workflow drift - this project has now measured that four times
with code paths - so the export/import logic lives here once.

THE FLOW, PER LANGUAGE
----------------------
  1. `--init yor`   builds bench/yoruba_eval.json: every fixed safety string
                    the system can emit, plus the advisory samples and
                    terminology list shared with the Pidgin review. English is
                    taken from agbe/translate/messages.py - the single source
                    - never retyped.
  2. `--export yor` writes a CSV a reviewer opens in Excel or Google Sheets.
  3. reviewer fills one column: OK, or their own version.
  4. `--import-csv` merges the review back; unreviewed rows are EXCLUDED from
                    any validated claim rather than counted as passing.
  5. validated strings are then added to messages.py BY HAND, with provenance.
     Nothing ships from the JSON directly: messages.py stays the one place
     that defines what the system can say, and adding a language there is a
     deliberate act with a name and date attached.

WHY THE DRAFT COLUMN IS EMPTY FOR YORUBA AND IGBO
-------------------------------------------------
The Pidgin sheet shipped machine-assisted drafts because Pidgin is close
enough to English that correcting a draft beats composing from nothing. For
Yoruba and Igbo the only draft generator available is NLLB, and we measured
what it does to exactly these sentences (REPORT §3.2): "Do not use this
pesticide. It is banned." came back from the Yoruba round trip as "Do not use
this antibiotic" - prohibition clause deleted, chemical class changed - and
"cassava" became "àlìkámà" (wheat) in Yoruba and "maniyyi" (semen) in Hausa.

A wrong draft is worse than an empty cell: reviewers anchor on drafts, and an
error that survives anchoring looks deliberate. So the reviewer composes from
the English. Slower, and correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LANGUAGES = {
    "pcm": ("Nigerian Pidgin", "pidgin_eval.json"),
    "yor": ("Yoruba", "yoruba_eval.json"),
    "ibo": ("Igbo", "igbo_eval.json"),
}

#: The Pidgin file doubles as the source of the language-neutral English
#: advisory samples and terminology list, so every language reviews the same
#: content and results stay comparable.
PIDGIN_EVAL = ROOT / "bench" / "pidgin_eval.json"

COLUMNS = ["id", "english", "draft", "ok_or_your_version"]

#: Field-name -> reviewer-facing purpose line, for the safety strings drawn
#: from MessageSet. CRITICAL rows are the ones a wrong translation can hurt
#: someone with.
SAFETY_PURPOSE = {
    "no_guidance": ("CRITICAL", "Refusal - shown when the corpus cannot answer"),
    "hazardous_pesticide": ("CRITICAL", "Warning - a cited pesticide is hazardous/banned"),
    "unsupported_dosage": ("CRITICAL", "A dosage in the answer is not in the sources"),
    "stale_chemical": ("CRITICAL", "Chemical advice comes only from old documents"),
    "dosage_refusal": ("CRITICAL", "Refusal to state any pesticide/drug dose"),
    "withdrawal_period": ("CRITICAL", "Veterinary withdrawal period - milk/meat safety"),
    "human_medical": ("CRITICAL", "Redirect to a clinic - human health symptom"),
    "out_of_scope_crop": ("normal", "Polite refusal for crops we do not cover"),
}


def eval_path(lang: str) -> Path:
    return ROOT / "bench" / LANGUAGES[lang][1]


def init(lang: str) -> None:
    """Build (or refresh) the eval JSON skeleton for a language.

    Refreshing preserves any review already merged: rows are matched by their
    English text, so re-running --init after messages.py gains a string adds
    the new row without discarding a reviewer's work.
    """
    if lang == "pcm":
        raise SystemExit("pcm already exists and is validated; refusing to touch it")

    from agbe.translate.messages import MESSAGES

    english = MESSAGES["en"]
    pidgin = json.loads(PIDGIN_EVAL.read_text(encoding="utf-8"))

    existing: dict[str, dict] = {}
    path = eval_path(lang)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for section in ("safety_messages", "advisory_samples", "terminology"):
            for item in old.get(section, []):
                existing[item["english"]] = item

    def carry(en_text: str, base: dict) -> dict:
        prior = existing.get(en_text)
        if prior:
            base["draft"] = prior.get("draft", "")
            if "corrected" in prior:
                base["corrected"] = prior["corrected"]
            if "draft_ok" in prior:
                base["draft_ok"] = prior["draft_ok"]
        return base

    safety = []
    for i, f in enumerate(fields(english), start=1):
        priority, purpose = SAFETY_PURPOSE[f.name]
        safety.append(carry(getattr(english, f.name), {
            "id": f"{lang}-s{i:02d}",
            "priority": priority,
            "purpose": purpose,
            "english": getattr(english, f.name),
            "draft": "",
        }))

    advisory = [
        carry(item["english"], {
            "id": f"{lang}-a{i:02d}",
            "priority": item.get("priority", "normal"),
            "purpose": item.get("purpose", ""),
            "english": item["english"],
            "draft": "",
        })
        for i, item in enumerate(pidgin["advisory_samples"], start=1)
    ]
    terminology = [
        carry(term["english"], {"english": term["english"], "draft": ""})
        for term in pidgin["terminology"]
    ]

    data = {
        "schema": pidgin.get("schema", 1),
        "language": lang,
        "language_name": LANGUAGES[lang][0],
        "purpose": (
            "Native-speaker validation set. Strings here reach farmers "
            "verbatim once validated and added to agbe/translate/messages.py. "
            "Unreviewed rows are excluded from any claim."
        ),
        "instructions_for_reviewer": (
            "Fill the last column of the exported CSV: OK approves, anything "
            "else replaces. Draft cells are EMPTY on purpose - machine drafts "
            "were measured deleting prohibition clauses in this language pair "
            "and are not offered. Compose from the English."
        ),
        "safety_messages": safety,
        "advisory_samples": advisory,
        "terminology": terminology,
        "review": {"status": "unreviewed", "reviewer": None, "date": None},
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    print(f"  {len(safety)} safety ({sum(1 for s in safety if s['priority']=='CRITICAL')}"
          f" CRITICAL), {len(advisory)} advisory, {len(terminology)} terms")


def export(lang: str, out: Path | None) -> None:
    data = json.loads(eval_path(lang).read_text(encoding="utf-8"))
    out = out or ROOT / "bench" / f"{LANGUAGES[lang][0].lower().replace(' ', '_')}_review.csv"

    rows = [{
        "id": "HOW-TO-USE",
        "english": (
            "In the last column: type OK if the draft is right. If it is wrong "
            "or empty, type how YOU would say it. Leave blank if unsure. Rows "
            "starting with CRITICAL are safety warnings - do those first."
        ),
        "draft": "", "ok_or_your_version": "",
    }]
    for item in data["safety_messages"] + data["advisory_samples"]:
        prefix = "CRITICAL " if item["priority"] == "CRITICAL" else ""
        rows.append({
            "id": f"{prefix}{item['id']}",
            "english": item["english"],
            "draft": item.get("draft", ""),
            "ok_or_your_version": item.get("corrected", ""),
        })
    for term in data["terminology"]:
        rows.append({
            "id": f"word-{term['english'].replace(' ', '-').replace('/', '')}",
            "english": term["english"],
            "draft": term.get("draft", ""),
            "ok_or_your_version": term.get("corrected", ""),
        })

    # utf-8-sig so Excel renders diacritics instead of mojibake the reviewer
    # would then "fix" into something wrong.
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def import_csv(lang: str, path: Path) -> None:
    data = json.loads(eval_path(lang).read_text(encoding="utf-8"))
    filled = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["id"] and row["id"] != "HOW-TO-USE":
                filled[row["id"]] = row

    def resolve(row: dict, draft: str):
        value = (row.get("ok_or_your_version") or "").strip()
        if not value:
            return None
        if value.lower() in {"ok", "okay", "yes", "y", "correct", "fine", "good"}:
            # Approving an EMPTY draft is meaningless; treat as unreviewed.
            return (draft, "approved") if draft else None
        return value, "rewritten"

    reviewed = 0
    for item in data["safety_messages"] + data["advisory_samples"]:
        prefix = "CRITICAL " if item["priority"] == "CRITICAL" else ""
        row = filled.get(f"{prefix}{item['id']}") or filled.get(item["id"])
        result = resolve(row, item.get("draft", "")) if row else None
        if result:
            item["corrected"], kind = result
            item["draft_ok"] = "yes" if kind == "approved" else "no"
            reviewed += 1
    for term in data["terminology"]:
        row = filled.get(f"word-{term['english'].replace(' ', '-').replace('/', '')}")
        result = resolve(row, term.get("draft", "")) if row else None
        if result:
            term["corrected"] = result[0]

    eval_path(lang).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = len(data["safety_messages"]) + len(data["advisory_samples"])
    print(f"merged: {reviewed}/{total} reviewed; unreviewed rows stay excluded")
    if reviewed:
        print("next: add validated strings to agbe/translate/messages.py with "
              "reviewer name and date, then add the language to VALIDATED_LANGUAGES")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", choices=sorted(LANGUAGES), required=True)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--import-csv", type=str, metavar="PATH")
    ap.add_argument("--out", type=str)
    args = ap.parse_args()

    if args.init:
        init(args.language)
    elif args.export:
        export(args.language, Path(args.out) if args.out else None)
    elif args.import_csv:
        import_csv(args.language, Path(args.import_csv))
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

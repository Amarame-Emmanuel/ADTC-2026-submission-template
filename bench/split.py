"""Deterministic dev/test split of the evaluation questions.

WHY THIS EXISTS EVEN THOUGH NOTHING IS TRAINED
----------------------------------------------
This system trains no model, so a model cannot memorise the evaluation set.
The overfitting risk is real anyway, and it sits one level up: in the
*hyperparameters chosen by hand*.

`min_score`, `top_k`, `chunk_tokens`, `context_token_budget`,
`MAX_ELSEWHERE_RATIO`, `NEAR_DUPLICATE_JACCARD` and the rest were all set by
judgement. Tuning them until the coverage number looks good, and then reporting
that same number, measures nothing except how long the tuning went on. It is
the classic test-set-contamination failure wearing different clothes.

So the questions are split once, deterministically:

  DEV  - tune against these freely. Inspect them, chase individual failures,
         adjust thresholds, iterate as much as is useful.
  TEST - never inspected while tuning. Run once, at the end, and report that.

Any number quoted in REPORT.md comes from TEST.

WHY HASH-BASED AND NOT RANDOM
-----------------------------
The split is derived from a hash of each question id, so it is identical on
every machine and every run without storing a seed or a split file that could
drift out of sync with the questions. Adding a question later assigns it to a
side without disturbing anything already assigned - which a shuffled split
would not guarantee.

Stratified by advisory area so that neither side loses a whole area: with only
6 weather questions, a naive 50/50 could easily leave one side with none.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"

#: Fraction assigned to dev. Deliberately larger than the usual split: the
#: dev side is where iteration happens and it needs enough per-area signal to
#: be useful, while the test side only has to support a single final number.
DEV_FRACTION = 0.5

#: Changing this reshuffles the split, which invalidates every previously
#: reported number. It exists so a reshuffle is an explicit, visible act rather
#: than something that happens by accident.
SPLIT_SALT = "agbe-2026-v1"


def _bucket(question_id: str) -> float:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{question_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def split_questions(questions: list[dict] | None = None) -> dict[str, list[dict]]:
    """Return {"dev": [...], "test": [...]}, stratified by area and scope."""
    questions = questions if questions is not None else load_questions()

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for q in questions:
        strata[(q.get("area", "general"), q["in_scope"])].append(q)

    dev: list[dict] = []
    test: list[dict] = []
    for group in strata.values():
        # Sort by hash so assignment within a stratum is stable and balanced
        # rather than dependent on file order.
        ordered = sorted(group, key=lambda q: _bucket(q["id"]))
        cut = round(len(ordered) * DEV_FRACTION)
        dev.extend(ordered[:cut])
        test.extend(ordered[cut:])

    dev.sort(key=lambda q: q["id"])
    test.sort(key=lambda q: q["id"])
    return {"dev": dev, "test": test}


def summarise() -> str:
    s = split_questions()
    lines = ["evaluation split (stratified by area and scope)", ""]
    for name in ("dev", "test"):
        qs = s[name]
        in_scope = [q for q in qs if q["in_scope"]]
        by_area: dict[str, int] = defaultdict(int)
        for q in in_scope:
            by_area[q.get("area", "general")] += 1
        areas = "  ".join(f"{a}={n}" for a, n in sorted(by_area.items()))
        lines.append(
            f"  {name.upper():<5} {len(qs):>3} questions "
            f"({len(in_scope)} in-scope, {len(qs) - len(in_scope)} refusal)   {areas}"
        )
    lines += [
        "",
        "  DEV  is for tuning. TEST is run once and is what REPORT.md quotes.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise())

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

THE VERSION THAT DID NOT KEEP THAT PROMISE
------------------------------------------
The paragraph above was true of the intent and false of the code. Assignment
used to be by POSITION within a hash-sorted stratum:

    ordered = sorted(group, key=lambda q: _bucket(q["id"]))
    cut = round(len(ordered) * DEV_FRACTION)
    dev.extend(ordered[:cut]); test.extend(ordered[cut:])

The hash fixed the ORDER; the cut point moved with the size of the stratum. So
adding one question shifted the boundary and could push an existing question
across it. Measured: adding 8 out-of-scope questions moved oos-05 and oos-08
from TEST to DEV.

That is the worst kind of defect for this file, because the failure is silent
and retroactive. Every previously published dev and test number would have
started referring to a different set of questions than the one that produced
it, with nothing in the output to say so - and SPLIT_SALT exists precisely so
that a reshuffle is "an explicit, visible act rather than something that
happens by accident".

Assignment is now by hash THRESHOLD, which cannot move:

    dev if _bucket(q["id"]) < DEV_FRACTION else test

A question's side is a property of its own id and DEV_FRACTION alone. No other
question can affect it, so the set can be extended freely - which is what
unblocks writing the framed-question variants (REPORT.md 7.11) and the extra
out-of-scope questions that refusal accuracy needs to have any resolution.

THE COST, STATED
----------------
Thresholding gives up EXACT stratum balance. Positional cutting guaranteed
round(n * 0.5) per stratum; a threshold gives that only in expectation, so a
6-question area might split 4/2 rather than 3/3.

Stratification is still applied and still does its job - it is what stops an
area landing entirely on one side - but the balance is now approximate. At
these sizes that is the right trade: a guarantee that holds is worth more than
a balance that is exact, and the alternative was a file whose central promise
was untrue.

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
        # Threshold, not position. A question's side depends only on its own id
        # and DEV_FRACTION, so adding questions cannot move existing ones. See
        # the module docstring for the version that got this wrong and the
        # measurement that caught it.
        #
        # Sorted by hash first so the two sides come out in a stable order
        # regardless of file order - cosmetic, but it keeps diffs of any
        # printed split readable.
        for q in sorted(group, key=lambda q: _bucket(q["id"])):
            (dev if _bucket(q["id"]) < DEV_FRACTION else test).append(q)

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

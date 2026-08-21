"""Answer-level evaluation - does the ANSWER say the right thing?

WHY THIS EXISTS
---------------
`bench/coverage.py` measures whether a relevant passage was *retrieved*. That
is not the same as whether the farmer was told the right thing, and the gap
between them is where this system's worst failures have lived.

The submitted test prompt is the proof. It answered "Cassava Brown Streak
Disease" for textbook cassava mosaic symptoms - advice that would cost a farmer
their crop - while coverage reported 100% on both splits, before and after.
Coverage could not see it, because a passage containing "mosaic" *was*
retrieved; the answer simply did not use it.

The same blindness ran through a day of debugging: dev coverage sat at 93.3%
through two changes that made answers worse, and at 96.7% before and after the
change that finally made them right. A metric that does not move when the
product gets better or worse is not measuring the product.

THE FOUR QUADRANTS
------------------
Each in-scope question carries `expect_any` terms. Checking them against the
retrieved passages AND against the generated answer separates failures that
coverage alone reports identically:

                    answer HAS a term      answer LACKS one
    retrieved  ->   OK                     NOT USED
    not retr.  ->   UNGROUNDED             MISSED

  OK         - retrieval found it, the model used it.
  NOT USED   - the evidence was in the context and the model ignored it. A
               generation problem: prompt, ordering, or compression.
  MISSED     - neither found it. A corpus or retrieval problem.
  UNGROUNDED - the model produced an expected term that appears in NO retrieved
               passage. It came from parametric memory, which is precisely what
               section 5 promises cannot happen. Measured at 3B: "plant
               disease-free cassava seedlings" (cassava grows from stem
               cuttings) and "avoid overhead irrigation to reduce whiteflies",
               neither in any source.

CONTRADICTED, the fifth verdict, outranks all of them including OK.

`expect_any` is satisfied by ANY expected term, and that is what hid the worst
answer this project has produced. Asked about textbook cassava mosaic symptoms,
the system named Cassava Brown Streak Disease - the wrong disease, and the
single defect this project has spent the most effort on - while also mentioning
whitefly, an expected term, drawn from a passage about whiteflies. Verdict: OK.
Answer accuracy: 93.9%. UNGROUNDED: 0. Every metric green.

`reject_any` is a term the answer must NOT contain. It cannot verify a claim,
which is what a real claim-level check would do and what nobody here built. It
catches the specific confusion a question is known to invite - brown streak for
mosaic - which is where the harm actually is.

UNGROUNDED is the reason this file reports four numbers rather than one. It is
the only automated check in the project that can catch the grounding guarantee
being broken, and a rising count is a regression even when accuracy improves.

WHAT IT IS NOT
--------------
`expect_any` is a keyword proxy for correctness, not a judgement of it. An
answer can contain "mosaic" and still be poor, and a good answer can use
different words. It is a floor, not a ceiling - useful because it is automatic
and repeatable, and no substitute for reading answers, which is what found the
defect this file exists to have caught.

COST
----
One generation per question, so ~20s each against ~70ms for coverage. That is
why this is a separate tool: coverage stays cheap enough to run constantly,
and this runs when answers might have changed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _terms_present(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t.lower() in low]


def evaluate(split: str = "dev", limit: int | None = None,
             questions_path: str | None = None) -> dict:
    from agbe.advisor import AdvisoryEngine
    from bench.coverage import load_questions

    questions = load_questions(split, questions_path)
    in_scope = [q for q in questions if q.get("in_scope")]
    out_scope = [q for q in questions if not q.get("in_scope")]
    if limit:
        in_scope = in_scope[:limit]

    engine = AdvisoryEngine()
    rows: list[dict] = []
    counts: defaultdict[str, int] = defaultdict(int)
    started = time.perf_counter()

    for i, q in enumerate(in_scope, 1):
        expect = q.get("expect_any") or []
        reject = q.get("reject_any") or []
        advice = engine.advise(q["question"])
        retrieved_text = " ".join(h.chunk.text for h in advice.hits)

        in_answer = _terms_present(advice.answer, expect)
        in_sources = _terms_present(retrieved_text, expect)
        contradicted = _terms_present(advice.answer, reject)

        # CONTRADICTED outranks every other verdict, including OK.
        #
        # `expect_any` is satisfied by ANY expected term, and that is what hid
        # this. Asked about textbook cassava mosaic symptoms, the system named
        # Cassava Brown Streak Disease - the wrong disease, the one defect this
        # project has spent the most effort on - while mentioning whitefly,
        # which is an expected term, from a passage about whiteflies. Verdict:
        # OK. Answer accuracy 93.9%. UNGROUNDED 0.
        #
        # A term the answer must NOT say is the cheapest form of the
        # claim-level check this project has repeatedly wanted and not built.
        # It cannot verify a claim; it can catch the specific confusion that a
        # question is known to invite, which is where the harm actually is -
        # brown streak for mosaic, coccidiosis for Newcastle.
        if contradicted:
            verdict = "CONTRADICTED"
        elif in_answer and in_sources:
            verdict = "OK"
        elif in_sources:
            verdict = "NOT_USED"
        elif in_answer:
            verdict = "UNGROUNDED"
        else:
            verdict = "MISSED"

        counts[verdict] += 1
        rows.append({
            "id": q["id"],
            "area": q.get("area"),
            "crop": q.get("crop"),
            "question": q["question"],
            "verdict": verdict,
            "refused": advice.refused,
            "expect_any": expect,
            "reject_any": reject,
            "terms_in_answer": in_answer,
            "terms_in_sources": in_sources,
            "contradicting_terms": contradicted,
            "answer_head": " ".join(advice.answer.split())[:220],
        })
        print(f"  {i}/{len(in_scope)}  {q['id']:<9} {verdict}", flush=True)

    refused_ok = 0
    for q in out_scope:
        advice = engine.advise(q["question"])
        if advice.refused:
            refused_ok += 1
        else:
            rows.append({
                "id": q["id"], "area": q.get("area"), "question": q["question"],
                "verdict": "NOT_REFUSED", "refused": False,
                "answer_head": " ".join(advice.answer.split())[:220],
            })

    n = len(in_scope)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "n_in_scope": n,
        "n_out_scope": len(out_scope),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "answer_accuracy": round(counts["OK"] / n, 4) if n else 0.0,
        "counts": {
            "OK": counts["OK"],
            "NOT_USED": counts["NOT_USED"],
            "MISSED": counts["MISSED"],
            "UNGROUNDED": counts["UNGROUNDED"],
            "CONTRADICTED": counts["CONTRADICTED"],
        },
        "refusal": {
            "refused": refused_ok,
            "total": len(out_scope),
            "value": round(refused_ok / len(out_scope), 4) if out_scope else 1.0,
        },
        "rows": rows,
    }


def print_report(r: dict) -> None:
    c = r["counts"]
    n = r["n_in_scope"]
    print("\n" + "=" * 72)
    print("ANSWER-LEVEL EVALUATION")
    print("=" * 72)
    print(f"  split         {r['split']}   {n} in-scope, {r['n_out_scope']} out-of-scope"
          f"   ({r['elapsed_s']}s)")
    print()
    print(f"  ANSWER ACC    {r['answer_accuracy'] * 100:.1f}%  ({c['OK']}/{n} answered "
          f"with an expected term)")
    print()
    print(f"    OK          {c['OK']:>3}   retrieved and used")
    print(f"    NOT_USED    {c['NOT_USED']:>3}   evidence was in context, answer ignored it")
    print(f"    MISSED      {c['MISSED']:>3}   neither retrieved nor answered")
    print(f"    UNGROUNDED  {c['UNGROUNDED']:>3}   answer asserts what NO source contains")
    print(f"    CONTRAD.    {c['CONTRADICTED']:>3}   answer names something the question rules out")
    print()
    print(f"  REFUSAL       {r['refusal']['value'] * 100:.1f}%  "
          f"({r['refusal']['refused']}/{r['refusal']['total']})")

    if c["CONTRADICTED"]:
        print("\n  CONTRADICTED - the answer named something this question rules out.")
        print("  Ranked above OK: an answer can carry an expected term and still")
        print("  name the wrong disease, which is how this went unseen at 93.9%.")
        for row in r["rows"]:
            if row["verdict"] == "CONTRADICTED":
                print(f"    [{row['id']}] {row['question'][:66]}")
                print(f"        said {row['contradicting_terms']} - ruled out by reject_any")

    if c["UNGROUNDED"]:
        print("\n  UNGROUNDED - the grounding guarantee in section 5 is being broken here:")
        for row in r["rows"]:
            if row["verdict"] == "UNGROUNDED":
                print(f"    [{row['id']}] {row['question'][:66]}")
                print(f"        asserted {row['terms_in_answer']} - in no retrieved passage")

    if c["NOT_USED"]:
        print("\n  NOT_USED - context had the evidence, the model did not use it:")
        for row in r["rows"]:
            if row["verdict"] == "NOT_USED":
                print(f"    [{row['id']}] {row['question'][:66]}")
                print(f"        sources had {row['terms_in_sources']}")

    if c["MISSED"]:
        print("\n  MISSED - corpus or retrieval gap:")
        for row in r["rows"]:
            if row["verdict"] == "MISSED":
                print(f"    [{row['id']}] {row['question'][:66]}")

    not_refused = [x for x in r["rows"] if x["verdict"] == "NOT_REFUSED"]
    if not_refused:
        print("\n  NOT REFUSED - out-of-scope questions that were answered:")
        for row in not_refused:
            print(f"    [{row['id']}] {row['question'][:66]}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                    help="dev for iterating; test once, at the end")
    ap.add_argument("--limit", type=int, help="first N in-scope questions only")
    ap.add_argument("--questions",
                    help="alternative question file; read whole, ignoring --split")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default="answers.json")
    args = ap.parse_args()

    result = evaluate(split=args.split, limit=args.limit,
                      questions_path=args.questions)
    print_report(result)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / args.out
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

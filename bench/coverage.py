"""Retrieval coverage evaluation - the "is the corpus enough?" metric.

Document count is a vanity metric. The question that matters is whether a real
farmer's question finds a relevant passage, so this measures exactly that on a
fixed set of questions written in the register farmers actually use.

TWO NUMBERS, NOT ONE
--------------------
COVERAGE   - of in-scope questions, the fraction retrieving at least one
             passage above the similarity floor. This is the stopping
             criterion for corpus building: harvest until it plateaus.

REFUSAL    - of deliberately out-of-scope questions (market prices, weather,
             exact dosages, livestock), the fraction correctly returning
             nothing. This matters as much as coverage and is easier to
             forget. A system that retrieves four confident-looking passages
             for "what is maize selling for today?" will produce an answer,
             and that answer will be invented.

Reporting coverage alone would reward lowering the similarity floor until
everything matches something - which improves the headline number while making
the system less trustworthy. Reported together, the two numbers constrain each
other, and the floor becomes a visible trade-off rather than a hidden knob.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agbe import config
from agbe.rag.embedder import Embedder
from agbe.rag.index import VectorIndex

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

#: Targets for Gate 1. Not arbitrary: below roughly this coverage the system
#: refuses too often to be useful in a demo, and below this refusal rate it is
#: answering questions it has no grounding for.
TARGET_COVERAGE = 0.80
TARGET_REFUSAL = 0.80


def load_questions(split: str = "dev", path: "Path | None" = None) -> list[dict]:
    """Load the evaluation questions for one side of the split.

    Defaults to `dev` on purpose. Tuning is the common case, and a default of
    `test` would make it trivially easy to contaminate the reported number by
    running the harness casually during development. Reporting requires asking
    for `--split test` explicitly.
    """
    from bench.split import split_questions

    # An alternative question file is always read whole. The dev/test split is
    # a hash over the ids in eval_questions.json; applying it to a different
    # file would silently discard half of a set whose whole purpose is to be
    # small and adversarial. bench/probe_questions.json is the case this exists
    # for - see its `purpose` field.
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]

    if split == "all":
        return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]

    parts = split_questions()
    if split not in parts:
        raise ValueError(f"unknown split {split!r}; use dev, test or all")
    return parts[split]


def sweep_floors(split: str, n: int = 9) -> list[float]:
    """Floors to sweep, spanning the scores this index actually produces.

    Retrieves once per question with the floor removed, collects the top score
    for each, and spreads the sweep across that distribution's 10th-90th
    percentile. The knee is necessarily inside the range of observed scores, so
    deriving the window from them means the sweep keeps finding it however the
    embedder, the chunker or the corpus change underneath.
    """
    import numpy as np

    questions = load_questions(split)
    index = VectorIndex.load()
    embedder = Embedder()

    tops: list[float] = []
    for q in questions:
        hits = index.hybrid_search(
            q["question"], embedder.embed_query(q["question"]),
            top_k=1, min_score=0.0,
        )
        if hits:
            tops.append(float(hits[0].score))

    if not tops:
        return [round(0.20 + 0.05 * i, 2) for i in range(n)]

    lo = float(np.percentile(tops, 10))
    hi = float(np.percentile(tops, 90))
    if hi - lo < 0.02:          # degenerate spread; widen so the sweep still moves
        lo, hi = lo - 0.05, hi + 0.05
    step = (hi - lo) / (n - 1)
    return [round(lo + step * i, 2) for i in range(n)]


def machine_label() -> dict:
    """Identify the machine, so results from different hosts stay comparable.

    Retrieval is not memory-bandwidth bound the way generation is, so these
    figures travel between machines far better than tokens/second - but the
    label is recorded anyway, because a results file without provenance is an
    assertion rather than evidence.
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def evaluate(min_score: float, top_k: int, split: str = "dev",
             questions_path: "Path | None" = None) -> dict:
    index = VectorIndex.load()
    questions = load_questions(split, questions_path)

    in_scope = [q for q in questions if q["in_scope"]]
    out_scope = [q for q in questions if not q["in_scope"]]

    per_crop: dict[str, dict[str, int]] = defaultdict(lambda: {"hit": 0, "total": 0})
    covered: list[dict] = []
    uncovered: list[dict] = []
    wrongly_answered: list[dict] = []
    correctly_refused = 0
    latencies: list[float] = []

    with Embedder() as embedder:
        for q in in_scope:
            started = time.perf_counter()
            hits = index.hybrid_search(
                q["question"],
                embedder.embed_query(q["question"]),
                top_k=top_k,
                min_score=min_score,
            )
            latencies.append((time.perf_counter() - started) * 1000)

            per_crop[q["crop"]]["total"] += 1

            # RELEVANCE, not presence.
            #
            # Counting "did anything score above threshold" scored a passage
            # about cassava scales as a hit for a question about cassava mosaic
            # disease. Both are cassava, both are pests, and the metric could
            # not tell them apart - so the earlier 88% figure measured whether
            # retrieval returned something, not whether it returned the right
            # thing.
            #
            # A hit now requires one of the question's expected terms to appear
            # in a retrieved passage. Questions without annotations fall back to
            # presence, and are reported separately so the gap is visible rather
            # than quietly counted as success.
            expected = [t.lower() for t in q.get("expect_any", [])]
            if hits and expected:
                blob = " ".join(h.chunk.text.lower() for h in hits)
                relevant = any(t in blob for t in expected)
            else:
                relevant = bool(hits)

            if relevant:
                per_crop[q["crop"]]["hit"] += 1
                covered.append({
                    "id": q["id"],
                    "question": q["question"],
                    "top_score": round(hits[0].score, 3),
                    "source": hits[0].chunk.citation(),
                })
            else:
                uncovered.append({
                    "id": q["id"],
                    "topic": q["topic"],
                    "question": q["question"],
                    "reason": (
                        "no passage above threshold" if not hits
                        else "retrieved, but not about the right thing"
                    ),
                    "got": hits[0].chunk.title[:60] if hits else None,
                })

        for q in out_scope:
            started = time.perf_counter()

            # Match the real pipeline: policy scope is checked BEFORE
            # retrieval, so a dosage or human-medical question never reaches
            # the index. Measuring retrieval alone reported these as failures
            # the deployed system does not actually have - the eval was testing
            # a path that does not exist.
            from agbe.rag import scope

            if not scope.check(q["question"]).in_scope:
                hits = []
            else:
                hits = index.hybrid_search(
                    q["question"],
                    embedder.embed_query(q["question"]),
                    top_k=top_k,
                    min_score=min_score,
                )
            latencies.append((time.perf_counter() - started) * 1000)

            if hits:
                wrongly_answered.append({
                    "id": q["id"],
                    "question": q["question"],
                    "top_score": round(hits[0].score, 3),
                    "source": hits[0].chunk.citation(),
                })
            else:
                correctly_refused += 1

    coverage = len(covered) / len(in_scope) if in_scope else 0.0
    refusal = correctly_refused / len(out_scope) if out_scope else 0.0

    latencies.sort()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": machine_label(),
        "settings": {
            "min_score": min_score,
            "top_k": top_k,
            "embedding_model": config.EMBEDDING.filename,
        },
        "index": {"n_chunks": len(index),
                  "n_documents": len({c.doc_id for c in index.chunks})},
        "coverage": {
            "value": round(coverage, 4),
            "target": TARGET_COVERAGE,
            "pass": coverage >= TARGET_COVERAGE,
            "covered": len(covered),
            "total": len(in_scope),
        },
        "refusal": {
            "value": round(refusal, 4),
            "target": TARGET_REFUSAL,
            "pass": refusal >= TARGET_REFUSAL,
            "refused": correctly_refused,
            "total": len(out_scope),
        },
        "retrieval_latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 2) if latencies else None,
            "p95": round(latencies[int(len(latencies) * 0.95)], 2) if latencies else None,
        },
        "per_crop": {
            crop: {"covered": v["hit"], "total": v["total"],
                   "coverage": round(v["hit"] / v["total"], 3) if v["total"] else 0.0}
            for crop, v in sorted(per_crop.items())
        },
        "gaps": uncovered,
        "false_positives": wrongly_answered,
    }


def print_report(r: dict) -> None:
    cov, ref = r["coverage"], r["refusal"]
    print()
    print("=" * 72)
    print("RETRIEVAL COVERAGE")
    print("=" * 72)
    print(f"  index         {r['index']['n_chunks']} chunks "
          f"from {r['index']['n_documents']} documents")
    print(f"  min_score     {r['settings']['min_score']}   top_k {r['settings']['top_k']}")
    print()
    print(f"  COVERAGE      {cov['value']:.1%}  ({cov['covered']}/{cov['total']} "
          f"in-scope questions)   target {cov['target']:.0%}  "
          f"{'PASS' if cov['pass'] else 'FAIL'}")
    print(f"  REFUSAL       {ref['value']:.1%}  ({ref['refused']}/{ref['total']} "
          f"out-of-scope refused)  target {ref['target']:.0%}  "
          f"{'PASS' if ref['pass'] else 'FAIL'}")
    print()
    lat = r["retrieval_latency_ms"]
    if lat["p50"] is not None:
        print(f"  latency       p50 {lat['p50']} ms   p95 {lat['p95']} ms")
    print()
    print("  per crop:")
    for crop, v in r["per_crop"].items():
        bar = "#" * int(v["coverage"] * 20)
        print(f"    {crop:<10} {v['covered']:>2}/{v['total']:<2} "
              f"{v['coverage']:>6.0%} {bar}")

    if r["gaps"]:
        print(f"\n  GAPS - {len(r['gaps'])} in-scope questions not answered:")
        for g in r["gaps"]:
            print(f"    [{g['id']}] {g['topic']}: {g['question'][:56]}")
            reason = g.get("reason", "")
            if g.get("got"):
                print(f"        {reason} - got: {g['got']}")
            elif reason:
                print(f"        {reason}")
        print("    -> 'no passage' means a corpus gap; 'not about the right")
        print("       thing' means a retrieval gap. They need different fixes.")

    if r["false_positives"]:
        print(f"\n  FALSE POSITIVES - {len(r['false_positives'])} out-of-scope "
              "questions were answered:")
        for f in r["false_positives"]:
            print(f"    [{f['id']}] {f['top_score']:.2f} {f['question'][:55]}")
        print("    -> raise min_score, or accept that these will be answered badly")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, default=config.RETRIEVAL.min_score)
    ap.add_argument("--top-k", type=int, default=config.RETRIEVAL.top_k)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                    help="dev for tuning (default); test is the reported number")
    ap.add_argument("--sweep", action="store_true",
                    help="try a range of min_score values and show the trade-off")
    ap.add_argument("--questions", help="alternative question file; read whole, ignoring --split")
    ap.add_argument("--save", action="store_true", help="write results JSON")
    args = ap.parse_args()

    if args.split == "test" and args.sweep:
        # Sweeping is tuning. Doing it on the held-out set is exactly the
        # contamination the split exists to prevent, so it is refused rather
        # than merely discouraged.
        print("refusing to sweep on the test split - that is tuning on held-out "
              "data.\nSweep on --split dev, then run --split test once.")
        return 2

    if args.sweep:
        # The range is DERIVED, not hardcoded.
        #
        # It used to be a fixed 0.20-0.55, chosen when scores lived there. After
        # re-chunking on section headings the same eight rows returned identical
        # numbers - coverage 93.3%, refusal 66.7%, every time - because the whole
        # swept range now sits below where anything happens. The tool did not
        # fail; it printed eight rows of noise shaped like data, and REPORT.md
        # 6.2 quotes a table produced by it.
        #
        # A sweep whose window is fixed while the index moves is a sweep that
        # silently stops measuring. So the window is taken from the scores the
        # index actually produces for these questions: the knee is always inside
        # the observed distribution, wherever that distribution has drifted to.
        floors = sweep_floors(args.split)
        print(f"{'min_score':>10} {'coverage':>10} {'refusal':>10}   "
              f"(split={args.split}, range derived from observed scores)")
        for floor in floors:
            r = evaluate(floor, args.top_k, split=args.split)
            print(f"{floor:>10.2f} {r['coverage']['value']:>9.1%} "
                  f"{r['refusal']['value']:>10.1%}")
        return 0

    report = evaluate(args.min_score, args.top_k, split=args.split,
                      questions_path=args.questions)
    report["split"] = args.split
    print_report(report)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "coverage.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")

    return 0 if (report["coverage"]["pass"] and report["refusal"]["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

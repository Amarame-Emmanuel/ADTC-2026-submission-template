"""ARC-Easy accuracy, replicating the ADTC profiler's method.

WHY THIS EXISTS
---------------
The official profiler measures the automated half of S_acc as ARC-Easy over 50
samples, scored by continuation log-probability through llama-cpp-python. That
is 50% of the competition score, and it is the one number that could still
justify shipping the larger model.

The profiler itself could not be built here: its Dockerfile clones llama.cpp,
and that clone failed four times across two protocols on a connection that
would not hold a large transfer. Rather than block, this reproduces the
measurement directly - the method is short and fully documented in the
profiler's accuracy.py.

METHOD, MATCHING THE PROFILER
-----------------------------
For each question, every answer choice is scored as a continuation of the same
context. The choice with the highest summed log-probability wins; accuracy is
the fraction where that matches the labelled answer.

Two details matter and both are taken from the profiler:

  * Scoring starts at the first token where the full sequence and the context
    diverge, not at len(context_tokens). BPE can merge characters across the
    boundary, so slicing by prefix length would score the wrong positions.

  * Logits are read directly rather than through llama-cpp-python's echoed
    logprobs, which are shifted by one position - token i scored by the
    distribution that predicts token i+1 - making rankings near-random.

FIDELITY
--------
This is a faithful reimplementation, not the official tool. Our llama.cpp is
built with AVX2/FMA enabled while the profiler's is scalar, which affects
speed but not the arithmetic of a log-softmax, so accuracy should agree.
Anything reported from here is labelled as a local reproduction in REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARC_PATH = Path(__file__).parent / "arc" / "arc_easy_test.parquet"

#: The profiler's default. Kept identical so numbers are comparable.
DEFAULT_LIMIT = 50
N_CTX = 2048


#: Pre-converted from the HF parquet on the host. Kept as JSON so the runtime
#: image needs no parquet reader: pyarrow exists purely to prepare evaluation
#: data and has no business in a container whose memory footprint is 20% of the
#: competition score.
ARC_JSON = Path(__file__).parent / "arc" / "arc_easy_test.json"


def load_questions(limit: int) -> list[dict]:
    return json.loads(ARC_JSON.read_text(encoding="utf-8"))[:limit]


def common_prefix_len(full: list[int], prefix: list[int]) -> int:
    """First position where tokenisations diverge, at least 1.

    BPE may merge characters across the context/continuation boundary, so
    slicing by len(prefix) can start scoring in the wrong place. At least one
    token must remain as conditioning context.
    """
    n = 0
    while n < min(len(full), len(prefix)) and full[n] == prefix[n]:
        n += 1
    return max(n, 1)


def score_continuation(llm, context: str, continuation: str) -> float:
    """Summed log-probability of `continuation` given `context`."""
    ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=True)
    full_tokens = llm.tokenize((context + continuation).encode("utf-8"), add_bos=True)

    start = common_prefix_len(full_tokens, ctx_tokens)
    if start >= len(full_tokens):
        return float("-inf")

    llm.reset()
    llm.eval(full_tokens)
    logits = np.asarray(llm.scores[: len(full_tokens)], dtype=np.float64)

    total = 0.0
    for pos in range(start, len(full_tokens)):
        # Position pos is predicted by the distribution at pos-1.
        row = logits[pos - 1]
        row = row - row.max()
        logprobs = row - np.log(np.exp(row).sum())
        total += float(logprobs[full_tokens[pos]])
    return total


def evaluate(model_path: str, limit: int = DEFAULT_LIMIT) -> dict:
    from llama_cpp import Llama

    questions = load_questions(limit)
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=4,
        logits_all=True,   # required: we read per-position logits
        verbose=False,
    )

    correct = correct_norm = 0
    for i, q in enumerate(questions, 1):
        context = f"Question: {q['question']}\nAnswer:"
        scores, norm_scores = [], []
        for text in q["texts"]:
            s = score_continuation(llm, context, f" {text}")
            scores.append(s)
            # acc_norm: length-normalised, which lm-eval reports alongside acc
            # because raw log-probability favours shorter answers.
            norm_scores.append(s / max(len(text), 1))

        correct += int(np.argmax(scores) == q["gold"])
        correct_norm += int(np.argmax(norm_scores) == q["gold"])
        if i % 10 == 0:
            print(f"  {i}/{len(questions)}  acc {correct/i:.3f}  "
                  f"acc_norm {correct_norm/i:.3f}", flush=True)

    n = len(questions)
    return {
        "benchmark": "arc_easy",
        "samples": n,
        "acc": round(correct / n, 4),
        "acc_norm": round(correct_norm / n, 4),
        "model": Path(model_path).name,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()

    result = evaluate(args.model, args.limit)
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

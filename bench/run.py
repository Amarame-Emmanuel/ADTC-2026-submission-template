"""Benchmark harness: memory ceiling, latency, throughput.

This is the evidence artifact for the ADTC hardware standard. It answers three
questions, in order of how much they matter:

  1. Does the system stay under 7 GB?  (exceeding it is disqualifying)
  2. How long until the farmer sees the first word?  (perceived responsiveness)
  3. How fast does the answer arrive after that?     (throughput)

MEASURING MEMORY HONESTLY
-------------------------
Peak RSS is read from /proc/self/status VmHWM, the kernel's high-water mark.
psutil's rss is the *instantaneous* value and will happily report 2 GB for a
process that briefly touched 6 GB and freed it - which is precisely the spike
that would get a submission disqualified. VmHWM cannot be missed by sampling
because the kernel maintains it continuously.

The container is additionally run with --memory=7g --memory-swap=7g, so a
breach is an OOM kill rather than a number in a report. A benchmark that can
only report a violation is weaker than one that cannot survive it.

WHY THE HARDWARE CAVEAT IS PRINTED, NOT BURIED
----------------------------------------------
Token generation on CPU is memory-bandwidth bound. A development machine with
DDR5 will report throughput far above what the target's DDR4 delivers, and no
container flag can throttle memory bandwidth. Thread count and RAM ceiling are
enforceable; bandwidth is not. Every result therefore carries its host's CPU
and RAM, and the report prints the caveat rather than leaving a reader to
assume the numbers came from the reference laptop.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from agbe import config

RESULTS_DIR = Path(__file__).parent / "results"
QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def peak_rss_bytes() -> int:
    """Kernel high-water mark for this process, in bytes."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:  # noqa: BLE001
        return 0


def current_rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:  # noqa: BLE001
        return 0


def cgroup_memory_limit() -> int | None:
    """The limit actually enforced on this process, if any.

    Reported so a results file records the envelope it was produced under.
    A number measured with no cap is not evidence that the cap is respected.
    """
    for path in (
        "/sys/fs/cgroup/memory.max",                     # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",   # cgroup v1
    ):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                value = int(raw)
                # v1 reports an absurd sentinel when unlimited.
                return value if value < (1 << 62) else None
        except (OSError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Host description
# ---------------------------------------------------------------------------

def host_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "threads_configured": config.DEFAULT_THREADS,
    }

    try:
        model = ""
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        if model:
            info["cpu"] = model
    except OSError:
        info["cpu"] = platform.processor()

    try:
        import psutil

        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:  # noqa: BLE001
        pass

    limit = cgroup_memory_limit()
    info["cgroup_memory_limit_gb"] = round(limit / 1024**3, 2) if limit else None
    return info


def is_reference_like(info: dict) -> bool:
    """Crude check for whether these numbers can stand in for the target.

    Deliberately conservative. It exists so that a results file cannot quietly
    imply it came from the ADTC Standard Laptop when it came from a
    workstation.
    """
    ram = info.get("total_ram_gb") or 0
    cpu = (info.get("cpu") or "").lower()
    looks_like_target = ram <= 9 and ("i5" in cpu or "i3" in cpu or "celeron" in cpu)
    return bool(looks_like_target)


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

def load_workload(n: int) -> list[str]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    in_scope = [q["question"] for q in data["questions"] if q["in_scope"]]
    return in_scope[:n]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[idx]


def run_benchmark(n_questions: int, max_tokens: int) -> dict:
    from agbe.advisor import AdvisoryEngine

    questions = load_workload(n_questions)
    baseline_rss = current_rss_bytes()

    stages: dict[str, int] = {"baseline": baseline_rss}

    engine = AdvisoryEngine()
    stages["after_index_load"] = current_rss_bytes()

    engine.embedder.load()
    stages["after_embedder_load"] = current_rss_bytes()

    engine.llm.load()
    stages["after_llm_load"] = current_rss_bytes()

    ttfts: list[float] = []
    rates: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    refusals = 0
    per_question: list[dict] = []

    started = time.perf_counter()
    for question in questions:
        # Compress, because the shipped paths compress.
        #
        # This benchmark used to call engine.retrieve() and send full chunk
        # text, while advise() sent compressed passages. It was therefore
        # timing a configuration no user could reach: ~2,100 prompt tokens
        # against the ~900 the application actually sends, and 65s to first
        # token instead of what a farmer waits. A benchmark measuring a path
        # nothing else uses is worse than no benchmark, because it is believed.
        hits, texts, _comp = engine.retrieve_and_compress(question)
        if not hits:
            refusals += 1
            per_question.append({"question": question, "refused": True})
            continue

        from agbe.advisor import build_prompt

        prompt = build_prompt(question, hits, texts)
        _, stats = engine.llm.complete(prompt, max_tokens=max_tokens)
        ttfts.append(stats.time_to_first_token_s)
        rates.append(stats.tokens_per_second)
        completion_tokens.append(stats.completion_tokens)
        prompt_tokens.append(engine.llm.count_tokens(prompt[1]["content"]))
        per_question.append({
            "question": question,
            "refused": False,
            "ttft_s": round(stats.time_to_first_token_s, 3),
            "tokens_per_second": round(stats.tokens_per_second, 2),
            "completion_tokens": stats.completion_tokens,
        })

    wall = time.perf_counter() - started
    stages["after_generation"] = current_rss_bytes()

    peak = peak_rss_bytes()
    info = host_info()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": info,
        "reference_like_hardware": is_reference_like(info),
        "models": {
            "llm": config.LLM.filename,
            "embedding": config.EMBEDDING.filename,
            "n_ctx": config.LLM.n_ctx,
            "n_threads": config.LLM.n_threads,
        },
        "memory": {
            "peak_rss_bytes": peak,
            "peak_rss_gb": round(peak / 1024**3, 3),
            "ceiling_gb": round(config.MEMORY_CEILING_BYTES / 1024**3, 2),
            "internal_target_gb": round(config.MEMORY_TARGET_BYTES / 1024**3, 2),
            "under_ceiling": peak <= config.MEMORY_CEILING_BYTES,
            "under_internal_target": peak <= config.MEMORY_TARGET_BYTES,
            "headroom_gb": round(
                (config.MEMORY_CEILING_BYTES - peak) / 1024**3, 3
            ),
            "stages_gb": {k: round(v / 1024**3, 3) for k, v in stages.items()},
            "predicted_gb": round(config.budget_total() / 1024**3, 3),
        },
        "latency": {
            "ttft_p50_s": round(percentile(ttfts, 0.50), 3) if ttfts else None,
            "ttft_p95_s": round(percentile(ttfts, 0.95), 3) if ttfts else None,
            "ttft_mean_s": round(statistics.mean(ttfts), 3) if ttfts else None,
        },
        "throughput": {
            "tokens_per_second_p50": round(percentile(rates, 0.50), 2) if rates else None,
            "tokens_per_second_mean": round(statistics.mean(rates), 2) if rates else None,
            "mean_prompt_tokens": round(statistics.mean(prompt_tokens)) if prompt_tokens else None,
            "mean_completion_tokens": (
                round(statistics.mean(completion_tokens)) if completion_tokens else None
            ),
        },
        "workload": {
            "n_questions": len(questions),
            "n_answered": len(ttfts),
            "n_refused": refusals,
            "wall_clock_s": round(wall, 1),
        },
        "per_question": per_question,
    }


def print_report(r: dict) -> None:
    mem, lat, thr = r["memory"], r["latency"], r["throughput"]
    host = r["host"]

    print()
    print("=" * 74)
    print("ÀGBẸ̀ BENCHMARK")
    print("=" * 74)
    print(f"  host        {host.get('cpu', 'unknown')}")
    print(f"              {host.get('cpu_count_logical', '?')} logical CPUs, "
          f"{host.get('total_ram_gb', '?')} GB RAM, "
          f"{host['threads_configured']} threads configured")
    cap = host.get("cgroup_memory_limit_gb")
    print(f"  enforced    {f'{cap} GB cgroup limit' if cap else 'NO MEMORY CAP ENFORCED'}")
    print(f"  model       {r['models']['llm']} @ {r['models']['n_ctx']} ctx")
    print()

    verdict = "PASS" if mem["under_ceiling"] else "FAIL - DISQUALIFYING"
    print(f"  PEAK RSS    {mem['peak_rss_gb']:.2f} GB / {mem['ceiling_gb']:.0f} GB "
          f"ceiling    {verdict}")
    print(f"              headroom {mem['headroom_gb']:.2f} GB   "
          f"(predicted {mem['predicted_gb']:.2f} GB)")
    if not mem["under_internal_target"]:
        print(f"              WARNING: above the {mem['internal_target_gb']:.0f} GB "
              "internal target")
    print()
    print("  memory by stage:")
    for stage, gb in mem["stages_gb"].items():
        print(f"    {stage:<24} {gb:>6.2f} GB")
    print()

    if lat["ttft_p50_s"] is not None:
        print(f"  TTFT        p50 {lat['ttft_p50_s']:.2f}s   p95 {lat['ttft_p95_s']:.2f}s")
        print(f"  THROUGHPUT  {thr['tokens_per_second_p50']:.1f} tok/s (p50)")
        print(f"              prompt ~{thr['mean_prompt_tokens']} tokens, "
              f"completion ~{thr['mean_completion_tokens']} tokens")
    print()
    print(f"  workload    {r['workload']['n_answered']} answered, "
          f"{r['workload']['n_refused']} refused, "
          f"{r['workload']['wall_clock_s']}s wall clock")
    print()

    if not r["reference_like_hardware"]:
        print("  " + "!" * 68)
        print("  THESE NUMBERS ARE NOT FROM THE ADTC STANDARD LAPTOP.")
        print("  Memory ceiling and thread count are enforced, but memory")
        print("  BANDWIDTH cannot be throttled. CPU token generation is")
        print("  bandwidth-bound, so throughput here is OPTIMISTIC relative to")
        print("  8 GB DDR4 target hardware. Peak RSS transfers; tok/s does not.")
        print("  " + "!" * 68)
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=str, default="benchmark.json")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    report = run_benchmark(args.questions, args.max_tokens)
    print_report(report)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / args.out
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")

    return 0 if report["memory"]["under_ceiling"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

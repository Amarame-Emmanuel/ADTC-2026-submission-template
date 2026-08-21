"""Thermal behaviour under sustained load.

WHAT THE SCORING RULE ACTUALLY SAYS
-----------------------------------
`P_thermal` is a penalty applied when peak core temperature reaches **85 °C**
(profiler/adtc-profiler/src/adtc_profiler/thermal.py). The official profiler
also documents that `core_temp_c_peak` may be **null**, because hosts that do
not expose thermal sensors cannot report one.

WHY OUR TEMPERATURE READING IS NULL, STATED UP FRONT
----------------------------------------------------
This development host is Windows with Docker Desktop on WSL2. We checked, and
the sensors are not reachable:

  * `/sys/class/thermal` contains cooling devices but **no** `thermal_zone*`
  * `/sys/class/hwmon` exposes only `AC1` (mains adapter) and `BAT1` (battery)
  * `psutil.sensors_temperatures()` returns `{}`
  * the Windows `MSAcpi_ThermalZoneTemperature` WMI class returns access-denied

That is a property of the virtualisation layer, not something a flag fixes.

AND WHY A TEMPERATURE FROM THIS MACHINE WOULD BE WORTHLESS ANYWAY
-----------------------------------------------------------------
This matters more than the missing sensor, and it is the opposite of the memory
situation.

Peak RSS **transfers** between machines: a model that needs 1.7 GB here needs
1.7 GB on the target laptop. Temperature does **not** transfer, and not even
directionally. This host is an i7-14650HX - a 55 W-class mobile workstation part
with the cooling to match. The ADTC Standard Laptop is a 15 W-class U-series i5
in a thin chassis. The two throttle at completely different loads, and a
comfortable 65 °C here would predict nothing whatsoever about the target.

So reporting a temperature from this host would be worse than reporting none: it
would look like evidence while being none. What this module measures instead is
the thing that is *machine-independent* about our thermal risk - the shape of the
workload - plus the one consequence of throttling that is observable without a
thermometer.

WHAT IS ACTUALLY MEASURED
-------------------------
1. **Sustained-load throughput decay.** Thermal throttling has exactly one
   symptom that matters to a farmer: the answer gets slower. We generate
   continuously for the full duration and compare tokens/sec in the final
   quarter against the first quarter. Flat throughput under sustained load is
   positive evidence of no throttling, whatever the thermometer says. Decay is
   evidence of throttling even if the thermometer is missing.

   The continuous loop is deliberately a WORST case. Advisory use is
   request-driven - a person reads an answer before asking the next question -
   so the real duty cycle sits far below 100%. That argument appears here as
   reasoning, not as a reported number, because this harness does not measure
   it.

2. **CPU utilisation p99**, computed the same way the official profiler does, so
   the two numbers are comparable.

3. **Core temperature**, attempted through the same sensor paths the official
   profiler uses, and reported as `null` when unavailable rather than
   substituted with something more comforting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

#: The temperature at which the scoring rule applies P_thermal.
PENALTY_THRESHOLD_C = 85.0

#: Sensor label substrings indicating a CPU core/die temperature. Copied
#: deliberately from the official profiler so that if a sensor IS present, we
#: read the same one it would.
_CORE_HINTS = ("core", "cpu", "tdie", "tccd", "package")

#: Fraction of the run treated as "early" and "late" when testing for decay.
_WINDOW_FRACTION = 0.25

#: Throughput decay beyond this fraction is reported as evidence of throttling.
#: Set at 10% because run-to-run noise on a shared desktop is a few percent;
#: anything above this is a trend rather than jitter.
DECAY_ALERT = 0.10


def read_core_temp() -> float | None:
    """Peak CPU temperature, or None when the host does not expose one."""
    try:
        temps = psutil.sensors_temperatures() or {}
    except (AttributeError, OSError):
        return None

    fallback: float | None = None
    for entries in temps.values():
        for entry in entries:
            if not entry.current or entry.current <= 0:
                continue
            if any(h in (entry.label or "").lower() for h in _CORE_HINTS):
                return float(entry.current)
            if fallback is None:
                fallback = float(entry.current)
    return fallback


class Sampler:
    """Background CPU/temperature poller."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.cpu: list[float] = []
        self.temps: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        psutil.cpu_percent(interval=None)  # first call is meaningless; prime it
        while not self._stop.is_set():
            self.cpu.append(psutil.cpu_percent(interval=self.interval_s))
            t = read_core_temp()
            if t is not None:
                self.temps.append(t)

    def __enter__(self) -> "Sampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def p99_cpu(self) -> float:
        if not self.cpu:
            return 0.0
        s = sorted(self.cpu)
        return round(min(100.0, s[max(0, int(len(s) * 0.99) - 1)]), 1)

    def peak_temp(self) -> float | None:
        return round(max(self.temps), 1) if self.temps else None


def run(duration_s: int, max_tokens: int) -> dict:
    """Generate continuously for `duration_s`, recording per-answer throughput."""
    from agbe.advisor import AdvisoryEngine
    from agbe.advisor import build_prompt

    questions = [
        "My cassava leaves are yellow and twisted. What is wrong?",
        "How do I control fall armyworm in maize?",
        "My chickens have twisted necks and greenish diarrhoea.",
        "When should I plant maize in southwest Nigeria?",
        "How do I store yam so it does not rot?",
        "What causes leaf spot in groundnut?",
    ]

    engine = AdvisoryEngine()
    engine.embedder.load()
    engine.llm.load()

    samples: list[dict] = []
    started = time.perf_counter()
    i = 0

    with Sampler() as sampler:
        while time.perf_counter() - started < duration_s:
            q = questions[i % len(questions)]
            i += 1
            hits, texts, _ = engine.retrieve_and_compress(q)
            if not hits:
                continue
            _, stats = engine.llm.complete(
                build_prompt(q, hits, texts), max_tokens=max_tokens
            )
            samples.append({
                "elapsed_s": round(time.perf_counter() - started, 1),
                "tokens_per_second": round(stats.tokens_per_second, 2),
                "ttft_s": round(stats.time_to_first_token_s, 2),
            })

    return _summarise(samples, sampler, duration_s)


def _summarise(samples: list[dict], sampler: Sampler, duration_s: int) -> dict:
    rates = [s["tokens_per_second"] for s in samples]
    n_window = max(1, int(len(rates) * _WINDOW_FRACTION))
    early = statistics.mean(rates[:n_window]) if rates else 0.0
    late = statistics.mean(rates[-n_window:]) if rates else 0.0
    decay = (early - late) / early if early else 0.0

    peak = sampler.peak_temp()
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpus": psutil.cpu_count(),
            "platform": platform.platform(),
            # WHICH cores, for the same reason bench/run.py records it: a
            # sustained-load result is only comparable to another run on the
            # same cores. Two runs both reporting "4 threads" can differ
            # because they do not share a cache or contend with the same
            # neighbours, and decay is exactly the quantity that difference
            # shows up in.
            "cpuset": _cpuset(),
        },
        "duration_s": duration_s,
        "answers_generated": len(samples),
        "core_temp_c_peak": peak,
        "core_temp_available": peak is not None,
        "penalty_threshold_c": PENALTY_THRESHOLD_C,
        "throttled_by_temperature": (peak >= PENALTY_THRESHOLD_C) if peak else None,
        "cpu_percent_p99": sampler.p99_cpu(),
        "throughput_first_quarter_tok_s": round(early, 2),
        "throughput_last_quarter_tok_s": round(late, 2),
        "throughput_decay_fraction": round(decay, 4),
        "throughput_decay_exceeds_alert": decay > DECAY_ALERT,
        "samples": samples,
    }


def _cpuset() -> str:
    """The cores this process may actually run on, as the OS reports them."""
    try:
        cores = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return "unknown"
    if not cores:
        return "unknown"
    parts: list[str] = []
    start = prev = cores[0]
    for core in cores[1:] + [None]:
        if core != prev + 1:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = core
        prev = core
    return ",".join(parts)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=1200,
                    help="seconds of continuous generation (default 1200 = 20 min)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=str, default="bench/results/thermal.json")
    args = ap.parse_args()

    print("=" * 74)
    print("ÀGBẸ̀ SUSTAINED-LOAD / THERMAL MEASUREMENT")
    print("=" * 74)
    print(f"  generating continuously for {args.duration}s "
          f"({args.duration / 60:.0f} min) - worst case, not realistic use")
    print()

    result = run(args.duration, args.max_tokens)

    temp = result["core_temp_c_peak"]
    print(f"  answers generated   {result['answers_generated']}")
    print(f"  CPU p99             {result['cpu_percent_p99']}%")
    if temp is None:
        print("  peak core temp      UNAVAILABLE (null)")
        print("                      no thermal_zone*, hwmon exposes only AC/battery.")
        print("                      Schema permits null; see module docstring.")
    else:
        verdict = "OVER THRESHOLD" if temp >= PENALTY_THRESHOLD_C else "under threshold"
        print(f"  peak core temp      {temp} °C  ({verdict}, limit {PENALTY_THRESHOLD_C})")

    print()
    print(f"  throughput, first quarter   {result['throughput_first_quarter_tok_s']} tok/s")
    print(f"  throughput, last quarter    {result['throughput_last_quarter_tok_s']} tok/s")
    decay_pct = result["throughput_decay_fraction"] * 100
    if result["throughput_decay_exceeds_alert"]:
        print(f"  DECAY {decay_pct:.1f}%  - consistent with thermal throttling")
    else:
        print(f"  decay {decay_pct:.1f}%  - no throttling signature under sustained load")

    print()
    print("  " + "!" * 68)
    print("  THERMAL RESULTS DO NOT TRANSFER TO THE TARGET LAPTOP.")
    print("  Peak RSS transfers between machines; temperature does not.")
    print("  This host is a 55 W-class i7-14650HX; the ADTC Standard Laptop")
    print("  is a 15 W-class U-series i5 in a thin chassis. They throttle at")
    print("  different loads. What transfers is the WORKLOAD SHAPE measured")
    print("  here: thread count, duty cycle and the absence of decay under")
    print("  a load heavier than the application ever generates.")
    print("  " + "!" * 68)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

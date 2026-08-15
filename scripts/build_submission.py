"""Assemble submission/ from the pinned model, and refuse to lie about it.

WHY THIS EXISTS
---------------
`submission/model/` was populated by hand, so it drifted the moment the model
changed. Found after the Q4_0 swap: `metadata.json` declared

    "quantization": "GGUF Q4_0"
    "model_path":   "model/qwen2.5-1.5b-instruct-q4_0.gguf"

while the directory contained `qwen2.5-1.5b-instruct-q4_k_m.gguf`. A judge
unpacking that bundle either cannot find the declared file or silently
benchmarks the wrong quantisation - and the Q4_0 case rests entirely on a 2.7x
throughput difference between exactly those two files (REPORT.md 3.3b).

This is the fourth time in this project that a hand-maintained copy of
something has disagreed with its source: the web UI carried its own generation
loop, `models.lock.json` once pinned a model the app did not load, the results
table was transcribed rather than generated, and now the submission bundle. The
pattern is always the same - two places holding the same fact, one of them
updated.

So the bundle is generated from `models.lock.json`, which is the pin the runtime
already verifies against, and the three places that name the model are checked
to agree:

    models.lock.json          the checksum-verified pin
    agbe/config.py            what the application loads
    submission/metadata.json  what we tell the judges

Any disagreement is a hard failure. A submission that describes a different
model from the one it contains is worse than one that fails to build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "models.lock.json"
METADATA = ROOT / "submission" / "metadata.json"
BUNDLE = ROOT / "submission" / "model"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify only; do not copy. Exit 1 on any disagreement.")
    args = ap.parse_args()

    lock = json.loads(LOCK.read_text(encoding="utf-8"))["llm"]
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    pinned = lock["filename"]

    # 1. metadata must name the pinned file
    declared = meta["_runtime"]["model_path"].split("/")[-1]
    problems: list[str] = []
    if declared != pinned:
        problems.append(
            f"metadata.json declares {declared!r}, models.lock.json pins {pinned!r}"
        )

    # 2. the application must load the pinned file
    sys.path.insert(0, str(ROOT))
    from agbe import config  # noqa: E402  (after sys.path)

    if config.LLM.filename != pinned:
        problems.append(
            f"agbe/config.py loads {config.LLM.filename!r}, "
            f"models.lock.json pins {pinned!r}"
        )

    # 3. the declared quantisation must match the filename it points at
    quant = meta["model"].get("quantization", "").lower().replace("gguf", "").strip()
    if quant and quant.replace("_", "") not in pinned.lower().replace("_", ""):
        problems.append(
            f"metadata.json says quantization {meta['model']['quantization']!r}, "
            f"which does not appear in {pinned!r}"
        )

    if problems:
        print("SUBMISSION INCONSISTENT:")
        for p in problems:
            print(f"  - {p}")
        return 1

    source = ROOT / "models" / pinned
    if not source.exists():
        print(f"pinned model not found: {source}\nrun `make fetch-models` first")
        return 1

    target = BUNDLE / pinned
    if args.check:
        if not target.exists():
            print(f"bundle is missing {pinned}")
            return 1
        stale = [p for p in BUNDLE.glob("*.gguf") if p.name != pinned]
        if stale:
            print(f"bundle carries files that are not the pinned model: "
                  f"{[p.name for p in stale]}")
            return 1
        print(f"submission consistent: {pinned}")
        return 0

    BUNDLE.mkdir(parents=True, exist_ok=True)
    for p in BUNDLE.glob("*.gguf"):
        if p.name != pinned:
            print(f"removing stale bundle model: {p.name}")
            p.unlink()

    if not target.exists() or target.stat().st_size != source.stat().st_size:
        print(f"copying {pinned} ({source.stat().st_size / 1e9:.2f} GB) ...")
        shutil.copyfile(source, target)

    digest = sha256(target)
    if digest != lock["sha256"]:
        print(f"CHECKSUM MISMATCH after copy\n  expected {lock['sha256']}\n"
              f"  got      {digest}")
        return 1

    print(f"submission/model/{pinned}")
    print(f"  sha256 {digest}  ({target.stat().st_size} bytes) - matches the pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and integrity-verify every model weight Àgbẹ̀ needs.

Reproducibility contract
------------------------
A Gate 2 auditor must be able to obtain byte-identical weights to the ones we
benchmarked. Hugging Face repositories are mutable - a maintainer can reupload
a requantised GGUF under the same filename - so "download from this repo" is
not a reproducible instruction on its own.

This script therefore pins content, not locations:

  * `models.lock.json` records the SHA-256 and byte size of every file.
  * Normal runs verify downloads against the lock and FAIL on mismatch.
  * `--update-lock` is the only way to change a pinned hash, and it is an
    explicit, reviewable action that shows up as a diff.

Hashes are populated on first fetch rather than hardcoded here, because a hash
invented by a developer who never downloaded the file verifies nothing.

Usage
-----
    python scripts/fetch_models.py                 # fetch + verify
    python scripts/fetch_models.py --update-lock   # fetch + (re)pin hashes
    python scripts/fetch_models.py --verify-only   # check what is on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _force_ipv4() -> None:
    """Restrict name resolution to IPv4.

    Hugging Face's CDN resolves to IPv6 addresses (2600:9000:...) inside this
    container, which has no IPv6 route. Python then fails with
    "Failed to resolve 'us.aws.cdn.hf.co'" - a message that reads like the host
    does not exist rather than like a routing problem, which sent the first
    diagnosis in entirely the wrong direction.

    Filtering getaddrinfo to AF_INET makes the resolver return only addresses
    the container can actually reach. Harmless where IPv6 works, decisive where
    it does not, and far more portable than asking every operator to reconfigure
    their Docker daemon.
    """
    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        results = original(host, port, socket.AF_INET, type, proto, flags)
        return results or original(host, port, family, type, proto, flags)

    socket.getaddrinfo = ipv4_only


_force_ipv4()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agbe import config  # noqa: E402

LOCK_PATH = Path(__file__).resolve().parent.parent / "models.lock.json"

#: Read in 8 MiB blocks. The target machine has 8 GB of RAM and we may be
#: hashing a 2 GB file; reading it whole would be a self-inflicted OOM.
_HASH_BLOCK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    """A single downloadable file."""

    key: str
    repo_id: str
    filename: str
    dest: Path
    note: str


def artifacts() -> list[Artifact]:
    """Files fetched directly as-is.

    NLLB is deliberately absent: it ships as PyTorch weights and must be
    converted to CTranslate2 format first. That conversion needs torch, which
    is banned from the runtime image, so it happens in the separate converter
    image via `make convert-models`.
    """
    return [
        Artifact(
            key="llm",
            repo_id=config.LLM.repo_id,
            filename=config.LLM.filename,
            dest=config.LLM.path,
            note="Instruction model, Q4_K_M (~1.9 GB)",
        ),
        Artifact(
            key="embedding",
            repo_id=config.EMBEDDING.repo_id,
            filename=config.EMBEDDING.filename,
            dest=config.EMBEDDING.path,
            note="Retrieval embeddings, runs on the same llama.cpp runtime (~50 MB)",
        ),
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_BLOCK):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock() -> dict[str, dict]:
    if not LOCK_PATH.exists():
        return {}
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def save_lock(lock: dict[str, dict]) -> None:
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def download(art: Artifact, attempts: int = 5) -> Path:
    """Fetch one artifact into MODELS_DIR.

    huggingface_hub caches into its own blob layout with symlinks; we copy the
    resolved file to a stable path so the container bind-mount and the lock
    file both refer to something predictable.

    Retried with backoff because these are multi-gigabyte files and the
    connections this project is built for are slow. hf_hub_download resumes
    from its cache between attempts, so a retry continues rather than
    restarting - which matters when a single file is ~1.9 GB.
    """
    from huggingface_hub import hf_hub_download

    art.dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {art.repo_id}/{art.filename}")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            cached = hf_hub_download(
                repo_id=art.repo_id,
                filename=art.filename,
                # Long timeout: the default assumes a fast link and aborts a
                # download that is merely slow.
                etag_timeout=60,
            )
            cached_path = Path(cached)
            if art.dest.exists() and art.dest.samefile(cached_path):
                return art.dest
            shutil.copyfile(cached_path, art.dest)
            return art.dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            short = str(exc).split("\n")[0][:120]
            print(f"    attempt {attempt}/{attempts} failed: {short}")
            if attempt < attempts:
                delay = min(60, 5 * attempt)
                print(f"    retrying in {delay}s (download resumes from cache)")
                time.sleep(delay)

    raise RuntimeError(f"gave up after {attempts} attempts: {last_error}")


def verify(art: Artifact, lock: dict[str, dict]) -> tuple[bool, str]:
    """Check an on-disk file against its pinned hash."""
    if not art.dest.exists():
        return False, "missing"

    entry = lock.get(art.key)
    if entry is None:
        return False, "not pinned in models.lock.json (run --update-lock)"

    size = art.dest.stat().st_size
    if size != entry.get("size"):
        return False, f"size mismatch: got {human(size)}, expected {human(entry['size'])}"

    actual = sha256_of(art.dest)
    if actual != entry.get("sha256"):
        return False, f"SHA-256 mismatch\n      expected {entry['sha256']}\n      got      {actual}"

    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="record hashes of downloaded files into models.lock.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify files already on disk; download nothing",
    )
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lock = load_lock()
    arts = artifacts()

    print(f"models dir: {config.MODELS_DIR}")
    print(f"lock file : {LOCK_PATH}\n")

    failures: list[str] = []

    for art in arts:
        print(f"[{art.key}] {art.note}")

        if not args.verify_only and not art.dest.exists():
            try:
                download(art)
            except Exception as exc:  # noqa: BLE001 - surfaced to the operator
                print(f"  FAILED: {exc}\n")
                failures.append(f"{art.key}: download failed ({exc})")
                continue

        if args.update_lock:
            if not art.dest.exists():
                failures.append(f"{art.key}: cannot pin, file missing")
                print("  FAILED: cannot pin a file that is not on disk\n")
                continue
            lock[art.key] = {
                "repo_id": art.repo_id,
                "filename": art.filename,
                "sha256": sha256_of(art.dest),
                "size": art.dest.stat().st_size,
            }
            print(f"  pinned {lock[art.key]['sha256'][:16]}... "
                  f"({human(lock[art.key]['size'])})\n")
            continue

        ok, detail = verify(art, lock)
        print(f"  {'OK' if ok else 'FAILED'}: {detail}\n")
        if not ok:
            failures.append(f"{art.key}: {detail}")

    if args.update_lock:
        if failures:
            # A lock file that pins only the artifacts which happened to
            # download is worse than no lock file: it looks authoritative while
            # silently omitting whatever failed, and the next run would report
            # "not pinned" for the missing entries rather than "the last fetch
            # was incomplete".
            print("NOT writing the lock file - some artifacts failed to "
                  "download.\nRe-run once all downloads succeed.\n")
        else:
            save_lock(lock)
            print(f"wrote {LOCK_PATH.name} - commit this file\n")

    # The CT2 translation model is produced by a different pipeline; report its
    # status so an operator is not left wondering why Yoruba is unavailable.
    if config.TRANSLATION.path.exists():
        print(f"[translation] present at {config.TRANSLATION.path}")
    else:
        print("[translation] NOT BUILT - run `make convert-models` "
              "(needs the converter image; torch is not in the runtime image)")

    if failures:
        print("\n" + "=" * 60)
        print("FAILURES")
        for f in failures:
            print(f"  - {f}")
        print("=" * 60)
        return 1

    print("\nall model artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

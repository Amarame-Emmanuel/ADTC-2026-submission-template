#!/usr/bin/env python3
"""Capture Gate 1 screenshots by driving the real interface.

WHY A BROWSER AND NOT AN IMAGE EDITOR
-------------------------------------
Every screenshot here is a photograph of the actual system answering an actual
question, produced by typing into the real UI and waiting for the real streamed
response. Nothing is staged, and no answer text is injected into the page.

That distinction is the whole point. A screenshot is a *record* of behaviour. A
hand-composed one showing the answer we wish the model gave would be a
fabrication, and it would be indistinguishable from the real thing to a judge -
which is exactly why it must not be done.

The consequence is that these images show whatever the system actually does,
including the refusals. The dosage refusal shot is deliberately included: a
reviewer should see that the system declines to invent a pesticide dose, because
that is a feature and the most important one.

WHY IT WAITS SO LONG
--------------------
Time to first token is ~22 s on this hardware and generation runs to completion
after that, so each capture allows generous time. The wait is on the UI
signalling completion, not on a fixed sleep - a fixed sleep would silently
photograph a half-streamed answer the day the machine is busy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import time
import urllib.request

from playwright.sync_api import sync_playwright

#: Viewport chosen to look like a modest laptop rather than a desktop monitor,
#: because that is the machine this project targets.
VIEWPORT = {"width": 1280, "height": 900}

#: Generation can take a couple of minutes for a long answer on four pinned
#: cores. Well above the measured p95.
ANSWER_TIMEOUT_MS = 300_000


SHOTS = [
    {
        "name": "01-landing",
        "question": None,
        "caption": "The interface on load. No question asked yet.",
    },
    {
        "name": "02-crop-diagnosis",
        "question": "My cassava leaves are yellow and twisted, and the plants are small.",
        "caption": "Crop advisory with the sources each claim came from.",
    },
    {
        "name": "03-livestock",
        "question": "My chickens are dying with twisted necks and greenish diarrhoea.",
        "caption": "Livestock advisory - the Newcastle disease case.",
    },
    {
        "name": "04-dosage-refusal",
        "question": "How many ml of insecticide should I put in my 20 litre sprayer?",
        "caption": "The safety layer refusing to invent a pesticide dose.",
    },
    {
        "name": "05-pidgin",
        "question": "My goat no dey chop again and e dey shake. Wetin I go do?",
        "caption": "A question asked in Nigerian Pidgin.",
    },
]


def capture(page, shot: dict, outdir: Path) -> dict:
    if shot["question"] is not None:
        page.fill("#q", shot["question"])
        page.click("#send")

        # Wait for the answer to finish rather than for a fixed duration. The
        # send button re-enables when the stream closes, which is the UI's own
        # signal that it is done - more reliable than watching text stop
        # changing, which cannot distinguish "finished" from "slow".
        page.wait_for_function(
            "() => !document.querySelector('#send').disabled",
            timeout=ANSWER_TIMEOUT_MS,
        )
        page.wait_for_timeout(700)  # let the final repaint settle

    path = outdir / f"{shot['name']}.png"
    page.screenshot(path=str(path), full_page=True)

    answer = page.inner_text("#answer").strip()
    sources = page.inner_text("#sources").strip() if page.is_visible("#sources") else ""
    print(f"  {path.name:26} {len(answer):>5} chars"
          f"{'  (+sources)' if sources else ''}")
    return {
        "file": path.name,
        "question": shot["question"],
        "caption": shot["caption"],
        "answer_chars": len(answer),
        "had_sources": bool(sources),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://agbe-ui:8000")
    ap.add_argument("--out", default="/out")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Wait for the server here, in a real loop. The first version of this wait
    # lived in the Makefile as a python -c one-liner built around a list
    # comprehension - and urlopen() RAISES on a server that is not up yet, so
    # the exception aborted the comprehension on the first attempt and the
    # `|| true` after it swallowed the failure. A retry loop that cannot retry,
    # hidden by the operator that made it optional.
    deadline = time.monotonic() + 180
    while True:
        try:
            urllib.request.urlopen(f"{args.url}/health", timeout=2)
            break
        except Exception:
            if time.monotonic() > deadline:
                print(f"server at {args.url} never became healthy", file=sys.stderr)
                return 1
            time.sleep(2)

    print(f"capturing from {args.url}")
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for shot in SHOTS:
            # A fresh page per shot so each screenshot shows one interaction
            # rather than an accumulating transcript.
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
            page.goto(args.url, wait_until="networkidle", timeout=60_000)
            try:
                results.append(capture(page, shot, outdir))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {shot['name']:26} FAILED: {exc}", file=sys.stderr)
            page.close()
        browser.close()

    index = outdir / "README.md"
    lines = [
        "# Screenshots",
        "",
        "Captured by `scripts/screenshots.py` driving the real interface against",
        "the real engine. Every answer shown was generated by the system at",
        "capture time; none is staged or edited.",
        "",
    ]
    for r in results:
        lines += [f"### {r['file']}", "", r["caption"], ""]
        if r["question"]:
            lines += [f"> {r['question']}", ""]
        lines += [f"![{r['caption']}]({r['file']})", ""]
    index.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{len(results)}/{len(SHOTS)} captured -> {outdir}")
    return 0 if len(results) == len(SHOTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

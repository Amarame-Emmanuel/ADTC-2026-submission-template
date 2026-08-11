#!/usr/bin/env python3
"""Deprecated wrapper. The review-sheet flow lives in scripts/lang_sheet.py.

This file once held the whole Pidgin export/import implementation. When
Yoruba and Igbo reviews needed the same flow, the choice was copy or extract -
and this project has measured, four times, what happens to copies. Extracted.

Kept as a wrapper because the Pidgin review instructions in circulation name
this script; both spellings do the same thing:

    python scripts/pidgin_sheet.py --export
    python scripts/lang_sheet.py --language pcm --export
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lang_sheet import main as _main  # noqa: E402


def main() -> int:
    if "--language" not in sys.argv:
        sys.argv[1:1] = ["--language", "pcm"]
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

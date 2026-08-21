"""Source files must not contain control characters.

WHY THIS FILE EXISTS
--------------------
A regex was written as `rains?\\b` and reached disk as `rains?<0x08>` - a literal
BACKSPACE byte where the escape belonged. The pattern compiled, matched nothing,
and the fix it was part of silently did not work. It read correctly in an editor
and in `grep` output, because a terminal renders 0x08 by moving the cursor back
rather than by showing anything.

That has happened more than once in this project, always the same way: a `\\b`
passing through a shell heredoc into a Python string literal, losing one level
of escaping on the way. The failure mode is what makes it worth a test - a
corrupted pattern raises nothing, fails no assertion of its own, and simply
stops matching.

WHAT COUNTS AS A CONTROL CHARACTER
----------------------------------
Everything below 0x20 except tab, newline and carriage return, plus DEL. Those
three are the only ones a source file has any business containing.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Tab, newline, carriage return. Nothing else below 0x20 is legitimate.
ALLOWED = {0x09, 0x0A, 0x0D}


def python_sources() -> list[Path]:
    """Every source file this project owns, excluding vendored trees."""
    out: list[Path] = []
    for path in ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {"profiler", "__pycache__", ".git", "model2", "corpus"}:
            continue
        out.append(path)
    return sorted(out)


def test_there_are_sources_to_check() -> None:
    """A glob that silently matches nothing would make this suite vacuous."""
    assert len(python_sources()) > 20


@pytest.mark.parametrize("path", python_sources(), ids=lambda p: p.name)
def test_no_control_characters(path: Path) -> None:
    text = io.open(path, encoding="utf-8", newline="").read()
    bad = [
        (i, ord(ch))
        for i, ch in enumerate(text)
        if ord(ch) < 0x20 and ord(ch) not in ALLOWED or ord(ch) == 0x7F
    ]
    if bad:
        i, code = bad[0]
        context = text[max(0, i - 40):i + 10].replace("\n", " ")
        pytest.fail(
            f"{path.relative_to(ROOT)} contains {len(bad)} control character(s); "
            f"first is 0x{code:02X} at offset {i}, near: ...{context}..."
        )

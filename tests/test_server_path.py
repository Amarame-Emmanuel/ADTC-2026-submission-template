"""The web server must be a consumer of the tested path, not a reimplementation.

WHY THIS FILE EXISTS
--------------------
`/ask` carried its own copy of the generation loop for most of this project's
life, and the copy drifted: it missed the context-compression fix (the browser
kept 65 s TTFT after the benchmark said 22 s), missed the query-preparation fix
of REPORT §6.3 (pasting the submitted test prompt into the UI reproduced the
exact bug the fix was for), and the interface hardcoded `language: "en"`,
disabling Pidgin detection the server itself supported.

Nothing caught any of that, and the reason is recorded in requirements-dev.txt:
httpx was added with the comment "TestClient needs httpx; without it the UI
cannot be exercised in tests at all" - and then no server test was ever
written. The dependency for this file predates the file. Every path with a
test absorbed the fixes; the one path without a test absorbed none of them.

These tests deliberately avoid TestClient anyway: they call the endpoint
function and consume the StreamingResponse directly, so they need nothing
beyond what the runtime image already ships and cannot be skipped for a
missing dev dependency.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agbe.rag.chunker import Chunk
from agbe.rag.index import SearchHit
from agbe.ui import server
from agbe.ui.server import Question, ask


def _chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id="doc-test",
        title="Goat husbandry field guide",
        publisher="Test Extension Service",
        year="2023",
        licence="CC BY 4.0",
        source_url="https://example.org/goats",
        page_start=3,
        page_end=3,
        chunk_index=0,
    )


class StubEngine:
    """Records which engine entry points the server touches.

    The assertions in this file are about *routing*, not model output: the
    server must reach retrieval through `retrieve_and_compress` (the entry
    point carrying Pidgin normalisation, instruction stripping and
    compression) and generation through `guarded_stream` (the one dosage-
    guarded loop). A stub proves routing without loading a model.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.pieces = ["Feed the goats ", "fresh forage ", "twice daily."]

    def retrieve_and_compress(self, question: str, top_k=None, language="en"):
        self.calls.append("retrieve_and_compress")
        hits = [SearchHit(chunk=_chunk("Goats need fresh forage twice daily."),
                          score=0.81, rank=0)]
        return hits, ["Goats need fresh forage twice daily."], {"tokens_after": 12}

    def guarded_stream(self, question, hits, texts, no_guidance=None):
        self.calls.append("guarded_stream")
        for piece in self.pieces:
            yield piece, None

    def _scope_message(self, verdict, messages):
        return "scope-refusal-message"


def _events(question: Question) -> list[tuple[str, dict]]:
    """Drive /ask and parse its SSE stream into (event, data) pairs."""

    async def run():
        resp = await ask(question, request=None)
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(chunks)

    raw = asyncio.run(run())
    out = []
    for block in raw.split("\n\n"):
        lines = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "event" in lines and "data" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture()
def stub(monkeypatch):
    engine = StubEngine()
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    return engine


def test_ask_routes_through_the_shared_entry_points(stub):
    events = _events(Question(text="My goats are not eating well."))

    assert stub.calls == ["retrieve_and_compress", "guarded_stream"], (
        "the server must consume the tested path, not reimplement it - "
        f"got {stub.calls}"
    )

    tokens = [d["t"] for e, d in events if e == "token"]
    assert tokens == stub.pieces, (
        "every streamed token must come from guarded_stream, unmodified"
    )
    assert any(e == "sources" for e, _ in events)
    assert any(e == "done" for e, _ in events)


def test_ask_autodetects_pidgin_when_language_is_auto(stub):
    """The interface sends language: "auto". It once sent "en", which turned
    the server's working detection off for every real user."""
    events = _events(
        Question(text="My goat no dey chop again, wetin dey worry am?")
    )
    languages = [d["language"] for e, d in events if e == "language"]
    assert languages == ["pcm"]


def test_ask_refuses_dosage_before_touching_retrieval(stub):
    events = _events(
        Question(text="How many ml of insecticide for my 20 litre sprayer?")
    )
    assert stub.calls == [], "a scope refusal must never reach retrieval"
    assert any(e == "refused" for e, _ in events)


def test_default_language_is_auto():
    """The pydantic default is the contract the interface relies on."""
    assert Question(text="hello").language == "auto"

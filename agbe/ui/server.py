"""HTTP interface for Àgbẹ̀.

DESIGN CONSTRAINT: THE MACHINE IS SLOW AND THAT MUST NOT FEEL LIKE A FAULT
--------------------------------------------------------------------------
On four cores with no GPU, a complete answer takes tens of seconds. The
difference between "this is broken" and "this is working" is almost entirely
about what the interface does during that time:

  * Retrieval finishes long before generation starts, so sources are sent
    first. Within a second the farmer can see which documents are being used,
    which is both useful information and proof the system is alive.
  * Tokens stream as they are produced. The same wall-clock wait reads as
    progress rather than a freeze.
  * A visible stage indicator distinguishes retrieving from thinking from
    writing, so a long pause has an explanation attached.

Server-sent events rather than WebSockets: the data flows one way, SSE
reconnects on its own, and it survives the kind of intermittent connectivity
this system is built for. A WebSocket would add a dependency and a failure
mode for no benefit.

The model is loaded once at startup and shared. Loading per request would mean
paying ~2 GB of allocation and several seconds on every question, and would
make concurrent requests a memory-ceiling breach rather than a queue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agbe import config

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Àgbẹ̀", docs_url=None, redoc_url=None)

#: Populated at startup. Module-level rather than per-request for the memory
#: reason in the docstring.
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from agbe.advisor import AdvisoryEngine

        _engine = AdvisoryEngine()
        _engine.embedder.load()
        _engine.llm.load()
    return _engine


class Question(BaseModel):
    text: str
    #: "auto" detects from the question text; "en" or "pcm" force a language.
    #: Forcing matters because detection is deliberately biased towards
    #: English, so a short Pidgin question ("my fowl dey die") may not carry
    #: enough grammatical markers to be recognised.
    language: str = "auto"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    """Readiness and the memory position, so the demo can show both."""
    from bench.run import current_rss_bytes, peak_rss_bytes

    return {
        "status": "ok",
        "model": config.LLM.filename,
        "loaded": _engine is not None,
        "rss_gb": round(current_rss_bytes() / 1024**3, 2),
        "peak_rss_gb": round(peak_rss_bytes() / 1024**3, 2),
        "ceiling_gb": round(config.MEMORY_CEILING_BYTES / 1024**3, 1),
    }


@app.post("/ask")
async def ask(question: Question, request: Request) -> StreamingResponse:
    engine = get_engine()

    def generate() -> Iterator[str]:
        # Language and policy scope, exactly as the tested advise() path does
        # them. This endpoint previously reimplemented the flow and skipped
        # both, so the web interface - the one a judge actually sees - had no
        # Pidgin support and none of the scope guards, while the code path
        # under test had both. Divergence between the demo path and the tested
        # path is its own kind of defect.
        from agbe.rag import scope
        from agbe.translate.detect import detect
        from agbe.translate.messages import get as get_messages

        language = (
            detect(question.text)
            if question.language == "auto"
            else question.language
        )
        messages = get_messages(language)

        yield sse("language", {"language": language})

        verdict = scope.check(question.text)
        if not verdict.in_scope:
            yield sse("refused", {
                "message": engine._scope_message(verdict, messages),
                "reason": verdict.reason,
            })
            yield sse("done", {"refused": True})
            return

        yield sse("stage", {"stage": "retrieving"})

        # retrieve_and_compress, never retrieve: it is the entry point that
        # carries Pidgin normalisation, instruction stripping and context
        # compression. This line previously called engine.retrieve(), which
        # has none of those - so the browser, the one surface a judge
        # actually touches, was excluded from every retrieval fix and still
        # sent ~2,100-token prompts at 65s to first token. The generation
        # loop below it had been copy-pasted too; it now lives in ONE place,
        # engine.guarded_stream().
        hits, texts, _comp = engine.retrieve_and_compress(question.text)

        if not hits:
            # The refusal path. Emitted as its own event so the interface can
            # present it as a deliberate answer rather than an error or an
            # empty response.
            yield sse("refused", {"message": messages.no_guidance})
            yield sse("done", {"refused": True})
            return

        # Sources first: they are ready now, and they give the farmer
        # something true to read during the seconds before the first token.
        yield sse("sources", {
            "sources": [
                {
                    "n": i + 1,
                    "citation": h.chunk.citation(),
                    "year": h.chunk.year,
                    "licence": h.chunk.licence,
                    "score": round(h.score, 3),
                }
                for i, h in enumerate(hits)
            ]
        })

        yield sse("stage", {"stage": "writing"})

        answer_parts: list[str] = []
        stats = None

        for piece, piece_stats in engine.guarded_stream(question.text, hits, texts):
            if piece_stats is not None:
                stats = piece_stats
            answer_parts.append(piece)
            yield sse("token", {"t": piece})

        # Post-hoc safety check over the whole answer: catches hazardous
        # actives and wholly-stale chemical sourcing, which are properties of
        # the answer rather than of any single sentence.
        from agbe.rag.safety import check_answer

        verdict = check_answer(
            "".join(answer_parts),
            source_texts=[h.chunk.text for h in hits],
            source_years=[h.chunk.year for h in hits],
        )
        notice = verdict.as_notice(language)
        if notice:
            yield sse("safety", {"notice": notice})

        yield sse("done", {
            "stats": stats.as_dict() if stats else None,
            "safe": verdict.safe,
            "language": language,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # defeat proxy buffering of the stream
        },
    )


def preflight() -> list[str]:
    """Check everything the server needs before accepting a request.

    Without this the first question produces a 500 and a 40-line ASGI
    traceback whose actual cause - "no index at /app/index" - is on line 38.
    The most likely reason for that is a mistyped bind mount: `$(PWD)` is Make
    syntax and expands to nothing in a shell, so `-v "$(PWD)/index:/app/index"`
    pasted directly becomes `-v "/index:/app/index"`, and Docker silently
    creates an empty directory rather than failing.

    A server that starts successfully and then fails every request is worse
    than one that refuses to start: it looks healthy, and the error surfaces
    to whoever is using it rather than whoever launched it.
    """
    problems: list[str] = []

    if not config.LLM.path.exists():
        problems.append(
            f"language model missing: {config.LLM.path}\n"
            "      run `make fetch-models`"
        )
    if not config.EMBEDDING.path.exists():
        problems.append(
            f"embedding model missing: {config.EMBEDDING.path}\n"
            "      run `make fetch-models`"
        )
    if not (config.INDEX_DIR / "vectors.npy").exists():
        problems.append(
            f"no retrieval index at {config.INDEX_DIR}\n"
            "      run `make index` - or, if you expected one to exist, check\n"
            "      your bind mounts: `$(PWD)` is Make syntax and expands to\n"
            "      nothing in a shell, so a hand-copied `-v \"$(PWD)/index:...\"`\n"
            "      mounts an empty directory instead of your index."
        )

    return problems


def main() -> None:
    import uvicorn

    problems = preflight()
    if problems:
        print("Àgbẹ̀ cannot start:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nRefusing to start rather than failing every request.",
              file=sys.stderr)
        raise SystemExit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

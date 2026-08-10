"""Language model wrapper.

Thin by design. The interesting decisions in this system live in retrieval,
grounding and safety; this module's only jobs are to load a quantized model
under a memory budget, stream tokens, and report the timings the benchmark
harness needs.

WHY STREAMING IS NOT OPTIONAL
-----------------------------
On the target hardware - four cores, no GPU, DDR4 - generation runs at single
digit to low double digit tokens per second. A 250-token answer therefore
takes tens of seconds. Delivered as one block at the end, that reads as a
frozen application; delivered token by token, the same wall-clock time reads
as a system that is working.

Time to first token is the number that actually governs perceived
responsiveness, and it is dominated by prompt processing, not generation. That
is why retrieval sends four passages rather than ten: every retrieved chunk is
prefill work the farmer waits through before seeing a single word.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from agbe import config


@dataclass
class GenerationStats:
    """Timings for one generation, consumed by the benchmark harness."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_s: float = 0.0
    total_time_s: float = 0.0
    stopped_early: bool = False
    fields: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Generation rate excluding prefill.

        Measured after the first token so prompt processing does not
        contaminate it - mixing the two produces a number that improves when
        the prompt gets shorter and tells you nothing about generation.
        """
        gen_time = self.total_time_s - self.time_to_first_token_s
        return self.completion_tokens / gen_time if gen_time > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_s": round(self.time_to_first_token_s, 3),
            "total_s": round(self.total_time_s, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "stopped_early": self.stopped_early,
            **self.fields,
        }


class LLM:
    """Quantized instruction model on llama.cpp."""

    def __init__(
        self,
        model_path: Path | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path or config.LLM.path)
        self.n_ctx = n_ctx or config.LLM.n_ctx
        self.n_threads = n_threads or config.LLM.n_threads
        self._llm = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model missing: {self.model_path}\nrun `make fetch-models` first"
            )

        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            # Prefill batch. Larger batches speed up prompt processing but
            # enlarge the compute buffer, which is resident for the whole
            # session. 512 is the point where the buffer stops paying for
            # itself on a four-core CPU.
            n_batch=512,
            # The KV cache is the one allocation that grows with conversation
            # length. Reserving it up front means a long session cannot creep
            # over the ceiling mid-demo; we would rather fail at load than at
            # the worst possible moment.
            use_mlock=False,
            use_mmap=True,
            verbose=False,
        )

    def warm(self) -> float:
        """Force the weights resident and the compute buffers allocated.

        Weights are mmap'd, so they page in from disk lazily on first use. That
        cost lands entirely on whoever asks the first question: measured here at
        **93 seconds** to first token, against ~22 seconds once warm. The
        difference is disk I/O, not inference.

        A farmer opening the app and waiting a minute and a half concludes it is
        broken. Doing that work at startup - while a splash screen is showing
        and nobody is waiting on an answer - moves the cost somewhere it does no
        harm.

        Returns seconds taken, so startup logs can report it.
        """
        self.load()
        assert self._llm is not None

        started = time.perf_counter()
        self._llm.create_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0.0,
        )
        return time.perf_counter() - started

    def close(self) -> None:
        self._llm = None
        gc.collect()

    def __enter__(self) -> "LLM":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- generation --------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        self.load()
        assert self._llm is not None
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[tuple[str, GenerationStats]]:
        """Yield (token_text, stats) as generation proceeds.

        Stats are updated in place and are only complete once the iterator is
        exhausted; the caller gets a reference on every yield so a UI can show
        live throughput without a second channel.
        """
        self.load()
        assert self._llm is not None

        stats = GenerationStats()
        started = time.perf_counter()
        first_token_at: float | None = None

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.LLM.max_tokens,
            temperature=temperature if temperature is not None else config.LLM.temperature,
            top_p=top_p if top_p is not None else config.LLM.top_p,
            stop=stop or [],
            stream=True,
        )

        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if not piece:
                continue

            if first_token_at is None:
                first_token_at = time.perf_counter()
                stats.time_to_first_token_s = first_token_at - started

            stats.completion_tokens += 1
            stats.total_time_s = time.perf_counter() - started
            yield piece, stats

        stats.total_time_s = time.perf_counter() - started
        if first_token_at is None:
            # Model produced nothing at all. Surfaced rather than swallowed:
            # silent empty output is indistinguishable from a hung process.
            stats.stopped_early = True

    def complete(self, messages: list[dict[str, str]], **kwargs) -> tuple[str, GenerationStats]:
        """Non-streaming convenience wrapper. Used by benchmarks and tests."""
        pieces: list[str] = []
        stats = GenerationStats()
        for piece, stats in self.stream(messages, **kwargs):
            pieces.append(piece)
        return "".join(pieces), stats

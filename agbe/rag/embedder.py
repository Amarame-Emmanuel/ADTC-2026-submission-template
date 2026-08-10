"""Passage and query embeddings, served by llama.cpp.

WHY LLAMA.CPP AND NOT SENTENCE-TRANSFORMERS
-------------------------------------------
sentence-transformers is the obvious choice and the wrong one here. It pulls
PyTorch, which costs roughly 700 MB-1 GB resident before a single weight
loads - about 14% of the entire 7 GB budget spent on a framework whose only
job would be a 33M-parameter encoder.

bge-small-en-v1.5 runs as a GGUF through the llama.cpp runtime we already load
for the language model. Cost: ~50 MB and no new framework to build, audit or
ship. This is the single largest memory saving in the system.

THE ASYMMETRY THAT MATTERS
--------------------------
BGE models are trained with an instruction prefix on the *query* side only.
Embedding a query without it, or a passage with it, degrades retrieval
measurably - and silently, which is worse. `embed_query` and `embed_passages`
are separate methods rather than one method with a flag so that the asymmetry
is impossible to get wrong at a call site.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from agbe import config

#: The prefix BGE v1.5 was trained with. Queries only - never passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Thin wrapper over a GGUF embedding model.

    Loaded lazily: constructing this object should be cheap, because the
    server builds its object graph at import time and we do not want to pay
    50 MB plus load latency before the first request that actually needs it.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path or config.EMBEDDING.path)
        self.n_ctx = n_ctx or config.EMBEDDING.n_ctx
        self.n_threads = n_threads or config.EMBEDDING.n_threads
        self.dim = config.EMBEDDING.dim
        self._llm = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"embedding model missing: {self.model_path}\n"
                "run `make fetch-models` first"
            )

        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(self.model_path),
            embedding=True,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_ctx,
            verbose=False,
        )

    def close(self) -> None:
        """Release the model.

        Indexing loads the embedder and the corpus at once; serving needs the
        language model resident at the same time. Being able to hand memory
        back explicitly - rather than waiting for the garbage collector to
        notice - is what keeps the peak below the ceiling.
        """
        self._llm = None
        gc.collect()

    def __enter__(self) -> "Embedder":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- embedding ---------------------------------------------------------

    def _embed_raw(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch in a single llama.cpp call.

        create_embedding accepts a list. An earlier version passed one string
        per call inside a Python loop, paying the Python-to-C crossing and a
        context reset 6,939 times during a full index build. The forward passes
        dominate either way - this is CPU inference on a 33M-parameter encoder -
        but there is no reason to add per-item overhead on top of them.
        """
        self.load()
        assert self._llm is not None
        if not texts:
            return np.zeros((0, self.dim), np.float32)

        result = self._llm.create_embedding(list(texts))

        vectors: list[np.ndarray] = []
        for item in result["data"]:
            vec = np.asarray(item["embedding"], dtype=np.float32)
            # llama.cpp returns token-level embeddings for some model
            # architectures and a single pooled vector for others. Mean-pool
            # when we get a matrix so both shapes are handled.
            if vec.ndim == 2:
                vec = vec.mean(axis=0)
            vectors.append(vec)

        return np.vstack(vectors)

    @staticmethod
    def normalise(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise rows so cosine similarity is a plain dot product.

        Normalising once at index time turns every later similarity search
        into a single matrix multiply, which is what makes brute-force search
        fast enough to avoid an ANN index entirely.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        return matrix / norms

    def embed_passages(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        progress: bool = False,
    ) -> np.ndarray:
        """Embed corpus passages. No prefix - see module docstring.

        `progress` prints a live rate and ETA. Indexing thousands of passages
        on four CPU cores takes tens of minutes, and a silent process for that
        long is indistinguishable from a hung one - which cost most of an hour
        here before anyone could tell whether it was working.
        """
        out: list[np.ndarray] = []
        total = len(texts)
        started = time.perf_counter()

        for start in range(0, total, batch_size):
            out.append(self._embed_raw(texts[start : start + batch_size]))

            if progress:
                done = min(start + batch_size, total)
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(
                    f"\r  embedding {done}/{total} "
                    f"({100 * done / total:5.1f}%)  "
                    f"{rate:5.1f} chunks/s  eta {eta / 60:5.1f} min",
                    end="",
                    flush=True,
                )

        if progress and total:
            print()

        if not out:
            return np.zeros((0, self.dim), np.float32)
        return self.normalise(np.vstack(out))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query, with the BGE instruction prefix."""
        return self.normalise(self._embed_raw([QUERY_PREFIX + text]))[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.normalise(self._embed_raw([QUERY_PREFIX + t for t in texts]))

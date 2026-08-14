"""Central configuration for Àgbẹ̀.

Every knob that moves the memory needle lives in this file. The 7 GB ceiling
from the ADTC hardware standard is a disqualification threshold, not a
performance target, so it is expressed here as a constant that the watchdog
(`bench/watchdog.py`) enforces and that `MEMORY_BUDGET` documents component by
component.

The budget below is an *estimate*. It is superseded by measured peak RSS from
`make bench`, which is what REPORT.md cites. If the two disagree, the
measurement is right and this table is stale - fix the table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("AGBE_MODELS_DIR", ROOT / "models"))
CORPUS_DIR = Path(os.environ.get("AGBE_CORPUS_DIR", ROOT / "corpus"))
INDEX_DIR = Path(os.environ.get("AGBE_INDEX_DIR", ROOT / "index"))


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------

#: Hard limit from the ADTC hardware standard. Exceeding it is disqualifying
#: (S_total = 0), so anything measuring against this treats a breach as a test
#: failure rather than a warning.
MEMORY_CEILING_BYTES: int = 7 * 1024**3

#: We hold ourselves to a stricter internal budget than the disqualification
#: line. The gap absorbs allocator fragmentation, page-cache pressure and the
#: fact that our dev hardware cannot reproduce the target's memory bandwidth or
#: its background OS load. A run that exceeds this is a bug to investigate even
#: though it would not itself be disqualifying.
MEMORY_TARGET_BYTES: int = 5 * 1024**3


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

#: The reference laptop is a 4-core / 8-thread i5. Our development box has 24
#: threads, so leaving this unset would silently benchmark a machine nobody in
#: the target market owns. Every entry point pins it.
DEFAULT_THREADS: int = int(os.environ.get("AGBE_THREADS", "4"))


@dataclass(frozen=True)
class LLMConfig:
    """Core instruction-following model.

    Q4_K_M rather than a smaller quant: Q4_K_M is the knee of the
    quality/size curve for 3B-class models, and we have the headroom for it
    once torch is out of the picture. Q3 variants save ~400 MB but degrade
    instruction-following noticeably, which matters more here than the RAM,
    because a wrong agronomic instruction is worse than a slow one.
    """

    #: Qwen2.5-1.5B, chosen on measurement rather than assumption.
    #:
    #: ARC-Easy over 50 samples - the profiler's own benchmark and sample count
    #: - gave 0.740 acc / 0.780 acc_norm against the 3B's 0.740 / 0.760. The 3B
    #: showed no accuracy advantage while costing 10.7 points of S_eff (2.47 GB
    #: peak RSS against 1.71 GB, both full-application) and half the throughput.
    #: It needed to win accuracy by >4.3 points to break even and won by zero.
    #: See REPORT.md §3.3.
    repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

    #: Overridable so the A/B in §3.3 can be re-run against another GGUF in
    #: models/ without editing source. That comparison was previously only
    #: reproducible by patching this line, which is why its two columns ended up
    #: measured on different harnesses and had to be re-stated as not
    #: like-for-like. A reviewer should be able to reproduce a claim with an
    #: environment variable, not a diff.
    filename: str = os.environ.get(
        "AGBE_LLM_FILENAME", "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )

    #: Sized from what the pipeline actually sends, not from what the model
    #: could accept. Measured prompts run ~773 tokens (retrieval is capped at
    #: `RetrievalConfig.context_token_budget`, and top_k is fixed), and
    #: `max_tokens` below caps the answer at 512. Worst case is therefore under
    #: 1,300 tokens, so 2048 leaves ~750 tokens of headroom.
    #:
    #: This was 4096, which cost ~60 MB of KV cache reserved for context the
    #: system has no path to producing. Qwen2.5's grouped-query attention (2 KV
    #: heads) makes the cache cheap per token, which is exactly why the waste
    #: went unnoticed - cheap per token is not the same as free.
    #:
    #: The cache is reserved up front rather than grown on demand: we would
    #: rather fail at load than creep over the ceiling mid-demo.
    n_ctx: int = 2048
    n_threads: int = DEFAULT_THREADS

    #: Low temperature: this is an advisory system where a confidently
    #: invented pesticide dosage is an actual safety failure, not a style
    #: preference.
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


@dataclass(frozen=True)
class EmbeddingConfig:
    """Retrieval embeddings, served by the *same* llama.cpp runtime as the LLM.

    This is the single biggest architectural saving in the system. The obvious
    choice - sentence-transformers - pulls PyTorch, which costs roughly
    700 MB-1 GB resident before a single weight loads. Running bge-small as a
    GGUF through llama.cpp costs ~50 MB and adds no new framework to audit,
    build or ship.

    Queries are embedded in English, and this is a real limitation rather than
    a design choice we got for free: a question asked in Nigerian Pidgin lands
    in the wrong part of the embedding space and retrieves poorly. Pidgin
    speakers get validated refusals and safety warnings in their language, not
    answers. See REPORT.md §7.
    """

    repo_id: str = "CompendiumLabs/bge-small-en-v1.5-gguf"
    filename: str = "bge-small-en-v1.5-q8_0.gguf"
    dim: int = 384
    n_ctx: int = 512
    n_threads: int = DEFAULT_THREADS

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


@dataclass(frozen=True)
class TranslationConfig:
    """Translation bridge - PRESENT BUT NOT SHIPPED.

    Configuration for NLLB-200-distilled-600M via CTranslate2. Nothing in the
    running system uses it: `make convert-models` was never run, so the model
    does not exist on disk and no code path loads it.

    It is kept because the bridge is the right way to make Pidgin *questions*
    work - the embedder is English-only, so a question asked in Pidgin
    currently retrieves poorly. What ships instead is human-validated fixed
    Pidgin for every message the system emits (agbe/translate/messages.py),
    which is strictly better for safety text and costs no memory at all.

    Do not treat the presence of this config as a claim of Yoruba, Hausa or
    Igbo support. None are wired or validated. See REPORT.md §3.2 and §9.
    """

    repo_id: str = "facebook/nllb-200-distilled-600M"
    ct2_dirname: str = "nllb-200-distilled-600M-int8"
    quantization: str = "int8"
    src_lang_yo: str = "yor_Latn"
    src_lang_en: str = "eng_Latn"
    n_threads: int = DEFAULT_THREADS

    #: Loaded on demand and released when idle. An English-only session should
    #: not pay 600 MB for a translator it never calls.
    lazy_load: bool = True

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.ct2_dirname


@dataclass(frozen=True)
class RetrievalConfig:
    """Chunking and retrieval parameters.

    `top_k` is small on purpose. Every retrieved chunk is prefill tokens, and
    prefill is compute-bound on a CPU-only machine - it is the dominant term in
    time-to-first-token, which is the latency the user actually feels. Four
    chunks at ~450 tokens keeps us near 2k prompt tokens including the system
    preamble.
    """

    chunk_tokens: int = 450
    chunk_overlap_tokens: int = 60

    #: Measured on the dev split against the full 31,682-chunk index.
    #:
    #: Raised from 4 after the corpus grew 4x. Counter-intuitively, expanding
    #: the corpus made retrieval WORSE at top_k=4: every document added is also
    #: a competitor for the same four slots, so passages that previously placed
    #: were pushed out. Two questions that passed before the expansion started
    #: failing.
    #:
    #:     top_k   coverage   refusal   prompt   TTFT
    #:       4       83.3%      100%     710     17.5s
    #:       6       93.3%      100%     838     20.7s
    #:       8       93.3%      100%       -        -
    #:      12       93.3%      100%       -        -
    #:
    #: 6 is the knee - nothing above it helps. It buys 10 points of coverage
    #: for 3.2 seconds, and costs far less prefill than the passage count
    #: suggests because `context_token_budget` compresses six passages harder
    #: rather than concatenating them: the prompt grows 18%, not 50%.
    top_k: int = 6

    #: Chunks scoring below this are dropped even if they are in the top k.
    #: An empty retrieval is a valid, safe outcome: the system then says it has
    #: no local guidance rather than answering from parametric memory.
    #:
    #: MEASURED, not guessed. The original 0.35 was a guess and it was badly
    #: wrong: at that floor "How do I fix my motorcycle engine?" scored 0.41
    #: and returned passages, and 0 of 3 out-of-scope questions were refused.
    #:
    #: bge-small has a compressed similarity range - unrelated text still
    #: scores 0.4-0.5 - so any floor below ~0.55 accepts everything. Swept on
    #: the DEV split (bench/split.py), never on the reported test set:
    #:
    #:     floor   coverage   refusal
    #:     0.55      100%        0%
    #:     0.62      100%       33%
    #:     0.70       88%      100%
    #:
    #: 0.70 costs 12.5% false refusals to buy correct refusal of every
    #: out-of-scope question. That trade is deliberate: a false refusal sends a
    #: farmer to their extension officer, while a false answer about a
    #: pesticide or a sick animal can cost them the crop or the animal.
    min_score: float = 0.70

    #: Total tokens of retrieved context allowed into the prompt after
    #: sentence-level compression (agbe/rag/compress.py).
    #:
    #: This is the single most important latency knob in the system. Time to
    #: first token is prompt_tokens / prefill_rate, and prefill already runs at
    #: ~70% of hardware peak, so the numerator is the only lever. Uncompressed
    #: retrieval produced ~1,456 prompt tokens and 74s TTFT on development
    #: hardware - projecting to ~125s on the slower reference laptop.
    #:
    #: 700 tokens of context plus a ~70-token system prompt targets roughly
    #: 800 total, or about 40s here and 65s on target. Lowering it further
    #: trades answer grounding for latency; the coverage evaluation is what
    #: says how much of that trade is affordable.
    context_token_budget: int = 700


LLM = LLMConfig()
EMBEDDING = EmbeddingConfig()
TRANSLATION = TranslationConfig()
RETRIEVAL = RetrievalConfig()


#: Component-by-component estimate, in bytes, used by `bench` to report
#: predicted-vs-measured. Keeping the prediction in code means a regression
#: shows up as a diff rather than as a stale paragraph in a report.
#:
#: That is exactly what went wrong with the previous version of this dict, so
#: the failure is worth recording where it happened. It kept budgeting 1.90 GB
#: of weights for the 3B rejected in REPORT.md 3.3, and 620 MB for the NLLB
#: bridge rejected in 3.2, long after both decisions were made. It therefore
#: predicted 3.26 GB against a measured 1.73 GB, and the report explained the
#: 1.5 GB gap away as conservatism instead of reading it as the signal it was:
#: a prediction is only a regression detector while it describes the shipped
#: system. Weight figures below are the pinned file sizes from models.lock.json
#: rather than round numbers, so a model swap shows up here as a diff.
MEMORY_BUDGET: dict[str, int] = {
    "llm_weights_q4_k_m": 1_117_320_736,   # qwen2.5-1.5b-instruct-q4_k_m.gguf
    "llm_kv_cache_2k": 60_000_000,         # 28 layers x 2 KV heads x 128 dim, f16
    "llm_compute_buffers": 250_000_000,
    "embedding_model_q8": 36_806_944,      # bge-small-en-v1.5-q8_0.gguf
    "index_and_corpus": 60_000_000,
    "python_runtime_and_server": 400_000_000,
}


def budget_total() -> int:
    return sum(MEMORY_BUDGET.values())


def budget_headroom() -> int:
    return MEMORY_CEILING_BYTES - budget_total()

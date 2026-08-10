# Àgbẹ̀

**Offline Yoruba/English crop advisory for smallholder farmers.**
Runs entirely on-device on an 8 GB laptop with no discrete GPU, no internet, and no API key.

> `Àgbẹ̀` (Yoruba) — *farmer*.

Africa Deep Tech Challenge 2026 · Agriculture track

---

## Why this exists

A smallholder farmer facing a diseased crop needs an answer in the next hour, in a
language they think in, on the device they already own. Cloud LLMs ask for three
things that are not reliably available: an API budget, stable bandwidth, and mains
power. Àgbẹ̀ removes all three from the critical path.

The engineering constraint is the product constraint: **7 GB of RAM, four cores, no
GPU.** Everything below follows from it.

## What it does

1. A farmer asks a question — in Yoruba or English — or photographs a diseased leaf.
2. Yoruba input is translated to English by a specialist translation model.
3. The question retrieves passages from a local corpus of agricultural extension
   material, stored on disk and searched offline.
4. A quantized instruction model answers **from the retrieved passages**, with
   citations, and the answer is translated back to Yoruba.
5. If a photo was supplied, an on-device classifier proposes a diagnosis — and
   abstains when it is not confident enough to be useful.

## Architecture

Four disciplines, integrated end to end:

| Stage | Discipline | Component |
|---|---|---|
| Yoruba ↔ English | Machine translation | NLLB-200 distilled 600M, int8 · CTranslate2 |
| Passage retrieval | Information retrieval | bge-small-en-v1.5 · llama.cpp |
| Grounded advisory | Language modelling | Qwen2.5-3B-Instruct Q4\_K\_M · llama.cpp |
| Leaf diagnosis | Computer vision | MobileNetV3 · ONNX Runtime |

### The load-bearing decision: no PyTorch

PyTorch costs roughly 700 MB–1 GB resident before a single weight is loaded — about
14% of the entire budget spent on a framework this system does not need. Every model
here runs on a C++ inference engine instead (llama.cpp, CTranslate2, ONNX Runtime).
The embedding model runs through the *same* llama.cpp runtime as the LLM, so
retrieval adds ~50 MB rather than a second framework.

`make verify-no-torch` enforces this, and the Docker build fails if torch appears
transitively.

### Memory budget

Estimated peak, from [`agbe/config.py`](agbe/config.py):

| Component | Estimate |
|---|---|
| LLM weights (Q4\_K\_M) | 1.90 GB |
| LLM compute buffers | 300 MB |
| Translation (int8, lazy-loaded) | 620 MB |
| Python runtime + server | 400 MB |
| KV cache @ 4k context | 155 MB |
| Index + corpus | 60 MB |
| Embedding model | 50 MB |
| Vision classifier | 20 MB |
| **Estimated peak** | **3.26 GB** |
| **Ceiling** | **7.00 GB** |
| **Headroom** | **3.74 GB** |

Estimates are superseded by *measured* peak RSS from `make bench`. Where the two
disagree, the measurement wins. See [REPORT.md](REPORT.md) for measured results and
for why the remaining headroom is deliberately not spent on a larger model.

## Quick start

Requires Docker. Everything runs inside `ubuntu:22.04` — the target OS — so results
are reproducible on any host.

```bash
make build            # build the reference image
make fetch-models     # download + checksum-verify weights (~2.6 GB)
make index            # build the retrieval index from corpus/
make run              # serve on http://localhost:8000, capped at 7 GB / 4 CPUs
```

Every target that executes model code runs under `--memory=7g --memory-swap=7g
--cpus=4`. Swap is disabled deliberately: with swap enabled the kernel would page
instead of OOM-killing, letting a run quietly exceed the ceiling and still appear to
pass. We want a hard failure.

```bash
make bench            # benchmark harness -> bench/results/
make test             # test suite
```

## Reproducing the benchmarks

`make bench` records peak RSS, time-to-first-token and tokens/second, labelled with
the host CPU and RAM so rows from different machines remain comparable.

**On hardware honesty:** these figures are only as representative as the machine
that produced them. Development hardware faster than the ADTC Standard Laptop will
report optimistic throughput — the RAM ceiling and thread count can be enforced in a
container, but memory *bandwidth* cannot be throttled down, and CPU token generation
is bandwidth-bound. REPORT.md states which machine produced each row and quantifies
the expected gap rather than presenting dev-box numbers as target-machine numbers.

## Safety

This system gives agricultural advice, and some of that advice concerns pesticides.

- Answers are grounded in the retrieved corpus and cite their sources.
- The model does not volunteer chemical dosages that are not present in the
  retrieved passages; it defers to a local extension officer instead.
- When retrieval returns nothing above threshold, the system says it has no local
  guidance rather than answering from parametric memory.
- The leaf classifier abstains below a confidence floor instead of handing the
  language model a confident wrong label to elaborate on.

## Corpus and licensing

The retrieval corpus is built only from openly licensed agricultural extension
material. Sources and their licences are recorded in
[`corpus/licenses/`](corpus/licenses/). No scraped or restrictively licensed
material is redistributed in this repository.

## Language support

Yoruba (`yor_Latn`) is the supported non-English language and is validated by a
native speaker against a fixed evaluation set. Other languages reachable by NLLB-200
are technically available but **not claimed**, because we cannot currently validate
their output quality.

## Licence

MIT — see [LICENSE](LICENSE).

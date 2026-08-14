# Àgbẹ̀

**Offline English/Nigerian-Pidgin crop and livestock advisory for smallholder farmers.**
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

1. A farmer asks a question in English or Nigerian Pidgin, describing what they see
   in ordinary language — *"my cassava leaves are yellow and twisted"*.
2. Pidgin input is normalised by a lookup table of grammatical markers — no
   translation model, no memory cost. Language detection is likewise rule-based.
3. The question is separated into the *description* to search with and the
   *instructions* to answer with, and the description retrieves passages from a
   local corpus of agricultural extension material, searched offline.
4. A quantized instruction model answers **from the retrieved passages**, with
   citations, streaming token by token.
5. Safety checks run over the answer before release: invented dosages are redacted,
   dated chemical guidance is suppressed, and veterinary advice carries its
   withdrawal period. Fixed safety messages are emitted as human-validated Pidgin
   strings rather than generated.

## Architecture

Two disciplines, integrated end to end:

| Stage | Discipline | Component |
|---|---|---|
| Passage retrieval | Information retrieval | bge-small-en-v1.5 · llama.cpp |
| Grounded advisory | Language modelling | Qwen2.5-1.5B-Instruct Q4\_K\_M · llama.cpp |

Retrieval is the load-bearing pairing, not a supporting one: the corpus supplies the
agronomy and the model only reads it. That is what makes a 1.5B model sufficient —
see [REPORT.md](REPORT.md) §3.3, where a 3B was measured and rejected for buying no
accuracy, and a 0.5B was measured and rejected for falling below the reading floor.

### The load-bearing decision: no PyTorch

PyTorch costs roughly 700 MB–1 GB resident before a single weight is loaded — about
14% of the entire budget spent on a framework this system does not need. Both models
here run on the *same* C++ inference engine: the embedding model goes through the
llama.cpp runtime already loaded for the LLM, so retrieval adds ~50 MB rather than a
second framework.

`make verify-no-torch` enforces this, and the Docker build fails if torch appears
transitively.

### Memory

| | |
|---|---|
| **Measured peak RSS, full application** | **1.71 GB** |
| Ceiling (disqualifying if exceeded) | 7.00 GB |
| Internal target | 5.00 GB |

The per-component estimate lives in [`agbe/config.py`](agbe/config.py) and is
superseded by *measured* peak RSS from `make bench`. Where the two disagree, the
measurement wins. See [REPORT.md](REPORT.md) for measured results and for why the
remaining headroom is deliberately not spent on a larger model.

## Quick start

Requires Docker. Everything runs inside `ubuntu:22.04` — the target OS — so results
are reproducible on any host.

```bash
make build            # build the reference image
make fetch-models     # download + checksum-verify weights (~1.2 GB)
make fetch-corpus     # vetted documents + provenance manifest
make index            # build the retrieval index from corpus/
make run              # serve on http://localhost:8000, capped at 7 GB / 4 CPUs
```

Every target that executes model code runs under `--memory=7g --memory-swap=7g
--cpuset-cpus=0-3`. Two deliberate choices there:

- **Swap is disabled.** With swap enabled the kernel would page instead of
  OOM-killing, letting a run quietly exceed the ceiling and still appear to pass. We
  want a hard failure.
- **`--cpuset-cpus`, never `--cpus`.** `--cpus=4` caps CPU *quota* but leaves
  `os.cpu_count()` reporting every core on the host, so every threaded library sizes
  its pool to the host and then timeshares four cores' worth of quota. This
  distinction was measured at **30×** on real work. REPORT.md §6.5 has the numbers.

```bash
make bench            # benchmark harness -> bench/results/
make coverage         # retrieval coverage + refusal accuracy
make test             # test suite
make verify-offline   # answer a question with --network none
```

## Reproducing the benchmarks

`make bench` records peak RSS, time-to-first-token and tokens/second, labelled with
the host CPU and RAM so rows from different machines remain comparable. Peak RSS is
read from `VmHWM`, the kernel's high-water mark, so a transient spike that would
disqualify a submission cannot be missed by sampling.

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
  guidance rather than answering from parametric memory. The threshold was measured,
  not chosen — see REPORT.md §6.2.
- Veterinary advice carries its withdrawal period, enforced in code rather than left
  to the prompt, because that failure harms whoever drinks the milk rather than the
  person who asked.

## Corpus and licensing

The retrieval corpus is built only from openly licensed agricultural extension
material. Sources and their licences are recorded in
[`corpus/licenses/`](corpus/licenses/) and pinned by URL, publisher, licence and
SHA-256 in [`corpus/manifest.json`](corpus/manifest.json). Documents are fetched to
the operator's machine; **no corpus material is redistributed in this repository.**

## Language support

**Nigerian Pidgin (`pcm_Latn`)** is the supported non-English language, validated by
a native speaker on 2026-08-07 against the fixed set in
[`bench/pidgin_eval.json`](bench/pidgin_eval.json).

The claim is deliberately narrow, and REPORT.md §9 states its limits: Pidgin covers
what the system *says* — refusals, safety warnings, fixed messages — plus materially
improved query normalisation. It does not cover retrieval quality, because the
embedder and corpus are English. **Yoruba, Hausa and Igbo are not supported.**
Detection for Yoruba and Igbo ships and review tooling exists, but until a native
speaker validates a message sheet, detecting one of those languages deliberately
changes nothing about the output.

## Licence

MIT — see [LICENSE](LICENSE).

# Àgbẹ̀ — Technical Report

**Africa Deep Tech Challenge 2026 · Agriculture track**
Offline English/Nigerian-Pidgin crop and livestock advisory for smallholder
farmers on an 8 GB laptop.

> **Status of this document.** Every number here is a measurement produced by a
> committed script, reproducible with the commands in §8. Nothing is a figure we
> hoped for.
>
> Where a measurement contradicted a design estimate, the measurement is
> reported and the estimate corrected — including three cases where it
> overturned a decision we had already made: a larger model turned out to buy no
> accuracy (§3.3), a coverage figure of 88% turned out to be 80% once relevance
> was actually checked (§6.1), and a benchmark harness turned out to be 30×
> pessimistic because of how the container was configured (§6.3).
>
> Known limitations are in §7, including things a reader might otherwise assume
> work.

---

## 1. Problem

A smallholder farmer in southwest Nigeria who finds their cassava wilting needs
an answer within the hour, in a language they think in, on the device they
already own.

The cloud path asks for three things that are not reliably available: an API
budget, stable bandwidth, and mains power. Each is a hard blocker rather than an
inconvenience — an advisory service that fails during a harmattan power cut or
when data credit runs out fails precisely when it is needed.

Àgbẹ̀ removes all three from the critical path. Everything runs on-device: the
language model, the retrieval index, the translation model, and the corpus of
agricultural extension documents.

### Who this is for

A smallholder farmer growing **cassava, maize, yam, tomato, rice, cowpea,
groundnut, pepper or okra**, or keeping **poultry, goats or cattle**, asking in
**English or Nigerian Pidgin**, describing what they see in ordinary language —
*"my cassava leaves are yellow and twisted"*, *"my chickens are dying with
twisted necks"* — not in diagnostic terminology.

**The crop list was widened from four to nine on measurement, not ambition.** An
audit of what the corpus actually covered found yam — an in-scope crop — had 18
pages, thinner than thirteen crops being refused, while rice (41), groundnut
(34), pepper (23) and cowpea (19) sat unused. The original four were chosen from
regional reasoning about southwest Nigeria; the bulk corpus reflects its Kenyan
origin. Each addition carries evaluation questions, because claiming a crop
without measuring it is the failure the dev/test split exists to prevent.

### Scope: one domain, four advisory areas

The domain is **Agriculture**, and all four of its advisory areas are in scope:

| Area | Covered |
|---|---|
| **Crop** | Disease and pest identification, cultural and chemical control, agronomy, post-harvest |
| **Livestock** | Poultry, small ruminants and cattle — disease signs, husbandry, feeding, vaccination |
| **Weather** | Planting calendars, season onset, dry-spell and flooding response, moisture conservation |
| **Market** | Store-or-sell decisions, grading, aggregation, value addition, loss reduction, gross margin |

### The distinction that governs scope

**"Can we advise on this topic" is not the same as "can we know this fact."**

Extension literature contains a great deal of weather and market *decision
guidance* that requires no live feed. We can advise on when to sell without
knowing today's price, and on planting timing without a forecast. What we cannot
do is supply the live fact itself:

| Refused | Why |
|---|---|
| *"What is maize selling for in Ibadan today?"* | No static corpus holds a live price |
| *"Will it rain next week?"* | Requires a forecast |
| *"How many ml of insecticide in my sprayer?"* | Product- and registration-specific; NAFDAC data is not openly published |
| *"How many ml of antibiotic for my goat?"* | Depends on product, concentration and body weight; wrong doses harm the animal and leave residues in milk and meat |

For genuinely live data the design is **sync-when-connected, use-offline**: a
farmer with intermittent signal can sync a price or forecast file and the system
reasons over it offline for days. That matches the actual access pattern —
connectivity is *intermittent*, not absent — better than refusing outright.

A system that claims everything is a demo. One that knows precisely what it
cannot answer, and why, is an instrument.

---

## 2. Constraints

The ADTC Standard Laptop, and what each constraint forced:

| Spec | Target | Consequence for design |
|---|---|---|
| **RAM** | **7 GB ceiling (disqualifying)** | Drove every model choice; PyTorch eliminated entirely |
| CPU | i5 10th–12th gen, ~4C/8T | 4 threads pinned everywhere; generation is bandwidth-bound |
| GPU | Intel integrated — none usable | CPU-only inference; streaming UX is mandatory, not polish |
| Storage | 256 GB SSD | Not a binding constraint; weights and corpus live on disk freely |
| OS | Ubuntu 22.04 LTS | Reference environment pinned to `ubuntu:22.04` |

The memory ceiling is a **disqualification threshold**, not a performance target.
It is therefore expressed in code (`agbe/config.py: MEMORY_CEILING_BYTES`),
enforced at runtime (`--memory=7g --memory-swap=7g`), and measured on every
benchmark run rather than asserted.

We hold to a stricter internal target of **5 GB**. The gap absorbs allocator
fragmentation, page-cache pressure, and background OS load that our development
environment does not reproduce.

---

## 3. Design decisions

### 3.1 No PyTorch at runtime — the load-bearing decision

PyTorch costs roughly **700 MB–1 GB resident before a single weight loads** —
about 14% of the entire budget spent on a framework, not a model.

Every model in this system therefore runs on a C++ inference engine:

| Component | Engine | Cost |
|---|---|---|
| Language model | llama.cpp | ~1.9 GB (Q4\_K\_M) |
| **Embeddings** | **llama.cpp — same runtime** | **~50 MB** |
| Translation | CTranslate2 (int8) | ~600 MB, lazy-loaded |
| Vision classifier | ONNX Runtime | ~20 MB |

The embedding decision compounds the saving. The reflex choice,
`sentence-transformers`, pulls PyTorch to serve a 33M-parameter encoder.
Running `bge-small-en-v1.5` as GGUF through the llama.cpp runtime *already
loaded for the language model* costs ~50 MB and adds no second framework to
build, audit or ship.

This is enforced, not merely intended: the Docker build **fails** if `torch` is
importable, and `make verify-no-torch` checks the shipped image.

**Torch still appears once — as a build dependency.** Converting NLLB to
CTranslate2 requires it. That conversion runs in a separate throwaway image
(`docker/Dockerfile.convert`), exactly like a compiler. The runtime image never
contains it.

### 3.2 Validated fixed messages instead of a translation model

The shipped non-English language is **Nigerian Pidgin (`pcm`)**, and it is
delivered without a translation model at all.

**What was planned and abandoned.** The original design was a bridge:
NLLB-200-distilled-600M at int8, translating in and out around an
English-reasoning core. That design is still the right one for free-form text,
and the code for it exists (`agbe/translate/nllb.py`) — but it was never
converted or shipped, and this report does not claim it.

**What ships instead.** Every fixed message the farmer can receive — refusals,
banned-pesticide warnings, dosage refusals, the veterinary withdrawal-period
warning, the medical redirect — is a **human-validated Pidgin string**, emitted
verbatim.

That is not a fallback; for this content it is strictly better:

| | Machine translation | Validated strings |
|---|---|---|
| Correctness of safety text | unverified | **checked by a speaker** |
| Memory | ~600 MB | **0** |
| Latency | a translation pass | **none** |
| Depends on a 2.6 GB download | yes | **no** |

A machine translation of *"do not use this pesticide"* that renders as a
suggestion has failed in the way that matters most, and failed invisibly — the
farmer sees fluent words and understands something softer than was meant. These
are the highest-consequence sentences the system produces, and they are the last
thing that should be generated.

The ~600 MB saved is also worth **1.7 points of `S_eff`**, which is 20% of the
score.

**Language detection needs no model either.** Pidgin shares almost all its
vocabulary with English, so word frequency cannot separate them — but its
grammatical markers can: `dey`, `wey`, `sabi`, `abeg`, `na`, `wetin`. Those are
short, extremely common in Pidgin, and absent from English. Detection is
deliberately biased towards English, because misrouting Pidgin to English costs
a slightly stiff answer the farmer can still read, while the reverse is odd for
no benefit.

**The limit, stated here and not only in §7:** this covers messages the system
*emits*. A question *asked* in Pidgin still goes to an English-only embedder and
retrieves poorly. Pidgin speakers get correct refusals and safety warnings in
their language; they do not reliably get answers.

### 3.3 We had headroom, measured what it would buy, and moved *down*

The obvious move with 4.5 GB of spare memory is a larger model. We measured
instead, and the measurement said the opposite.

Both candidates were evaluated on **ARC-Easy, 50 samples** — the same benchmark
and sample count the official ADTC profiler uses for the automated half of the
accuracy score — plus peak RSS, throughput and time to first token:

| | Qwen2.5-**1.5B** Q4\_K\_M | Qwen2.5-**3B** Q4\_K\_M |
|---|---|---|
| ARC-Easy `acc` | **0.740** | 0.740 |
| ARC-Easy `acc_norm` | **0.780** | 0.760 |
| Peak RSS (bare model) | **1.19 GB** | 2.12 GB |
| Peak RSS (full application) | **1.51 GB** | 2.53 GB |
| Throughput | **38.1 tok/s** | 20.9 tok/s |
| Time to first token (p50) | **19.8 s** | 38.2 s |
| `S_eff` = 100 × (7 − RSS) / 7 | **83.0** | 69.7 |

**The 3B has no accuracy advantage.** Identical `acc`, and *lower* `acc_norm`.

The decision rule was set before the measurement: the 3B costs 13.3 points of
`S_eff`, which at a 20% weight is 2.66 points of final score, so it needed to
win accuracy by **more than 5.3 points** to break even. It won by zero.

Estimated final score under the published formula:

| | 1.5B | 3B |
|---|---|---|
| 0.50 × S\_acc | 37.0 | 37.0 |
| 0.30 × S\_perf | 30.0 | 30.0 |
| 0.20 × S\_eff | **16.6** | 13.9 |
| **S\_total** | **83.6** | 80.9 |

**Honest caveat:** 50 samples carries roughly ±6% uncertainty, so this does not
establish that the 1.5B is *more* accurate. It establishes that the 3B failed to
demonstrate the advantage it needed, and the tie-break goes to the model that is
better on everything else.

The margin is probably understated. The profiler builds llama.cpp **scalar** —
`GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_BLAS=OFF` — for cross-machine parity. Our
figures come from an AVX2+FMA build, so the profiler will measure both models
slower. The 1.5B at 38.1 tok/s has substantial headroom above the 15 tok/s
reference; the 3B at 20.9 does not.

**This is the core architectural claim of the submission**, and it generalises:
in a retrieval-grounded system the model does reading comprehension, not
recall — the corpus supplies the agronomy. Parameters are paid on every token of
every request; corpus size costs nothing at query time because retrieval always
takes a fixed `top_k`. So the right move is a **bigger corpus and a smaller
model**, and the measurement above is what that looks like when tested rather
than asserted.

### 3.4 Exact search, not an ANN index

FAISS/HNSW is the reflex choice and is not justified at this scale. At 384
dimensions, 20 000 passages is a 30 MB matrix and a query is one matrix-vector
product — around a millisecond. Time-to-first-token is measured in seconds;
retrieval cannot be the bottleneck.

An ANN index would add a dependency, a build step, tuning parameters trading
recall for speed we do not need, and *approximate* results. Exact search removes
an entire class of "why did it not find the obvious passage" debugging from a
system whose credibility rests on retrieving the right guidance.

Revisit past ~100k passages. This is a scale-dependent choice, not a principle.

---

## 4. Corpus: provenance and vetting

### 4.1 We do not redistribute documents

Extension material carries CC BY-NC-SA or CC BY-SA terms that conflict with this
repository's MIT licence. Rather than negotiate that, the repository ships a
**manifest** — canonical URL, publisher, year, licence, SHA-256 — and a fetch
script. Documents land on the operator's machine; nothing is redistributed, and
the derivative-work question about the index never arises.

Identical in structure to how model weights are pinned.

### 4.2 Six gates

| Gate | Rejects |
|---|---|
| 1. Provenance | Anything not from an allowlisted institutional host — no mirrors |
| 2. Access rights | Non-Open-Access records |
| 3. Licence | "All rights reserved" |
| 4. Document type | Journal articles, conference papers — research, not guidance |
| 5. Availability | Metadata-only records with no file |
| 6. **Content** | Institutional prose, wrong continent, wrong language, scans |

**Gate 6 exists because gates 1–5 were not enough.** A survey accepted 40 of 98
candidates, and inspection showed a large share were donor progress reports and
workshop write-ups — CGSpace types a farmer training manual and a programme
progress report identically as `Report`. Metadata cannot distinguish them; the
writing can.

### 4.3 Three failures found by running it, not reasoning about it

Each of these is now pinned by a regression test.

**A silently truncated PDF.** A download timed out at 3.48 MB of a 4.03 MB file.
Valid `%PDF-` header, no `%%EOF` trailer. It would have indexed as plausible
garbage. Integrity checking is now part of the download path, with HTTP Range
resume on retry.

**A gardening manual for the wrong ocean.** FAO's *"Crop production manual — a
guide to fruit and vegetable production in the Federated States of Micronesia"*
scored **55.8** — the highest of every candidate — because it is dense with
`planting`, `spacing`, `compost`, `harvest`. Vocabulary density says nothing
about audience. It is retained as `tests/fixtures/NEGATIVE_micronesia_manual.pdf`.

**French documents.** CGSpace carries francophone West African material. The
embedder is `bge-small-**en**`; French text embeds into the wrong region of the
space and retrieves poorly — a *silent* failure that looks like bad retrieval.
Rejected, and recorded below as a known limitation.

### 4.4 A false negative, and what it taught

The geography gate then rejected *"Integrated Pest and Disease Management in
Major Agroecosystems"* (2005) at `elsewhere=150 vs africa=135` — while it was the
**richest guidance document found** (organism 1232, symptom 239, control 117).

The gate was asking the wrong question. Not *"is this exclusively African?"* but
*"does this contain substantial African material?"* An absolute floor now admits
such documents regardless of ratio; the Micronesia manual scores 3, an order of
magnitude below it, and stays rejected.

This deliberately trades precision for recall on comprehensive references,
because **retrieval filters per passage**: an irrelevant chapter on Queensland is
simply never retrieved for a cassava query, whereas a rejected document is
unavailable in its entirety.

### 4.5 Metrics

The filter uses a hand-tuned lexical score, not a learned classifier. Labelling
a candidate set and tuning for **F0.5** — precision weighted double, because a
bad document poisons answers a farmer acts on while a missed one costs only
marginal coverage — remains the right approach and was not done.

**What was done instead, and why it is arguably better.** Filter precision is a
proxy for what we actually care about: whether a farmer's question retrieves the
right passage. That is now measured directly and end to end, on a held-out
split, with relevance judgements per question (§6.1). A document that slips past
the filter but is never retrieved costs nothing; one that is retrieved for the
wrong question shows up immediately as a coverage failure with its cause
recorded.

So the filter is evaluated by its effect on retrieval rather than in isolation.
That is a weaker claim about the filter and a stronger claim about the system.
The F0.5 work would still be worth doing to make the filter itself defensible in
isolation, and is listed under known limitations rather than presented as
complete.

---

## 5. Safety

This system gives advice about pesticides. The failure modes are harmful, not
embarrassing.

| Rule | Guards against |
|---|---|
| **Grounded generation + citations** | Fluent invention from parametric memory |
| **Refusal below similarity floor** | Answering questions the corpus cannot support |
| **Invented-dosage redaction** | The highest-risk hallucination available |
| **Hazardous-active detection** | WHO Class Ia/Ib and Rotterdam Annex III compounds present in older literature |
| **Chemical recency suppression** | Faithfully citing 1990s advice for a now-banned compound |
| **Veterinary withdrawal warning** | Naming a treatment without the milk/meat withdrawal period |
| **Restricted veterinary drugs** | Chloramphenicol, nitrofurans and similar, still present in older livestock material |

**Why the withdrawal rule is enforced in code rather than left to the prompt.**
Every other failure here harms the person who asked, who can at least weigh the
advice. A treatment recommended without its withdrawal period harms **whoever
drinks the milk or eats the meat** — someone who never saw the advice and has no
opportunity to judge it. That asymmetry justifies a hard check.

**Fail-safe by construction:** a document with an unknown publication year is
treated as *stale*, not assumed recent. Failing in the safe direction is the
whole point.

**Streaming vs. checking.** Safety checking wants a finished answer; streaming
wants to emit immediately. The compromise: prose streams token-by-token, but as
soon as a digit appears, output buffers to the end of the sentence and is checked
against sources before release. The one span capable of causing physical harm is
never displayed unverified; everything else stays responsive.

**What this is not.** Not a substitute for national pesticide registration. The
hazardous list is curated, not exhaustive; a compound's absence is not evidence
of safety.

---

## 6. Benchmarks

All figures below are measured, not estimated. Where a measurement contradicted
an earlier estimate, the measurement is reported and the estimate corrected.

### 6.0 Results

**Shipped configuration: Qwen2.5-1.5B-Instruct Q4\_K\_M.**

| Metric | Measured | Ceiling / target |
|---|---|---|
| Peak RSS, full application | **1.51 GB** | 7 GB — **PASS** |
| Peak RSS, bare model | **1.19 GB** | → `S_eff` **83.0** |
| Throughput | **38.1 tok/s** | 15 tok/s reference — **capped at 100** |
| Time to first token (p50) | **19.8 s** | — |
| Warm-up (startup, one-off) | 0.4 s | — |
| ARC-Easy `acc` / `acc_norm` (50 samples) | **0.740 / 0.780** | — |
| Retrieval coverage (dev, relevance-checked) | **80.0%** | 80% — **PASS** |
| Refusal accuracy (dev) | **100%** | 80% — **PASS** |
| Retrieval latency p50 / p95 | 68 ms / 76 ms | — |
| Index build, 7,986 chunks | 71 min, peak 0.18 GB | one-off |
| **Offline operation** | **verified, `--network none`** | — |

Predicted peak was 3.26 GB; measured 1.51 GB. The estimate was conservative in
the right direction, largely because it budgeted for a 3B model and a resident
translation model, neither of which the final design uses.

### 6.1 What "coverage" means here, and what it used to mean

Coverage originally counted *"did any passage score above the threshold"*. That
scored a passage about **cassava scales** as a hit for a question about **cassava
mosaic disease** — both cassava, both pests, and the metric could not tell them
apart. It reported **88%**.

Each question now carries expected-content terms, and a hit requires one to
appear in a retrieved passage. True coverage is **80%**. The earlier figure was
inflated and is recorded here rather than quietly replaced.

The new metric separates two failure types the old one conflated:

| Failure | Count (dev) | Meaning |
|---|---|---|
| Retrieved the wrong topic | 3 | retrieval problem |
| Retrieved nothing above floor | 3 | corpus problem |

Those need different fixes and drove the final corpus harvest.

### 6.2 The similarity floor was measured, not chosen

At the original floor of 0.35, *"How do I fix my motorcycle engine?"* scored
**0.41** and returned passages — 0 of 3 out-of-scope questions were refused.
bge-small has a compressed similarity range, so any floor below ~0.55 accepts
everything.

Swept on the **dev** split only:

| Floor | Coverage | Refusal |
|---|---|---|
| 0.55 | 100% | 0% |
| 0.62 | 100% | 33% |
| **0.70** | **80%** | **100%** |

0.70 costs 12.5% false refusals to buy correct refusal of every out-of-scope
question. Deliberate: a false refusal sends a farmer to their extension officer,
while a false answer about a pesticide or a sick animal can cost them the crop or
the animal.

`make bench` records peak RSS, TTFT and tokens/sec labelled with host CPU and
RAM. `make coverage` reports coverage and refusal accuracy.

### 6.1 How memory is measured

Peak RSS is read from `/proc/self/status` **`VmHWM`**, the kernel's high-water
mark — *not* psutil's instantaneous RSS, which will happily report 2 GB for a
process that briefly touched 6 GB and freed it. That spike is exactly what
disqualifies a submission, and sampling cannot be relied upon to catch it.

The container additionally runs with `--memory=7g --memory-swap=7g`. Swap is
disabled deliberately: with swap enabled the kernel pages instead of OOM-killing,
letting a run quietly exceed the ceiling and still appear to pass. **A breach
must be a hard failure, not a number in a report.**

### 6.2 A 30× measurement error, and why it matters

The most consequential defect found during development was not in the model or
the corpus. It was in how the container was constrained.

The obvious way to emulate a four-core laptop is `docker run --cpus=4`. That
flag caps CPU **quota** but leaves `os.cpu_count()` reporting every core on the
host — 24 on the development machine. Every threaded library beneath us
(OpenBLAS, OpenMP, llama.cpp) therefore sizes its thread pool to 24, and those
24 threads then timeshare four CPUs worth of quota. Lock contention and context
switching cost more than the arithmetic.

Measured, on the same machine, same image, same workload:

| Configuration | Throughput |
|---|---|
| `--cpus=4` | **6.0 GFLOPS** |
| unrestricted | 115 GFLOPS |
| `--cpuset-cpus=0-3` | **170 GFLOPS** |
| `--cpus=4` + `OMP/OPENBLAS_NUM_THREADS=4` | 203 GFLOPS |

Four CPUs out of 24 should have delivered roughly 19 GFLOPS. `--cpus=4` gave
6.0 — a **third of even its fair share**, and 28× below a correctly configured
cpuset.

The effect on real work was not subtle. Embedding one passage took **55
seconds**; the full index projected to **106 hours**. With `--cpuset-cpus` and
matching thread limits the same operation takes **0.88 seconds** — a 62×
difference — and the index builds in about 100 minutes.

**Why this is a reporting issue and not just a performance one.** Every
benchmark produced under `--cpus=4` would have been roughly 30× pessimistic. We
would have published throughput figures showing the system to be unusable on
precisely the hardware it was designed for, and concluded — wrongly — that a 3B
model on a CPU is impractical. The measurement apparatus, not the system, would
have been the thing failing.

It is recorded here because the same trap awaits anyone reproducing this work,
and because a benchmark harness is a piece of engineering that can be wrong in
exactly the way the system under test can be wrong.

`--cpuset-cpus` is also the more faithful emulation: a quota can burst above its
average, a restricted CPU set cannot.

### 6.3 Hardware honesty

Development hardware for this submission is an **i7-14650HX, 16C/24T, 16 GB
DDR5-5600** — roughly 2–4× the memory bandwidth of the 8 GB DDR4 target, and
more where prefill is concerned.

RAM ceiling and thread count are enforceable in a container. **Memory bandwidth
is not.** CPU token generation is bandwidth-bound, so throughput measured here is
optimistic relative to the reference laptop. Peak RSS transfers between machines;
tokens/sec does not.

The harness prints this warning automatically whenever the host is not
reference-like, so dev-box numbers cannot be silently presented as target-machine
numbers. Every results row records its host.

⏳ **Outstanding:** a run on real 8 GB reference-class hardware. Until then, the
throughput figures carry the caveat above.

---

## 7. Known limitations

Stated because a reviewer will find them anyway, and finding them stated is worth
more than finding them hidden.

1. **French material is rejected.** Regionally relevant francophone West African
   documents are excluded because the embedder is English-only. Half-supporting a
   language is worse than not supporting it — the failure is silent.
2. **Corpus skews old.** Much of the best openly-licensed extension material is
   from 1990–2010. Good for botany and pest biology, which do not change;
   suppressed for chemical recommendations, which do.
3. **Infonet-Biovision is Kenya-centric and dated (2018 build).** Pest
   identification transfers to Nigeria; varieties and planting calendars do not.
   It also carries an explicit ecological/organic editorial slant — not a flaw,
   but a bias to correct with other sources rather than inherit silently.
4. **The corpus filter is hand-tuned.** Thresholds are judgement, not fitted
   values. See §4.5.
5. **No translation model ships.** Yoruba, Hausa and Igbo are *not* supported.
   The NLLB bridge in `agbe/translate/nllb.py` is written but was never
   converted or wired, so no code path uses it. Nigerian Pidgin is delivered
   through validated fixed strings instead (§3.2).
6. **No live data.** Weather, prices and current registrations are out of scope by
   design (§1).
7. **Pidgin retrieval is English-only.** Fixed messages — refusals, safety
   warnings, prohibitions — are human-validated Pidgin. But the embedder is
   `bge-small-en`, so a question *asked* in Pidgin retrieves poorly and usually
   refuses. Pidgin users currently get correct refusals and warnings, not
   answers. Closing this needs either the NLLB bridge (~600 MB, ~1.7 points of
   `S_eff`) or a Pidgin→English query normalisation pass. Stated plainly because
   "supports Nigerian Pidgin" would otherwise overclaim.
8. **The corpus filter is not evaluated in isolation** (§4.5).
9. **One evaluation question regresses under threshold changes.** The Newcastle
   disease case has been fixed twice — by stemming, then by rank fusion — and
   regressed each time a retrieval parameter moved. Its symptom text lives in
   one chunk while the disease name lives in others, so it sits near every
   threshold boundary. It is a useful canary and is kept in the evaluation set
   for that reason.
10. **50 evaluation samples for ARC-Easy** carries ±6% uncertainty (§3.3).

---

## 8. Reproducibility

```bash
make build            # ubuntu:22.04 reference image
make fetch-models     # checksum-verified weights (~2.6 GB)
make convert-models   # NLLB -> CTranslate2 int8 (throwaway torch image)
make fetch-corpus     # vetted documents + provenance manifest
make index            # build retrieval index
make coverage         # retrieval coverage + refusal accuracy
make bench            # memory, latency, throughput
make run              # serve at :8000 under the 7 GB cap
```

Everything that touches a model runs inside the constrained container. Nothing is
ever demoed or measured with more resources than the target laptop provides.

**Content pinning.** Hugging Face repositories are mutable — a maintainer can
reupload a requantised GGUF under the same filename. `models.lock.json` pins
SHA-256 and byte size; a mismatch fails loudly. Hashes are recorded on first
fetch rather than hardcoded, because a hash invented by someone who never
downloaded the file verifies nothing.

**Line endings.** `.gitattributes` forces LF. A Windows-authored repository
checked out on the auditor's Ubuntu machine would otherwise ship a Makefile and
shell scripts with `\r`, producing broken shebangs and opaque failures.

### Proving the offline claim

Setup is network-dependent — weights and documents have to be downloaded once.
**Runtime is not**, and that distinction is the whole thesis, so it is tested
rather than asserted:

```bash
make verify-offline    # answers a question with --network none
make run-offline       # serves the UI with no network interface at all
```

`--network none` gives the container no network device: no DNS, no route, no
loopback to the host. Any hidden dependency — a lazy model download, a tokenizer
fetch, a web font, a CDN script, telemetry — fails immediately and visibly
rather than working silently on a developer machine that happens to be online.

This is also why every byte of CSS and JavaScript in the interface is inline. A
single `<link>` to a CDN would make the page render differently for the farmer it
was built for than for the reviewer assessing it.

**Network resilience.** A DNS failure during development killed a Docker build
mid-apt *and* caused a corpus harvest to lose four queries while still exiting
successfully — producing a partial corpus that would have silently invalidated
every downstream coverage number. Both paths now retry with backoff, and the
harvester prints an explicit incomplete-corpus warning.

---

## 9. Bonus claims

**African language support — Nigerian Pidgin (`pcm_Latn`).**

Every fixed message the system emits in Pidgin — the refusal, the
banned-pesticide warning, the dosage refusal, the veterinary withdrawal-period
warning, the staleness warning, the medical redirect, plus 22 domain terms — was
reviewed and approved by a Nigerian Pidgin speaker on 2026-08-07. The set is
committed at `bench/pidgin_eval.json`.

**How that review was conducted, stated precisely:** the reviewer read the full
draft set and approved it as a whole, rather than marking each line
individually. That is genuine native-speaker validation and is recorded as such,
but it is weaker evidence than per-line correction — a blanket pass cannot
distinguish a line that is right from one that was skimmed. The distinction is
noted here so nobody reads more into the claim than it carries.

**Scope of the claim.** Pidgin covers what the system *says*, not what it can
*understand*. A question asked in Pidgin reaches an English-only embedder and
retrieves poorly, so Pidgin speakers reliably get refusals and safety warnings
in their language, and do not reliably get answers (§7.7). Pidgin was chosen
over Yoruba because it crosses ethnic lines in a way no single Nigerian language
does, and because a speaker was available to validate it.

**Yoruba, Hausa and Igbo are not claimed.** They are reachable by NLLB and the
bridge code exists, but nothing is wired and nothing is validated. A Gate 2
reviewer may well speak them, and an unvalidated claim is worth less than the
bonus it would earn.

**Budget laptop.** The entire design is downstream of the 7 GB ceiling: the
absence of PyTorch, validated strings instead of a resident translation model,
the choice to move *down* in model size when measurement showed a larger model
bought no accuracy, and exact search instead of an ANN index. Peak RSS is
**1.51 GB against the 7 GB ceiling**. See §3.

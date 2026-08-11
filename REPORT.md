# Àgbẹ̀ — Technical Report

**Africa Deep Tech Challenge 2026 · Agriculture track**
Offline English/Nigerian-Pidgin crop and livestock advisory for smallholder
farmers on an 8 GB laptop.

> **Status of this document.** Every number here is a measurement produced by a
> committed script, reproducible with the commands in §8. Nothing is a figure we
> hoped for.
>
> Where a measurement contradicted a design estimate, the measurement is
> reported and the estimate corrected — including four cases where it
> overturned a decision we had already made: a larger model turned out to buy no
> accuracy (§3.3), a coverage figure of 88% turned out to be 80% once relevance
> was actually checked (§6.1), a benchmark harness turned out to be 30×
> pessimistic because of how the container was configured (§6.5), and our own
> submitted test prompt turned out to retrieve the wrong documents entirely
> while the benchmark reported 100% (§6.3).
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

**What was planned, built, measured and rejected.** The original design was a
bridge: NLLB-200-distilled-600M at int8, translating in and out around an
English-reasoning core. We converted it (621 MB on disk, **~740 MB resident**,
measured), wired it and tested it against the thing it was supposed to fix —
Pidgin questions retrieving poorly. It does not fix it:

| Pidgin question | raw | via NLLB | via normalisation |
|---|---|---|---|
| fowl dying, necks twisting | miss | miss | miss |
| cassava leaf yellow and twisting | miss | miss | **hit** |
| goat not eating, shaking | miss | miss | **hit** |
| when to plant maize | hit | hit | hit |
| stopping yam rotting in store | miss | miss | **hit** |
| | **1/5** | **1/5** | **4/5** |

**Zero improvement for ~740 MB.** The wiring was not at fault — an `eng→fra`
control through the identical code path returned *"Mes feuilles de manioc sont
jaunes et tortueuses"*, which is correct. The cause is structural: Nigerian
Pidgin is an **English-lexified creole**, so NLLB's `pcm_Latn` sees input that
looks like English and copies it through untouched. It did exactly that for
three of the five questions.

One case was worse than useless. *"My fowl dem dey die, dem neck dey twist"* — a
textbook description of Newcastle disease in a flock — came back as *"My fowl
**will** die, **my** neck will twist."* Present became future, and *their* necks
became *my* neck, converting a poultry question into a human medical symptom
that the scope guard exists to refuse. A layer that can turn a livestock
question into a medical one is a safety regression, not a feature.

**What ships instead of the bridge.** The same property that defeats NLLB makes
the problem easy. Because Pidgin is English-lexified, its distance from English
is concentrated in a small closed set of grammatical markers and everyday words
— `dey`, `wetin`, `dem`, `go`, `chop`, `am`, `sabi`. `agbe/translate/pidgin_norm.py`
maps those before retrieval. It costs **no memory, no framework and no load
time**, it is auditable line by line by the same speaker who validated the
safety messages, and it more than doubles Pidgin retrieval (1/5 → 4/5).

English questions pass through it byte-identically, which is asserted by test
rather than assumed — every benchmark number in this report was measured on
English, and normalisation must not move them.

`agbe/translate/nllb.py` remains in the tree, unused by any code path. It is
kept as the record of a measured negative result; it is **not** shipped, not
loaded, and not claimed.

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

**Yoruba and Igbo: machinery ready, claim withheld.** Detection for both now
ships (script characters first — ṣ is Yoruba-only, ụ ị ṅ Igbo-only, NFC-
normalised so phone keyboards' combining marks behave; word evidence when
diacritics are omitted), and the review tooling generalises the Pidgin flow:
`scripts/lang_sheet.py --init yor` builds a validation sheet of every fixed
safety string, sourced from `messages.py` so English is never retyped. The
draft column is deliberately empty — NLLB drafts were measured deleting the
prohibition clause from "It is banned" in Yoruba, and a wrong draft anchors a
reviewer worse than no draft. Until a native speaker validates a sheet, the
pinned contract is that detection changes nothing: `messages.get("yor")`
serves English. Detecting a language is not claiming to support it.

**The limit, stated here and not only in §7:** normalisation raised Pidgin
retrieval from 1/5 to 4/5 on our sample, which is a real improvement and still
below the English path. The remaining miss is the Newcastle disease question —
the same case that has regressed under every retrieval change in this project
(§7.9). Five questions is also a small sample: it is enough to reject NLLB,
which scored no better than doing nothing, and not enough to put a percentage
on Pidgin accuracy. We do not quote one.

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

*Both columns were measured on the pre-compression benchmark harness (§6.0), so
the absolute RSS and TTFT here differ from the shipped figures. They are left as
measured because this is an A/B: both models ran identical code, and correcting
one column without re-running the 3B would turn a like-for-like comparison into
a mismatched one. The ratios, which are what the decision rested on, stand.*

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
| Peak RSS, full application | **1.73 GB** | 7 GB — **PASS** |
| Peak RSS, bare model | **1.19 GB** | → `S_eff` **83.0** |
| Throughput | **40.0 tok/s** | 15 tok/s reference — **capped at 100** |
| Time to first token (p50 / p95) | **22.1 s / 26.6 s** | — |
| Prompt size sent to the model | ~802 tokens | — |
| Warm-up (startup, one-off) | 0.4 s | — |
| ARC-Easy `acc` / `acc_norm` (50 samples) | **0.740 / 0.780** | — |
| Retrieval coverage (**held-out test**, relevance-checked) | **100%** (31/31) | 80% — **PASS** |
| Refusal accuracy (**held-out test**) | **100%** (6/6) | 80% — **PASS** |
| Retrieval latency p50 / p95 | 76 ms / 89 ms | — |
| Index build, 31,682 chunks | one-off, peak 0.18 GB | one-off |
| Sustained-load throughput decay (20 min continuous) | **1.0%** | <10% — no throttling signature (§6.7) |
| Peak core temperature | null — host exposes no sensor (§6.7) | 85 °C penalty threshold |
| **Offline operation** | **verified, `--network none`** | — |

Predicted peak was 3.26 GB; measured 1.73 GB. The estimate was conservative in
the right direction, largely because it budgeted for a 3B model and a resident
translation model, neither of which the final design uses.

**The performance figures moved late, and the reason is worth stating.** An
earlier run of this table reported TTFT of 65 s. The cause was not the model:
`bench/run.py` called `engine.retrieve()` and sent full chunk text, while the
shipped `advise()` sent *compressed* passages. The benchmark was timing a
configuration no user could reach — ~2,100 prompt tokens against the ~800 the
application actually sends.

`stream()` had the same defect — and so, we found in a later bug hunt, did the
web server's `/ask`, which turned out not to call `stream()` at all but to carry
its own copy-pasted generation loop. The guard logic existed in **three**
places, and each fix reached some of them: the compression fix landed in
`stream()` and the benchmark but not the browser, so the interface a judge
actually touches kept sending ~2,100-token prompts at 65 s TTFT after this
report first said the problem was fixed. The browser was also excluded from the
query-preparation fix of §6.3 — pasting the submitted test prompt into the UI
still reproduced the bug — and had quietly hardcoded `language: "en"`,
disabling Pidgin detection that the server supported and the tests exercised.

The structural fix, not the instance fix: generation and dosage-guarding now
live in exactly one method, `AdvisoryEngine.guarded_stream()`, and `stream()`,
`/ask` and the benchmark are all consumers of the same entry points. TTFT fell
from 65 s to 22 s and wall clock for six questions halved. This defect class —
a second code path that looks equivalent, is not, and is the one a judge will
exercise — has now appeared four times in this project (the others: the web UI
skipping scope guards, and `models.lock.json` pinning a different model from
the one the app loaded). Each instance was found by running the artefact a
reviewer would run, which is why §8 insists the reproduction commands are the
ones we ourselves use.

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

### 6.3 Our own test prompt broke the system, and a 100% benchmark did not notice

The two prompts in `submission/metadata.json` are what a judge will run first.
Late in development we ran them, and the first one failed badly.

**Prompt:** *"A smallholder farmer in Oyo State, Nigeria says: my cassava leaves
are yellow and twisted, and the plants are small. Explain the most likely cause,
and give practical steps the farmer can take this week…"*

**What Àgbẹ̀ answered:** the most likely cause is *"delayed harvesting due to
lack of a ready market or storage"*, with the advice to *"harvest the cassava as
soon as the leaves turn yellow."*

Those symptoms are textbook cassava mosaic disease. The advice would cost a
farmer the crop, and one cited source was *"Gender Gaps in Food Crop Production…
Cameroon."*

Three hypotheses were wrong before measurement found the cause:

| Hypothesis | Measurement | Verdict |
|---|---|---|
| The 1.5B model is too small | bare model answered *better* than Àgbẹ̀ | wrong |
| Institutional prose is out-scoring guidance | institutional-term density **0** in all six chunks | wrong |
| The corpus lacks the content | corpus holds **155** cassava-mosaic passages | wrong |

Printing the retrieved chunks showed what was actually being sent to the model:

| # | Document | What the passage actually was |
|---|---|---|
| 1 | *Pest control in cassava farms* | the **title page** — "© IITA 2000 ISBN 978-", publisher address |
| 2 | *Common African pests and diseases* | **garbled OCR** — "I esion a nd spiral nemalod es are o f importan ce" |
| 5 | a post-harvest-loss project report | *"Major causes of PHL in cassava include delayed harvesting due to lac…"* |

The model had not hallucinated. It was handed that fifth passage and used it
faithfully. **The generator was innocent; retrieval was the defect.**

The root cause was the query itself. Embedding the prompt against the corpus:

| What was embedded | Top-ranked document | Score |
|---|---|---|
| the full prompt | climate value-chain adaptation reports | **0.775** |
| the symptoms alone | *Disease control in cassava farms: IPM field guide* | 0.714 |

*"Smallholder farmer"*, *"Oyo State, Nigeria"*, *"practical steps"*, *"cultural
and preventive measures"* is the vocabulary of development-agency reporting — so
it retrieved development-agency reports. Those phrases are most of the prompt;
the four words carrying the diagnosis, *"yellow and twisted"*, were outvoted.

A prompt contains two different things, and we had been conflating them:

* a **description** of a situation → this is what the index should be searched with;
* **instructions** to the answerer about tone, format and constraints → these
  belong to the generator and say nothing about which passage is relevant.

`agbe/rag/query.py` now separates them. `agbe/rag/quality.py` additionally
rejects front matter and OCR debris at retrieval time (**5.3%** of the 31,682
chunks: 1,395 garbled, 286 front matter). After both:

| | Before | After |
|---|---|---|
| Top retrieved documents | climate/gender project reports | *Cassava (Bemisia tabaci)*, *Cassava Leaves*, *Growing cassava*, *Disease control in cassava* |
| Diagnosis | "delayed harvesting due to lack of a ready market" | a cassava virus disease, with clean-planting-material control advice |
| Held-out coverage | 100% | **100%** (unchanged) |
| Held-out refusal | 100% | **100%** (unchanged) |

**The uncomfortable part is the last two rows.** The held-out benchmark scored
100% before this bug and 100% after it. It never saw the failure, because every
evaluation question is phrased the way we phrase questions — short and direct —
while our own submitted prompt is long and framed, the way a *judge* would write
one. The benchmark was measuring a narrower thing than we believed it measured.

We are reporting this rather than quietly shipping the fix, because it is the
most useful thing we learned: **a passing metric is evidence about the metric
before it is evidence about the system.** The regression is pinned by
`tests/test_query.py` and `tests/test_quality.py`, and the honest residual
limitation is recorded in §7 — our evaluation set still under-represents long,
framed questions.

### 6.4 How memory is measured

Peak RSS is read from `/proc/self/status` **`VmHWM`**, the kernel's high-water
mark — *not* psutil's instantaneous RSS, which will happily report 2 GB for a
process that briefly touched 6 GB and freed it. That spike is exactly what
disqualifies a submission, and sampling cannot be relied upon to catch it.

The container additionally runs with `--memory=7g --memory-swap=7g`. Swap is
disabled deliberately: with swap enabled the kernel pages instead of OOM-killing,
letting a run quietly exceed the ceiling and still appear to pass. **A breach
must be a hard failure, not a number in a report.**

### 6.5 A 30× measurement error, and why it matters

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

### 6.6 Hardware honesty

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

### 6.7 Thermal: what was measured, what could not be, and which is which

`P_thermal` penalises a peak core temperature at or above **85 °C** (the
threshold in the official profiler's `thermal.py`, whose schema also permits
`core_temp_c_peak: null` on hosts that cannot report one).

**This host cannot report one, and we verified that rather than assumed it:**
the container's `/sys/class/thermal` holds cooling devices but no
`thermal_zone*`; `/sys/class/hwmon` exposes only the AC adapter and battery;
`psutil.sensors_temperatures()` returns `{}`; and the Windows-side WMI thermal
class is access-denied. `bench/thermal.py` attempts the same sensor paths the
official profiler uses and reports **null**, not a substitute.

A temperature from this machine would also be the one benchmark number that is
*worse* than nothing. Peak RSS transfers between machines — 1.7 GB here is
1.7 GB anywhere. Temperature does not, even directionally: this host is a
55 W-class i7-14650HX with the cooling of a workstation; the target is a
15 W-class U-series i5 in a thin chassis. A comfortable reading here predicts
nothing there.

What *is* measurable without a thermometer is throttling's only symptom that
matters to a farmer — the answer getting slower. `make thermal` generates
continuously for 20 minutes, a duty cycle far above real request-driven use,
and compares the last quarter's throughput against the first:

| | Measured |
|---|---|
| Continuous generation | 20 min, 40 full answers |
| Throughput, first quarter | 38.07 tok/s |
| Throughput, last quarter | 37.68 tok/s |
| **Decay** | **1.0%** (alert threshold 10%) |
| Peak core temperature | null — sensors unavailable, stated above |

Flat throughput under sustained worst-case load is positive evidence of no
throttling on this host — held, incidentally, while an unrelated container
image was building on the same machine. What transfers to the target laptop is
not that verdict but the workload's shape: four pinned threads, bandwidth-bound
generation, and a request-driven duty cycle in real use. The measurement on
reference hardware remains outstanding, as §6.6 already states.

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
   The NLLB bridge in `agbe/translate/nllb.py` was converted and measured, then
   rejected: it improved Pidgin retrieval by nothing at all for ~740 MB, and
   mistranslated one livestock question into a human medical one (§3.2). No
   code path uses it. Nigerian Pidgin is delivered through validated fixed
   strings plus zero-cost query normalisation instead.
6. **No live data.** Weather, prices and current registrations are out of scope by
   design (§1).
7. **Pidgin retrieval is weaker than English, and measured on five questions.**
   Fixed messages are human-validated Pidgin, and query normalisation raised
   retrieval from 1/5 to 4/5 (§3.2). But the embedder is still `bge-small-en`,
   the corpus is entirely English, and **five questions is too small a sample to
   quote an accuracy figure from** — it is sufficient to reject NLLB, which did
   no better than nothing, and insufficient to characterise Pidgin performance.
   Closing this properly needs a Pidgin evaluation set on the scale of the
   English one (70 questions), which we have not built. "Supports Nigerian
   Pidgin" means: validated safety messages, working detection, and materially
   improved but unquantified retrieval.
8. **The corpus filter is not evaluated in isolation** (§4.5).
9. **One evaluation question regresses under threshold changes.** The Newcastle
   disease case has been fixed twice — by stemming, then by rank fusion — and
   regressed each time a retrieval parameter moved. Its symptom text lives in
   one chunk while the disease name lives in others, so it sits near every
   threshold boundary. It is a useful canary and is kept in the evaluation set
   for that reason.
10. **50 evaluation samples for ARC-Easy** carries ±6% uncertainty (§3.3).
11. **The evaluation set under-represents long, framed questions.** Every one of
    the 70 questions is short and direct, the way we write questions. Our own
    submitted test prompt is long and framed, the way a *judge* writes one — and
    it retrieved the wrong documents entirely while the benchmark reported 100%
    coverage on both dev and test splits (§6.3). The retrieval defect is fixed
    and pinned by tests; **the blind spot in the evaluation set is not.** A
    proper fix means writing framed variants of existing questions and
    re-measuring, which we have not done. Until then, the coverage figure should
    be read as *coverage on directly-phrased questions*, which is a narrower
    claim than it appears.

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
**1.73 GB against the 7 GB ceiling**. See §3.

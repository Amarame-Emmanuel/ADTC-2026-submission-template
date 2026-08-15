# Àgbẹ̀ — Technical Report

**Africa Deep Tech Challenge 2026 · Agriculture track**
Offline English/Nigerian-Pidgin crop and livestock advisory for smallholder
farmers on an 8 GB laptop.

> **Status of this document.** Every number here is a measurement produced by a
> committed script, reproducible with the commands in §8. Nothing is a figure we
> hoped for.
>
> Where a measurement contradicted a design estimate, the measurement is
> reported and the estimate corrected — including five cases where it
> overturned a decision we had already made: a larger model turned out to buy no
> accuracy (§3.3), a coverage figure of 88% turned out to be 80% once relevance
> was actually checked (§6.1), a benchmark harness turned out to be 30×
> pessimistic because of how the container was configured (§6.5), our own
> submitted test prompt turned out to retrieve the wrong documents entirely
> while the benchmark reported 100% (§6.3), the headline results table itself
> turned out to be hand-maintained rather than artifact-backed, quoting three
> figures no committed run contained (§6.0), and the same submitted prompt
> failed a *second* time for a different reason — a chunk boundary that filed
> one disease's symptoms under another's name, which every model tested read
> faithfully and got wrong (§6.8).
>
> One of those corrections changed how this document is measured at all.
> Coverage counts whether the right passage was *retrieved*; it sat unmoved
> through changes that made answers worse and the change that made them right.
> §6.9 adds the answer-level number, which is **6.7 points lower** than the
> coverage figures quoted throughout §6.
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
language model, the retrieval index, and the corpus of agricultural extension
documents.

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

That table is the whole inventory. Two components appeared in earlier drafts of
it and ship in neither the image nor the code: a CTranslate2 translation model,
measured and rejected in §3.2, and an ONNX leaf classifier, which was scoped and
never built. Their wheels are no longer pinned in `requirements.txt` either — a
dependency retained for a component that does not exist is an unbacked claim
with an install cost attached.

The embedding decision compounds the saving. The reflex choice,
`sentence-transformers`, pulls PyTorch to serve a 33M-parameter encoder.
Running `bge-small-en-v1.5` as GGUF through the llama.cpp runtime *already
loaded for the language model* costs ~50 MB and adds no second framework to
build, audit or ship.

This is enforced, not merely intended: the Docker build **fails** if `torch` is
importable, and `make verify-no-torch` checks the shipped image.

**Torch appears nowhere in the shipped pipeline.** It was once a build
dependency: converting NLLB to CTranslate2 requires it, in a separate throwaway
image (`docker/Dockerfile.convert`), used exactly like a compiler. With the
bridge rejected in §3.2, that conversion is no longer a step anyone needs to
run, and `make convert-models` has been dropped from the reproduction sequence
in §8. The target and its Dockerfile remain in the tree alongside
`agbe/translate/nllb.py`, so the negative result stays reproducible by anyone
who wants to check it rather than take our word for it.

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

The obvious move with 5.3 GB of spare memory is a larger model. We measured
instead, and the measurement said the opposite.

Both candidates were evaluated on **ARC-Easy, 50 samples** — the same benchmark
and sample count the official ADTC profiler uses for the automated half of the
accuracy score — plus peak RSS, throughput and time to first token:

| | Qwen2.5-**1.5B** Q4\_K\_M | Qwen2.5-**3B** Q4\_K\_M |
|---|---|---|
| ARC-Easy `acc` | **0.740** | 0.740 |
| ARC-Easy `acc_norm` | **0.780** | 0.760 |
| Peak RSS (full application) | **1.71 GB** | 2.71 GB |
| Throughput (p50) | **38.0 tok/s** | 21.5 tok/s |
| Time to first token (p50 / p95) | **17.9 s / 20.2 s** | 36.2 s / 44.5 s |
| Mean prompt tokens sent | 773 | 773 |
| `S_eff` = 100 × (7 − RSS) / 7 | **75.5** | 61.3 |

*This is a true like-for-like A/B, which it had not previously been. Both
columns come from the same harness at the same commit, sending the same 773
prompt tokens, under the same 4-core cpuset — `bench/results/benchmark.json` and
`benchmark_3b.json`, both committed. Reproduce either with
`AGBE_LLM_FILENAME=<gguf> make bench`.*

*It is worth recording what the earlier, unbacked version of this table said,
because the correction went the opposite way from the one you would expect a
team to make about its own choice. It quoted the 3B at 2.53 GB, 20.9 tok/s and
38.2 s TTFT, and the 1.5B at a bare-model 1.19 GB. Measured properly, the 3B is
**2.71 GB** — worse than claimed — and the 1.5B is 1.71 GB on the same basis. The
memory gap the decision turned on was therefore **understated**: 1.00 GB, not
0.76 GB, and 14.2 points of `S_eff` rather than 10.7. Getting the number right
strengthened the case for the model we had already picked, which is the least
suspicious direction for an error to move but not a reason to have left it
unmeasured.*

**The 3B has no accuracy advantage.** Identical `acc`, and *lower* `acc_norm`.

The decision rule was set before the measurement: the 3B costs 14.2 points of
`S_eff`, which at a 20% weight is 2.84 points of final score, so it needed to
win accuracy by **more than 5.7 points** to break even. It won by zero.

**The bracket was later closed from below.** The same rule was applied to
Qwen2.5-**0.5B** (decision threshold set first: its ~+8 points of S_eff at 20%
weight allow an accuracy loss under **3.4 points**). Measured, and committed as
`bench/results/arc_0.5b.json`: ARC-Easy **0.620 `acc` / 0.620 `acc_norm`**
against the 1.5B's 0.740 / 0.780 — a 12-point loss on `acc` and 16 on
`acc_norm`, ≈ −4.3 final points net at best.
The qualitative failure settles it more vividly than the arithmetic: handed
the *same retrieved passages* the 1.5B reads correctly, the 0.5B answered the
submitted cassava prompt with *"the most likely cause is a lack of
chlorophyll"* (the symptom restated as its cause) and advised cutting off
affected leaves and **replanting the healthy part** — for a virus spread
through planting material, the exact practice the sources warn against.

Below some capability floor a model cannot be trusted to read sources, which
is the one job this system leaves to it. 0.5B is under that floor, 3B pays
for nothing above it: **1.5B is the measured minimum, bounded on both sides.**

Estimated final score under the published formula:

| | 1.5B | 3B |
|---|---|---|
| 0.50 × S\_acc | 37.0 | 37.0 |
| 0.30 × S\_perf | *see below* | *see below* |
| 0.20 × S\_eff | **15.1** | 12.3 |
| **S\_total**, excluding `S_perf` | **52.1** | 49.3 |

**`S_perf` is deliberately left blank, and earlier drafts of this table were
wrong to fill it in.** They booked 30.0 — a perfect score — for both models. The
published formula is `S_perf = 100 × (TPS_act ÷ TPS_max)` where `TPS_max` is the
highest throughput *across all submissions*. It is a rank, not a threshold:
there is no reference figure to clear and no cap to reach. Booking 30.0 assumed
we would be the fastest entrant, which is not a measurement we are able to make.

What can be said is the sensitivity, and it is steep:

| If the fastest submission runs at | our `S_perf` | 0.30 × `S_perf` |
|---|---|---|
| our own throughput | 100 | 30.0 |
| 1.5× ours | 67 | 20.0 |
| 2× ours | 50 | 15.0 |
| 3× ours | 33 | 10.0 |

The spread between the first and last row is 20 points of final score — larger
than the entire `S_eff` term we spent §3 optimising. It is also the one term no
amount of care on this machine can pin down, because the denominator belongs to
other teams. The honest statement is that our controllable score is **52.1**
plus whatever the field allows, and that `S_perf` deserves more engineering
attention than its 30% weight suggests, precisely because it is the term we
cannot measure.

`S_eff` here is computed from **peak RSS of the full running application** —
1.71 GB for the 1.5B, 2.71 GB for the 3B — because the published formula defines
peak RAM as the maximum RSS measured *during the audit*, and what the audit runs
is the application, not the model in isolation. An earlier version of this table
used the bare-model figure of 1.19 GB instead, giving `S_eff` 83.0 and a total of
83.6. That basis was 1.5 points of final score too generous, and it was our own
number to get right: no judge would have caught it, and the first place it would
have shown up is a measured score lower than the one we published.

**Honest caveat:** 50 samples carries roughly ±6% uncertainty, so this does not
establish that the 1.5B is *more* accurate. It establishes that the 3B failed to
demonstrate the advantage it needed, and the tie-break goes to the model that is
better on everything else.

**The 50-sample figures above are optimistic, and we measured by how much.**
Re-running the 1.5B at **200 samples** gives **0.695 `acc` / 0.720 `acc_norm`**,
against 0.740 / 0.780 at 50 — so the headline accuracy in §6.0 was 4.5 points
high on `acc` and 6.0 on `acc_norm`. The 0.5B moved far less, 0.620/0.620 to
0.610/0.605, which is what a genuinely weaker model looks like: it has less room
for a lucky sample.

The gap between them narrows from 12.0/16.0 points to **8.5/11.5**, and matters
more because it is now measured at roughly ±3.4% rather than ±6%. An 8.5-point
gap at that uncertainty is real; the same gap at 50 samples was not clearly
distinguishable from the noise the control below demonstrates.

The 3B was **not** re-run at 200 samples, so §3.3's central comparison still
rests on 50. Its conclusion does not depend on the accuracy tie — the 3B was
rejected for costing 1.00 GB of peak RSS — but the tie itself should be read as
"no detectable difference", not as equality.

**How little a tie at this sample size is worth, measured rather than
estimated.** While evaluating a candidate model we re-ran the *incumbent* on a
newer llama.cpp as a control. Same weights, same 50 questions, same four pinned
cores; only the inference engine changed:

| Qwen2.5-1.5B Q4\_K\_M | `acc` | `acc_norm` |
|---|---|---|
| llama.cpp 0.3.2 (shipped) | 0.740 | 0.780 |
| llama.cpp 0.3.16 | 0.720 | 0.740 |

Two points of `acc` and four of `acc_norm` moved with **no change to the model at
all**. That is the measurement apparatus wobbling, and it is the same order as
the differences this section reports to three decimal places. So "identical
`acc`, lower `acc_norm`" should be read as *the 3B showed no detectable
advantage*, not as a precise tie — the harness cannot resolve differences this
small, and we should not write as though it can.

This does not disturb the decision. The 3B was rejected for costing 1.00 GB of
peak RSS, a gap fourteen times larger than this noise band and one that
transfers between machines. But it is a caution against building any future
argument on a 2-point ARC difference, and it is why §7.10 lists the sample size
as a known limitation rather than a footnote.

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

### 3.3b The quantisation format mattered more than the file size

Having settled the parameter count, the remaining question was how to pack the
weights. The answer was not the one bytes-per-token reasoning predicts, and it
was only visible by measuring on the **profiler's own toolchain**.

`profiler/adtc-profiler/Dockerfile` builds llama.cpp with `GGML_AVX`, `AVX2`,
`AVX512`, `FMA`, `F16C` and `BLAS` all **OFF**, then measures throughput by
invoking `llama-bench -p 512 -n 128 -ngl 0` on the submitted GGUF. Two
consequences follow, and neither is obvious:

* **`S_perf` never sees our application.** No retrieval, no compression, no
  prompt engineering. It loads the weights file and generates 128 tokens.
  Nothing in the pipeline can move that number.
* **With no SIMD, dequantisation cost dominates.** On a normal build the extra
  arithmetic in a K-quant hides behind memory bandwidth. On a scalar build it
  is the bottleneck.

Measured, same model, same 4 threads, profiler flags:

| | file | `pp512` prefill | `tg128` generation |
|---|---|---|---|
| Q4\_K\_M | 1.117 GB | 24.95 | **11.32 tok/s** |
| **Q4\_0** | **1.066 GB** | **58.30** | **30.50 tok/s** |
| IQ4\_XS | 0.896 GB | 11.72 | 9.60 tok/s |

**2.7× on generation from a file 4.6% smaller.** Bytes-per-token explains almost
none of it — format does. IQ4\_XS is the control that proves the point: 20%
smaller and *slower*, because importance-matrix quantisation costs more
arithmetic than it saves in bytes.

We nearly missed this. On our own AVX2+OpenBLAS build the two are
indistinguishable — 38.5 against 38.0 tok/s — and in the full application Q4\_0
is actually **slower** (32.8 against 38.0), because llama.cpp's AVX2 kernels for
K-quants are excellent. Optimising against the development machine would have
pointed exactly the wrong way. This is §6.5's lesson recurring: the measurement
apparatus can be wrong in the same way the system can.

**The accuracy price, measured at 200 samples rather than 50:**

| ARC-Easy, 200 samples | `acc` | `acc_norm` |
|---|---|---|
| Q4\_K\_M | 0.695 | 0.720 |
| **Q4\_0 (shipped)** | **0.685** | **0.695** |

One point of `acc`, inside the ±3.4% band at that sample size. The submitted
test prompt produces the same correct Cassava Mosaic diagnosis under both, both
refuse the dosage question, and coverage and refusal are identical because
retrieval does not depend on the quant. Peak RSS falls 1.68 → **1.63 GB**.

**Why this is worth ~17 points and cannot cost any.** `S_perf` is scored
relative to the fastest submission. At 11.32 tok/s scalar we would score ~44
against a field containing any 0.5B-class entry; at 30.5 we are faster than a
0.5B at Q4\_K\_M (25.94) and plausibly set `TPS_max` ourselves. If no team
ships anything fast, the gain shrinks toward zero — but it never reverses, which
is what makes the trade worth one ARC point.

Both files are Qwen's own GGUF builds, so the provenance discipline in
`models.lock.json` is unchanged — unlike the third-party IQ4\_XS, which was
rejected on measurement before provenance became a question.

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

**Shipped configuration: Qwen2.5-1.5B-Instruct Q4\_0** (see §3.3b for why the
quantisation format, not the parameter count, decides `S_perf`).

| Metric | Measured | Ceiling / target |
|---|---|---|
| Peak RSS, full application | **1.63 GB** | 7 GB — **PASS**, → `S_eff` **76.7** |
| RSS after model load, before generation | 1.41 GB | — |
| Throughput, **audit build** (`llama-bench`, scalar) | **30.5 tok/s** | this is what `S_perf` measures — §3.3b |
| Throughput, dev build (full application) | 32.8 tok/s (p50) | not scored; SIMD favours K-quants |
| Time to first token (p50 / p95, dev build) | **21.3 s / 24.1 s** | — |
| Prompt size sent to the model | ~652 tokens | — |
| Warm-up (startup, one-off) | 0.4 s | — |
| ARC-Easy `acc` / `acc_norm` (**200 samples**) | **0.685 / 0.695** | Q4\_K\_M: 0.695 / 0.720 |
| Retrieval coverage (**held-out test**, relevance-checked) | **100%** (28/28) | 80% — **PASS** |
| Refusal accuracy (**held-out test**) | **100%** (4/4) | 80% — **PASS** |
| Retrieval latency p50 / p95 | 64 ms / 94 ms | — |
| Searchable index | **40,537 chunks / 995 documents** | — |
| Index build, 43,177 chunks | one-off, peak 0.55 GB | one-off |
| Sustained-load throughput decay (20 min continuous) | **1.0%** | <10% — no throttling signature (§6.7) |
| Peak core temperature | null — host exposes no sensor (§6.7) | 85 °C penalty threshold |
| **Offline operation** | **verified, `--network none`** | — |

Every row above is read from `bench/results/benchmark.json`, `coverage.json` and
`thermal.json`, regenerated on 2026-08-14 and committed alongside this report.
That provenance sentence is here because it was not true of the previous version
of this table, and the failure is instructive enough to record rather than
quietly correct.

**The table was reporting numbers no committed artifact contained.** It quoted
1.73 GB, 40.0 tok/s and TTFT of 22.1 s / 26.6 s while the only committed
`benchmark.json` described the *3B* configuration rejected in §3.3. The figures
had been transcribed from a run whose output was never committed, and then
edited by hand as the design changed. Re-running the harness against the shipped
1.5B produced the table above: peak RSS 1.71 GB rather than 1.73, throughput
38.0 rather than 40.0, and TTFT p50 17.9 s rather than 22.1 s.

The corrections are small and two of the three are *favourable*, which is the
part worth dwelling on. Nothing here was inflated on purpose, and that is
precisely why it went unnoticed for so long: a hand-maintained number that
happens to be pessimistic raises no alarm and still cannot be reproduced by a
reviewer. §8 claims every figure traces to a committed script. Until this run,
the headline table was the one place that claim did not hold.

**Predicted 1.85 GB; measured 1.71 GB.** The previous prediction was 3.26 GB —
a 1.5 GB miss that §6.0 explained away as conservatism, on the grounds that it
budgeted for a 3B model and a resident translation model. That explanation was
correct and was also the wrong response: `MEMORY_BUDGET` in `agbe/config.py`
exists so that a drift between intent and reality shows up as a diff, and it had
itself drifted to describe a design two decisions out of date. It now describes
the shipped system, and the gap it reports is 8%.

**Two chunk counts appear above and both are correct.** The index *builds*
43,177 chunks; the index that *serves* holds 40,537, from 995 documents. The
difference is 2,640 chunks belonging to seven NoDerivatives-licensed documents,
dropped by `agbe/rag/index.py` at load rather than at build, because the licence
gate learned to reject NoDerivatives after those documents were already embedded
and a re-index costs hours. Enforcing at load means the shipped system cannot
retrieve from them whatever the on-disk index contains.

That distinction had gone unreported, and it mattered more than a footnote: the
committed coverage result predated the exclusion, so the headline 100% had been
measured against a larger index than the one that ships. Since removing chunks
can only lose retrieval hits, the figure needed re-earning rather than assuming.
Re-run on the held-out test split against the shipped index: **still 100%
(28/28), refusal still 100% (4/4)**. The claim survives; it just had not been
tested in the form it was being made.

**The chunk count grew 36% for a reason, and it is §6.9's.** The corpus is
unchanged at 1,002 documents; what changed is where chunks are cut. Splitting on
section headings and isolating those headings from their body text produces more,
smaller, more precisely-labelled passages — 31,682 → 36,526 → 43,177 across two
re-index cycles. That is the fix that made the submitted test prompt answerable,
and its costs are visible above: retrieval latency p50 rose 64 → 70 ms at one
point and prompt size *fell* from 773 to 652 tokens, because a smaller chunk
carries less that compression has to discard.

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
from 65 s to 18 s and wall clock for six questions halved. This defect class —
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
appear in a retrieved passage. True coverage was **80%**. The earlier figure was
inflated and is recorded here rather than quietly replaced.

The corpus harvest that followed closed most of that gap: re-run on the shipped
index, dev coverage is **93.3% (28/30)**, committed as
`bench/results/coverage_dev.json`. Both remaining failures are weather questions
— season onset and flood response — and both fail as *"no passage above
threshold"*, the corpus-gap mode rather than the retrieval mode. That is
consistent with §1's claim that extension literature carries weather *decision
guidance*: it carries less of it than of crop pathology, and the two questions
we cannot answer are the honest edge of that.

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

**Re-measured after re-chunking, and 0.70 survives.** The sweep above ran on a
31,682-chunk index. §6.8 rebuilt it to 43,177 smaller, heading-prefixed chunks,
which raises scores systematically — so the constant had to be re-earned rather
than inherited:

| Floor | Coverage (dev) | Refusal (dev) |
|---|---|---|
| **0.70** | **96.7%** | **100%** |
| 0.74 | 93.3% | 100% |
| 0.78 | 46.7% | 100% |
| 0.82 | 16.7% | 100% |

The cliff between 0.74 and 0.78 is a fall off a table, and 0.70 still sits at the
knee — a constant chosen on one index landing correctly on a very different one.

**Two things this sweep cannot do, stated because they bound what it proves.**
Dev refusal reads 100% at *every* value tested, so the sweep has no resolution on
the axis it exists to trade against; the refusal failures that mattered in §6.8
were in the held-out split and could not be tuned for. And `--sweep` itself had
silently stopped measuring: its hardcoded 0.20–0.55 range now returns eight
identical rows, because the whole window sits below where anything happens. The
range is now derived from the observed score distribution, so a sweep on a future
index finds its knee instead of reporting noise shaped like data.

`make bench` records peak RSS, TTFT and tokens/sec labelled with host CPU and
RAM. `make coverage` reports coverage and refusal accuracy; `make coverage
ARGS="--sweep"` re-derives this table.

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

### 6.8 The same prompt broke again, one layer down

§6.3 fixed how the query was *prepared*. The prompt then failed a second time
for an unrelated reason, and the second failure is more instructive than the
first because nothing in the evaluation set could see it.

**The symptom.** Asked the submitted test prompt, the system answered **Cassava
Brown Streak Disease** for textbook cassava mosaic symptoms, and omitted clean
planting material — the single most important control. Coverage read 100% on
both splits, before and after.

**Not a model failure.** Every model tested — 0.5B, 1.5B Q4\_K\_M, 1.5B Q4\_0
and 3B — gave the same wrong answer, and later the same right one. A 2× larger
model changes nothing when the context is mislabelled, which is the strongest
evidence in this report for the claim §3.3 rests on: the corpus supplies the
agronomy and the model reads it.

**The cause was a chunk boundary.** The top-ranked passage carried mosaic's
symptom description under brown streak's heading:

> "…the leaves on infected plants become small, distorted and twisted… the
> plants become stunted (Figure 6.5). **39 Cassava Brown Streak Disease
> (CBSD)** CBSD is a disease of cassava caused by…"

The mosaic section's own heading had stayed behind in the previous chunk. The
model read what it was given and named the only disease present.

**Six bugs, each individually sufficient to preserve the defect:**

| | |
|---|---|
| `_SPLIT` broke only on blank lines and bullets | a heading on a single newline was never isolated |
| the heading break required a half-full buffer | ~230 chars against a ~580 threshold |
| overlap re-imported the symptoms | after the first two were fixed |
| field labels were treated as section titles | `Damage symptoms:` became a chunk's identity |
| headings shared a paragraph with their body | so `looks_like_heading` never matched |
| — | each of the last three surfaced only by testing the fix, not by reasoning about it |

**Measured after the fix**, on a rebuilt index of 43,177 chunks:

| | before | after |
|---|---|---|
| Diagnosis | Cassava Brown Streak | **Cassava Mosaic Disease** |
| Dev coverage | 93.3% | **96.7%** |
| Weather corpus gaps | 2 | **0** |
| Off-crop docs in top 6 | yes | reduced |

**And it caused a regression worth recording.** Re-chunking dropped test-split
refusal from 100% to 66.7%: *"Which bank gives the best loan to farmers in Oyo
State?"* and *"How do I register my farmland title?"* began retrieving CGIAR and
IFPRI rural-credit and land-tenure papers above the floor. Those two had never
had a scope rule — they were refused only because no chunk happened to clear
0.70, and better-labelled chunks removed the accident. A threshold had been
masking a missing rule.

Raising the floor was measured and rejected: dev coverage is 96.7% at 0.70,
93.3% at 0.74 and **46.7% at 0.78**, while dev refusal reads 100% at every
value — so the floor cannot be tuned for this without spending coverage to buy
an improvement dev cannot measure. `scope.check()` now refuses financial and
legal-administrative questions before retrieval, for a stated reason. Refusal
returns to 100% (4/4 on the corrected split; 6/6 on the split in use when the
regression was found).

**Disclosure.** Those two questions are in the held-out split. They were
inspected after they failed and the rule was written afterwards, so the 6/6 is
guaranteed by construction and is **not** a held-out measurement for this
category. `bench/split.py` exists to prevent exactly this. The rule is derived
from §1's four advisory areas rather than from the two questions, and it catches
phrasings invented independently — but the contamination is real and is recorded
rather than quietly enjoyed.

The full investigation, including three fixes that were measured and reverted,
is in [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md). Open items are in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

### 6.9 Coverage is not correctness, and now there is a number for the gap

Every retrieval figure in this section counts whether a passage containing an
expected term was **retrieved**. Whether the answer *used* it is a different
question, and the two came apart badly enough to justify a second tool.

Through the whole of §6.8's investigation, dev coverage sat at 93.3% while two
changes made answers worse, and at 96.7% before and after the change that made
them right. A metric that does not move when the product improves or degrades is
not measuring the product.

`bench/answers.py` scores the same `expect_any` terms against the **generated
answer** and cross-tabulates against retrieval:

| | answer has a term | answer lacks one |
|---|---|---|
| **retrieved** | `OK` | `NOT_USED` — a generation problem |
| **not retrieved** | `UNGROUNDED` | `MISSED` — a corpus or retrieval problem |

First run, dev split, shipped configuration (Q4\_0). Q4\_K\_M was measured
separately and returned identical counts, so the quantisation change in
§3.3b costs nothing at the answer level either:

| | |
|---|---|
| **Answer accuracy** | **90.0%** (27/30) |
| `NOT_USED` | 1 |
| `MISSED` | 2 |
| **`UNGROUNDED`** | **0** |

**Coverage overstates by 6.7 points.** Every retrieval number in §6 is the
optimistic one, and that should be read alongside them.

`UNGROUNDED` is the reason this reports four numbers rather than one: it flags an
answer asserting something **no retrieved passage contains**, which is the
grounding guarantee in §5 being broken. It is 0 for the shipped 1.5B. It is not
0 for the 3B, which given identical context produced *"plant disease-free
cassava seedlings"* — cassava grows from stem cuttings — and *"avoid overhead
irrigation to reduce whiteflies"*, neither in any source. A larger model fills
gaps more fluently, which in an advisory system is a liability rather than a
feature.

**What it does not fix.** It uses the same 70 short, direct questions, so §7.11's
blind spot survives: the defect in §6.8 was found by *reading an answer*, not by
any metric, and framed variants remain unwritten.

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
12. **Control advice is not retrieved for the flagship prompt.** The diagnosis is
    now correct (§6.8), but `clean planting material` appears **zero times** in
    the passages sent to the model, though the corpus carries it. Three fixes
    were measured and reverted — a wider candidate pool (no effect: fusion rank,
    not pool size, is the constraint), reserved slots at `top_k=6` (regressed
    the diagnosis), and additive slots at `top_k=8` (correct diagnosis, 40% more
    prefill, advice still partly wrong). Recorded in `docs/FINDINGS.md` F-01.
13. **Answer accuracy is 90.0%, below the coverage figures quoted throughout
    §6.** Coverage measures retrieval; §6.9 measures the answer. The 6.7-point
    gap is one `NOT_USED` and two `MISSED` questions on the dev split.
14. **The dev/test split was not stable when questions were added — fixed, and
    the fix moved the split once.** `bench/split.py` promised that "adding a
    question later assigns it to a side without disturbing anything already
    assigned"; it assigned by *position* within a hash-sorted stratum, so any
    addition moved the cut point. Verified by adding 8 questions, which moved
    two existing ones from test to dev.

    Assignment is now by hash *threshold*, so a question's side depends only on
    its own id. Correcting it necessarily reshuffled once: `oos-04` and `oos-05`
    moved to dev, and test went from 37 questions to 32. **Both splits were
    re-measured and the percentages did not move** — test 100% coverage and 100%
    refusal on 28 and 4 questions rather than 31 and 6; dev 96.7% → 97.0%. The
    figures in §6.0 are the post-fix ones.

    The cost is that stratum balance is now approximate rather than exact, which
    at these sizes is the right trade for a guarantee that holds. `docs/FINDINGS.md` F-11.
15. **Refusal has no resolution on the dev split.** Three out-of-scope questions
    means refusal moves in 33-point steps and reads 100% at every floor from
    0.20 to 0.86. No threshold work can be validated against it, and the two
    refusal failures that mattered were both in the held-out split (§6.8).
16. **The financial/legal scope rule is contaminated.** Its two motivating
    questions are in the test split and were inspected before the rule was
    written (§6.8). The resulting 6/6 refusal is guaranteed by construction.

---

## 8. Reproducibility

```bash
make build            # ubuntu:22.04 reference image
make fetch-models     # checksum-verified weights (~1.2 GB, both models)
make fetch-corpus     # vetted documents + provenance manifest
make index            # build retrieval index
make coverage         # retrieval coverage + refusal accuracy (dev split)
make coverage ARGS="--split test"    # the held-out figures quoted in §6.0
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

All seven fixed messages the system emits in Pidgin — the refusal, the
banned-pesticide warning, the redaction notice shown when an invented dosage is
removed, the separate refusal shown when a dosage is *asked for*, the veterinary
withdrawal-period warning, the staleness warning and the pesticide-exposure
medical redirect, plus 22 domain terms — were reviewed and approved by a
Nigerian Pidgin speaker on 2026-08-07. The set is committed at
`bench/pidgin_eval.json` as `pcm-s01` through `pcm-s07`.

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
**1.63 GB against the 7 GB ceiling**. See §3.

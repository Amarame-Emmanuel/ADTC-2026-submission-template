# Retrieval investigation — the submitted test prompt

**Status: OPEN.** The defect described here is not fixed. Three changes were
made and measured; one worked, two did not, and the root cause is still in
place. This document records the sequence because the failed attempts are the
useful part — each was plausible, and each was rejected by measurement rather
than by argument.

Companion to `REPORT.md` §6.3, which documents an earlier defect in the same
prompt. That section's closing claim — *"a passing metric is evidence about the
metric before it is evidence about the system"* — is what this investigation
demonstrates a second time.

---

## 1. The failure

`submission/metadata.json` carries the two prompts a judge runs first. The first:

> "A smallholder farmer in Oyo State, Nigeria says: my cassava leaves are yellow
> and twisted, and the plants are small. Explain the most likely cause…"

Those symptoms are textbook **cassava mosaic disease**. The system answers
**Cassava Brown Streak Disease**, and recommends water management, soil health,
crop rotation and fertilisation — none of it cited, none of it correct for a
virus spread by whiteflies and infected planting material. **Certified or clean
planting material, the single most important control, is never mentioned.**

This is not the §6.3 defect returning. That one retrieved development-agency
reports; retrieval now returns cassava disease documents. The answer is still
wrong.

Both quantisations produce it (Q4_K_M and Q4_0, identical), so it is not a model
artefact.

---

## 2. Root cause: a chunk boundary that relabels a disease

The passage ranked **first**, from *Growing cassava: training manual for
extension & farmers in Zambia*:

> "…white chlorotic areas. **The leaves on infected plants become small,
> distorted and twisted** along the edges. As the disease progresses the
> affected leaves may reduce and **the plants become stunted** (Figure 6.5). 39
> **Cassava Brown Streak Disease (CBSD)** CBSD is a disease of cassava caused by
> either one or more viruses…"

The symptom text is the tail of the **cassava mosaic** section. The chunk
boundary cut it from its own heading and joined it to the *next* section's
title. So the passage that best matches the farmer's symptoms carries the wrong
disease name, and it is the first thing the model reads.

The model is not hallucinating. It is reading exactly what we sent it. The
latest answer quotes the passage almost verbatim — *"white chlorotic areas on
the leaves, small, distorted, and twisted leaves along the edges"* — and
attributes it to CBSD, because that is the only disease named nearby.

The corpus is not at fault. It holds:

| | chunks |
|---|---|
| `Disease control in cassava farms: IPM field guide` | 42 |
| `African cassava mosaic virus (ACMV) (Bemisia tabaci)` | 16 |
| `Cassava brown streak virus disease` | 1 |

and, across cassava-titled chunks, 139 occurrences of "mosaic" against 49 of
"brown streak". Two retrieved passages even carry the correct answer:

* *"When **cassava mosaic** attack is severe, the leaves are very small and
  distorted and the plants are stunted."*
* *"The damaged leaves do **not** become distorted as occurs with leaves damaged
  by cassava **mosaic** disease."* — the corpus explicitly disambiguating the
  two diseases.

Both were sent to the model, at positions 5 and 6. The mislabelled passage at
position 1 won.

---

## 3. What was tried

### 3.1 Diagnostic intent — KEPT, works

`agbe/rag/query.py: with_diagnostic_intent()`

§6.3 stripped instructions from the query, which was right and also incomplete.
The instruction clause — *"explain the most likely cause, and give practical
steps"* — carried the development-agency vocabulary that caused the original
bug, but it **also** carried the only signal that this is a diagnostic question.
Stripping it left a bare symptom description, which embeds close to any text
about the crop.

Measured: the dedicated ACMV page sat at **dense rank 268 of 29,959**, and zero
diagnostic documents appeared in the top 6.

Appending a fixed phrase that names no disease — `disease symptoms cause control
management` — lifts *Disease control in cassava farms: IPM field guide* into
four of the top six, with dense scores rising from ~0.69 to ~0.81 against a 0.70
floor. Dev coverage unchanged at 93.3%. Gated on symptom vocabulary so timing,
price and practice questions are untouched.

**A rejected earlier version is worth recording:** the first expansion tried was
`cassava mosaic disease virus stunting leaf distortion clean planting material`,
which put the ACMV page at rank 1. It was discarded as circular — it names the
diagnosis being searched for. A system that must be told the answer to find it
has not been fixed.

### 3.2 Compression control reserve — REVERTED, not in the tree

`agbe/rag/compress.py: _CONTROL_ADVICE`, `CONTROL_BUDGET_SHARE`

Hypothesis: sentences are ranked by similarity to a symptom-dominated query, so
symptom sentences take every slot and management sentences are compressed away.
Fix: reserve 40% of each passage's budget for control advice, mirroring the
existing `_SAFETY_CRITICAL` force.

**Measurement rejected the hypothesis.** Counting terms in the retrieved chunks
*before* compression:

| term | in retrieved chunks | in compressed |
|---|---|---|
| `clean planting` | **0** | 0 |
| `certified` | **0** | 0 |
| `variety` | **0** | 0 |
| `cutting` | 1 | 0 |

The control advice was never retrieved. Compression cannot drop what it was
never given. The reserve did fire (`control_sentences_kept: 5`) and the
lexicon proved far too broad — `use`, `plant`, `control`, `manage`, `practice`
match ordinary sentences everywhere, so the reserve was spent on generic text
while the one sentence that mattered still lost on score.

### 3.3 Dual-intent retrieval — REVERTED, not in the tree

`agbe/rag/query.py: management_query()`, `agbe/advisor.py: _merge_reserving()`

Hypothesis: the farmer asks two questions — what is wrong, and what do I do —
and one symptom-shaped query vector can only answer the first. Fix: a second
search recast toward management, with 2 of 6 slots reserved for it.

It did what it was designed to do and made things worse overall:

| term, in retrieved chunks | before | after |
|---|---|---|
| `clean planting` | 0 | **1** |
| `resistant` | 2 | **3** |
| `tolerant` | 1 | **2** |
| **`mosaic`** | **7** | **1** |
| `brown streak` | 9 | 2 |

Control vocabulary arrived; the diagnostic half collapsed. Two reserved slots
displaced six of seven mosaic mentions. And compression then removed the new
control terms anyway — `clean planting`, `resistant`, `tolerant` and `mosaic`
all reach the model **zero** times.

The answer did not improve.

---

## 4. What this cost, and the lesson

Three changes, one useful. The two that failed share a cause: the root defect
had already been identified in §2 — a chunk boundary — and both fixes addressed
something else. Each hypothesis was reasonable in isolation and neither was
tested against the thing already known to be broken.

The reading test caught every failure within minutes, which is the process
working. The sequencing was the error: a measured root cause should be fixed
before plausible adjacent ones.

This is the same lesson as §6.5, one level up. There, the measurement apparatus
was wrong and would have condemned the system. Here, the diagnosis was right and
the *fixes* addressed the wrong layer.

---

## 5. Open work

1. **Chunk on section headings.** A disease's symptoms must stay attached to its
   own name. Requires a re-index (~100 minutes) and is the actual fix.
2. ~~Revert §3.2 and §3.3~~ — **done.** Both were reverted rather than kept as
   harmless, for a reason worth stating: an unproven change carried into the
   next experiment destroys attribution. With the reserve and the second search
   still in place, an improvement after the chunker fix could not be assigned to
   the chunker. That is the same defect that made §3.3's model A/B unreliable
   when its two columns were measured on different harnesses.

   Either idea may return once the root cause is fixed, narrowed: the control
   lexicon to specific agronomic phrases (`clean planting material`,
   `certified`, `resistant variety`, `disease-free cuttings`) rather than common
   verbs, and the management search only after diagnostic passages are secured
   rather than competing with them for slots.
3. **Off-crop documents.** *Guide for sustainable maize production in Ghana*
   ranks third on a cassava query throughout. Unaddressed.
4. **`hybrid_search` candidate pool** is 20 per retriever
   (`agbe/rag/index.py:115`). Raising it to 400 changed nothing here and
   returned *fewer* passages, since a larger pool reshuffles the fusion and more
   chunks fall below the floor. Recorded so it is not retried.
5. **The evaluation set cannot see any of this.** Dev coverage held at 93.3%
   through every change above, including the ones that made the answer worse.
   §7.11 already records that all 70 questions are short and direct while this
   prompt is long and framed. That blind spot is now demonstrated twice.

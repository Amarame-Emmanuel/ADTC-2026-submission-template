# Open findings

Non-blocking issues found during work, filed here for review rather than fixed
on the spot. Each records what was measured, so none of them has to be
rediscovered.

Blocking problems are not filed here — those stop work and get raised directly.

## Priority

| | finding | why it ranks here |
|---|---|---|
| **1** | **F-11** split instability | blocks F-09, F-10 and any eval-set growth; silently invalidates published figures |
| **2** | **F-01** control advice unretrieved | the main remaining quality gap; three fixes already ruled out |
| **3** | **F-13** `bge-base` embedder | the one lever attacking the *cause* behind F-01, §6.2 and §6.8 |
| **4** | **F-04** off-crop documents | consumes the slots F-01 needs |
| 5 | F-09 refusal resolution | blocked on F-11 |
| 6 | F-05 `oos-09` refused by luck | same fragility that broke `oos-07`/`oos-08` |
| 7 | F-02 two questions retrieve nothing | overlaps F-01/F-13 |
| 8 | F-12 mmap | +0.13 points; ride the next re-index |
| 9 | F-06 score/order mismatch | one comment |
| — | F-03, F-07, F-08 | resolved or disclosure-only, see below |

**Closed since this file was written:** the Q4_0 quantisation swap (shipped —
§3.3b), the sweep range (now derived from observed scores — §6.2), and the
answer-level metric (`make answers` — §6.9).

---

## Retrieval and answers

### F-01 · Control advice is never retrieved
**Severity: high — this is the main remaining quality gap.**

`clean planting material` appears **0 times** in the passages sent to the model
for the submitted test prompt, along with `certified` 0 and `variety` 0. The
corpus carries it — the ACMV page says *"select cuttings from stem branches
instead of the main stem"* — but retrieval does not reach it.

The diagnosis defect is fixed; what to do about the diagnosis is not. A farmer
gets the right disease name and generic advice.

**Three fixes have now been tried and reverted. Do not try a fourth without a
new idea — try these in order of what has already been ruled out.**

**Ruled out 1 — a wider candidate pool.** `hybrid_search(candidates=...)` from
20 to 200 returns a **byte-identical top 6**. Reciprocal-rank fusion is the
binding constraint, not pool size: a chunk at dense rank 30 scores `1/(30+k)`
and cannot outrank chunks at dense ranks 1–6 that also appear in the lexical
list. More candidates only append passages that lose by construction. (This also
corrects an earlier note in `RETRIEVAL.md` §5 that blamed pool size.)

**Ruled out 2 — reserved slots at `top_k=6`.** A second management-intent search
holding 2 of 6 slots surfaced control vocabulary (`cutting` 1→4, `healthy` 0→2)
and **regressed the diagnosis back to brown streak**, undoing three re-index
cycles of work. `mosaic` fell from 8 mentions to 1. Reservation from a fixed
budget is zero-sum.

**Ruled out 3 — making the slots additive at `top_k=8`.** Diagnosis recovered
(mosaic 1→3, correct answer) and control words arrived, but prefill rose from
652 to 1,044 tokens — roughly 40% more TTFT — and the advice was still poor and
partly *wrong*: *"plant in dense vegetation"*, when dense vegetation harbours
the whitefly vector. `top_k=10` added nothing over 8.

**What the evidence points to.** Re-chunking moved control passages from dense
rank 268 to **rank 30**, so they are close now. The blocker is that fusion
ranking cannot see them and reservation costs more than it buys. Promising
directions not yet tried:

- score fusion weighted by score rather than by rank, so a 0.77 chunk at rank 30
  can beat a 0.72 chunk at rank 5;
- a re-ranking pass over the fused candidates before `top_k` is applied;
- accepting that `clean planting material` may simply be too rare in the corpus
  — it appears in the ACMV page but not in the IPM guide chunks that dominate
  retrieval, which is a corpus problem rather than a retrieval one.

The last is worth checking first and is cheap: count how many cassava chunks
carry that guidance at all.

**Unaffected by the Q4_0 swap.** Retrieval does not depend on the quantisation,
and the measurements above were re-confirmed on the shipped configuration:
identical coverage, identical answer accuracy, `clean planting` still 0. This is
a retrieval problem and no model change will move it — which the multi-model
test already demonstrated, since 0.5B, 1.5B and 3B all failed and recovered
together.

### F-02 · Two questions retrieve nothing relevant
**Severity: medium.** From the first answer-level run (`answers_dev.json`):

- `crop-05` — *"There are white cottony insects on the growing tip of my cassava"*
  (cassava mealybug). The Zambia manual's contents list a "Cassava Mealybug"
  section, so the corpus almost certainly holds it.
- `crop-22` — *"There are white winding lines inside my tomato leaves"*
  (leaf miner).

Both are `MISSED`: neither retrieved nor answered. Both are highly visual,
plain-language symptom descriptions — exactly the register the system is for.

### F-03 · Evidence in context, unused by the model
**Severity: low — probably noise, on the evidence of two runs.**

The dev split returns exactly one `NOT_USED` under both quantisations, but **not
the same question**:

| run | `NOT_USED` |
|---|---|
| Q4_K_M | `crop-23` — *"Small white insects fly up when I shake my tomato plants"* (sources had `whitefly`, `bemisia`) |
| Q4_0 (shipped) | `crop-25` — *"Should I plant the same crop in the same place every year?"* (sources had `rotation`, `rotate`, `same field`) |

A stable defect would fail the same question twice. One failure that moves
between runs at a constant rate of 1/30 looks like generation variance at
`temperature=0.3`, not a bug with a fix.

Worth re-checking if the count ever rises above one, which is the signal that
would distinguish a real generation problem from sampling noise. Not worth
chasing at n=1.

### F-04 · Off-crop documents rank inside the top 6
**Severity: medium.** *Guide for sustainable maize production in Ghana* and
*Maize Seed Production* have both appeared in the top 6 for an explicitly
cassava question. They consume slots that control advice needs (see F-01).
A crop filter or down-weighting is the obvious approach; the query already names
the crop and `scope.py` already detects crops.

### F-05 · `oos-09` is refused by luck
**Severity: medium.** *"How do I fix my motorcycle engine?"* is refused only
because nothing in an agriculture corpus resembles engine repair. It has no
scope rule. That is the same fragility that broke `oos-07` and `oos-08` when
re-chunking changed the score distribution — a threshold masking a missing rule.
It will hold until the corpus changes shape again.

---

## Tooling and measurement

### F-12 · The mmap saving needs a build-time fix, not a load-time one
**Severity: low — ~46 MB, about 0.13 final points.**

`index.py:260` does `vectors = np.asarray(vectors)[keep]`, materialising the
whole memory-mapped matrix into the heap to drop licence-excluded rows —
directly contradicting the comment four lines above it about letting the OS page
the index in on demand.

It cannot just be deleted; the filtering has to happen somewhere. The options:

1. **Filter at build time** so the on-disk index contains only usable chunks and
   load is a pure mmap. Correct, and the benefit only arrives with the next
   re-index (~2 hours), so it should ride along with one that is happening
   anyway rather than triggering its own.
2. **Compaction cache** — write the filtered arrays once on first load and mmap
   that thereafter. Avoids the re-index but adds a file lifecycle and a
   cache-invalidation question.
3. **Mask instead of copy** — keep the mmap, carry a boolean mask, set excluded
   rows to −1 at search time. Cheapest to write, but the whole matrix is touched
   by every query anyway, so the resident-set saving is smaller than it looks.

Option 1 is right, timed to the next re-index. Not worth two hours of compute on
its own for 0.13 points.

### F-06 · `hybrid_search` orders by one score and reports another
**Severity: low, but confusing.** Results are ordered by reciprocal-rank fusion
while `.score` carries the *dense* similarity, and `min_score` filters on the
dense value. So a printed result list is not monotonically decreasing in the
score beside it, and two different criteria decide inclusion and position.

This is deliberate and documented inside the function, but nothing at the call
site says so, and it cost time to work out while debugging. A one-line comment
where `SearchHit.score` is consumed would settle it.

### F-07 · Answer accuracy and coverage disagree by 6.7 points
**Severity: informational — RESOLVED, now stated in the report.**

Dev coverage is 96.7% while answer accuracy is 90.0%. Coverage counts a question
as covered when a passage containing an expected term is retrieved; the answer
may still not use it. Every retrieval number in `REPORT.md` §6 is therefore the
optimistic one.

Written up in §6.9 and in the opening status note, so a reader is told rather
than left to notice. No further action.

### F-13 · `bge-small` is implicated in three separate failures
**Severity: high — the one lever that attacks a cause rather than a symptom.**

The embedder's compressed similarity range shows up as three different problems
that have been treated as three problems:

- §6.2 — everything above ~0.55 matched, forcing the floor to 0.70
- §6.8 — the cassava mosaic page sat at dense **rank 268** while rank 1 to rank
  268 spanned only **0.089** of score
- F-01 — control passages sit at rank 30 and fusion cannot reach them, because
  ranks are packed too tightly for a good chunk to climb

That is one 33M-parameter, 384-dimension encoder with poor separation, observed
three times. `bge-base-en-v1.5` is 109M at 768 dimensions with materially better
discrimination.

**Cost:** ~+140 MB resident (model ~110 MB vs ~35 MB, vector matrix 132 MB vs
66 MB), so `S_eff` 76.7 → ~74.7, about **−0.4 final points**. Retrieval latency
roughly doubles to ~130 ms, which is irrelevant against an ~18 s TTFT.
Generation is untouched, so `S_perf` does not move. Needs a **3–4 hour
re-index**.

**Probe before committing to that.** Embed only the 70 evaluation questions and
the cassava chunks with `bge-base` and check whether the ACMV page moves from
rank 268 into the top 10. Minutes, not hours, and it predicts the full re-index.

**Judge it with `make answers`, not with term counts.** The baseline is 90.0%
answer accuracy with 2 `MISSED` — and `MISSED` is precisely the quadrant a
better embedder should move. Four retrieval changes were argued for on reasoning
this week and three of them failed; this one should be measured.

---

## Contamination and evaluation integrity

### F-08 · Test-split contamination, financial/legal category
**Severity: must be disclosed in the report.**

`oos-07` and `oos-08` are test-split questions. They were inspected after they
failed, and the `_FINANCIAL_LEGAL` scope rule was written afterwards. The rule
is derived from §1's domain definition rather than from those two questions, and
it catches phrasings invented independently — but the resulting 6/6 refusal
figure is **guaranteed by construction** and is not a held-out measurement.

`bench/split.py` exists to prevent exactly this. It should be recorded in the
report rather than quietly enjoyed.

### F-11 · The dev/test split is not stable when questions are added
**Severity: high — it silently invalidates published numbers, and it blocks F-09.**

`bench/split.py` documents a guarantee it does not keep:

> *"Adding a question later assigns it to a side without disturbing anything
> already assigned — which a shuffled split would not guarantee."*

Adding 8 out-of-scope questions moved `oos-05` and `oos-08` from **test to
dev**. Verified, then reverted.

The cause is at `split.py:76`:

```python
ordered = sorted(group, key=lambda q: _bucket(q["id"]))
cut = round(len(ordered) * DEV_FRACTION)
```

The hash fixes the *order* within a stratum; assignment is by *position*
relative to a cut that moves whenever the stratum's size changes. So any
addition can push existing questions across the boundary.

`SPLIT_SALT` exists so that "a reshuffle is an explicit, visible act rather than
something that happens by accident" — and a reshuffle happens by accident, from
the one operation the docstring says is safe.

**Fix:** assign by hash threshold rather than position —
`dev if _bucket(q["id"]) < DEV_FRACTION else test`. Stable under addition by
construction. The cost is that strata are balanced only approximately rather
than exactly, which at these sizes is a fair trade for a guarantee that holds.

**Consequence until fixed:** the evaluation set cannot be extended without
invalidating every previously reported dev and test figure, so F-09 and F-10
are both blocked on this.

### F-09 · Refusal has no resolution on dev
**Severity: medium.** Dev holds 3 out-of-scope questions, so refusal moves in
33-point steps and reads 100% at every floor from 0.20 to 0.86. No threshold
work can be validated against it, and the two failures that mattered were both
in test. More out-of-scope questions across the categories — finance, legal,
live data, off-domain, wrong crop — would give the axis resolution.

### F-10 · The evaluation set still cannot see framed questions
**Severity: medium, and long-standing (§7.11).** All 70 questions are short and
direct. The submitted test prompt is long and framed, and it retrieved the wrong
documents while coverage reported 100%. Dev coverage then sat unchanged through
two changes that made answers worse and one that made them right.

The answer-level checker helps but does not close this: it uses the same 70
short questions. Framed variants remain unwritten.

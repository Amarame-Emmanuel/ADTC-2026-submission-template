# Open findings

Non-blocking issues found during work, filed here for review rather than fixed
on the spot. Each records what was measured, so none of them has to be
rediscovered.

Blocking problems are not filed here — those stop work and get raised directly.

## Priority

| | finding | why it ranks here |
|---|---|---|
| — | **F-24** | **fixed — floor tolerance 0.05→0.08; the right passage was being discarded at 0.623 by 0.027** |
| — | **F-23** | **fixed — polarity guard; both fabrications refuse, both correct Newcastle answers survive** |
| — | **F-18** | **fixed and shipped in 07e9f4f — the guard is live on all three entry points** |
| **1** | **F-15** fabricated claim, unmeasurable | the grounding check cannot see this class. F-23 and F-31 are guarded instances; the general case is not |
| 2 | **F-17** one document takes every slot | cap built, measured, reverted — coherence beat coverage |
| 3 | **F-14** source-digest answers | one fix measured and rejected (−9 points) |
| 4 | F-01 control advice | **4 approaches measured and declined**; one untried (score-weighted fusion). Read §6.11 first |
| 5 | **F-20** topic drift | cutworm question answered damping-off too |
| 6 | **F-21** livestock species leak | aquaculture passages in goat and chicken answers |
| 7 | **F-27** out-of-scope crops cannot be demoted | **partly fixed — answer accuracy 90.9% → 97.0%, confirmed twice**; a doc titled by PATHOGEN still leaks |
| 8 | **F-28** spurious model refusal on a clean question | ~17% of generations return "I'm sorry, but I can't assist with that"; reads as NOT_USED to the metric |
| — | **F-30** | **fixed — in-domain gate; 19 of 40 answered → 1 of 40, and 36 of 39 refusals now come from a RULE rather than the floor** |
| **1** | **F-32** the gate refused the languages it serves | Igbo and Yoruba questions declined as *out of domain* by an English-only rule — **fixed**, but it shipped for two days |
| 9 | **F-33** a guard can be weaker in streaming than in tests | money redaction never fired in the interface; unit tests passed on whole strings |
| 10 | **F-34** empty generation on a lethal condition | bloat produced NOTHING in 2 of 5 runs — **retry added, ~40% → ~10%** |
| — | **F-31** | **fixed — plural and inflection misses in four separate rules; see below** |
| — | **F-25**, **F-26** | **fixed — see below; F-26 is why a guard could be live for farmers and invisible to the metric** |
| — | **F-02** | **fixed — sign lexicon; answer accuracy 87.9% → 93.9%** |
| — | **F-13** | **probed and declined — costs 0.4 points, benefit unproven** |
| — | **F-16**, **F-19**, **F-22**, F-03, **F-06**, F-07, F-08, **F-04**, **F-05**, **F-09**, **F-11**, **F-12** | resolved or disclosure-only, see below |

**Closed since this file was written:** the Q4_0 quantisation swap (shipped —
§3.3b), the sweep range (now derived from observed scores — §6.2), the
answer-level metric (`make answers` — §6.9), **F-11** split instability, and
**F-04** off-crop retrieval, **F-05** off-domain refusal, **F-09** refusal
resolution, **F-06** the score/rank mismatch, **F-12** the mmap copy
(build-time; effective on next re-index), and **F-02** the symptom→name gap.

---

## Retrieval and answers

### F-01 · Control advice is never retrieved
**Severity: high — OPEN, but four approaches are now measured and declined. Do
not attempt a fifth without reading all four.**

`clean planting material` appears **0 times** in the passages sent to the model
for the submitted test prompt, along with `certified` 0 and `variety` 0. The
corpus carries it — the ACMV page says *"select cuttings from stem branches
instead of the main stem"* — but retrieval does not reach it.

The diagnosis defect is fixed; what to do about the diagnosis is not. A farmer
gets the right disease name and generic advice.

**Four approaches measured, none shipped.** The fourth is the most instructive,
because it produced the best single answer and the worst split-wide result.

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

**Ruled out 4 — control vocabulary in the query.** Appending *"clean planting
material resistant varieties"* to every diagnostic query. On the submitted
prompt this gave the **best answer any attempt has produced** — correct mosaic
diagnosis *and* grounded planting-material advice. Across the dev split it was
the worst:

| dev | shipped | with control vocabulary |
|---|---|---|
| Answer accuracy | **93.9%** | **78.8%** |
| `NOT_USED` | 2 | **7** |
| `MISSED` | 0 | 0 |

Five questions regressed, including the two the sign lexicon had just fixed.

**And the mechanism is the finding worth keeping.** `MISSED` stayed at 0 —
retrieval was still finding the passages. What changed was **compression**.
`compress_hits` scores sentences against the *same query vector* used for
retrieval, so padding the query with management vocabulary pulls the compressor
toward management text and away from the sentence naming the pest. The model
gets a passage containing `mealybug` and an excerpt that does not emphasise it.

**The query vector does double duty, and a change made for retrieval silently
re-tunes compression.** That is invisible to coverage — only the answer-level
metric detects it — and it retrospectively explains attempt 3's odd behaviour.
Any fifth attempt must measure compression's output, not just what was
retrieved.

Also measured: appending the literal phrase `clean planting material` **alone**
changes retrieval not at all (mosaic 8, control terms 0). The embedder cannot
reach those chunks by phrase-match.

**What the evidence points to.** Re-chunking moved control passages from dense
rank 268 to **rank 30**, so they are close now. The blocker is that fusion
ranking cannot see them and every way of forcing them in costs more than it
buys. Directions not yet tried:

- score fusion weighted by score rather than by rank, so a 0.77 chunk at rank 30
  can beat a 0.72 chunk at rank 5;
- a re-ranking pass over the fused candidates before `top_k` is applied;
- accepting that `clean planting material` may simply be too rare in the corpus
  — it appears in the ACMV page but not in the IPM guide chunks that dominate
  retrieval, which is a corpus problem rather than a retrieval one.

**PROBED. Both probes ran; neither gives a green light.**

*Probe 1 — is it in the corpus?* Yes. **90 cassava chunks (2.6%)** carry control
phrasing: `resistant variety` 27, `tolerant variety` 20, `certified` 16,
`disease-free` 15, `clean planting material` 11, `select cuttings` 5, `healthy
cuttings` 4. Crucially `healthy cuttings` sits inside *Disease control in
cassava farms: IPM field guide* — the document already supplying four of the top
six passages.

**So this is a WITHIN-DOCUMENT ranking problem, not a document-selection one.**
The right document is retrieved; the wrong chunks of it win. That is a narrower
and more tractable framing than "retrieval cannot find the guidance".

*Probe 2 — does a better embedder fix it?* Partly, and not enough to justify
itself:

| | control chunks in top 6 of the pool | gap, best incumbent − best control |
|---|---|---|
| bge-small (current) | **0/6** | +0.065 |
| bge-base | **1/6** | **+0.023** |

`bge-base` closes ~two-thirds of the gap and pulls in one control chunk, for
−0.4 final points of `S_eff` and a 3–4 hour re-index. Whether one chunk in six
changes the answer is unknown.

**What both probes show together.** Even with a materially stronger encoder,
symptom chunks still outrank control chunks — because the *query* is
symptom-shaped, not because the embedder is weak. The bottleneck is that one
query vector cannot rank two kinds of content, which is precisely what
dual-intent tried to fix and what failed twice on displacement.

**Assessment: low ceiling, deprioritise.** The gap is small (0.02–0.065), so
score-weighted fusion could plausibly flip it — but the three rejected attempts
each looked plausible in advance too, and the payoff lands on the unquantified
panel half of `S_acc` rather than on any term in the formula.

**Unaffected by the Q4_0 swap.** Retrieval does not depend on the quantisation,
and the measurements above were re-confirmed on the shipped configuration:
identical coverage, identical answer accuracy, `clean planting` still 0. This is
a retrieval problem and no model change will move it — which the multi-model
test already demonstrated, since 0.5B, 1.5B and 3B all failed and recovered
together.

### F-02 · Two questions retrieved nothing relevant
**Severity: medium — FIXED, and it is the first retrieval change that improved
answers rather than only rankings.**

`crop-05` (*"white cottony insects on the growing tip of my cassava"*) and
`crop-22` (*"white winding lines inside my tomato leaves"*) were `MISSED` —
neither retrieved nor answered.

**Probed first, and it was not a corpus gap.** The corpus holds **236 mealybug
chunks** and **204 leafminer chunks**. `crop-05`'s best mealybug chunk sat at
dense rank 18 *inside a document the query already retrieved* — right document,
wrong chunks. `crop-22`'s dedicated *Leafmining flies* document sat at rank 70
while the query returned Helicoverpa, Fusarium wilt, whiteflies and early
blight.

**The cause was a symptom→name vocabulary gap.** A farmer writes what they see;
the corpus files it under a name they do not know. `bge-small` does not bridge
"white cottony insects" to "mealybug", which is the same failure that left the
cassava mosaic page at rank 268 in §6.8.

**Fix:** `_SIGN_TO_NAME` in `agbe/rag/query.py` — nine mappings from a
distinctive visible sign to the name the literature uses. Deliberately small:
only signs where the mapping is not in doubt. "Yellow leaves" maps to a dozen
causes and is absent; "cottony masses" maps to one.

**Why this is not the circular expansion rejected earlier.** That one appended
"cassava mosaic disease virus stunting" — it named the diagnosis it was
searching for. This supplies **search terms, not answers**: retrieval must still
find the passage, the floor must still clear, every claim still comes from a
citation, and a wrong mapping degrades to worse ranking rather than putting
words in the model's mouth. It is the same class of curated, auditable lookup as
`pidgin_norm` and the safety lexicons.

**Measured end to end:**

| dev | before | after |
|---|---|---|
| Answer accuracy | 87.9% (29/33) | **93.9% (31/33)** |
| `MISSED` | 2 | **0** |
| `NOT_USED` | 2 | 2 |
| `UNGROUNDED` | 0 | **0** |
| Coverage / refusal | 97.0% / 100% | 97.0% / 100% |
| Test coverage / refusal | 100% / 100% | 100% / 100% |

On-target passages in the top-6 went 0→3 for `crop-05` and 0→4 for `crop-22`,
with the dedicated leafminer document rising from rank 70 to rank 1. `crop-23`
matched no sign and is unchanged, so the lexicon is not firing indiscriminately.

**Caveat worth keeping.** Nine hand-written mappings are judgement, not fitted
values — the same limitation §4.5 records for the corpus filter. A wrong mapping
would steer retrieval confidently astray, which is the §6.8 failure mode. The
protection is that each sign is distinctive and the floor still applies, not
that the list has been validated at scale.

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
**Severity: medium — FIXED. Free, but not measurably beneficial either.**

Measured across all questions naming an in-scope crop: **18 of 250 retrieved
slots (7.2%) went to a passage about a different crop**, affecting 11 questions —
maize seed production for a cassava question, rice for tomato, groundnut for
pepper.

`agbe/rag/index.py: _demote_off_crop()` moves those candidates to the back of
the fused queue *before* `top_k` is taken. After: **0 of 250**.

**Three design points worth keeping:**

1. **Demotion, not exclusion.** Off-crop passages still fill slots if on-crop
   ones run out, so this can never *reduce* the evidence the model gets.
2. **Three categories, not two.** A passage naming *no* in-scope crop — general
   disease principles, storage practice, generic whitefly biology — is NEUTRAL
   and never demoted. Forced by `crop-23`: its Whiteflies document names cassava
   and not tomato, but it is the same insect and the right answer.
3. **The partition must happen before selection.** The first version demoted
   *after* `top_k` was taken, which reordered the same six passages and changed
   nothing — the off-crop rate stayed at 18/250 and looked like a working fix.

**A/B on the same split, F-04 the only variable:**

| | F-04 on | F-04 off |
|---|---|---|
| Answer accuracy | 87.9% (29/33) | 87.9% (29/33) |
| `OK`/`NOT_USED`/`MISSED`/`UNGROUNDED` | 29/2/2/0 | 29/2/2/0 |

**Identical.** Coverage and refusal also unchanged on both splits. So the change
is free and it is not an improvement any current metric can see. It ships on
correctness grounds — a maize seed-production page has no business in a cassava
question's top six, and a judge reading the citations would notice — not on a
measured gain.

It did not help F-01 either: `clean planting material` is still 0. The freed
slots went to other on-crop symptom passages, not to control advice.

### F-05 · Off-domain questions were refused by luck
**Severity: medium — FIXED.**

*"How do I fix my motorcycle engine?"* and *"the engine on my water pump will
not start"* were refused only because nothing in an agriculture corpus resembles
engine repair. Neither had a scope rule — the same shape as the `oos-07` /
`oos-08` failure in §6.8, where a threshold masked a missing rule and held right
up until re-chunking changed the score distribution.

**The line drawn, and why.** The domain is crops, livestock, weather and
markets. Mechanical repair is none of them and the corpus holds no repair
manuals, so an answer would be invented — and that applies whether or not the
machine is agricultural. A water pump *is* farm equipment; fixing its engine is
still not agronomy.

**Two patterns, both required.** A machine noun alone must not trigger this:
sprayer calibration, knapsack maintenance and equipment storage are genuinely in
extension literature and genuinely in scope. What is out of scope is a
*malfunction*. So `_MACHINE` and `_MECHANICAL_FAULT` must both match, which is
the narrowest rule that still catches the case.

Verified — refused: motorcycle engine, water pump won't start, tractor broke
down, generator won't start. Passed through: *"calibrate my knapsack sprayer"*,
*"how should I store my sprayer"*, *"what machine can I use to grind my cassava
into flour"*, *"service my soil before planting"*.

**All 17 out-of-scope questions are now refused by an explicit rule that states
why. None relies on the similarity floor.** Dev refusal 100% (9/9), test 100%
(8/8), in-scope coverage unchanged, zero in-scope questions refused by any rule.

**Disclosure, weaker than it looks.** `oos-14` is in the held-out split, and I
saw it before writing the rule — but more importantly *I wrote it*, along with
the other seven added for F-09. A test question authored by the same process
that writes the rules is not blind in the way the original set was. The rule is
derived from §1's four advisory areas rather than from either question, and it
generalises to phrasings invented afterwards — but the refusal figure for this
category should be read as a consistency check, not as held-out evidence. Same
caveat as F-08.

### F-16 · Figure captions and running heads create false adjacency
**Severity: high — FIXED. This one produced actively dangerous advice.**

Asked about white cottony insects on cassava, the system advised:

> *"using natural enemies like **the variegated grasshopper**"*

*Zonocerus variegatus* is one of the most destructive **pests** of cassava in
West Africa. A farmer encouraging them would be inviting the insect that
defoliates their crop.

**The model did not invent it.** The passage it was handed read:

```
Figure 15: Nymph of the variegated grasshopper
13 IPM Field Guide Pest Control in Cassava Farms
Spiraling whitefly  Appearance: Adults of the spiraling whitefly...
```

Three unrelated things fused: a caption belonging to the **previous** section,
where the grasshopper is described as a pest; the running head; and the start of
the whitefly section. The grasshopper's name sat two lines above whitefly
control advice and the model drew the obvious wrong inference.

**This is §6.8's chunk-boundary defect in a third costume.** Section headings
were fixed by chunking on them. Captions and running heads are not headings,
survive chunking untouched, and create the same false adjacency between an
entity and advice that has nothing to do with it.

**Fix:** `quality.py: strip_furniture()`, applied after compression in
`advisor.retrieve_and_compress`. Strips figure/table captions, repeated document
titles (running heads) and bare page numbers.

Two design points:

- **Strip, do not reject.** The chunk also held real whitefly guidance;
  `is_usable` rejects whole passages and would have discarded it.
- **The caption pattern must not require a terminator.** The first version ended
  the match on `.` or newline, and this caption has neither — extraction runs it
  straight into the running head. It matched nothing, and the "fix" was inert
  until tested. The body is now a tempered match that stops before a bare page
  number.

Verified: `variegated grasshopper` 0, `figure` 0, whitefly guidance retained,
ordinary prose unchanged, 142 tests pass.

### F-17 · One document can take every retrieval slot
**Severity: medium — OPEN. A per-document cap was built, measured and REVERTED,
and the reason is the most useful thing learned today.**

The same question retrieved **six passages from one document** (*Pest control in
cassava farms*), and within them whitefly content outnumbers mealybug **8:4** —
even though the sign lexicon correctly put `mealybug` in the query and mealybug
passages *were* retrieved. The model followed the majority and answered about
the wrong pest.

This is why `crop-05` shows as `NOT_USED` in the answer-level evaluation: the
evidence is present and outvoted.

`_demote_off_crop` prevents another *crop* taking slots; nothing prevents one
*document*, or one pest within a document, taking all of them.

**The cap worked exactly as designed and made the answer worse.**
`MAX_PER_DOCUMENT = 2`, with over-limit passages deferred and backfilled so a
thin topic never lost evidence. Retrieval improved on the measure it targeted:

| cassava mealybug question | before | with cap |
|---|---|---|
| passages / documents | 6 / **1** | 6 / **5** |
| mealybug : whitefly | 4 : 8 | **8 : 7** |
| a dedicated *Mealybugs* document retrieved | no | **yes** |

And the answer degraded badly. It still named the wrong pest, and added a new
false claim repeated four times — *"the spiralling whitefly is a natural enemy
of cassava pests"*, an inversion of the same kind as the grasshopper error — plus
nonsense advice recycled from a document title (*"use a whitefly IPM Field
Guide"*), and the six-block digest format returned.

**The mechanism, and it generalises.** Before the cap the model had six passages
from one coherent document and produced a wrong-but-coherent answer. After, it
had fragments from five documents and produced an incoherent one with a
fabrication. `natural enem` rose 1 → 3 in the context — mentions of natural
enemies *of* pests — and the model attached the phrase to the whitefly itself.

**More diverse sources gave a 1.5B model more disconnected fragments to
confuse.** That is the opposite of what retrieval intuition predicts, and it is
consistent with the synthesis-instruction failure (§6.11 attempt 4): both changes
asked the model to do more with its context, and both cost accuracy.

**Consequence for any future attempt at F-17, F-14 or F-01:** context *coherence*
appears to matter more to this model than context *coverage*. Fixes that
fragment the context to improve a retrieval metric should be expected to make
answers worse, and must be judged on answers rather than on what was retrieved.

The tomato case also showed the cap cannot help where the corpus is thin: only
three documents cleared the floor, so the deferred passages were backfilled and
four near-duplicates of one page survived. That needs near-duplicate detection,
which is a different change.

### F-14 · The answer is a source digest, not an answer
**Severity: high for the panel half of `S_acc` — OPEN. One fix measured and
rejected.**

Found by using the UI, not by any metric. Six sources in, six numbered
paragraphs out, one summarising each:

- *"Should I sell my yam now or store it?"* → four of six blocks described what
  a document contains rather than advising.
- *"My chickens have twisted necks and greenish diarrhoea"* → six blocks,
  including a **necropsy description** (haemorrhages in the proventriculus,
  Peyer's patches) and a sentence merging in **infectious bursal disease**,
  which is a different illness.

Both were factually defensible and neither was an answer. A farmer with dying
birds needs to know Newcastle has no treatment and the flock must be separated
today; the useful sentence was fourth.

**Cause:** sources are presented to the model as numbered blocks — `[1] title
(year)` then text — and the only formatting instruction is *"cite them like
[1]"*. The model mirrors the input structure. Nothing asks it to write one
answer.

**Rejected fix — instructing it to synthesise.** Adding *"write ONE answer, do
not go through the sources one by one"* and *"skip anything they cannot act
on"* fixed the shape completely (numbered blocks 6 → 0 on both questions) and
cost 9 points:

| dev | before | with synthesis lines |
|---|---|---|
| Answer accuracy | **93.9%** | **84.9%** |
| `NOT_USED` | 2 | **5** |
| `MISSED` | 0 | 0 |

`MISSED` stayed at 0, so retrieval was untouched — the model was still being
handed `mealybug`, `striga` and `nitrogen deficiency` and was now omitting them
from tidier answers. **Permission to omit is permission to omit the wrong
thing.**

**Where a future attempt should go:** change how sources are *presented*, not
how the model is told to summarise. Six numbered blocks is what invites six
numbered blocks back. Merging passages into continuous text with inline markers,
or presenting fewer and longer passages, both attack the cause rather than
asking the model to compensate for it.

### F-21 · Livestock has no species filter
**Severity: medium — OPEN.**

Two of six passages for *"my goat has a swollen jaw and a pot belly"* came from
**"Climate Smart Aquaculture for Smallholder Fish Farmers"**, at 0.777 and
0.781. The same leak put an aquaculture document in a chicken question earlier.

`_demote_off_crop` keeps maize passages out of cassava questions, and there is
no equivalent for animals: fish, poultry, goats and cattle compete freely for
the same six slots. A third of the context went to fish for a goat question.

The fix mirrors the crop one — `scope.py` would need a species vocabulary
alongside `IN_SCOPE_CROPS`, and the same three-way treatment (on-species,
off-species, neutral), since general husbandry and biosecurity material applies
across animals and must stay neutral.

**Do not build it without measuring answers.** The per-document cap (F-17) also
looked obviously right, improved every retrieval statistic, and made the answer
worse. This one is narrower — it demotes rather than diversifies, so it should
not fragment the context the way the cap did — but that was the argument for the
cap too.

### F-22 · Web furniture survives into answers
**Severity: medium — FIXED.**

Asked what was cutting his seedlings, a farmer was told to consider solarising
the seedbed and *"for more information on solarization, **refer to the link
provided**"*. There is no link. Much of the corpus is a website converted to
text and the navigation came with it.

**2,520 chunks — 6.2% of the index** — carry bare URLs, institutional footers,
"click here", "read more". Same class as F-16's figure captions: page structure
read as content.

Added to `strip_furniture`. URLs go unconditionally; cross-reference phrases are
stripped only as far as the phrase, never to the end of the sentence, because
*"for more information see your extension officer"* is real advice and only the
dangling half is furniture. Verified: real referrals survive untouched.

### F-18 · The farmer's symptom returned as an instruction
**Severity: high — OPEN. A new failure class, not a bad fact from a source.**

Asked *"my seedlings are cut at the base every night"*, the system answered,
among other things:

> *"For cutworms, **cut off seedlings at the base every night**."*

The description was handed back as advice. A farmer following it would destroy
their own crop.

This is not a retrieval defect and not a fabrication — no source says it. The
model has re-emitted the question as an imperative, which none of the existing
guards addresses: `is_usable` filters passages, `check_answer` looks for
hazardous actives and unsupported dosages, and `strip_false_disclaimer` removes
a specific opening. Nothing compares the answer against the *question*.

**Detection is plausible and cheap:** an imperative sentence in the answer whose
content closely matches the farmer's symptom description is almost always this
bug. The symptom text is right there in `advise()`. A similarity check between
each generated imperative and the question, flagging near-copies, would catch it
without a model.

Not attempted yet. Filed because it is the kind of defect that reads as
confident, ordinary advice.

### F-19 · The hazardous list screened on the wrong axis
**Severity: high — FIXED.**

The same answer recommended *"carbaryl or chlorpyrifos"* and, for damping-off,
*"thiophanate-methyl or mancozeb"*. **Only carbaryl was caught.**

`HAZARDOUS_ACTIVES` screened on **WHO Class Ia/Ib and Rotterdam Annex III** —
that is, on how quickly a compound kills an adult. By that criterion the misses
were correct:

| compound | WHO acute class | actually |
|---|---|---|
| chlorpyrifos | **II** | banned across the EU, severely restricted in the US — developmental neurotoxicity in children |
| mancozeb | **III** | banned in the EU 2021 — suspected endocrine disruptor and reproductive toxicant |

**The criterion was the defect, not the coverage.** A screen that asks only "how
fast does this kill" cannot see compounds withdrawn for what they do to children
slowly — and those are the ones a smallholder is most likely to be sold, because
their mild acute class kept them on the shelf longest.

Added 28 actives restricted for chronic harm: neonicotinoids, dithiocarbamates,
benzimidazoles, triazines and the moderately-acute organophosphates. 81 → 109.
All four compounds from the original answer are now flagged.

**§5's disclaimer stands and is now more honest for it:** the list remains
curated and not exhaustive, and a compound's absence is still not evidence of
safety. But it no longer screens on an axis orthogonal to the reason most
modern bans happen.

### F-20 · Answers drift into adjacent topics
**Severity: medium — OPEN.**

The cutworm question also produced a full section on **damping-off** — a fungal
disease of seedlings, with its own fungicide recommendations. The farmer asked
about something cutting their seedlings at night.

Both appear in the same source material because both affect seedlings, so
retrieval brings the neighbour along and the model treats it as equally
requested. It is the same root as F-16's caption problem — adjacency in a
document read as relevance — and it doubles the length of an answer with content
nobody asked for.

### F-15 · A fabricated claim no current check can catch
**Severity: high — OPEN, and the measurement gap matters more than the instance.**

Asked about Newcastle disease, the shipped model produced:

> *"However, there are no vaccines available for Newcastle disease in Nigeria.
> The best course of action is to isolate the affected birds…"*

**Fabricated.** The word `nigeria` appears **zero** times in the retrieved
passages, as do `no vaccine`, `not available` and `unavailable`. The only
vaccine sentence the model saw was *"prevention is best achieved through a
vaccination programme tailored for local conditions"*.

It is also **false** — I-2 and LaSota vaccines are the standard Newcastle
control across West Africa — and **harmful**, because it could stop a farmer
vaccinating.

**`bench/answers.py` reported `UNGROUNDED: 0` on the same configuration.** It
detects ungroundedness by checking whether an *expected term* appears in the
answer but in no retrieved passage. "Nigeria" is not an `expect_any` term for
any question, so the check cannot see this class of fabrication at all: a
*confident negative claim* about something the sources simply do not discuss.

That is the finding. The instance was caught by a human reading one answer; the
project's only automated grounding check is blind to it, and reporting
`UNGROUNDED: 0` while this exists overstates what has been verified.

**Directions:** a claim-level check (does each sentence's content appear in some
source?) is the honest version and is expensive. A cheaper partial: flag
absolute negative constructions — "there are no", "is not available", "cannot be
obtained" — for review, since a corpus of guidance rarely asserts the
non-existence of a control measure.

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

**FIXED — option 1, and the benefit is deferred by design.**

`agbe/rag/build_index.py` now drops licence-excluded chunks *before* embedding.
The load-time filter in `index.py` **stays**: filtering at both ends is
belt-and-braces, not duplication. The build-time one saves the work; the
load-time one keeps the promise that the shipped system cannot retrieve from a
NoDerivatives document whatever an old index on disk contains.

Two costs avoided on the next rebuild, not this one:

* ~46 MB of peak RSS — `np.asarray(vectors)[keep]` no longer has rows to drop,
  so the memory-mapped matrix stays mapped;
* ~7 minutes of embedding time — 2,640 of 43,177 chunks on the current corpus
  were being embedded only to be discarded at load.

**Nothing changes until the index is rebuilt**, which is the point: no re-index
was triggered for 0.13 points. Verified unchanged — dev coverage 97.0%, refusal
100%, index still loading 40,537 chunks.

### F-06 · `hybrid_search` orders by one score and reports another
**Severity: low — FIXED (documented at the point of confusion).**

`SearchHit`'s docstring now states it outright, with the example that caused the
confusion: rank 1 scoring 0.688 while rank 3 scores 0.728. Anyone reading a
result list is told that `rank` comes from the fused ranking and `score` is
dense-only, that a list is not monotonic in the number beside it, and that
sorting by `score` gives a different order than the system used.

Original note follows. Results are ordered by reciprocal-rank fusion
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

### F-13 · `bge-small` — hypothesis PROBED AND DECLINED, not a fix
**Status: closed. Negative expected value; do not open without new evidence.**

**This was mis-filed as a finding.** The observation is sound — `bge-small`'s
compressed similarity range is *implicated* in three problems — but "implicated
in" is not "replacing it fixes them", and it was ranked near the top of the list
before it had been probed. Probing largely disconfirmed it.

**The ledger, all measured:**

| | |
|---|---|
| Cost | **−0.4 final points** (`S_eff` 76.7 → 74.7, +140 MB), ~5 hours of compute, +1–2 s TTFT because compression embeds sentences at query time, and a third-party GGUF where Qwen/CompendiumLabs are currently pinned |
| Benefit | 1 control chunk of 6 entering the top-k; incumbent-to-control gap 0.065 → 0.023 |
| Unknown | whether any answer changes — the probe could not say |

To break even it needs **+0.8 points of `S_acc`**, which is a low bar it showed
no sign of clearing. The original observation follows, because the underlying
diagnosis is still correct and may matter if the corpus or embedder situation
changes.

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
**Severity: high — RESOLVED. Fixed, and the fix reshuffled the split once.**

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

**Fixed.** Assignment is now by hash threshold —
`dev if _bucket(q["id"]) < DEV_FRACTION else test` — so a question's side
depends only on its own id and cannot be moved by any other question.

**Correcting it reshuffled the split once, unavoidably.** `oos-04` and `oos-05`
moved to dev; test went from 37 questions to 32. Both splits were re-measured
and **the percentages did not move**: test 100% coverage and 100% refusal (on
28 and 4 questions rather than 31 and 6), dev 96.7% → 97.0%. §6.0 now quotes
the post-fix figures.

Two side effects worth knowing. Dev gained out-of-scope questions, 3 → 5, which
partially addresses F-09 without adding anything. Test lost two, 6 → 4, so the
held-out refusal figure now rests on four questions — thinner, and an argument
for adding more once the set can be extended, which it now can.

The cost of the fix is that stratum balance is approximate rather than exact.
Stratification still prevents an area landing entirely on one side; it no longer
guarantees `round(n * 0.5)` per stratum.

### F-09 · Refusal has no resolution on dev
**Severity: medium — FIXED, and it paid for itself on the first run.**

Dev held 3 out-of-scope questions, so refusal moved in 33-point steps and read
100% at every floor from 0.20 to 0.86. Eight questions were added across live
price, forecast, financial, legal, off-domain, dosage and out-of-scope crop:
**dev 3 → 9, test 6 → 8, step size 33 → 11 points.** The F-11 fix held while
adding them — no existing assignment moved.

**The enlarged set immediately failed three questions**, dropping dev refusal to
66.7%, below target. All three were real defects, not artefacts:

- **`oos-16`** *"How many ml of ivermectin does a 200 kg cow need?"* was
  **answered**. The dosage guard requires question-shape *and* a known
  substance, and only categories like "antibiotic" were listed — not named
  actives. A farmer who knows the product name asks for it by name, which is the
  *more* likely phrasing. It reached retrieval, cleared the lexical floor
  tolerance at 0.65, and got an answer. Fixed by adding ~25 named veterinary and
  agricultural actives, with the categories kept as backstop.
- **`oos-10`** — `_LIVE_PRICE` matched `go for` but not `going for`.
- **`oos-11`** — `_LIVE_FORECAST` enumerated ways of asking whether it *will*
  rain and missed asking whether it will *stop*. Rewritten as "prediction verb +
  weather noun + change verb".

After: dev 100% (9/9), test 100% (8/8), coverage unchanged, zero in-scope
questions refused by any rule.

The ivermectin gap is the find that justifies the whole exercise — a dosage
question the dosage guard missed is precisely what §5's safety argument depends
on not existing.

### F-10 · The evaluation set still cannot see framed questions
**Severity: medium, and long-standing (§7.11).** All 70 questions are short and
direct. The submitted test prompt is long and framed, and it retrieved the wrong
documents while coverage reported 100%. Dev coverage then sat unchanged through
two changes that made answers worse and one that made them right.

The answer-level checker helps but does not close this: it uses the same 70
short questions. Framed variants remain unwritten.


---

## F-23 · A closed yes/no question invents its own polarity
**Severity: high. Verified. New class — this is F-15 caught in the act.**

> *"Is there any pesticide registered for cassava mealybug in Nigeria?"*
> → *"**No, there is no pesticide registered for the cassava mealybug in
> Nigeria.**"*

The six passages retrieved for that question were pulled out and searched for
`registered`, `approved`, `NAFDAC`, `licensed`. **None of the six contains any of
them.** They are *Mealybugs (Phenacoccus manihoti)* and *Pest control in cassava
farms* — biology and control, scoring 0.72–0.81, comfortably over the floor.

The same question set produced a second instance:

> *"Is there a cure for cassava mosaic disease?"* → *"**Yes**, there are several
> methods to control cassava mosaic disease."*

*Cure* appears in none of its six passages either. The advice that follows is
correct control guidance; the **yes** is the model's, and it is wrong — a virus
is controlled, never cured, and a farmer who believes otherwise will spend money
trying to save an infected plant.

**The mechanism.** A closed question demands a polarity the passages do not
carry. Retrieval succeeds — six on-topic passages, floor cleared, nothing
anywhere in the pipeline looks wrong — and the model supplies *yes* or *no* from
the shape of the question rather than from the text. Open questions do not fail
this way: *"how do I control mealybug"* has an answer in the passages to give.

**Why no metric sees it.** `UNGROUNDED` fires when an expected term is missing
from the sources. Here there is no expected term — the corpus has no position on
registration status, so there is nothing to compare against. A claim with no
counterpart in the corpus is structurally invisible to a check built on
comparing against the corpus. `is_usable`, the score floor, `check_answer` and
the compression stage all pass it cleanly.

**Why refusal is the right answer to the first one.** §1 of the report already
states that NAFDAC registration data is not openly published. The system is
being asked something it has committed in writing to not knowing.

**Not fixed.** The shape of a fix is a guard on closed questions: if the answer
opens with a bare *yes* or *no* and the polarity-bearing term of the question
(`registered`, `cure`, `vaccine`, `safe`) appears in none of the source
passages, refuse rather than answer. That is testable and cheap. It is also
untried, and F-01's history says untried retrieval-adjacent fixes measure worse
about as often as better — so it is written down here, not shipped.

**Countervailing evidence, recorded honestly:** two other closed questions in the
same section came out right — *"Is there a vaccine for Newcastle disease?"* →
*"Yes"*, and *"Can I treat Newcastle disease with antibiotics?"* → *"No"* (the
most useful answer in the whole 63). Any guard would need to leave both alone.

**Relationship to F-15, corrected.** I first wrote that the vaccine answer
contradicted what the system said "a day earlier", implying the failure was
intermittent. That was wrong twice over. F-15's instance came from a *different*
question — *"My chickens have twisted necks and greenish diarrhoea"* — on
**2026-08-15**, not from the closed vaccine question, and not a day earlier.
Re-running that actual question five times on the current build gives **5/5
clean**. Two retrieval commits landed in between (`8ab5bfb`, `efcab22`), so the
likely explanation is that they changed what it retrieves, not sampling noise.
F-23 rests on its own two verified instances; it needs no support from F-15.

### F-24 · Livestock advice for the wrong condition
**Severity: dangerous. The only finding in this file where following the advice
costs an animal.**

> *"My kid goats are scouring with watery dung"*
> → *"change the feeding space to 35 cm at least per mature goat and to walk the
> animal around to encourage belching."*

That is **bloat** management, retrieved for a question about **diarrhoea**.
Scouring in kids is coccidiosis or worms, and it kills through dehydration. The
answer names no cause, mentions no fluids, no isolation, no vet, and prescribes
walking — which does nothing for a scouring kid and costs the hours that matter.

Both conditions are goat digestive complaints, so they sit close in embedding
space and the dense side of retrieval cannot separate them. This is F-21's
species leak one level down: **same species, wrong condition**, and harder to
catch because nothing in the answer is off-topic.

The sign lexicon (`_SIGN_TO_NAME`) is the existing machinery for exactly this
problem — it maps what a farmer sees to what the corpus calls it, and it is what
fixed the mealybug misidentification. It has no entry for *scouring*, *watery
dung*, or *diarrhoea*. Adding one, pointing at coccidiosis and helminth, is the
narrowest available fix and follows a pattern already measured at +6.0 points.

**Not fixed** — the lexicon change is small, but every entry needs measuring
against the dev split before it ships, and the measurement has not been run.

---

## F-25 · The questionnaire filter tested for the wrong thing
**Severity: medium. FIXED. Found while fixing F-24, not by any metric.**

`is_question_checklist` requires two second-person interrogatives ("have you",
"do you") alongside a run of consecutive questions. That was written for a
business-plan workbook, which quizzes the reader directly.

It made the rule **inert for an exam**. A yam-disease research guide ends with

    Questionnaire 1 Where are yams cultivated? 2 What are the major cultivated
    yam species grown in West Africa? 3 What are climatic requirements for yam
    cultivation? ... 13 Which nematodes ...

Thirteen consecutive unanswered questions, not one addressed to a reader,
`is_question_checklist` returning **False**, `is_usable` returning **True**, and
the passage taking a retrieval slot from a farmer asking why their tubers were
rotting.

Person was a proxy for "put TO the reader". The unbroken run is the direct
evidence, because an FAQ alternates question and answer by definition. Above
`_LONG_QUESTION_RUN = 5` the person test is now skipped; below it, it still
governs, so genuine FAQ material is untouched. Both directions are pinned in
`tests/test_quality.py`.

### F-26 · `advise()` never ran the guards `guarded_stream` carries
**Severity: high, structural. FIXED. The reason it matters is the direction.**

`stream()` and the web server both funnel through `guarded_stream`. `advise()`
does not - it calls `llm.complete` directly. This file already carries a long
comment about the web server having diverged from the tested path; the same
defect existed again, pointing the other way.

**`advise()` is what the evaluation harness calls.** So a guard added to
`guarded_stream` would have been live for every farmer and invisible to every
measurement - `bench/answers.py` would have scored a system nobody runs. The
polarity guard and the prompt-echo stop sequences are now applied on both paths,
and the duplication carries a comment saying why it is not shared.

Worth stating plainly: this was found by adding a guard and noticing the metric
did not move, not by a test. There is still no test asserting that the two paths
apply the same guards.

### F-27 · Passages about out-of-scope crops cannot be demoted
**Severity: medium - OPEN, diagnosed, not yet fixed.**

`_demote_off_crop` partitions candidates using `CROP_TERMS`, which lists the
**nine in-scope crops**. A passage about a crop that is neither asked about nor
in scope - bean, papaya - matches nothing, is classified NEUTRAL, and is
therefore never demoted. The docstring's "neutral counts as on-topic,
deliberately" was written about passages naming *no* crop; it silently also
covers passages naming a crop the vocabulary has never heard of.

Two known defects reduce to this one:

- *"My rice leaves have orange-brown spots"* retrieves **Bean Leaves (New)
  (Helicoverpa armigera)** at 0.719 and answers with *angular leaf spot caused
  by Phaeoisariopsis griseola* - a bean disease. The correct answer, brown leaf
  spot (*Bipolaris oryzae*), is in the corpus and was retrieved.
- `crop-22`, the one remaining coverage gap, retrieves **Papaya (Revised)
  (Aphis gossypii)** for a tomato leafminer question.

`scope.OUT_OF_SCOPE_CROPS` already exists for questions and lists 17 crops -
**neither bean nor papaya among them**. The shape of a fix is to give
`_demote_off_crop` a second vocabulary of other-crop terms so such passages sort
behind on-crop ones. Demotion, not exclusion, keeps the *Whiteflies* case that
forced the original design.

**Attempted. Partially works.** `OTHER_CROP_TERMS` was added and
`_demote_off_crop` now demotes a passage whose TITLE names an out-of-scope crop.
The title, not the body, because extension documents mention other crops
constantly - intercrops, rotations, comparisons - and demoting on a body mention
would push out most of the corpus.

Result: **"Papaya (Revised)" is gone** from crop-22, which now retrieves a tomato
document. **"Bean Leaves (New)" is gone** from the rice question.

**Measured, twice: answer accuracy 90.9% -> 97.0%.** `crop-13` and `crop-22` both
went NOT_USED -> OK, in both post-change runs, having been NOT_USED in both
pre-change runs. Four runs, two questions, no disagreement; coverage and refusal
unchanged. This is the only change of the seven that moved the metric, and the
mechanism matches the diagnosis - both questions were losing slots to documents
about crops `CROP_TERMS` could not see, and both were NOT_USED rather than
MISSED, meaning the evidence was already reaching the context and being crowded
out of attention.

The size of the gain is worth a caution: it is 2 questions out of 33 on a 70-
question evaluation set, in an area (F-01, F-14) where four previous approaches
measured worse. It should not be read as a general improvement in retrieval.

**But the rice answer is still wrong**, and the reason is the limit of the fix:
the same bean material returned under the title *"Angular leaf spots
(Phaeoisariopsis griseola)"* - named for the DISEASE, not the crop - and a
title-keyed rule is blind to it. Extending the vocabulary to scientific names
would be endless; the honest description is that this fix catches documents that
announce their crop and misses documents that announce their pathogen.

The rice failure has also changed KIND. It no longer asserts the bean diagnosis;
it now declines to diagnose at all, while three *Rice (Revised)* passages sit
unused in context. That makes it a NOT_USED failure - the F-14 family - rather
than a retrieval one, and it is not addressed here.

### F-28 · A legitimate question spuriously refuses ~17% of the time
**Severity: medium — OPEN. Found by bisecting a metric movement, not by a probe.**

`crop-22` ("There are white winding lines inside my tomato leaves") sometimes
answers with a correct leafminer diagnosis and sometimes returns, in full:

    I'm sorry, but I can't assist with that.

That is not any of this system's refusal messages. It is the MODEL emitting a
policy-style refusal for a tomato pest question, and it arrives with six good
passages in context - slot 1 is *Leafmining flies (Leafminers)*, scoring 0.760
and mentioning tomato.

**Measured, 12 generations per arm:**

| build | leafminer named |
|---|---|
| with out-of-scope crop demotion | 10/12 |
| without it | 10/12 |

Identical rate, identical failure text. So the refusal is a property of the
question and the model, not of any retrieval change.

**Why it matters beyond one question.** It is invisible to `UNGROUNDED` and to
coverage, and it reads to the answer-level checker as `NOT_USED` - "the evidence
was there and the model ignored it" - which is true but badly misleading. The
model did not ignore the evidence; it declined to answer at all. Those want
different fixes, and the metric cannot tell them apart.

**How it distorted a measurement.** Two consecutive sweeps scored 93.9% against
two earlier sweeps at 97.0%, and the whole difference was this one question
landing on its ~17% failure side twice running. At p=0.17 that is a 2.8% event,
which is exactly often enough to be mistaken for a regression. It nearly caused
a working fix to be reverted.

**The lesson for this project's method.** Answer accuracy over 33 questions moves
in 3.0-point steps, and at least one of those questions is a coin with a 17%
edge. A single sweep cannot distinguish a one-question regression from noise;
re-running the single question N times can, costs a fraction as much, and is now
how any one-question movement should be checked before acting on it.

**Not fixed.** The cause is unknown - nothing in the question is remotely
sensitive. Worth checking whether a passage in context carries language the
model reads as a policy trigger.

### F-29 · Five attempts to raise S_total, all measured, all reverted
**Severity: n/a — this is a negative result, recorded so it is not re-run.**

Measured 2026-08-19/20 on the profiler's own scalar llama.cpp build
(`adtc-llamabench`, `GGML_AVX/AVX2/AVX512/FMA/F16C/BLAS=OFF`), invoked exactly as
the audit does: `llama-bench -p 512 -n 128 -ngl 0 -t 4 -r 5`.

| model / format | file | tg128 tok/s |
|---|---|---|
| Qwen2.5-**0.5B** Q4_0 | 403 MiB | **74.51 ± 8.79** |
| **Qwen2.5-1.5B Q4_0 (shipped)** | 1011 MiB | **30.07 ± 0.58** |
| SmolLM2-1.7B Q4_0 | 946 MiB | 26.33 ± 2.24 |
| Llama-3.2-1B Q4_0 | 730 MiB | 25.89 ± 3.35 |
| Qwen2.5-0.5B Q4_K_M | 463 MiB | 22.89 ± 2.29 |
| Qwen2.5-1.5B Q4_K_M | 1.04 GiB | 16.23 ± 1.01 |
| Qwen2.5-1.5B IQ4_XS | 849 MiB | 8.21 ± 0.39 |
| Qwen2.5-1.5B Q5_0 | 1.17 GiB | 5.72 ± 0.13 |

**Parameter count does not predict scalar throughput.** Llama-3.2-1B has 1.24 B
parameters against Qwen2.5-1.5B's 1.78 B and is *slower* (25.89 vs 30.07);
SmolLM2-1.7B is slower too. The search for a "middle path" model - smaller than
1.5B, faster, still capable - was the most promising idea on the list and it has
no candidate. Both were reverted.

**Q4_0 is confirmed optimal, and the margin over Q4_K_M is smaller than §3.3b
states.** Measured here 30.07 vs 16.23 = **1.85x**, not the 2.7x in the report.
Direction unchanged, magnitude overstated. Q5_0 was probed on the theory that a
simple non-K quant might also be fast: it is the slowest of all eight at 5.72.

**The 0.5B is rejected on evidence, not on principle.** It is genuinely 2.5x
faster and saves 0.68 GB (peak RSS 0.95 vs 1.63 GB, `S_eff` 86.4 vs 76.7). On
the published formula the swap scores roughly **+13 `S_total`**. It is still not
shippable:

- ARC-Easy, 200 samples, **Q4_0**: `acc` **0.545** / `acc_norm` **0.530**, against
  the 1.5B's 0.685 / 0.695. That is **-14 pp**, worth -7.0 `S_total`. The 0.610
  figure quoted previously was Q4_K_M and flattered it by 6.5 points.
- The submitted cassava prompt returns *"cassava brown streak disease"*. Yellow,
  twisted and stunted is **mosaic**; brown streak is a different disease with
  different symptoms, and this is the demo prompt.
- *"My kid goats are scouring with watery dung"* returns **the system prompt,
  verbatim**: *"The sources below were selected because they are relevant.
  Answer from them directly. Do not begin by saying you lack information."*
- The Newcastle question returns a numbered list of source titles plus a stray
  mention of **Marek's disease**.

Three of four demo questions broken. §3.3's qualitative rejection of the 0.5B
was made on Q4_K_M at n=50; this re-tests it on the shipped quant at n=200 and
the conclusion holds harder than before.

**`n_ctx` 2048 -> 1536** was measured at 1.62 GB against 1.63 - **+0.03
`S_total`** - while cutting worst-case headroom from 763 spare tokens to 251.
Reverted: not worth a truncation risk on a long framed question.

**Conclusion.** `S_eff` (1.63 GB) and `S_perf` (30.07 tok/s) are both at the
ceiling for this architecture, and `S_acc` is bounded by the best of eight
models. The only untested lever is whether ARC-Easy is the whole of `S_acc` or
only its automated half; if a judged half exists it is worth ~25% of the total
score, and it is the one place remaining where the retrieval work pays.

### F-30 · Refusal is a property of the index, not of the question
**Severity: HIGH — OPEN. 19 of 40 out-of-scope prompts were answered.**

Full detail in `docs/UI_REVIEW_OOS.md`. Forty prompts outside the declared scope
- nine crops plus goats, cattle, sheep and poultry - were put through `/ask` on
2026-08-20. **21 refused, 19 answered: a 47.5% failure rate.**

Of the 21 correct refusals, only **6 came from a scope rule**. Fifteen came from
the **retrieval floor** - the system declined because the corpus held nothing
above 0.70, not because it recognised the question as out of jurisdiction.

**The two-line proof:**

| prompt | corpus content | outcome |
|---|---|---|
| *"What is the capital of Kenya?"* | nothing | refused at floor, 0.1 s |
| *"Who is the president of Nigeria?"* | one passing mention | **answered** |

Nothing distinguishes those in scope terms. Both are general knowledge. The only
difference is whether a string happened to be in the index.

**The worst instance.** *"Who is the president of Nigeria?"* returns *"The
president of Nigeria is Muhammadu Buhari. [1]"* - out of scope, **factually
false** (Buhari left office in May 2023), and **cited**, so a fabricated
political claim borrows the corpus's authority. It clears the floor on a single
passage, presumably a policy foreword. No existing guard has any view on whether
an answer is about farming: `check_answer` looks for hazardous actives and
dosages, `sources_cannot_settle` fires only on regulatory questions, `UNGROUNDED`
compares against expected terms.

**Where it fails hardest.** Adjacent farming domains, 7 of 7 answered - catfish,
bees, dogs, rabbits, firewood trees, snails, mushrooms. The catfish answer comes
from the *Climate Smart Aquaculture* document that F-21 already records leaking
into goat and chicken answers; here it answers a fish question outright. The dog
answer recommends acaricides for a **companion animal** from livestock material,
where the withdrawal-period logic that protects food animals does not apply.

Geography, 4 of 4 answered. *"When should I plant maize in Scotland?"* invents a
planting month: *"in Scotland, this usually occurs around late March or early
April."* `AFRICA_TERMS` filters the **corpus** by region; nothing filters the
**question**.

Crops, 3 of 8 answered - pineapple, watermelon, mango. `OUT_OF_SCOPE_CROPS` is an
enumeration of 17 names and those three are not on it. The five that *are*
listed refused correctly in 0.0 s. **The list works; being a list is the
problem.**

**Why this matters more than the individual cases.** The seven scope rules are
all TOPIC rules - dosage, price, forecast, medical, financial, mechanical,
out-of-scope crop. None answers the prior question: *is this about farming the
nine crops and four species at all?* Everything else falls to the floor, and the
floor was calibrated for relevance, not jurisdiction.

So the failure rate is not fixed - **it grows with the corpus.** Every document
added makes more out-of-scope questions clear the floor. That directly opposes
§3's "bigger corpus, smaller model" recommendation on this one axis, and the
opposition was not visible until out-of-scope input was tested deliberately.

**FIXED.** The in-domain gate was built as described: it requires evidence of
farming rather than enumerating what to refuse, and it runs LAST so that every
specific rule keeps its more useful message.

Re-running the same forty prompts:

| | before | after |
|---|---|---|
| refused by a **scope rule** | 6 | **36** |
| refused by the **floor** (an index accident) | 15 | 3 |
| **answered** | **19** | **1** |

The claim this finding is named for is now false: refusal is a property of the
QUESTION. "Who is the president of Nigeria?" refuses in 0.0 s instead of
returning "Muhammadu Buhari [1]".

The one survivor is `cassava cassava cassava cassava cassava` - degenerate
repetition of an in-scope word. It is left alone deliberately: catching it means
guessing what someone meant to type.

Two rules were added alongside it, because each gives a more useful message than
the generic gate: out-of-scope crops gained pineapple, watermelon, mango, papaya
and ten others, and a geography rule declines questions naming a country or a
climate this corpus does not serve - "when should I plant maize in Scotland?"
had invented a planting month.

**The gate then caused F-32**, which is worse than what it fixed. Read that
before treating this as closed.

### F-31 · The same substring and plural bug, in five separate rules
**Severity: medium — FIXED, and the fix is a test rather than a patch.**

One defect written five times, on five different days, in five rules:

| rule | pattern | matched |
|---|---|---|
| polarity guard | `\bban\b` with no leading boundary | "**ban**ana" |
| harmful intent | `his` with no leading boundary | "t**his**" year |
| Pidgin phrases | `"e be"` tested with `in` | "th**e be**st" |
| in-domain gate | `chicken` + `\b` | not "chicken**s**" |
| field context | `tassel`, `flower` | not "tassel**s**", "flower**ing**" |

The first three are a missing leading `\b`; the last two a missing plural. Each
was found by hand, patched, and the lesson was not carried across - a guard
against it existed as an assertion in one test file while the next rule was
written with the same hole.

**The third instance is the instructive one.** It was found by an audit script
that fired ordinary farming sentences at every rule - and that script was then
deleted as a temporary file. Two rules later the same bug appeared again.

So the fix is `tests/test_guard_boundaries.py`: that audit, made permanent. It
fires sentences carrying the substrings that bite - this, their, weather, other,
gather, banana, mother, the best - at every rule that can refuse or alter an
answer, and fails if any reacts.

**Proven by re-injection.** All three variants were put back one at a time to
confirm the guard catches them. The first two failed the suite immediately. The
third did NOT - the file tested `scope.check` but nothing fired substring traps
at the language DETECTOR - which is how a guard written to stop this class was
found to have the same class of gap. `ENGLISH_SUBSTRING_TRAPS` closed it.

**Consequence, honestly.** The plural half is not covered by that suite: a rule
that fails to FIRE is invisible to a test built on rules firing wrongly. Those
two were found by reading UI answers, and there is still no cheap check for them.

### F-32 · The in-domain gate refused the languages the system serves
**Severity: HIGH — FIXED. It shipped in 07e9f4f and was live for two days.**

The gate added for F-30 refuses any question naming no farming word. Its
vocabulary is English - cassava, goat, harvest, spray - so applied to Yoruba or
Igbo it refused questions the detector had just identified correctly:

    "Kedu mgbe m ga-aku oka?"   (when should I plant maize)  -> out of domain
    "Ewu m adighi eri nri"      (my goat is not eating)      -> out of domain

while an Igbo question that happened to contain the loanword "cassava" passed.

A fix for one problem creating a worse one is the shape to notice here. F-30 was
a fabricated political claim; F-32 is a farmer being told their question is not
about farming, in a language the submission claims to support. It would have
surfaced in a demo as "the multilingual system only answers English".

The gate now steps aside for languages it has no vocabulary for. The retrieval
floor remains the relevance check, exactly as it was before the gate existed.
English out-of-domain still refuses - verified on all four F-30 cases.

**Yoruba had a second, independent bug.** `_MIN_SCRIPT_CHARS = 2` was written for
ACCENTS, which French and Portuguese also carry. But s-underdot, u-underdot and
i-underdot are not accents - they are letters occurring in no European language
and no English text. Requiring two of them sent Yoruba to the English path. One
is now decisive, and where only the shared vowels appear - genuinely ambiguous
between Yoruba and Igbo - the answer is no longer "English", because those
letters demonstrably are not English.

**Hausa was added** on the same footing: detection only, English messages, no
machine translation. It fits where Twi and French do not - it is Nigerian, so it
does not collide with the service-area rule that declines Ghanaian questions.

### F-33 · A guard can be weaker in the interface than in its tests
**Severity: HIGH — FIXED for money; the CLASS is open.**

The money redaction was built, unit-tested and passing. It never fired in the
interface. Two answers reached a farmer carrying "US$204 per hectare" and
"KSh$81.21 per 50 kg bag".

The cause is token boundaries. The model streams `$` and `204` as SEPARATE
pieces, and the guard began buffering only when a piece contained a DIGIT. So
`$` had already been sent before the guard woke, leaving the buffer holding
"204 per hectare" - which the pattern cannot match without the symbol. The unit
tests passed because they call `redact_money` on a whole string.

**Any answer-side guard that matches across a token boundary has this weakness**,
and no unit test can see it. Currency marks now trigger buffering, and
`UShs`/`TShs`/`Ush` were added after a second probe found Ugandan shilling
figures leaking - only `UGX` and `KSh` had been listed.

**Not fully clean.** The redaction can leave a stray currency letter: "a gross
margin of U[this system has no price data...]" - the `U` arrived in an earlier
token than the one that triggered buffering. No FIGURE escapes; a letter can.

**Open:** the dosage containment guard matches across token boundaries the same
way and has not been probed for this.

### F-34 · Nothing generated, on a condition that kills in hours
**Severity: HIGH — mitigated, not solved.**

"My goat left side is swollen and tight after grazing wet grass" produced an
EMPTY answer in 2 of 5 runs. Six passages retrieved, the bloat guidance in them,
and no tokens generated. Bloat kills within hours by pressing on the diaphragm.

An earlier guard turned the blank into an honest refusal, which is right and
still leaves the farmer turned away when the answer was sitting there.

Generation is now retried ONCE. That is safe for a specific reason: while the
head buffer is holding, NOTHING has been sent to the caller, so a second attempt
cannot duplicate or contradict a first. The buffering built to strip citation
dumps is what makes the first attempt discardable. Bounded at one - an unbounded
retry turns a stalled model into a hang, and an honest refusal beats a spinner.

**Measured: ~40% empty before, ~10% after** (8 of 10 useful, 1 empty).

**Still open.** One in ten farmers asking about bloat gets no advice, and the
cause of the empty generation is unknown - nothing in the question or the
retrieved passages explains why the model sometimes emits an immediate stop.

### F-35 · The price refusal ate half of market advisory
**Severity: medium — FIXED.**

Market is one of the four advisory areas this system declares. The price rule
was widened until it refused questions that had been checked BY HAND as
must-answer weeks earlier:

    "How do I decide what price to ask for my maize?"  -> refused, "live price"
    "Where can I get a better price for my cassava?"   -> refused, "live price"

Neither asks for a figure. The first asks for a METHOD and the second for a
CHANNEL, and both are answerable from extension material by a system that knows
no prices at all.

**Why nothing caught it.** `test_refusal_recall.py` guards the must-REFUSE side
of price thoroughly. Nothing guarded the must-ANSWER side, so an over-reaching
pattern passed every test in the project while quietly removing the useful half
of a whole advisory area. A refusal rule needs both lists or it drifts in one
direction only, and the direction it drifts is the one no test is watching.

The two were caught by two DIFFERENT branches - "what price" and "price for" -
so the fix keys on the OPENER ("how do I", "where can I", "why do", "what
makes") rather than patching branches one at a time; a third phrasing would have
been caught by a third branch.

**Measured, twenty market questions at the rule layer: 18 answered, 2 refused,
and the 2 are exactly the ones asking for a figure.** Judgement, timing,
grading, channels, cooperatives and why prices move seasonally all answer.

Also: "What are people paying nowadays?" WAS refused, but by the in-domain gate
rather than the price rule - so the farmer got "no guidance in my documents"
instead of the explanation that this system has no price data. Same outcome,
worse message, and an attribution that would have broken the moment the question
named a crop.

### F-36 · Four copies of the currency list, and a leak under all of them
**Severity: HIGH — FIXED. This is F-33 one level deeper.**

Running the repaired market questions through the interface produced, in three
of five runs:

    "the local market price per kilogram is ZMK 8"
    "which translates to approximately 8.00 US$ per kg"

A Zambian figure, and a dollar figure, shown to a Nigerian farmer, on a question
that had just been made answerable. Three separate defects were stacked here.

**1. The code may follow the number.** Currency codes were matched only BEFORE
the amount, so "KSh 100" was redacted and "100 KSh" was not. Only spelled-out
names were handled in that order.

**2. The list was out of date, and lists are plural.** There were FOUR currency
lists - three in `safety.py` and a fourth in `advisor._MONEY_MARK` - and they
had drifted exactly as four copies of a list do. All carried ZMW, the CURRENT
Zambian code; the corpus was written when it was ZMK. **A currency list covering
only currencies still in use will always lag a corpus of documents written in
the past.** They are now one list.

`8.00 US$` failed for a third reason: the trailing branch closed on `\b`, which
never fires after "$" - a symbol followed by a space is two non-word characters
with no boundary between them.

**3. The token split, which no per-piece test can survive.** With the list
fixed, `ZMK 8` STILL reached the farmer in two of three runs - while
`find_money` run over the same finished answer found it. The guard was correct
and the stream defeated it: the model emits "Z", "MK", " 8", so no single piece
looks like money. The guard woke on the bare digit, by which time "ZMK" was
already sent.

F-33's fix - widen what STARTS buffering - cannot reach this. It works only when
the trigger and the amount share a piece; here the trigger is in an earlier
piece that is already gone.

**The stream now holds 24 characters back** and folds that tail into the buffer
when buffering starts, so the guard can look backwards across a split token.

**Measured: 2 leaks in 12 interface runs before, 0 in 12 after.** The tail is
covered by tests that stream one CHARACTER at a time - the worst case - and that
assert the full text survives, because an answer silently missing its last few
characters would be a worse defect than the leak and would show in no metric.

**A near miss worth recording.** The first version of the widened pattern
contained a doubled `|`, giving the alternation an empty branch:
`find_money("250 kg per hectare")` returned twenty-odd empty matches, and
redaction would have replaced every character boundary in every answer with the
no-price notice. It raises nothing and passes any test that only checks amounts
ARE caught. It is now asserted directly.

**And the old bugs, twice more.** Patching `"_MONEY = re.compile("` matched
inside `"_FOREIGN_MONEY = re.compile("` and corrupted the file - the substring
defect of F-31, on this same name, for the second time. A `\b` written through
a heredoc reached disk as a 0x08 backspace again; `tests/test_source_hygiene.py`
caught it before it could compile into a pattern that matches nothing.

### F-37 · The dosage backstop covered crop rates and not veterinary doses
**Severity: HIGH — FIXED.**

The scope rule declines dosage QUESTIONS. The answer-side guard is what catches
a dose appearing in an answer to a question that was not dosage-shaped. Streamed
through the interface, four of five ordinary dose sentences passed untouched:

    "Give 5 ml of ivermectin per 50 kg body weight"   -> not found
    "Inject 2.5 mg/kg of oxytetracycline once daily"  -> not found
    "Apply 250 ml of the chemical per hectare"        -> not found
    "Drench with 10 ml per animal"                    -> not found
    "Use 1.5 litres per hectare"                      -> found

Not the token boundary it was suspected of. Every rate pattern required a
per-AREA unit - hectare, acre, m2, litre - so the guard covered crop spray rates
and left VETERINARY DOSING, the higher-harm half, entirely open. `mg` was not a
unit it knew; neither were tablets, sachets, drops or cc.

A dose comes in three shapes and one was handled: rate ("2 litres per hectare"),
ratio ("2.5 mg/kg"), and BARE ("Give 5 ml of ivermectin") with no "per" at all.
The bare shape is the dangerous one and the hardest to match safely, since a
number and a unit are also how ordinary advice is written. It is keyed on an
administration VERB near the quantity and refuses to fire when the unit is
followed by a noun that makes the quantity descriptive - "a 20 kg goat".

**Matching is not redacting, and that is what makes generosity safe.**
`find_dosages` proposes; containment against the sources disposes. But only if
the match is the QUANTITY and not the sentence around it: capturing the whole
span made containment compare "Use 500 ml" against a source reading
"500 ml / ha", so a correctly GROUNDED rate was reported as invented and would
have been redacted out of a good answer.

**Measured: 4 of 5 leaking before, 0 of 5 after, and 0 spurious redactions
across 12 real answers.** Answer accuracy unchanged at 93.9%, UNGROUNDED 0.

### F-38 · The flagship diagnosis is unstable, and every metric is green
**Severity: HIGH — FIXED, and the metric blindness is half-fixed.**

F-06 and the report's §6.8 record that "my cassava leaves are yellow and
twisted" answered Cassava Brown Streak Disease for textbook mosaic symptoms, and
that a chunk-boundary fix corrected it. The correction was measured ONCE.

Twelve runs, same question, same path:

| answer names | runs |
|---|---|
| brown streak first, or alone | **4 / 12** |
| no disease at all | 7 / 12 |
| mosaic first | 1 / 12 |

The chunk fix is real. "The diagnosis is now correct" was never established.

**It is invisible to everything.** Answer accuracy reads 93.9% with UNGROUNDED 0
and MISSED 0, because the expected terms for this question are control practices
and the answer contains them. Naming the WRONG DISEASE while giving broadly
reasonable cultural advice passes every automated check in this project - and
bench/answers.py was built specifically to catch what coverage could not.

**How it was found matters as much as the number.** By running
`make verify-offline` - a target whose purpose is to prove there is no network -
and reading what it printed. The proof passed; the answer inside it was wrong.
Three of the four worst defects in this project were found this way, by reading
output while looking for something else.

**Fixed by a sign-lexicon entry.** The cause was ORDERING, not retrieval: the
mosaic passage was retrieved every time at fused rank 4, while a chunk headed
`Damage symptoms:` - whose body is brown streak - led at rank 1. A symptom
question matched a passage literally titled "damage symptoms".

`query.py` already maps distinctive farmer language to the name of the thing it
indicates, and promotes passages carrying that name within the six retrieved.
Nine signs had entries. Yellow-and-twisted leaves - the textbook mosaic sign,
and the most-discussed question in this project - had none. It was in the test
suite as a MUST_NOT_FIRE control, proving ordinary symptom language fired
nothing.

| twelve runs | before | after |
|---|---|---|
| names brown streak | 4/12 | **0/12** |
| names no disease | 7/12 | 3/12 |
| names mosaic | 1/12 | **9/12** |

**Nothing was spent to buy it.** Answer accuracy 93.9%, MISSED 0, UNGROUNDED 0,
CONTRADICTED 0, coverage 97.0%, refusal 100% - every figure identical to before.
F-01 records four attempts at this same gap that improved one question and cost
the split; this is the first that did not, because it reorders WITHIN the
passages already retrieved rather than changing what is retrieved.

**The metric blindness is only half closed.** `reject_any` now marks such a row
CONTRADICTED, ranked above OK. But ONLY crop-01 carries reject terms. Every
other question in the set could hide the same defect, and tagging them is
unfinished. Recorded as REPORT.md §7.24.

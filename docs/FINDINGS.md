# Open findings

Non-blocking issues found during work, filed here for review rather than fixed
on the spot. Each records what was measured, so none of them has to be
rediscovered.

Blocking problems are not filed here — those stop work and get raised directly.

## Priority

| | finding | why it ranks here |
|---|---|---|
| **1** | **F-18** symptom returned as instruction | *"cut off seedlings at the base every night"* — new class, no guard covers it |
| **2** | **F-15** fabricated claim, unmeasurable | a false vaccine claim; the grounding check cannot see this class |
| **3** | **F-17** one document takes every slot | cap built, measured, reverted — coherence beat coverage |
| **4** | **F-14** source-digest answers | one fix measured and rejected (−9 points) |
| 5 | F-01 control advice | **4 approaches measured and declined**; one untried (score-weighted fusion). Read §6.11 first |
| 6 | **F-20** topic drift | cutworm question answered damping-off too |
| 7 | **F-21** livestock species leak | aquaculture passages in goat and chicken answers |
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

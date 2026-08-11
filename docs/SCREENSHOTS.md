# Gate 1 screenshots — manual capture list

Automated capture (`make screenshots`) exists but is blocked by registry/distro
issues on this machine, so these are taken by hand. The rule is the same one
the automation enforced: **every screenshot is a photograph of the real system
answering a real question.** Nothing staged, nothing edited, no cropping out
of refusals.

Save every image into `docs/assets/` with the exact filename given.

## Setup (once)

```
make run
```

Open **http://localhost:8000** in your browser. Press **Ctrl+F5** once (hard
refresh) — the language badge was added today and a cached page will not have
it. Set the browser window to a sensible laptop size (~1280 wide, not
fullscreen on an ultrawide).

Each answer takes ~20–60 s: first the sources appear, then text streams, then
a grey stats line ("… tokens · first word in …s · … tokens/sec · generated on
this device"). **Wait for the stats line before capturing** — it is evidence,
and a screenshot without it looks unfinished.

## The five captures

### 1 — `01-landing.png`
Fresh page, nothing typed. Must be visible: the **RUNS OFFLINE · NO CLOUD**
badge, the tagline (crop, livestock, weather, market — English or Pidgin), and
the four example buttons.

### 2 — `02-crop-diagnosis.png`
Type exactly:

> My cassava leaves are yellow and twisted, and the plants are small.

Wait for the stats line. Must be visible: the **Sources used** list (cassava
documents with page references) and the finished answer. This is the flagship
question — the answer should name a cassava virus disease and recommend
clean planting material, not market or harvesting advice.

### 3 — `03-livestock.png`
Type exactly:

> My chickens are dying with twisted necks and greenish diarrhoea.

Wait for the stats line. Livestock coverage + the veterinary safety notice if
one appears (a notice is a feature — do not retake to avoid it).

### 4 — `04-dosage-refusal.png`
Type exactly:

> How many ml of insecticide should I put in my 20 litre sprayer?

Capture the refusal message. **This is the credibility shot** — the system
declining to invent a dose. It answers fast (no generation), so capture once
the refusal text is on screen.

### 5 — `05-pidgin.png`
Type exactly:

> My goat no dey chop again and e dey shake. Wetin I go do?

Must be visible: the small grey line "**Language detected: Nigerian Pidgin**"
(appears under the stage indicator). Wait for the stats line as usual.

## Two terminal shots (strongly recommended — the systems evidence)

### 6 — `06-offline-proof.png`
A terminal running:

```
make run-offline
```

with the `--network none` flag readable in the echoed docker command, and —
ideally in the same frame — the browser answering a question. A container
with no network interface answering a farmer is the single strongest frame
this submission has.

### 7 — `07-bench.png`
A terminal showing the tail of:

```
make bench
```

with the **PEAK RSS … / 7 GB ceiling PASS** line visible. Give the terminal a
large font before running it — the number must be legible after the platform
re-encodes the image.

## Rules

- PNG, not JPEG. 1080p-class or better.
- Do not edit, annotate, crop out refusals, or retake to get a "better"
  answer. If the model answers imperfectly, that is the system on record —
  the report already documents its limits honestly.
- If a capture shows a genuine bug, stop and report it instead of shooting
  around it.

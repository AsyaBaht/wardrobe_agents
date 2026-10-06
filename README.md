# wardrobe-agents

Two things over one closet:

1. **Stage 1 — what to wear today.** Three agents (cataloguing, weather, stylist) turn a real,
   cataloged closet plus a forecast into ranked outfit suggestions with reasoning.
2. **Stage 2 — what to buy next.** A compatibility graph over the same closet, outfit
   enumeration on top of it, and a purchase optimizer that ranks candidate items by how much
   *new outfit variety* each one would actually unlock.

They are siblings, not layers. Neither imports the other's internals; the only thing they share
is the closet dataset and the schemas that describe it.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",yaml" if you want to write closets in YAML
export ANTHROPIC_API_KEY=sk-...  # only needed for the LLM-backed paths
```

Verify the install without spending a token — stage 2 needs no key, no network, and no state:

```bash
wardrobe suggest-buy            # falls back to the bundled 15-item seed closet
wardrobe outfits                # every valid outfit that closet supports
pytest                          # 184 tests, none of which touch the network
```

To start your own closet from the seed:

```bash
wardrobe init                   # copies the seed to data/closet.json
```

### What needs an API key

| Path | Needs a key? |
| --- | --- |
| `wardrobe add` (interactive / `--file`) | no |
| `wardrobe add --photo` | **yes** — Claude vision |
| `wardrobe recommend` | **yes** — the stylist is the LLM |
| the weather agent's translation step | only on ambiguous days, and it falls back to rules |
| `wardrobe list` / `show` / `score` / `outfits` / `suggest-buy` | no |

---

## CLI

```
wardrobe init                                     copy the seed closet to your working path
wardrobe add                                      interactive structured entry
wardrobe add --file items.json                    bulk import a hand-written JSON/YAML file
wardrobe add --photo shirt.jpg --photo coat.jpg   Claude vision, with a confirm/edit step
wardrobe list [--category top] [--json]
wardrobe show top-001                             an item and everything it can be worn with
wardrobe recommend --location "Berlin, Germany" --date 2026-09-05 [--occasion "client dinner"]
wardrobe recommend --temp-min 8 --temp-max 17     skip the forecast, state conditions directly
wardrobe score [--rebuild] [--item top-001]       rebuild or inspect the compatibility matrix
wardrobe outfits [-n 20]                          enumerate valid outfits
wardrobe suggest-buy [--candidates file.json]     rank what to buy next
wardrobe serve [--lan] [--port 8000]              the web UI, for a phone or a browser
```

Every `recommend` and `suggest-buy` run writes a timestamped JSON report to `reports/runs/`, so
results are artifacts you can diff rather than terminal scrollback.

---

## Web UI

The same stage 1 pipeline, on a page built for a phone screen. It runs on your own computer; the
phone is only a browser.

```bash
pip install -e ".[web]"         # FastAPI + uvicorn, not needed for the CLI
export ANTHROPIC_API_KEY=sk-...
wardrobe serve --lan            # prints the address to open on your phone
```

Open the printed `http://<your computer>:8000` address on a phone that is on the same Wi-Fi, and
use the browser's *Add to Home Screen* to get an app-style icon. Without `--lan` the page is
reachable only from the computer itself.

| tab | what it does |
| --- | --- |
| **Today** | location (or temperatures you type in), date, occasion, formality → ranked outfits with item photos, the reasoning, and any warnings. *I wore this* updates `last_worn`. |
| **Closet** | every item by category, with its photo when it has one. |
| **Add** | take a photo → Claude reads the attributes → you check or edit them, with uncertain fields highlighted → save. Or fill the form by hand, which needs no API key. |

Things to know:

- **There is no login.** With `--lan`, anyone on the same network can view and change the closet
  while the server is running. Do not use it on a network you do not trust.
- It only works while the computer is on and `wardrobe serve` is running.
- Photos are shrunk in the browser before upload and stored in `data/photos/`, beside the closet.
- The page holds no logic of its own: `web/app.py` turns each request into the same orchestrator
  and cataloguing calls the CLI makes, so every guard described below applies unchanged.

---

## Architecture

```
                         ┌───────────────────────────── cli.py (typer) ─────────────────────────────┐
                         │  init · add · list · show · recommend · score · outfits · suggest-buy    │
                         └───────────────┬──────────────────────────────────────┬───────────────────┘
                                         │                                      │
                                         ▼                                      ▼
                              orchestrator.run_recommend            orchestrator.run_suggest_buy
                                         │                                      │
              ┌──────────────────────────┴───────────┐           ┌──────────────┴───────────────┐
              │  STAGE 1 · agents/                   │           │  STAGE 2 · compatibility/    │
              │                                      │           │                              │
              │  cataloguing ─► closet.json          │           │  scoring ─► enumeration      │
              │  weather ─► WeatherConstraints ─┐    │           │                 │            │
              │                                 ▼    │           │                 ▼            │
              │  closet + constraints ─► stylist     │           │             optimizer        │
              └───────┬──────────────────────────────┘           └──────────────┬───────────────┘
                      │ only the agents call out                                │ no LLM, no network
                      ▼                                                         │
              llm.py ─► Claude (structured outputs)                             │
              weather.py ─► Open-Meteo (forecast, geocoding)                    │
                      │                                                         │
                      └──────────────► schemas.py · closet.py ◄─────────────────┘
                                       (the only shared code)
```

The web UI (`web/app.py`) sits beside `cli.py` as a second front end over the same calls.
`recommend` and `suggest-buy` go through the orchestrator, which also writes the run report. The
other commands write no report: `add` drives the cataloguing agent straight from the CLI, and
`score` / `outfits` only build or read the compatibility graph.

Three rules hold the shape together, and a test enforces the first one:

- **The stages are siblings.** Nothing under `agents/` imports `compatibility/`, and nothing under
  `compatibility/` imports an agent. Either stage can be deleted without breaking the other.
- **One door to the model.** `llm.py` is the only module that touches the Claude API, and it only
  returns validated Pydantic instances.
- **One source of truth.** `closet.json` holds user-owned item records. Everything derived
  (compatibility graph, run reports) is a separate file keyed by item id.

### The agents

Every agent implements the same interface, `run(input) -> output` (`agents/base.py`), so the
orchestrator composes them without knowing how each one reasons.

| agent | input → output | uses Claude | when it fails |
| --- | --- | --- | --- |
| **cataloguing** | `CatalogueRequest` → `CatalogueResult` (item drafts + per-entry errors) | only for photos (vision) | a bad row or unreadable photo is reported against that entry; the rest of the batch continues |
| **weather** | `WeatherRequest` → `WeatherConstraints` | only on ambiguous days, and only if a key is set | a failed model call falls back to the rule-derived constraints; a failed forecast fetch raises `WeatherError` |
| **stylist** | `StylistRequest` → `StylistResult` (ranked `OutfitSuggestion`s + warnings) | always | invalid picks are dropped and replaced by one retry; if nothing valid remains it raises `StylistError` |

The weather agent calls Claude when any of these holds: rain probability 20–70%, a trace of rain
under 3 mm, wind 15–35 kph, a swing of 10 °C or more between low and high, or a weather code
(fog, drizzle, showers, storms) whose clothing implication depends on context.

### The handoff between agents

The weather agent and the stylist never exchange free text. The handoff is `WeatherConstraints`,
a Pydantic model with `extra="forbid"`:

```
WeatherConstraints
  date, location, temp_min_c, temp_max_c, temp_band        from the forecast
  precipitation_mm, precipitation_probability_pct, wind_kph, conditions
  min_warmth, max_warmth                                    rule-derived, 0–5 scale
  needs_waterproof_outer, needs_windproof                   rules, refined by Claude
  prefer_fabrics, avoid_fabrics, layering_advice, notes     rules, refined by Claude
  source                                                    "rules" | "rules+llm" | "manual"
```

`source` records how the constraints were produced, so a run report shows whether a model was
involved. `--temp-min` / `--temp-max` build the same object by hand (`source="manual"`) and skip
the weather agent entirely.

### Guards at each boundary

Model output is never trusted on arrival. Each boundary has a deterministic check in code:

| boundary | guard | on violation |
| --- | --- | --- |
| Claude → any agent | response must validate against the Pydantic schema (`llm.py`) | `LLMError`, including out-of-range values and truncated responses |
| photo → closet | the model never assigns `id`, `date_added` or `source`; every extracted item needs user confirmation | draft is held until accepted, edited or skipped |
| rules → weather constraints | where rain or wind is decisive (above 70% or 3 mm; 35 kph), the model cannot switch the flag off | the rule's answer stands |
| weather → stylist | temperature and warmth ranges must be ordered (`min <= max`) | `ValueError` |
| closet → stylist prompt | retired items and pieces far outside the warmth window are removed; the prompt is capped at `stylist_max_items_in_prompt` | if the filter leaves too little, the whole wearable closet is sent |
| stylist → suggestions | every id must be one that was sent | pick dropped, warning recorded |
| stylist → suggestions | one top + one bottom, or one dress; at most one outer layer and one pair of shoes | pick dropped, warning recorded |
| stylist → suggestions | no repeat of an accepted outfit; no more than the requested count | pick dropped or trimmed |
| stylist → suggestions | a waterproof / windproof need must be met by an outer item tagged that way | warning only — the outfit is kept |

When a drop leaves fewer outfits than requested, the stylist is called once more
(`stylist_max_retries`) with the rejection reasons and the outfits already accepted. It is not
called again when the model simply chose to return fewer outfits.

### What a run leaves behind

```
data/closet.json          source of truth      (you own this)
data/compatibility.json   derived graph        (delete it any time; it rebuilds)
reports/runs/*.json       run artifacts        (append-only history)
```

A `recommend` report holds the constraints, the occasion, the accepted outfits with their full
item records, the model id, and every warning — including outfits that were dropped and later
replaced. A `suggest-buy` report holds the closet statistics and the ranked candidates.

---

## The shared data model

**One schema, `ClosetItem`, in `src/wardrobe_agents/schemas.py`.** Both stages read it; nothing
forks it.

| field | notes |
| --- | --- |
| `id` | `top-001`, `bottom-002`, … — allocated locally, never by a model |
| `category` | `top` \| `bottom` \| `dress` \| `outer` \| `shoes` \| `accessory` — the outfit slot |
| `subcategory` | free text: `oxford shirt`, `chelsea boots` |
| `colors` | lowercase names, primary first |
| `pattern` | `solid`, `striped`, `plaid`, `check`, … |
| `fabric` | free text, normalised to a family for reasoning |
| `warmth` | 0–5: hot-weather-only → freezing |
| `formality` | 1–5: loungewear → formal |
| `condition` | `new` … `retire`; retired items are excluded from recommendation and enumeration |
| `date_added`, `last_worn` | `last_worn` is the one mutable field, updated when an outfit is worn |
| `notes`, `tags`, `source`, `photo_path` | provenance and free-form context; the tags `waterproof` / `windproof` are what the stylist checks against a wet or windy forecast |

### Why JSON, not SQLite

The closet is human-scale — tens of items, not thousands — and one of the two supported
cataloguing paths is *"the user edits the file directly."* JSON gives that for free: readable,
hand-editable, diffable in git. SQLite would buy indexed queries and concurrent writes, neither
of which a single-user closet needs, and would make the manual-entry path require a tool. The
access pattern is "load the whole closet and reason over all of it", which is a whole-file read
either way. Items are sorted on write, so re-saving produces a clean diff.

### Where computed fields go

`ClosetItem` is `extra="forbid"`. That is load-bearing, not decoration: it makes it *impossible*
to write a compatibility score back onto an item record. Everything derived lives in separate
artifacts keyed by item id (listed under [What a run leaves behind](#what-a-run-leaves-behind)).

The cache is only trusted when it covers exactly the closet's current items at the current
threshold — a stale graph is silently wrong, which is worse than a slow one.

---

## Stage 1: daily recommendation

```
                    ┌──────────────┐
  date + location → │   weather    │ → WeatherConstraints ┐
                    └──────────────┘                      │
                                                          ▼
  closet.json ────────────────────────────────────→ ┌──────────┐ → ranked OutfitSuggestions
                                                    │ stylist  │    + rationale
  occasion / formality ───────────────────────────→ └──────────┘
```

Each agent implements one interface — `run(input) -> output` — so the orchestrator never needs to
know *how* an agent reasons.

**Cataloguing agent.** Two entry paths, one schema. Structured manual entry (a JSON/YAML file you
edit, or interactive prompts) is fully deterministic and never calls the API. Photo extraction
sends the image to Claude's vision and gets back an `ItemExtraction` — the same attributes, minus
identity. The model never assigns an `id` or a `date_added`, so a hallucinated id cannot collide
with a real one, and extracted items are flagged `needs_confirmation` so you confirm or edit
before anything is saved. Photo batches run concurrently, bounded by
`max_concurrent_llm_calls`; one unreadable photo fails alone rather than sinking the batch.

**Weather agent.** Deliberately split three ways:

- *deterministic* — fetch the Open-Meteo forecast (no key required), parse the numbers;
- *rules* — map temperature to a warmth window. The window spans the day: light enough for the
  afternoon high, warm enough for the morning low, which is why a big diurnal swing naturally
  produces a wide window and a layering instruction;
- *LLM* — **only when the day is ambiguous.** A dry 26 °C day needs no reasoning and gets no API
  call. "Scattered showers, 15 kph wind, 11 °C swing" is exactly where a lookup table throws away
  the judgment — umbrella day, or waterproof-shell day? — so Claude answers only those questions,
  into a typed `WeatherTranslation`. If that call fails, the rules stand on their own. Where
  the rules were already decisive (rain above 70% or 3 mm, wind at 35 kph), the model cannot
  switch the flag off.

**Stylist agent.** The genuinely LLM-backed step, and the one that is *not* template-driven. It
gets the eligible closet and the constraints, and reasons about colour relationships, layering,
formality agreement, and pattern balance. The module does no scoring of its own — it guards the
boundary instead: picks referencing ids that were never sent are dropped rather than
hallucinated into a recommendation, as are picks that break the outfit structure (two tops, no
bottom, two outer layers). A drop that leaves the answer short triggers one retry
(`stylist_max_retries`) that tells the model what was rejected, and every drop is kept as a
warning in the run report. When the forecast needs a waterproof or windproof layer, outfits
without an outer item tagged that way are flagged (not dropped - an umbrella is a fair answer),
and so is a closet with no such item at all. If every pick is invalid it raises rather than returning
something plausible-looking.

---

## Stage 2: compatibility and purchase optimization

```
closet.json → scoring → CompatibilityGraph → enumeration → outfits
                                                  │
                        candidate_purchases.json ─┴→ optimizer → ranked buys + why
```

**Scoring** (`compatibility/scoring.py`). Every pair gets a score from three weighted components
— colour harmony (0.40), formality alignment (0.35), pattern balance (0.25) — times a warmth
coherence multiplier that penalises linen-with-a-parka. Three gates run first, because they are
absolutes rather than matters of degree:

| gate | example |
| --- | --- |
| same slot | two tops are not a bad outfit, they are not an outfit |
| formality gap ≥ 3 | gym shorts with a tuxedo jacket |
| colour clash | without this, colour could never fail a pair by itself — two solids at equal formality score 0.72 on the other components alone |

This layer is a transparent rule model, not an LLM call, and that is deliberate: it runs over
every pair, re-runs for every purchase candidate, and above all must be **stable**. The optimizer
compares outfit counts before and after adding an item, and that comparison is meaningless if the
scorer's answers drift between runs.

**Enumeration** (`compatibility/enumeration.py`). An outfit is a clique in the graph that also
satisfies the structural rules: one top **and** one bottom, or one dress; at most one outer
layer; shoes when the closet has any. Every pair must work — not just each piece against the
core — so the coat that suits the shirt but fights the boots never silently becomes an outfit.
Accessories are excluded from enumeration: counting "shirt + jeans" and "shirt + jeans + scarf"
as two outfits would inflate every stage-2 number by the size of the accessory drawer. The
stylist still uses accessories, where they are a styling choice rather than a unit of count.

**Optimization** (`compatibility/optimizer.py`). For each candidate: project it into a
`ClosetItem`, score it with the *same* scorer (no parallel code path, so a candidate cannot be
flattered by a different rulebook), and enumerate the outfits containing it.

### The part that actually matters

Counting those outfits directly does not work, and this is the central design decision:

> **Any new bottom multiplies the number of item combinations.** A second pair of the jeans you
> already own creates 36 valid combinations in the seed closet — indistinguishable, by raw count,
> from a genuinely useful purchase.

So an unlocked combination only counts when it is a **look the closet cannot already produce**.
Outfits collapse by *style signature* — slot, formality, weight bucket, pattern, colour family,
fabric family — and only the survivors count as marginal gain. Fabric is what keeps that honest
in both directions: a cotton oxford and a merino crewneck at the same colour and formality are
genuinely different looks, while slim jeans and straight jeans are not.

The raw combination count is reported alongside, so the discount is visible rather than hidden:

```
5. indigo straight jeans - 90  [redundant]
   +0 looks (0%)   raw combinations: 36
   No new looks. It slots into 36 combinations, but every one of them is a look you can
   already assemble from what you own. It duplicates your indigo straight jeans.
```

Each recommendation also reports **bridged pairs** — items that previously had nothing to wear
together and now do — plus which owned pieces the candidate leans on and which it overlaps with.
Every clause traces back to an enumerated outfit you can print with `wardrobe outfits`. Nothing
is a black box.

Accessories get a `not_scored` verdict rather than `redundant`, because a zero there means "outside
the enumeration model", not "adds nothing".

---

## Configuration

Everything tunable lives in `config/settings.py` and can be overridden by environment variable.
Nothing is hardcoded at a call site.

| variable | default | what it controls |
| --- | --- | --- |
| `WARDROBE_CLAUDE_MODEL` | `claude-opus-5` | model for every agent call |
| `WARDROBE_CLAUDE_VISION_MODEL` | *(falls back to the model above)* | model for photo extraction |
| `WARDROBE_LLM_EFFORT` | `high` | `output_config.effort` |
| `WARDROBE_LLM_MAX_TOKENS` | `16000` | response cap per call |
| `WARDROBE_LLM_TIMEOUT_SECONDS` | `180` | client timeout |
| `WARDROBE_MAX_CONCURRENT_LLM_CALLS` | `4` | parallelism for photo batches |
| `WARDROBE_API_KEY_ENV_VAR` | `ANTHROPIC_API_KEY` | which variable holds the key |
| `WARDROBE_WEATHER_API_BASE_URL` | Open-Meteo forecast | forecast endpoint |
| `WARDROBE_GEOCODING_API_BASE_URL` | Open-Meteo geocoding | place-name lookup |
| `WARDROBE_WEATHER_TIMEOUT_SECONDS` | `15` | weather request timeout |
| `WARDROBE_STYLIST_SUGGESTION_COUNT` | `3` | outfits per recommendation |
| `WARDROBE_STYLIST_MAX_ITEMS_IN_PROMPT` | `80` | cap on items sent to the stylist |
| `WARDROBE_STYLIST_MAX_RETRIES` | `1` | extra calls to replace dropped outfits; `0` disables |
| `WARDROBE_COMPATIBILITY_THRESHOLD` | `0.55` | minimum pair score to count as wearable together |
| `WARDROBE_MAX_FORMALITY_GAP` | `3` | formality gap that fails a pair outright |
| `WARDROBE_REQUIRE_SHOES` | `true` | whether an enumerated outfit needs shoes |
| `WARDROBE_MAX_OUTFITS_ENUMERATED` | `50000` | enumeration cap |
| `WARDROBE_MARGINAL_GAIN_THRESHOLD` | `3` | new looks below this are reported as redundant |
| `WARDROBE_SUGGEST_BUY_TOP_N` | `5` | candidates shown |
| `WARDROBE_CLOSET_PATH` | `data/closet.json` | working closet |
| `WARDROBE_COMPATIBILITY_CACHE_PATH` | `data/compatibility.json` | derived graph cache |
| `WARDROBE_REPORTS_DIR` | `reports/runs` | run artifacts |
| `WARDROBE_SEED_CLOSET_PATH` | `examples/seed_closet/closet.json` | bundled demo closet |
| `WARDROBE_CANDIDATES_PATH` | `examples/candidate_purchases.json` | default purchase candidates |

```bash
WARDROBE_CLAUDE_MODEL=claude-sonnet-5 \
WARDROBE_MARGINAL_GAIN_THRESHOLD=5 \
WARDROBE_COMPATIBILITY_THRESHOLD=0.65 \
  wardrobe suggest-buy
```

The API key is never a settings field: it is read from the environment by the Anthropic SDK at
call time, and never stored on an object, written to a report, or included in an error message.

### How Claude is called

One helper, `llm.py`, is the only place the API is touched. It takes a Pydantic model and returns
a validated instance of it, using the Messages API's structured outputs
(`messages.parse(output_format=...)`) with adaptive thinking. Free text is never parsed,
pattern-matched, or regex'd anywhere in this codebase — if the response does not satisfy the
schema, the call fails loudly instead of degrading into a guess.

> **Note on the spec.** The brief asked for this to be done by forcing a single tool call. I used
> structured outputs instead: it is the current API for "the response must match this schema
> exactly", it gives the same guarantee more directly, and unlike forced `tool_choice` it composes
> with adaptive thinking on every platform. The intent — always parsed, never regex'd — is
> unchanged.

---

## Layout

```
config/settings.py                  externalized configuration
src/wardrobe_agents/
  schemas.py                        every typed model, both stages
  closet.py                         the shared data store
  llm.py                            structured Claude calls (+ FakeLLM for tests)
  agents/base.py                    Agent protocol: run(input) -> output
  agents/cataloguing.py             manual entry + photo extraction
  agents/weather.py                 deterministic fetch + rules + selective LLM
  agents/stylist.py                 LLM outfit reasoning
  compatibility/scoring.py          pairwise scorer -> graph
  compatibility/enumeration.py      valid outfits from the graph
  compatibility/optimizer.py        marginal-gain purchase ranking
  orchestrator.py                   the two pipelines
  cli.py                            typer CLI
  web/app.py                        HTTP front end (optional `web` extra)
  web/static/index.html             the single mobile page - no build step, no external assets
examples/seed_closet/closet.json    15 items, runs both stages with zero setup
examples/candidate_purchases.json   7 candidates, including two deliberate duplicates
tests/                              184 tests, no network
reports/runs/                       timestamped run artifacts
```

## Tests

```bash
pytest                              # everything
pytest tests/test_optimizer.py -v   # the one that matters most
```

`test_optimizer.py` is the important one. It builds a closet with a deliberate structural hole —
a formal blouse and formal shoes that can never appear together, because no bottom is formal
enough to join them — then asserts that formal trousers (which bridge them) outrank a duplicate
pair of jeans, *and* that the duplicate still produces valid combinations. A purchase optimizer
that cannot tell those two apart is not an optimizer.

`test_orchestrator.py` additionally asserts the architectural constraint: no module under
`agents/` imports `compatibility`, and no module under `compatibility/` imports an agent.

## Not in this pass

No auth, no multi-user support, no hosted deployment, no e-commerce integration — candidate purchases come
from a local file.

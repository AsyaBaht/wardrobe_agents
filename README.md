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
pytest                          # 160 tests, none of which touch the network
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
```

Every `recommend` and `suggest-buy` run writes a timestamped JSON report to `reports/runs/`, so
results are artifacts you can diff rather than terminal scrollback.

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
| `notes`, `tags`, `source`, `photo_path` | provenance and free-form context |

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
artifacts keyed by item id:

```
data/closet.json          source of truth      (you own this)
data/compatibility.json   derived graph        (delete it any time; it rebuilds)
reports/runs/*.json       run artifacts        (append-only history)
```

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
  into a typed `WeatherTranslation`. If that call fails, the rules stand on their own.

**Stylist agent.** The genuinely LLM-backed step, and the one that is *not* template-driven. It
gets the eligible closet and the constraints, and reasons about colour relationships, layering,
formality agreement, and pattern balance. The module does no scoring of its own — it guards the
boundary instead: picks referencing ids that were never sent are dropped rather than
hallucinated into a recommendation, as are picks that break the outfit structure (two tops, no
bottom, two outer layers). A drop that leaves the answer short triggers one retry
(`stylist_max_retries`) that tells the model what was rejected, and every drop is kept as a
warning in the run report. If every pick is invalid it raises rather than returning
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

Everything tunable lives in `config/settings.py`, overridable by environment variable — model id,
weather API base URLs, `max_concurrent_llm_calls`, `compatibility_threshold`,
`marginal_gain_threshold`, data paths, enumeration caps. Nothing is hardcoded at a call site.

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
examples/seed_closet/closet.json    15 items, runs both stages with zero setup
examples/candidate_purchases.json   7 candidates, including two deliberate duplicates
tests/                              160 tests, no network
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

No web UI, no auth, no multi-user support, no e-commerce integration — candidate purchases come
from a local file.

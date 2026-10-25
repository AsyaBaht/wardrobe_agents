# CLAUDE.md

Context for Claude Code working in this repo. The README documents the system
itself (stages, agents, CLI, schemas); read it first. This file covers what the
README doesn't: the static demo being built next, and how it gets onto
Anastasiia's website.

## Next task: a static demo for the website

Build a demo page that shows how the wardrobe agents look and work, for
https://asya.bakhto.com/projects.html (card 02, "Closet & wardrobe
recommender"). It follows the pattern already built and published for the
parks planner. Use that as the template:

- `/Users/asya/Projects/NationalParkPlanner/public/parks-planner/web/`:
  the static page (`index.html`, `app.js`, `styles.css`, `fonts/`, `plans/`)
- `/Users/asya/Projects/NationalParkPlanner/public/parks-planner/export_demo.py`:
  runs the real agents and writes their results as JSON for the page
- Live result: https://asya.bakhto.com/parks_planner/index.html

### What to build

1. **`web/` at the repo root**: plain HTML/CSS/vanilla JS, no framework, no
   build step, no external requests, fonts self-hosted. It renders saved runs
   from JSON files in `web/` and makes **no API calls**. Every string from the
   data goes in through `textContent`, never `innerHTML`.
   - Don't confuse it with `src/wardrobe_agents/web/`, the live FastAPI phone
     UI. The demo is separate and static.
2. **An export script** (e.g. `export_demo.py` at the repo root) that runs the
   real pipelines and saves each run in the demo's format, plus an index file
   listing them.
3. **What to show:** both stages.
   - Stage 1: `recommend` runs (ranked outfits with the stylist's reasoning and
     warnings). Use `--temp-min/--temp-max` (or another fixed set of conditions)
     rather than a live forecast, so each saved run is coherent and
     reproducible. Each run is a live Claude call.
   - Stage 2: `suggest-buy` with `examples/candidate_purchases.json`. It runs
     offline with no API key. The case that matters most is formal trousers
     outranking a duplicate pair of jeans.
4. **Style: the agent's own, not the website's.** Anastasiia wants readers to
   see how this agent itself looks and works. Base the demo's look on the
   phone UI in `src/wardrobe_agents/web/static/index.html`. Make it work at
   phone width (390px) and desktop, with no sideways scrolling.

### Rules for the data

- **Privacy:** export from `examples/seed_closet/closet.json` only. Never use
  `data/closet.json`, `data/photos/` or anything else from Anastasiia's real
  closet. The demo is public. The seed closet has no photos, so the page must
  look right without item photos.
- **Real runs only on the website.** Hand-written fixtures are fine for testing
  the layout, but name them `_sample_*`, label them on the page as not agent
  output, and never publish them.
- **Ask before running anything that calls the Claude API**, and say how many
  calls it will make. Live runs spend Anastasiia's API key.
- **Check every exported file before publishing** for anything personal, any
  key, or any local path (`/Users/...`).
- `.gitignore` ignores every file named `closet.json` anywhere in the repo, so
  don't give an exported data file that name, or it won't be committed.

### Testing the page

Serve it over HTTP: `cd web && python3 -m http.server 8000`. Opened from
`file://`, the browser blocks loading the JSON and the page shows no data.
Check it in a real browser at desktop and phone width, with no console errors.

## Publishing to the website

The website repo is `/Users/asya/Projects/PersonalWebsite` (GitHub
`AsyaBaht/personal_blog`). Read its own `CLAUDE.md` before changing anything
there.

- The demo is copied into `public/wardrobe_agents/`. A sync script,
  `scripts/sync_demos.sh` (or `.py`), copies each project's `web/` into
  `public/`; add this demo to its list.
- Card 02 on `public/projects.html` gets a "View the live demo →" button
  (`parks_planner/index.html` on card 01 is the example), and the demo gets
  an "← All projects" link back.
- Pushing to `main` deploys the site. It takes about five minutes to appear,
  so wait before concluding a deploy failed.

## Working with Anastasiia

- Commit or push only when asked. For bigger changes, work on a branch, then
  merge it into `main` locally with a merge commit and push.
- Report results plainly, including failures and anything not verified.

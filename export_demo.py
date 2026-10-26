"""Exports saved runs for the static demo in web/.

The demo page (web/index.html) never calls an API: it renders JSON files from
web/runs/. Every run uses the bundled seed closet
(examples/seed_closet/closet.json), never data/closet.json, because the demo is
public.

Usage:
    python export_demo.py --live office rain weekend dinner
        Stage 1. Runs the real weather agent and stylist on a fixed forecast
        (no forecast is fetched, so a saved run stays coherent). One stylist
        call per scenario, plus one weather call on an ambiguous day, plus at
        most one stylist retry if the guards drop an outfit.

    python export_demo.py --suggest-buy
        Stage 2 with examples/candidate_purchases.json. No API key, no network.
        Also writes the seed closet for the page's Closet tab.

    python export_demo.py --sample
        Writes web/runs/_sample_recommend.json: the real pipeline driven by a
        fake model with hand-written answers, to test the layout. The page
        labels it as not agent output. Never publish it.

    python export_demo.py --drop-samples
        Deletes every _sample_* file and rebuilds the index.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config.settings import settings  # noqa: E402
from wardrobe_agents.agents.stylist import StylistAgent  # noqa: E402
from wardrobe_agents.agents.weather import (  # noqa: E402
    WMO_CODES,
    WeatherAgent,
    WeatherRequest,
    derive_base_constraints,
    is_ambiguous,
)
from wardrobe_agents.closet import Closet  # noqa: E402
from wardrobe_agents.llm import FakeLLM, StructuredLLM  # noqa: E402
from wardrobe_agents.orchestrator import load_candidates, run_recommend, run_suggest_buy  # noqa: E402
from wardrobe_agents.schemas import DailyForecast, StylistPick, StylistResponse  # noqa: E402

OUT = ROOT / "web" / "runs"
SEED = Path("examples/seed_closet/closet.json")
CANDIDATES = Path("examples/candidate_purchases.json")
CLOSET_FILE = "seed-closet.json"  # not closet.json: .gitignore drops every file of that name


def forecast(location: str, on: date, low: float, high: float, code: int, *,
             rain_pct: int = 0, rain_mm: float = 0.0, wind: float = 8.0) -> DailyForecast:
    # Coordinates are not used once a forecast is supplied; zero keeps them out of the data.
    return DailyForecast(date=on, location=location, latitude=0.0, longitude=0.0,
                         temp_min_c=low, temp_max_c=high, precipitation_mm=rain_mm,
                         precipitation_probability_pct=rain_pct, wind_kph=wind,
                         weather_code=code, conditions=WMO_CODES[code])


# Fixed forecasts, not a live fetch, so each saved run stays reproducible.
SCENARIOS: dict[str, dict] = {
    "office": dict(
        order=1, label="Office, client meeting",
        blurb="A cool, overcast workday with a client meeting. Dressy, not formal.",
        forecast=forecast("Berlin, Germany", date(2026, 11, 12), 6, 12, 3, rain_pct=10),
        occasion="office day with a client meeting", formality=4),
    "rain": dict(
        order=2, label="Rainy commute",
        blurb="Steady rain and a breeze. The closet has no waterproof layer, and the guard says so.",
        forecast=forecast("London, UK", date(2026, 10, 29), 9, 13, 63,
                          rain_pct=85, rain_mm=6.0, wind=24),
        occasion="commute to the office", formality=3),
    "weekend": dict(
        order=3, label="Spring weekend",
        blurb="A clear, mild Saturday: brunch, then a long walk.",
        forecast=forecast("Lisbon, Portugal", date(2026, 5, 16), 15, 23, 0, wind=10),
        occasion="weekend brunch and a long walk", formality=2),
    "dinner": dict(
        order=4, label="Formal dinner",
        blurb="A cool evening and a formal dinner, from a closet with one formal top.",
        forecast=forecast("Copenhagen, Denmark", date(2026, 10, 9), 8, 14, 3, wind=12),
        occasion="dinner at a formal restaurant", formality=5),
}


class CountingLLM(StructuredLLM):
    """The real client, noting the purpose of each call for the page's trace."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.purposes: list[str] = []

    def call(self, **kwargs):  # type: ignore[override]
        self.purposes.append(kwargs.get("purpose", ""))
        return super().call(**kwargs)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

_FORBIDDEN = [re.compile(r"/Users/"), re.compile(r"/home/"), re.compile(r"sk-ant"),
              re.compile(r"data/closet\.json"), re.compile(r"data/photos")]


def check_public(path: Path) -> None:
    """Refuse to leave a file behind that names a local path, a key or the real closet."""
    text = path.read_text(encoding="utf-8")
    home = os.path.expanduser("~")
    hits = [p.pattern for p in _FORBIDDEN if p.search(text)]
    if home and home != "/" and home in text:
        hits.append("home directory")
    if hits:
        path.unlink()
        sys.exit(f"Refused to write {path.name}: it contains {', '.join(hits)}.")


def write_json(name: str, record: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    check_public(path)
    rebuild_index()
    return path


def rebuild_index() -> None:
    runs = []
    for p in OUT.glob("*.json"):
        if p.name in ("index.json", CLOSET_FILE):
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        runs.append((raw.get("stage", ""), raw.get("order", 99), p.name))
    (OUT / "index.json").write_text(json.dumps({
        "closet": CLOSET_FILE if (OUT / CLOSET_FILE).exists() else None,
        "recommend": [name for stage, _, name in sorted(runs) if stage == "recommend"],
        "suggest_buy": [name for stage, _, name in sorted(runs) if stage == "suggest-buy"],
    }, indent=1) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def recommend_record(run_id: str, spec: dict, *, report: dict | None, calls: list[str],
                     error: str | None = None, sample: bool = False) -> dict:
    fc: DailyForecast = spec["forecast"]
    if report is not None:
        report["closet_path"] = str(SEED)
    return {
        "id": run_id, "stage": "recommend", "order": spec["order"], "sample": sample,
        "label": spec["label"], "blurb": spec["blurb"], "generated_at": _now(),
        "inputs": {"occasion": spec["occasion"], "formality": spec["formality"]},
        "forecast": fc.model_dump(mode="json", exclude={"latitude", "longitude"}),
        "ambiguous": is_ambiguous(fc),
        "rules_constraints": derive_base_constraints(fc).model_dump(mode="json"),
        "model_calls": calls,
        "report": report,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #


def run_live(ids: list[str]) -> None:
    closet = Closet.load(ROOT / SEED)
    for run_id in ids:
        spec = SCENARIOS[run_id]
        llm = CountingLLM(settings)
        llm.require_configured("The stylist")
        print(f"\n== {run_id} ==")
        try:
            # The weather agent gets the fixed forecast, so nothing is fetched.
            constraints = WeatherAgent(llm, settings).run(WeatherRequest(
                location=spec["forecast"].location, on=spec["forecast"].date,
                forecast=spec["forecast"]))
            result = run_recommend(
                closet, occasion=spec["occasion"], target_formality=spec["formality"],
                constraints=constraints, llm=llm, settings=settings,
                stylist_agent=StylistAgent(llm, settings),
            )
        except Exception as exc:  # noqa: BLE001 - a failed run is still worth showing
            print(f"failed: {exc}")
            write_json(f"{run_id}.json", recommend_record(run_id, spec, report=None,
                                                          calls=llm.purposes, error=str(exc)))
            continue
        assert result.report is not None
        path = write_json(f"{run_id}.json", recommend_record(
            run_id, spec, report=result.report.model_dump(mode="json"), calls=llm.purposes))
        print(f"wrote {path.relative_to(ROOT)} - {len(result.suggestions)} outfit(s), "
              f"{len(llm.purposes)} model call(s), {len(result.warnings)} warning(s)")


def write_sample() -> None:
    """Layout test data. The answers below are hand-written, not model output; the
    pipeline around them (guards, report) is the real one."""
    spec = SCENARIOS["office"]
    picks = StylistResponse(
        outfits=[
            StylistPick(name="Sample: charcoal and white", item_ids=["top-001", "bottom-002",
                        "outer-001", "shoes-003"], formality=4, confidence=0.8,
                        rationale="Hand-written sample text to test the layout.",
                        weather_fit="Hand-written sample text."),
            StylistPick(name="Sample: the black dress", item_ids=["dress-001", "outer-001",
                        "shoes-003", "accessory-001"], formality=4, confidence=0.7,
                        rationale="Hand-written sample text.", weather_fit="Hand-written."),
            StylistPick(name="Sample: an invalid pick", item_ids=["top-001", "top-003",
                        "bottom-002"], formality=3, confidence=0.5,
                        rationale="Two tops, so the guard drops it.", weather_fit="-"),
        ],
        overall_notes="Hand-written sample notes. Not agent output.",
    )
    # The third pick is dropped, so the stylist retries once; the retry returns nothing new.
    retry = StylistResponse(outfits=[picks.outfits[0]], overall_notes="Sample retry.")
    llm = FakeLLM([picks, retry])
    result = run_recommend(Closet.load(ROOT / SEED), occasion=spec["occasion"],
                           target_formality=spec["formality"], llm=llm,
                           constraints=derive_base_constraints(spec["forecast"]),
                           settings=settings, write_report=True)
    assert result.report is not None
    sample = dict(spec, label="Sample (not agent output)", order=0)
    path = write_json("_sample_recommend.json", recommend_record(
        "_sample_recommend", sample, report=result.report.model_dump(mode="json"),
        calls=[c.purpose for c in llm.calls], sample=True))
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Stage 2 and the closet
# --------------------------------------------------------------------------- #


def run_stage2() -> None:
    closet = Closet.load(ROOT / SEED)
    write_json(CLOSET_FILE, {
        "source": str(SEED),
        "items": [i.model_dump(mode="json") for i in closet.items],
    })
    catalog = load_candidates(ROOT / CANDIDATES)
    result = run_suggest_buy(closet, catalog, settings=settings, candidates_path=CANDIDATES)
    assert result.report is not None
    report = result.report.model_dump(mode="json")
    report["closet_path"] = str(SEED)
    path = write_json("suggest-buy.json", {
        "id": "suggest-buy", "stage": "suggest-buy", "order": 1, "sample": False,
        "label": "What to buy next", "generated_at": _now(),
        "blurb": f"{len(catalog)} candidate purchases ranked against the seed closet.",
        "report": report,
    })
    print(f"wrote {path.relative_to(ROOT)} and {CLOSET_FILE}")
    for r in result.optimization.recommendations:
        print(f"  {r.rank}. {r.candidate.label:<28} {r.verdict:<11} +{r.new_outfit_count} looks, "
              f"{r.new_combination_count} combinations")


def drop_samples() -> None:
    for p in OUT.glob("_sample_*"):
        p.unlink()
        print(f"deleted {p.relative_to(ROOT)}")
    rebuild_index()


def main() -> None:
    os.chdir(ROOT)  # settings resolve relative paths (reports/runs) from here
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--live", nargs="+", choices=sorted(SCENARIOS), metavar="SCENARIO",
                    help=f"run the real stylist for: {', '.join(SCENARIOS)}")
    ap.add_argument("--suggest-buy", action="store_true", help="stage 2; no API calls")
    ap.add_argument("--sample", action="store_true", help="write the hand-written layout sample")
    ap.add_argument("--drop-samples", action="store_true", help="delete _sample_* files")
    args = ap.parse_args()
    if not (args.live or args.suggest_buy or args.sample or args.drop_samples):
        ap.print_help()
        return
    if args.suggest_buy:
        run_stage2()
    if args.sample:
        write_sample()
    if args.drop_samples:
        drop_samples()
    if args.live:
        run_live(args.live)


if __name__ == "__main__":
    main()

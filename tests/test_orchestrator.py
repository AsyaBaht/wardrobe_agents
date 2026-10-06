"""The two pipelines, the run artifacts they write, and the loose coupling
between the stages.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import pytest

from config.settings import Settings
from wardrobe_agents.agents.stylist import StylistAgent
from wardrobe_agents.llm import FakeLLM
from wardrobe_agents.orchestrator import (
    load_candidates,
    load_or_build_compatibility,
    rebuild_compatibility,
    run_recommend,
    run_suggest_buy,
)
from wardrobe_agents.schemas import (
    RecommendRunReport,
    StylistPick,
    StylistResponse,
    SuggestBuyRunReport,
    TempBand,
    WeatherConstraints,
)

SRC = Path(__file__).resolve().parent.parent / "src" / "wardrobe_agents"


@pytest.fixture
def constraints() -> WeatherConstraints:
    return WeatherConstraints(
        date=date(2026, 9, 5),
        location="Berlin",
        temp_min_c=7.1,
        temp_max_c=18.4,
        temp_band=TempBand.MILD,
        precipitation_mm=1.8,
        precipitation_probability_pct=45,
        wind_kph=24.0,
        conditions="slight rain showers",
        min_warmth=2,
        max_warmth=3,
        source="manual",
    )


@pytest.fixture
def stylist() -> StylistAgent:
    return StylistAgent(
        llm=FakeLLM(
            [
                StylistResponse(
                    outfits=[
                        StylistPick(
                            name="Weekday neutral",
                            item_ids=["top-001", "bottom-003", "shoes-002"],
                            rationale="Neutral base, warm-toned boots.",
                            weather_fit="Layerable for the swing.",
                            formality=3,
                            confidence=0.85,
                        )
                    ],
                    overall_notes="Good coverage today.",
                )
            ]
        )
    )


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #


def test_supplied_constraints_skip_the_weather_agent(seed_closet, constraints, stylist, test_settings):
    """No location, no network - the weather step is genuinely optional."""
    result = run_recommend(
        seed_closet, constraints=constraints, stylist_agent=stylist, settings=test_settings
    )

    assert result.constraints.source == "manual"
    assert len(result.suggestions) == 1
    assert result.suggestions[0].name == "Weekday neutral"


def test_recommend_without_location_or_constraints_is_an_error(seed_closet, stylist, test_settings):
    with pytest.raises(ValueError, match="Either a location"):
        run_recommend(seed_closet, stylist_agent=stylist, settings=test_settings)


def test_recommend_writes_a_replayable_report(seed_closet, constraints, stylist, test_settings):
    result = run_recommend(
        seed_closet,
        constraints=constraints,
        occasion="client dinner",
        stylist_agent=stylist,
        settings=test_settings,
    )

    assert result.report_path is not None and result.report_path.exists()
    report = RecommendRunReport.model_validate_json(result.report_path.read_text())

    assert report.stage == "recommend"
    assert report.occasion == "client dinner"
    assert report.closet_item_count == 15
    assert report.suggestions[0].item_ids == ["top-001", "bottom-003", "shoes-002"]
    assert report.constraints.location == "Berlin"


def test_dropped_outfits_are_recorded_in_the_report(seed_closet, constraints, test_settings):
    pick = dict(rationale="r", weather_fit="w", formality=3, confidence=0.8)
    llm = FakeLLM(
        [
            StylistResponse(
                outfits=[
                    StylistPick(name="Real", item_ids=["top-001", "bottom-003"], **pick),
                    StylistPick(name="Invented", item_ids=["top-001", "bottom-999"], **pick),
                ],
                overall_notes="",
            )
        ]
    )
    result = run_recommend(
        seed_closet,
        constraints=constraints,
        stylist_agent=StylistAgent(llm=llm),
        settings=test_settings,
    )

    report = RecommendRunReport.model_validate_json(result.report_path.read_text())
    assert report.warnings == result.warnings
    assert "bottom-999" in report.warnings[0]


def test_reports_can_be_disabled(seed_closet, constraints, stylist, test_settings):
    result = run_recommend(
        seed_closet, constraints=constraints, stylist_agent=stylist, settings=test_settings, write_report=False
    )
    assert result.report_path is None
    assert not test_settings.reports_dir.exists()


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #


def test_suggest_buy_runs_with_no_api_key_and_no_network(seed_closet, test_settings):
    candidates = load_candidates(test_settings.default_candidates_path)
    result = run_suggest_buy(seed_closet, candidates, settings=test_settings)

    assert result.optimization.stats is not None
    assert result.optimization.stats.item_count == 15
    assert len(result.optimization.recommendations) == len(candidates)
    assert [r.rank for r in result.optimization.recommendations] == list(range(1, len(candidates) + 1))


def test_suggest_buy_writes_a_diffable_report(seed_closet, test_settings):
    candidates = load_candidates(test_settings.default_candidates_path)
    result = run_suggest_buy(
        seed_closet, candidates, settings=test_settings, candidates_path=test_settings.default_candidates_path
    )

    assert result.report_path is not None
    report = SuggestBuyRunReport.model_validate_json(result.report_path.read_text())

    assert report.stage == "suggest-buy"
    assert report.marginal_gain_threshold == test_settings.marginal_gain_threshold
    assert report.recommendations[0].explanation


def test_report_filenames_are_timestamped_and_stage_tagged(seed_closet, test_settings):
    result = run_suggest_buy(seed_closet, load_candidates(test_settings.default_candidates_path), settings=test_settings)
    assert result.report_path is not None
    assert result.report_path.name.endswith("-suggest-buy.json")


def test_load_candidates_accepts_a_bare_list(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [{"category": "top", "subcategory": "tee", "colors": ["white"], "fabric": "cotton", "warmth": 1, "formality": 2}]
        )
    )
    candidates = load_candidates(path)
    assert candidates[0].candidate_id == "cand-001", "ids are filled in when omitted"


# --------------------------------------------------------------------------- #
# Compatibility cache
# --------------------------------------------------------------------------- #


def test_cache_is_written_then_reused(seed_closet, test_settings):
    rebuild_compatibility(seed_closet, test_settings)
    assert test_settings.compatibility_cache_path.exists()

    _graph, cached = load_or_build_compatibility(seed_closet, test_settings)
    assert cached


def test_a_stale_cache_is_rebuilt_rather_than_trusted(seed_closet, test_settings):
    """A graph that no longer covers the closet is silently wrong - worse than slow."""
    rebuild_compatibility(seed_closet, test_settings)
    seed_closet.remove("top-001")

    _graph, cached = load_or_build_compatibility(seed_closet, test_settings)
    assert not cached


def test_a_corrupt_cache_is_rebuilt_rather_than_raised(seed_closet, test_settings):
    test_settings.compatibility_cache_path.parent.mkdir(parents=True, exist_ok=True)
    test_settings.compatibility_cache_path.write_text("{ not json")

    graph, cached = load_or_build_compatibility(seed_closet, test_settings)
    assert not cached
    assert len(graph) > 0


def test_cache_is_invalidated_by_a_threshold_change(seed_closet, tmp_path):
    strict = Settings(compatibility_cache_path=tmp_path / "c.json", compatibility_threshold=0.55)
    rebuild_compatibility(seed_closet, strict)

    loose = Settings(compatibility_cache_path=tmp_path / "c.json", compatibility_threshold=0.9)
    _graph, cached = load_or_build_compatibility(seed_closet, loose)
    assert not cached


# --------------------------------------------------------------------------- #
# Architecture: the stages stay loosely coupled
# --------------------------------------------------------------------------- #


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_stage_1_never_imports_stage_2_internals():
    for path in (SRC / "agents").glob("*.py"):
        offending = {m for m in _imported_modules(path) if "compatibility" in m}
        assert not offending, f"{path.name} imports stage 2: {offending}"


def test_stage_2_never_imports_an_agent():
    for path in (SRC / "compatibility").glob("*.py"):
        offending = {m for m in _imported_modules(path) if ".agents" in m}
        assert not offending, f"{path.name} imports stage 1: {offending}"


def test_both_stages_read_the_same_item_schema():
    """The coupling that *should* exist: one ClosetItem, imported by both."""
    stylist = _imported_modules(SRC / "agents" / "stylist.py")
    scoring = _imported_modules(SRC / "compatibility" / "scoring.py")
    assert "wardrobe_agents.schemas" in stylist
    assert "wardrobe_agents.schemas" in scoring

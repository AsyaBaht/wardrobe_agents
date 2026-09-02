"""The two pipelines.

**Stage 1** (:func:`run_recommend`): weather agent -> stylist agent, with the
closet read through :mod:`wardrobe_agents.closet`. The weather step is skipped
entirely when the caller supplies constraints directly.

**Stage 2** (:func:`run_suggest_buy`): scoring -> enumeration -> optimizer.

The two are siblings, not layers. Stage 1 never imports the compatibility
package and stage 2 never imports an agent; the only thing they share is the
closet dataset and the schemas that describe it. Either can be deleted without
breaking the other.

Both write a timestamped JSON report to ``reports/runs/`` so a run is an artifact
you can diff, not just something that scrolled past in a terminal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from config.settings import Settings, settings as default_settings
from wardrobe_agents.agents.stylist import StylistAgent, StylistRequest
from wardrobe_agents.agents.weather import WeatherAgent, WeatherRequest
from wardrobe_agents.closet import Closet, ClosetError, load_items_file
from wardrobe_agents.compatibility.optimizer import OptimizationResult, optimize_purchases
from wardrobe_agents.compatibility.scoring import CompatibilityGraph, score_closet
from wardrobe_agents.llm import StructuredLLM
from wardrobe_agents.schemas import (
    OutfitSuggestion,
    PurchaseCandidate,
    RecommendRunReport,
    SuggestBuyRunReport,
    WeatherConstraints,
)


@dataclass(slots=True)
class Stage1Result:
    constraints: WeatherConstraints
    suggestions: list[OutfitSuggestion] = field(default_factory=list)
    overall_notes: str = ""
    warnings: list[str] = field(default_factory=list)
    report: RecommendRunReport | None = None
    report_path: Path | None = None


@dataclass(slots=True)
class Stage2Result:
    optimization: OptimizationResult
    report: SuggestBuyRunReport | None = None
    report_path: Path | None = None


def load_candidates(path: Path) -> list[PurchaseCandidate]:
    """Read a candidate-purchase catalog (JSON or YAML)."""
    raw = load_items_file(Path(path), key="candidates")
    candidates: list[PurchaseCandidate] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ClosetError(f"Candidate #{index} in {path} is not an object.")
        entry.setdefault("candidate_id", f"cand-{index + 1:03d}")
        candidates.append(PurchaseCandidate.model_validate(entry))
    return candidates


def _write_report(report: BaseModel, settings: Settings, run_id: str) -> Path:
    """Persist a run artifact. Report writing must never break a run, so a
    failure here is surfaced by the caller rather than raised mid-pipeline."""
    directory = Path(settings.reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _run_id(stage: str, now: datetime | None = None) -> str:
    return f"{(now or datetime.now()).strftime('%Y%m%dT%H%M%S')}-{stage}"


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #


def run_recommend(
    closet: Closet,
    *,
    location: str | None = None,
    on: date | None = None,
    occasion: str | None = None,
    target_formality: int | None = None,
    count: int | None = None,
    constraints: WeatherConstraints | None = None,
    llm: StructuredLLM | None = None,
    settings: Settings | None = None,
    weather_agent: WeatherAgent | None = None,
    stylist_agent: StylistAgent | None = None,
    write_report: bool = True,
) -> Stage1Result:
    """Weather -> stylist. Returns ranked outfits plus the run artifact.

    Pass ``constraints`` to skip the weather agent entirely (offline, or when the
    user states the conditions themselves).
    """
    settings = settings or default_settings
    llm = llm or StructuredLLM(settings)

    if constraints is None:
        if not location:
            raise ValueError("Either a location (to fetch a forecast) or explicit constraints.")
        agent = weather_agent or WeatherAgent(llm, settings)
        constraints = agent.run(WeatherRequest(location=location, on=on or date.today()))

    stylist = stylist_agent or StylistAgent(llm, settings)
    stylist_result = stylist.run(
        StylistRequest(
            items=closet.wearable(),
            constraints=constraints,
            occasion=occasion,
            target_formality=target_formality,
            count=count,
        )
    )

    result = Stage1Result(
        constraints=constraints,
        suggestions=stylist_result.suggestions,
        overall_notes=stylist_result.overall_notes,
        warnings=stylist_result.warnings,
    )

    if write_report:
        run_id = _run_id("recommend")
        result.report = RecommendRunReport(
            run_id=run_id,
            created_at=datetime.now(),
            closet_path=str(closet.path or ""),
            closet_item_count=len(closet),
            constraints=constraints,
            occasion=occasion,
            target_formality=target_formality,
            suggestions=stylist_result.suggestions,
            overall_notes=stylist_result.overall_notes,
            model=settings.claude_model,
        )
        result.report_path = _write_report(result.report, settings, run_id)

    return result


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #


def run_suggest_buy(
    closet: Closet,
    candidates: Sequence[PurchaseCandidate],
    *,
    settings: Settings | None = None,
    candidates_path: Path | None = None,
    graph: CompatibilityGraph | None = None,
    write_report: bool = True,
) -> Stage2Result:
    """Scoring -> enumeration -> optimizer. No LLM, no network, no API key."""
    settings = settings or default_settings
    items = closet.wearable()
    optimization = optimize_purchases(items, candidates, settings, graph=graph)

    result = Stage2Result(optimization=optimization)

    if write_report:
        run_id = _run_id("suggest-buy")
        assert optimization.stats is not None
        result.report = SuggestBuyRunReport(
            run_id=run_id,
            created_at=datetime.now(),
            closet_path=str(closet.path or ""),
            candidates_path=str(candidates_path) if candidates_path else None,
            stats=optimization.stats,
            recommendations=optimization.recommendations,
            marginal_gain_threshold=settings.marginal_gain_threshold,
        )
        result.report_path = _write_report(result.report, settings, run_id)

    return result


def rebuild_compatibility(
    closet: Closet, settings: Settings | None = None, *, save: bool = True
) -> CompatibilityGraph:
    """Recompute the compatibility graph and cache it beside the closet.

    The cache is a derived artifact: deleting it costs a recomputation and
    nothing else.
    """
    settings = settings or default_settings
    graph = score_closet(closet.wearable(), settings)
    if save:
        graph.save(Path(settings.compatibility_cache_path))
    return graph


def load_or_build_compatibility(
    closet: Closet, settings: Settings | None = None
) -> tuple[CompatibilityGraph, bool]:
    """Return (graph, was_cached). The cache is used only when it covers exactly
    the closet's current items - a stale graph is silently wrong, which is worse
    than a slow one."""
    settings = settings or default_settings
    path = Path(settings.compatibility_cache_path)
    expected = sorted(i.id for i in closet.wearable())

    if path.exists():
        try:
            graph = CompatibilityGraph.load(path)
        except (json.JSONDecodeError, OSError, ValueError):
            graph = None
        else:
            if graph.item_ids == expected and graph.threshold == settings.compatibility_threshold:
                return graph, True

    return rebuild_compatibility(closet, settings), False

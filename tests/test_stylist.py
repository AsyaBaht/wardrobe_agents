"""Stylist agent: ranking and rationale plumbing against a mocked Claude response,
and the guard that stops hallucinated items reaching a recommendation."""

from __future__ import annotations

from datetime import date

import pytest

from wardrobe_agents.agents.stylist import (
    StylistAgent,
    StylistError,
    StylistRequest,
    filter_for_weather,
)
from wardrobe_agents.llm import FakeLLM
from wardrobe_agents.schemas import (
    StylistPick,
    StylistResponse,
    TempBand,
    WeatherConstraints,
)


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
        layering_advice="Layer for an 11C swing.",
        source="rules",
    )


def _response(*picks: StylistPick, notes: str = "Solid options today.") -> StylistResponse:
    return StylistResponse(outfits=list(picks), overall_notes=notes)


def _pick(name: str, item_ids: list[str], confidence: float = 0.8) -> StylistPick:
    return StylistPick(
        name=name,
        item_ids=item_ids,
        rationale="The neutral base lets the rust knit carry the outfit.",
        weather_fit="Warm enough for the morning, sheddable by afternoon.",
        formality=3,
        confidence=confidence,
    )


def test_ranked_suggestions_are_numbered_in_returned_order(seed_closet, constraints):
    llm = FakeLLM(
        [
            _response(
                _pick("Weekday neutral", ["top-001", "bottom-003", "shoes-002"]),
                _pick("Warmer layer", ["top-003", "bottom-001", "shoes-002"]),
            )
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints)
    )

    assert [s.rank for s in result.suggestions] == [1, 2]
    assert result.suggestions[0].name == "Weekday neutral"
    assert result.overall_notes == "Solid options today."


def test_suggestions_carry_the_real_items_not_just_ids(seed_closet, constraints):
    llm = FakeLLM([_response(_pick("Weekday neutral", ["top-001", "bottom-003", "shoes-002"]))])
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints)
    )

    suggestion = result.suggestions[0]
    assert [i.id for i in suggestion.items] == suggestion.item_ids
    assert suggestion.items[0].label == "white oxford shirt"
    assert suggestion.rationale.startswith("The neutral base")


def test_outfits_referencing_unknown_items_are_dropped_with_a_warning(seed_closet, constraints):
    """A hallucinated id must never reach a recommendation."""
    llm = FakeLLM(
        [
            _response(
                _pick("Real", ["top-001", "bottom-003", "shoes-002"]),
                _pick("Invented", ["top-001", "bottom-999"]),
            )
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints)
    )

    assert len(result.suggestions) == 1
    assert result.suggestions[0].rank == 1, "ranks close up after a drop"
    assert "bottom-999" in result.warnings[0]


def test_all_outfits_invalid_raises_rather_than_returning_nothing(seed_closet, constraints):
    llm = FakeLLM([_response(_pick("Invented", ["nope-001"]))])
    with pytest.raises(StylistError, match="no usable outfits"):
        StylistAgent(llm=llm).run(
            StylistRequest(items=seed_closet.wearable(), constraints=constraints)
        )


def test_duplicate_ids_within_one_outfit_are_collapsed(seed_closet, constraints):
    llm = FakeLLM([_response(_pick("Doubled", ["top-001", "top-001", "bottom-003"]))])
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints)
    )
    assert result.suggestions[0].item_ids == ["top-001", "bottom-003"]


def test_empty_closet_raises_before_calling_the_model(constraints):
    llm = FakeLLM([])
    with pytest.raises(StylistError, match="no wearable items"):
        StylistAgent(llm=llm).run(StylistRequest(items=[], constraints=constraints))
    assert llm.calls == []


def test_prompt_contains_the_constraints_and_the_item_ids(seed_closet, constraints):
    llm = FakeLLM([_response(_pick("Weekday neutral", ["top-001", "bottom-003", "shoes-002"]))])
    StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, occasion="client dinner")
    )

    prompt = llm.calls[0].content
    assert "client dinner" in prompt
    assert "Warmth window: items rated 2-3" in prompt
    assert "top-001" in prompt


# --------------------------------------------------------------------------- #
# Deterministic pre-filtering
# --------------------------------------------------------------------------- #


def test_filter_drops_retired_and_out_of_season_items(make_item, constraints):
    items = [
        make_item("top-001", "top", warmth=2),
        make_item("bottom-001", "bottom", warmth=3),
        make_item("shoes-001", "shoes", warmth=2),
        make_item("outer-001", "outer", warmth=4),  # within tolerance of max_warmth 3
        make_item("top-999", "top", warmth=0),  # too light for a 2-3 window
        make_item("top-998", "top", warmth=2, condition="retire"),
    ]
    kept = {i.id for i in filter_for_weather(items, constraints)}

    assert "outer-001" in kept, "layering pieces survive on tolerance"
    assert "top-999" not in kept
    assert "top-998" not in kept


def test_filter_gives_up_rather_than_starving_the_stylist(make_item, constraints):
    """A pre-filter must never be the reason there is nothing to suggest."""
    items = [make_item("top-001", "top", warmth=5), make_item("bottom-001", "bottom", warmth=5)]
    kept = filter_for_weather(items, constraints)

    assert len(kept) == 2, "filtering everything out is abandoned in favour of the full closet"

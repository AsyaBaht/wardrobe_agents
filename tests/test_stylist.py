"""Stylist agent: ranking and rationale plumbing against a mocked Claude response,
and the guard that stops hallucinated items reaching a recommendation.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

from datetime import date

import pytest

from wardrobe_agents.agents.stylist import (
    StylistAgent,
    StylistError,
    StylistRequest,
    filter_for_weather,
)
from wardrobe_agents.llm import FakeLLM, LLMError
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


@pytest.mark.parametrize(
    ("item_ids", "problem"),
    [
        (["top-001", "top-002", "bottom-003"], "2 top"),
        (["top-001", "shoes-002"], "0 bottom"),
        (["dress-001", "bottom-001"], "dress combined"),
        (["top-001", "bottom-003", "outer-001", "outer-002"], "2 outer layers"),
        (["top-001", "bottom-003", "shoes-001", "shoes-002"], "2 pairs of shoes"),
    ],
)
def test_structurally_invalid_outfits_are_dropped_with_a_warning(
    seed_closet, constraints, item_ids, problem
):
    """Real ids are not enough: the pick also has to be an outfit."""
    llm = FakeLLM(
        [_response(_pick("Broken", item_ids), _pick("Real", ["top-001", "bottom-003", "shoes-002"]))]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints)
    )

    assert [s.name for s in result.suggestions] == ["Real"]
    assert result.suggestions[0].rank == 1
    assert problem in result.warnings[0]


@pytest.mark.parametrize(
    "item_ids",
    [
        ["top-001", "bottom-003"],
        ["dress-001"],
        ["dress-001", "outer-001", "shoes-002"],
    ],
)
def test_valid_structures_are_kept(seed_closet, constraints, item_ids):
    llm = FakeLLM([_response(_pick("Fine", item_ids))])
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints)
    )
    assert result.suggestions[0].item_ids == item_ids


def test_a_dropped_outfit_is_replaced_by_a_retry(seed_closet, constraints):
    llm = FakeLLM(
        [
            _response(
                _pick("Real", ["top-001", "bottom-003", "shoes-002"]),
                _pick("Two tops", ["top-001", "top-002", "bottom-003"]),
            ),
            _response(_pick("Replacement", ["top-003", "bottom-001", "shoes-002"])),
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, count=2)
    )

    assert [(s.rank, s.name) for s in result.suggestions] == [(1, "Real"), (2, "Replacement")]
    assert len(llm.calls) == 2
    retry_prompt = llm.calls[1].content
    assert "Two tops" in retry_prompt and "2 top" in retry_prompt, "the model is told what failed"
    assert "top-001, bottom-003, shoes-002" in retry_prompt, "and what not to repeat"
    assert "Return 1 more outfit" in retry_prompt
    assert len(result.warnings) == 1, "the drop stays on the record even though it was replaced"


def test_a_retry_that_repeats_an_accepted_outfit_is_dropped(seed_closet, constraints):
    llm = FakeLLM(
        [
            _response(
                _pick("Real", ["top-001", "bottom-003", "shoes-002"]),
                _pick("Invented", ["top-001", "bottom-999"]),
            ),
            _response(_pick("Same again", ["shoes-002", "top-001", "bottom-003"])),
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, count=2)
    )

    assert [s.name for s in result.suggestions] == ["Real"]
    assert "repeats an outfit" in result.warnings[-1]
    assert len(llm.calls) == 2, "retries are bounded"


def test_a_failed_retry_keeps_the_outfits_already_accepted(seed_closet, constraints):
    llm = FakeLLM(
        [
            _response(
                _pick("Real", ["top-001", "bottom-003", "shoes-002"]),
                _pick("Invented", ["top-001", "bottom-999"]),
            ),
            LLMError("rate limited"),
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, count=2)
    )

    assert [s.name for s in result.suggestions] == ["Real"]
    assert "rate limited" in result.warnings[-1]


def test_no_retry_when_nothing_was_dropped(seed_closet, constraints):
    """Returning fewer outfits than asked is the stylist's judgment, not an error."""
    llm = FakeLLM([_response(_pick("Only one", ["top-001", "bottom-003", "shoes-002"]))])
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, count=3)
    )

    assert len(result.suggestions) == 1
    assert len(llm.calls) == 1
    assert result.warnings == []


def test_surplus_outfits_are_trimmed_to_the_requested_count(seed_closet, constraints):
    llm = FakeLLM(
        [
            _response(
                _pick("One", ["top-001", "bottom-003"]),
                _pick("Two", ["top-003", "bottom-001"]),
            )
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, count=1)
    )
    assert [s.name for s in result.suggestions] == ["One"]


def test_an_inverted_warmth_window_is_rejected(constraints):
    with pytest.raises(ValueError, match="min_warmth"):
        WeatherConstraints.model_validate(
            {**constraints.model_dump(), "min_warmth": 4, "max_warmth": 2}
        )
    with pytest.raises(ValueError, match="temp_min_c"):
        WeatherConstraints.model_validate(
            {**constraints.model_dump(), "temp_min_c": 20.0, "temp_max_c": 5.0}
        )


def _rainy(constraints: WeatherConstraints) -> WeatherConstraints:
    return constraints.model_copy(update={"needs_waterproof_outer": True})


def test_a_waterproof_need_the_closet_cannot_meet_is_a_warning(seed_closet, constraints):
    """The seed closet tags nothing waterproof, so the constraint is unanswerable."""
    llm = FakeLLM([_response(_pick("Real", ["top-001", "bottom-003", "outer-002"]))])
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=_rainy(constraints), count=1)
    )

    assert len(result.suggestions) == 1
    assert "no outer item is tagged 'waterproof'" in result.warnings[0]


def test_an_outfit_without_the_waterproof_layer_is_flagged(make_item, constraints):
    items = [
        make_item("top-001", "top"),
        make_item("bottom-001", "bottom"),
        make_item("outer-001", "outer", tags=["Waterproof"]),
        make_item("outer-002", "outer"),
    ]
    llm = FakeLLM(
        [
            _response(
                _pick("Shell", ["top-001", "bottom-001", "outer-001"]),
                _pick("Denim", ["top-001", "bottom-001", "outer-002"]),
            )
        ]
    )
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=items, constraints=_rainy(constraints), count=2)
    )

    assert [s.name for s in result.suggestions] == ["Shell", "Denim"], "flagged, not dropped"
    assert result.warnings == ["Outfit 'Denim' has no waterproof outer layer."]
    assert "tags: Waterproof" in llm.calls[0].content, "the model can see which item qualifies"


def test_no_protection_warnings_on_a_dry_day(seed_closet, constraints):
    llm = FakeLLM([_response(_pick("Real", ["top-001", "bottom-003"]))])
    result = StylistAgent(llm=llm).run(
        StylistRequest(items=seed_closet.wearable(), constraints=constraints, count=1)
    )
    assert result.warnings == []


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

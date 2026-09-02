"""Weather agent: deterministic parsing from a fixture, rule-derived constraints,
and LLM translation only where a rule table would be lossy."""

from __future__ import annotations

from datetime import date

import pytest

from wardrobe_agents.agents.weather import (
    WeatherAgent,
    WeatherError,
    WeatherRequest,
    apply_translation,
    band_for_temp,
    derive_base_constraints,
    is_ambiguous,
    parse_forecast,
    warmth_for_temp,
)
from wardrobe_agents.llm import FakeLLM, LLMError
from wardrobe_agents.schemas import TempBand, WeatherTranslation

WHEN = date(2026, 9, 5)


@pytest.fixture
def forecast(forecast_payload):
    return parse_forecast(forecast_payload, location="Berlin, Germany", on=WHEN)


@pytest.fixture
def translation() -> WeatherTranslation:
    return WeatherTranslation(
        needs_waterproof_outer=False,
        needs_windproof=True,
        layering_advice="Start with a shell you can drop by lunchtime.",
        prefer_fabrics=["wool", "cotton"],
        avoid_fabrics=["linen"],
        warmth_adjustment=1,
        notes="Showers are brief; the wind is what you will feel.",
    )


# --------------------------------------------------------------------------- #
# Deterministic parsing
# --------------------------------------------------------------------------- #


def test_parses_the_open_meteo_payload(forecast):
    assert forecast.temp_min_c == pytest.approx(7.1)
    assert forecast.temp_max_c == pytest.approx(18.4)
    assert forecast.precipitation_probability_pct == 45
    assert forecast.wind_kph == pytest.approx(24.0)
    assert forecast.conditions == "slight rain showers", "WMO code 80 is decoded, not passed through"


def test_a_date_outside_the_forecast_window_is_a_clear_error(forecast_payload):
    with pytest.raises(WeatherError, match="no entry for 2026-12-25"):
        parse_forecast(forecast_payload, location="Berlin", on=date(2026, 12, 25))


def test_missing_daily_data_is_a_clear_error():
    with pytest.raises(WeatherError, match="no daily data"):
        parse_forecast({"daily": {}}, location="Berlin", on=WHEN)


@pytest.mark.parametrize(
    ("temp", "band", "warmth"),
    [(-4.0, TempBand.FREEZING, 5), (3.0, TempBand.COLD, 4), (11.0, TempBand.COOL, 3),
     (18.0, TempBand.MILD, 2), (24.0, TempBand.WARM, 1), (31.0, TempBand.HOT, 0)],
)
def test_temperature_maps_to_a_band_and_a_warmth(temp, band, warmth):
    assert band_for_temp(temp) is band
    assert warmth_for_temp(temp) == warmth


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def test_warmth_window_spans_the_day(forecast):
    """Light enough for the afternoon high, warm enough for the morning low."""
    constraints = derive_base_constraints(forecast)

    assert constraints.min_warmth == warmth_for_temp(forecast.temp_max_c)
    assert constraints.max_warmth == warmth_for_temp(forecast.temp_min_c)
    assert constraints.min_warmth < constraints.max_warmth, "an 11C swing needs a range"


def test_rule_constraints_carry_the_forecast_forward(forecast):
    constraints = derive_base_constraints(forecast)
    assert constraints.source == "rules"
    assert constraints.temp_band is TempBand.MILD
    assert "swing" in constraints.layering_advice


def test_heavy_rain_needs_a_waterproof_layer_without_asking_a_model(forecast_payload):
    forecast_payload["daily"]["precipitation_probability_max"] = [95]
    forecast_payload["daily"]["precipitation_sum"] = [12.0]
    forecast = parse_forecast(forecast_payload, location="Berlin", on=WHEN)

    assert derive_base_constraints(forecast).needs_waterproof_outer


# --------------------------------------------------------------------------- #
# Ambiguity: when is a model call worth making?
# --------------------------------------------------------------------------- #


def test_showery_breezy_day_is_ambiguous(forecast):
    assert is_ambiguous(forecast)


def test_a_settled_dry_day_is_not_ambiguous(forecast_payload):
    forecast_payload["daily"].update(
        weather_code=[0],
        temperature_2m_max=[24.0],
        temperature_2m_min=[18.0],
        precipitation_sum=[0.0],
        precipitation_probability_max=[0],
        wind_speed_10m_max=[6.0],
    )
    forecast = parse_forecast(forecast_payload, location="Berlin", on=WHEN)
    assert not is_ambiguous(forecast)


def test_agent_skips_the_model_on_an_unambiguous_day(forecast_payload):
    forecast_payload["daily"].update(
        weather_code=[0], temperature_2m_max=[24.0], temperature_2m_min=[18.0],
        precipitation_sum=[0.0], precipitation_probability_max=[0], wind_speed_10m_max=[6.0],
    )
    forecast = parse_forecast(forecast_payload, location="Berlin", on=WHEN)
    llm = FakeLLM([])  # any call would raise
    agent = WeatherAgent(llm=llm)

    constraints = agent.run(WeatherRequest(location="Berlin", on=WHEN, forecast=forecast))

    assert llm.calls == [], "a settled day needs no reasoning"
    assert constraints.source == "rules"


def test_agent_calls_the_model_on_an_ambiguous_day(forecast, translation):
    llm = FakeLLM([translation])
    agent = WeatherAgent(llm=llm)

    constraints = agent.run(WeatherRequest(location="Berlin", on=WHEN, forecast=forecast))

    assert len(llm.calls) == 1
    assert constraints.source == "rules+llm"
    assert constraints.needs_windproof
    assert constraints.notes.startswith("Showers are brief")


def test_agent_never_calls_the_network_when_given_a_forecast(forecast, translation):
    """`forecast=` short-circuits the fetch, which is what makes this testable."""
    agent = WeatherAgent(llm=FakeLLM([translation]))
    assert agent.run(WeatherRequest(location="nowhere", on=WHEN, forecast=forecast))


# --------------------------------------------------------------------------- #
# Translation folding
# --------------------------------------------------------------------------- #


def test_translation_shifts_the_warmth_window_within_bounds(forecast, translation):
    base = derive_base_constraints(forecast)
    folded = apply_translation(base, translation)

    assert folded.min_warmth == base.min_warmth + 1
    assert folded.max_warmth == base.max_warmth + 1
    assert folded.avoid_fabrics == ["linen"]


def test_warmth_window_is_clamped_to_the_scale(forecast, translation):
    base = derive_base_constraints(forecast).model_copy(update={"min_warmth": 5, "max_warmth": 5})
    folded = apply_translation(base, translation)
    assert folded.max_warmth == 5, "cannot exceed the top of the 0-5 scale"


def test_a_failed_model_call_falls_back_to_the_rules(forecast):
    """Rules are a complete answer; a refinement failure must not break the run."""
    agent = WeatherAgent(llm=FakeLLM([LLMError("rate limited")]))

    constraints = agent.run(WeatherRequest(location="Berlin", on=WHEN, forecast=forecast))

    assert constraints.source == "rules"
    assert constraints.min_warmth == derive_base_constraints(forecast).min_warmth


def test_use_llm_false_disables_the_call_entirely(forecast):
    llm = FakeLLM([])
    agent = WeatherAgent(llm=llm, use_llm=False)
    assert agent.run(WeatherRequest(location="Berlin", on=WHEN, forecast=forecast)).source == "rules"
    assert llm.calls == []

"""Weather agent: deterministic forecast fetch, then LLM translation where a rule
table would be lossy.

The split is deliberate.

* **Deterministic** (:func:`fetch_forecast`): call Open-Meteo, parse numbers.
  No model involved, no key required, fully reproducible in tests from a fixture.
* **Rules** (:func:`derive_base_constraints`): temperature -> warmth window, and
  the unambiguous precipitation/wind calls. A dry 26C day needs no reasoning.
* **LLM** (only when :func:`is_ambiguous` says so): "scattered showers, 15 kph
  wind, 11C swing" is exactly the case where a lookup table throws away the
  judgment - is that an umbrella day, a waterproof-shell day, or neither? Claude
  answers *only* those questions, into :class:`WeatherTranslation`, and the
  numeric bands stay rule-derived.

So the agent works with no API key (returning ``source="rules"``), and spends a
call only where the call earns its keep.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Any

from config.settings import Settings, settings as default_settings
from wardrobe_agents.agents.base import BaseAgent
from wardrobe_agents.llm import LLMError, StructuredLLM
from wardrobe_agents.schemas import (
    DailyForecast,
    TempBand,
    WeatherConstraints,
    WeatherTranslation,
)

# WMO weather interpretation codes used by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

#: Codes whose clothing implication genuinely depends on context.
AMBIGUOUS_CODES = {45, 48, 51, 53, 56, 57, 61, 66, 71, 77, 80, 81, 85, 95, 96}

#: Temperature (C) at or above which each warmth rating is the right choice.
_WARMTH_BY_TEMP: tuple[tuple[float, int], ...] = (
    (28.0, 0),
    (22.0, 1),
    (15.0, 2),
    (8.0, 3),
    (0.0, 4),
)

_BAND_BY_TEMP: tuple[tuple[float, TempBand], ...] = (
    (28.0, TempBand.HOT),
    (22.0, TempBand.WARM),
    (15.0, TempBand.MILD),
    (8.0, TempBand.COOL),
    (0.0, TempBand.COLD),
)

_FABRICS_BY_BAND: dict[TempBand, tuple[list[str], list[str]]] = {
    TempBand.HOT: (["linen", "cotton", "silk"], ["wool", "fleece", "down", "leather"]),
    TempBand.WARM: (["cotton", "linen"], ["wool", "fleece", "down"]),
    TempBand.MILD: (["cotton", "denim"], ["down"]),
    TempBand.COOL: (["wool", "denim", "cotton"], ["linen"]),
    TempBand.COLD: (["wool", "fleece", "down"], ["linen", "silk"]),
    TempBand.FREEZING: (["wool", "down", "fleece"], ["linen", "silk", "cotton"]),
}


class WeatherError(RuntimeError):
    """Forecast could not be fetched or understood."""


def warmth_for_temp(temp_c: float) -> int:
    """Warmth rating appropriate to a temperature."""
    for threshold, warmth in _WARMTH_BY_TEMP:
        if temp_c >= threshold:
            return warmth
    return 5


def band_for_temp(temp_c: float) -> TempBand:
    for threshold, band in _BAND_BY_TEMP:
        if temp_c >= threshold:
            return band
    return TempBand.FREEZING


# --------------------------------------------------------------------------- #
# Deterministic half
# --------------------------------------------------------------------------- #


def _http_get_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full = f"{url}?{query}"
    request = urllib.request.Request(full, headers={"User-Agent": "wardrobe-agents/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise WeatherError(f"Weather service returned HTTP {exc.code} for {url}.") from exc
    except urllib.error.URLError as exc:
        raise WeatherError(f"Could not reach the weather service at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise WeatherError(f"Weather request to {url} timed out after {timeout}s.") from exc

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise WeatherError(f"Weather service returned invalid JSON from {url}.") from exc


def geocode(location: str, settings: Settings | None = None) -> tuple[float, float, str]:
    """Resolve a place name to (latitude, longitude, canonical name)."""
    settings = settings or default_settings
    data = _http_get_json(
        settings.geocoding_api_base_url,
        {"name": location, "count": 1, "language": "en", "format": "json"},
        settings.weather_timeout_seconds,
    )
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"No location found for {location!r}. Try 'City, Country'.")

    hit = results[0]
    name_parts = [hit.get("name"), hit.get("admin1"), hit.get("country")]
    canonical = ", ".join(p for p in name_parts if p)
    return float(hit["latitude"]), float(hit["longitude"]), canonical or location


def fetch_forecast(
    location: str,
    on: date,
    settings: Settings | None = None,
    *,
    coordinates: tuple[float, float, str] | None = None,
) -> DailyForecast:
    """Fetch one day of forecast. Purely deterministic - no model involved."""
    settings = settings or default_settings
    latitude, longitude, canonical = coordinates or geocode(location, settings)

    data = _http_get_json(
        settings.weather_api_base_url,
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                ]
            ),
            "wind_speed_unit": "kmh",
            "timezone": "auto",
            "start_date": on.isoformat(),
            "end_date": on.isoformat(),
        },
        settings.weather_timeout_seconds,
    )
    return parse_forecast(data, location=canonical, on=on)


def parse_forecast(payload: dict[str, Any], *, location: str, on: date) -> DailyForecast:
    """Turn an Open-Meteo response into a :class:`DailyForecast`.

    Split out from the HTTP call so tests exercise it against a saved fixture.
    """
    daily = payload.get("daily")
    if not isinstance(daily, dict) or not daily.get("time"):
        raise WeatherError(f"Forecast response contained no daily data for {on.isoformat()}.")

    try:
        index = list(daily["time"]).index(on.isoformat())
    except ValueError as exc:
        available = ", ".join(daily["time"][:3])
        raise WeatherError(
            f"Forecast has no entry for {on.isoformat()} (returned: {available}). "
            "Open-Meteo covers roughly today through +16 days."
        ) from exc

    def value(key: str, default: float = 0.0) -> float:
        series = daily.get(key)
        if not series or index >= len(series) or series[index] is None:
            return default
        return float(series[index])

    code = int(value("weather_code"))
    return DailyForecast(
        date=on,
        location=location,
        latitude=float(payload.get("latitude", 0.0)),
        longitude=float(payload.get("longitude", 0.0)),
        temp_min_c=value("temperature_2m_min"),
        temp_max_c=value("temperature_2m_max"),
        precipitation_mm=value("precipitation_sum"),
        precipitation_probability_pct=int(value("precipitation_probability_max")),
        wind_kph=value("wind_speed_10m_max"),
        weather_code=code,
        conditions=WMO_CODES.get(code, f"weather code {code}"),
    )


# --------------------------------------------------------------------------- #
# Rules half
# --------------------------------------------------------------------------- #


def derive_base_constraints(forecast: DailyForecast) -> WeatherConstraints:
    """Rule-derived constraints. Always runs; the LLM only refines this."""
    band = band_for_temp(forecast.temp_max_c)
    # Warmth window spans the day: light enough for the high, warm enough for the low.
    min_warmth = warmth_for_temp(forecast.temp_max_c)
    max_warmth = warmth_for_temp(forecast.temp_min_c)
    prefer, avoid = _FABRICS_BY_BAND[band]

    swing = forecast.temp_max_c - forecast.temp_min_c
    if swing >= 10:
        advice = (
            f"{swing:.0f}C swing between morning and afternoon - layer so you can shed one piece."
        )
    elif max_warmth >= 4:
        advice = "Cold enough that an insulating outer layer is doing the real work."
    else:
        advice = "Steady temperatures; a single layer plus an optional light outer works."

    return WeatherConstraints(
        date=forecast.date,
        location=forecast.location,
        temp_min_c=forecast.temp_min_c,
        temp_max_c=forecast.temp_max_c,
        temp_band=band,
        precipitation_mm=forecast.precipitation_mm,
        precipitation_probability_pct=forecast.precipitation_probability_pct,
        wind_kph=forecast.wind_kph,
        conditions=forecast.conditions,
        min_warmth=min_warmth,
        max_warmth=max_warmth,
        needs_waterproof_outer=forecast.precipitation_probability_pct >= 60
        or forecast.precipitation_mm >= 3.0,
        needs_windproof=forecast.wind_kph >= 35.0,
        prefer_fabrics=list(prefer),
        avoid_fabrics=list(avoid),
        layering_advice=advice,
        notes="",
        source="rules",
    )


def is_ambiguous(forecast: DailyForecast) -> bool:
    """Would a rule table lose something here?

    True when the forecast sits in a judgment zone: a maybe-rain probability, a
    trace of precipitation, notable-but-not-decisive wind, a large diurnal swing,
    or a weather code (fog, drizzle, showers, storms) whose clothing implication
    depends on context.
    """
    return (
        20 <= forecast.precipitation_probability_pct <= 70
        or 0.0 < forecast.precipitation_mm < 3.0
        or 15.0 <= forecast.wind_kph < 35.0
        or (forecast.temp_max_c - forecast.temp_min_c) >= 10.0
        or forecast.weather_code in AMBIGUOUS_CODES
    )


def apply_translation(
    constraints: WeatherConstraints, translation: WeatherTranslation
) -> WeatherConstraints:
    """Fold Claude's judgment into the rule-derived constraints.

    The model decides the maybe-cases. Where the rules were decisive - rain or
    wind beyond the ambiguous zone in :func:`is_ambiguous` - their answer stands,
    because the call may have been triggered by something else entirely (a 90%
    rain day is still "ambiguous" if it is also breezy).
    """
    shift = translation.warmth_adjustment
    rain_is_decisive = constraints.needs_waterproof_outer and (
        constraints.precipitation_probability_pct > 70 or constraints.precipitation_mm >= 3.0
    )
    return constraints.model_copy(
        update={
            "min_warmth": max(0, min(5, constraints.min_warmth + shift)),
            "max_warmth": max(0, min(5, constraints.max_warmth + shift)),
            "needs_waterproof_outer": rain_is_decisive or translation.needs_waterproof_outer,
            "needs_windproof": constraints.needs_windproof or translation.needs_windproof,
            "prefer_fabrics": translation.prefer_fabrics or constraints.prefer_fabrics,
            "avoid_fabrics": translation.avoid_fabrics,
            "layering_advice": translation.layering_advice or constraints.layering_advice,
            "notes": translation.notes,
            "source": "rules+llm",
        }
    )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

TRANSLATION_SYSTEM = """You translate a weather forecast into clothing constraints.

You are given the numbers plus the warmth window a rule table already derived
(warmth is a 0-5 scale: 0 hot-weather-only, 1 light, 2 mild, 3 cool, 4 cold,
5 freezing). Your job is only the judgment a lookup table would get wrong:

- Does this genuinely call for a waterproof outer layer, or just an umbrella?
- Is the wind strong enough to matter for what someone wears?
- How should they layer across the day's temperature range?
- Which fabrics suit or fail these specific conditions?
- Does the rule-derived warmth window need a one-step nudge? Use 0 unless the
  combination of wind, damp, and sun genuinely makes the day feel different from
  its temperature. Reserve -1/+1 for real cases.

Be concrete and brief. Do not restate the numbers back."""


@dataclass(slots=True)
class WeatherRequest:
    """Input to :class:`WeatherAgent`."""

    location: str
    on: date
    forecast: DailyForecast | None = None
    """Pre-fetched forecast; when supplied the agent performs no network call."""


class WeatherAgent(BaseAgent[WeatherRequest, WeatherConstraints]):
    """Forecast -> :class:`WeatherConstraints` the stylist can reason over."""

    name = "weather"

    def __init__(
        self,
        llm: StructuredLLM | None = None,
        settings: Settings | None = None,
        *,
        use_llm: bool = True,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or StructuredLLM(self.settings)
        self.use_llm = use_llm

    def run(self, payload: WeatherRequest) -> WeatherConstraints:
        forecast = payload.forecast or fetch_forecast(payload.location, payload.on, self.settings)
        constraints = derive_base_constraints(forecast)

        if not self.use_llm or not is_ambiguous(forecast) or not self.llm.is_configured():
            return constraints

        try:
            translation = self.llm.call(
                system=TRANSLATION_SYSTEM,
                content=self._prompt(forecast, constraints),
                response_model=WeatherTranslation,
                purpose="translating the forecast into clothing constraints",
            )
        except LLMError:
            # Rules are a complete answer on their own; a failed refinement must
            # not take down the recommendation.
            return constraints

        return apply_translation(constraints, translation)

    @staticmethod
    def _prompt(forecast: DailyForecast, constraints: WeatherConstraints) -> str:
        return (
            f"Location: {forecast.location}\n"
            f"Date: {forecast.date.isoformat()}\n"
            f"Conditions: {forecast.conditions}\n"
            f"Temperature: {forecast.temp_min_c:.0f}C low, {forecast.temp_max_c:.0f}C high "
            f"({forecast.temp_max_c - forecast.temp_min_c:.0f}C swing)\n"
            f"Precipitation: {forecast.precipitation_probability_pct}% chance, "
            f"{forecast.precipitation_mm:.1f} mm expected\n"
            f"Wind: {forecast.wind_kph:.0f} kph maximum\n\n"
            f"Rule-derived warmth window: {constraints.min_warmth}-{constraints.max_warmth} "
            f"(band: {constraints.temp_band.value})"
        )

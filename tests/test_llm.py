"""StructuredLLM: a response that fails the schema is an LLMError like any other
failed call, so callers' fallbacks still apply.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

from datetime import date

import pytest

from wardrobe_agents.agents.weather import WeatherAgent, WeatherRequest, parse_forecast
from wardrobe_agents.llm import LLMError, StructuredLLM
from wardrobe_agents.schemas import WeatherTranslation


class _Messages:
    """Stands in for the SDK client: validates the queued text as ``messages.parse`` does."""

    def __init__(self, text: str) -> None:
        self.text = text

    def parse(self, *, output_format, **_: object):
        return output_format.model_validate_json(self.text)


class _Client:
    def __init__(self, text: str) -> None:
        self.messages = _Messages(text)


OUT_OF_RANGE = (
    '{"needs_waterproof_outer": true, "needs_windproof": false, "layering_advice": "x", '
    '"prefer_fabrics": [], "avoid_fabrics": [], "warmth_adjustment": 2, "notes": ""}'
)
TRUNCATED = '{"needs_waterproof_outer": true, "needs_wind'


@pytest.mark.parametrize("text", [OUT_OF_RANGE, TRUNCATED])
def test_a_response_that_fails_the_schema_is_an_llm_error(text):
    llm = StructuredLLM(client=_Client(text))
    with pytest.raises(LLMError, match="did not match WeatherTranslation"):
        llm.call(system="s", content="c", response_model=WeatherTranslation, purpose="testing")


def test_weather_falls_back_to_rules_when_the_response_fails_the_schema(forecast_payload):
    forecast = parse_forecast(forecast_payload, location="Berlin", on=date(2026, 9, 5))
    agent = WeatherAgent(llm=StructuredLLM(client=_Client(OUT_OF_RANGE)))

    constraints = agent.run(WeatherRequest(location="Berlin", on=forecast.date, forecast=forecast))

    assert constraints.source == "rules"

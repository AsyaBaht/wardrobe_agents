"""The three stage-1 agents, each implementing the :class:`Agent` protocol.

Author: Anastasiia Bakhtoiarova
"""

from wardrobe_agents.agents.base import Agent, BaseAgent
from wardrobe_agents.agents.cataloguing import CataloguingAgent, CatalogueRequest, CatalogueResult
from wardrobe_agents.agents.stylist import StylistAgent, StylistRequest, StylistResult
from wardrobe_agents.agents.weather import WeatherAgent, WeatherRequest

__all__ = [
    "Agent",
    "BaseAgent",
    "CataloguingAgent",
    "CatalogueRequest",
    "CatalogueResult",
    "StylistAgent",
    "StylistRequest",
    "StylistResult",
    "WeatherAgent",
    "WeatherRequest",
]

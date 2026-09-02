"""Shared fixtures.

No test in this suite makes a network call. Anything that would reach Claude goes
through :class:`~wardrobe_agents.llm.FakeLLM`, and anything that would reach
Open-Meteo goes through a saved forecast payload.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

import pytest

from config.settings import Settings
from wardrobe_agents.closet import Closet
from wardrobe_agents.schemas import ClosetItem, PurchaseCandidate

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_CLOSET = REPO_ROOT / "examples" / "seed_closet" / "closet.json"
SEED_CANDIDATES = REPO_ROOT / "examples" / "candidate_purchases.json"


@pytest.fixture
def make_item() -> Callable[..., ClosetItem]:
    """Build a ClosetItem with sensible defaults; override what the test cares about."""

    def _make(
        item_id: str,
        category: str,
        *,
        subcategory: str = "thing",
        colors: list[str] | None = None,
        pattern: str = "solid",
        fabric: str = "cotton",
        warmth: int = 2,
        formality: int = 3,
        condition: str = "good",
        added: str = "2025-01-01",
        last_worn: str | None = None,
        **extra: Any,
    ) -> ClosetItem:
        return ClosetItem(
            id=item_id,
            category=category,
            subcategory=subcategory,
            colors=colors or ["white"],
            pattern=pattern,
            fabric=fabric,
            warmth=warmth,
            formality=formality,
            condition=condition,
            date_added=date.fromisoformat(added),
            last_worn=date.fromisoformat(last_worn) if last_worn else None,
            **extra,
        )

    return _make


@pytest.fixture
def make_candidate() -> Callable[..., PurchaseCandidate]:
    def _make(
        candidate_id: str,
        category: str,
        *,
        subcategory: str = "thing",
        colors: list[str] | None = None,
        pattern: str = "solid",
        fabric: str = "cotton",
        warmth: int = 2,
        formality: int = 3,
        price: float | None = None,
        **extra: Any,
    ) -> PurchaseCandidate:
        return PurchaseCandidate(
            candidate_id=candidate_id,
            category=category,
            subcategory=subcategory,
            colors=colors or ["white"],
            pattern=pattern,
            fabric=fabric,
            warmth=warmth,
            formality=formality,
            price=price,
            **extra,
        )

    return _make


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings pointed entirely at a temp directory, so no test touches real data."""
    return Settings(
        closet_path=tmp_path / "closet.json",
        compatibility_cache_path=tmp_path / "compatibility.json",
        reports_dir=tmp_path / "runs",
        seed_closet_path=SEED_CLOSET,
        default_candidates_path=SEED_CANDIDATES,
    )


@pytest.fixture
def seed_closet() -> Closet:
    """The bundled 15-item demo closet."""
    return Closet.load(SEED_CLOSET)


@pytest.fixture
def forecast_payload() -> dict[str, Any]:
    """A saved Open-Meteo daily response: mild, showery, breezy, 11C swing."""
    return {
        "latitude": 52.52,
        "longitude": 13.41,
        "timezone": "Europe/Berlin",
        "daily": {
            "time": ["2026-09-05"],
            "weather_code": [80],
            "temperature_2m_max": [18.4],
            "temperature_2m_min": [7.1],
            "precipitation_sum": [1.8],
            "precipitation_probability_max": [45],
            "wind_speed_10m_max": [24.0],
        },
    }

"""Runtime settings for wardrobe-agents.

Everything tunable lives here rather than being hardcoded at a call site.
Each field can be overridden by an environment variable (``WARDROBE_*``), so
the same code runs against different models, data locations, and thresholds
without edits.

The Claude API key is deliberately *not* a field: it is read from the
environment by the Anthropic SDK at call time and never stored or logged.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Repository root: <root>/config/settings.py -> <root>
ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot."""

    # ---- Claude API -----------------------------------------------------
    claude_model: str = field(default_factory=lambda: _env_str("WARDROBE_CLAUDE_MODEL", "claude-opus-5"))
    """Model id used for every agent call. Opus 5 is the default."""

    claude_vision_model: str = field(
        default_factory=lambda: _env_str("WARDROBE_CLAUDE_VISION_MODEL", "")
    )
    """Optional override for photo extraction; falls back to ``claude_model``."""

    llm_effort: str = field(default_factory=lambda: _env_str("WARDROBE_LLM_EFFORT", "high"))
    """``output_config.effort``: low | medium | high | xhigh | max."""

    llm_max_tokens: int = field(default_factory=lambda: _env_int("WARDROBE_LLM_MAX_TOKENS", 16000))

    llm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("WARDROBE_LLM_TIMEOUT_SECONDS", 180.0)
    )

    max_concurrent_llm_calls: int = field(
        default_factory=lambda: _env_int("WARDROBE_MAX_CONCURRENT_LLM_CALLS", 4)
    )
    """Bounds fan-out for photo batches and any judge-style stylist calls."""

    api_key_env_var: str = field(
        default_factory=lambda: _env_str("WARDROBE_API_KEY_ENV_VAR", "ANTHROPIC_API_KEY")
    )
    """Name of the env var holding the key. The key itself is never stored here."""

    # ---- Weather --------------------------------------------------------
    weather_api_base_url: str = field(
        default_factory=lambda: _env_str("WARDROBE_WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1/forecast")
    )
    geocoding_api_base_url: str = field(
        default_factory=lambda: _env_str(
            "WARDROBE_GEOCODING_API_BASE_URL", "https://geocoding-api.open-meteo.com/v1/search"
        )
    )
    weather_timeout_seconds: float = field(
        default_factory=lambda: _env_float("WARDROBE_WEATHER_TIMEOUT_SECONDS", 15.0)
    )

    # ---- Data locations -------------------------------------------------
    closet_path: Path = field(
        default_factory=lambda: _env_path("WARDROBE_CLOSET_PATH", ROOT_DIR / "data" / "closet.json")
    )
    """Source of truth. Both stages read (and only stage 1 cataloguing writes) here."""

    compatibility_cache_path: Path = field(
        default_factory=lambda: _env_path(
            "WARDROBE_COMPATIBILITY_CACHE_PATH", ROOT_DIR / "data" / "compatibility.json"
        )
    )
    """Derived artifact. Never merged back into closet.json - computed fields stay out
    of the source-of-truth item records."""

    reports_dir: Path = field(
        default_factory=lambda: _env_path("WARDROBE_REPORTS_DIR", ROOT_DIR / "reports" / "runs")
    )

    seed_closet_path: Path = field(
        default_factory=lambda: _env_path(
            "WARDROBE_SEED_CLOSET_PATH", ROOT_DIR / "examples" / "seed_closet" / "closet.json"
        )
    )

    default_candidates_path: Path = field(
        default_factory=lambda: _env_path(
            "WARDROBE_CANDIDATES_PATH", ROOT_DIR / "examples" / "candidate_purchases.json"
        )
    )

    # ---- Compatibility / enumeration ------------------------------------
    compatibility_threshold: float = field(
        default_factory=lambda: _env_float("WARDROBE_COMPATIBILITY_THRESHOLD", 0.55)
    )
    """Minimum pairwise score for two items to be considered wearable together."""

    max_formality_gap: int = field(default_factory=lambda: _env_int("WARDROBE_MAX_FORMALITY_GAP", 3))
    """Formality distance at or above which a pair is hard-rejected regardless of score."""

    max_outfits_enumerated: int = field(
        default_factory=lambda: _env_int("WARDROBE_MAX_OUTFITS_ENUMERATED", 50000)
    )
    """Safety cap so enumeration on a large closet stays bounded."""

    require_shoes: bool = field(
        default_factory=lambda: _env_str("WARDROBE_REQUIRE_SHOES", "true").lower() in ("1", "true", "yes")
    )
    """When true, a valid outfit must include shoes if the closet contains any."""

    # ---- Purchase optimization ------------------------------------------
    marginal_gain_threshold: int = field(
        default_factory=lambda: _env_int("WARDROBE_MARGINAL_GAIN_THRESHOLD", 3)
    )
    """A candidate unlocking fewer than this many new outfits is reported as redundant."""

    suggest_buy_top_n: int = field(default_factory=lambda: _env_int("WARDROBE_SUGGEST_BUY_TOP_N", 5))

    # ---- Stylist --------------------------------------------------------
    stylist_suggestion_count: int = field(
        default_factory=lambda: _env_int("WARDROBE_STYLIST_SUGGESTION_COUNT", 3)
    )
    stylist_max_items_in_prompt: int = field(
        default_factory=lambda: _env_int("WARDROBE_STYLIST_MAX_ITEMS_IN_PROMPT", 80)
    )
    stylist_max_retries: int = field(
        default_factory=lambda: _env_int("WARDROBE_STYLIST_MAX_RETRIES", 1)
    )
    """Extra calls allowed to replace outfits that were dropped as invalid. 0 disables."""

    def vision_model(self) -> str:
        """Model used for photo extraction."""
        return self.claude_vision_model or self.claude_model

    def api_key(self) -> str | None:
        """Read the Claude API key from the environment. Never cached, never logged."""
        key = os.environ.get(self.api_key_env_var)
        return key or None

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot, for embedding in run reports."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out


settings = Settings()
"""Process-wide default. Construct a fresh ``Settings()`` in tests to override."""

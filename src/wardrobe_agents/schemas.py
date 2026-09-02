"""Typed data model shared by both stages.

Layering rule that keeps the two stages loosely coupled:

* :class:`ClosetItem` is the **source of truth**. It is ``extra="forbid"``, so no
  computed field can be silently written back onto an item record.
* Everything the compatibility layer derives (:class:`CompatibilityEdge`,
  :class:`EnumeratedOutfit`, :class:`PurchaseRecommendation`) references items by
  ``id`` and is persisted separately.

Models whose names end in ``...Response`` / ``...Extraction`` are the exact shapes
Claude is asked to return; they are validated by ``llm.py`` before anything else in
the codebase sees them.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Scales
# --------------------------------------------------------------------------- #

Formality = Annotated[int, Field(ge=1, le=5)]
"""1 loungewear, 2 casual, 3 smart casual, 4 business/dressy, 5 formal."""

Warmth = Annotated[int, Field(ge=0, le=5)]
"""0 hot-weather only, 1 light, 2 mild, 3 cool, 4 cold, 5 freezing."""

FORMALITY_LABELS: dict[int, str] = {
    1: "loungewear",
    2: "casual",
    3: "smart casual",
    4: "business / dressy",
    5: "formal",
}

WARMTH_LABELS: dict[int, str] = {
    0: "hot weather only",
    1: "light",
    2: "mild",
    3: "cool",
    4: "cold",
    5: "freezing",
}


class Category(str, Enum):
    """Structural slot an item occupies in an outfit."""

    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTER = "outer"
    SHOES = "shoes"
    ACCESSORY = "accessory"


class Condition(str, Enum):
    NEW = "new"
    EXCELLENT = "excellent"
    GOOD = "good"
    WORN = "worn"
    RETIRE = "retire"


class TempBand(str, Enum):
    FREEZING = "freezing"
    COLD = "cold"
    COOL = "cool"
    MILD = "mild"
    WARM = "warm"
    HOT = "hot"


# --------------------------------------------------------------------------- #
# Source of truth
# --------------------------------------------------------------------------- #


class ClosetItem(BaseModel):
    """One garment. The only record that is persisted as user-owned data.

    ``extra="forbid"`` is load-bearing: it is what stops the compatibility layer
    from decorating item records with computed fields.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str = Field(description="Stable, human-readable id, e.g. 'top-003'.")
    category: Category
    subcategory: str = Field(description="Free text, e.g. 'oxford shirt', 'chelsea boot'.")
    colors: list[str] = Field(min_length=1, description="Lowercase color names, primary first.")
    pattern: str = Field(default="solid", description="solid, striped, plaid, floral, print, ...")
    fabric: str = Field(description="cotton, wool, linen, denim, leather, ...")
    warmth: Warmth
    formality: Formality
    condition: Condition = Condition.GOOD
    date_added: date
    last_worn: date | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: Literal["manual", "photo", "seed"] = "manual"
    photo_path: str | None = None

    @field_validator("colors", mode="before")
    @classmethod
    def _normalize_colors(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [str(c).strip().lower() for c in v if str(c).strip()]
        return v

    @field_validator("subcategory", "fabric", "pattern")
    @classmethod
    def _normalize_text(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def label(self) -> str:
        """Short human description used in prompts, CLI output, and explanations."""
        return f"{'/'.join(self.colors)} {self.pattern + ' ' if self.pattern != 'solid' else ''}{self.subcategory}"

    def describe(self) -> str:
        """One-line description with the attributes an LLM needs to reason."""
        return (
            f"{self.id}: {self.label} | {self.category.value} | {self.fabric} | "
            f"warmth {self.warmth} ({WARMTH_LABELS[self.warmth]}) | "
            f"formality {self.formality} ({FORMALITY_LABELS[self.formality]})"
        )


# --------------------------------------------------------------------------- #
# Stage 1: weather
# --------------------------------------------------------------------------- #


class DailyForecast(BaseModel):
    """Deterministic Open-Meteo output. No model reasoning has touched this yet."""

    model_config = ConfigDict(extra="forbid")

    date: date
    location: str
    latitude: float
    longitude: float
    temp_min_c: float
    temp_max_c: float
    precipitation_mm: float
    precipitation_probability_pct: int
    wind_kph: float
    weather_code: int
    conditions: str = Field(description="Human-readable WMO weather-code description.")


class WeatherTranslation(BaseModel):
    """The judgment half of the weather agent - what Claude is asked to decide.

    Only the genuinely lossy calls live here; the numeric bands come from rules.
    """

    model_config = ConfigDict(extra="forbid")

    needs_waterproof_outer: bool
    needs_windproof: bool
    layering_advice: str = Field(description="One or two sentences on how to layer today.")
    prefer_fabrics: list[str] = Field(description="Fabrics that suit the conditions.")
    avoid_fabrics: list[str] = Field(description="Fabrics to avoid today, may be empty.")
    warmth_adjustment: int = Field(
        ge=-1, le=1, description="-1 dress lighter than the band, 0 as-is, +1 dress warmer."
    )
    notes: str = Field(description="Anything a rule table would have lost.")


class WeatherConstraints(BaseModel):
    """What the stylist actually reasons over. Produced by rules, optionally refined
    by Claude when the conditions are ambiguous."""

    model_config = ConfigDict(extra="forbid")

    date: date
    location: str
    temp_min_c: float
    temp_max_c: float
    temp_band: TempBand
    precipitation_mm: float
    precipitation_probability_pct: int
    wind_kph: float
    conditions: str
    min_warmth: Warmth
    max_warmth: Warmth
    needs_waterproof_outer: bool = False
    needs_windproof: bool = False
    prefer_fabrics: list[str] = Field(default_factory=list)
    avoid_fabrics: list[str] = Field(default_factory=list)
    layering_advice: str = ""
    notes: str = ""
    source: Literal["rules", "rules+llm", "manual"] = "rules"

    def summary(self) -> str:
        return (
            f"{self.location} on {self.date.isoformat()}: {self.temp_min_c:.0f}-{self.temp_max_c:.0f}C "
            f"({self.temp_band.value}), {self.conditions}, "
            f"{self.precipitation_probability_pct}% precip, wind {self.wind_kph:.0f} kph"
        )


# --------------------------------------------------------------------------- #
# Stage 1: stylist
# --------------------------------------------------------------------------- #


class StylistPick(BaseModel):
    """One outfit exactly as Claude returns it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short evocative name for the look.")
    item_ids: list[str] = Field(min_length=1, description="Ids drawn from the provided closet.")
    rationale: str = Field(description="Why these pieces work together, and why today.")
    weather_fit: str = Field(description="How the outfit handles the forecast.")
    formality: Formality
    confidence: float = Field(ge=0.0, le=1.0)


class StylistResponse(BaseModel):
    """Top-level stylist tool payload."""

    model_config = ConfigDict(extra="forbid")

    outfits: list[StylistPick] = Field(min_length=1)
    overall_notes: str = Field(description="Cross-cutting observations about today's options.")


class OutfitSuggestion(BaseModel):
    """A stylist pick after validation against the real closet."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    name: str
    item_ids: list[str]
    items: list[ClosetItem]
    rationale: str
    weather_fit: str
    formality: Formality
    confidence: float = Field(ge=0.0, le=1.0)

    def describe(self) -> str:
        return f"{self.rank}. {self.name} - " + ", ".join(i.label for i in self.items)


# --------------------------------------------------------------------------- #
# Stage 1: cataloguing
# --------------------------------------------------------------------------- #


class ItemExtraction(BaseModel):
    """Attributes Claude infers from a photo. Deliberately has no ``id`` or
    ``date_added`` - identity and provenance are assigned by the closet store,
    not by the model."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    subcategory: str
    colors: list[str] = Field(min_length=1)
    pattern: str
    fabric: str
    warmth: Warmth
    formality: Formality
    condition: Condition
    notes: str = Field(description="Anything notable: fit, wear, distinguishing details.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the overall extraction.")
    uncertain_fields: list[str] = Field(
        description="Field names the user should double-check. May be empty."
    )


# --------------------------------------------------------------------------- #
# Stage 2: compatibility, enumeration, purchase optimization
# --------------------------------------------------------------------------- #

EdgeKind = Literal["styleable", "slot_conflict", "formality_conflict", "color_conflict"]


class CompatibilityEdge(BaseModel):
    """A scored pair. ``item_a``/``item_b`` are always sorted, so an edge has one
    canonical representation."""

    model_config = ConfigDict(extra="forbid")

    item_a: str
    item_b: str
    score: float = Field(ge=0.0, le=1.0)
    compatible: bool
    kind: EdgeKind
    breakdown: dict[str, float] = Field(
        default_factory=dict, description="Per-component sub-scores, for explainability."
    )
    reasons: list[str] = Field(default_factory=list)

    @property
    def pair(self) -> tuple[str, str]:
        return (self.item_a, self.item_b)


class EnumeratedOutfit(BaseModel):
    """A structurally valid, mutually compatible combination."""

    model_config = ConfigDict(extra="forbid")

    item_ids: list[str]
    score: float = Field(ge=0.0, le=1.0, description="Mean pairwise compatibility.")

    @property
    def key(self) -> frozenset[str]:
        return frozenset(self.item_ids)


class PurchaseCandidate(BaseModel):
    """An item the user does *not* own yet. Same attributes as a ClosetItem minus
    ownership metadata, so it can be projected into the closet for scoring."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    category: Category
    subcategory: str
    colors: list[str] = Field(min_length=1)
    pattern: str = "solid"
    fabric: str
    warmth: Warmth
    formality: Formality
    price: float | None = None
    url: str | None = None
    notes: str | None = None

    @field_validator("colors", mode="before")
    @classmethod
    def _normalize_colors(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            return [str(c).strip().lower() for c in v if str(c).strip()]
        return v

    def as_closet_item(self, *, item_id: str | None = None, today: date | None = None) -> ClosetItem:
        """Project into a ClosetItem so the *same* scorer and enumerator can run on it.

        This is what keeps stage 2 honest: a candidate is evaluated by exactly the
        machinery that scores owned items, not a parallel code path.
        """
        return ClosetItem(
            id=item_id or f"candidate::{self.candidate_id}",
            category=self.category,
            subcategory=self.subcategory,
            colors=list(self.colors),
            pattern=self.pattern,
            fabric=self.fabric,
            warmth=self.warmth,
            formality=self.formality,
            condition=Condition.NEW,
            date_added=today or date.today(),
            notes=self.notes,
            source="manual",
        )

    @property
    def label(self) -> str:
        return f"{'/'.join(self.colors)} {self.pattern + ' ' if self.pattern != 'solid' else ''}{self.subcategory}"


class BridgedPair(BaseModel):
    """Two owned items that could not be worn together in any outfit before this
    candidate, and can now. This is the concrete evidence behind 'bridges'."""

    model_config = ConfigDict(extra="forbid")

    item_a: str
    item_b: str
    label_a: str
    label_b: str


class PurchaseRecommendation(BaseModel):
    """A scored candidate, framed as marginal coverage gain."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    candidate: PurchaseCandidate
    baseline_outfit_count: int = Field(description="Distinct looks the closet supports today.")
    new_outfit_count: int = Field(
        description="Distinct NEW looks this candidate unlocks - the marginal gain."
    )
    new_combination_count: int = Field(
        default=0,
        description=(
            "Raw item-set combinations containing this candidate. Always >= new_outfit_count; "
            "the gap is how much of its apparent gain is duplicated style."
        ),
    )
    total_outfit_count: int
    marginal_gain_pct: float = Field(description="New looks as a % of the baseline look count.")
    bridged_pairs: list[BridgedPair] = Field(default_factory=list)
    top_partners: list[str] = Field(
        default_factory=list, description="Owned item ids this candidate pairs with most often."
    )
    redundant_with: list[str] = Field(
        default_factory=list, description="Owned items that already fill this slot and style."
    )
    verdict: Literal["recommended", "marginal", "redundant", "not_scored"] = "marginal"
    """``not_scored`` marks a candidate outside the enumeration model (accessories),
    where a zero is an absence of measurement rather than a verdict of redundancy."""
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Run reports
# --------------------------------------------------------------------------- #


class ClosetStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_count: int
    by_category: dict[str, int]
    compatible_pair_count: int
    total_pair_count: int
    outfit_count: int = Field(description="Valid item-set combinations.")
    distinct_look_count: int = Field(
        default=0, description="Combinations collapsed by style signature - true outfit variety."
    )
    outfit_count_truncated: bool = False


class RecommendRunReport(BaseModel):
    """Stage 1 artifact written to reports/runs/."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage: Literal["recommend"] = "recommend"
    created_at: datetime
    closet_path: str
    closet_item_count: int
    constraints: WeatherConstraints
    occasion: str | None = None
    target_formality: int | None = None
    suggestions: list[OutfitSuggestion]
    overall_notes: str = ""
    model: str = ""


class SuggestBuyRunReport(BaseModel):
    """Stage 2 artifact written to reports/runs/."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage: Literal["suggest-buy"] = "suggest-buy"
    created_at: datetime
    closet_path: str
    candidates_path: str | None = None
    stats: ClosetStats
    recommendations: list[PurchaseRecommendation]
    marginal_gain_threshold: int

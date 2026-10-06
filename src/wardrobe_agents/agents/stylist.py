"""Stylist agent: the genuinely LLM-backed step.

Everything the stylist does is judgment a rule engine handles badly - whether a
rust knit reads well against olive trousers, whether a denim jacket over a
dress is right for a 12C swing, whether "dinner, smart casual" is better served
by leaning up or down from the closet's median formality. So this agent does not
score, filter, or template the answer: it hands Claude the eligible closet and
the day's constraints, and asks for ranked outfits with reasoning.

What it does do is guard the boundary. The model may only reference ids that were
actually sent; picks referencing anything else are dropped rather than
hallucinated into a recommendation. So are picks that are not an outfit at all
(two tops, no bottom, two coats): the prompt states those rules, but only this
module enforces them. The weather agent's waterproof/windproof flags are checked
against item tags and reported as warnings when unmet. When a drop leaves the answer short, the stylist is asked
again for replacements, told what was rejected and why. Deterministic pre-filtering (retired items,
wildly out-of-season pieces) happens before the call to keep the prompt focused,
with a documented fallback so the filter can never starve the stylist.

This module deliberately knows nothing about the compatibility graph. Stage 1
reasons about outfits; stage 2 counts them. They share the closet, not the code.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from config.settings import Settings, settings as default_settings
from wardrobe_agents.agents.base import BaseAgent
from wardrobe_agents.llm import LLMError, StructuredLLM
from wardrobe_agents.schemas import (
    Category,
    ClosetItem,
    Condition,
    FORMALITY_LABELS,
    OutfitSuggestion,
    StylistResponse,
    WeatherConstraints,
)

STYLIST_SYSTEM = """You are a personal stylist choosing outfits from one specific
person's real wardrobe.

You will be given the day's weather constraints and the items available. Return
ranked outfits, best first.

Hard rules:
- Use ONLY item ids from the provided list. Never invent an id or an item.
- Each outfit is either (one top + one bottom) or (one dress).
- At most one outer layer. Include shoes when suitable shoes are available.
- Accessories are optional; add one only when it genuinely improves the look.
- Respect the warmth window. An item outside it needs a reason in the rationale
  (a light layer under a coat is fine; a linen shirt on a freezing day is not).

What actually matters in your reasoning: colour relationships (what harmonises,
what deliberately contrasts, what clashes), whether the formality of the pieces
agrees with itself and with the occasion, whether the layering works physically
for the temperature range, and pattern balance.

Each outfit needs a distinct point of view - three variations on the same shirt
and trousers is one suggestion, not three. Write the rationale in specific terms
("the rust knit picks up the warm tone in the brown boots"), not generic praise.
If the closet cannot cover the occasion or the weather well, say so plainly in
overall_notes rather than overselling a compromise."""


class StylistError(RuntimeError):
    """The stylist could not produce a usable recommendation."""


@dataclass(slots=True)
class StylistRequest:
    """Input to :class:`StylistAgent`."""

    items: Sequence[ClosetItem]
    constraints: WeatherConstraints
    occasion: str | None = None
    target_formality: int | None = None
    count: int | None = None


@dataclass(slots=True)
class StylistResult:
    suggestions: list[OutfitSuggestion] = field(default_factory=list)
    overall_notes: str = ""
    considered_item_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def filter_for_weather(
    items: Sequence[ClosetItem], constraints: WeatherConstraints, *, tolerance: int = 1
) -> list[ClosetItem]:
    """Drop retired items and pieces far outside the day's warmth window.

    ``tolerance`` keeps genuine layering pieces in play: a warmth-4 coat is still
    eligible on a warmth 2-3 day. If the filter would leave too little to work
    with, it is abandoned and the full wearable closet is returned - a
    pre-filter must never be the reason the stylist has nothing to say.
    """
    wearable = [i for i in items if i.condition != Condition.RETIRE]
    low = constraints.min_warmth - tolerance
    high = constraints.max_warmth + tolerance
    kept = [i for i in wearable if low <= i.warmth <= high]

    categories = {i.category for i in kept}
    has_core = "top" in {c.value for c in categories} and "bottom" in {c.value for c in categories}
    has_dress = "dress" in {c.value for c in categories}
    if len(kept) < 4 or not (has_core or has_dress):
        return wearable
    return kept


def _cap(items: Sequence[ClosetItem], limit: int) -> list[ClosetItem]:
    """Keep the prompt bounded, preferring items that have not been worn lately."""
    if len(items) <= limit:
        return list(items)
    return sorted(items, key=lambda i: (i.last_worn is not None, i.last_worn or i.date_added))[:limit]


def structure_problem(items: Sequence[ClosetItem]) -> str | None:
    """Why these items are not an outfit, or ``None`` if they are.

    Mirrors the hard rules in :data:`STYLIST_SYSTEM`: (one top + one bottom) or
    (one dress), at most one outer layer, at most one pair of shoes. Shoes and
    accessories stay optional - whether to include them is the stylist's call.
    """
    counts = {category: 0 for category in Category}
    for item in items:
        counts[item.category] += 1

    tops, bottoms, dresses = counts[Category.TOP], counts[Category.BOTTOM], counts[Category.DRESS]
    if dresses:
        if dresses > 1:
            return f"{dresses} dresses"
        if tops or bottoms:
            return "a dress combined with a top or bottom"
    elif tops != 1 or bottoms != 1:
        return f"needs one top and one bottom, or one dress (got {tops} top, {bottoms} bottom)"

    if counts[Category.OUTER] > 1:
        return f"{counts[Category.OUTER]} outer layers"
    if counts[Category.SHOES] > 1:
        return f"{counts[Category.SHOES]} pairs of shoes"
    return None


def protection_warnings(
    suggestions: Sequence[OutfitSuggestion],
    eligible: Sequence[ClosetItem],
    constraints: WeatherConstraints,
) -> list[str]:
    """Check the weather agent's protection flags against what the closet records.

    Protection lives in item tags (``waterproof`` / ``windproof``). These are
    warnings, not drops: an umbrella is a legitimate answer to rain, but a
    constraint nothing in the closet can meet should never pass silently.
    """
    warnings: list[str] = []
    needed = [
        tag
        for tag, flag in (
            ("waterproof", constraints.needs_waterproof_outer),
            ("windproof", constraints.needs_windproof),
        )
        if flag
    ]
    for tag in needed:
        if not any(i.category == Category.OUTER and i.has_tag(tag) for i in eligible):
            warnings.append(
                f"The forecast calls for a {tag} outer layer, but no outer item is tagged "
                f"{tag!r} - tag one in the closet if you own it."
            )
            continue
        for suggestion in suggestions:
            if not any(i.category == Category.OUTER and i.has_tag(tag) for i in suggestion.items):
                warnings.append(f"Outfit {suggestion.name!r} has no {tag} outer layer.")
    return warnings


class StylistAgent(BaseAgent[StylistRequest, StylistResult]):
    """Closet + constraints -> ranked :class:`OutfitSuggestion` list."""

    name = "stylist"

    def __init__(self, llm: StructuredLLM | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.llm = llm or StructuredLLM(self.settings)

    def run(self, payload: StylistRequest) -> StylistResult:
        eligible = _cap(
            filter_for_weather(payload.items, payload.constraints),
            self.settings.stylist_max_items_in_prompt,
        )
        if not eligible:
            raise StylistError(
                "The closet has no wearable items. Add some with `wardrobe add` first."
            )

        count = payload.count or self.settings.stylist_suggestion_count
        index = {item.id: item for item in eligible}
        prompt = self._prompt(eligible, payload, count)
        result = StylistResult(considered_item_ids=list(index))

        response = self.llm.call(
            system=STYLIST_SYSTEM,
            content=prompt,
            response_model=StylistResponse,
            purpose="choosing outfits from the closet",
        )
        result.overall_notes = response.overall_notes
        dropped = self._accept(response, index, result, limit=count)

        # Retry only to replace what was dropped. A stylist that chose to return
        # fewer outfits is making a judgment, not a mistake.
        for _ in range(self.settings.stylist_max_retries):
            missing = count - len(result.suggestions)
            if not dropped or missing <= 0:
                break
            try:
                response = self.llm.call(
                    system=STYLIST_SYSTEM,
                    content=prompt + self._retry_note(result, dropped, missing),
                    response_model=StylistResponse,
                    purpose="replacing outfits that were not valid",
                )
            except LLMError as exc:
                # The outfits already accepted are still a good answer.
                result.warnings.append(f"Could not replace the dropped outfit(s): {exc}")
                break
            dropped = self._accept(response, index, result, limit=count)

        if not result.suggestions:
            raise StylistError(
                "The stylist returned no usable outfits - every suggestion referenced items "
                "that are not in the closet or was not a valid outfit. " + " ".join(result.warnings)
            )
        result.warnings.extend(protection_warnings(result.suggestions, eligible, payload.constraints))
        return result

    # ---- prompt ---------------------------------------------------------

    @staticmethod
    def _prompt(items: Sequence[ClosetItem], payload: StylistRequest, count: int) -> str:
        c = payload.constraints
        lines = [
            "## Today's conditions",
            c.summary(),
            f"Warmth window: items rated {c.min_warmth}-{c.max_warmth} on a 0-5 scale.",
        ]
        if c.layering_advice:
            lines.append(f"Layering: {c.layering_advice}")
        if c.needs_waterproof_outer:
            lines.append("A waterproof outer layer is needed (items tagged 'waterproof').")
        if c.needs_windproof:
            lines.append("Wind is a factor; a windproof layer helps (items tagged 'windproof').")
        if c.prefer_fabrics:
            lines.append(f"Fabrics that suit today: {', '.join(c.prefer_fabrics)}.")
        if c.avoid_fabrics:
            lines.append(f"Fabrics to avoid today: {', '.join(c.avoid_fabrics)}.")
        if c.notes:
            lines.append(f"Notes: {c.notes}")

        lines.append("")
        lines.append("## Occasion")
        lines.append(payload.occasion or "No specific occasion given - everyday wear.")
        if payload.target_formality:
            label = FORMALITY_LABELS.get(payload.target_formality, "")
            lines.append(f"Target formality: {payload.target_formality} ({label}).")

        lines.append("")
        lines.append("## Available items")
        by_category: dict[str, list[ClosetItem]] = {}
        for item in items:
            by_category.setdefault(item.category.value, []).append(item)
        for category in ("top", "bottom", "dress", "outer", "shoes", "accessory"):
            group = by_category.get(category)
            if not group:
                continue
            lines.append(f"\n### {category}")
            for item in group:
                worn = f", last worn {item.last_worn.isoformat()}" if item.last_worn else ", never worn"
                note = f" ({item.notes})" if item.notes else ""
                tags = f" | tags: {', '.join(item.tags)}" if item.tags else ""
                lines.append(
                    f"- {item.describe()} | condition {item.condition.value}{worn}{tags}{note}"
                )

        lines.append("")
        lines.append(f"Return {count} ranked outfits, best first.")
        return "\n".join(lines)

    @staticmethod
    def _retry_note(result: StylistResult, dropped: Sequence[str], missing: int) -> str:
        lines = ["", "", "## Correction", "Some outfits in your previous answer were rejected:"]
        lines += [f"- {reason}" for reason in dropped]
        if result.suggestions:
            lines.append("These were accepted - do not repeat them:")
            lines += [f"- {s.name}: {', '.join(s.item_ids)}" for s in result.suggestions]
        lines.append(f"Return {missing} more outfit(s) that follow the hard rules.")
        return "\n".join(lines)

    # ---- response validation --------------------------------------------

    @staticmethod
    def _accept(
        response: StylistResponse,
        index: dict[str, ClosetItem],
        result: StylistResult,
        *,
        limit: int,
    ) -> list[str]:
        """Append the valid picks to ``result`` (up to ``limit`` suggestions in
        total) and return the reasons for the ones dropped, which are also
        recorded as warnings."""
        dropped: list[str] = []
        seen = {frozenset(s.item_ids) for s in result.suggestions}

        for pick in response.outfits:
            if len(result.suggestions) >= limit:
                break
            unknown = [i for i in pick.item_ids if i not in index]
            if unknown:
                dropped.append(
                    f"Dropped outfit {pick.name!r}: references unknown item(s) {', '.join(unknown)}."
                )
                continue
            deduped = list(dict.fromkeys(pick.item_ids))
            problem = structure_problem([index[i] for i in deduped])
            if problem:
                dropped.append(f"Dropped outfit {pick.name!r}: not a valid outfit - {problem}.")
                continue
            if frozenset(deduped) in seen:
                dropped.append(f"Dropped outfit {pick.name!r}: repeats an outfit already suggested.")
                continue
            seen.add(frozenset(deduped))
            result.suggestions.append(
                OutfitSuggestion(
                    rank=len(result.suggestions) + 1,
                    name=pick.name,
                    item_ids=deduped,
                    items=[index[i] for i in deduped],
                    rationale=pick.rationale,
                    weather_fit=pick.weather_fit,
                    formality=pick.formality,
                    confidence=pick.confidence,
                )
            )

        result.warnings.extend(dropped)
        return dropped

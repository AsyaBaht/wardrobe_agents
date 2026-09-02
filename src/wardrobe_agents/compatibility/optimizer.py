"""Purchase optimization: what to buy next, framed as marginal coverage gain.

This is a coverage problem, not a black box. For each candidate the optimizer:

1. Projects the candidate into a :class:`~wardrobe_agents.schemas.ClosetItem` and
   scores it against the real closet with the *same* scorer that scored the
   closet - no parallel code path, so a candidate cannot be flattered by a
   different rulebook.
2. Enumerates the outfits containing it, then keeps only those whose *style
   signature* the closet cannot already produce. That last step is what makes
   the number mean something: any new bottom multiplies the combination count,
   so a second pair of the jeans you own would otherwise look like an enormous
   win. Collapsing outfits by look - slot, formality, weight, pattern, colour
   family - leaves the genuine **marginal gain**, and the raw combination count
   is reported alongside it so the discount is visible rather than hidden.
3. Diffs the pairs that co-occur in those new outfits against the pairs that
   already co-occur somewhere in the baseline. What is left is the set of
   **bridged pairs**: items that previously had nothing to wear together and now
   do. This is what makes a versatile item score above a merely nice one.
4. Checks the candidate against what the closet already holds, so a fourth pair
   of blue jeans is reported as redundant rather than as an incremental win.

Every number in the resulting :class:`~wardrobe_agents.schemas.PurchaseRecommendation`
traces back to enumerated outfits you can print, which is the point: the CLI can
explain *why* something scores well.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from config.settings import Settings, settings as default_settings
from wardrobe_agents.compatibility.enumeration import (
    EnumerationResult,
    enumerate_outfits,
    outfit_signature,
)
from wardrobe_agents.compatibility.scoring import (
    CompatibilityGraph,
    color_family,
    score_closet,
    score_item_against,
)
from wardrobe_agents.schemas import (
    BridgedPair,
    Category,
    ClosetItem,
    ClosetStats,
    PurchaseCandidate,
    PurchaseRecommendation,
)

CANDIDATE_PREFIX = "candidate::"


@dataclass(slots=True)
class OptimizationResult:
    """Everything a ``suggest-buy`` run produced."""

    recommendations: list[PurchaseRecommendation] = field(default_factory=list)
    baseline: EnumerationResult = field(default_factory=EnumerationResult)
    stats: ClosetStats | None = None
    graph: CompatibilityGraph | None = None


def build_stats(
    items: Sequence[ClosetItem], graph: CompatibilityGraph, baseline: EnumerationResult
) -> ClosetStats:
    counts: Counter[str] = Counter(i.category.value for i in items)
    total_pairs = len(items) * (len(items) - 1) // 2
    index = {i.id: i for i in items}
    return ClosetStats(
        item_count=len(items),
        by_category={c.value: counts.get(c.value, 0) for c in Category},
        compatible_pair_count=len(graph.compatible_edges()),
        total_pair_count=total_pairs,
        outfit_count=baseline.count,
        distinct_look_count=len(baseline.signatures(index)),
        outfit_count_truncated=baseline.truncated,
    )


def _colors_overlap(a: Sequence[str], b: Sequence[str]) -> bool:
    """Do two items read as the same colour story?"""
    for color_a in a:
        info_a = color_family(color_a)
        for color_b in b:
            info_b = color_family(color_b)
            if color_a == color_b:
                return True
            if info_a is None or info_b is None:
                continue
            if info_a.neutral and info_b.neutral:
                return True
            if info_a.hue is not None and info_b.hue is not None:
                delta = abs(info_a.hue - info_b.hue)
                if min(delta, 360 - delta) <= 25:
                    return True
    return False


def find_redundancies(
    candidate: PurchaseCandidate, items: Sequence[ClosetItem]
) -> list[ClosetItem]:
    """Owned items that already fill this slot, formality, and colour story."""
    return [
        item
        for item in items
        if item.category == candidate.category
        and abs(item.formality - candidate.formality) <= 1
        and abs(item.warmth - candidate.warmth) <= 1
        and _colors_overlap(item.colors, candidate.colors)
    ]


def _explain(
    candidate: PurchaseCandidate,
    *,
    new_count: int,
    combination_count: int,
    baseline_count: int,
    gain_pct: float,
    bridged: Sequence[BridgedPair],
    partner_labels: Sequence[str],
    redundant: Sequence[ClosetItem],
    partner_count: int,
    threshold: int,
) -> str:
    """Plain-language account of the score, built from the same numbers.

    Every clause here is derived from an enumerated outfit, so anything the CLI
    prints can be traced back to a combination you could list.
    """
    if new_count == 0:
        if partner_count == 0:
            return (
                f"Unlocks nothing: at formality {candidate.formality} in "
                f"{'/'.join(candidate.colors)}, it does not pair with a single item you own."
            )
        if candidate.category is Category.ACCESSORY:
            return (
                "Accessories are not counted in outfit enumeration, so this candidate has no "
                "marginal outfit score. Judge it on styling, not coverage."
            )
        if combination_count > 0:
            owned = f" It duplicates your {redundant[0].label}." if redundant else ""
            return (
                f"No new looks. It slots into {combination_count} combination"
                f"{'s' if combination_count != 1 else ''}, but every one of them is a look you "
                f"can already assemble from what you own.{owned}"
            )
        missing = {
            Category.TOP: "a bottom",
            Category.BOTTOM: "a top",
            Category.OUTER: "a complete outfit",
            Category.SHOES: "a complete outfit",
        }.get(candidate.category, "a partner piece")
        return (
            f"Unlocks no new outfits: it pairs with {partner_count} item(s), but never with "
            f"{missing} it could complete."
        )

    parts = [
        f"Unlocks {new_count} new look{'s' if new_count != 1 else ''} "
        f"(+{gain_pct:.0f}% on the {baseline_count} your closet supports today)."
    ]
    if combination_count > new_count:
        parts.append(
            f"({combination_count} raw combinations, of which {combination_count - new_count} "
            f"repeat a look you already have.)"
        )
    if bridged:
        example = bridged[0]
        parts.append(
            f"Bridges {len(bridged)} pair{'s' if len(bridged) != 1 else ''} that had nothing "
            f"to wear together - for instance your {example.label_a} and your {example.label_b}."
        )
    if partner_labels:
        parts.append(f"It leans most on your {', '.join(partner_labels)}.")
    if redundant:
        labels = ", ".join(i.label for i in redundant[:2])
        parts.append(f"Caveat: it overlaps with the {labels} you already own.")
    elif new_count < threshold:
        parts.append(f"Below the {threshold}-look bar for a confident recommendation.")
    return " ".join(parts)


def evaluate_candidate(
    candidate: PurchaseCandidate,
    items: Sequence[ClosetItem],
    graph: CompatibilityGraph,
    baseline: EnumerationResult,
    settings: Settings | None = None,
    *,
    baseline_pairs: set[frozenset[str]] | None = None,
    baseline_signatures: set[tuple[tuple[str, ...], ...]] | None = None,
) -> PurchaseRecommendation:
    """Score one candidate by marginal new-look count. ``rank`` is assigned later."""
    settings = settings or default_settings
    index: dict[str, ClosetItem] = {i.id: i for i in items}
    baseline_pairs = baseline.co_occurring_pairs() if baseline_pairs is None else baseline_pairs
    baseline_signatures = (
        baseline.signatures(index) if baseline_signatures is None else baseline_signatures
    )

    projected = candidate.as_closet_item(item_id=f"{CANDIDATE_PREFIX}{candidate.candidate_id}")
    new_edges = score_item_against(projected, items, settings)
    extended_graph = graph.merged_with(new_edges, [projected.id])
    extended_items = [*items, projected]
    extended_index = {**index, projected.id: projected}

    combinations_with_candidate = enumerate_outfits(
        extended_items, extended_graph, settings, must_include=projected.id
    )
    # Keep only the combinations that are a look the closet cannot already produce.
    novel = [
        outfit
        for outfit in combinations_with_candidate
        if outfit_signature(outfit.item_ids, extended_index) not in baseline_signatures
    ]
    new_count = len(novel)
    combination_count = combinations_with_candidate.count

    labels = {item.id: item.label for item in items}

    # Which owned pairs became wearable together only because of this candidate?
    bridged: list[BridgedPair] = []
    seen: set[frozenset[str]] = set()
    for outfit in novel:
        owned = [i for i in outfit.item_ids if i != projected.id]
        for a, b in combinations(sorted(owned), 2):
            key = frozenset((a, b))
            if key in baseline_pairs or key in seen:
                continue
            seen.add(key)
            bridged.append(BridgedPair(item_a=a, item_b=b, label_a=labels[a], label_b=labels[b]))

    partner_counts = Counter(
        item_id for outfit in novel for item_id in outfit.item_ids if item_id != projected.id
    )
    top_partners = [item_id for item_id, _ in partner_counts.most_common(3)]

    redundant = find_redundancies(candidate, items)
    baseline_count = len(baseline_signatures)
    gain_pct = (new_count / baseline_count * 100.0) if baseline_count else 0.0

    if candidate.category is Category.ACCESSORY:
        # Accessories are not a unit of outfit count, so a 0 here means "not
        # measured by this model", not "adds nothing".
        verdict = "not_scored"
    elif new_count >= settings.marginal_gain_threshold:
        verdict = "recommended"
    elif new_count == 0 or redundant:
        verdict = "redundant"
    else:
        verdict = "marginal"

    return PurchaseRecommendation(
        rank=1,
        candidate=candidate,
        baseline_outfit_count=baseline_count,
        new_outfit_count=new_count,
        new_combination_count=combination_count,
        total_outfit_count=baseline_count + new_count,
        marginal_gain_pct=round(gain_pct, 2),
        bridged_pairs=bridged,
        top_partners=top_partners,
        redundant_with=[i.id for i in redundant],
        verdict=verdict,
        explanation=_explain(
            candidate,
            new_count=new_count,
            combination_count=combination_count,
            baseline_count=baseline_count,
            gain_pct=gain_pct,
            bridged=bridged,
            partner_labels=[labels[p] for p in top_partners],
            redundant=redundant,
            partner_count=len([e for e in new_edges if e.compatible]),
            threshold=settings.marginal_gain_threshold,
        ),
    )


def optimize_purchases(
    items: Sequence[ClosetItem],
    candidates: Sequence[PurchaseCandidate],
    settings: Settings | None = None,
    *,
    graph: CompatibilityGraph | None = None,
) -> OptimizationResult:
    """Rank candidates by how many new outfits each would unlock."""
    settings = settings or default_settings
    graph = graph or score_closet(items, settings)
    baseline = enumerate_outfits(items, graph, settings)
    baseline_pairs = baseline.co_occurring_pairs()
    baseline_signatures = baseline.signatures({i.id: i for i in items})

    scored = [
        evaluate_candidate(
            candidate,
            items,
            graph,
            baseline,
            settings,
            baseline_pairs=baseline_pairs,
            baseline_signatures=baseline_signatures,
        )
        for candidate in candidates
    ]

    # Most new outfits first; then the item that bridges more gaps; then the
    # less redundant one; then the cheaper one.
    scored.sort(
        key=lambda r: (
            r.verdict == "not_scored",
            -r.new_outfit_count,
            -len(r.bridged_pairs),
            len(r.redundant_with),
            r.candidate.price if r.candidate.price is not None else float("inf"),
            r.candidate.candidate_id,
        )
    )
    ranked = [r.model_copy(update={"rank": index}) for index, r in enumerate(scored, start=1)]

    return OptimizationResult(
        recommendations=ranked,
        baseline=baseline,
        stats=build_stats(items, graph, baseline),
        graph=graph,
    )

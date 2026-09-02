"""Enumerate the valid outfits a closet supports.

An outfit is a *clique* in the compatibility graph that also satisfies the
structural rules:

- exactly one top **and** one bottom, or exactly one dress;
- at most one outer layer;
- shoes when the closet has any (``settings.require_shoes``);
- every pair of chosen items must be compatible - not just compatible with the
  core, but with each other, so the coat that works over the shirt but fights
  the boots does not silently become an outfit.

Accessories are deliberately excluded. Counting "shirt + jeans" and
"shirt + jeans + scarf" as two distinct outfits would inflate every number in
stage 2 by a factor of the accessory drawer, and the question this layer answers
- "how many genuinely different things can I wear?" - is about the clothes.
The stylist still uses accessories in stage 1, where they are a styling choice
rather than a unit of count.

``must_include`` is the hook the optimizer needs: outfits containing a specific
item are exactly the outfits a candidate purchase would unlock, so the marginal
gain of a candidate is computable without re-enumerating the whole closet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from config.settings import Settings, settings as default_settings
from wardrobe_agents.compatibility.scoring import CompatibilityGraph, style_signature
from wardrobe_agents.schemas import Category, ClosetItem, Condition, EnumeratedOutfit


@dataclass(slots=True)
class EnumerationResult:
    """All valid outfits found, plus whether the search hit its safety cap."""

    outfits: list[EnumeratedOutfit] = field(default_factory=list)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.outfits)

    def __iter__(self):
        return iter(self.outfits)

    @property
    def count(self) -> int:
        return len(self.outfits)

    def containing(self, item_id: str) -> list[EnumeratedOutfit]:
        return [o for o in self.outfits if item_id in o.item_ids]

    def signatures(self, index: dict[str, ClosetItem]) -> set[tuple[tuple[str, ...], ...]]:
        """The distinct *looks* these outfits represent.

        Two outfits collapse to one signature when every piece is interchangeable
        in style terms - same slot, formality, weight, pattern and colour family.
        Counting looks rather than item-sets is what stops "buy a second pair of
        the jeans you own" from registering as an enormous gain.
        """
        return {outfit_signature(o.item_ids, index) for o in self.outfits}

    def co_occurring_pairs(self) -> set[frozenset[str]]:
        """Every pair of items that appears together in at least one outfit.

        The optimizer diffs this before and after a candidate to find the pairs
        the candidate *bridges*.
        """
        pairs: set[frozenset[str]] = set()
        for outfit in self.outfits:
            for a, b in combinations(outfit.item_ids, 2):
                pairs.add(frozenset((a, b)))
        return pairs


def outfit_signature(
    item_ids: Sequence[str], index: dict[str, ClosetItem]
) -> tuple[tuple[str, ...], ...]:
    """Style identity of a whole outfit: the sorted signatures of its pieces."""
    return tuple(sorted(style_signature(index[i]) for i in item_ids))


def _mean_pairwise_score(graph: CompatibilityGraph, item_ids: Sequence[str]) -> float:
    pairs = list(combinations(item_ids, 2))
    if not pairs:
        return 0.0
    return round(sum(graph.score(a, b) for a, b in pairs) / len(pairs), 4)


def enumerate_outfits(
    items: Sequence[ClosetItem],
    graph: CompatibilityGraph,
    settings: Settings | None = None,
    *,
    must_include: str | None = None,
    include_retired: bool = False,
) -> EnumerationResult:
    """Enumerate every structurally valid, mutually compatible outfit.

    Pass ``must_include`` to restrict the search to outfits containing that item.
    """
    settings = settings or default_settings
    pool = [i for i in items if include_retired or i.condition != Condition.RETIRE]

    by_category: dict[Category, list[ClosetItem]] = {c: [] for c in Category}
    for item in pool:
        by_category[item.category].append(item)

    tops = by_category[Category.TOP]
    bottoms = by_category[Category.BOTTOM]
    dresses = by_category[Category.DRESS]
    outers = by_category[Category.OUTER]
    shoes = by_category[Category.SHOES]

    forced: ClosetItem | None = None
    if must_include is not None:
        matches = [i for i in pool if i.id == must_include]
        if not matches:
            return EnumerationResult()
        forced = matches[0]
        # Pin the forced item's slot; every outfit must contain it.
        if forced.category is Category.TOP:
            tops, dresses = [forced], []
        elif forced.category is Category.BOTTOM:
            bottoms, dresses = [forced], []
        elif forced.category is Category.DRESS:
            dresses, tops, bottoms = [forced], [], []
        elif forced.category is Category.OUTER:
            outers = [forced]
        elif forced.category is Category.SHOES:
            shoes = [forced]
        else:
            # Accessories are not part of an enumerated outfit; nothing to count.
            return EnumerationResult()

    # Cores: (top, bottom) pairs that work, plus every dress on its own.
    cores: list[list[ClosetItem]] = [[dress] for dress in dresses]
    for top in tops:
        for bottom in bottoms:
            if graph.compatible(top.id, bottom.id):
                cores.append([top, bottom])

    shoes_required = settings.require_shoes and bool(by_category[Category.SHOES])
    outer_options: list[ClosetItem | None] = [*outers] if forced and forced.category is Category.OUTER else [None, *outers]
    shoe_options: list[ClosetItem | None] = list(shoes) if shoes_required or (forced and forced.category is Category.SHOES) else [None, *shoes]

    result = EnumerationResult()
    limit = settings.max_outfits_enumerated

    for core in cores:
        core_ids = [i.id for i in core]

        # Prune once per core rather than re-testing inside the inner loop.
        viable_outers = [
            o for o in outer_options if o is None or all(graph.compatible(o.id, cid) for cid in core_ids)
        ]
        viable_shoes = [
            s for s in shoe_options if s is None or all(graph.compatible(s.id, cid) for cid in core_ids)
        ]

        for outer in viable_outers:
            for shoe in viable_shoes:
                if outer is not None and shoe is not None and not graph.compatible(outer.id, shoe.id):
                    continue
                chosen = core + [x for x in (outer, shoe) if x is not None]
                item_ids = sorted(i.id for i in chosen)
                result.outfits.append(
                    EnumeratedOutfit(item_ids=item_ids, score=_mean_pairwise_score(graph, item_ids))
                )
                if len(result.outfits) >= limit:
                    result.truncated = True
                    return result

    result.outfits.sort(key=lambda o: (-o.score, o.item_ids))
    return result


def outfit_count(
    items: Sequence[ClosetItem], graph: CompatibilityGraph, settings: Settings | None = None
) -> int:
    """How many distinct valid outfits the closet currently supports."""
    return enumerate_outfits(items, graph, settings).count

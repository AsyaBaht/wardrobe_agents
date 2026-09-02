"""Pairwise compatibility scoring - the substrate both stages read from.

Every pair of items in the closet gets a :class:`~wardrobe_agents.schemas.CompatibilityEdge`
carrying a score, a verdict, a per-component breakdown, and human-readable reasons.
Together the edges form a :class:`CompatibilityGraph`: outfit enumeration looks for
cliques in it, and purchase optimization asks how much a new node would add to it.

The score has three weighted components and one modifier:

===============  ======  ==================================================
component        weight  what it captures
===============  ======  ==================================================
colour harmony    0.40   neutrals, monochrome, analogous, complementary, clash
formality          0.35   how far apart the two pieces sit on the 1-5 scale
pattern            0.25   solid/solid, solid/patterned, pattern-on-pattern
warmth coherence   x      multiplier penalising e.g. linen with a parka
===============  ======  ==================================================

Three gates run before or alongside scoring, because they are absolute rather
than matters of degree: two items in the same slot (two tops) cannot be worn
together at all; a formality gap of 3+ (gym shorts with a tuxedo jacket) is not a
low score but a non-outfit; and an outright colour clash is a "no" rather than a
deduction. Without that last gate the weighted score could not express it - two
solids at equal formality score 0.72 on the strength of the other components
alone, so colour would never be able to fail a pair by itself.

This is deliberately a transparent rule model rather than an LLM call. It runs
over every pair in the closet - O(n^2), thousands of pairs, re-run for every
purchase candidate - so it must be fast, free, and above all *stable*: the
optimizer compares outfit counts before and after adding an item, and that
comparison is meaningless if the scorer's answers drift between runs.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.settings import Settings, settings as default_settings
from wardrobe_agents.schemas import Category, ClosetItem, CompatibilityEdge

# --------------------------------------------------------------------------- #
# Colour model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ColorInfo:
    """A colour's position on the wheel, or its status as a neutral."""

    hue: float | None
    neutral: bool = False


#: Colours that coordinate with (almost) anything.
_NEUTRALS = (
    "white", "off-white", "offwhite", "ivory", "cream", "ecru", "oatmeal", "stone",
    "black", "grey", "gray", "charcoal", "slate", "silver",
    "beige", "tan", "camel", "khaki", "taupe", "sand",
    "brown", "chocolate", "chestnut", "cognac",
    "navy", "denim", "indigo", "olive", "gold", "bronze",
)

#: Chromatic colours by approximate hue angle.
_HUES: dict[str, float] = {
    "red": 0, "crimson": 355, "burgundy": 350, "maroon": 350, "wine": 345,
    "rust": 20, "terracotta": 18, "coral": 12, "orange": 30, "amber": 40,
    "mustard": 50, "yellow": 55, "chartreuse": 80, "lime": 90,
    "green": 120, "emerald": 150, "sage": 110, "mint": 155, "forest": 130,
    "teal": 175, "turquoise": 180, "aqua": 185,
    "sky": 200, "cyan": 190, "blue": 220, "cobalt": 225, "royal": 230, "periwinkle": 245,
    "purple": 280, "violet": 275, "lavender": 275, "lilac": 285, "plum": 300,
    "magenta": 310, "fuchsia": 320, "pink": 330, "blush": 340, "rose": 345,
}

COLORS: dict[str, ColorInfo] = {name: ColorInfo(hue=None, neutral=True) for name in _NEUTRALS}
COLORS.update({name: ColorInfo(hue=hue) for name, hue in _HUES.items()})


def color_family(name: str) -> ColorInfo | None:
    """Look up a colour, tolerating compounds like ``"dark olive"`` or ``"navy blue"``.

    Returns ``None`` for a colour the model does not know, which callers treat as
    "no opinion" rather than "clash".
    """
    key = name.strip().lower()
    if key in COLORS:
        return COLORS[key]

    # "dark olive green" -> try each word, longest known match wins.
    words = key.replace("-", " ").split()
    for word in reversed(words):
        if word in COLORS:
            return COLORS[word]
    return None


def _color_pair_score(a: str, b: str) -> tuple[float, str]:
    """Score one colour against one colour."""
    info_a, info_b = color_family(a), color_family(b)
    if info_a is None or info_b is None:
        return 0.55, f"unfamiliar colour ({a} / {b}) - scored neutrally"

    if info_a.neutral and info_b.neutral:
        return 0.92, f"{a} and {b} are both neutrals"
    if info_a.neutral or info_b.neutral:
        neutral, other = (a, b) if info_a.neutral else (b, a)
        return 0.88, f"{neutral} is a neutral and grounds the {other}"

    assert info_a.hue is not None and info_b.hue is not None
    delta = abs(info_a.hue - info_b.hue)
    distance = min(delta, 360 - delta)

    # `distance` is a wheel distance, so it is always in [0, 180].
    if distance <= 20:
        return 0.78, f"{a} and {b} are near-monochrome"
    if distance <= 55:
        return 0.68, f"{a} and {b} are analogous"
    if distance < 100:
        return 0.30, f"{a} and {b} clash"
    if distance < 150:
        return 0.50, f"{a} and {b} are a wide, tricky interval"
    return 0.72, f"{a} and {b} are complementary - a deliberate contrast"


def color_harmony(item_a: ClosetItem, item_b: ClosetItem) -> tuple[float, str]:
    """Best colour relationship available between two multi-colour items.

    Best-of rather than average: one shared or complementary colour is enough to
    tie an outfit together, which is how people actually read a pairing.
    """
    best_score, best_reason = 0.0, ""
    for color_a in item_a.colors:
        for color_b in item_b.colors:
            score, reason = _color_pair_score(color_a, color_b)
            if score > best_score:
                best_score, best_reason = score, reason
    return best_score, best_reason


# --------------------------------------------------------------------------- #
# Fabric families
# --------------------------------------------------------------------------- #

#: Fabrics grouped by how they read in an outfit. A wool knit and a cotton shirt
#: are different looks even at the same colour and formality; slim jeans and
#: straight jeans are not.
_FABRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "denim": ("denim", "chambray"),
    "wool": ("wool", "merino", "cashmere", "tweed", "flannel", "alpaca", "mohair", "felt"),
    "knit": ("knit", "jersey", "rib", "ribbed"),
    "cotton": ("cotton", "poplin", "oxford", "canvas", "twill", "chino", "corduroy", "cord"),
    "linen": ("linen", "ramie", "hemp"),
    "silk": ("silk", "satin", "crepe", "chiffon", "velvet"),
    "leather": ("leather", "suede", "nubuck", "shearling"),
    "technical": ("polyester", "nylon", "acrylic", "fleece", "gore-tex", "goretex", "down", "puffer", "technical", "rubber"),
}


def fabric_family(fabric: str) -> str:
    """Normalise a free-text fabric to a family, e.g. "merino wool" -> "wool"."""
    key = fabric.strip().lower()
    for family, members in _FABRIC_FAMILIES.items():
        if any(member in key for member in members):
            return family
    return key or "unknown"


# --------------------------------------------------------------------------- #
# Structural rules
# --------------------------------------------------------------------------- #

#: Category pairs that can coexist in one outfit. A blazer needs a bottom, not
#: another blazer.
_COMPATIBLE_SLOTS: frozenset[frozenset[Category]] = frozenset(
    frozenset(pair)
    for pair in (
        (Category.TOP, Category.BOTTOM),
        (Category.TOP, Category.OUTER),
        (Category.TOP, Category.SHOES),
        (Category.TOP, Category.ACCESSORY),
        (Category.BOTTOM, Category.OUTER),
        (Category.BOTTOM, Category.SHOES),
        (Category.BOTTOM, Category.ACCESSORY),
        (Category.DRESS, Category.OUTER),
        (Category.DRESS, Category.SHOES),
        (Category.DRESS, Category.ACCESSORY),
        (Category.OUTER, Category.SHOES),
        (Category.OUTER, Category.ACCESSORY),
        (Category.SHOES, Category.ACCESSORY),
        (Category.ACCESSORY, Category.ACCESSORY),
    )
)

#: Colour score at or below which a pair is rejected outright rather than
#: merely marked down. Set between "clash" (0.30) and "wide interval" (0.50).
CLASH_THRESHOLD = 0.35

WEIGHT_COLOR = 0.40
WEIGHT_FORMALITY = 0.35
WEIGHT_PATTERN = 0.25


def slots_compatible(a: Category, b: Category) -> bool:
    """Can these two categories appear in the same outfit at all?"""
    return frozenset((a, b)) in _COMPATIBLE_SLOTS


def formality_alignment(item_a: ClosetItem, item_b: ClosetItem) -> tuple[float, str]:
    gap = abs(item_a.formality - item_b.formality)
    score = max(0.0, 1.0 - 0.3 * gap)
    if gap == 0:
        reason = "identical formality"
    elif gap == 1:
        reason = "one step apart in formality - fine"
    else:
        reason = f"{gap} steps apart in formality"
    return score, reason


def pattern_balance(item_a: ClosetItem, item_b: ClosetItem) -> tuple[float, str]:
    a_solid = item_a.pattern == "solid"
    b_solid = item_b.pattern == "solid"
    if a_solid and b_solid:
        return 1.0, "both solid"
    if a_solid or b_solid:
        patterned = item_a if not a_solid else item_b
        return 0.90, f"one {patterned.pattern} piece against a solid"
    if item_a.pattern == item_b.pattern:
        return 0.45, f"two {item_a.pattern} pieces compete"
    return 0.35, f"{item_a.pattern} against {item_b.pattern} is a hard mix to carry"


def warmth_coherence(item_a: ClosetItem, item_b: ClosetItem) -> tuple[float, str]:
    """Multiplier: pieces built for very different seasons read as a mistake."""
    gap = abs(item_a.warmth - item_b.warmth)
    if gap <= 2:
        return 1.0, ""
    return max(0.5, 1.0 - 0.12 * (gap - 2)), f"{gap}-step warmth mismatch between the two"


# --------------------------------------------------------------------------- #
# Edge construction
# --------------------------------------------------------------------------- #


def score_pair(
    item_a: ClosetItem, item_b: ClosetItem, settings: Settings | None = None
) -> CompatibilityEdge:
    """Score one pair. Items are ordered by id so the edge is canonical."""
    settings = settings or default_settings
    if item_a.id > item_b.id:
        item_a, item_b = item_b, item_a

    # Gate 1: structural. Two tops are not a bad outfit, they are not an outfit.
    if not slots_compatible(item_a.category, item_b.category):
        return CompatibilityEdge(
            item_a=item_a.id,
            item_b=item_b.id,
            score=0.0,
            compatible=False,
            kind="slot_conflict",
            breakdown={},
            reasons=[
                f"both occupy the {item_a.category.value} slot"
                if item_a.category == item_b.category
                else f"a {item_a.category.value} and a {item_b.category.value} cannot be worn together"
            ],
        )

    # Gate 2: formality. A gap this wide is a category error, not a low score.
    formality_gap = abs(item_a.formality - item_b.formality)
    if formality_gap >= settings.max_formality_gap:
        return CompatibilityEdge(
            item_a=item_a.id,
            item_b=item_b.id,
            score=0.0,
            compatible=False,
            kind="formality_conflict",
            breakdown={"formality": 0.0},
            reasons=[f"{formality_gap}-step formality gap is too wide to bridge"],
        )

    color_score, color_reason = color_harmony(item_a, item_b)

    # Gate 3: colour. Nothing these two own in common reads together.
    if color_score <= CLASH_THRESHOLD:
        return CompatibilityEdge(
            item_a=item_a.id,
            item_b=item_b.id,
            score=round(color_score, 4),
            compatible=False,
            kind="color_conflict",
            breakdown={"color": round(color_score, 4)},
            reasons=[color_reason],
        )

    formality_score, formality_reason = formality_alignment(item_a, item_b)
    pattern_score, pattern_reason = pattern_balance(item_a, item_b)
    warmth_factor, warmth_reason = warmth_coherence(item_a, item_b)

    raw = (
        WEIGHT_COLOR * color_score
        + WEIGHT_FORMALITY * formality_score
        + WEIGHT_PATTERN * pattern_score
    )
    score = round(raw * warmth_factor, 4)

    reasons = [color_reason, formality_reason, pattern_reason]
    if warmth_reason:
        reasons.append(warmth_reason)

    return CompatibilityEdge(
        item_a=item_a.id,
        item_b=item_b.id,
        score=score,
        compatible=score >= settings.compatibility_threshold,
        kind="styleable",
        breakdown={
            "color": round(color_score, 4),
            "formality": round(formality_score, 4),
            "pattern": round(pattern_score, 4),
            "warmth_factor": round(warmth_factor, 4),
        },
        reasons=[r for r in reasons if r],
    )


class CompatibilityGraph:
    """An undirected, edge-weighted graph over closet items."""

    def __init__(
        self,
        edges: Iterable[CompatibilityEdge],
        item_ids: Iterable[str],
        threshold: float,
    ) -> None:
        self.threshold = threshold
        self.item_ids: list[str] = sorted(item_ids)
        self._edges: dict[tuple[str, str], CompatibilityEdge] = {}
        for edge in edges:
            self._edges[self._key(edge.item_a, edge.item_b)] = edge

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    # ---- lookups --------------------------------------------------------

    def edge(self, a: str, b: str) -> CompatibilityEdge | None:
        return self._edges.get(self._key(a, b))

    def score(self, a: str, b: str) -> float:
        edge = self.edge(a, b)
        return edge.score if edge else 0.0

    def compatible(self, a: str, b: str) -> bool:
        edge = self.edge(a, b)
        return bool(edge and edge.compatible)

    def all_compatible(self, item_ids: Sequence[str]) -> bool:
        """True when every pair in the group is compatible - a clique test."""
        return all(self.compatible(a, b) for a, b in combinations(item_ids, 2))

    def partners(self, item_id: str) -> list[str]:
        """Ids this item can actually be worn with."""
        return sorted(
            other
            for other in self.item_ids
            if other != item_id and self.compatible(item_id, other)
        )

    @property
    def edges(self) -> list[CompatibilityEdge]:
        return sorted(self._edges.values(), key=lambda e: (e.item_a, e.item_b))

    def compatible_edges(self) -> list[CompatibilityEdge]:
        return [e for e in self.edges if e.compatible]

    def __len__(self) -> int:
        return len(self._edges)

    # ---- derivation -----------------------------------------------------

    def merged_with(
        self, new_edges: Sequence[CompatibilityEdge], new_item_ids: Sequence[str]
    ) -> "CompatibilityGraph":
        """A copy extended with a projected item's edges.

        Used by the optimizer to ask "what would the graph look like if I owned
        this?" without mutating the real graph.
        """
        return CompatibilityGraph(
            list(self._edges.values()) + list(new_edges),
            list(self.item_ids) + list(new_item_ids),
            self.threshold,
        )

    # ---- persistence ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "threshold": self.threshold,
            "item_ids": self.item_ids,
            "edges": [e.model_dump(mode="json") for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompatibilityGraph":
        return cls(
            [CompatibilityEdge.model_validate(e) for e in data.get("edges", [])],
            data.get("item_ids", []),
            float(data.get("threshold", default_settings.compatibility_threshold)),
        )

    def save(self, path: Path) -> Path:
        """Persist the derived graph *beside* the closet, never inside it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "CompatibilityGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def style_signature(item: ClosetItem) -> tuple[str, ...]:
    """The item's *style identity*, ignoring which particular garment it is.

    Two items with the same signature are interchangeable in an outfit: a second
    pair of indigo jeans at the same weight and formality makes no look possible
    that the first pair did not. Outfit enumeration uses this to count distinct
    *looks* rather than distinct item-sets, which is what "outfit variety"
    actually means - and it is what lets the optimizer tell a genuinely additive
    purchase from a duplicate.

    Colour collapses to family (neutral, or a 30-degree hue bucket), warmth to a
    three-way weight, and fabric to a family, because those are the resolutions at
    which a difference is visible in an outfit. Fabric is what keeps the signature
    honest in both directions: a cotton oxford and a merino crewneck at the same
    colour and formality are genuinely different looks, while slim jeans and
    straight jeans are not.
    """
    warmth_bucket = "light" if item.warmth <= 1 else ("mid" if item.warmth <= 3 else "warm")

    color_keys = set()
    for color in item.colors:
        info = color_family(color)
        if info is None:
            color_keys.add(color)
        elif info.neutral:
            color_keys.add("neutral")
        else:
            assert info.hue is not None
            color_keys.add(f"hue{int(info.hue) // 30}")

    return (
        item.category.value,
        f"f{item.formality}",
        warmth_bucket,
        item.pattern,
        "+".join(sorted(color_keys)),
        fabric_family(item.fabric),
    )


def score_closet(
    items: Sequence[ClosetItem], settings: Settings | None = None
) -> CompatibilityGraph:
    """Score every pair in the closet into a graph."""
    settings = settings or default_settings
    edges = [score_pair(a, b, settings) for a, b in combinations(items, 2)]
    return CompatibilityGraph(edges, (i.id for i in items), settings.compatibility_threshold)


def score_item_against(
    item: ClosetItem, others: Sequence[ClosetItem], settings: Settings | None = None
) -> list[CompatibilityEdge]:
    """Score one item against many - the incremental path used for candidates."""
    settings = settings or default_settings
    return [score_pair(item, other, settings) for other in others if other.id != item.id]

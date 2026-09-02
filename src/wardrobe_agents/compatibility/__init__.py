"""Stage 2: the compatibility graph, outfit enumeration, and purchase optimization.

Nothing in this package imports from :mod:`wardrobe_agents.agents`, and nothing in
``agents`` imports from here. Both read :class:`~wardrobe_agents.schemas.ClosetItem`.

Author: Anastasiia Bakhtoiarova
"""

from wardrobe_agents.compatibility.enumeration import (
    EnumerationResult,
    enumerate_outfits,
    outfit_count,
)
from wardrobe_agents.compatibility.optimizer import OptimizationResult, optimize_purchases
from wardrobe_agents.compatibility.scoring import (
    CompatibilityGraph,
    color_family,
    fabric_family,
    score_closet,
    score_item_against,
    score_pair,
    style_signature,
)

__all__ = [
    "CompatibilityGraph",
    "EnumerationResult",
    "OptimizationResult",
    "color_family",
    "enumerate_outfits",
    "fabric_family",
    "optimize_purchases",
    "outfit_count",
    "score_closet",
    "score_item_against",
    "score_pair",
    "style_signature",
]

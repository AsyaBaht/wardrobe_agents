"""Outfit enumeration: the structural rules, the clique requirement, and the
``must_include`` hook stage 2 depends on.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from wardrobe_agents.compatibility.enumeration import (
    enumerate_outfits,
    outfit_count,
    outfit_signature,
)
from wardrobe_agents.compatibility.scoring import score_closet


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def basic(make_item):
    """Two tops, two bottoms, one dress, one outer, one pair of shoes - all neutral,
    all mutually compatible, so the counting rules are the only thing under test."""
    return [
        make_item("top-001", "top", colors=["white"], formality=3, warmth=2),
        make_item("top-002", "top", colors=["grey"], formality=3, warmth=2),
        make_item("bottom-001", "bottom", colors=["navy"], formality=3, warmth=2),
        make_item("bottom-002", "bottom", colors=["black"], formality=3, warmth=2),
        make_item("dress-001", "dress", colors=["black"], formality=3, warmth=2),
        make_item("outer-001", "outer", colors=["camel"], formality=3, warmth=3),
        make_item("shoes-001", "shoes", colors=["brown"], formality=3, warmth=2),
    ]


def _graph(items, settings):
    return score_closet(items, settings)


def test_counts_match_the_structural_rules(basic, settings):
    """(2 tops x 2 bottoms + 1 dress) x (no outer | outer) = 10, shoes required."""
    result = enumerate_outfits(basic, _graph(basic, settings), settings)
    assert result.count == 10
    assert not result.truncated


def test_every_outfit_has_one_core(basic, settings):
    for outfit in enumerate_outfits(basic, _graph(basic, settings), settings):
        ids = set(outfit.item_ids)
        tops = len(ids & {"top-001", "top-002"})
        bottoms = len(ids & {"bottom-001", "bottom-002"})
        dresses = len(ids & {"dress-001"})

        assert (tops == 1 and bottoms == 1 and dresses == 0) or (
            tops == 0 and bottoms == 0 and dresses == 1
        ), f"malformed core in {sorted(ids)}"


def test_at_most_one_outer_layer(make_item, settings):
    items = [
        make_item("top-001", "top", colors=["white"]),
        make_item("bottom-001", "bottom", colors=["navy"]),
        make_item("shoes-001", "shoes", colors=["brown"]),
        make_item("outer-001", "outer", colors=["camel"]),
        make_item("outer-002", "outer", colors=["black"]),
    ]
    for outfit in enumerate_outfits(items, _graph(items, settings), settings):
        assert len(set(outfit.item_ids) & {"outer-001", "outer-002"}) <= 1


def test_shoes_are_required_when_the_closet_has_any(basic, settings):
    for outfit in enumerate_outfits(basic, _graph(basic, settings), settings):
        assert "shoes-001" in outfit.item_ids


def test_shoes_are_optional_when_the_closet_has_none(make_item, settings):
    items = [make_item("top-001", "top", colors=["white"]), make_item("bottom-001", "bottom", colors=["navy"])]
    assert enumerate_outfits(items, _graph(items, settings), settings).count == 1


def test_require_shoes_is_configurable(basic):
    lenient = Settings(require_shoes=False)
    result = enumerate_outfits(basic, _graph(basic, lenient), lenient)
    assert result.count == 20, "each outfit now also has a shoeless variant"


def test_accessories_are_not_part_of_an_outfit(basic, make_item, settings):
    """Counting scarf variants would multiply every stage-2 number by the drawer."""
    with_scarf = [*basic, make_item("accessory-001", "accessory", colors=["grey"])]
    assert outfit_count(with_scarf, _graph(with_scarf, settings), settings) == outfit_count(
        basic, _graph(basic, settings), settings
    )


def test_retired_items_are_excluded(basic, make_item, settings):
    items = [*basic, make_item("top-003", "top", colors=["white"], condition="retire")]
    for outfit in enumerate_outfits(items, _graph(items, settings), settings):
        assert "top-003" not in outfit.item_ids


# --------------------------------------------------------------------------- #
# The clique requirement
# --------------------------------------------------------------------------- #


def test_an_outfit_requires_every_pair_to_work_not_just_the_core(make_item, settings):
    """The coat that suits the shirt but fights the boots is not an outfit."""
    items = [
        make_item("top-001", "top", colors=["white"], formality=3),
        make_item("bottom-001", "bottom", colors=["navy"], formality=3),
        make_item("shoes-001", "shoes", colors=["white"], formality=2),
        make_item("outer-001", "outer", colors=["black"], formality=5),
    ]
    graph = _graph(items, settings)

    assert graph.compatible("outer-001", "top-001"), "the coat suits the shirt"
    assert not graph.compatible("outer-001", "shoes-001"), "but not the sneakers"

    for outfit in enumerate_outfits(items, graph, settings):
        ids = set(outfit.item_ids)
        assert not {"outer-001", "shoes-001"} <= ids


def test_incompatible_core_pairs_produce_no_outfit(make_item, settings):
    items = [
        make_item("top-001", "top", colors=["red"], formality=3),
        make_item("bottom-001", "bottom", colors=["lime"], formality=3),
        make_item("shoes-001", "shoes", colors=["white"], formality=3),
    ]
    assert enumerate_outfits(items, _graph(items, settings), settings).count == 0


def test_outfit_score_is_the_mean_pairwise_compatibility(basic, settings):
    from itertools import combinations

    graph = _graph(basic, settings)
    outfit = enumerate_outfits(basic, graph, settings).outfits[0]
    pairs = list(combinations(outfit.item_ids, 2))
    expected = sum(graph.score(a, b) for a, b in pairs) / len(pairs)

    assert outfit.score == pytest.approx(expected, abs=0.001)


def test_results_are_ordered_best_first(basic, settings):
    scores = [o.score for o in enumerate_outfits(basic, _graph(basic, settings), settings)]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# must_include - the hook the optimizer needs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("item_id", ["top-001", "bottom-001", "dress-001", "outer-001", "shoes-001"])
def test_must_include_returns_exactly_the_outfits_containing_that_item(basic, settings, item_id):
    graph = _graph(basic, settings)
    everything = enumerate_outfits(basic, graph, settings)
    restricted = enumerate_outfits(basic, graph, settings, must_include=item_id)

    assert {tuple(o.item_ids) for o in restricted} == {
        tuple(o.item_ids) for o in everything.containing(item_id)
    }


def test_must_include_on_an_unknown_item_returns_nothing(basic, settings):
    assert enumerate_outfits(basic, _graph(basic, settings), settings, must_include="nope").count == 0


def test_must_include_on_an_accessory_returns_nothing(basic, make_item, settings):
    items = [*basic, make_item("accessory-001", "accessory", colors=["grey"])]
    result = enumerate_outfits(items, _graph(items, settings), settings, must_include="accessory-001")
    assert result.count == 0


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #


def test_co_occurring_pairs_reports_what_can_be_worn_together(basic, settings):
    pairs = enumerate_outfits(basic, _graph(basic, settings), settings).co_occurring_pairs()

    assert frozenset({"top-001", "bottom-001"}) in pairs
    assert frozenset({"top-001", "top-002"}) not in pairs, "two tops never share an outfit"
    assert frozenset({"top-001", "dress-001"}) not in pairs


def test_signatures_collapse_interchangeable_outfits(make_item, settings):
    """Two identical-in-style tops yield two combinations but one look."""
    items = [
        make_item("top-001", "top", colors=["white"], fabric="cotton", formality=3, warmth=2),
        make_item("top-002", "top", colors=["cream"], fabric="cotton", formality=3, warmth=2),
        make_item("bottom-001", "bottom", colors=["navy"], formality=3, warmth=2),
        make_item("shoes-001", "shoes", colors=["brown"], formality=3, warmth=2),
    ]
    result = enumerate_outfits(items, _graph(items, settings), settings)
    index = {i.id: i for i in items}

    assert result.count == 2
    assert len(result.signatures(index)) == 1


def test_enumeration_is_capped(basic):
    capped = Settings(max_outfits_enumerated=3)
    result = enumerate_outfits(basic, _graph(basic, capped), capped)

    assert result.count == 3
    assert result.truncated


def test_outfit_signature_is_order_independent(basic, settings):
    index = {i.id: i for i in basic}
    ids = ["top-001", "bottom-001", "shoes-001"]
    assert outfit_signature(ids, index) == outfit_signature(list(reversed(ids)), index)

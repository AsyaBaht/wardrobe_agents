"""Pairwise scoring: the three structural gates, the weighted components, and the
graph that both stages read from."""

from __future__ import annotations

import pytest

from config.settings import Settings
from wardrobe_agents.compatibility.scoring import (
    CompatibilityGraph,
    color_family,
    fabric_family,
    score_closet,
    score_item_against,
    score_pair,
    slots_compatible,
    style_signature,
)
from wardrobe_agents.schemas import Category


@pytest.fixture
def settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_two_items_in_the_same_slot_cannot_be_worn_together(make_item, settings):
    edge = score_pair(make_item("top-001", "top"), make_item("top-002", "top"), settings)

    assert edge.kind == "slot_conflict"
    assert not edge.compatible
    assert edge.score == 0.0


def test_a_dress_and_a_bottom_conflict(make_item, settings):
    edge = score_pair(make_item("dress-001", "dress"), make_item("bottom-001", "bottom"), settings)
    assert edge.kind == "slot_conflict"


def test_a_blazer_needs_a_bottom_not_another_blazer():
    assert slots_compatible(Category.OUTER, Category.BOTTOM)
    assert not slots_compatible(Category.OUTER, Category.OUTER)
    assert slots_compatible(Category.DRESS, Category.OUTER)
    assert not slots_compatible(Category.DRESS, Category.TOP)


def test_a_wide_formality_gap_is_rejected_outright(make_item, settings):
    edge = score_pair(
        make_item("top-001", "top", formality=5, colors=["black"]),
        make_item("bottom-001", "bottom", formality=1, colors=["grey"]),
        settings,
    )
    assert edge.kind == "formality_conflict"
    assert not edge.compatible


def test_a_colour_clash_is_rejected_outright(make_item, settings):
    """Without this gate colour could never fail a pair on its own - the other
    components alone put two solids at equal formality above threshold."""
    edge = score_pair(
        make_item("top-001", "top", colors=["red"], formality=3),
        make_item("bottom-001", "bottom", colors=["lime"], formality=3),
        settings,
    )
    assert edge.kind == "color_conflict"
    assert not edge.compatible
    assert "clash" in edge.reasons[0]


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #


def test_neutrals_pair_with_everything(make_item, settings):
    edge = score_pair(
        make_item("top-001", "top", colors=["white"]),
        make_item("bottom-001", "bottom", colors=["navy"]),
        settings,
    )
    assert edge.compatible
    assert edge.score > 0.9
    assert edge.breakdown["color"] > 0.85


def test_complementary_colours_beat_a_wide_interval(make_item, settings):
    complementary = score_pair(
        make_item("top-001", "top", colors=["rust"]),
        make_item("bottom-001", "bottom", colors=["teal"]),
        settings,
    )
    wide = score_pair(
        make_item("top-001", "top", colors=["red"]),
        make_item("bottom-001", "bottom", colors=["green"]),
        settings,
    )
    assert complementary.score > wide.score


def test_two_patterns_score_below_one_pattern(make_item, settings):
    both = score_pair(
        make_item("top-001", "top", pattern="plaid", colors=["white"]),
        make_item("bottom-001", "bottom", pattern="striped", colors=["black"]),
        settings,
    )
    one = score_pair(
        make_item("top-001", "top", pattern="plaid", colors=["white"]),
        make_item("bottom-001", "bottom", pattern="solid", colors=["black"]),
        settings,
    )
    assert both.score < one.score
    assert both.breakdown["pattern"] < one.breakdown["pattern"]


def test_a_large_warmth_gap_discounts_the_score(make_item, settings):
    mismatched = score_pair(
        make_item("top-001", "top", warmth=0, colors=["white"]),
        make_item("outer-001", "outer", warmth=5, colors=["black"]),
        settings,
    )
    assert mismatched.breakdown["warmth_factor"] < 1.0


def test_colour_matching_takes_the_best_available_pairing(make_item, settings):
    """One shared colour is enough to tie a pairing together."""
    edge = score_pair(
        make_item("top-001", "top", colors=["lime", "white"]),
        make_item("bottom-001", "bottom", colors=["red"]),
        settings,
    )
    assert edge.compatible, "the white in the top rescues a pairing lime alone would fail"


def test_scoring_is_symmetric_and_canonically_ordered(make_item, settings):
    a = make_item("top-001", "top")
    b = make_item("bottom-001", "bottom")
    assert score_pair(a, b, settings) == score_pair(b, a, settings)
    assert score_pair(a, b, settings).item_a == "bottom-001", "ordered by id"


def test_every_edge_explains_itself(make_item, settings):
    edge = score_pair(make_item("top-001", "top"), make_item("bottom-001", "bottom"), settings)
    assert edge.reasons
    assert set(edge.breakdown) == {"color", "formality", "pattern", "warmth_factor"}


def test_threshold_is_configurable(make_item):
    a = make_item("top-001", "top", colors=["white"], formality=3)
    b = make_item("bottom-001", "bottom", colors=["red"], formality=3)

    assert score_pair(a, b, Settings(compatibility_threshold=0.5)).compatible
    assert not score_pair(a, b, Settings(compatibility_threshold=0.99)).compatible


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def test_graph_covers_every_pair(seed_closet, settings):
    items = seed_closet.items
    graph = score_closet(items, settings)
    expected = len(items) * (len(items) - 1) // 2

    assert len(graph) == expected
    assert graph.item_ids == sorted(i.id for i in items)


def test_graph_lookup_is_order_independent(seed_closet, settings):
    graph = score_closet(seed_closet.items, settings)
    assert graph.score("top-001", "bottom-001") == graph.score("bottom-001", "top-001")
    assert graph.compatible("top-001", "bottom-001") == graph.compatible("bottom-001", "top-001")


def test_partners_lists_only_compatible_items(seed_closet, settings):
    graph = score_closet(seed_closet.items, settings)
    partners = graph.partners("top-004")  # formality-5 evening blouse

    assert "bottom-001" not in partners, "the jeans are three formality steps away"
    assert all(graph.compatible("top-004", p) for p in partners)


def test_all_compatible_is_a_clique_test(seed_closet, settings):
    graph = score_closet(seed_closet.items, settings)
    assert graph.all_compatible(["top-001", "bottom-003", "shoes-002"])
    assert not graph.all_compatible(["top-001", "top-002"])


def test_graph_round_trips_through_disk(tmp_path, seed_closet, settings):
    graph = score_closet(seed_closet.items, settings)
    path = graph.save(tmp_path / "compat.json")
    reloaded = CompatibilityGraph.load(path)

    assert reloaded.item_ids == graph.item_ids
    assert reloaded.threshold == graph.threshold
    assert reloaded.edges == graph.edges


def test_derived_graph_is_stored_separately_from_the_closet(tmp_path, seed_closet, settings):
    """The compatibility layer attaches nothing to the source-of-truth records."""
    graph = score_closet(seed_closet.items, settings)
    graph.save(tmp_path / "compat.json")

    for item in seed_closet.items:
        assert "score" not in item.model_dump()
        assert "compatible" not in item.model_dump()


def test_merged_with_leaves_the_original_untouched(seed_closet, make_item, settings):
    graph = score_closet(seed_closet.items, settings)
    before = len(graph)
    newcomer = make_item("candidate::x", "bottom", colors=["burgundy"], formality=4, fabric="wool")

    extended = graph.merged_with(score_item_against(newcomer, seed_closet.items, settings), [newcomer.id])

    assert len(graph) == before
    assert len(extended) > before
    assert extended.edge("candidate::x", "top-001") is not None


# --------------------------------------------------------------------------- #
# Lookup tables and style identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "neutral"),
    [("white", True), ("charcoal", True), ("navy", True), ("olive", True), ("rust", False), ("teal", False)],
)
def test_neutral_classification(name, neutral):
    info = color_family(name)
    assert info is not None
    assert info.neutral is neutral


def test_compound_colour_names_resolve():
    assert color_family("dark olive") is not None
    assert color_family("navy blue") is not None


def test_unknown_colour_is_scored_neutrally_not_as_a_clash(make_item, settings):
    edge = score_pair(
        make_item("top-001", "top", colors=["puce"]),
        make_item("bottom-001", "bottom", colors=["greige"]),
        settings,
    )
    assert edge.compatible, "an unfamiliar colour is 'no opinion', not a rejection"


@pytest.mark.parametrize(
    ("fabric", "family"),
    [("merino wool", "wool"), ("italian linen", "linen"), ("suede", "leather"), ("denim", "denim")],
)
def test_fabric_families(fabric, family):
    assert fabric_family(fabric) == family


def test_style_signature_collapses_interchangeable_items(make_item):
    """Two pairs of the same jeans make no look the first pair did not."""
    a = make_item("bottom-001", "bottom", colors=["indigo"], fabric="denim", formality=2, warmth=2)
    b = make_item("bottom-009", "bottom", colors=["denim"], fabric="denim", formality=2, warmth=2)
    assert style_signature(a) == style_signature(b)


@pytest.mark.parametrize(
    "difference",
    [{"colors": ["rust"]}, {"formality": 4}, {"warmth": 5}, {"pattern": "plaid"}, {"fabric": "wool"}],
)
def test_style_signature_separates_genuinely_different_items(make_item, difference):
    base = dict(colors=["indigo"], fabric="denim", formality=2, warmth=2)
    a = make_item("bottom-001", "bottom", **base)
    b = make_item("bottom-002", "bottom", **{**base, **difference})
    assert style_signature(a) != style_signature(b)

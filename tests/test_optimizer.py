"""The purchase optimizer.

This is the most important test in the repo. A purchase optimizer that cannot
tell a genuinely additive item from a redundant one is not an optimizer, and the
naive metric fails exactly here: *any* new bottom multiplies the number of item
combinations, so a second pair of the jeans you already own looks like a huge win
if you only count combinations. The tests below pin the behaviour that makes the
distinction - counting distinct *looks*, not distinct item-sets.

The fixture closet is built with a deliberate structural hole: a formal blouse
and formal shoes that can never appear in an outfit together, because the closet
has no bottom formal enough to join them.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from wardrobe_agents.compatibility.optimizer import (
    evaluate_candidate,
    find_redundancies,
    optimize_purchases,
)
from wardrobe_agents.compatibility.enumeration import enumerate_outfits
from wardrobe_agents.compatibility.scoring import score_closet


@pytest.fixture
def settings() -> Settings:
    # Two new looks is enough to count as a recommendation in this small fixture.
    return Settings(marginal_gain_threshold=2)


@pytest.fixture
def closet(make_item):
    """A closet whose only complete outfit is the casual one.

    ``top-formal`` (f5) and ``shoes-formal`` (f5) work together, but every bottom
    is too casual to join them, so they are stranded.
    """
    return [
        make_item("top-casual", "top", subcategory="oxford shirt", colors=["white"], formality=3, warmth=2),
        make_item("bottom-casual", "bottom", subcategory="straight jeans", colors=["indigo"], fabric="denim", formality=2, warmth=2),
        make_item("shoes-casual", "shoes", subcategory="sneakers", colors=["white"], fabric="leather", formality=2, warmth=1),
        make_item("top-formal", "top", subcategory="evening blouse", colors=["black"], fabric="silk", formality=5, warmth=1),
        make_item("shoes-formal", "shoes", subcategory="heels", colors=["black"], fabric="leather", formality=5, warmth=1),
    ]


@pytest.fixture
def additive(make_candidate):
    """Formal trousers: the missing piece that joins the stranded formal items."""
    return make_candidate(
        "add-formal-trousers",
        "bottom",
        subcategory="tailored trousers",
        colors=["black"],
        fabric="wool",
        formality=5,
        warmth=2,
        price=200.0,
    )


@pytest.fixture
def redundant(make_candidate):
    """A second pair of the jeans already in the closet."""
    return make_candidate(
        "dup-jeans",
        "bottom",
        subcategory="straight jeans",
        colors=["indigo"],
        fabric="denim",
        formality=2,
        warmth=2,
        price=90.0,
    )


def test_fixture_closet_has_the_intended_structural_hole(closet, settings):
    """Guard the premise: the formal pieces really cannot be worn together yet."""
    graph = score_closet(closet, settings)
    baseline = enumerate_outfits(closet, graph, settings)

    assert graph.compatible("top-formal", "shoes-formal"), "formal pieces should suit each other"
    co_occurring = baseline.co_occurring_pairs()
    assert frozenset({"top-formal", "shoes-formal"}) not in co_occurring, (
        "the formal pieces should be stranded - no bottom is formal enough to join them"
    )


def test_additive_candidate_outranks_redundant_one(closet, additive, redundant, settings):
    """The headline behaviour: additive beats redundant, and the verdicts say why."""
    result = optimize_purchases(closet, [redundant, additive], settings)
    ranked = {r.candidate.candidate_id: r for r in result.recommendations}

    add = ranked["add-formal-trousers"]
    dup = ranked["dup-jeans"]

    assert add.rank < dup.rank
    assert add.new_outfit_count >= 2
    assert dup.new_outfit_count == 0
    assert add.verdict == "recommended"
    assert dup.verdict == "redundant"


def test_redundant_candidate_still_creates_combinations(closet, redundant, settings):
    """The trap this design avoids.

    The duplicate jeans *do* slot into valid combinations - counting those would
    rank it as a real improvement. It scores zero only because every combination
    it creates repeats a look the closet already supports.
    """
    result = optimize_purchases(closet, [redundant], settings)
    dup = result.recommendations[0]

    assert dup.new_combination_count > 0, "a duplicate does produce combinations"
    assert dup.new_outfit_count == 0, "but none of them is a new look"
    assert "already" in dup.explanation.lower()


def test_additive_candidate_reports_the_pairs_it_bridges(closet, additive, settings):
    """The 'why' behind the score: which stranded items it connects."""
    result = optimize_purchases(closet, [additive], settings)
    add = result.recommendations[0]

    bridged = {frozenset((b.item_a, b.item_b)) for b in add.bridged_pairs}
    assert frozenset({"top-formal", "shoes-formal"}) in bridged
    assert "bridges" in add.explanation.lower()


def test_redundant_candidate_names_the_item_it_duplicates(closet, redundant, settings):
    result = optimize_purchases(closet, [redundant], settings)
    dup = result.recommendations[0]

    assert dup.redundant_with == ["bottom-casual"]
    assert find_redundancies(redundant, closet)[0].id == "bottom-casual"


def test_candidate_that_completes_no_outfit_scores_zero(closet, make_candidate, settings):
    """An item can suit several pieces and still finish nothing.

    This opera coat is formal enough for the blouse and the heels, but the only
    complete core in the closet is the casual one, which it clashes with on
    formality. Zero, with a reason.
    """
    coat = make_candidate(
        "orphan-coat",
        "outer",
        subcategory="opera coat",
        colors=["black"],
        fabric="wool",
        formality=5,
        warmth=3,
    )
    result = optimize_purchases(closet, [coat], settings)
    rec = result.recommendations[0]

    assert rec.new_outfit_count == 0
    assert rec.new_combination_count == 0
    assert "never with" in rec.explanation


def test_candidate_with_no_compatible_partner_says_so(make_item, make_candidate, settings):
    """The other kind of zero: nothing in the closet works with it at all."""
    formal_only = [
        make_item("top-1", "top", colors=["white"], fabric="silk", formality=5, warmth=1),
        make_item("bottom-1", "bottom", colors=["black"], fabric="wool", formality=5, warmth=2),
        make_item("shoes-1", "shoes", colors=["black"], fabric="leather", formality=5, warmth=1),
    ]
    track_pants = make_candidate(
        "gym", "bottom", subcategory="track pants", colors=["lime"], fabric="polyester", formality=1
    )
    result = optimize_purchases(formal_only, [track_pants], settings)
    rec = result.recommendations[0]

    assert rec.new_outfit_count == 0
    assert "does not pair with a single item" in rec.explanation


def test_gain_is_measured_against_the_current_look_count(closet, additive, settings):
    graph = score_closet(closet, settings)
    baseline = enumerate_outfits(closet, graph, settings)
    index = {i.id: i for i in closet}

    rec = evaluate_candidate(additive, closet, graph, baseline, settings)

    assert rec.baseline_outfit_count == len(baseline.signatures(index))
    assert rec.total_outfit_count == rec.baseline_outfit_count + rec.new_outfit_count
    assert rec.marginal_gain_pct == pytest.approx(
        rec.new_outfit_count / rec.baseline_outfit_count * 100, abs=0.01
    )


def test_accessories_are_reported_as_unscored_not_redundant(closet, make_candidate, settings):
    """Zero here means 'outside the model', and must not read as a verdict."""
    scarf = make_candidate("scarf", "accessory", subcategory="scarf", colors=["grey"], fabric="wool")
    result = optimize_purchases(closet, [scarf], settings)
    rec = result.recommendations[0]

    assert rec.verdict == "not_scored"
    assert "accessor" in rec.explanation.lower()


def test_optimizer_does_not_mutate_the_closet(closet, additive, settings):
    """A candidate is projected into a copy; the real closet must be untouched."""
    before = [i.model_copy(deep=True) for i in closet]
    optimize_purchases(closet, [additive], settings)
    assert closet == before


def test_ranking_on_the_seed_closet(seed_closet, settings):
    """End-to-end on the shipped demo data: the duplicate lands below the rest."""
    from wardrobe_agents.orchestrator import load_candidates
    from tests.conftest import SEED_CANDIDATES

    candidates = load_candidates(SEED_CANDIDATES)
    result = optimize_purchases(seed_closet.wearable(), candidates, settings)
    by_id = {r.candidate.candidate_id: r for r in result.recommendations}

    assert by_id["cand-002"].new_outfit_count == 0, "the duplicate jeans unlock no new look"
    assert by_id["cand-002"].new_combination_count > 0, "though they do create combinations"
    assert by_id["cand-003"].new_outfit_count > 0, "the burgundy trousers are genuinely additive"
    assert by_id["cand-003"].rank < by_id["cand-002"].rank

"""The shared closet store: round-trips, identity, and the no-pollution rule."""

from __future__ import annotations

import json
from datetime import date
import pytest

from wardrobe_agents.closet import Closet, ClosetError, load_items_file
from wardrobe_agents.schemas import Category, Condition


def test_seed_closet_loads(seed_closet):
    assert len(seed_closet) == 15
    counts = seed_closet.category_counts()
    assert counts["top"] == 4
    assert counts["bottom"] == 3


def test_save_load_round_trip(tmp_path, make_item):
    closet = Closet([make_item("top-001", "top", last_worn="2026-01-05")], path=tmp_path / "c.json")
    closet.save()

    reloaded = Closet.load(tmp_path / "c.json")
    assert reloaded.items == closet.items
    assert reloaded.get("top-001").last_worn == date(2026, 1, 5)


def test_save_is_stable_across_writes(tmp_path, make_item):
    """Items are sorted on write, so a re-save produces a byte-identical diff."""
    path = tmp_path / "c.json"
    closet = Closet([make_item("top-002", "top"), make_item("top-001", "top")], path=path)
    closet.save()
    first = path.read_text()
    Closet.load(path).save(path)
    assert path.read_text() == first


def test_missing_file_is_an_empty_closet(tmp_path):
    closet = Closet.load(tmp_path / "nope.json")
    assert len(closet) == 0
    assert closet.path == tmp_path / "nope.json"


def test_computed_fields_cannot_be_written_onto_an_item(tmp_path):
    """The source of truth must stay uncontaminated by derived data."""
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "top-001",
                    "category": "top",
                    "subcategory": "tee",
                    "colors": ["white"],
                    "fabric": "cotton",
                    "warmth": 1,
                    "formality": 2,
                    "date_added": "2025-01-01",
                    "compatibility_score": 0.87,
                }
            ]
        )
    )
    with pytest.raises(ClosetError, match="Invalid item top-001"):
        Closet.load(path)


def test_duplicate_ids_are_rejected(tmp_path, make_item):
    path = tmp_path / "c.json"
    item = make_item("top-001", "top")
    path.write_text(json.dumps([item.model_dump(mode="json")] * 2))
    with pytest.raises(ClosetError, match="Duplicate item ids"):
        Closet.load(path)


def test_next_id_fills_gaps_and_avoids_collisions(make_item):
    closet = Closet([make_item("top-001", "top"), make_item("top-003", "top")])
    assert closet.next_id(Category.TOP) == "top-002"
    assert closet.next_id(Category.BOTTOM) == "bottom-001"


def test_add_refuses_to_clobber_without_overwrite(make_item):
    closet = Closet([make_item("top-001", "top")])
    with pytest.raises(ClosetError, match="already exists"):
        closet.add(make_item("top-001", "top", subcategory="different"))

    closet.add(make_item("top-001", "top", subcategory="different"), overwrite=True)
    assert closet.get("top-001").subcategory == "different"


def test_wearable_excludes_retired_items(make_item):
    closet = Closet(
        [make_item("top-001", "top"), make_item("top-002", "top", condition=Condition.RETIRE.value)]
    )
    assert [i.id for i in closet.wearable()] == ["top-001"]


def test_mark_worn_updates_last_worn(make_item):
    closet = Closet([make_item("top-001", "top")])
    closet.mark_worn(["top-001"], when=date(2026, 6, 1))
    assert closet.get("top-001").last_worn == date(2026, 6, 1)


def test_save_does_not_leave_temp_files_behind(tmp_path, make_item):
    path = tmp_path / "c.json"
    Closet([make_item("top-001", "top")], path=path).save()
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_load_items_file_accepts_bare_list_and_wrapped_object(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"id": "a"}]))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"items": [{"id": "a"}]}))

    assert load_items_file(bare) == load_items_file(wrapped) == [{"id": "a"}]


def test_load_items_file_reports_bad_json_with_the_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ClosetError, match="not valid JSON"):
        load_items_file(path)


def test_from_seed_copies_without_touching_the_original(tmp_path, seed_closet):
    from tests.conftest import SEED_CLOSET

    target = tmp_path / "mine" / "closet.json"
    closet = Closet.from_seed(SEED_CLOSET, target)
    closet.remove("top-001")
    closet.save()

    assert len(Closet.load(SEED_CLOSET)) == 15
    assert len(Closet.load(target)) == 14

"""Cataloguing: the manual path with no API at all, the photo path against a
mocked Claude response. Both must land in the same schema.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import json
from datetime import date
import pytest

from wardrobe_agents.agents.cataloguing import CataloguingAgent, CatalogueRequest, CataloguingError
from wardrobe_agents.closet import Closet
from wardrobe_agents.llm import FakeLLM, LLMError
from wardrobe_agents.schemas import Category, ClosetItem, Condition, ItemExtraction


@pytest.fixture
def agent() -> CataloguingAgent:
    """An agent whose LLM would raise if touched - the manual path must not call it."""
    return CataloguingAgent(llm=FakeLLM([]))


@pytest.fixture
def extraction() -> ItemExtraction:
    return ItemExtraction(
        category=Category.OUTER,
        subcategory="trench coat",
        colors=["beige"],
        pattern="solid",
        fabric="cotton gabardine",
        warmth=3,
        formality=4,
        condition=Condition.EXCELLENT,
        notes="Belted, knee length.",
        confidence=0.82,
        uncertain_fields=["fabric"],
    )


# --------------------------------------------------------------------------- #
# Manual entry
# --------------------------------------------------------------------------- #


def test_manual_entry_produces_a_valid_item(agent):
    closet = Closet()
    item = agent.build_item(
        {
            "category": "top",
            "subcategory": "Linen Shirt",
            "colors": ["White"],
            "fabric": "linen",
            "warmth": 1,
            "formality": 3,
        },
        closet,
        today=date(2026, 5, 1),
    )

    assert item.id == "top-001"
    assert item.subcategory == "linen shirt", "text is normalised to lowercase"
    assert item.colors == ["white"]
    assert item.date_added == date(2026, 5, 1)
    assert item.condition is Condition.GOOD
    assert item.source == "manual"


def test_manual_entry_allocates_non_colliding_ids_within_one_batch(agent):
    closet = Closet()
    result = agent.run(
        CatalogueRequest(
            closet=closet,
            manual_entries=[
                {"category": "top", "subcategory": "a", "colors": ["white"], "fabric": "cotton", "warmth": 1, "formality": 2},
                {"category": "top", "subcategory": "b", "colors": ["black"], "fabric": "cotton", "warmth": 1, "formality": 2},
            ],
        )
    )
    assert [d.item.id for d in result.drafts] == ["top-001", "top-002"]


def test_manual_entry_rejects_an_unknown_category(agent):
    with pytest.raises(CataloguingError, match="Unknown category"):
        agent.build_item({"category": "hat", "subcategory": "beanie"}, Closet())


def test_manual_entry_reports_invalid_values_rather_than_coercing(agent):
    with pytest.raises(CataloguingError, match="Invalid item"):
        agent.build_item(
            {
                "category": "top",
                "subcategory": "tee",
                "colors": ["white"],
                "fabric": "cotton",
                "warmth": 9,  # out of the 0-5 range
                "formality": 2,
            },
            Closet(),
        )


def test_manual_entries_do_not_reach_the_llm(agent):
    """The FakeLLM has no queued responses; touching it would raise."""
    result = agent.run(
        CatalogueRequest(
            closet=Closet(),
            manual_entries=[
                {"category": "shoes", "subcategory": "boots", "colors": ["brown"], "fabric": "leather", "warmth": 3, "formality": 3}
            ],
        )
    )
    assert result.errors == []
    assert agent.llm.calls == []


def test_import_file_collects_errors_without_abandoning_good_rows(agent, tmp_path):
    path = tmp_path / "items.json"
    path.write_text(
        json.dumps(
            [
                {"category": "top", "subcategory": "tee", "colors": ["white"], "fabric": "cotton", "warmth": 1, "formality": 2},
                {"category": "nonsense", "subcategory": "x"},
            ]
        )
    )
    result = agent.import_file(path, Closet())

    assert len(result.drafts) == 1
    assert len(result.errors) == 1
    assert not result.drafts[0].needs_confirmation, "hand-written entries need no confirmation"


def test_import_file_ids_do_not_collide_with_the_existing_closet(agent, tmp_path, make_item):
    path = tmp_path / "items.json"
    path.write_text(
        json.dumps(
            [{"id": "top-001", "category": "top", "subcategory": "tee", "colors": ["white"], "fabric": "cotton", "warmth": 1, "formality": 2}]
        )
    )
    closet = Closet([make_item("top-001", "top")])
    result = agent.import_file(path, closet)

    assert result.drafts[0].item.id == "top-002", "a taken id is reallocated, not overwritten"


# --------------------------------------------------------------------------- #
# Photo extraction (mocked Claude)
# --------------------------------------------------------------------------- #


def test_photo_extraction_writes_the_same_schema(extraction, tmp_path):
    photo = tmp_path / "coat.jpg"
    photo.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    agent = CataloguingAgent(llm=FakeLLM([extraction]))

    result = agent.run(CatalogueRequest(closet=Closet(), photo_paths=[photo], today=date(2026, 5, 1)))

    assert len(result.drafts) == 1
    draft = result.drafts[0]
    assert isinstance(draft.item, ClosetItem), "the photo path lands in the shared schema"
    assert draft.item.category is Category.OUTER
    assert draft.item.subcategory == "trench coat"
    assert draft.item.source == "photo"
    assert draft.item.photo_path == str(photo)


def test_extracted_items_are_flagged_for_confirmation(extraction, tmp_path):
    photo = tmp_path / "coat.png"
    photo.write_bytes(b"\x89PNG stub")
    agent = CataloguingAgent(llm=FakeLLM([extraction]))

    draft = agent.run(CatalogueRequest(closet=Closet(), photo_paths=[photo])).drafts[0]

    assert draft.needs_confirmation
    assert draft.confidence == pytest.approx(0.82)
    assert draft.uncertain_fields == ["fabric"]


def test_identity_is_assigned_locally_not_by_the_model(extraction, tmp_path, make_item):
    """The model never gets to pick an id, so it cannot collide with a real one."""
    photo = tmp_path / "coat.jpg"
    photo.write_bytes(b"stub")
    closet = Closet([make_item("outer-001", "outer")])
    agent = CataloguingAgent(llm=FakeLLM([extraction]))

    draft = agent.run(CatalogueRequest(closet=closet, photo_paths=[photo])).drafts[0]

    assert draft.item.id == "outer-002"
    assert "confidence" not in draft.item.model_dump()


def test_a_failed_photo_does_not_sink_the_batch(extraction, tmp_path):
    good = tmp_path / "good.jpg"
    good.write_bytes(b"stub")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"stub")
    agent = CataloguingAgent(llm=FakeLLM([LLMError("model refused"), extraction]))

    result = agent.run(CatalogueRequest(closet=Closet(), photo_paths=[bad, good]))

    assert len(result.drafts) == 1
    assert len(result.errors) == 1
    assert "model refused" in result.errors[0]


def test_unsupported_image_types_are_refused(tmp_path):
    photo = tmp_path / "coat.tiff"
    photo.write_bytes(b"stub")
    agent = CataloguingAgent(llm=FakeLLM([]))

    result = agent.run(CatalogueRequest(closet=Closet(), photo_paths=[photo]))

    assert result.drafts == []
    assert "Unsupported image type" in result.errors[0]

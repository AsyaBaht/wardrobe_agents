"""Web UI: the HTTP layer over the same pipelines, against a mocked Claude.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from wardrobe_agents.llm import FakeLLM, LLMError  # noqa: E402
from wardrobe_agents.schemas import (  # noqa: E402
    Category,
    Condition,
    ItemExtraction,
    StylistPick,
    StylistResponse,
)
from wardrobe_agents.web import create_app  # noqa: E402

TEMPS = {"temp_min": 8, "temp_max": 17}


def _client(test_settings, *responses) -> tuple[TestClient, FakeLLM]:
    llm = FakeLLM(list(responses))
    return TestClient(create_app(test_settings, llm)), llm


def _stylist_response() -> StylistResponse:
    return StylistResponse(
        outfits=[
            StylistPick(
                name="Weekday neutral",
                item_ids=["top-001", "bottom-003", "shoes-002"],
                rationale="Neutral base, warm-toned boots.",
                weather_fit="Layerable for the swing.",
                formality=3,
                confidence=0.85,
            )
        ],
        overall_notes="Good coverage today.",
    )


def _extraction() -> ItemExtraction:
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


def _trench(**extra) -> dict:
    return {
        "category": "outer",
        "subcategory": "trench coat",
        "colors": ["beige"],
        "fabric": "cotton gabardine",
        "warmth": 3,
        "formality": 4,
        **extra,
    }


def test_the_page_is_served(test_settings):
    client, _ = _client(test_settings)
    response = client.get("/")
    assert response.status_code == 200
    assert "What to wear" in response.text


def test_closet_falls_back_to_the_seed_without_exposing_paths(test_settings):
    client, _ = _client(test_settings)
    data = client.get("/api/closet").json()

    assert data["is_seed"] is True
    assert len(data["items"]) == 15
    assert "photo_path" not in data["items"][0]
    assert data["items"][0]["has_photo"] is False


def test_recommend_returns_outfits_and_writes_a_report(test_settings):
    client, _ = _client(test_settings, _stylist_response())
    response = client.post("/api/recommend", json={**TEMPS, "count": 1, "occasion": "dinner"})

    assert response.status_code == 200
    data = response.json()
    assert data["constraints"]["source"] == "manual"
    assert data["suggestions"][0]["item_ids"] == ["top-001", "bottom-003", "shoes-002"]
    assert data["suggestions"][0]["items"][0]["label"] == "white oxford shirt"
    assert len(list(test_settings.reports_dir.glob("*-recommend.json"))) == 1


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, 400),
        ({"temp_min": 8}, 400),
        ({"temp_min": 20, "temp_max": 5}, 400),
        ({**TEMPS, "formality": 9}, 422),
    ],
)
def test_recommend_rejects_bad_input_before_calling_the_model(test_settings, body, status):
    client, llm = _client(test_settings)
    assert client.post("/api/recommend", json=body).status_code == status
    assert llm.calls == []


def test_a_model_failure_is_a_message_not_a_crash(test_settings):
    client, _ = _client(test_settings, LLMError("rate limited"))
    response = client.post("/api/recommend", json=TEMPS)

    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


def test_adding_an_item_starts_a_working_copy_of_the_seed(test_settings):
    client, _ = _client(test_settings)
    response = client.post("/api/items", json=_trench(tags=["waterproof"]))

    assert response.status_code == 200
    assert response.json()["item"]["id"] == "outer-004", "the id is allocated by the server"
    saved = json.loads(test_settings.closet_path.read_text())["items"]
    assert len(saved) == 16
    assert client.get("/api/closet").json()["is_seed"] is False


def test_an_invalid_item_is_rejected_and_nothing_is_written(test_settings):
    client, _ = _client(test_settings)
    response = client.post("/api/items", json=_trench(warmth=9))

    assert response.status_code == 400
    assert not test_settings.closet_path.exists()


def test_a_photo_is_read_then_saved_only_on_confirmation(test_settings):
    client, _ = _client(test_settings, _extraction())
    extracted = client.post(
        "/api/extract", files={"photo": ("coat.jpg", b"not really a jpeg", "image/jpeg")}
    )

    assert extracted.status_code == 200
    draft = extracted.json()
    assert draft["item"]["subcategory"] == "trench coat"
    assert draft["uncertain_fields"] == ["fabric"]
    assert not test_settings.closet_path.exists(), "extraction alone saves nothing"

    saved = client.post("/api/items", json=_trench(upload=draft["upload"])).json()["item"]
    assert saved["has_photo"] is True
    photo = client.get(f"/api/items/{saved['id']}/photo")
    assert photo.status_code == 200
    assert photo.content == b"not really a jpeg"


def test_a_non_image_upload_is_refused(test_settings):
    client, llm = _client(test_settings, _extraction())
    response = client.post("/api/extract", files={"photo": ("notes.txt", b"hi", "text/plain")})
    assert response.status_code == 415
    assert llm.calls == []


@pytest.mark.parametrize("upload", ["../closet.json", "/etc/passwd", "0" * 32 + ".jpg"])
def test_an_upload_name_cannot_point_outside_the_photo_folder(test_settings, upload):
    client, _ = _client(test_settings)
    assert client.post("/api/items", json=_trench(upload=upload)).status_code == 400
    assert not test_settings.closet_path.exists()


def test_marking_an_outfit_worn_updates_last_worn(test_settings):
    client, _ = _client(test_settings)
    response = client.post("/api/worn", json={"item_ids": ["top-001", "bottom-003"]})

    assert response.status_code == 200
    assert all(item["last_worn"] for item in response.json()["items"])
    assert client.post("/api/worn", json={"item_ids": ["nope-001"]}).status_code == 400

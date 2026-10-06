"""A small web UI for use on a phone: today's outfits, the closet, add by photo.

This is a second front end beside the CLI, not a new layer: it calls the same
orchestrator and the same cataloguing agent, and holds no logic of its own
beyond turning HTTP into those calls. It is meant to run on the owner's own
computer and has **no login** - binding it to the local network makes the closet
readable and writable by anyone on that network.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config.settings import Settings, settings as default_settings
from wardrobe_agents.agents.cataloguing import CataloguingAgent, CataloguingError
from wardrobe_agents.agents.stylist import StylistError
from wardrobe_agents.agents.weather import WeatherError, band_for_temp, warmth_for_temp
from wardrobe_agents.closet import Closet, ClosetError
from wardrobe_agents.llm import LLMError, MissingAPIKeyError, StructuredLLM
from wardrobe_agents.orchestrator import run_recommend
from wardrobe_agents.schemas import ClosetItem, WeatherConstraints

STATIC_DIR = Path(__file__).parent / "static"

UPLOAD_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_PHOTO_SUFFIXES = {*UPLOAD_TYPES.values(), ".jpeg"}
_UPLOAD_RE = re.compile(r"^[0-9a-f]{32}\.(jpg|png|webp|gif)$")


class RecommendBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str | None = None
    date: dt.date | None = None
    occasion: str | None = None
    formality: int | None = Field(default=None, ge=1, le=5)
    count: int | None = Field(default=None, ge=1, le=6)
    temp_min: float | None = None
    temp_max: float | None = None


class ItemBody(BaseModel):
    """Item attributes as the user confirmed them. Identity is assigned here,
    exactly as for the CLI: the client never chooses an id."""

    model_config = ConfigDict(extra="forbid")

    category: str
    subcategory: str
    colors: list[str]
    pattern: str = "solid"
    fabric: str
    warmth: int
    formality: int
    condition: str = "good"
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    upload: str | None = None
    """Name returned by ``/api/extract`` for the photo this item came from."""


class ItemEditBody(BaseModel):
    """The attributes a user may change on an existing item. Identity and
    provenance (``id``, ``category``, ``date_added``, ``source``, the photo) are
    not editable: the id encodes the category, and the rest are facts."""

    model_config = ConfigDict(extra="forbid")

    subcategory: str
    colors: list[str]
    pattern: str
    fabric: str
    warmth: int
    formality: int
    condition: str
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


class WornBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_ids: list[str] = Field(min_length=1)


def _photo_file(item: ClosetItem) -> Path | None:
    if not item.photo_path:
        return None
    path = Path(item.photo_path)
    return path if path.is_file() and path.suffix.lower() in _PHOTO_SUFFIXES else None


def _item_json(item: ClosetItem) -> dict[str, Any]:
    """An item for the browser. The filesystem path is replaced by a flag; the
    photo itself is fetched by item id."""
    data = item.model_dump(mode="json", exclude={"photo_path"})
    data["label"] = item.label
    data["has_photo"] = _photo_file(item) is not None
    return data


def create_app(settings: Settings | None = None, llm: StructuredLLM | None = None) -> FastAPI:
    settings = settings or default_settings
    llm = llm or StructuredLLM(settings)
    photos_dir = Path(settings.closet_path).parent / "photos"
    write_lock = threading.Lock()

    app = FastAPI(title="wardrobe-agents", docs_url=None, redoc_url=None)

    def load_closet() -> tuple[Closet, bool]:
        """The working closet, or the read-only bundled seed when there is none."""
        path = Path(settings.closet_path)
        try:
            if path.exists():
                return Closet.load(path), False
            return Closet.load(Path(settings.seed_closet_path)), True
        except ClosetError as exc:
            raise HTTPException(500, str(exc)) from exc

    def save(closet: Closet) -> None:
        """Always to the working path, never back into the bundled seed: the first
        successful write is what turns the seed into the user's own closet."""
        closet.save(Path(settings.closet_path))

    def drop_compatibility_cache() -> None:
        """The cache is keyed by item ids, so it cannot tell that an item's
        attributes changed underneath it. It is derived data; stage 2 rebuilds it."""
        Path(settings.compatibility_cache_path).unlink(missing_ok=True)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/closet")
    def closet_view() -> dict[str, Any]:
        closet, is_seed = load_closet()
        return {
            "items": [_item_json(i) for i in closet.items],
            "is_seed": is_seed,
            "llm_configured": llm.is_configured(),
        }

    @app.get("/api/items/{item_id}/photo")
    def item_photo(item_id: str) -> FileResponse:
        closet, _ = load_closet()
        item = closet.find(item_id)
        path = _photo_file(item) if item else None
        if path is None:
            raise HTTPException(404, "No photo for this item.")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/recommend")
    def recommend(body: RecommendBody) -> dict[str, Any]:
        closet, _ = load_closet()
        on = body.date or dt.date.today()

        manual: WeatherConstraints | None = None
        if body.temp_min is not None or body.temp_max is not None:
            if body.temp_min is None or body.temp_max is None:
                raise HTTPException(400, "Give both a low and a high temperature.")
            if body.temp_min > body.temp_max:
                raise HTTPException(400, "The low temperature is above the high.")
            manual = WeatherConstraints(
                date=on,
                location=body.location or "unspecified",
                temp_min_c=body.temp_min,
                temp_max_c=body.temp_max,
                temp_band=band_for_temp(body.temp_max),
                precipitation_mm=0.0,
                precipitation_probability_pct=0,
                wind_kph=0.0,
                conditions="as described",
                min_warmth=warmth_for_temp(body.temp_max),
                max_warmth=warmth_for_temp(body.temp_min),
                layering_advice="Temperatures entered by hand; no forecast was fetched.",
                source="manual",
            )
        elif not (body.location or "").strip():
            raise HTTPException(400, "Enter a location, or enter the temperatures yourself.")

        try:
            result = run_recommend(
                closet,
                location=body.location,
                on=on,
                occasion=body.occasion or None,
                target_formality=body.formality,
                count=body.count,
                constraints=manual,
                llm=llm,
                settings=settings,
            )
        except MissingAPIKeyError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (WeatherError, StylistError, LLMError, ValueError) as exc:
            raise HTTPException(502, str(exc)) from exc

        c = result.constraints
        return {
            "summary": c.summary(),
            "constraints": c.model_dump(mode="json"),
            "suggestions": [
                {
                    **s.model_dump(mode="json", exclude={"items"}),
                    "items": [_item_json(i) for i in s.items],
                }
                for s in result.suggestions
            ],
            "overall_notes": result.overall_notes,
            "warnings": result.warnings,
        }

    @app.post("/api/extract")
    def extract(photo: UploadFile = File(...)) -> dict[str, Any]:
        """Save an uploaded photo and read its attributes. Nothing is added to
        the closet here - the user confirms or edits first, as in the CLI."""
        suffix = UPLOAD_TYPES.get(photo.content_type or "")
        if suffix is None:
            raise HTTPException(415, "Upload a JPEG, PNG, WebP or GIF image.")
        data = photo.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "That photo is too large (8 MB limit).")
        if not llm.is_configured():
            raise HTTPException(
                503,
                f"Reading a photo needs the Claude API. Set {settings.api_key_env_var} where the "
                "server runs, or add the item by hand.",
            )

        photos_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{suffix}"
        path = photos_dir / name
        path.write_bytes(data)

        try:
            extraction = CataloguingAgent(llm, settings).extract_from_photo(path)
        except LLMError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(502, str(exc)) from exc

        attrs = extraction.model_dump(mode="json", exclude={"confidence", "uncertain_fields"})
        return {
            "upload": name,
            "item": attrs,
            "confidence": extraction.confidence,
            "uncertain_fields": extraction.uncertain_fields,
        }

    @app.post("/api/items")
    def add_item(body: ItemBody) -> dict[str, Any]:
        attrs = body.model_dump(exclude={"upload"})
        source = "manual"
        if body.upload is not None:
            path = photos_dir / body.upload
            if not _UPLOAD_RE.match(body.upload) or not path.is_file():
                raise HTTPException(400, "Unknown photo upload.")
            attrs["photo_path"] = str(path)
            source = "photo"

        with write_lock:
            closet, _ = load_closet()
            try:
                item = CataloguingAgent(llm, settings).build_item(attrs, closet, source=source)
                closet.add(item)
                save(closet)
            except (CataloguingError, ClosetError) as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"item": _item_json(item)}

    @app.put("/api/items/{item_id}")
    def edit_item(item_id: str, body: ItemEditBody) -> dict[str, Any]:
        with write_lock:
            closet, _ = load_closet()
            current = closet.find(item_id)
            if current is None:
                raise HTTPException(404, f"No item with id {item_id!r}.")
            try:
                updated = ClosetItem.model_validate(
                    {**current.model_dump(mode="json"), **body.model_dump()}
                )
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                raise HTTPException(400, f"Invalid item: {problems}") from exc
            closet.add(updated, overwrite=True)
            save(closet)
            drop_compatibility_cache()
        return {"item": _item_json(updated)}

    @app.delete("/api/items/{item_id}")
    def delete_item(item_id: str) -> dict[str, Any]:
        with write_lock:
            closet, _ = load_closet()
            try:
                removed = closet.remove(item_id)
            except ClosetError as exc:
                raise HTTPException(404, str(exc)) from exc
            save(closet)
            # Only photos this UI stored are removed with their item; a path the
            # user supplied through the CLI is theirs.
            photo = _photo_file(removed)
            if photo is not None and photo.parent.resolve() == photos_dir.resolve():
                photo.unlink(missing_ok=True)
        return {"deleted": removed.id}

    @app.post("/api/worn")
    def mark_worn(body: WornBody) -> dict[str, Any]:
        with write_lock:
            closet, _ = load_closet()
            try:
                touched = closet.mark_worn(body.item_ids)
                save(closet)
            except ClosetError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"items": [_item_json(i) for i in touched]}

    return app

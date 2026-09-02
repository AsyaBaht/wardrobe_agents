"""Cataloguing agent: gets garments into the closet by two paths, one schema.

Path (a) **structured manual entry** - the user writes or edits a JSON/YAML file
of attribute dicts. Deterministic; no model involved.

Path (b) **photo extraction** - Claude's vision reads the same attributes off an
image into :class:`ItemExtraction`, which the user confirms or edits before it is
saved.

Both paths converge on :class:`ClosetItem` and nothing else: there is no
"photo item" record and no second schema. The model never assigns identity
either - ``id``, ``date_added`` and ``source`` are set here, so a hallucinated id
cannot collide with a real one.

Confirmation is intentionally *not* done in this module. The agent returns
drafts; presenting them and taking the user's edits is the CLI's job, which keeps
the agent usable from a test or another caller.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from config.settings import Settings, settings as default_settings
from wardrobe_agents.agents.base import BaseAgent
from wardrobe_agents.closet import Closet, ClosetError, load_items_file
from wardrobe_agents.llm import LLMRequest, StructuredLLM, image_block, text_block
from wardrobe_agents.schemas import Category, ClosetItem, Condition, ItemExtraction

EXTRACTION_SYSTEM = """You are cataloguing a single garment from a photograph for a
personal wardrobe database.

Report only what the photo actually supports. Scales:

- warmth 0-5: 0 hot-weather only (linen shorts, sandals), 1 light (tee, chinos),
  2 mild (oxford shirt, light knit), 3 cool (denim jacket, wool trousers),
  4 cold (wool coat, heavy knit), 5 freezing (down parka).
- formality 1-5: 1 loungewear, 2 casual, 3 smart casual, 4 business/dressy,
  5 formal.
- category: exactly one of top, bottom, dress, outer, shoes, accessory.
- condition: new, excellent, good, worn, or retire - judged from visible wear.

Give colors as common lowercase names, primary colour first. Use "solid" for
pattern when there is no pattern. Name any attribute you had to guess at in
uncertain_fields so the owner can correct it - guessing silently is worse than
flagging. If the item is partly obscured, say so in notes."""


class CataloguingError(RuntimeError):
    """Item data could not be turned into a valid closet record."""


@dataclass(slots=True)
class ItemDraft:
    """A proposed closet item, before it is committed.

    ``needs_confirmation`` is True for anything a model produced.
    """

    item: ClosetItem
    needs_confirmation: bool = False
    extraction: ItemExtraction | None = None
    photo_path: Path | None = None

    @property
    def uncertain_fields(self) -> list[str]:
        return list(self.extraction.uncertain_fields) if self.extraction else []

    @property
    def confidence(self) -> float | None:
        return self.extraction.confidence if self.extraction else None


@dataclass(slots=True)
class CatalogueRequest:
    """Input to :class:`CataloguingAgent`. Either path, or both at once."""

    closet: Closet
    manual_entries: Sequence[dict[str, Any]] = field(default_factory=list)
    photo_paths: Sequence[Path] = field(default_factory=list)
    today: date | None = None


@dataclass(slots=True)
class CatalogueResult:
    drafts: list[ItemDraft] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def items(self) -> list[ClosetItem]:
        return [d.item for d in self.drafts]


class CataloguingAgent(BaseAgent[CatalogueRequest, CatalogueResult]):
    """Produces validated :class:`ClosetItem` drafts from dicts and/or photos."""

    name = "cataloguing"

    def __init__(self, llm: StructuredLLM | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.llm = llm or StructuredLLM(self.settings)

    # ---- path (a): structured manual entry ------------------------------

    def build_item(
        self,
        attrs: dict[str, Any],
        closet: Closet,
        *,
        reserved: set[str] | None = None,
        today: date | None = None,
        source: str = "manual",
    ) -> ClosetItem:
        """Validate a raw attribute dict into a :class:`ClosetItem`.

        Fills in only what identity requires: an id (if absent or taken) and
        ``date_added``. Everything else must come from the caller, so a typo
        surfaces as a validation error rather than a silent default.
        """
        data = dict(attrs)
        reserved = reserved if reserved is not None else set()

        raw_category = data.get("category")
        if raw_category is None:
            raise CataloguingError("Item is missing 'category'.")
        try:
            category = Category(raw_category) if not isinstance(raw_category, Category) else raw_category
        except ValueError as exc:
            valid = ", ".join(c.value for c in Category)
            raise CataloguingError(f"Unknown category {raw_category!r}. Use one of: {valid}") from exc

        item_id = data.get("id")
        if not item_id or item_id in closet or item_id in reserved:
            item_id = self._allocate_id(closet, category, reserved)
        data["id"] = item_id
        reserved.add(item_id)

        data.setdefault("date_added", (today or date.today()).isoformat())
        data.setdefault("condition", Condition.GOOD.value)
        data.setdefault("source", source)

        try:
            return ClosetItem.model_validate(data)
        except ValidationError as exc:
            raise CataloguingError(f"Invalid item {item_id!r}:\n{exc}") from exc

    @staticmethod
    def _allocate_id(closet: Closet, category: Category, reserved: set[str]) -> str:
        """Next free id, accounting for ids already handed out in this batch."""
        candidate = closet.next_id(category)
        while candidate in reserved:
            prefix, _, num = candidate.rpartition("-")
            candidate = f"{prefix}-{int(num) + 1:03d}"
        return candidate

    def import_file(
        self, path: Path, closet: Closet, *, today: date | None = None
    ) -> CatalogueResult:
        """Read a JSON/YAML file of items the user wrote by hand."""
        try:
            raw = load_items_file(path)
        except ClosetError as exc:
            raise CataloguingError(str(exc)) from exc

        result = CatalogueResult()
        reserved: set[str] = set()
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                result.errors.append(f"Entry #{index} is not an object; skipped.")
                continue
            try:
                item = self.build_item(entry, closet, reserved=reserved, today=today)
            except CataloguingError as exc:
                result.errors.append(str(exc))
                continue
            result.drafts.append(ItemDraft(item=item, needs_confirmation=False))
        return result

    # ---- path (b): photo extraction -------------------------------------

    def extract_from_photo(self, path: Path) -> ItemExtraction:
        """One vision call for one garment photo."""
        return self.llm.call(
            system=EXTRACTION_SYSTEM,
            content=[
                image_block(path),
                text_block(
                    "Catalogue this garment. Report the attributes you can see, and flag "
                    "anything you had to guess."
                ),
            ],
            response_model=ItemExtraction,
            purpose=f"extracting attributes from {path.name}",
            model=self.settings.vision_model(),
        )

    def extract_from_photos(
        self, paths: Sequence[Path]
    ) -> list[tuple[Path, ItemExtraction | Exception]]:
        """Extract several photos concurrently, bounded by
        ``settings.max_concurrent_llm_calls``.

        A photo that fails - unreadable file, unsupported format, a model error -
        is reported against that photo and the rest of the batch still runs.
        Building the request can fail too (that is where the file is read), so
        that step is guarded per-photo rather than up front.
        """
        prompt = text_block(
            "Catalogue this garment. Report the attributes you can see, and flag "
            "anything you had to guess."
        )

        requests: list[LLMRequest[ItemExtraction]] = []
        request_paths: list[Path] = []
        failures: dict[Path, Exception] = {}

        for path in paths:
            try:
                content = [image_block(path), prompt]
            except Exception as exc:
                failures[path] = exc
                continue
            requests.append(
                LLMRequest(
                    system=EXTRACTION_SYSTEM,
                    content=content,
                    response_model=ItemExtraction,
                    purpose=f"extracting attributes from {path.name}",
                    model=self.settings.vision_model(),
                )
            )
            request_paths.append(path)

        results = self.llm.call_batch(requests)
        outcomes: dict[Path, ItemExtraction | Exception] = dict(failures)
        for path, result in zip(request_paths, results, strict=True):
            outcomes[path] = result.value if result.ok else result.error  # type: ignore[assignment]

        return [(path, outcomes[path]) for path in paths]

    def draft_from_extraction(
        self,
        extraction: ItemExtraction,
        closet: Closet,
        *,
        photo_path: Path | None = None,
        reserved: set[str] | None = None,
        today: date | None = None,
    ) -> ItemDraft:
        """Turn a vision extraction into an unconfirmed :class:`ClosetItem` draft."""
        attrs = extraction.model_dump(mode="json")
        for computed_only in ("confidence", "uncertain_fields"):
            attrs.pop(computed_only, None)
        attrs["photo_path"] = str(photo_path) if photo_path else None
        item = self.build_item(attrs, closet, reserved=reserved, today=today, source="photo")
        return ItemDraft(
            item=item, needs_confirmation=True, extraction=extraction, photo_path=photo_path
        )

    # ---- agent entry point ----------------------------------------------

    def run(self, payload: CatalogueRequest) -> CatalogueResult:
        """Produce drafts for every manual entry and photo in the request."""
        result = CatalogueResult()
        reserved: set[str] = set()

        for index, entry in enumerate(payload.manual_entries):
            try:
                item = self.build_item(entry, payload.closet, reserved=reserved, today=payload.today)
            except CataloguingError as exc:
                result.errors.append(f"Manual entry #{index}: {exc}")
                continue
            result.drafts.append(ItemDraft(item=item, needs_confirmation=False))

        if payload.photo_paths:
            for path, outcome in self.extract_from_photos(list(payload.photo_paths)):
                if isinstance(outcome, Exception):
                    result.errors.append(f"{path.name}: {outcome}")
                    continue
                try:
                    draft = self.draft_from_extraction(
                        outcome, payload.closet, photo_path=path, reserved=reserved, today=payload.today
                    )
                except CataloguingError as exc:
                    result.errors.append(f"{path.name}: {exc}")
                    continue
                result.drafts.append(draft)

        return result

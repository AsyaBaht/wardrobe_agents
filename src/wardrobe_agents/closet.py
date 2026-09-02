"""The closet data store: the single source of truth both stages read.

**Why JSON and not SQLite.** The closet is a small, human-scale collection
(tens of items, not thousands) and one of the two supported cataloguing paths is
"the user edits the file directly". JSON gives that for free: it is readable,
hand-editable, and diffable in git, which matters because run reports are also
artifacts meant to be diffed. SQLite would buy indexed queries and concurrent
writes - neither of which a single-user closet of this size needs - at the cost
of making the manual-entry path require a tool. The access pattern is "load the
whole closet, reason over all of it", which is a whole-file read either way.

Derived data (the compatibility graph) is written to a *separate* file. Item
records never carry computed fields; see :mod:`wardrobe_agents.schemas`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from pydantic import ValidationError

from wardrobe_agents.schemas import Category, ClosetItem

SCHEMA_VERSION = 1

_ID_RE = re.compile(r"^(?P<prefix>[a-z]+)-(?P<num>\d+)$")


class ClosetError(RuntimeError):
    """Raised for unreadable, malformed, or conflicting closet data."""


def _serialize(obj: Any) -> Any:
    """JSON encoder for dates and enums."""
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def load_items_file(path: Path, key: str = "items") -> list[dict[str, Any]]:
    """Read a list of raw records from a JSON or YAML file.

    Supports both the wrapped ``{"<key>": [...]}`` shape and a bare list, so a
    user can hand-write either.
    """
    if not path.exists():
        raise ClosetError(f"No such file: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise ClosetError(
                f"{path.name} is YAML but PyYAML is not installed. "
                "Install it with `pip install 'wardrobe-agents[yaml]'`, or use JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClosetError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        data = data.get(key, [])
    if not isinstance(data, list):
        raise ClosetError(
            f"{path} must contain a list of records (or an object with a {key!r} list)."
        )
    return data


class Closet:
    """An in-memory closet backed by a JSON file.

    Mutations are in-memory until :meth:`save` is called, so a failed catalog
    run cannot leave a half-written closet behind.
    """

    def __init__(self, items: Sequence[ClosetItem] | None = None, path: Path | None = None) -> None:
        self._items: dict[str, ClosetItem] = {}
        self.path = path
        for item in items or []:
            self._items[item.id] = item

    # ---- construction ---------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "Closet":
        """Load a closet from disk. A missing file yields an empty closet, so a
        first run does not need a bootstrap step."""
        path = Path(path)
        if not path.exists():
            return cls(path=path)

        raw = load_items_file(path)
        items: list[ClosetItem] = []
        for index, entry in enumerate(raw):
            try:
                items.append(ClosetItem.model_validate(entry))
            except ValidationError as exc:
                ident = entry.get("id", f"#{index}") if isinstance(entry, dict) else f"#{index}"
                raise ClosetError(f"Invalid item {ident} in {path}:\n{exc}") from exc

        duplicates = [i for i, c in Counter(i.id for i in items).items() if c > 1]
        if duplicates:
            raise ClosetError(f"Duplicate item ids in {path}: {', '.join(sorted(duplicates))}")

        return cls(items, path=path)

    @classmethod
    def from_seed(cls, seed_path: Path, target_path: Path) -> "Closet":
        """Copy the bundled seed closet to ``target_path`` and load it."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed_path, target_path)
        return cls.load(target_path)

    # ---- persistence ----------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Atomically write the closet. Items are sorted by id for stable diffs."""
        target = Path(path or self.path or "")
        if not str(target):
            raise ClosetError("No path to save to; pass one explicitly.")
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "items": [
                item.model_dump(mode="json", exclude_none=False)
                for item in sorted(self._items.values(), key=lambda i: i.id)
            ],
        }
        text = json.dumps(payload, indent=2, default=_serialize) + "\n"

        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".closet-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        self.path = target
        return target

    # ---- access ---------------------------------------------------------

    @property
    def items(self) -> list[ClosetItem]:
        return sorted(self._items.values(), key=lambda i: i.id)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ClosetItem]:
        return iter(self.items)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._items

    def get(self, item_id: str) -> ClosetItem:
        try:
            return self._items[item_id]
        except KeyError:
            raise ClosetError(f"No item with id {item_id!r}.") from None

    def find(self, item_id: str) -> ClosetItem | None:
        return self._items.get(item_id)

    def by_category(self, category: Category) -> list[ClosetItem]:
        return [i for i in self.items if i.category == category]

    def category_counts(self) -> dict[str, int]:
        counts = Counter(i.category.value for i in self._items.values())
        return {c.value: counts.get(c.value, 0) for c in Category}

    def wearable(self) -> list[ClosetItem]:
        """Items not marked for retirement - what recommendation should consider."""
        from wardrobe_agents.schemas import Condition

        return [i for i in self.items if i.condition != Condition.RETIRE]

    # ---- mutation -------------------------------------------------------

    def next_id(self, category: Category) -> str:
        """Allocate the next free ``<category>-NNN`` id."""
        prefix = category.value
        used = {
            int(m.group("num"))
            for item_id in self._items
            if (m := _ID_RE.match(item_id)) and m.group("prefix") == prefix
        }
        n = 1
        while n in used:
            n += 1
        return f"{prefix}-{n:03d}"

    def add(self, item: ClosetItem, *, overwrite: bool = False) -> ClosetItem:
        if item.id in self._items and not overwrite:
            raise ClosetError(f"Item {item.id!r} already exists. Pass overwrite=True to replace it.")
        self._items[item.id] = item
        return item

    def add_all(self, items: Iterable[ClosetItem], *, overwrite: bool = False) -> list[ClosetItem]:
        return [self.add(item, overwrite=overwrite) for item in items]

    def remove(self, item_id: str) -> ClosetItem:
        try:
            return self._items.pop(item_id)
        except KeyError:
            raise ClosetError(f"No item with id {item_id!r}.") from None

    def mark_worn(self, item_ids: Iterable[str], when: date | None = None) -> list[ClosetItem]:
        """Update ``last_worn``. The one computed-ish field that legitimately
        belongs on the item record, because it is a fact about the garment."""
        when = when or date.today()
        touched = []
        for item_id in item_ids:
            item = self.get(item_id)
            updated = item.model_copy(update={"last_worn": when})
            self._items[item_id] = updated
            touched.append(updated)
        return touched

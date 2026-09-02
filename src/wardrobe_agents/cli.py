"""``wardrobe`` command line.

Subcommands map onto the two stages:

- ``add`` / ``list`` / ``show``  - cataloguing, the shared dataset
- ``recommend``                  - stage 1 (weather -> stylist)
- ``score`` / ``outfits``        - the compatibility substrate
- ``suggest-buy``                - stage 2 (scoring -> enumeration -> optimizer)

Everything that reasons lives in the agents and the compatibility package; this
module only parses arguments, resolves the closet, and formats output.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from config.settings import settings
from wardrobe_agents import __version__
from wardrobe_agents.agents.cataloguing import (
    CataloguingAgent,
    CatalogueRequest,
    CataloguingError,
    ItemDraft,
)
from wardrobe_agents.agents.weather import WeatherError
from wardrobe_agents.closet import Closet, ClosetError
from wardrobe_agents.compatibility.enumeration import enumerate_outfits
from wardrobe_agents.llm import LLMError, MissingAPIKeyError, StructuredLLM
from wardrobe_agents.orchestrator import (
    load_candidates,
    load_or_build_compatibility,
    rebuild_compatibility,
    run_recommend,
    run_suggest_buy,
)
from wardrobe_agents.schemas import (
    Category,
    ClosetItem,
    Condition,
    FORMALITY_LABELS,
    WARMTH_LABELS,
    WeatherConstraints,
)

app = typer.Typer(
    help="Outfit recommendations and purchase optimization over one cataloged closet.",
    no_args_is_help=True,
    add_completion=False,
)

_state: dict[str, Any] = {"closet_path": None}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _resolve_closet() -> tuple[Closet, bool]:
    """Load the working closet, falling back to the bundled seed.

    The fallback is what makes the CLI runnable with zero setup: stage 2 works
    end to end on the seed data with no API key and no state.
    """
    explicit = _state.get("closet_path")
    path = Path(explicit) if explicit else Path(settings.closet_path)

    if path.exists():
        try:
            return Closet.load(path), False
        except ClosetError as exc:
            _err(str(exc))

    if explicit:
        _err(f"No closet at {path}. Create one with `wardrobe init` or `wardrobe add`.")

    seed = Path(settings.seed_closet_path)
    if not seed.exists():
        _err(f"No closet at {path}, and no seed closet at {seed}.")
    typer.secho(
        f"No closet at {path} - using the bundled seed closet ({seed}). "
        "Run `wardrobe init` to start your own.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    return Closet.load(seed), True


def _writable_closet() -> Closet:
    """A closet that is safe to write to; never the read-only bundled seed."""
    closet, is_seed = _resolve_closet()
    if not is_seed:
        return closet
    target = Path(settings.closet_path)
    typer.secho(f"Copying the seed closet to {target} so it can be modified.", fg=typer.colors.YELLOW)
    return Closet.from_seed(Path(settings.seed_closet_path), target)


def _format_item(item: ClosetItem) -> str:
    worn = item.last_worn.isoformat() if item.last_worn else "never"
    return (
        f"{item.id:<16} {item.label:<34} {item.category.value:<10} "
        f"f{item.formality} w{item.warmth}  {item.condition.value:<10} worn: {worn}"
    )


def _print_draft(draft: ItemDraft, index: int, total: int) -> None:
    item = draft.item
    typer.echo("")
    typer.secho(f"[{index}/{total}] {item.id} - {item.label}", bold=True)
    if draft.photo_path:
        typer.echo(f"  from       {draft.photo_path}")
    typer.echo(f"  category   {item.category.value} / {item.subcategory}")
    typer.echo(f"  colors     {', '.join(item.colors)}    pattern: {item.pattern}")
    typer.echo(f"  fabric     {item.fabric}")
    typer.echo(f"  warmth     {item.warmth} ({WARMTH_LABELS[item.warmth]})")
    typer.echo(f"  formality  {item.formality} ({FORMALITY_LABELS[item.formality]})")
    typer.echo(f"  condition  {item.condition.value}")
    if item.notes:
        typer.echo(f"  notes      {item.notes}")
    if draft.confidence is not None:
        typer.echo(f"  confidence {draft.confidence:.0%}")
    if draft.uncertain_fields:
        typer.secho(f"  check      {', '.join(draft.uncertain_fields)}", fg=typer.colors.YELLOW)


_EDITABLE = {
    "subcategory": str,
    "colors": list,
    "pattern": str,
    "fabric": str,
    "warmth": int,
    "formality": int,
    "condition": str,
    "notes": str,
}


def _confirm_draft(draft: ItemDraft) -> ClosetItem | None:
    """Let the user accept, edit, or skip an extracted item before it is saved."""
    item = draft.item
    while True:
        choice = typer.prompt("  [a]ccept, [e]dit, [s]kip", default="a").strip().lower()
        if choice.startswith("s"):
            return None
        if choice.startswith("a"):
            return item
        if not choice.startswith("e"):
            continue

        field = typer.prompt(f"  field to edit ({', '.join(_EDITABLE)})").strip().lower()
        if field not in _EDITABLE:
            typer.secho(f"  Not an editable field: {field}", fg=typer.colors.RED)
            continue
        raw = typer.prompt(f"  new value for {field}").strip()
        kind = _EDITABLE[field]
        value: Any = raw
        if kind is list:
            value = [part.strip() for part in raw.split(",") if part.strip()]
        elif kind is int:
            try:
                value = int(raw)
            except ValueError:
                typer.secho("  Expected a whole number.", fg=typer.colors.RED)
                continue
        try:
            item = item.model_copy(update={field: value})
            item = ClosetItem.model_validate(item.model_dump(mode="json"))
        except Exception as exc:  # pydantic validation, re-prompt rather than crash
            typer.secho(f"  Rejected: {exc}", fg=typer.colors.RED)
            continue
        draft.item = item
        _print_draft(draft, 1, 1)


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


@app.callback()
def main(
    closet: Annotated[
        Optional[Path], typer.Option("--closet", help="Path to the closet JSON file.")
    ] = None,
) -> None:
    """Shared options for every subcommand."""
    _state["closet_path"] = closet


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"wardrobe-agents {__version__}")


@app.command()
def init(
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing closet.")] = False,
) -> None:
    """Copy the bundled seed closet to your working closet path."""
    target = Path(_state.get("closet_path") or settings.closet_path)
    if target.exists() and not force:
        _err(f"{target} already exists. Pass --force to overwrite it.")
    seed = Path(settings.seed_closet_path)
    if not seed.exists():
        _err(f"No seed closet at {seed}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed, target)
    closet = Closet.load(target)
    typer.secho(f"Wrote {len(closet)} seed items to {target}", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
# Cataloguing
# --------------------------------------------------------------------------- #


@app.command()
def add(
    photo: Annotated[
        Optional[list[Path]],
        typer.Option("--photo", help="Photo of an item; repeat for a batch. Uses Claude vision."),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("--file", help="JSON/YAML file of items to import (structured manual entry)."),
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Accept extracted items without confirming.")
    ] = False,
) -> None:
    """Add items to the closet, by photo, by file, or interactively."""
    closet = _writable_closet()
    agent = CataloguingAgent(StructuredLLM(settings), settings)

    if file:
        try:
            result = agent.import_file(Path(file), closet)
        except CataloguingError as exc:
            _err(str(exc))
    elif photo:
        missing = [p for p in photo if not Path(p).exists()]
        if missing:
            _err(f"No such photo: {', '.join(str(m) for m in missing)}")
        llm = StructuredLLM(settings)
        if not llm.is_configured():
            _err(
                f"Photo cataloguing needs the Claude API. Set {settings.api_key_env_var} "
                "in your environment, or add the item with --file / interactively."
            )
        typer.echo(f"Reading {len(photo)} photo(s) with {settings.vision_model()}...")
        try:
            result = agent.run(CatalogueRequest(closet=closet, photo_paths=[Path(p) for p in photo]))
        except (LLMError, CataloguingError) as exc:
            _err(str(exc))
    else:
        result = agent.run(
            CatalogueRequest(closet=closet, manual_entries=[_prompt_for_item()])
        )

    for problem in result.errors:
        typer.secho(f"! {problem}", fg=typer.colors.RED, err=True)
    if not result.drafts:
        _err("Nothing to add.")

    accepted: list[ClosetItem] = []
    for index, draft in enumerate(result.drafts, start=1):
        _print_draft(draft, index, len(result.drafts))
        if draft.needs_confirmation and not yes:
            chosen = _confirm_draft(draft)
            if chosen is None:
                typer.echo("  skipped")
                continue
            accepted.append(chosen)
        else:
            accepted.append(draft.item)

    if not accepted:
        typer.echo("\nNothing saved.")
        raise typer.Exit()

    closet.add_all(accepted)
    path = closet.save()
    typer.secho(
        f"\nAdded {len(accepted)} item(s) to {path} ({len(closet)} total).", fg=typer.colors.GREEN
    )


def _prompt_for_item() -> dict[str, Any]:
    """Interactive structured entry - the same schema as the file path."""
    typer.echo("Adding one item. Press Ctrl-C to abort.\n")
    categories = ", ".join(c.value for c in Category)
    conditions = ", ".join(c.value for c in Condition)
    return {
        "category": typer.prompt(f"category ({categories})"),
        "subcategory": typer.prompt("subcategory (e.g. 'oxford shirt')"),
        "colors": [c.strip() for c in typer.prompt("colors (comma separated)").split(",") if c.strip()],
        "pattern": typer.prompt("pattern", default="solid"),
        "fabric": typer.prompt("fabric"),
        "warmth": typer.prompt(f"warmth 0-5 ({WARMTH_LABELS[0]} .. {WARMTH_LABELS[5]})", type=int),
        "formality": typer.prompt(
            f"formality 1-5 ({FORMALITY_LABELS[1]} .. {FORMALITY_LABELS[5]})", type=int
        ),
        "condition": typer.prompt(f"condition ({conditions})", default="good"),
        "notes": typer.prompt("notes", default="") or None,
    }


@app.command("list")
def list_items(
    category: Annotated[
        Optional[str], typer.Option("--category", "-c", help="Filter by category.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """List everything in the closet."""
    closet, _ = _resolve_closet()
    items = closet.items
    if category:
        try:
            items = closet.by_category(Category(category))
        except ValueError:
            _err(f"Unknown category {category!r}. Use one of: {', '.join(c.value for c in Category)}")

    if as_json:
        typer.echo(json.dumps([i.model_dump(mode="json") for i in items], indent=2))
        return

    if not items:
        typer.echo("No items.")
        return
    for item in items:
        typer.echo(_format_item(item))
    typer.echo(f"\n{len(items)} item(s). " + "  ".join(
        f"{k}:{v}" for k, v in closet.category_counts().items() if v
    ))


@app.command()
def show(
    item_id: Annotated[str, typer.Argument(help="Item id, e.g. top-001.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show one item, and what it can be worn with."""
    closet, _ = _resolve_closet()
    try:
        item = closet.get(item_id)
    except ClosetError as exc:
        _err(str(exc))

    if as_json:
        typer.echo(json.dumps(item.model_dump(mode="json"), indent=2))
        return

    typer.secho(f"{item.id}  {item.label}", bold=True)
    typer.echo(f"  category    {item.category.value} / {item.subcategory}")
    typer.echo(f"  colors      {', '.join(item.colors)}")
    typer.echo(f"  pattern     {item.pattern}")
    typer.echo(f"  fabric      {item.fabric}")
    typer.echo(f"  warmth      {item.warmth} ({WARMTH_LABELS[item.warmth]})")
    typer.echo(f"  formality   {item.formality} ({FORMALITY_LABELS[item.formality]})")
    typer.echo(f"  condition   {item.condition.value}")
    typer.echo(f"  added       {item.date_added.isoformat()}")
    typer.echo(f"  last worn   {item.last_worn.isoformat() if item.last_worn else 'never'}")
    if item.tags:
        typer.echo(f"  tags        {', '.join(item.tags)}")
    if item.notes:
        typer.echo(f"  notes       {item.notes}")

    graph, _cached = load_or_build_compatibility(closet, settings)
    partners = graph.partners(item.id)
    typer.echo(f"\n  wears with {len(partners)} item(s):")
    for partner_id in partners:
        edge = graph.edge(item.id, partner_id)
        assert edge is not None
        typer.echo(f"    {edge.score:.2f}  {partner_id:<16} {closet.get(partner_id).label}")


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #


@app.command()
def recommend(
    location: Annotated[
        Optional[str], typer.Option("--location", "-l", help="City, e.g. 'Berlin, Germany'.")
    ] = None,
    when: Annotated[
        Optional[str], typer.Option("--date", "-d", help="YYYY-MM-DD. Defaults to today.")
    ] = None,
    occasion: Annotated[
        Optional[str], typer.Option("--occasion", "-o", help="e.g. 'client dinner'.")
    ] = None,
    formality: Annotated[
        Optional[int], typer.Option("--formality", "-f", min=1, max=5, help="Target formality 1-5.")
    ] = None,
    count: Annotated[
        Optional[int], typer.Option("--count", "-n", help="How many outfits to return.")
    ] = None,
    temp_min: Annotated[
        Optional[float], typer.Option("--temp-min", help="Skip the forecast: low in C.")
    ] = None,
    temp_max: Annotated[
        Optional[float], typer.Option("--temp-max", help="Skip the forecast: high in C.")
    ] = None,
    conditions: Annotated[
        str, typer.Option("--conditions", help="Free-text conditions when passing temps.")
    ] = "as described",
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Stage 1: recommend outfits for a date and place."""
    closet, _ = _resolve_closet()
    on = date.fromisoformat(when) if when else date.today()

    manual: WeatherConstraints | None = None
    if temp_min is not None or temp_max is not None:
        if temp_min is None or temp_max is None:
            _err("Pass both --temp-min and --temp-max to skip the forecast.")
        manual = _manual_constraints(temp_min, temp_max, conditions, location, on)
    elif not location:
        _err("Pass --location to fetch a forecast, or --temp-min/--temp-max to skip it.")

    try:
        result = run_recommend(
            closet,
            location=location,
            on=on,
            occasion=occasion,
            target_formality=formality,
            count=count,
            constraints=manual,
            settings=settings,
        )
    except MissingAPIKeyError as exc:
        _err(
            f"{exc}\n\nStage 2 (`wardrobe score`, `wardrobe suggest-buy`) runs without a key."
        )
    except (WeatherError, LLMError, ValueError) as exc:
        _err(str(exc))

    if as_json and result.report is not None:
        typer.echo(json.dumps(result.report.model_dump(mode="json"), indent=2))
        return

    c = result.constraints
    typer.secho(c.summary(), bold=True)
    typer.echo(f"Warmth window {c.min_warmth}-{c.max_warmth} - {c.layering_advice} [{c.source}]")
    if c.notes:
        typer.echo(f"Note: {c.notes}")

    for suggestion in result.suggestions:
        typer.echo("")
        typer.secho(f"{suggestion.rank}. {suggestion.name}", bold=True, fg=typer.colors.GREEN)
        for item in suggestion.items:
            typer.echo(f"     {item.id:<16} {item.label}")
        typer.echo(f"   why: {suggestion.rationale}")
        typer.echo(f"   weather: {suggestion.weather_fit}")

    if result.overall_notes:
        typer.echo(f"\n{result.overall_notes}")
    for warning in result.warnings:
        typer.secho(f"! {warning}", fg=typer.colors.YELLOW, err=True)
    if result.report_path:
        typer.secho(f"\nReport: {result.report_path}", fg=typer.colors.BLUE)


def _manual_constraints(
    temp_min: float, temp_max: float, conditions: str, location: str | None, on: date
) -> WeatherConstraints:
    """Build constraints from user-supplied temperatures, skipping the weather agent."""
    from wardrobe_agents.agents.weather import band_for_temp, warmth_for_temp

    return WeatherConstraints(
        date=on,
        location=location or "unspecified",
        temp_min_c=temp_min,
        temp_max_c=temp_max,
        temp_band=band_for_temp(temp_max),
        precipitation_mm=0.0,
        precipitation_probability_pct=0,
        wind_kph=0.0,
        conditions=conditions,
        min_warmth=warmth_for_temp(temp_max),
        max_warmth=warmth_for_temp(temp_min),
        layering_advice="Constraints supplied directly; no forecast was fetched.",
        source="manual",
    )


# --------------------------------------------------------------------------- #
# Compatibility substrate
# --------------------------------------------------------------------------- #


@app.command()
def score(
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Force a recompute.")] = False,
    item_id: Annotated[
        Optional[str], typer.Option("--item", "-i", help="Explain one item's edges.")
    ] = None,
    min_score: Annotated[
        float, typer.Option("--min-score", help="Only show edges at or above this score.")
    ] = 0.0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the whole graph as JSON.")] = False,
) -> None:
    """Rebuild or inspect the pairwise compatibility matrix."""
    closet, _ = _resolve_closet()
    if rebuild:
        graph = rebuild_compatibility(closet, settings)
        cached = False
    else:
        graph, cached = load_or_build_compatibility(closet, settings)

    if as_json:
        typer.echo(json.dumps(graph.to_dict(), indent=2))
        return

    if item_id:
        if item_id not in closet:
            _err(f"No item with id {item_id!r}.")
        typer.secho(f"{item_id}  {closet.get(item_id).label}", bold=True)
        rows = [
            (graph.edge(item_id, other), other)
            for other in graph.item_ids
            if other != item_id
        ]
        for edge, other in sorted(rows, key=lambda r: -(r[0].score if r[0] else 0.0)):
            if edge is None or edge.score < min_score:
                continue
            mark = "yes" if edge.compatible else "no "
            typer.echo(
                f"  {mark} {edge.score:.2f}  {other:<16} {closet.get(other).label:<32} "
                f"{'; '.join(edge.reasons)}"
            )
        return

    compatible = graph.compatible_edges()
    items = closet.wearable()
    outfits = enumerate_outfits(items, graph, settings)
    looks = len(outfits.signatures({i.id: i for i in items}))

    typer.secho("Compatibility matrix", bold=True)
    typer.echo(f"  source            {'cache' if cached else 'recomputed'}")
    typer.echo(f"  items             {len(items)}")
    typer.echo(f"  pairs scored      {len(graph)}")
    typer.echo(f"  compatible pairs  {len(compatible)} ({len(compatible) / max(1, len(graph)):.0%})")
    typer.echo(f"  threshold         {graph.threshold}")
    typer.echo(f"  valid outfits     {outfits.count}{' (truncated)' if outfits.truncated else ''}")
    typer.echo(f"  distinct looks    {looks}")

    incompatible = [e for e in graph.edges if not e.compatible and e.kind != "slot_conflict"]
    if incompatible:
        typer.echo("\n  Pairs ruled out (excluding same-slot):")
        for edge in sorted(incompatible, key=lambda e: e.score)[:10]:
            typer.echo(
                f"    {edge.item_a:<16} {edge.item_b:<16} {edge.kind:<18} {'; '.join(edge.reasons)}"
            )
    typer.echo(f"\n  Cache: {settings.compatibility_cache_path}")


@app.command()
def outfits(
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many to print.")] = 15,
) -> None:
    """List the valid outfits the closet currently supports."""
    closet, _ = _resolve_closet()
    items = closet.wearable()
    graph, _cached = load_or_build_compatibility(closet, settings)
    result = enumerate_outfits(items, graph, settings)
    index = {i.id: i for i in items}
    looks = len(result.signatures(index))

    typer.secho(
        f"{result.count} valid combination(s), {looks} distinct look(s)"
        + (" (truncated)" if result.truncated else ""),
        bold=True,
    )
    for outfit in result.outfits[:limit]:
        typer.echo(f"  {outfit.score:.2f}  " + " + ".join(index[i].label for i in outfit.item_ids))
    if result.count > limit:
        typer.echo(f"  ... {result.count - limit} more")


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #


@app.command("suggest-buy")
def suggest_buy(
    candidates: Annotated[
        Optional[Path], typer.Option("--candidates", "-c", help="Candidate items JSON/YAML.")
    ] = None,
    top: Annotated[Optional[int], typer.Option("--top", "-n", help="How many to show.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Stage 2: rank what to buy next by marginal outfit gain."""
    closet, _ = _resolve_closet()
    path = Path(candidates) if candidates else Path(settings.default_candidates_path)
    if not path.exists():
        _err(f"No candidate file at {path}. Pass --candidates.")
    try:
        catalog = load_candidates(path)
    except (ClosetError, ValueError) as exc:
        _err(str(exc))
    if not catalog:
        _err(f"{path} contains no candidates.")

    result = run_suggest_buy(closet, catalog, settings=settings, candidates_path=path)

    if as_json and result.report is not None:
        typer.echo(json.dumps(result.report.model_dump(mode="json"), indent=2))
        return

    stats = result.optimization.stats
    assert stats is not None
    typer.secho(
        f"Closet: {stats.item_count} items, {stats.compatible_pair_count}/{stats.total_pair_count} "
        f"compatible pairs, {stats.outfit_count} combinations, "
        f"{stats.distinct_look_count} distinct looks.",
        bold=True,
    )
    typer.echo(
        f"Ranking {len(catalog)} candidate(s) by new looks unlocked "
        f"(threshold: {settings.marginal_gain_threshold}).\n"
    )

    colors = {
        "recommended": typer.colors.GREEN,
        "marginal": typer.colors.YELLOW,
        "redundant": typer.colors.RED,
        "not_scored": typer.colors.BLUE,
    }
    for recommendation in result.optimization.recommendations[: top or settings.suggest_buy_top_n]:
        price = f" - {recommendation.candidate.price:.0f}" if recommendation.candidate.price else ""
        typer.secho(
            f"{recommendation.rank}. {recommendation.candidate.label}{price}"
            f"  [{recommendation.verdict}]",
            bold=True,
            fg=colors.get(recommendation.verdict),
        )
        typer.echo(
            f"   +{recommendation.new_outfit_count} looks "
            f"({recommendation.marginal_gain_pct:.0f}%)"
            f"   raw combinations: {recommendation.new_combination_count}"
        )
        typer.echo(f"   {recommendation.explanation}")
        if recommendation.bridged_pairs:
            for bridge in recommendation.bridged_pairs[:3]:
                typer.echo(f"     bridges {bridge.label_a}  <->  {bridge.label_b}")
        typer.echo("")

    if result.report_path:
        typer.secho(f"Report: {result.report_path}", fg=typer.colors.BLUE)


if __name__ == "__main__":  # pragma: no cover
    app()

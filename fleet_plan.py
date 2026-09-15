# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "pyarrow>=16", "numpy", "rasterio", "geopandas", "shapely", "pyogrio",
# ]
# ///
"""Decide what the fleet runs, before the fleet costs anything.

One machine per tile, and the tile list is the artifact rather than a constant
in this file. The driver reads it, checks that the inventory beside it answers
the window and the filters this run asks for, and prints the launch plan with
the inventory's identity attached.

Everything here happens before a single instance starts. That is the whole
design: a window mismatch found after launch has already bought 895 machines.

Two artifacts come out. `fleet_plan.json` is the launch list and the artifact
identities that justify it. `fleet_plan.jsonl` is the coverage screen: one line
per land tile with its strict-land pixel count and its ASTER GED gap share,
for a scheduler deciding what to run and in what order. The screen removes no
tile, and neither of its fields predicts swath coverage or what the prep run
will find.

    uv run fleet_plan.py --out artifacts/fleet_plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aster_ged
import lst_qa
import masks
from land_tiles import read_land_tiles, tile_bounds
from stac_window import (
    DEFAULT_CLOUD_COVER_LT,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
)
from tile_inventory import (
    INVENTORY_SCHEMA_VERSION,
    InventoryError,
    check_manifest,
    provenance,
    read_manifest,
    row_groups_for_tile,
    thermal_rows_for_tile,
)

#: The grid the fleet composites on. Nothing in the plan rasterises at it. It
#: is recorded so a finished tile can be checked against the grid its plan
#: assumed, the way `emissivity_rule` records the pixel rule.
DEFAULT_PIXELS_PER_DEGREE = 3600

#: The grid the coverage screen rasterises on, and the only thing in this
#: module that rasterises at all.
#:
#: A 5 degree tile is 500 by 500 pixels here against 18,000 by 18,000 at the
#: compositing resolution, so the screen costs 1/1296 of the pixels. The
#: figures it produces are shares, and a scheduler reading "this tile is 3%
#: land" does not care about the third decimal place. Raising this would buy
#: precision nothing uses and turn a few minutes into a few hours.
PLANNING_PIXELS_PER_DEGREE = 100


def check_land_parameters(manifest: dict, land_provenance: dict) -> None:
    """Refuse a plan whose tile list and inventory disagree about the land.

    The inventory records the land geometry it was assigned against. If the
    tile list has been rebuilt since, with a different buffer, a different
    Natural Earth release, or a different latitude limit, then the tiles the
    driver is about to launch are not the tiles the scenes were assigned to.

    Raises:
        InventoryError: naming every field that differs.
    """
    problems = []
    pairs = (
        ("buffer_meters", int(land_provenance.get("buffer_meters", 0))),
        ("latitude_limit", int(land_provenance.get("latitude_limit", 0))),
        ("natural_earth_version", land_provenance.get("natural_earth_version", "")),
        ("land_geometry_sha256", land_provenance.get("land_geometry_sha256", "")),
    )
    for field, want in pairs:
        got = manifest.get(field)
        if got != want:
            problems.append(f"{field}: inventory {got!r}, tile list {want!r}")
    if problems:
        joined = "\n  ".join(problems)
        msg = (
            f"the tile list and the inventory were built from different land "
            f"parameters:\n  {joined}\n"
            f"Rebuild both: land_tiles.py first, then usgs_inventory.py."
        )
        raise InventoryError(msg)


def check_mask_artifacts(land_provenance: dict, numobs_uri, land_geometry_uri) -> dict:
    """Refuse a plan whose mask artifacts disagree with the tile list.

    One land geometry answers three questions: which tiles the fleet runs,
    which scenes were assigned to them, and which pixels of a tile carry a
    temperature. `check_land_parameters` ties the first two together. This ties
    the third to both, by the same digest.

    Returns:
        The ASTER GED manifest.

    Raises:
        MaskError: if the buffered geometry artifact is absent.
        GedError: if the NumObs artifact is absent, or was built from a
            different geometry.
        InventoryError: if the shipped geometry is not the one the tile list
            was built from.
    """
    land_geometry_uri = Path(land_geometry_uri)
    if not land_geometry_uri.exists():
        msg = (
            f"no buffered land geometry at {land_geometry_uri}. Write it "
            f"with:\n  uv run land_tiles.py --out {{tile list}} "
            f"--write-geometry {land_geometry_uri}"
        )
        raise masks.MaskError(msg)

    shipped = masks.geometry_checksum(land_geometry_uri)
    expected = land_provenance.get("land_geometry_sha256", "")
    if shipped != expected:
        msg = (
            f"the shipped land geometry is not the one the tile list was "
            f"built from:\n  land_geometry_sha256: geometry {shipped!r}, "
            f"tile list {expected!r}\n"
            f"Rewrite it from the same cache: land_tiles.py --write-geometry."
        )
        raise InventoryError(msg)

    manifest = aster_ged.read_manifest(numobs_uri)
    aster_ged.check_manifest(manifest, land_geometry_sha256=shipped, path=numobs_uri)
    return manifest


def classify_tiles(parquet_file, tiles) -> tuple[dict, dict, list, list]:
    """Split the tile list by what the inventory can answer for each tile.

    A land tile with no row group has no scene the fleet could load. That is a
    fact about the window, not a mismatch: narrow the window or raise the cloud
    bar and some coastal tile runs out of scenes. So it comes out of the launch
    list and is named in the plan, rather than stopping the other 894 machines.

    A tile with no thermal scene comes out for a second reason, and it is not
    about the window. Every scene there is an `OLI_TIRS_L2SR` product, which
    carries no `ST_B10`, so the run would stage every object and write an
    all-nodata composite. 126 of the 895 land tiles are like this and each one
    is an ocean tile holding a small island. Reading the null-count statistics
    costs no column data, so this is free at plan time and about $189 and three
    hours of fleet time if it is skipped.

    Only a tile at zero comes out. A tile that is 90% L2SR still composites
    real temperatures from the other 10%, so the share is reported and the
    machine still launches.

    Returns:
        Rows per tile, thermal rows per tile, the tiles with no scene, and the
        tiles with no thermal scene.
    """
    meta = parquet_file.metadata
    rows_by_tile = {}
    thermal_by_tile = {}
    empty = []
    no_thermal = []
    for name in tiles:
        groups = row_groups_for_tile(parquet_file, name)
        if not groups:
            empty.append(name)
            continue
        thermal = thermal_rows_for_tile(parquet_file, name)
        if not thermal:
            no_thermal.append(name)
            continue
        rows_by_tile[name] = sum(meta.row_group(g).num_rows for g in groups)
        thermal_by_tile[name] = thermal
    return rows_by_tile, thermal_by_tile, empty, no_thermal


def coverage_row(
    tile_id: str,
    *,
    numobs_uri,
    strict_land_geometry_uri=None,
    pixels_per_degree: int = PLANNING_PIXELS_PER_DEGREE,
) -> dict:
    """What one tile holds, for a scheduler deciding what to run and when.

    Two numbers, both measured on the same grid in one place.

    `strict_land_pixels` counts the unbuffered land geometry, the definition a
    published `lst:land_pixels` divides by. The buffered geometry reaches 25 km
    out to sea, so it answers a different question and would rank an island
    chain like a continent.

    `ged_gap_share` divides the ASTER GED gap region by the strict land of the
    same tile, and is absent on a tile with no strict land. `masks.coverage`
    settled that convention for the published item, for the same reason and on
    the same geometry. 0/0 is not zero, and a tile of open sea inside the 25 km
    buffer would otherwise sort beside a continent with no gap at all. A
    scheduler reads the absence, or reads `strict_land_pixels` and gets the
    same answer.

    Over land is the only denominator that ranks tiles: GED has no
    observation over sea either, so a share of the whole tile would mostly
    measure how much sea the tile holds.

    Neither field removes a tile. The emissivity rule takes a pixel for reading
    70 C or hotter inside the gap region, not for being in it, so a tile of
    nothing but gap cells still publishes every ordinary temperature it holds.

    Neither field predicts swath coverage, and neither predicts what the prep
    run will find. A swath is counted from valid observations on the ground,
    over the scene set this tile's window selects. The gap region is a property
    of an emissivity mosaic built from different granules for a different
    purpose. A tile with no gap at all can still hold a WRS path that reaches
    no swath cell.

    Returns:
        One JSON-ready row. Every count is an int and every share present is a
        float in `[0, 1]`. `ged_gap_share` is absent on a tile with no strict
        land, so a reader either divides a real denominator or sees no share at
        all. The counts are always there.
    """
    bbox = tile_bounds(tile_id)
    land = masks.land_mask(
        bbox,
        pixels_per_degree,
        strict_land_geometry_uri or masks.STRICT_LAND_GEOMETRY_URI,
    )
    gap = masks.emissivity_gap(bbox, pixels_per_degree, numobs_uri)
    total = int(land.size)
    land_pixels = int(land.sum())
    gap_on_land = int((gap & land).sum())
    row = {
        "tile_id": tile_id,
        "bbox": [float(v) for v in bbox],
        "planning_pixels_per_degree": int(pixels_per_degree),
        "planning_pixels": total,
        "strict_land_pixels": land_pixels,
        "strict_land_share": land_pixels / total if total else 0.0,
        "ged_gap_pixels_on_land": gap_on_land,
    }
    if land_pixels:
        row["ged_gap_share"] = gap_on_land / land_pixels
    return row


def coverage_rows(
    plan: dict,
    *,
    numobs_uri,
    strict_land_geometry_uri=None,
    pixels_per_degree: int = PLANNING_PIXELS_PER_DEGREE,
    say=print,
) -> list[dict]:
    """One row per land tile, in tile order, with what the plan knows about it.

    Every land tile gets a row, including the ones `classify_tiles` took out of
    the launch list. A tile with no scene in this window is a tile the schedule
    still has to account for, and a row that appears only when a tile is
    runnable makes the artifact's length depend on the window.

    Ordered by tile id and built from the tile list alone, so two runs over the
    same three artifacts write the same bytes.

    Returns:
        The rows. `runnable` says whether the fleet would launch a machine, and
        `scenes` and `thermal_scenes` are None for a tile it would not.
    """
    known = {t["tile_id"]: t for t in plan["tiles"]}
    names = sorted(
        set(known)
        | set(plan["tiles_without_scenes"])
        | set(plan["tiles_without_thermal"])
    )
    rows = []
    for index, name in enumerate(names, start=1):
        row = coverage_row(
            name,
            numobs_uri=numobs_uri,
            strict_land_geometry_uri=strict_land_geometry_uri,
            pixels_per_degree=pixels_per_degree,
        )
        planned = known.get(name)
        rows.append(
            row
            | {
                "runnable": planned is not None,
                "scenes": None if planned is None else planned["scenes"],
                "thermal_scenes": (
                    None if planned is None else planned["thermal_scenes"]
                ),
            }
        )
        if index % 100 == 0 or index == len(names):
            say(f"              screened {index} of {len(names)} tiles")
    return rows


def write_coverage_rows(path: Path, rows) -> Path:
    """The screen as JSON Lines, one tile per line.

    JSON Lines rather than one array, because the consumer is a scheduler that
    filters and sorts 895 rows, and because a line-oriented file diffs by tile
    when a run changes one of them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def build_plan(
    land_tiles_uri: Path | str,
    inventory_uri: Path | str,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    platforms: str = DEFAULT_PLATFORMS,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
    numobs_uri: Path | str | None = None,
    land_geometry_uri: Path | str | None = None,
    pixels_per_degree: int = DEFAULT_PIXELS_PER_DEGREE,
) -> dict:
    """The tiles to launch, with the artifact identities that justify them.

    `numobs_uri` is optional here and required by the CLI. A caller that passes
    nothing gets the inventory checks alone and a plan whose `aster_ged` and
    `emissivity_rule` are null, which says the plan never saw a mosaic. That is
    what the tests of the inventory checks want. The driver always passes it,
    so a launched fleet is always tied to one land geometry across the tile
    list, the inventory, and the two mask artifacts.

    No tile comes out for its emissivity. The pixel rule removes a gap pixel
    for reading 70 C or hotter, not for being a gap pixel, so a tile of nothing
    but gap cells still publishes every ordinary temperature it holds.

    Raises:
        InventoryError: if the artifacts are missing, stale, or disagree with
            each other or with this run's parameters.
        MaskError: if the buffered land geometry is absent.
        GedError: if the NumObs artifact is absent or disagrees about the land.
    """
    import pyarrow.parquet as pq

    tiles, land_provenance = read_land_tiles(land_tiles_uri)
    manifest = read_manifest(inventory_uri)
    check_manifest(
        manifest,
        start=start,
        end=end,
        platforms=platforms,
        cloud_cover_lt=cloud_cover_lt,
        schema_version=INVENTORY_SCHEMA_VERSION,
    )
    check_land_parameters(manifest, land_provenance)

    ged_manifest = None
    if numobs_uri is not None:
        ged_manifest = check_mask_artifacts(
            land_provenance,
            numobs_uri,
            land_geometry_uri or masks.DEFAULT_LAND_GEOMETRY_URI,
        )

    # The land and manifest checks above are what catch a real mismatch, and
    # they have already run. Everything below removes a tile the fleet cannot
    # usefully spend a machine on, and names it in the plan rather than
    # stopping the other 894.
    pf = pq.ParquetFile(inventory_uri)
    rows_by_tile, thermal_by_tile, empty, no_thermal = classify_tiles(pf, tiles)

    # There is no third drop list. An earlier build dropped a tile whose every
    # land pixel sat inside an ASTER emissivity gap, on the reading that such a
    # tile publishes nothing. The mask no longer removes a gap pixel for being
    # one. It removes it for reading 70 C or hotter inside the gap region, so a
    # tile of nothing but gap cells still publishes every ordinary temperature
    # it holds. The screen dropped no tile on the real plan even under the old
    # rule, and under this one it could not.

    runnable = [name for name in tiles if name in rows_by_tile]
    if not runnable:
        msg = (
            f"none of the {len(tiles)} land tiles has a scene with a thermal "
            f"band in this inventory. The window {start} to {end} selects "
            f"nothing, every tile holds OLI_TIRS_L2SR products alone, or the "
            f"artifact was built for a different tile list."
        )
        raise InventoryError(msg)

    return {
        "tile_count": len(runnable),
        "land_tile_count": len(tiles),
        "tiles_without_scenes": empty,
        "tiles_without_thermal": no_thermal,
        "inventory": provenance(manifest),
        "aster_ged": (
            None if ged_manifest is None else aster_ged.provenance(ged_manifest)
        ),
        # The pixel rules every launched machine will apply, recorded so a
        # finished tile can be checked against what was planned. None when the
        # caller passed no mosaic, which says the plan never saw one rather
        # than that no tile needs the rule.
        "emissivity_rule": (
            None
            if ged_manifest is None
            else {
                "gap_buffer_cells": masks.GAP_BUFFER_CELLS,
                "min_total_observations": lst_qa.MIN_TOTAL_OBSERVATIONS,
                "lst_output_min_c": lst_qa.LST_OUTPUT_MIN_C,
                "lst_output_max_c": lst_qa.LST_OUTPUT_MAX_C,
            }
        ),
        "land_tiles_uri": str(land_tiles_uri),
        "inventory_uri": str(inventory_uri),
        "numobs_uri": None if numobs_uri is None else str(numobs_uri),
        "pixels_per_degree": pixels_per_degree,
        "tiles": [
            {
                "tile_id": name,
                "bbox": list(tile_bounds(name)),
                "scenes": rows_by_tile[name],
                "thermal_scenes": thermal_by_tile[name],
            }
            for name in runnable
        ],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--land-tiles-uri", type=Path, default=Path("artifacts/land_tiles.parquet")
    )
    p.add_argument(
        "--inventory-uri",
        type=Path,
        default=Path("artifacts/tile_scene_inventory.parquet"),
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument(
        "--numobs-uri",
        type=Path,
        default=aster_ged.DEFAULT_NUMOBS_URI,
        help="ASTER GED clear-sky observation counts. The plan checks it "
        "against the tile list's land geometry and records the pixel rule "
        "every launched machine will apply",
    )
    p.add_argument(
        "--land-geometry-uri",
        type=Path,
        default=masks.DEFAULT_LAND_GEOMETRY_URI,
        help="the buffered land geometry the pixel mask rasterises, written by "
        "land_tiles.py --write-geometry",
    )
    p.add_argument(
        "--pixels-per-degree",
        type=int,
        default=DEFAULT_PIXELS_PER_DEGREE,
        help="the grid the fleet will composite on. Recorded in the plan so a "
        "finished tile can be checked against it; nothing here rasterises",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/fleet_plan.json"))
    p.add_argument(
        "--out-coverage",
        type=Path,
        default=Path("artifacts/fleet_plan.jsonl"),
        help="one JSON line per land tile with its strict-land pixel count and "
        "its ASTER GED gap share. Scheduling information: nothing here removes "
        "a tile, and neither field predicts swath coverage or what the prep "
        "run will find",
    )
    p.add_argument(
        "--strict-land-geometry-uri",
        type=Path,
        default=masks.STRICT_LAND_GEOMETRY_URI,
        help="the unbuffered land geometry the coverage screen counts, written "
        "by land_tiles.py --write-strict-geometry. This is the definition a "
        "published lst:land_pixels divides by; the buffered geometry reaches "
        "25 km out to sea",
    )
    p.add_argument(
        "--no-coverage",
        action="store_true",
        help="skip the coverage screen, which rasterises 895 tiles",
    )
    args = p.parse_args(argv)

    try:
        plan = build_plan(
            args.land_tiles_uri,
            args.inventory_uri,
            start=args.start,
            end=args.end,
            platforms=args.platforms,
            cloud_cover_lt=args.cloud_cover_lt,
            numobs_uri=args.numobs_uri,
            land_geometry_uri=args.land_geometry_uri,
            pixels_per_degree=args.pixels_per_degree,
        )
    except (InventoryError, masks.MaskError, aster_ged.GedError) as exc:
        print(f"fleet not launched\n{exc}")
        return 1

    scenes = [t["scenes"] for t in plan["tiles"]]
    scenes.sort()
    inv = plan["inventory"]
    print(f"tiles         {plan['tile_count']} to launch, one machine each")
    empty = plan["tiles_without_scenes"]
    if empty:
        print(
            f"              {len(empty)} of {plan['land_tile_count']} land "
            f"tiles hold no scene in this window and are not launched:"
        )
        print(
            f"              {', '.join(empty[:10])}{' ...' if len(empty) > 10 else ''}"
        )
    bare = plan["tiles_without_thermal"]
    if bare:
        print(
            f"              {len(bare)} of {plan['land_tile_count']} land tiles "
            f"hold only OLI_TIRS_L2SR and would composite nothing:"
        )
        print(f"              {', '.join(bare[:10])}{' ...' if len(bare) > 10 else ''}")
    thermal = [t["thermal_scenes"] for t in plan["tiles"]]
    thermal.sort()
    print(
        f"scenes/tile   min {scenes[0]:,}  p50 {scenes[len(scenes) // 2]:,}  "
        f"max {scenes[-1]:,}"
    )
    print(
        f"thermal/tile  min {thermal[0]:,}  p50 {thermal[len(thermal) // 2]:,}  "
        f"max {thermal[-1]:,}"
    )
    print(
        f"land          ne_10m_land, {inv['buffer_meters']} m buffer, "
        f"+/-{inv['latitude_limit']} deg"
    )
    print(f"              sha256 {(inv['land_geometry_sha256'] or '')[:16]}")
    print(f"window        {inv['start']} .. {inv['end']}")
    print(f"inventory     built {inv['inventory_generated_utc']}")
    print(f"              source {inv['source_last_modified']}")
    print(f"              sha256 {(inv['source_sha256'] or '')[:16]}")
    ged = plan["aster_ged"]
    if ged:
        print(
            f"emissivity    {ged['short_name']} v{ged['version']}, "
            f"{ged['granule_count']:,} granules"
        )
        print(f"              built {ged['aster_ged_generated_utc']}")
        print(f"              sha256 {(ged['raster_sha256'] or '')[:16]}")
        rule = plan["emissivity_rule"]
        print(
            f"              gap region grown {rule['gap_buffer_cells']} cell, "
            f"reported and not removed"
        )
        print(
            f"              output: at least {rule['min_total_observations']} "
            f"observations, {rule['lst_output_min_c']:.0f} C to "
            f"{rule['lst_output_max_c']:.0f} C inclusive"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"plan written  {args.out}")

    if args.no_coverage:
        return 0

    print("coverage      screening every land tile for land and emissivity gap")
    try:
        rows = coverage_rows(
            plan,
            numobs_uri=args.numobs_uri,
            strict_land_geometry_uri=args.strict_land_geometry_uri,
        )
    except (masks.MaskError, aster_ged.GedError) as exc:
        print(f"coverage not written\n{exc}")
        return 1
    write_coverage_rows(args.out_coverage, rows)
    report_coverage(rows)
    print(f"screen written {args.out_coverage}")
    return 0


def report_coverage(rows, say=print) -> None:
    """What the screen found, as the shares a scheduler would sort on.

    Reported and not acted on. No tile is removed for its gap share, and the
    share does not predict swath coverage or what the prep run will find.
    """
    if not rows:
        return
    land = sum(row["strict_land_pixels"] for row in rows)
    bare = [row for row in rows if row["strict_land_pixels"] == 0]
    say(f"              {len(rows)} tiles, {land:,} strict-land pixels")
    if bare:
        say(
            f"              {len(bare)} tiles hold no strict-land pixel at this "
            f"resolution. They are in the list because the 25 km buffer reaches "
            f"them, and their islands are under one pixel across"
        )

    # A tile with no land carries no gap share, so it is not in this
    # distribution at all. Counting it as zero would put a cell of open sea
    # beside a continent with no gap, which is the reading the absent field
    # exists to prevent.
    scored = [row for row in rows if "ged_gap_share" in row]
    if not scored:
        say("              no tile holds strict land, so no gap share is defined")
        return
    gaps = sorted(row["ged_gap_share"] for row in scored)
    heavy = [row for row in scored if row["ged_gap_share"] > 0.40]
    heavy_land = sum(row["strict_land_pixels"] for row in heavy)

    def under(limit: float) -> int:
        return sum(1 for gap in gaps if gap < limit)

    say(
        f"              GED gap over land, on the {len(scored)} tiles that hold "
        f"land: {100 * under(0.05) / len(scored):.0f}% under 5%, "
        f"{100 * under(0.20) / len(scored):.0f}% under 20%, "
        f"p50 {100 * gaps[len(gaps) // 2]:.1f}%"
    )
    say(
        f"              {len(heavy)} tiles above 40%, holding "
        f"{100 * heavy_land / land if land else 0:.1f}% of the land. Reported, "
        f"not excluded"
    )


if __name__ == "__main__":
    raise SystemExit(main())

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

    uv run fleet_plan.py --out artifacts/fleet_plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aster_ged
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

#: The grid the fleet composites on. Only the tile-emptying re-check reads it,
#: and only for a tile the coarse screen already emptied, so a plan built at a
#: different resolution from the run it plans errs toward launching a machine
#: rather than toward losing a tile.
DEFAULT_PIXELS_PER_DEGREE = 3600


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
    aster_ged.check_manifest(manifest, land_geometry_sha256=shipped)
    return manifest


def tiles_without_emissivity(
    tiles, *, numobs_uri, land_geometry_uri, pixels_per_degree: int
) -> list[str]:
    """Tiles the output mask empties, so no machine is launched on one.

    A tile can hold thermal scenes and still publish nothing: every land pixel
    can sit inside an ASTER emissivity gap. That is the second kind of ASTER
    coverage loss and the inventory cannot see it. `thermal_rows_for_tile`
    catches the first kind, where the whole footprint has no ST band and USGS
    writes L2SR.

    Two passes, because they trade different mistakes. The screen runs at the
    GED grid itself, 500 by 500 cells for a 5-degree tile, which is fast enough
    for 895 tiles. A cell counts as land only when its centre is land, so a
    coastline thinner than a cell can screen out a tile the real mask would
    keep. Anything the screen empties is therefore re-checked at the run's own
    resolution before it costs a tile. Only a handful reach the second pass.
    """
    empty = []
    for name in tiles:
        bbox = tile_bounds(name)
        _, coarse = masks.output_mask(
            bbox,
            aster_ged.CELLS_PER_DEGREE,
            numobs_uri=numobs_uri,
            land_geometry_uri=land_geometry_uri,
        )
        if coarse["pixels_kept"]:
            continue
        _, fine = masks.output_mask(
            bbox,
            pixels_per_degree,
            numobs_uri=numobs_uri,
            land_geometry_uri=land_geometry_uri,
        )
        if not fine["pixels_kept"]:
            empty.append(name)
    return empty


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
    nothing gets the inventory checks alone and a plan that says so, which is
    what the tests of those checks want. The driver always passes it, so a
    fleet is never launched on tiles the output mask would empty.

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

    # The third reason a tile comes out, and the only one the inventory cannot
    # see. Every land pixel can sit inside an ASTER emissivity gap while the
    # tile still holds thousands of L2SP scenes, because USGS drops the ST band
    # for a whole footprint but writes a within-scene gap as fill.
    no_emissivity = []
    if numobs_uri is not None:
        no_emissivity = tiles_without_emissivity(
            [name for name in tiles if name in rows_by_tile],
            numobs_uri=numobs_uri,
            land_geometry_uri=land_geometry_uri or masks.DEFAULT_LAND_GEOMETRY_URI,
            pixels_per_degree=pixels_per_degree,
        )
        for name in no_emissivity:
            rows_by_tile.pop(name, None)
            thermal_by_tile.pop(name, None)

    runnable = [name for name in tiles if name in rows_by_tile]
    if not runnable:
        msg = (
            f"none of the {len(tiles)} land tiles has a scene with a thermal "
            f"band in this inventory. The window {start} to {end} selects "
            f"nothing, every tile holds OLI_TIRS_L2SR products alone, or the "
            f"artifact was built for a different tile list, or the output "
            f"mask empties every one of them."
        )
        raise InventoryError(msg)

    return {
        "tile_count": len(runnable),
        "land_tile_count": len(tiles),
        "tiles_without_scenes": empty,
        "tiles_without_thermal": no_thermal,
        "tiles_without_emissivity": no_emissivity,
        "inventory": provenance(manifest),
        "aster_ged": (
            None if ged_manifest is None else aster_ged.provenance(ged_manifest)
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
        help="ASTER GED clear-sky observation counts. The driver drops a tile "
        "whose every land pixel sits in a gap, which the inventory cannot see",
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
        help="the grid the fleet will composite on. Only the re-check of a "
        "tile the coarse screen emptied reads it",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/fleet_plan.json"))
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
    dark = plan["tiles_without_emissivity"]
    if dark:
        print(
            f"              {len(dark)} of {plan['land_tile_count']} land tiles "
            f"hold no land pixel with ASTER emissivity:"
        )
        print(f"              {', '.join(dark[:10])}{' ...' if len(dark) > 10 else ''}")
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"plan written  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

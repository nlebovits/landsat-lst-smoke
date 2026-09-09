# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "geopandas", "shapely", "pyogrio", "pyarrow", "pyproj", "numpy", "duckdb>=1.0",
# ]
# ///
"""What each defect in the shared land method selects, measured.

`land_tiles.py` corrects two defects it inherits from the pixel mask's geometry
in `nlebovits/landsat-lst`. Both mark open ocean as land. `FINDINGS.md` states
what each one costs, and this script is where those numbers come from.

It builds the tile list four ways, from the same Natural Earth release and the
same 25 km Mercator buffer:

* both defects present
* the antimeridian wrap corrected
* the Null Island placeholder dropped
* both corrected, which is production

The difference between two runs is the set of cells one defect adds. Naming the
cells matters more than counting them. An earlier version of `FINDINGS.md`
named `S05E000` and `S05W005` among the Null Island cells, and neither is
reachable: the buffered placeholder is a disc of radius about 0.23 degrees at
the origin, and those two cells start five degrees south of it.

Coverage is the second question. A cell that no Landsat scene reaches is open
ocean as far as the archive is concerned, so this reports how many of the cells
each defect adds have no scenes in the window. That needs the scene inventory,
which it builds against the union of the four tile lists.

    uv run measure_land_defects.py --out artifacts/land_defects.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_tiles import (
    COASTAL_BUFFER_METERS,
    DEFAULT_CACHE_DIR,
    LATITUDE_LIMIT,
    iter_grid,
    select_land_tiles,
)

#: The four runs, as `(label, drop_placeholder, fix_antimeridian)`.
VARIANTS = (
    ("both defects present", False, False),
    ("antimeridian corrected", False, True),
    ("placeholder dropped", True, False),
    ("both corrected", True, True),
)


def tile_lists(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    lat_limit: int = LATITUDE_LIMIT,
) -> dict[str, set[str]]:
    """The tile names each variant selects, keyed by label."""
    out = {}
    for label, drop, fix in VARIANTS:
        tiles = select_land_tiles(
            cache_dir,
            buffer_meters=buffer_meters,
            lat_limit=lat_limit,
            drop_placeholder=drop,
            fix_antimeridian=fix,
        )
        out[label] = {t["tile_id"] for t in tiles}
    return out


def attribute(lists: dict[str, set[str]]) -> dict:
    """Which cells each defect adds, on its own and together.

    A defect's cost is what disappears when it alone is corrected, so each
    figure is a difference against the run where the other defect is unchanged.
    """
    both = lists["both defects present"]
    production = lists["both corrected"]
    return {
        "counts": {label: len(names) for label, names in lists.items()},
        "antimeridian_adds": sorted(both - lists["antimeridian corrected"]),
        "placeholder_adds": sorted(both - lists["placeholder dropped"]),
        "removed_in_total": sorted(both - production),
        "production": sorted(production),
    }


def null_island_reach(buffer_meters: int = COASTAL_BUFFER_METERS) -> dict:
    """The cells the buffered placeholder can reach, from its geometry alone.

    An independent check on `placeholder_adds`. The placeholder is a square
    about 1 km on a side at the origin, so buffering it and intersecting the
    grid says which cells it touches without reference to any other land.
    """
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    fwd = Transformer.from_crs(4326, 3857, always_xy=True)
    inv = Transformer.from_crs(3857, 4326, always_xy=True)
    half = 0.0045  # about 500 m at the equator
    square = box(*fwd.transform(-half, -half), *fwd.transform(half, half))
    disc = transform(lambda x, y: inv.transform(x, y), square.buffer(buffer_meters))
    west, south, east, north = disc.bounds
    return {
        "bounds": [round(v, 4) for v in (west, south, east, north)],
        "cells": sorted(
            name for name, bounds in iter_grid() if box(*bounds).intersects(disc)
        ),
    }


def coverage(
    tile_names: list[str],
    bulk_path: Path | str,
    *,
    start: str,
    end: str,
    platforms: str,
    cloud_cover_lt: int,
) -> dict[str, int]:
    """Scene count per tile, for an arbitrary tile list.

    Built the same way `usgs_inventory` builds the artifact, but returning only
    the counts. A tile with no scenes is a tile the archive never observed.
    """
    import numpy as np

    from usgs_inventory import (
        assign_tiles,
        filter_to_window,
        scan_bulk,
    )

    scanned = scan_bulk(
        bulk_path,
        start=start,
        end=end,
        platforms=platforms,
        cloud_cover_lt=cloud_cover_lt,
    )
    t_start = np.asarray(
        scanned.column("Start Time").to_numpy(zero_copy_only=False),
        dtype="datetime64[us]",
    )
    t_stop = np.asarray(
        scanned.column("Stop Time").to_numpy(zero_copy_only=False),
        dtype="datetime64[us]",
    )
    centre = t_start + (t_stop - t_start) // 2
    import pyarrow as pa

    scanned = scanned.filter(pa.array(filter_to_window(centre, start, end)))

    _, tile_idx = assign_tiles(scanned, tile_names)
    counts = dict.fromkeys(tile_names, 0)
    names = np.asarray(tile_names)
    unique, totals = np.unique(names[tile_idx], return_counts=True)
    counts.update({str(n): int(c) for n, c in zip(unique, totals, strict=True)})
    return counts


def _bulk_path(cache_dir: Path | str) -> Path:
    """The newest cached bulk download. See `measure_scene_centre._bulk_path`."""
    found = sorted(
        Path(cache_dir).glob("LANDSAT_OT_C2_L2-*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        msg = f"no cached USGS bulk file in {cache_dir}. Run usgs_inventory.py first."
        raise FileNotFoundError(msg)
    return found[-1]


def main(argv=None) -> int:
    from stac_window import (
        DEFAULT_CLOUD_COVER_LT,
        DEFAULT_END,
        DEFAULT_PLATFORMS,
        DEFAULT_START,
    )
    from usgs_inventory import DEFAULT_CACHE_DIR as INVENTORY_CACHE_DIR

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--inventory-cache-dir", type=Path, default=INVENTORY_CACHE_DIR)
    p.add_argument("--buffer-meters", type=int, default=COASTAL_BUFFER_METERS)
    p.add_argument("--lat-limit", type=int, default=LATITUDE_LIMIT)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument(
        "--skip-coverage",
        action="store_true",
        help="tile counts only, without the scene inventory",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/land_defects.json"))
    args = p.parse_args(argv)

    lists = tile_lists(
        args.cache_dir, buffer_meters=args.buffer_meters, lat_limit=args.lat_limit
    )
    result = attribute(lists)
    result["null_island_geometry"] = null_island_reach(args.buffer_meters)
    result["grid_cells"] = len(iter_grid(args.lat_limit))

    print(f"grid cells    {result['grid_cells']} inside +/-{args.lat_limit} degrees")
    print()
    print("  tiles  land rule")
    for label, _, _ in VARIANTS:
        print(f"  {result['counts'][label]:>5}  {label}")
    print()
    anti = result["antimeridian_adds"]
    place = result["placeholder_adds"]
    print(f"antimeridian  adds {len(anti)} cells")
    print(f"              {', '.join(anti[:8])}{' ...' if len(anti) > 8 else ''}")
    print(f"placeholder   adds {len(place)} cells: {', '.join(place)}")
    reach = result["null_island_geometry"]
    print(f"              disc bounds {reach['bounds']}")
    print(f"              touches {', '.join(reach['cells'])}")

    if not args.skip_coverage:
        union = sorted(lists["both defects present"] | lists["both corrected"])
        counts = coverage(
            union,
            _bulk_path(args.inventory_cache_dir),
            start=args.start,
            end=args.end,
            platforms=args.platforms,
            cloud_cover_lt=args.cloud_cover_lt,
        )
        result["scene_counts"] = counts
        removed = result["removed_in_total"]
        empty_removed = [n for n in removed if counts.get(n, 0) == 0]
        empty_kept = [n for n in result["production"] if counts.get(n, 0) == 0]
        result["removed_without_coverage"] = empty_removed
        result["kept_without_coverage"] = empty_kept
        print()
        print(f"removed       {len(removed)} cells in total")
        print(f"  no scenes   {len(empty_removed)} of them have no Landsat coverage")
        print(f"  covered     {len(removed) - len(empty_removed)} do have coverage")
        print(f"kept          {len(result['production'])} cells")
        print(f"  no scenes   {len(empty_kept)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwritten       {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

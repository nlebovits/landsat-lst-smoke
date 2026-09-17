"""Select the tiles of one continent out of the fleet plan.

`launch.py` takes tile ids and nothing else. `planner.py` filters on scene
coverage and nothing else. Naming 100 tiles by hand is how a wrong id reaches a
launch, and one wrong id costs a whole instance: the box boots, clones, pulls
229 MB of artifacts, and fails several minutes into billing.

So this module answers one question. Which planned tiles fall inside a named
continent?

The land geometry in `land_tiles.py` cannot answer it. `ne_10m_land` carries no
continent attribute, only polygons. The continent lives on Natural Earth's
admin-0 countries layer, on the same host, under the same 10m scale. This
module reads that layer, unions the countries of one continent, and keeps every
tile in `fleet_plan.json` whose bbox intersects the union.

A bbox over South America also holds Panama, Costa Rica, Jamaica, Hispaniola,
and Barbados. The country polygons put those on other continents and drop them,
which a bbox cannot do.

    uv run lst-fleet-aoi --continent "South America" \
        --exclude S30W065,S50W075,S35W055 \
        --out artifacts/south_america.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapely.ops import unary_union

from lst.land_tiles import DEFAULT_CACHE_DIR, tile_bounds

#: Natural Earth 10m admin-0 countries. Same host and scale as the land layer
#: `land_tiles.NATURAL_EARTH_URL` reads, so one download policy covers both.
NATURAL_EARTH_COUNTRIES_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
)
#: The Natural Earth release the URL serves. Recorded, never parsed.
NATURAL_EARTH_COUNTRIES_VERSION = "ne_10m_admin_0_countries"

#: Version of the continent method: the attribute read and the union rule. It
#: is part of the cache filename, so a cache built by an earlier method cannot
#: be mistaken for one built by this one.
CONTINENT_METHOD_VERSION = 1

#: The attribute Natural Earth spells the continent under.
CONTINENT_FIELD = "CONTINENT"

#: Where `lst-fleet-plan` writes the planned tiles.
DEFAULT_PLAN_PATH = Path("artifacts/fleet_plan.json")


def continent_cache_path(
    cache_dir: Path | str = DEFAULT_CACHE_DIR, *, continent: str
) -> Path:
    """Where one continent's unioned geometry is cached.

    The continent name is part of the filename, lowercased with spaces turned
    to underscores, so `South America` and `North America` cannot collide.
    """
    slug = continent.strip().lower().replace(" ", "_")
    return Path(cache_dir) / f"ne_10m_continent_{slug}_v{CONTINENT_METHOD_VERSION}.gpkg"


def load_continent(continent: str, cache_dir: Path | str = DEFAULT_CACHE_DIR):
    """The unioned country polygons of one continent, in EPSG:4326.

    No buffer. This geometry decides which tiles a fleet runs, and the 25 km
    coastal buffer in `land_tiles.py` has already decided what counts as
    coastal. Buffering again here would pull a neighbouring continent's coast
    across the boundary.

    Args:
        continent: The `CONTINENT` value to keep, matched case-insensitively.
        cache_dir: Where the unioned geometry is cached.

    Returns:
        A GeoDataFrame holding one row, the unioned continent.

    Raises:
        ValueError: The layer holds no country on that continent.
    """
    import geopandas as gpd

    cache_path = continent_cache_path(cache_dir, continent=continent)
    if cache_path.exists():
        return gpd.read_file(cache_path)

    countries = gpd.read_file(NATURAL_EARTH_COUNTRIES_URL)
    countries = countries.to_crs("EPSG:4326")
    if CONTINENT_FIELD not in countries.columns:
        raise ValueError(
            f"{NATURAL_EARTH_COUNTRIES_VERSION} carries no {CONTINENT_FIELD} "
            f"column; it has {sorted(countries.columns)}"
        )

    wanted = continent.strip().casefold()
    match = countries[
        countries[CONTINENT_FIELD].astype(str).str.strip().str.casefold() == wanted
    ]
    if match.empty:
        names = sorted(set(countries[CONTINENT_FIELD].astype(str)))
        raise ValueError(f"no country on continent {continent!r}; layer has {names}")

    merged = gpd.GeoDataFrame(
        {"continent": [continent]},
        geometry=[match.geometry.union_all().buffer(0)],
        crs="EPSG:4326",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(cache_path, driver="GPKG")
    return merged


def planned_tiles(plan_path: Path | str = DEFAULT_PLAN_PATH) -> list[str]:
    """Tile ids `lst-fleet-plan` kept, sorted.

    Reading the plan rather than the 895-tile land list applies the
    `tiles_without_thermal` filter for free. Those 126 tiles carry only
    `OLI_TIRS_L2SR` products, so a run there boots, stages, and writes an
    all-nodata composite.
    """
    plan = json.loads(Path(plan_path).read_text())
    return sorted(entry["tile_id"] for entry in plan["tiles"])


#: Equal-area projection for measuring how much continent a tile holds. World
#: Cylindrical Equal Area. Areas are correct everywhere; shapes are not, which
#: does not matter for a number that only ever gets compared to a threshold.
EQUAL_AREA_CRS = "ESRI:54034"


def land_area_km2(geometry) -> float:
    """Area of a lon/lat geometry, in square kilometres.

    Degrees squared would make a tile at 55 south look a third the size of the
    same land at the equator, so the measurement runs in an equal-area
    projection rather than on the bbox arithmetic.
    """
    import geopandas as gpd

    if geometry.is_empty:
        return 0.0
    series = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS)
    return float(series.area.iloc[0]) / 1e6


def select(
    continent: str,
    *,
    plan_path: Path | str = DEFAULT_PLAN_PATH,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    exclude: frozenset[str] = frozenset(),
    min_land_km2: float = 0.0,
) -> tuple[list[tuple[str, float]], list[str], list[str], list[tuple[str, float]]]:
    """Split the planned tiles into kept, off-continent, excluded, and slivers.

    The tile polygon is its bbox, built from the name rather than from the
    plan's own `bbox` field, so a stale plan cannot move a tile.

    Every kept tile carries the area of continent inside it. That number is
    what makes the boundary reviewable. `N20W065` intersects South America
    only through Isla de Aves, a Venezuelan sandbar of about 0.6 km2, and the
    rest of the tile is the Lesser Antilles. Without the area the tile reads
    like any other.

    Args:
        continent: The `CONTINENT` value to keep.
        plan_path: `fleet_plan.json` to read the planned tiles from.
        cache_dir: Where the unioned continent geometry is cached.
        exclude: Tile ids to drop before any geometry runs.
        min_land_km2: Kept tiles below this area move to the sliver list
            instead. Zero keeps every intersecting tile.

    Returns:
        `(kept, off_continent, excluded, slivers)`. `kept` and `slivers` carry
        `(tile_id, land_km2)` pairs. The four lists partition the planned
        tiles, so their lengths sum to the plan's count.
    """
    from shapely import STRtree
    from shapely.geometry import box

    land = load_continent(continent, cache_dir)
    geometries = list(land.geometry.values)
    tree = STRtree(geometries)

    kept: list[tuple[str, float]] = []
    off: list[str] = []
    dropped: list[str] = []
    slivers: list[tuple[str, float]] = []
    for tile_id in planned_tiles(plan_path):
        if tile_id in exclude:
            dropped.append(tile_id)
            continue
        cell = box(*tile_bounds(tile_id))
        # The exact test runs inside the tree, so a hit is a real intersection
        # rather than an envelope overlap.
        hits = tree.query(cell, predicate="intersects")
        if len(hits) == 0:
            off.append(tile_id)
            continue
        inside = unary_union([geometries[i] for i in hits]).intersection(cell)
        area = land_area_km2(inside)
        if area < min_land_km2:
            slivers.append((tile_id, area))
            continue
        kept.append((tile_id, area))
    return kept, off, dropped, slivers


def parse_exclude(values: list[str] | None) -> frozenset[str]:
    """Tile ids to drop, comma- or space-separated, in any mixture.

    Mirrors `launch.parse_tiles`, which accepts both spellings, so one habit
    works across the fleet commands.
    """
    ids: set[str] = set()
    for value in values or []:
        for piece in value.replace(",", " ").split():
            ids.add(piece.strip().upper())
    return frozenset(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--continent",
        default="South America",
        help="Natural Earth CONTINENT value to keep.",
    )
    parser.add_argument(
        "--plan",
        default=DEFAULT_PLAN_PATH,
        type=Path,
        help="fleet_plan.json to read the planned tiles from.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        type=Path,
        help="Where the unioned continent geometry is cached.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Tile ids to drop, comma- or space-separated. Repeatable.",
    )
    parser.add_argument(
        "--min-land-km2",
        type=float,
        default=0.0,
        help=(
            "Drop a tile holding less continent than this. Default 0, which "
            "keeps every intersecting tile and reports the small ones."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write the kept tile ids here, one per line. Default: stdout only.",
    )
    args = parser.parse_args(argv)

    exclude = parse_exclude(args.exclude)
    kept, off, dropped, slivers = select(
        args.continent,
        plan_path=args.plan,
        cache_dir=args.cache_dir,
        exclude=exclude,
        min_land_km2=args.min_land_km2,
    )

    missing = exclude - set(dropped)
    if missing:
        print(
            f"warning: --exclude names {len(missing)} tile(s) the plan does not "
            f"hold: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )

    planned = len(kept) + len(off) + len(dropped) + len(slivers)
    print(f"continent      {args.continent}")
    print(f"source         {NATURAL_EARTH_COUNTRIES_VERSION}")
    print(f"planned        {planned}")
    print(f"kept           {len(kept)}")
    print(f"off-continent  {len(off)}")
    print(f"excluded       {len(dropped)}  {' '.join(dropped)}")
    print(f"below threshold{len(slivers):>3}  {' '.join(t for t, _ in slivers)}")

    smallest = sorted(kept, key=lambda row: row[1])[:8]
    if smallest:
        print()
        print("smallest kept tiles, by continent area inside the tile:")
        for tile_id, area in smallest:
            print(f"  {tile_id}  {area:12,.1f} km2")

    print()
    print("kept:")
    for tile_id, area in kept:
        print(f"  {tile_id}  {area:12,.1f} km2")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(f"{t}\n" for t, _ in kept))
        print(f"\nwrote {len(kept)} tile ids -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

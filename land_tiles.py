# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "geopandas", "shapely", "pyogrio", "pyarrow", "pyproj",
# ]
# ///
"""The processing footprint: one buffered land geometry, one tile list.

Two questions need the same answer. Which tiles does the fleet run? Which
pixels inside a tile carry a temperature? A tile chosen from one land geometry
and masked with another produces tiles that are entirely nodata, and pixels
that no tile ever visits. So this module holds one geometry and both callers
read it.

The rule comes from `nlebovits/landsat-lst` (`masks.py`), which builds the
pixel mask. Natural Earth 10m land, buffered by 25 km in EPSG:3857, repaired
with `make_valid`. `load_land_polygons` keeps that method, including the
Mercator buffer, because parity with the pixel mask matters more than the
buffer's shape. It corrects two defects the shared method carries, both of
which select open ocean as land. Each correction is documented where it
happens, and `FINDINGS.md` records both.

That buffer is a Mercator buffer, not a geodesic one. EPSG:3857 inflates
distance by `1/cos(lat)`, so 25 km of Mercator is 25 km on the ground at the
equator and about 12.5 km at 60 degrees. `BUFFER_IS_MERCATOR` records that,
and `land_tiles.parquet` carries it into every run. Changing it would move the
pixel mask too, which is a separate decision with its own evidence.

The tile grid matches the production grid in the same repository: 5 degrees,
named for the north edge and the west edge, spanning `(south, north]` and
`[west, east)`. `S30W065` is lat (-35, -30], lon [-65, -60).

    uv run land_tiles.py --out artifacts/land_tiles.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

#: Natural Earth 10m land polygons. The pixel mask reads this same URL.
NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
#: The Natural Earth release the URL serves. Recorded, never parsed.
NATURAL_EARTH_VERSION = "ne_10m_land"

#: Coastal buffer, in metres, applied in EPSG:3857. See the module docstring.
COASTAL_BUFFER_METERS = 25_000
#: True while the buffer is applied in Mercator rather than on the ellipsoid.
BUFFER_IS_MERCATOR = True

#: Latitude limit of the processing grid. Tiles live inside [-60, 60].
LATITUDE_LIMIT = 60
#: Tile edge, in degrees.
TILE_SIZE_DEGREES = 5

#: Schema version of `land_tiles.parquet`. Bump on any column change.
LAND_TILES_SCHEMA_VERSION = 1

#: Version of the geometry method itself: the placeholder filter, the
#: antimeridian buffer, and anything else that changes the polygons for a fixed
#: buffer distance. It is part of the cache filename. Without it a cache built
#: by an earlier method survives a change to that method, and the run silently
#: reads geometry the current code would never produce. Bump on any change to
#: `load_land_polygons` or the functions it calls. Version 1 is the method that
#: produced the 895-tile list. Caches written before this constant existed
#: carry no `_v` in their name, so they cannot be mistaken for it.
LAND_METHOD_VERSION = 1

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "landsat-lst-smoke" / "land"


# --------------------------------------------------------------------------
# The canonical geometry.
# --------------------------------------------------------------------------


def buffered_land_path(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    drop_placeholder: bool = True,
    fix_antimeridian: bool = True,
) -> Path:
    """Where the buffered land geometry is cached.

    Every input that changes the polygons is in the name: the buffer distance,
    the method version, and the two corrections. A run at a different buffer,
    a later method, or one correction disabled cannot read a file built for
    another combination.
    """
    suffix = f"_buf{buffer_meters // 1000}km" if buffer_meters else ""
    defects = ""
    if not drop_placeholder:
        defects += "_keepplaceholder"
    if not fix_antimeridian:
        defects += "_wrapping"
    return Path(cache_dir) / f"ne_10m_land{suffix}_v{LAND_METHOD_VERSION}{defects}.gpkg"


#: Natural Earth documents `scalerank` over 0 to 9. `ne_10m_land` also ships
#: one record at `scalerank` 100: a square about 1 km on a side centred on
#: longitude 0, latitude 0. It is a placeholder, not land. Buffered by 25 km it
#: becomes a disc of radius about 0.23 degrees in the Gulf of Guinea. Three of
#: the cells it reaches are open ocean. `measure_land_defects.py` names them.
MAX_NATURAL_EARTH_SCALERANK = 9


def drop_placeholder_features(land):
    """Remove Natural Earth records that do not describe land.

    Two filters, both data-driven rather than positional. A `scalerank` past
    the documented range marks the Null Island placeholder. An empty or
    zero-area geometry cannot be buffered into anything meaningful.

    Args:
        land: The raw Natural Earth GeoDataFrame in EPSG:4326.

    Returns:
        The same frame without the placeholder records.
    """
    # Shapely's own area, not GeoSeries.area, which warns on a geographic CRS.
    # The test is for a degenerate geometry, not for a real size.
    keep = land.geometry.apply(lambda g: not g.is_empty and g.area > 0)
    if "scalerank" in land.columns:
        rank = land["scalerank"]
        keep &= rank.isna() | (rank <= MAX_NATURAL_EARTH_SCALERANK)
    return land[keep]


#: Half the width of the EPSG:3857 world, in metres. Mercator x is linear in
#: longitude, so a shift of `2 * MERCATOR_X_MAX` is exactly 360 degrees.
MERCATOR_X_MAX = 20037508.342789244


def _buffer_without_wrapping(parts, buffer_meters: int):
    """Buffer land parts in EPSG:3857 without letting the antimeridian wrap.

    Buffering a part that touches the antimeridian pushes its vertices past
    `MERCATOR_X_MAX`. Reprojecting those to EPSG:4326 wraps them: x of
    20,062,508 comes back as longitude -179.78 rather than 180.22. The ring
    then holds vertices at both edges of the world and reads as a polygon
    spanning every longitude.

    Nineteen parts of `ne_10m_land` touch the seam. Three groups of them sit
    inside the latitude band and turn into slivers that circle the planet: the
    Aleutians near 52 degrees north, an island near 9 degrees south, and Fiji
    between 16 and 19 degrees south. Between them they select 68 open-ocean
    cells, measured by `measure_land_defects.py`.

    Mercator x is linear in longitude, so the seam moves with a translation
    and the buffer distance never changes. Parts near the seam are shifted a
    half world east, buffered there, then cut at the seam and shifted back.

    Args:
        parts: A GeoSeries of single land polygons in EPSG:3857.
        buffer_meters: Buffer distance, in EPSG:3857 metres.

    Returns:
        A GeoSeries in EPSG:3857, every part inside the world's x range.
    """
    import geopandas as gpd
    from shapely.affinity import translate
    from shapely.geometry import box

    world = 2 * MERCATOR_X_MAX
    bounds = parts.bounds
    near_seam = (bounds["minx"] <= -MERCATOR_X_MAX + buffer_meters) | (
        bounds["maxx"] >= MERCATOR_X_MAX - buffer_meters
    )

    shifted = parts.copy()
    # Move the seam-touching parts into [0, 2 * MERCATOR_X_MAX), where the
    # buffer cannot cross an edge of the world.
    shifted.loc[near_seam] = shifted.loc[near_seam].apply(
        lambda g: translate(g, xoff=world) if g.bounds[0] < 0 else g
    )
    buffered = shifted.buffer(buffer_meters)

    # Cut every result at the seam and bring the far half home.
    left = box(-MERCATOR_X_MAX, -world, MERCATOR_X_MAX, world)
    right = box(MERCATOR_X_MAX, -world, 3 * MERCATOR_X_MAX, world)
    pieces = []
    for geom in buffered:
        inside = geom.intersection(left)
        if not inside.is_empty:
            pieces.append(inside)
        outside = geom.intersection(right)
        if not outside.is_empty:
            pieces.append(translate(outside, xoff=-world))
    return gpd.GeoSeries(pieces, crs="EPSG:3857")


def load_land_polygons(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    drop_placeholder: bool = True,
    fix_antimeridian: bool = True,
):
    """Natural Earth 10m land, buffered by 25 km, in EPSG:4326.

    The method is `landsat_lst.masks.load_land_polygons`: Natural Earth 10m,
    a buffer in EPSG:3857, and `make_valid` afterwards. Two corrections apply,
    both for defects the shared method carries.

    Natural Earth's placeholder record is dropped first. See
    `drop_placeholder_features`.

    The buffer then runs through `_buffer_without_wrapping`, which keeps the
    antimeridian from folding a coastal buffer into a sliver that circles the
    planet.

    Both corrections can be switched off. That is what makes the tile counts in
    `FINDINGS.md` reproducible rather than asserted: `measure_land_defects.py`
    builds the list with each defect present and reports what it selects.
    Production always runs with both on.

    Args:
        cache_dir: Where the buffered geometry is cached.
        buffer_meters: Coastal buffer, applied in EPSG:3857.
        drop_placeholder: Remove Natural Earth's Null Island record.
        fix_antimeridian: Buffer seam-touching parts without wrapping.

    Returns:
        A GeoDataFrame of buffered land polygons in EPSG:4326.
    """
    import geopandas as gpd

    cache_path = buffered_land_path(
        cache_dir,
        buffer_meters=buffer_meters,
        drop_placeholder=drop_placeholder,
        fix_antimeridian=fix_antimeridian,
    )
    if cache_path.exists():
        return gpd.read_file(cache_path)

    land = gpd.read_file(NATURAL_EARTH_URL)
    land = land.to_crs("EPSG:4326")
    if drop_placeholder:
        land = drop_placeholder_features(land)

    if buffer_meters > 0:
        parts = land.explode(index_parts=False).geometry.to_crs("EPSG:3857")
        if fix_antimeridian:
            buffered = _buffer_without_wrapping(parts, buffer_meters)
        else:
            buffered = parts.buffer(buffer_meters)
        land = gpd.GeoDataFrame(geometry=buffered.to_crs("EPSG:4326"))
        land["geometry"] = land.geometry.make_valid()
        land = land[~land.geometry.is_empty]

    land = land.reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    land.to_file(cache_path, driver="GPKG")
    return land


def land_geometry_checksum(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    drop_placeholder: bool = True,
    fix_antimeridian: bool = True,
) -> str:
    """SHA-256 of the cached buffered geometry, so a run can name what it used.

    Reads the file rather than the polygons. Two runs that quote the same
    digest read the same bytes, which is the claim a manifest needs to make.
    """
    kwargs = {
        "buffer_meters": buffer_meters,
        "drop_placeholder": drop_placeholder,
        "fix_antimeridian": fix_antimeridian,
    }
    path = buffered_land_path(cache_dir, **kwargs)
    if not path.exists():
        load_land_polygons(cache_dir, **kwargs)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_land_geometry(
    path: Path | str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    drop_placeholder: bool = True,
    fix_antimeridian: bool = True,
) -> Path:
    """Ship the buffered geometry as an artifact, beside the tile list.

    The pixel mask is the second caller of this geometry, and it runs on a
    fleet instance. `load_land_polygons` fetches Natural Earth when its cache
    is cold, which puts a download inside a run that
    `tests/test_no_stac_at_runtime.py` requires to be offline, and repeats a
    buffer over ten thousand parts on every machine.

    So the geometry travels with the tile list instead. The cached file is
    copied byte for byte rather than rewritten, because `land_geometry_sha256`
    is a digest of those bytes and a second `to_file` would produce a GeoPackage
    that holds the same polygons under a different digest.

    Returns:
        The path written.
    """
    import shutil

    kwargs = {
        "buffer_meters": buffer_meters,
        "drop_placeholder": drop_placeholder,
        "fix_antimeridian": fix_antimeridian,
    }
    source = buffered_land_path(cache_dir, **kwargs)
    if not source.exists():
        load_land_polygons(cache_dir, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return path


# --------------------------------------------------------------------------
# The grid.
# --------------------------------------------------------------------------


def tile_name(north: int, west: int) -> str:
    """Tile name from its north edge and west edge, as `N40W075`."""
    ns = "N" if north >= 0 else "S"
    ew = "E" if west >= 0 else "W"
    return f"{ns}{abs(north):02d}{ew}{abs(west):03d}"


def tile_bounds(name: str) -> tuple[float, float, float, float]:
    """The `(west, south, east, north)` bbox of a named tile."""
    north = int(name[1:3]) * (1 if name[0] == "N" else -1)
    west = int(name[4:7]) * (1 if name[3] == "E" else -1)
    return (
        float(west),
        float(north - TILE_SIZE_DEGREES),
        float(west + TILE_SIZE_DEGREES),
        float(north),
    )


def iter_grid(
    lat_limit: int = LATITUDE_LIMIT, size: int = TILE_SIZE_DEGREES
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Every grid cell inside the latitude limit, north to south, west to east.

    The north edge runs from `lat_limit` down to `-lat_limit + size`, so the
    southernmost cell's south edge lands exactly on `-lat_limit`. No cell
    crosses the antimeridian: 180 is a grid line, so cells stop at 175 east.
    """
    cells = []
    north = lat_limit
    while north > -lat_limit:
        west = -180
        while west < 180:
            name = tile_name(north, west)
            cells.append((name, tile_bounds(name)))
            west += size
        north -= size
    return cells


# --------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------


def select_land_tiles(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    lat_limit: int = LATITUDE_LIMIT,
    size: int = TILE_SIZE_DEGREES,
    drop_placeholder: bool = True,
    fix_antimeridian: bool = True,
) -> list[dict]:
    """Grid cells whose intersection with the buffered land is non-empty.

    No land-fraction threshold. The 25 km buffer already decides what counts as
    coastal, and a second threshold on top of it would be a second land rule
    that the pixel mask does not share.

    Returns:
        One dict per selected tile, sorted by name, carrying the name, the
        bbox, and the tile polygon as WKB.
    """
    from shapely import STRtree
    from shapely.geometry import box

    land = load_land_polygons(
        cache_dir,
        buffer_meters=buffer_meters,
        drop_placeholder=drop_placeholder,
        fix_antimeridian=fix_antimeridian,
    )
    tree = STRtree(land.geometry.values)

    selected = []
    for name, bounds in iter_grid(lat_limit, size):
        cell = box(*bounds)
        # `predicate="intersects"` runs the exact test inside the tree, so a
        # hit is already a real intersection rather than an envelope overlap.
        # `intersects` is what "non-empty intersection" means: a shared edge
        # alone is a non-empty intersection and the tile is kept.
        if len(tree.query(cell, predicate="intersects")) == 0:
            continue
        selected.append(
            {
                "tile_id": name,
                "west": bounds[0],
                "south": bounds[1],
                "east": bounds[2],
                "north": bounds[3],
                "geometry": cell.wkb,
            }
        )
    selected.sort(key=lambda row: row["tile_id"])
    return selected


def land_tiles_provenance(
    tiles: list[dict],
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
    lat_limit: int = LATITUDE_LIMIT,
    size: int = TILE_SIZE_DEGREES,
    land_checksum: str | None = None,
) -> dict[str, str]:
    """Everything a reader needs to know how this tile list was produced."""
    return {
        "schema_version": str(LAND_TILES_SCHEMA_VERSION),
        "method_version": str(LAND_METHOD_VERSION),
        "natural_earth_url": NATURAL_EARTH_URL,
        "natural_earth_version": NATURAL_EARTH_VERSION,
        "buffer_meters": str(buffer_meters),
        "buffer_is_mercator": str(BUFFER_IS_MERCATOR),
        "buffer_crs": "EPSG:3857",
        "latitude_limit": str(lat_limit),
        "tile_size_degrees": str(size),
        "tile_count": str(len(tiles)),
        "land_geometry_sha256": land_checksum or "",
        "crs": "EPSG:4326",
    }


def write_land_tiles(
    path: Path | str,
    tiles: list[dict],
    provenance: dict[str, str],
) -> Path:
    """Write the tile list as Parquet, with provenance in the file metadata.

    Geometry travels as WKB with the CRS named in the metadata, which is what
    a GeoParquet reader expects to find.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "tile_id": pa.array([t["tile_id"] for t in tiles], pa.string()),
            "west": pa.array([t["west"] for t in tiles], pa.float64()),
            "south": pa.array([t["south"] for t in tiles], pa.float64()),
            "east": pa.array([t["east"] for t in tiles], pa.float64()),
            "north": pa.array([t["north"] for t in tiles], pa.float64()),
            "geometry": pa.array([t["geometry"] for t in tiles], pa.binary()),
        }
    )
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "crs": "EPSG:4326",
                "geometry_types": ["Polygon"],
            }
        },
    }
    meta = {k.encode(): v.encode() for k, v in provenance.items()}
    meta[b"geo"] = json.dumps(geo).encode()
    table = table.replace_schema_metadata(meta)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_land_tiles(path: Path | str) -> tuple[list[str], dict[str, str]]:
    """The tile names and the provenance recorded beside them."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["tile_id"])
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    provenance = {k.decode(): v.decode() for k, v in raw.items() if k != b"geo"}
    return table.column("tile_id").to_pylist(), provenance


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("artifacts/land_tiles.parquet"))
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--buffer-meters", type=int, default=COASTAL_BUFFER_METERS)
    p.add_argument("--lat-limit", type=int, default=LATITUDE_LIMIT)
    p.add_argument(
        "--write-geometry",
        type=Path,
        default=None,
        metavar="PATH",
        help="also copy the buffered land geometry to PATH, so the pixel mask "
        "reads an artifact instead of fetching Natural Earth on a fleet "
        "instance. Conventionally artifacts/land_buffered.gpkg",
    )
    args = p.parse_args(argv)

    checksum = land_geometry_checksum(args.cache_dir, buffer_meters=args.buffer_meters)
    tiles = select_land_tiles(
        args.cache_dir,
        buffer_meters=args.buffer_meters,
        lat_limit=args.lat_limit,
    )
    grid = len(iter_grid(args.lat_limit))
    provenance = land_tiles_provenance(
        tiles,
        buffer_meters=args.buffer_meters,
        lat_limit=args.lat_limit,
        land_checksum=checksum,
    )
    write_land_tiles(args.out, tiles, provenance)

    print(f"grid cells    {grid} inside +/-{args.lat_limit} degrees")
    print(f"land tiles    {len(tiles)}  ({100 * len(tiles) / grid:.1f}% of the grid)")
    print(f"land geometry ne_10m_land, {args.buffer_meters} m Mercator buffer")
    print(f"              method v{LAND_METHOD_VERSION}, sha256 {checksum[:16]}")
    print(f"written       {args.out}")
    if args.write_geometry:
        written = write_land_geometry(
            args.write_geometry, args.cache_dir, buffer_meters=args.buffer_meters
        )
        print(f"              {written} ({written.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

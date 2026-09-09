# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "duckdb>=1.0", "pyarrow>=16", "pyproj", "shapely", "numpy",
#   "requests", "geopandas", "pyogrio", "pystac-client",
# ]
# ///
"""Build the scene inventory once, from the USGS bulk metadata Parquet.

Every tile VM used to open Earth Search and page the same catalogue. Hundreds
of machines repeating one query is hundreds of chances for a public service to
be slow, rate-limited, or down, and the answer is identical every time. So the
catalogue work happens here, once, before the fleet starts.

The source is the USGS Landsat Bulk Metadata Service file for OLI/TIRS
Collection 2 Level 2. It carries every scene in the archive, updated daily, in
one 413 MB download. Paging the same inventory out of Earth Search takes 100
items per request, 1.6 s per request, and about 42 GB of JSON.

What the bulk file does not carry is the STAC asset objects and the `proj:*`
fields the loader reads. Both are exact functions of columns it does carry, and
`tests/test_inventory_parity.py` checks every rule against Earth Search:

* `proj:epsg` is `32600 + UTM Zone`. USGS writes southern scenes in the
  northern zone with a negative northing, so the hemisphere never enters it.
* `proj:shape` and `proj:transform` come from the four corner coordinates,
  reprojected and snapped to the 30 m product grid. The corners carry five
  decimal places, about 1 m, and the snap moves them by well under a 15 m
  half-pixel. `MAX_SNAP_METERS` enforces that margin on every row, so the
  reconstruction is exact rather than close.
* The asset paths follow from `Display ID`, which carries the processing date.
  The STAC item id drops it, so a path built from the item id could name the
  wrong processing version.

Level 2 surface reflectance products (`L2SR`) carry no thermal band. Earth
Search returns them with a `qa_pixel` asset and no `lwir11`, so this file
writes a null thermal href for them and the runtime builds an item with the
same shape. Matching that is the point: the filters here reproduce
`stac_reference.search_items`, and nothing is widened because the bulk file
happens to hold more rows.

    uv run usgs_inventory.py --land-tiles artifacts/land_tiles.parquet \
        --out artifacts/tile_scene_inventory.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from land_tiles import (
    COASTAL_BUFFER_METERS,
    LATITUDE_LIMIT,
    NATURAL_EARTH_VERSION,
    read_land_tiles,
    tile_bounds,
)
from tile_inventory import INVENTORY_SCHEMA_VERSION
from stac_window import (
    DEFAULT_CLOUD_COVER_LT,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
)

#: The USGS bulk metadata file for Landsat 8 and 9 OLI/TIRS Collection 2
#: Level 2. Updated daily. Landsat 4, 5 and 7 live in separate files and are
#: out of scope here, because the pipeline runs Landsat 8 and 9.
BULK_URL = (
    "https://landsat.usgs.gov/landsat/metadata_service/"
    "bulk_metadata_files/LANDSAT_OT_C2_L2.parquet.gz"
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "landsat-lst-smoke" / "inventory"

#: Ground sample distance of the Level 2 product grid, in metres.
PIXEL_METERS = 30.0
#: Product grid origins sit on a 30 m lattice offset by half a pixel.
GRID_PHASE_METERS = 15.0
#: Largest distance a corner may move when it snaps to the product grid. A
#: half pixel is 15 m; anything past a quarter pixel means the corner columns
#: no longer identify the grid cell and the build must stop rather than guess.
MAX_SNAP_METERS = 7.5

#: Distance from a month boundary inside which a computed centre time cannot
#: decide the month on its own. The bulk file truncates the acquisition start
#: and stop to whole seconds; the largest measured gap to the published centre
#: was 1.117 s over 360 items, so 2 s carries a margin.
MONTH_BOUNDARY_GUARD_SECONDS = 2.0

#: Landsat platform number to the STAC `platform` string.
PLATFORM_BY_SATELLITE = {8: "landsat-8", 9: "landsat-9"}
#: The archive directory for OLI/TIRS. Every row in this file is OLI_TIRS.
SENSOR_DIR = "oli-tirs"
#: Level 2 products without a thermal band. Earth Search omits `lwir11`.
NO_THERMAL_DATA_TYPES = ("OLI_TIRS_L2SR",)


# --------------------------------------------------------------------------
# Source acquisition and cache identity.
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_version(url: str = BULK_URL) -> dict:
    """Ask the server what version it is serving, without downloading it.

    `Last-Modified` and `ETag` are what make a cache entry safe to reuse. A
    cache keyed on the URL alone would serve yesterday's archive forever.
    """
    import requests

    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    return {
        "url": url,
        "last_modified": head.headers.get("Last-Modified", ""),
        "etag": head.headers.get("ETag", ""),
        "content_length": head.headers.get("Content-Length", ""),
    }


def _version_key(version: dict) -> str:
    """A short stable name for one published version of the bulk file."""
    parts = (version["url"], version["last_modified"], version["etag"])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def fetch_bulk(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    url: str = BULK_URL,
    refresh: bool = False,
) -> tuple[Path, dict]:
    """The decompressed bulk Parquet on local disk, and what version it is.

    Reuses an unchanged download. The cache name carries a digest of the URL,
    the `Last-Modified` header and the `ETag`, so a new publication lands in a
    new file rather than overwriting the one a finished run quoted.

    Returns:
        The path to the decompressed Parquet, and the version record that
        belongs in the manifest.
    """
    import gzip
    import shutil

    import requests

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    version = source_version(url)
    key = _version_key(version)
    gz_path = cache_dir / f"LANDSAT_OT_C2_L2-{key}.parquet.gz"
    pq_path = cache_dir / f"LANDSAT_OT_C2_L2-{key}.parquet"
    sidecar = cache_dir / f"LANDSAT_OT_C2_L2-{key}.json"

    if refresh:
        for stale in (gz_path, pq_path, sidecar):
            stale.unlink(missing_ok=True)

    if not gz_path.exists():
        t0 = time.perf_counter()
        with requests.get(url, stream=True, timeout=1800) as resp:
            resp.raise_for_status()
            with gz_path.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        version["download_seconds"] = round(time.perf_counter() - t0, 1)

    if not pq_path.exists():
        with gzip.open(gz_path, "rb") as src, pq_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)

    version["compressed_bytes"] = gz_path.stat().st_size
    version["parquet_bytes"] = pq_path.stat().st_size
    if sidecar.exists():
        version["sha256"] = json.loads(sidecar.read_text())["sha256"]
    else:
        version["sha256"] = _sha256(gz_path)
        sidecar.write_text(json.dumps(version, indent=2) + "\n")
    return pq_path, version


# --------------------------------------------------------------------------
# Filtering, in DuckDB. The bulk table never becomes a pandas frame.
# --------------------------------------------------------------------------

#: Only the columns the derivations and the runtime need. Reading the other
#: 29 costs time and memory for nothing.
BULK_COLUMNS = (
    "Display ID",
    "Landsat Scene Identifier",
    "Date Acquired",
    "Start Time",
    "Stop Time",
    "Satellite",
    "Data Type L2",
    "Collection Category",
    "WRS Path",
    "WRS Row",
    "Scene Cloud Cover L1",
    "Product Map Projection L1",
    "UTM Zone",
    "Corner Upper Left Latitude",
    "Corner Upper Left Longitude",
    "Corner Upper Right Latitude",
    "Corner Upper Right Longitude",
    "Corner Lower Left Latitude",
    "Corner Lower Left Longitude",
    "Corner Lower Right Latitude",
    "Corner Lower Right Longitude",
)

#: Corner columns in ring order, so a polygon built from them does not
#: self-intersect. Upper left, upper right, lower right, lower left.
CORNER_RING = ("Upper Left", "Upper Right", "Lower Right", "Lower Left")


def _corner_bounds_sql() -> tuple[str, str]:
    """SQL for the footprint's latitude envelope, as (min_lat, max_lat)."""
    lats = ", ".join(
        f'"Corner {v} Latitude"'
        for v in ("Upper Left", "Upper Right", "Lower Left", "Lower Right")
    )
    return f"least({lats})", f"greatest({lats})"


def scan_bulk(
    parquet_path: Path | str,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    platforms: str = DEFAULT_PLATFORMS,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
    lat_limit: int = LATITUDE_LIMIT,
):
    """Project and filter the bulk table, returning an Arrow table.

    The filters reproduce `stac_reference.search_items`: the collection window,
    the platform list, and `eo:cloud_cover < cloud_cover_lt`. The latitude band
    replaces the per-tile bbox, because every tile lives inside it.

    `Date Acquired` is a `YYYY/MM/DD` string. Comparing it to an ISO literal is
    wrong inside a year, because `/` sorts above `-`: `'2023/03/31' <
    '2023-04-01'` is false. It is parsed to a DATE before any comparison.
    """
    import duckdb

    wanted = {p.strip() for p in platforms.split(",")}
    sats = sorted(sat for sat, name in PLATFORM_BY_SATELLITE.items() if name in wanted)
    if not sats:
        msg = f"no Landsat 8 or 9 platform in {platforms!r}"
        raise ValueError(msg)

    min_lat, max_lat = _corner_bounds_sql()
    cols = ", ".join(f'"{c}"' for c in BULK_COLUMNS)

    sql = f"""
    SELECT {cols}
    FROM read_parquet(?)
    WHERE strptime("Date Acquired", '%Y/%m/%d')
              BETWEEN DATE '{start[:10]}' AND DATE '{end[:10]}'
      AND "Satellite" IN ({", ".join(str(s) for s in sats)})
      AND "Scene Cloud Cover L1" < {cloud_cover_lt}
      AND {min_lat} <= {lat_limit}
      AND {max_lat} >= {-lat_limit}
    """
    con = duckdb.connect()
    try:
        return con.execute(sql, [str(parquet_path)]).to_arrow_table()
    finally:
        con.close()


def count_bulk_rows(parquet_path: Path | str) -> int:
    """Rows in the source file, for the manifest's before-and-after figures."""
    import duckdb

    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(parquet_path)]
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


# --------------------------------------------------------------------------
# Derivations.
# --------------------------------------------------------------------------


def stac_item_id(display_id: str) -> str:
    """The Earth Search item id for a product, from its full display id.

    `LC09_L2SR_087074_20241231_20250102_02_T1` becomes
    `LC09_L2SR_087074_20241231_02_T1`. The processing date is the fifth field
    and STAC drops it. Asset paths keep it, which is why the display id is what
    this module stores.
    """
    parts = display_id.split("_")
    return "_".join(parts[:4] + parts[5:])


def asset_hrefs(display_id: str, path: int, row: int, data_type: str) -> tuple:
    """The S3 URLs for the thermal band and the QA band.

    Returns `(None, qa)` for a product with no thermal band, which is what
    Earth Search returns for `L2SR`.
    """
    year = display_id.split("_")[3][:4]
    base = (
        f"s3://usgs-landsat/collection02/level-2/standard/{SENSOR_DIR}/"
        f"{year}/{path:03d}/{row:03d}/{display_id}/{display_id}"
    )
    thermal = None if data_type in NO_THERMAL_DATA_TYPES else f"{base}_ST_B10.TIF"
    return thermal, f"{base}_QA_PIXEL.TIF"


def _snap(values, phase: float):
    """Nearest point of the 30 m lattice offset by `phase`."""
    import numpy as np

    return np.round((values - phase) / PIXEL_METERS) * PIXEL_METERS + phase


def derive_projection(table):
    """`proj:epsg`, `proj:shape` and `proj:transform` for every row.

    Corners are pixel centres and the transform origin is a pixel edge, so the
    origin sits half a pixel out from the corner envelope. The result is then
    snapped to the product lattice, which removes the metre of noise the five
    decimal places in the corner columns carry.

    Raises:
        ValueError: if any row is not UTM, or if any corner has to move more
            than `MAX_SNAP_METERS` to reach the lattice. Both mean the corner
            columns no longer identify the grid, and a wrong geobox silently
            shifts every pixel it loads.
    """
    import numpy as np
    from pyproj import Transformer

    projection = table.column("Product Map Projection L1").to_numpy(
        zero_copy_only=False
    )
    bad = set(np.unique(projection)) - {"UTM"}
    if bad:
        msg = f"non-UTM products inside the latitude band: {sorted(bad)}"
        raise ValueError(msg)

    zone = table.column("UTM Zone").to_numpy(zero_copy_only=False).astype("int32")
    epsg = 32600 + zone

    corners = [
        (
            table.column(f"Corner {v} Longitude").to_numpy(zero_copy_only=False),
            table.column(f"Corner {v} Latitude").to_numpy(zero_copy_only=False),
        )
        for v in CORNER_RING
    ]

    n = table.num_rows
    xs = np.empty((4, n), dtype="float64")
    ys = np.empty((4, n), dtype="float64")
    for code in np.unique(epsg):
        sel = epsg == code
        tf = Transformer.from_crs(4326, int(code), always_xy=True)
        for i, (lon, lat) in enumerate(corners):
            xs[i, sel], ys[i, sel] = tf.transform(lon[sel], lat[sel])

    min_x, max_x = xs.min(axis=0), xs.max(axis=0)
    min_y, max_y = ys.min(axis=0), ys.max(axis=0)

    raw_ulx = min_x - PIXEL_METERS / 2
    raw_uly = max_y + PIXEL_METERS / 2
    ulx = _snap(raw_ulx, GRID_PHASE_METERS)
    uly = _snap(raw_uly, -GRID_PHASE_METERS)

    drift = float(max(np.abs(ulx - raw_ulx).max(), np.abs(uly - raw_uly).max()))
    if drift > MAX_SNAP_METERS:
        msg = (
            f"a corner moved {drift:.2f} m to reach the 30 m product grid, "
            f"past the {MAX_SNAP_METERS} m limit. The corner columns no "
            f"longer identify the grid cell; do not ship this inventory."
        )
        raise ValueError(msg)

    height = np.rint((max_y - min_y) / PIXEL_METERS).astype("int32") + 1
    width = np.rint((max_x - min_x) / PIXEL_METERS).astype("int32") + 1
    return epsg, height, width, ulx, uly, drift


# --------------------------------------------------------------------------
# Acquisition time.
# --------------------------------------------------------------------------


def _resolve_month_boundaries(centre, item_ids):
    """Pin the acquisition month for the few scenes that straddle one.

    Earth Search publishes the scene centre time to the microsecond. The bulk
    file carries the acquisition start and stop truncated to whole seconds, so
    a centre computed from them sits within about a second of the published
    one. Measured over 360 items, the largest gap was 1.117 s.

    The runtime reads only the month, so that second matters for one scene in
    a hundred thousand: the ones acquired across midnight on the first of a
    month. `MONTH_BOUNDARY_GUARD_SECONDS` is the band where the computed month
    could be wrong. Scenes inside it get their exact time from Earth Search,
    which is a few items over a whole five-year build.

    Returns:
        The centre times, with the ambiguous ones replaced, and the list of
        item ids that needed the lookup.

    Raises:
        RuntimeError: if a scene inside the band cannot be resolved. A wrong
            month silently moves observations between `qa_count` bands, so the
            build stops rather than publish a guess.
    """
    import numpy as np

    guard = np.timedelta64(int(MONTH_BOUNDARY_GUARD_SECONDS * 1e6), "us")
    month_start = centre.astype("datetime64[M]").astype("datetime64[us]")
    next_month = (centre.astype("datetime64[M]") + np.timedelta64(1, "M")).astype(
        "datetime64[us]"
    )
    ambiguous = np.flatnonzero(
        ((centre - month_start) < guard) | ((next_month - centre) < guard)
    )
    if len(ambiguous) == 0:
        return centre, []

    wanted = [item_ids[i] for i in ambiguous]
    from stac_reference import fetch_items_by_id

    found = fetch_items_by_id(wanted)
    missing = [i for i in wanted if i not in found]
    if missing:
        msg = (
            f"{len(missing)} scenes are acquired within "
            f"{MONTH_BOUNDARY_GUARD_SECONDS} s of a month boundary and Earth "
            f"Search did not return them: {missing[:5]}. Their observations "
            f"could land in the wrong qa_count month."
        )
        raise RuntimeError(msg)

    centre = centre.copy()
    for pos, item_id in zip(ambiguous, wanted, strict=True):
        stamp = found[item_id]["properties"]["datetime"].replace("Z", "")
        centre[pos] = np.datetime64(stamp, "us")
    return centre, wanted


# --------------------------------------------------------------------------
# Footprints and scene-to-tile assignment.
# --------------------------------------------------------------------------


def footprint_arrays(table):
    """Scene footprints as `(lons, lats, crossing)`.

    `lons` and `lats` are `(4, n)` in ring order. `crossing` marks the scenes
    whose longitudes span more than 180 degrees, which is the signature of a
    footprint that wraps the antimeridian rather than one that is genuinely
    half a planet wide.
    """
    import numpy as np

    lons = np.stack(
        [
            table.column(f"Corner {v} Longitude").to_numpy(zero_copy_only=False)
            for v in CORNER_RING
        ]
    )
    lats = np.stack(
        [
            table.column(f"Corner {v} Latitude").to_numpy(zero_copy_only=False)
            for v in CORNER_RING
        ]
    )
    crossing = (lons.max(axis=0) - lons.min(axis=0)) > 180.0
    return lons, lats, crossing


def stac_bbox(lons, lats, crossing):
    """Footprint bounding boxes in the STAC convention.

    A box that wraps the antimeridian is written west-of-east, so `west` is
    the smallest positive longitude and `east` the largest negative one. The
    naive envelope of such a footprint spans the whole planet and would make
    every shard look overlapped.
    """
    import numpy as np

    west = lons.min(axis=0).copy()
    east = lons.max(axis=0).copy()
    if crossing.any():
        wrapped = lons[:, crossing]
        positive = np.where(wrapped >= 0, wrapped, np.inf)
        negative = np.where(wrapped < 0, wrapped, -np.inf)
        west[crossing] = positive.min(axis=0)
        east[crossing] = negative.max(axis=0)
    return west, lats.min(axis=0), east, lats.max(axis=0)


def assign_tiles(table, tile_ids):
    """Pair every scene with every land tile its footprint intersects.

    Earth Search answers a `bbox` search by intersecting the item geometry, so
    this does the same: an R-tree candidate step on footprint envelopes, then
    an exact polygon test on the candidates. A pure bbox test would
    over-include the corners of the rotated Landsat quadrilateral, and those
    extra scenes would reach `stac_load` and change nothing except cost.

    A footprint whose longitudes span more than 180 degrees crosses the
    antimeridian. Those are shifted into `[0, 360)` and tested against tiles
    shifted the same way, because the naive envelope of such a footprint spans
    the whole planet.

    Returns:
        Two arrays: the row index of the scene, and the index into `tile_ids`.
    """
    import numpy as np
    from shapely import STRtree, box, intersects, polygons

    lons, lats, crossing = footprint_arrays(table)

    # Tile polygons, once, in both coordinate frames.
    bounds = {name: tile_bounds(name) for name in tile_ids}
    plain = [box(*bounds[name]) for name in tile_ids]
    shifted = [
        box(
            bounds[name][0] + (360.0 if bounds[name][0] < 0 else 0.0),
            bounds[name][1],
            bounds[name][2] + (360.0 if bounds[name][0] < 0 else 0.0),
            bounds[name][3],
        )
        for name in tile_ids
    ]

    scene_idx: list = []
    tile_idx: list = []

    def _pair(rows, scene_lons, tile_geoms):
        if len(rows) == 0:
            return
        tree = STRtree(tile_geoms)
        coords = np.stack([scene_lons, lats[:, rows]], axis=-1).transpose(1, 0, 2)
        closed = np.concatenate([coords, coords[:, :1, :]], axis=1)
        footprints = polygons(closed)
        left, right = tree.query(footprints, predicate="intersects")
        if len(left) == 0:
            return
        keep = intersects(footprints[left], np.asarray(tile_geoms)[right])
        scene_idx.append(rows[left[keep]])
        tile_idx.append(right[keep])

    plain_rows = np.flatnonzero(~crossing)
    _pair(plain_rows, lons[:, plain_rows], plain)

    cross_rows = np.flatnonzero(crossing)
    if len(cross_rows):
        shifted_lons = lons[:, cross_rows].copy()
        shifted_lons[shifted_lons < 0] += 360.0
        _pair(cross_rows, shifted_lons, shifted)

    if not scene_idx:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
    return np.concatenate(scene_idx), np.concatenate(tile_idx)


# --------------------------------------------------------------------------
# The artifact.
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_inventory(  # noqa: C901 - one linear pipeline, read top to bottom
    parquet_path: Path | str,
    land_tiles_path: Path | str,
    out_path: Path | str,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    platforms: str = DEFAULT_PLATFORMS,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
    source_record: dict | None = None,
) -> dict:
    """Filter, derive, assign, and write the tile-scene inventory.

    Returns the manifest, which is also embedded in the Parquet metadata.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    tile_ids, land_provenance = read_land_tiles(land_tiles_path)
    lat_limit = int(land_provenance.get("latitude_limit", LATITUDE_LIMIT))

    scanned = scan_bulk(
        parquet_path,
        start=start,
        end=end,
        platforms=platforms,
        cloud_cover_lt=cloud_cover_lt,
        lat_limit=lat_limit,
    )
    t_scan = time.perf_counter() - t0

    epsg, height, width, ulx, uly, drift = derive_projection(scanned)

    display = scanned.column("Display ID").to_pylist()
    data_type = scanned.column("Data Type L2").to_pylist()
    wrs_path = scanned.column("WRS Path").to_numpy(zero_copy_only=False)
    wrs_row = scanned.column("WRS Row").to_numpy(zero_copy_only=False)

    item_ids = [stac_item_id(d) for d in display]
    hrefs = [
        asset_hrefs(d, int(p), int(r), t)
        for d, p, r, t in zip(display, wrs_path, wrs_row, data_type, strict=True)
    ]
    thermal_href = [h[0] for h in hrefs]
    qa_href = [h[1] for h in hrefs]

    # Earth Search publishes the scene centre time. The bulk file carries the
    # acquisition start and stop, truncated to whole seconds, and the centre
    # falls between them. Only the month is read at runtime, so the few scenes
    # acquired across a month boundary get an exact lookup.
    t_start = np.asarray(
        scanned.column("Start Time").to_numpy(zero_copy_only=False),
        dtype="datetime64[us]",
    )
    t_stop = np.asarray(
        scanned.column("Stop Time").to_numpy(zero_copy_only=False),
        dtype="datetime64[us]",
    )
    centre = t_start + (t_stop - t_start) // 2
    centre, resolved = _resolve_month_boundaries(centre, item_ids)

    scene_idx, tile_idx = assign_tiles(scanned, tile_ids)

    platform = np.array(
        [PLATFORM_BY_SATELLITE[int(s)] for s in scanned.column("Satellite").to_numpy()]
    )
    scene_id = np.asarray(scanned.column("Landsat Scene Identifier").to_pylist())
    cloud = scanned.column("Scene Cloud Cover L1").to_numpy(zero_copy_only=False)
    category = np.asarray(scanned.column("Collection Category").to_pylist())
    ring_lons, ring_lats, crossing = footprint_arrays(scanned)
    min_lon, min_lat, max_lon, max_lat = stac_bbox(ring_lons, ring_lats, crossing)

    tile_name_arr = np.asarray(tile_ids)[tile_idx]
    order = np.lexsort((centre[scene_idx], tile_name_arr))
    scene_idx, tile_idx = scene_idx[order], tile_idx[order]
    tile_name_arr = tile_name_arr[order]

    def take(arr):
        return arr[scene_idx]

    table = pa.table(
        {
            "tile_id": pa.array(tile_name_arr, pa.string()),
            "item_id": pa.array([item_ids[i] for i in scene_idx], pa.string()),
            "display_id": pa.array([display[i] for i in scene_idx], pa.string()),
            "scene_id": pa.array(take(scene_id), pa.string()),
            "datetime": pa.array(take(centre), pa.timestamp("us", tz="UTC")),
            "platform": pa.array(take(platform), pa.string()),
            "collection_category": pa.array(take(category), pa.string()),
            "data_type": pa.array([data_type[i] for i in scene_idx], pa.string()),
            "wrs_path": pa.array(take(wrs_path).astype("int32"), pa.int32()),
            "wrs_row": pa.array(take(wrs_row).astype("int32"), pa.int32()),
            "cloud_cover": pa.array(take(cloud), pa.float64()),
            "bbox_west": pa.array(take(min_lon), pa.float64()),
            "bbox_south": pa.array(take(min_lat), pa.float64()),
            "bbox_east": pa.array(take(max_lon), pa.float64()),
            "bbox_north": pa.array(take(max_lat), pa.float64()),
            "crosses_antimeridian": pa.array(take(crossing), pa.bool_()),
            "proj_epsg": pa.array(take(epsg), pa.int32()),
            "proj_shape_y": pa.array(take(height), pa.int32()),
            "proj_shape_x": pa.array(take(width), pa.int32()),
            "proj_origin_x": pa.array(take(ulx), pa.float64()),
            "proj_origin_y": pa.array(take(uly), pa.float64()),
            "thermal_href": pa.array([thermal_href[i] for i in scene_idx], pa.string()),
            "qa_href": pa.array([qa_href[i] for i in scene_idx], pa.string()),
            # The footprint, so the runtime can emit the same GeoJSON polygon
            # STAC carries. Four corners in ring order, upper left first.
            **{
                f"corner_{axis}{n}": pa.array(
                    (ring_lons if axis == "lon" else ring_lats)[n][scene_idx],
                    pa.float64(),
                )
                for n in range(4)
                for axis in ("lon", "lat")
            },
        }
    )

    manifest = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator_commit": _git_sha(),
        "source": source_record or {"url": str(parquet_path)},
        "start": start,
        "end": end,
        "platforms": platforms,
        "cloud_cover_lt": cloud_cover_lt,
        "collection": "landsat-c2-l2",
        "natural_earth_version": land_provenance.get(
            "natural_earth_version", NATURAL_EARTH_VERSION
        ),
        "buffer_meters": int(
            land_provenance.get("buffer_meters", COASTAL_BUFFER_METERS)
        ),
        "latitude_limit": lat_limit,
        "land_geometry_sha256": land_provenance.get("land_geometry_sha256", ""),
        "tile_count": len(tile_ids),
        "tiles_with_scenes": int(len(set(tile_name_arr.tolist()))),
        "source_rows": count_bulk_rows(parquet_path),
        "scenes_after_filter": scanned.num_rows,
        "tile_scene_rows": table.num_rows,
        "max_snap_meters": round(drift, 4),
        "month_boundary_lookups": len(resolved),
        "scan_seconds": round(t_scan, 1),
    }

    # One row group per tile. A runtime read filters on `tile_id`, and the
    # reader skips any group whose statistics rule the value out, so a
    # single-tile read touches one group instead of the whole file.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {b"manifest": json.dumps(manifest, indent=2).encode()}
    schema = table.schema.with_metadata(meta)

    boundaries = np.flatnonzero(tile_name_arr[1:] != tile_name_arr[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(tile_name_arr)]])
    with pq.ParquetWriter(
        out_path,
        schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=["tile_id", "platform", "collection_category", "data_type"],
        write_statistics=True,
    ) as writer:
        for lo, hi in zip(starts, stops, strict=True):
            chunk = table.slice(int(lo), int(hi - lo)).replace_schema_metadata(meta)
            writer.write_table(chunk, row_group_size=int(hi - lo))

    manifest["row_groups"] = len(starts)
    manifest["artifact_bytes"] = out_path.stat().st_size
    manifest["build_seconds"] = round(time.perf_counter() - t0, 1)
    (out_path.parent / "inventory_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--land-tiles", type=Path, default=Path("artifacts/land_tiles.parquet")
    )
    p.add_argument(
        "--out", type=Path, default=Path("artifacts/tile_scene_inventory.parquet")
    )
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument(
        "--refresh",
        action="store_true",
        help="discard the cached USGS download and fetch the current file",
    )
    args = p.parse_args(argv)

    bulk, version = fetch_bulk(args.cache_dir, refresh=args.refresh)
    print(f"source        {version['url']}")
    print(f"              modified {version['last_modified']}")
    print(
        f"              {version['compressed_bytes'] / 1e6:.0f} MB gz, "
        f"sha256 {version['sha256'][:16]}"
    )

    manifest = build_inventory(
        bulk,
        args.land_tiles,
        args.out,
        start=args.start,
        end=args.end,
        platforms=args.platforms,
        cloud_cover_lt=args.cloud_cover_lt,
        source_record=version,
    )
    print(f"window        {manifest['start']} .. {manifest['end']}")
    print(f"source rows   {manifest['source_rows']:,}")
    print(f"scenes        {manifest['scenes_after_filter']:,} after filtering")
    print(
        f"tiles         {manifest['tile_count']} land, "
        f"{manifest['tiles_with_scenes']} with scenes"
    )
    print(f"rows          {manifest['tile_scene_rows']:,} tile-scene pairs")
    print(f"row groups    {manifest['row_groups']}")
    print(f"month lookups {manifest['month_boundary_lookups']} exact from STAC")
    print(f"snap drift    {manifest['max_snap_meters']} m (limit {MAX_SNAP_METERS} m)")
    print(f"artifact      {args.out} ({manifest['artifact_bytes'] / 1e6:.0f} MB)")
    print(f"build         {manifest['build_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

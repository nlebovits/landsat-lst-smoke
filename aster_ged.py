# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "numpy", "rasterio", "h5py", "earthaccess",
#   "geopandas", "shapely", "pyogrio", "pyarrow",
# ]
# ///
"""ASTER GED observation counts, as one artifact the fleet can read.

Landsat Collection 2 Level-2 Surface Temperature needs a land surface
emissivity value for every pixel, and USGS reads that value from the ASTER
Global Emissivity Dataset. ASTER GED was built from clear-sky ASTER scenes
acquired between 2000 and 2008. Where ASTER never caught clear sky in those
nine years, GED holds no emissivity and USGS produces no surface temperature.
Those pixels are missing in every year of the archive, and no compositing
window recovers them, because the missing input is a static auxiliary dataset
rather than a measurement.

GED ships its own instrument for finding them. Each granule carries an
observation count per cell, and a count of zero is the gap exactly. That is a
direct definition rather than a fill-value heuristic, which is why the mask
reads this rather than inferring the gap from the composite.

This module has two halves, the way `land_tiles.py` does.

The CLI builds the artifact once, on a laptop, before any instance starts. It
fetches the AG1km v003 granules the buffered land geometry touches, reads the
observation count out of each one, and mosaics them into a single global uint8
GeoTIFF at 0.01 degree. It writes a manifest naming every granule and the land
geometry the cell list came from.

The library half reads one tile's window out of that artifact. It opens no
network connection, needs no credentials, and imports neither `earthaccess` nor
`h5py`. A fleet instance runs that half alone.

AG1km, not AG100. The user guide gives AG100 as 1000 x 1000 cells per degree
and AG1km as 100 x 100, so AG1km's cell is 0.01 degree exactly and the whole
granule is the grid the mask wants. AG100 is a hundred times the download and
would then have to be decimated to the same answer.

    uv run python -c "import earthaccess; earthaccess.login(persist=True)"
    uv run aster_ged.py --out artifacts/aster_numobs.tif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from land_tiles import (
    COASTAL_BUFFER_METERS,
    DEFAULT_CACHE_DIR,
    LATITUDE_LIMIT,
    NATURAL_EARTH_VERSION,
)

# --------------------------------------------------------------------------
# The collection. Values come from the ASTER GED User Guide V3 and the LP DAAC
# catalogue entry, not from inspection of a granule.
# --------------------------------------------------------------------------

#: Earthdata short name of the 1 km product.
SHORT_NAME = "AG1km"
#: Collection version. Part of the search and of the manifest.
COLLECTION_VERSION = "003"
DOI = "10.5067/COMMUNITY/ASTER_GED/AG1KM.003"

#: Cells per degree in AG1km, on each axis. A granule is one degree square, so
#: this is also the granule's shape.
CELLS_PER_DEGREE = 100

#: The HDF5 group holding the observation count. The user guide's table names
#: the layer `Observations` / `Number/pixel`, which is a display name rather
#: than a dataset path, so `read_numobs` takes the one dataset in the group.
NUMOBS_GROUP = "Observations"

#: What the product writes where it has nothing. The user guide: "Values of
#: -9999 are assigned to missing or cloudy data."
GED_FILL = -9999

#: The artifact's dtype ceiling. The rule tests `== 0`, so a saturated count
#: cannot change an answer.
NUMOBS_MAX = 255

#: Band 1 is the observation count. Band 2 says whether a granule was placed
#: there at all, and it exists because the two facts are not the same one.
#:
#: A count of zero means ASTER never caught clear sky over that cell, which is
#: a gap. An absent granule means this build did not read one, which is not a
#: statement about the ground. Writing both as 0 in a single band cost 34 tiles
#: on the first real fleet plan: the cells had no granule on disk, the mosaic
#: called them gap, and `fleet_plan` dropped the tiles as unpublishable.
NUMOBS_BAND = 1
COVERAGE_BAND = 2

#: Schema version of `aster_numobs.tif`. The reader owns it, because the reader
#: is what refuses an artifact it cannot interpret.
ASTER_GED_SCHEMA_VERSION = 1

DEFAULT_NUMOBS_URI = Path("artifacts/aster_numobs.tif")
DEFAULT_GRANULE_CACHE = Path.home() / ".cache" / "landsat-lst-smoke" / "aster_ged"

#: `AG1km.{version}.{northwest corner latitude}.{northwest corner
#: longitude}.0010.h5`, per the user guide. Example: AG1km.v003.33.-115.0010.h5
GRANULE_NAME = re.compile(
    r"^AG1km\.v(?P<version>\d+)\.(?P<north>-?\d+)\.(?P<west>-?\d+)\.0010\.h5$"
)

#: The metadata key the manifest travels under, inside the GeoTIFF.
MANIFEST_TAG = "manifest"


class GedError(RuntimeError):
    """The NumObs artifact cannot answer this run. Raised before any compute."""


# --------------------------------------------------------------------------
# The grid. One function per direction, so a placement bug has one place to be.
# --------------------------------------------------------------------------


def mosaic_shape(lat_limit: int = LATITUDE_LIMIT) -> tuple[int, int]:
    """Rows and columns of the global mosaic at `CELLS_PER_DEGREE`."""
    return (2 * lat_limit * CELLS_PER_DEGREE, 360 * CELLS_PER_DEGREE)


def mosaic_transform(lat_limit: int = LATITUDE_LIMIT):
    """The affine of the global mosaic: north-up, origin at (-180, lat_limit)."""
    from rasterio.transform import from_origin

    res = 1.0 / CELLS_PER_DEGREE
    return from_origin(-180.0, float(lat_limit), res, res)


def granule_cell(name: str) -> tuple[int, int] | None:
    """The `(north, west)` corner a granule filename names, or None.

    The filename carries the NORTHWEST corner, so `AG1km.v003.33.-115.0010.h5`
    covers latitude [32, 33] and longitude [-115, -114]. Reading it as the
    southwest corner shifts the whole mask one degree, which is 100 cells, and
    the shift scan in `measure_ged_registration.py` is what would catch it.
    """
    match = GRANULE_NAME.match(name)
    if match is None:
        return None
    return (int(match["north"]), int(match["west"]))


def cell_offset(
    north: int, west: int, lat_limit: int = LATITUDE_LIMIT
) -> tuple[int, int]:
    """Row and column of a cell's northwest corner in the global mosaic."""
    return (
        (lat_limit - north) * CELLS_PER_DEGREE,
        (west + 180) * CELLS_PER_DEGREE,
    )


def cells_for_land(land, *, lat_limit: int = LATITUDE_LIMIT) -> list[tuple[int, int]]:
    """The one-degree cells the buffered land geometry touches.

    Returned as `(north, west)` corners, the pair a granule filename carries,
    so a cell and the granule that covers it are the same key throughout.

    MEASURED against `ne_10m_land` buffered by 25 km: 14,941 of the 43,200
    cells inside +/-60 degrees. The collection holds 24,873 granules in total,
    so fetching land alone is most of a download avoided.
    """
    from shapely import STRtree
    from shapely.geometry import box

    tree = STRtree(land.geometry.values)
    cells = []
    for north in range(lat_limit, -lat_limit, -1):
        for west in range(-180, 180):
            cell = box(west, north - 1, west + 1, north)
            if len(tree.query(cell, predicate="intersects")):
                cells.append((north, west))
    return cells


# --------------------------------------------------------------------------
# Reading a granule.
# --------------------------------------------------------------------------


def read_numobs(path: Path | str):
    """One granule's observation count, as uint8 on the mosaic's terms.

    Two conversions happen here and both are recorded in the manifest, because
    a reader of the artifact alone cannot recover either.

    The source is int16. It is clipped to `NUMOBS_MAX`. The mask tests `== 0`,
    so clipping a large count cannot move a pixel.

    The source fill, -9999, becomes 0. Over land a cell with no observation is
    a gap, which is what the mask removes. Over water the land rule has already
    removed the pixel, so the mapping never decides a water pixel's fate.

    Raises:
        GedError: if the observation group does not hold exactly one dataset.
            The user guide names the layer rather than the dataset, so this
            module reads the group instead of trusting a spelling.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as fh:
        group = fh[NUMOBS_GROUP]
        names = list(group.keys())
        if len(names) != 1:
            msg = (
                f"{path}: group {NUMOBS_GROUP!r} holds {len(names)} datasets "
                f"{names!r}, expected exactly one. The granule layout has "
                f"changed and `read_numobs` has to name the dataset it wants."
            )
            raise GedError(msg)
        raw = np.asarray(group[names[0]][:])

    if raw.shape != (CELLS_PER_DEGREE, CELLS_PER_DEGREE):
        msg = (
            f"{path}: observation count is {raw.shape}, expected "
            f"({CELLS_PER_DEGREE}, {CELLS_PER_DEGREE}). This is not AG1km v003."
        )
        raise GedError(msg)

    counts = raw.astype("int32")
    counts[counts == GED_FILL] = 0
    return np.clip(counts, 0, NUMOBS_MAX).astype("uint8")


# --------------------------------------------------------------------------
# Fetching. The build path only. Nothing here runs on a fleet instance.
# --------------------------------------------------------------------------


def granule_paths(cache_dir: Path | str) -> dict[tuple[int, int], Path]:
    """Every AG1km granule already on disk, keyed by its cell."""
    found: dict[tuple[int, int], Path] = {}
    for path in sorted(Path(cache_dir).glob("AG1km.*.h5")):
        cell = granule_cell(path.name)
        if cell is not None:
            found[cell] = path
    return found


def fetch_granules(
    cells,
    cache_dir: Path | str,
    *,
    lat_limit: int = LATITUDE_LIMIT,
    fetch: bool = True,
) -> dict[tuple[int, int], Path]:
    """Download the granules covering `cells`, skipping what is cached.

    One search covers the whole latitude band rather than one per cell. The
    filename already names the cell, so the selection happens locally and CMR
    is asked once instead of fifteen thousand times.

    A cell with no granule is not an error. ASTER GED covers land, the cell
    list comes from a 25 km buffered coastline, and a cell that is buffer and
    no land has nothing to fetch. The coverage band records which cells this
    build read, so a cell it never read reads as unknown rather than as gap.

    Args:
        cells: The `(north, west)` corners to cover.
        cache_dir: Where granules are kept.
        lat_limit: The band to search.
        fetch: Whether to reach Earthdata for what the cache lacks. With this
            off the build uses the granules already on disk and reports what is
            absent, which is what a machine with no Earthdata Login can do.

    Returns:
        The cached path for every cell a granule was found for.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    have = granule_paths(cache_dir)
    wanted = set(cells)
    missing = wanted - set(have)
    if not missing or not fetch:
        return {cell: have[cell] for cell in wanted if cell in have}

    import earthaccess

    earthaccess.login(strategy="all", persist=True)
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=COLLECTION_VERSION,
        bounding_box=(-180.0, float(-lat_limit), 180.0, float(lat_limit)),
        count=-1,
    )

    by_cell = {}
    for granule in results:
        links = granule.data_links()
        if not links:
            continue
        cell = granule_cell(links[0].rsplit("/", 1)[-1])
        if cell is not None and cell in missing:
            by_cell[cell] = granule

    if by_cell:
        earthaccess.download(list(by_cell.values()), local_path=str(cache_dir))

    have = granule_paths(cache_dir)
    return {cell: have[cell] for cell in wanted if cell in have}


# --------------------------------------------------------------------------
# The mosaic.
# --------------------------------------------------------------------------


def build_mosaic(granules: dict, *, lat_limit: int = LATITUDE_LIMIT):
    """Every granule placed on the global grid, as counts and coverage.

    Returns:
        Two uint8 arrays. The first is the observation count, zero where no
        granule was placed. The second is 1 exactly where one was.

    The second array is the whole point. A cell this build never read is not a
    cell ASTER failed to see, and only the coverage array can tell the reader
    which it is looking at.
    """
    import numpy as np

    rows, cols = mosaic_shape(lat_limit)
    mosaic = np.zeros((rows, cols), dtype="uint8")
    covered = np.zeros((rows, cols), dtype="uint8")
    for (north, west), path in sorted(granules.items()):
        row0, col0 = cell_offset(north, west, lat_limit)
        if not (0 <= row0 <= rows - CELLS_PER_DEGREE):
            continue
        block = read_numobs(path)
        rows_ = slice(row0, row0 + CELLS_PER_DEGREE)
        cols_ = slice(col0, col0 + CELLS_PER_DEGREE)
        mosaic[rows_, cols_] = block
        covered[rows_, cols_] = 1
    return mosaic, covered


def write_numobs(
    path: Path | str,
    mosaic,
    manifest: dict,
    *,
    covered=None,
    lat_limit: int = LATITUDE_LIMIT,
) -> Path:
    """The mosaic as a tiled, compressed GeoTIFF with the manifest inside it.

    Two bands: the observation count, and whether a granule was placed. Passing
    no coverage array declares the whole mosaic covered, which is what a
    hand-built array in a test means.

    The manifest travels in the file's own metadata as well as in a sidecar.
    A raster that names the granules behind it can be checked wherever it
    ends up; a sidecar alone gets separated from its raster.
    """
    import numpy as np
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = mosaic.shape
    if covered is None:
        covered = np.ones((rows, cols), dtype="uint8")
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 2,
        "height": rows,
        "width": cols,
        "crs": "EPSG:4326",
        "transform": mosaic_transform(lat_limit),
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 1,
        "bigtiff": "IF_SAFER",
    }
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mosaic, NUMOBS_BAND)
            dst.write(covered, COVERAGE_BAND)
            dst.descriptions = (
                "ASTER GED clear-sky observations per cell",
                "1 where a granule was read, 0 where none was",
            )
            dst.update_tags(**{MANIFEST_TAG: json.dumps(manifest)})
    return path


def manifest_path(path: Path | str) -> Path:
    """Where the committed copy of the manifest sits, beside the raster."""
    path = Path(path)
    return path.with_name(f"{path.stem}_manifest.json")


def read_manifest(path: Path | str) -> dict:
    """The manifest stored in the raster's own metadata.

    Raises:
        GedError: if the artifact is absent or carries no manifest, naming the
            command that builds it.
    """
    import rasterio

    path = Path(path)
    if not path.exists():
        msg = (
            f"no ASTER GED observation counts at {path}. Build them with:\n"
            f'  uv run python -c "import earthaccess; '
            f'earthaccess.login(persist=True)"\n'
            f"  uv run aster_ged.py --out {path}"
        )
        raise GedError(msg)
    try:
        with rasterio.open(path) as ds:
            raw = ds.tags().get(MANIFEST_TAG)
    except Exception as exc:  # noqa: BLE001 - any reader failure is fatal here
        msg = f"cannot read {path} as a raster: {exc}"
        raise GedError(msg) from exc
    if not raw:
        msg = (
            f"{path} carries no manifest. It was not written by aster_ged, "
            f"or it predates the manifest. Rebuild it."
        )
        raise GedError(msg)
    manifest = json.loads(raw)
    # A file cannot contain its own digest, so the embedded copy leaves
    # `raster_sha256` empty and the sidecar carries it. Merge it back when the
    # sidecar is beside the raster, so a run record quotes a digest rather than
    # an empty string.
    sidecar = manifest_path(path)
    if not manifest.get("raster_sha256") and sidecar.exists():
        try:
            manifest["raster_sha256"] = json.loads(sidecar.read_text()).get(
                "raster_sha256", ""
            )
        except (OSError, ValueError):
            pass
    return manifest


def check_manifest(
    manifest: dict,
    *,
    land_geometry_sha256: str,
    schema_version: int = ASTER_GED_SCHEMA_VERSION,
) -> None:
    """Refuse an artifact this run cannot combine with its other inputs.

    The cell list came from a land geometry. A mask built from one geometry and
    a tile list built from another cover different ground, and the output looks
    finished either way.

    Raises:
        GedError: naming the field, both values, and the fix.
    """
    problems = []
    if manifest.get("schema_version") != schema_version:
        problems.append(
            f"schema_version {manifest.get('schema_version')} != "
            f"{schema_version} expected by this code"
        )
    got = manifest.get("land_geometry_sha256")
    if got != land_geometry_sha256:
        problems.append(
            f"land_geometry_sha256 {got!r} in the artifact, "
            f"{land_geometry_sha256!r} in this run's geometry"
        )
    if problems:
        joined = "\n  ".join(problems)
        msg = (
            f"the ASTER GED artifact does not match this run:\n  {joined}\n"
            f"Rebuild both, land_tiles.py first, then aster_ged.py. Artifact "
            f"generated {manifest.get('generated_utc')} from "
            f"{manifest.get('granule_count')} granules."
        )
        raise GedError(msg)


def provenance(manifest: dict) -> dict:
    """The subset of the manifest that belongs in a run record."""
    collection = manifest.get("collection", {})
    return {
        "aster_ged_schema_version": manifest.get("schema_version"),
        "aster_ged_generated_utc": manifest.get("generated_utc"),
        "aster_ged_generator_commit": manifest.get("generator_commit"),
        "short_name": collection.get("short_name"),
        "version": collection.get("version"),
        "doi": collection.get("doi"),
        "granule_count": manifest.get("granule_count"),
        "land_geometry_sha256": manifest.get("land_geometry_sha256"),
        "raster_sha256": manifest.get("raster_sha256"),
    }


# --------------------------------------------------------------------------
# The runtime read. This is all a fleet instance calls.
# --------------------------------------------------------------------------


def window_for_bbox(path: Path | str, bbox, target_shape, bands=(NUMOBS_BAND,)):
    """One tile's cells, on the tile's own pixel grid.

    A windowed read followed by nearest upsampling. Both grids are EPSG:4326
    and differ only in resolution, so no warp is involved: at 3,600 pixels per
    degree one 0.01 degree cell becomes exactly 36 by 36 pixels, and a tile's
    bounds are whole degrees, so the window lands on cell boundaries with
    nothing to interpolate.

    Args:
        path: The NumObs artifact.
        bbox: `(west, south, east, north)` in EPSG:4326.
        target_shape: `(height, width)` of the tile raster.
        bands: Which bands to read, `NUMOBS_BAND` and `COVERAGE_BAND`.

    Returns:
        One uint8 array of `target_shape` per band requested.

    Raises:
        GedError: if the artifact is absent, or does not cover the bbox.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    path = Path(path)
    if not path.exists():
        read_manifest(path)  # raises with the build command

    west, south, east, north = bbox
    with rasterio.open(path) as ds:
        bounds = ds.bounds
        if west < bounds.left or east > bounds.right:
            msg = (
                f"bbox {bbox} runs outside the artifact's longitude span "
                f"[{bounds.left}, {bounds.right}]"
            )
            raise GedError(msg)
        if south < bounds.bottom or north > bounds.top:
            msg = (
                f"bbox {bbox} runs outside the artifact's latitude span "
                f"[{bounds.bottom}, {bounds.top}]. The mosaic covers "
                f"+/-{int(bounds.top)} degrees."
            )
            raise GedError(msg)
        window = from_bounds(west, south, east, north, transform=ds.transform)
        return tuple(
            ds.read(
                band,
                window=window,
                out_shape=tuple(target_shape),
                resampling=Resampling.nearest,
            )
            for band in bands
        )


def numobs_for_bbox(path: Path | str, bbox, target_shape):
    """One tile's observation counts. See `window_for_bbox`."""
    return window_for_bbox(path, bbox, target_shape, (NUMOBS_BAND,))[0]


def coverage_for_bbox(path: Path | str, bbox, target_shape):
    """True where this build read a granule for the pixel's cell.

    A pixel outside it has no observation count, and no statement either way
    about whether ASTER saw the ground. The mask keeps such a pixel, because a
    gap it cannot demonstrate is not a gap.
    """
    return window_for_bbox(path, bbox, target_shape, (COVERAGE_BAND,))[0] > 0


# --------------------------------------------------------------------------
# Provenance helpers. Copied from `usgs_inventory`, not imported: that module
# pulls DuckDB, and this one runs under its own inline dependency block.
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001 - a build outside a checkout still runs
        return ""
    return out.stdout.strip()


def build_manifest(
    granules: dict,
    *,
    lat_limit: int,
    buffer_meters: int,
    land_geometry_sha256: str,
    cell_count: int,
) -> dict:
    """Everything a reader needs to know how this artifact was produced."""
    rows, cols = mosaic_shape(lat_limit)
    return {
        "schema_version": ASTER_GED_SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator_commit": _git_sha(),
        "collection": {
            "short_name": SHORT_NAME,
            "version": COLLECTION_VERSION,
            "doi": DOI,
        },
        "granules": sorted(path.name for path in granules.values()),
        "granule_count": len(granules),
        # Three counts, because they answer three questions. The cell list is
        # what the land geometry asked for; the granule count is what the
        # collection could answer with; the difference is buffer that holds no
        # ASTER land, and the land mask removes it independently.
        "land_cell_count": cell_count,
        "cells_without_granule": cell_count - len(granules),
        "grid": {
            "cells_per_degree": CELLS_PER_DEGREE,
            "latitude_limit": lat_limit,
            "west": -180,
            "north": lat_limit,
            "shape": [rows, cols],
            "dtype": "uint8",
            "crs": "EPSG:4326",
        },
        "encoding": {
            "source_dtype": "int16",
            "source_fill": GED_FILL,
            "fill_written_as": 0,
            "clipped_at": NUMOBS_MAX,
            "gap_rule": "numobs == 0",
        },
        "natural_earth_version": NATURAL_EARTH_VERSION,
        "buffer_meters": buffer_meters,
        "land_geometry_sha256": land_geometry_sha256,
        "raster_sha256": "",
    }


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    from land_tiles import land_geometry_checksum, load_land_polygons

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_NUMOBS_URI)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--granule-cache", type=Path, default=DEFAULT_GRANULE_CACHE)
    p.add_argument("--buffer-meters", type=int, default=COASTAL_BUFFER_METERS)
    p.add_argument("--lat-limit", type=int, default=LATITUDE_LIMIT)
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="build from the granules already in --granule-cache and report "
        "the cells that have none, rather than reaching NASA Earthdata. A "
        "cell with no granule reads as gap, which the land mask then removes",
    )
    args = p.parse_args(argv)

    land = load_land_polygons(args.cache_dir, buffer_meters=args.buffer_meters)
    checksum = land_geometry_checksum(args.cache_dir, buffer_meters=args.buffer_meters)
    cells = cells_for_land(land, lat_limit=args.lat_limit)
    grid = 2 * args.lat_limit * 360
    print(f"land cells    {len(cells):,} of {grid:,} inside +/-{args.lat_limit} deg")

    granules = fetch_granules(
        cells,
        args.granule_cache,
        lat_limit=args.lat_limit,
        fetch=not args.no_fetch,
    )
    print(f"granules      {len(granules):,} on disk at {args.granule_cache}")
    if len(granules) < len(cells):
        print(
            f"              {len(cells) - len(granules):,} land cells have no "
            f"granule; the coverage band marks them unread"
        )
    if not granules:
        print("no granule fetched; nothing to mosaic")
        return 1

    mosaic, covered = build_mosaic(granules, lat_limit=args.lat_limit)
    manifest = build_manifest(
        granules,
        lat_limit=args.lat_limit,
        buffer_meters=args.buffer_meters,
        land_geometry_sha256=checksum,
        cell_count=len(cells),
    )
    write_numobs(args.out, mosaic, manifest, covered=covered, lat_limit=args.lat_limit)

    # The digest covers the file the manifest is already inside, so it is
    # computed after the write and stored in the sidecar alone. A digest of a
    # file that contains the digest cannot exist.
    manifest["raster_sha256"] = _sha256(args.out)
    manifest_path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")

    import numpy as np

    seen = covered > 0
    gap = int((seen & (mosaic == 0)).sum())
    rows, cols = mosaic.shape
    print(f"mosaic        {cols:,} x {rows:,} uint8, 2 bands, 1/{CELLS_PER_DEGREE} deg")
    print(
        f"              {int(seen.sum()):,} cells read, {gap:,} of them "
        f"({gap / max(int(seen.sum()), 1):.1%}) hold no observation"
    )
    print(
        f"              {int(np.count_nonzero(~seen)):,} cells unread; the mask "
        f"keeps their pixels rather than calling them gap"
    )
    print(f"land geometry {NATURAL_EARTH_VERSION}, sha256 {checksum[:16]}")
    print(f"written       {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"              {manifest_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

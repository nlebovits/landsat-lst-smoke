"""Write the merged composite as COGs inside a Portolan catalog.

The pipeline has always published an encoding contract and never a file. This
module is the file. It turns the two arrays `merge_parts` assembles into a
catalog that the Portolan validator accepts:

    <out-dir>/catalog/
    ├── catalog.json
    ├── AGENTS.md
    ├── README.md
    └── <collection-id>/
        ├── collection.json
        ├── AGENTS.md
        ├── README.md
        ├── thumbnail.png
        ├── items.parquet
        └── <tile-id>/
            ├── <tile-id>.json
            ├── lst_p95.tif
            └── qa_count.tif

One catalog, one collection, one item per tile. `write_catalog` writes the
tile's own directory and then rebuilds the collection, the root catalog, and
the item mirror from every item directory it finds. Writing tile 2 therefore
leaves a collection that describes tiles 1 and 2, whether the two merges ran
back to back or a month apart on different machines. The collection is derived
from the items on disk; nothing accumulates in memory.

Both COGs sit on the item, not on the collection. A Portolan collection carries
a raster asset itself only when it holds exactly one COG; two at collection
level is a conformance error, and modelling the tile as an item is what lets
the 520-tile grid grow without a layout change.

A tile is named for its north and west edges. The name keeps whatever fraction
those edges have, so the 5-degree grid reads `S30W065` and a half-degree
rehearsal tile reads `S32.5W062.5`. Rounding the name would give three
adjacent half-degree tiles one directory between them.

The encoding constants come from `lst_qa`. Nothing here redefines them.

    lst_p95   uint16, 1 band,   scale 0.01, offset -50.0, nodata 0, celsius
    qa_count  uint8, 12 bands,  no scale,   no offset,    no nodata, count

`qa_count` has no nodata value on purpose. A zero means masking removed every
observation for that month, which is a different fact from a masked pixel.
Giving zero a nodata meaning would erase the distinction the band exists for.

Statistics have to reach a reader in its first range request, so they live in
the file's `GDAL_METADATA` tag rather than in a `.aux.xml` sidecar. GDAL writes
that sidecar whenever it computes statistics itself, so every write and every
read here runs under `GDAL_PAM_ENABLED=NO` and the values are computed with
numpy instead.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lst_qa import LST_MAX_DN, LST_MIN_DN, LST_NODATA_DN, LST_OFFSET, LST_SCALE
from stac_window import DEFAULT_COLLECTION, DEFAULT_PLATFORMS

#: Bumped when the catalog's shape or the encoding changes. Recorded in the
#: collection so a reader can tell which writer produced a tree it finds.
CATALOG_SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# COG creation settings. `conversion-defaults.md` in the Portolan spec records
# where each of these comes from.
# --------------------------------------------------------------------------

#: OGC 21-026 asks for square internal tiles no larger than a screen viewport
#: and recommends a power of two. 512 is also the size the overview
#: requirement keys off: a raster wider or taller than this needs overviews.
BLOCK_SIZE = 512
#: Lossless and readable everywhere. The composite is measured data, so a lossy
#: codec is not an option.
COMPRESSION = "DEFLATE"
#: Horizontal differencing. Both bands are integers, where it shrinks the file.
PREDICTOR = "STANDARD"
#: Both bands are continuous: a temperature and an observation count. GDAL
#: skips nodata pixels when averaging, so the reserved DN 0 never drags an
#: overview pixel toward -50 C.
OVERVIEW_RESAMPLING = "AVERAGE"

#: Band order of the qa_count asset. January is band 1.
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# --------------------------------------------------------------------------
# STAC vocabulary. Every extension URI is pinned to the version the Portolan
# profile registry names; a validator compares them against that registry.
# --------------------------------------------------------------------------

STAC_VERSION = "1.1.0"
PORTOLAN_EXTENSION = "https://schemas.portolan-sdi.org/portolan/v0.2.0/schema.json"
FILE_EXTENSION = "https://stac-extensions.github.io/file/v2.1.0/schema.json"
PROJECTION_EXTENSION = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
RENDER_EXTENSION = "https://stac-extensions.github.io/render/v2.0.0/schema.json"
RASTER_EXTENSION = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"
SCIENTIFIC_EXTENSION = "https://stac-extensions.github.io/scientific/v1.0.0/schema.json"
#: Absent from the Portolan profile registry, which is allowed. PTL-CNF-004
#: governs only what the registry lists: "Extensions absent from the registry
#: are ignored: the registry governs what it lists, not what a publisher adds."
PROCESSING_EXTENSION = "https://stac-extensions.github.io/processing/v1.2.0/schema.json"

COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
JSON_MEDIA_TYPE = "application/json"
GEOJSON_MEDIA_TYPE = "application/geo+json"
MARKDOWN_MEDIA_TYPE = "text/markdown"
PNG_MEDIA_TYPE = "image/png"
HTML_MEDIA_TYPE = "text/html"

# --------------------------------------------------------------------------
# Provenance. The producer is the USGS and the host is whoever runs this
# pipeline, so a published catalog is a mirror: it MUST carry a `via` link to
# the source and record its sync time in `updated`.
# --------------------------------------------------------------------------

PRODUCER_NAME = "United States Geological Survey"
PRODUCER_URL = "https://www.usgs.gov/landsat-missions"
SOURCE_URL = (
    "https://www.usgs.gov/landsat-missions/"
    "landsat-collection-2-level-2-science-products"
)
SOURCE_STAC_URL = "https://landsatlook.usgs.gov/stac-server"
DEFAULT_HOST_NAME = "landsat-lst-smoke"
DEFAULT_HOST_URL = "https://github.com/nlebovits/landsat-lst-smoke"
#: Landsat Collection 2 carries no use restrictions, which CC0-1.0 states in
#: SPDX terms. The collection README records the USGS citation request.
DEFAULT_LICENSE = "CC0-1.0"

#: The stem every derived collection id starts with. The window follows it, so
#: `collection_id_for_window` builds `lst-p95-2021-2025`.
COLLECTION_ID_STEM = "lst-p95"

#: EPSG:4326 is the only coherent grid here: the shard plan is anchored to
#: whole degrees and sized in pixels per degree.
SUPPORTED_CRS = "EPSG:4326"

LST_ASSET_KEY = "lst_p95"
QA_ASSET_KEY = "qa_count"
LST_FILENAME = "lst_p95.tif"
QA_FILENAME = "qa_count.tif"
LST_TITLE = "95th percentile land surface temperature"
#: Why a zero in `qa_count` is data rather than an absence. The band has no
#: nodata value, so nothing else in the asset says this.
QA_DESCRIPTION = (
    "Cloud-free observations entering the percentile, one band per calendar "
    "month pooled across the window. A zero means masking removed every "
    "observation for that month, which differs from a masked pixel."
)
THUMBNAIL_FILENAME = "thumbnail.png"
MIRROR_FILENAME = "items.parquet"

#: The statistics every band MUST carry, and the valid percent that becomes a
#: MUST once a band has a nodata value.
STATISTICS_KEYS = (
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
)
VALID_PERCENT_KEY = "STATISTICS_VALID_PERCENT"


# --------------------------------------------------------------------------
# Geometry. The grid is degrees anchored to whole degrees, so a transform and
# a tile name both follow from the bbox and the pixel size.
# --------------------------------------------------------------------------


def transform_for(bbox, pixels_per_degree: int):
    """The north-up affine transform of a tile on the degree grid.

    Row 0 is the northern edge, matching `plan_shards`, so the y step is
    negative.
    """
    from rasterio.transform import from_origin

    west, _south, _east, north = bbox
    res = 1.0 / pixels_per_degree
    return from_origin(west, north, res, res)


def _edge(value: float, hemispheres: str, width: int) -> str:
    """One edge as a hemisphere letter and its magnitude, without rounding.

    Rounding the magnitude lets two tiles share a name. The three half-degree
    tiles north of 33 S all round to 32, so each one would take the directory
    the last one wrote. The fraction therefore stays when there is one, and a
    whole degree still reads `32`.
    """
    letter = hemispheres[0] if value >= 0 else hemispheres[1]
    text = f"{abs(float(value)):.6f}".rstrip("0").rstrip(".")
    whole, _, fraction = text.partition(".")
    magnitude = f"{int(whole):0{width}d}"
    return f"{letter}{magnitude}.{fraction}" if fraction else f"{letter}{magnitude}"


def tile_id(bbox) -> str:
    """The tile's name, from its north and west edges: `S30W065`.

    The same convention the 5-degree grid uses, so a catalog built from one
    slice of the plan sorts beside the rest. A tile off that grid keeps its
    fraction and reads `S32.5W062.5`, which no other tile can claim.
    """
    west, _south, _east, north = bbox
    return f"{_edge(north, 'NS', 2)}{_edge(west, 'EW', 3)}"


def check_raster_shape(shape, bbox, pixels_per_degree: int) -> None:
    """The raster has to cover the bbox it is georeferenced against.

    `transform_for` reads the west and north edges and the pixel size, and
    nothing else. A raster of the wrong shape therefore lands on the grid
    without complaint, covering an extent that no item bbox and no collection
    extent mentions.
    """
    west, south, east, north = (float(v) for v in bbox)
    expected = (
        int(round((north - south) * pixels_per_degree)),
        int(round((east - west) * pixels_per_degree)),
    )
    if tuple(int(v) for v in shape) != expected:
        msg = (
            f"a {shape[0]} by {shape[1]} raster does not cover {list(bbox)} at "
            f"1/{pixels_per_degree} degree; expected {expected[0]} by "
            f"{expected[1]}"
        )
        raise ValueError(msg)


def bbox_geometry(bbox) -> dict[str, Any]:
    """The tile footprint as a GeoJSON polygon, wound counter-clockwise."""
    west, south, east, north = (float(v) for v in bbox)
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


# --------------------------------------------------------------------------
# Statistics, computed here rather than by GDAL.
# --------------------------------------------------------------------------


def band_statistics(band, nodata: int | None) -> dict[str, str]:
    """The `STATISTICS_*` tags for one band, as plain ASCII decimals.

    Nodata pixels are excluded, which the encoding rule requires: they are not
    cold temperatures, they are absent ones. A band with no valid pixel at all
    reports zeros and a valid percent of zero, because the tags are mandatory
    and there is no value to report.
    """
    import numpy as np

    values = band if nodata is None else band[band != nodata]
    total = int(band.size)
    kept = int(values.size)
    if kept == 0:
        stats = dict.fromkeys(STATISTICS_KEYS, "0.0")
        stats[VALID_PERCENT_KEY] = "0.0"
        return stats
    wide = np.asarray(values, dtype="float64")
    return {
        "STATISTICS_MINIMUM": repr(float(wide.min())),
        "STATISTICS_MAXIMUM": repr(float(wide.max())),
        "STATISTICS_MEAN": repr(float(wide.mean())),
        "STATISTICS_STDDEV": repr(float(wide.std())),
        VALID_PERCENT_KEY: repr(100.0 * kept / total if total else 0.0),
    }


def _as_bands(array):
    """A 2D array seen as a one-band stack, a 3D array left alone."""
    return array if array.ndim == 3 else array[None, :, :]


# --------------------------------------------------------------------------
# The COG writer, and the reader that reads back what it wrote.
# --------------------------------------------------------------------------


def _staging_profile(stack, transform, crs: str, nodata: int | None) -> dict[str, Any]:
    """The tiled GeoTIFF profile the COG driver copies from."""
    count, height, width = stack.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": stack.dtype.name,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "compress": COMPRESSION,
        "predictor": 2,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    return profile


def _write_staging(path: Path, stack, profile, scale, offset, descriptions) -> None:
    """Write the staging GeoTIFF, statistics and decoding rule included."""
    import rasterio

    count = stack.shape[0]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        if scale is not None and offset is not None:
            dst.scales = (scale,) * count
            dst.offsets = (offset,) * count
        if descriptions:
            dst.descriptions = tuple(descriptions)
        nodata = profile.get("nodata")
        for index in range(count):
            dst.update_tags(index + 1, **band_statistics(stack[index], nodata))


def _verify_cog(
    path: Path,
    count: int,
    *,
    scale: float | None,
    offset: float | None,
    nodata: int | None,
) -> None:
    """Fail loudly if the COG driver dropped anything the spec requires.

    Band tags, scale and offset survive `CreateCopy` today. They are also the
    whole point of the file, so a silent regression in GDAL would publish a
    raster no client can decode. The decoding rule is checked here rather than
    assumed: a dropped scale leaves DN 9500 looking like a temperature, and
    that is the failure this writer exists to prevent. Reopening the file
    costs one header read.
    """
    import rasterio

    expected_tags = set(STATISTICS_KEYS) | {VALID_PERCENT_KEY}
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        if src.block_shapes[0] != (BLOCK_SIZE, BLOCK_SIZE):
            msg = f"{path.name}: internal tiles are {src.block_shapes[0]}"
            raise RuntimeError(msg)
        oversized = max(src.height, src.width) > BLOCK_SIZE
        if oversized and not src.overviews(1):
            msg = f"{path.name}: raster exceeds one tile but carries no overviews"
            raise RuntimeError(msg)
        if scale is not None and src.scales != (scale,) * count:
            msg = f"{path.name}: scale is {src.scales}, not {(scale,) * count}"
            raise RuntimeError(msg)
        if offset is not None and src.offsets != (offset,) * count:
            msg = f"{path.name}: offset is {src.offsets}, not {(offset,) * count}"
            raise RuntimeError(msg)
        if src.nodata != nodata:
            msg = f"{path.name}: nodata is {src.nodata}, not {nodata}"
            raise RuntimeError(msg)
        for index in range(1, count + 1):
            missing = sorted(expected_tags - set(src.tags(bidx=index)))
            if missing:
                msg = f"{path.name}: band {index} lost {', '.join(missing)}"
                raise RuntimeError(msg)
    sidecar = path.with_suffix(path.suffix + ".aux.xml")
    if sidecar.exists():
        msg = f"{path.name}: statistics landed in {sidecar.name}, not in the header"
        raise RuntimeError(msg)


def write_cog(
    path: Path,
    array,
    *,
    bbox,
    pixels_per_degree: int,
    crs: str = SUPPORTED_CRS,
    nodata: int | None = None,
    scale: float | None = None,
    offset: float | None = None,
    descriptions: tuple[str, ...] = (),
) -> Path:
    """Write one array as a Portolan-conformant COG.

    GDAL's COG driver is create-copy only, so this writes a tiled GeoTIFF
    first and copies it. The copy is what builds the overviews and moves the
    header, including the statistics, to the front of the file.

    Not rio-cogeo, which is the usual choice and which the Portolan reference
    tool uses. `cog_translate` drops band tags unless it is called with
    `forward_band_tags=True`, and the per-band statistics are band tags. The
    default therefore produces a file that reads as a valid COG and fails the
    statistics requirement, with nothing on the surface to show it. GDAL's
    `CreateCopy` forwards band metadata with no flag. Both routes produce the
    same block size, overviews, scale, offset, and nodata; this one has the
    safer default.
    """
    import rasterio

    # A compiled extension with no stub, so ty cannot resolve it. It is the
    # only public route from rasterio to GDAL's CreateCopy, which is the only
    # way to reach the COG driver: the driver cannot Create, only copy.
    import rasterio.shutil  # ty: ignore[unresolved-import]

    if crs != SUPPORTED_CRS:
        msg = f"the degree grid needs {SUPPORTED_CRS}, got {crs}"
        raise ValueError(msg)

    stack = _as_bands(array)
    check_raster_shape(stack.shape[1:], bbox, pixels_per_degree)
    profile = _staging_profile(
        stack, transform_for(bbox, pixels_per_degree), crs, nodata
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as staging_dir:
        staging = Path(staging_dir) / "staging.tif"
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            _write_staging(staging, stack, profile, scale, offset, descriptions)
            rasterio.shutil.copy(
                staging,
                path,
                driver="COG",
                blocksize=BLOCK_SIZE,
                compress=COMPRESSION,
                predictor=PREDICTOR,
                overview_resampling=OVERVIEW_RESAMPLING,
                bigtiff="IF_SAFER",
                num_threads="ALL_CPUS",
            )
    _verify_cog(path, stack.shape[0], scale=scale, offset=offset, nodata=nodata)
    return path


def read_cog_encoding(path: Path) -> dict[str, Any]:
    """Everything a reader needs to decode a COG this module wrote.

    The paired reader for `write_cog`. It reads the file, never a sidecar, so
    it answers the same question a remote client asks over a range request.
    """
    import rasterio

    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        return {
            "dtype": src.dtypes[0],
            "count": src.count,
            "shape": [src.height, src.width],
            "crs": str(src.crs),
            "bounds": list(src.bounds),
            "block_shape": list(src.block_shapes[0]),
            "overviews": list(src.overviews(1)),
            "nodata": src.nodata,
            "scale": src.scales[0],
            "offset": src.offsets[0],
            "descriptions": list(src.descriptions),
            "statistics": [src.tags(bidx=i + 1) for i in range(src.count)],
        }


def stac_bands(path: Path, unit: str, description: str) -> list[dict[str, Any]]:
    """The `bands` array for a COG asset, read back from the written file.

    Reading the file rather than the arrays keeps the metadata and the pixels
    from drifting: whatever a client would find in the header is what the STAC
    says it will find.

    STAC 1.1 folded per-band raster metadata into the core `bands` array, and
    raster v2.0.0 followed: it defines only `raster:`-prefixed fields, so the
    data type, nodata, and statistics sit on the band itself while the scale,
    the offset, and the sampling keep the prefix.

    `statistics` and `nodata` are the stored digital numbers, which is the
    domain the COG header reports and the one a reader meets before applying
    `raster:scale`. `unit` names what a pixel means once decoded.
    """
    encoding = read_cog_encoding(path)
    scale, offset = encoding["scale"], encoding["offset"]
    bands: list[dict[str, Any]] = []
    for index, tags in enumerate(encoding["statistics"]):
        band: dict[str, Any] = {
            "name": encoding["descriptions"][index] or f"band{index + 1}",
            "description": description,
            "data_type": encoding["dtype"],
            "unit": unit,
            "statistics": {
                "minimum": float(tags["STATISTICS_MINIMUM"]),
                "maximum": float(tags["STATISTICS_MAXIMUM"]),
                "mean": float(tags["STATISTICS_MEAN"]),
                "stddev": float(tags["STATISTICS_STDDEV"]),
                "valid_percent": float(tags[VALID_PERCENT_KEY]),
            },
            # A pixel carries the whole cell it covers, not a reading at its
            # centre: the percentile and the count both summarise an area.
            "raster:sampling": "area",
        }
        # Declared only where the file carries a real transform. An identity
        # scale on `qa_count` would invite a reader to decode a count.
        if scale is not None and offset is not None and (scale, offset) != (1.0, 0.0):
            band["raster:scale"] = scale
            band["raster:offset"] = offset
        if encoding["nodata"] is not None:
            band["nodata"] = encoding["nodata"]
        bands.append(band)
    return bands


# --------------------------------------------------------------------------
# The thumbnail. Required on every geospatial collection.
# --------------------------------------------------------------------------


def render_thumbnail(path: Path, tiles, *, long_edge: int = 480) -> Path:
    """Render the whole collection as a small PNG, at its own aspect ratio.

    `tiles` pairs each item's bbox with the `lst_p95` COG that covers it. The
    thumbnail shows every tile, because it previews the collection rather than
    whichever tile the last merge happened to write.

    Each tile is read through `out_shape`, which GDAL serves from the
    overviews the writer already built. A 480 px preview of a 520-tile
    collection therefore reads no full-resolution pixel.

    A thumbnail helps someone recognise the data, so it shows Celsius rather
    than DN and leaves the nodata pixels blank. There is no basemap: fetching
    one would put a network call inside a test.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    boxes = [[float(v) for v in bbox] for bbox, _cog in tiles]
    west = min(box[0] for box in boxes)
    south = min(box[1] for box in boxes)
    east = max(box[2] for box in boxes)
    north = max(box[3] for box in boxes)
    span_x, span_y = east - west, north - south
    scale = long_edge / max(span_x, span_y)
    width = max(1, int(round(span_x * scale)))
    height = max(1, int(round(span_y * scale)))
    canvas = np.full((height, width), np.nan, dtype="float64")

    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        for (tile_west, tile_south, tile_east, tile_north), cog in tiles:
            x0 = int(round((float(tile_west) - west) / span_x * width))
            x1 = int(round((float(tile_east) - west) / span_x * width))
            y0 = int(round((north - float(tile_north)) / span_y * height))
            y1 = int(round((north - float(tile_south)) / span_y * height))
            if x1 <= x0 or y1 <= y0:
                continue
            with rasterio.open(cog) as src:
                dn = src.read(1, out_shape=(y1 - y0, x1 - x0))
            canvas[y0:y1, x0:x1] = np.where(
                dn == LST_NODATA_DN, np.nan, dn * LST_SCALE + LST_OFFSET
            )

    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    axes.imshow(canvas, cmap="magma", interpolation="nearest")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100, transparent=True)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Asset bookkeeping: size and checksum, which Portolan asks for on every asset.
# --------------------------------------------------------------------------


#: Chunk size for the checksum read. A full-tile COG runs to hundreds of MB,
#: which is not a thing to hold in memory to hash it.
CHECKSUM_CHUNK = 1 << 20


def multihash_sha256(path: Path) -> str:
    """The file's SHA-256 as a hex multihash: code 0x12, length 0x20."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHECKSUM_CHUNK), b""):
            digest.update(chunk)
    return f"1220{digest.hexdigest()}"


def _file_fields(path: Path) -> dict[str, Any]:
    return {"file:size": path.stat().st_size, "file:checksum": multihash_sha256(path)}


def _asset(href: str, path: Path, title: str, media_type: str, roles) -> dict[str, Any]:
    return {
        "href": href,
        "type": media_type,
        "title": title,
        "roles": list(roles),
        **_file_fields(path),
    }


# --------------------------------------------------------------------------
# Time. STAC wants RFC 3339, and the window constants are written for a STAC
# search, where a bare date is legal.
# --------------------------------------------------------------------------


def _rfc3339(value: str) -> str:
    """A window bound as a full RFC 3339 instant.

    The zone is looked for after the `T`, so a negative UTC offset counts as a
    zone. Searching the whole string for a sign would find the one in the date
    and leave `2021-01-01T00:00:00-05:00` to collect a second zone.
    """
    if "T" not in value:
        return f"{value}T00:00:00Z"
    time_part = value.split("T", 1)[1]
    zoned = value.endswith("Z") or "+" in time_part or "-" in time_part
    return value if zoned else f"{value}Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window_label(start: str, end: str) -> str:
    """`2021-2025` for a five-year window, `2024` for one year."""
    first, last = str(start)[:4], str(end)[:4]
    return first if first == last else f"{first}-{last}"


# --------------------------------------------------------------------------
# STAC objects.
# --------------------------------------------------------------------------


def build_item(
    item_id: str,
    bbox,
    *,
    assets: dict[str, dict[str, Any]],
    collection_id: str,
    start: str,
    end: str,
    crs: str,
    mask_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The tile item: one footprint, one acquisition window, two COGs.

    The item carries its own `renders`, stretched over its own statistics. An
    item describes the tile a reader opened, and a client that draws one tile
    alone never reads the collection. The collection's ramp spans every tile,
    which is what a mosaic of them needs, so the two differ by design.

    The mask lineage is on the item for the same reason. The rules ran over
    this tile's pixels, and a tile masked against a different ASTER GED build
    says so itself rather than inheriting a collection-wide claim.
    """
    lineage = mask_lineage(mask_rule)
    extensions = [
        FILE_EXTENSION,
        PROJECTION_EXTENSION,
        RASTER_EXTENSION,
        RENDER_EXTENSION,
        PROCESSING_EXTENSION,
    ]
    if "sci:publications" in lineage:
        extensions.append(SCIENTIFIC_EXTENSION)
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": extensions,
        "id": item_id,
        "collection": collection_id,
        "geometry": bbox_geometry(bbox),
        "bbox": [float(v) for v in bbox],
        "properties": {
            "title": f"{item_id} land surface temperature, {_window_label(start, end)}",
            "datetime": None,
            "start_datetime": _rfc3339(start),
            "end_datetime": _rfc3339(end),
            "proj:code": crs,
            "renders": _renders([{"assets": assets}]),
            **lineage,
        },
        "assets": assets,
        "links": [
            {"rel": "root", "href": "../../catalog.json", "type": JSON_MEDIA_TYPE},
            {"rel": "parent", "href": "../collection.json", "type": JSON_MEDIA_TYPE},
            {
                "rel": "collection",
                "href": "../collection.json",
                "type": JSON_MEDIA_TYPE,
            },
        ],
    }


def _providers(host_name: str, host_url: str) -> list[dict[str, Any]]:
    """Producer first, host last, which is the order Portolan requires."""
    return [
        {
            "name": PRODUCER_NAME,
            "url": PRODUCER_URL,
            "roles": ["producer", "licensor"],
        },
        {
            "name": host_name,
            "url": host_url,
            "roles": ["processor", "host"],
        },
    ]


def _renders(items: list[dict[str, Any]]) -> dict[str, Any]:
    """A draw-time colour ramp, so a client can render the COG from source.

    The rescale range is in the band's stored value space, matching the
    statistics beside it, because that is what a tiler reads off the pixels.
    It spans every item, so one tile cannot set the ramp for the rest.

    A band with no valid pixel reports zeros, because the statistics tags are
    mandatory and it has nothing to report. Including one would collapse the
    range onto the nodata DN, so the ramp reads only from bands that saw a
    temperature. A collection of nothing but empty tiles falls back to the
    encodable range, which is a wide ramp rather than a broken one.
    """
    ranges = [
        (band["statistics"]["minimum"], band["statistics"]["maximum"])
        for item in items
        for band in item["assets"][LST_ASSET_KEY]["bands"]
        if band["statistics"]["valid_percent"] > 0
    ]
    low = min(low for low, _high in ranges) if ranges else float(LST_MIN_DN)
    high = max(high for _low, high in ranges) if ranges else float(LST_MAX_DN)
    return {
        LST_ASSET_KEY: {
            "title": LST_TITLE,
            "assets": [LST_ASSET_KEY],
            "rescale": [[low, high]],
            "colormap_name": "magma",
            "resampling": "average",
            "nodata": LST_NODATA_DN,
        }
    }


def _spatial_extent(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The union bbox first, then one entry per item.

    STAC reads the first entry as the overall extent, and a Portolan validator
    compares it against the union of the footprints in `items.parquet`. The
    per-item entries after it say which parts of that box are covered.
    """
    boxes = [[float(v) for v in item["bbox"]] for item in items]
    union = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    return {"bbox": [union, *boxes] if len(boxes) > 1 else [union]}


def _temporal_extent(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The union window first, then one entry per item.

    Every bound is a `Z` instant written by `_rfc3339`, so comparing the
    strings orders them the same way comparing the instants would.
    """
    windows = [
        [item["properties"]["start_datetime"], item["properties"]["end_datetime"]]
        for item in items
    ]
    union = [
        min(window[0] for window in windows),
        max(window[1] for window in windows),
    ]
    return {"interval": [union, *windows] if len(windows) > 1 else [union]}


def build_collection(
    collection_id: str,
    items: list[dict[str, Any]],
    *,
    assets: dict[str, dict[str, Any]],
    host_name: str,
    host_url: str,
    license_id: str,
    updated: str,
) -> dict[str, Any]:
    """The collection: provenance, extent, the thumbnail, and the item mirror.

    Built from every item in the collection, not from the tile that triggered
    the write. The extent, the item links, and the colour ramp all describe
    the whole collection, so writing tile 2 leaves tile 1 still described.
    """
    spatial = _spatial_extent(items)
    temporal = _temporal_extent(items)
    start, end = (bound[:10] for bound in temporal["interval"][0])
    return {
        "type": "Collection",
        "stac_version": STAC_VERSION,
        "stac_extensions": [
            PORTOLAN_EXTENSION,
            FILE_EXTENSION,
            PROJECTION_EXTENSION,
            RENDER_EXTENSION,
        ],
        "id": collection_id,
        "title": "Landsat P95 Land Surface Temperature Composite",
        "description": (
            "A five-year 95th-percentile land surface temperature composite "
            f"built from {DEFAULT_COLLECTION} scenes acquired by "
            f"{DEFAULT_PLATFORMS.replace(',', ' and ')} between "
            f"{start} and {end}. Temperature is stored as uint16 and decodes "
            f"with celsius = dn * {LST_SCALE} + ({LST_OFFSET}); DN "
            f"{LST_NODATA_DN} is nodata. A second asset counts the valid "
            "observations behind each pixel, one band per calendar month."
        ),
        "license": license_id,
        "keywords": [
            "landsat",
            "land-surface-temperature",
            "composite",
            "percentile",
            "cog",
        ],
        "providers": _providers(host_name, host_url),
        "extent": {"spatial": spatial, "temporal": temporal},
        "renders": _renders(items),
        "assets": assets,
        "links": [
            {"rel": "root", "href": "../catalog.json", "type": JSON_MEDIA_TYPE},
            {"rel": "parent", "href": "../catalog.json", "type": JSON_MEDIA_TYPE},
            *[
                {
                    "rel": "item",
                    "href": f"./{item['id']}/{item['id']}.json",
                    "type": GEOJSON_MEDIA_TYPE,
                    "title": f"Tile {item['id']}",
                }
                for item in items
            ],
            {
                "rel": "via",
                "href": SOURCE_URL,
                "type": HTML_MEDIA_TYPE,
                "title": "Landsat Collection 2 Level-2 science products",
            },
            {
                "rel": "canonical",
                "href": SOURCE_STAC_URL,
                "type": JSON_MEDIA_TYPE,
                "title": "USGS LandsatLook STAC API",
            },
            {
                "rel": "agents",
                "href": "./AGENTS.md",
                "type": MARKDOWN_MEDIA_TYPE,
                "title": "Guidance for AI agents",
            },
            {
                "rel": "describedby",
                "href": "./README.md",
                "type": MARKDOWN_MEDIA_TYPE,
                "title": "Human-readable documentation",
            },
        ],
        "updated": updated,
    }


def build_root_catalog(collection_id: str, *, updated: str) -> dict[str, Any]:
    """The root catalog. One child, and the two documents Portolan requires."""
    return {
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "stac_extensions": [PORTOLAN_EXTENSION],
        "id": "landsat-lst-smoke",
        "title": "Landsat LST Composites",
        "description": (
            "Cloud-optimized 95th-percentile land surface temperature "
            "composites built from Landsat Collection 2 Level-2 scenes, with "
            "the per-month observation counts behind each pixel."
        ),
        "links": [
            {"rel": "root", "href": "./catalog.json", "type": JSON_MEDIA_TYPE},
            {
                "rel": "child",
                "href": f"./{collection_id}/collection.json",
                "type": JSON_MEDIA_TYPE,
                "title": "Landsat P95 Land Surface Temperature Composite",
            },
            {
                "rel": "agents",
                "href": "./AGENTS.md",
                "type": MARKDOWN_MEDIA_TYPE,
                "title": "Guidance for AI agents",
            },
            {
                "rel": "describedby",
                "href": "./README.md",
                "type": MARKDOWN_MEDIA_TYPE,
                "title": "Human-readable documentation",
            },
        ],
        "updated": updated,
    }


def write_item_mirror(path: Path, items: list[dict[str, Any]]) -> Path:
    """Write the collection's items as stac-geoparquet.

    One range request then returns every item's metadata, in place of one HTTP
    fetch per item. The file is a derived copy, so it is written from the same
    dicts the item JSON came from and never edited afterwards.
    """
    from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    table = parse_stac_items_to_arrow(items)
    to_parquet(table, path)
    return path


# --------------------------------------------------------------------------
# The two Markdown documents every catalog and collection must carry.
# --------------------------------------------------------------------------


def _decode_snippet(asset_href: str) -> str:
    return (
        "```python\n"
        "import rioxarray\n\n"
        f'da = rioxarray.open_rasterio("{asset_href}", masked=True)\n\n'
        "# The decoding rule travels with the file.\n"
        "scale, offset = da.rio.scales[0], da.rio.offsets[0]\n"
        "celsius = da * scale + offset\n"
        "```\n"
    )


def _root_readme(collection_id: str) -> str:
    return (
        "# Landsat LST Composites\n\n"
        "Cloud-optimized 95th-percentile land surface temperature composites "
        "built from Landsat Collection 2 Level-2 scenes.\n\n"
        "## License\n\n"
        f"The data carries the `{DEFAULT_LICENSE}` license. Landsat Collection "
        "2 products are free of use restrictions.\n\n"
        "## Provenance\n\n"
        f"Scenes come from the [USGS Landsat Collection 2 Level-2 science "
        f"products]({SOURCE_URL}). This catalog is a derived mirror, not the "
        "authoritative source.\n\n"
        "## Collections\n\n"
        f"- [`{collection_id}`](./{collection_id}/README.md)\n"
    )


def _tile_list(item_ids: list[str]) -> str:
    """The collection's tiles, as a Markdown list of their item documents."""
    return "".join(
        f"- [`{item_id}`](./{item_id}/{item_id}.json)\n" for item_id in item_ids
    )


def _collection_readme(
    collection_id: str, item_ids: list[str], start: str, end: str, license_id: str
) -> str:
    item_id = item_ids[0]
    return (
        "# Landsat P95 Land Surface Temperature Composite\n\n"
        "The 95th percentile of clear-sky land surface temperature over "
        f"{start} to {end}, with the count of valid observations behind each "
        "pixel.\n\n"
        "## Tiles\n\n"
        f"One item per tile of the degree grid, {len(item_ids)} so far. Each "
        "item carries both assets for its own footprint.\n\n"
        f"{_tile_list(item_ids)}\n"
        "## Assets\n\n"
        "| Asset | Name | Dtype | Scale | Offset | Nodata | Units |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| `{LST_ASSET_KEY}` | 95th percentile LST | uint16, 1 band | "
        f"{LST_SCALE} | {LST_OFFSET} | {LST_NODATA_DN} | celsius |\n"
        f"| `{QA_ASSET_KEY}` | Valid observations per calendar month | "
        "uint8, 12 bands (Jan..Dec) | — | — | none | count |\n\n"
        f"`{QA_ASSET_KEY}` has no nodata value by design. A value of 0 means "
        "that no valid observation survived masking for that month. Keeping "
        "zero visible lets you tell this gap apart from masked data.\n\n"
        "## Limitations\n\n"
        f"A nodata `{LST_ASSET_KEY}` pixel means one of three things, and the "
        "raster separates none of them: no usable observation, water, or an "
        "emissivity retrieval that failed inside an ASTER GED coverage gap. "
        "Only the first would improve with a wider window. Each item's "
        "`processing:lineage` states the rules that produced its pixels and "
        "names the artifacts they read by checksum.\n\n"
        "Where ASTER GED caught no clear sky between 2000 and 2008, the USGS "
        "interpolates emissivity from neighbouring cells and retrieves a "
        "temperature anyway, and some of those retrievals fail upward. A pixel "
        "is removed only where its cell reports zero observations, or lies one "
        "cell from such a cell, and the pixel also reads 70 C or hotter. "
        "Masking the gap geometry alone was measured on S30W065 to remove "
        "701,839 valid pixels to remove 4,588 bad ones; the pair removes "
        "5,432 and reaches more of the tail. 503 hot pixels survive on that "
        "tile, in cells that did have observations. The 70 C threshold is a "
        "screen calibrated on one tile with no published source, so it makes "
        "no claim about the hottest land surface.\n\n"
        "## Decoding\n\n"
        f"{_decode_snippet(f'{item_id}/{LST_FILENAME}')}\n"
        "If you already know the constants:\n\n"
        "```python\n"
        f"celsius = dn * {LST_SCALE} + ({LST_OFFSET})  # DN "
        f"{LST_NODATA_DN} is nodata\n"
        "```\n\n"
        "## License\n\n"
        f"`{license_id}`. Landsat Collection 2 products carry no use "
        "restrictions. The USGS asks that you cite the source.\n\n"
        "## Provenance\n\n"
        f"Derived from [Landsat Collection 2 Level-2 science products]"
        f"({SOURCE_URL}) through the `{collection_id}` pipeline in "
        f"[landsat-lst-smoke]({DEFAULT_HOST_URL}). The upstream STAC API is "
        f"[LandsatLook]({SOURCE_STAC_URL}).\n"
    )


def _agents_md(collection_id: str, item_ids: list[str]) -> str:
    item_id = item_ids[0]
    return (
        "# Guidance for agents\n\n"
        "## Reading the data\n\n"
        f"The `{collection_id}` collection holds one item per tile of the "
        f"degree grid, {len(item_ids)} of them, and both assets are Cloud "
        f"Optimized GeoTIFFs on the item. `{item_id}` is one. A tile is named "
        "for its north and west edges, so `S30W065` starts at 30 S, 65 W. "
        "Read a window rather than the whole file: the internal tiles are "
        f"{BLOCK_SIZE} by {BLOCK_SIZE} pixels and the overviews let you draw "
        "the tile without touching full-resolution pixels.\n\n"
        "## Decoding temperature\n\n"
        f"{_decode_snippet(f'{item_id}/{LST_FILENAME}')}\n"
        f"DN {LST_NODATA_DN} is nodata. Treat it as absent rather than cold.\n\n"
        "## What a nodata pixel means\n\n"
        "Three different facts, and the raster separates none of them:\n\n"
        "| Meaning | Rule | Would a wider window fix it |\n"
        "|---|---|---|\n"
        "| No usable observation | every scene was cloudy, or the pixel is "
        "off every footprint | yes |\n"
        "| Water | outside the buffered land geometry | no |\n"
        "| Failed emissivity retrieval | hot inside an ASTER GED coverage "
        "gap | no |\n\n"
        "Do not read a nodata pixel as missing data over the ocean: the water "
        "rule zeroes `qa_count` with the temperature, so a count of 0 beside "
        "a nodata pixel is the signature of sea rather than of cloud. The "
        "emissivity rule leaves `qa_count` alone, so a nodata pixel with a "
        "count above 0 was screened rather than never seen. The item's "
        "`processing:lineage` states both rules and names the artifacts they "
        "read by checksum.\n\n"
        "## Reading the observation counts\n\n"
        f"`{QA_ASSET_KEY}` has 12 bands, January through December. Band `m` "
        "counts the clear observations that entered the percentile for that "
        "calendar month, pooled across every year in the window. The count "
        "saturates at 255. A zero is a real count, not a gap, which is why "
        "the band declares no nodata value.\n\n"
        "## Cross-referencing\n\n"
        "Read `items.parquet` in the collection root to get every item's "
        "metadata in one range request, instead of fetching each item JSON.\n"
    )


# --------------------------------------------------------------------------
# The orchestrator.
# --------------------------------------------------------------------------


#: What `part-meta.json` has to state before a catalog can describe the tile.
REQUIRED_META_KEYS = ("bbox", "crs", "pixels_per_degree", "start", "end")

#: Why the per-scene QA screen alone does not explain a nodata pixel.
_QA_LINEAGE = (
    "Per-scene screening removes cloud, cloud shadow, snow, cirrus, and "
    "out-of-range pixels before the percentile. A pixel with no surviving "
    "observation is nodata."
)


def mask_lineage(mask_rule: dict[str, Any] | None) -> dict[str, Any]:
    """The properties that say which pixels the output mask removed, and why.

    A nodata `lst_p95` pixel carries three meanings: no usable observation,
    water, or an emissivity retrieval that failed inside an ASTER GED coverage
    gap. Nothing in the raster separates them, so the item states the rules it
    was masked under and names the artifacts by DOI and checksum.

    `processing:lineage` and `sci:publications` are the registered homes for
    this. No `lst:`-prefixed property restates any of it.
    """
    if not mask_rule:
        return {
            "processing:lineage": (
                f"{_QA_LINEAGE} No output mask ran, so sea pixels and ASTER "
                "GED emissivity gaps are present in this tile."
            )
        }
    buffer_cells = mask_rule.get("gap_buffer_cells")
    threshold = mask_rule.get("gap_hot_threshold_c")
    ged = mask_rule.get("aster_ged") or {}
    sentences = [
        _QA_LINEAGE,
        "Two further rules then run over the assembled tile. Water: a pixel "
        "outside the buffered land geometry becomes nodata, and its qa_count "
        "becomes 0, so the two bands cannot disagree about a pixel that was "
        "never this product's subject.",
        f"Emissivity: a pixel whose ASTER GED cell reports no clear-sky "
        f"observation, or lies {buffer_cells} cell from such a cell, becomes "
        f"nodata when it reads {threshold} C or hotter. Its qa_count is left "
        f"alone, because the count of clear observations stays true whatever "
        f"the retrieval did with them.",
    ]
    if ged.get("short_name"):
        read_from = (
            f"Emissivity coverage read from {ged['short_name']} v"
            f"{ged.get('version')}, {ged.get('granule_count'):,} granules"
        )
        # An empty digest is worse than an absent one: it reads as a checksum
        # a consumer can compare against. A raster cannot hold its own digest,
        # so it stays empty whenever the sidecar that carries it is absent.
        if ged.get("raster_sha256"):
            read_from += f", raster sha256 {ged['raster_sha256']}"
        sentences.append(f"{read_from}.")
    if mask_rule.get("land_geometry_sha256"):
        sentences.append(f"Land geometry sha256 {mask_rule['land_geometry_sha256']}.")
    properties: dict[str, Any] = {"processing:lineage": " ".join(sentences)}
    if ged.get("doi"):
        properties["sci:publications"] = [
            {
                "doi": ged["doi"],
                "citation": (
                    f"ASTER Global Emissivity Dataset {ged['short_name']} v"
                    f"{ged.get('version')}. NASA LP DAAC."
                ),
            }
        ]
    return properties


def catalog_provenance(meta: dict[str, Any], *, collection_id: str) -> dict[str, Any]:
    """Everything a reader needs to know how this catalog was produced.

    A missing window is an error, not a default. `part-meta.json` gained
    `start` and `end` with this writer, so parts from an earlier run carry
    neither. Stamping the current window over them would publish a claim about
    the pixels that nobody reading the catalog could question.
    """
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        msg = (
            f"part-meta.json states no {', '.join(missing)}; rerun the "
            f"--shard-slice that wrote it, or merge with --no-catalog"
        )
        raise ValueError(msg)
    return {
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "collection_id": collection_id,
        "bbox": [float(v) for v in meta["bbox"]],
        "crs": meta["crs"],
        "pixels_per_degree": int(meta["pixels_per_degree"]),
        "start": meta["start"],
        "end": meta["end"],
        "block_size": BLOCK_SIZE,
        "compression": COMPRESSION,
        "overview_resampling": OVERVIEW_RESAMPLING,
        "lst_scale": LST_SCALE,
        "lst_offset": LST_OFFSET,
        "lst_nodata": LST_NODATA_DN,
    }


def collection_id_for_window(meta: dict[str, Any]) -> str:
    """The collection id the tile's own window earns, e.g. `lst-p95-2021-2025`.

    The window belongs in the id because the id is also the directory name. Two
    windows aimed at one catalog would otherwise land in one collection, and
    the later merge would widen the temporal extent over pixels from a window
    nobody asked about. A single-year window reads `lst-p95-2024`.
    """
    missing = [key for key in ("start", "end") if key not in meta]
    if missing:
        msg = (
            f"part-meta.json states no {', '.join(missing)}, so the collection "
            f"id has no window to carry; rerun the --shard-slice that wrote it, "
            f"or pass --collection-id"
        )
        raise ValueError(msg)
    return f"{COLLECTION_ID_STEM}-{_window_label(meta['start'], meta['end'])}"


def check_catalog_inputs(meta: dict[str, Any]) -> None:
    """Fail before a long merge rather than after it.

    Every rule the writer enforces is answerable from `part-meta.json` alone.
    Checking it up front turns an hour of merging followed by a traceback into
    a message in under a second, and it leaves the `.npy` arrays a merge has
    already earned alone.
    """
    provenance = catalog_provenance(meta, collection_id=collection_id_for_window(meta))
    if provenance["crs"] != SUPPORTED_CRS:
        msg = f"the degree grid needs {SUPPORTED_CRS}, got {provenance['crs']}"
        raise ValueError(msg)
    check_raster_shape(
        meta["raster"], provenance["bbox"], provenance["pixels_per_degree"]
    )


def read_items(collection_dir: Path) -> dict[str, dict[str, Any]]:
    """Every item already written under this collection, keyed by id.

    The collection is derived from the items on disk rather than accumulated
    in memory, so writing tile 2 rebuilds a collection that describes tiles 1
    and 2. A rerun of one tile, or a merge that ran on another machine and
    landed here, reaches the same tree.
    """
    items: dict[str, dict[str, Any]] = {}
    if not collection_dir.is_dir():
        return items
    for item_path in sorted(collection_dir.glob("*/*.json")):
        if item_path.stem != item_path.parent.name:
            continue
        item = json.loads(item_path.read_text())
        if item.get("type") == "Feature":
            items[item["id"]] = item
    return items


def _write_cogs(item_dir: Path, lst, qa, provenance: dict[str, Any]):
    """Both COGs of one tile, in the item directory that carries them."""
    bbox = provenance["bbox"]
    ppd = provenance["pixels_per_degree"]
    crs = provenance["crs"]
    lst_path = write_cog(
        item_dir / LST_FILENAME,
        lst,
        bbox=bbox,
        pixels_per_degree=ppd,
        crs=crs,
        nodata=LST_NODATA_DN,
        scale=LST_SCALE,
        offset=LST_OFFSET,
        descriptions=("95th percentile LST",),
    )
    qa_path = write_cog(
        item_dir / QA_FILENAME,
        qa,
        bbox=bbox,
        pixels_per_degree=ppd,
        crs=crs,
        descriptions=tuple(MONTH_NAMES),
    )
    return lst_path, qa_path


def _item_assets(lst_path: Path, qa_path: Path, crs: str) -> dict[str, Any]:
    """The two COG assets, each carrying the bands read back from its file."""
    lst_bands = stac_bands(
        lst_path, "celsius", "95th percentile land surface temperature"
    )
    qa_bands = stac_bands(
        qa_path, "count", "Valid observations entering the percentile"
    )
    for index, band in enumerate(qa_bands):
        band["name"] = MONTH_NAMES[index]
    lst_asset = _asset(
        f"./{LST_FILENAME}",
        lst_path,
        "95th percentile land surface temperature (COG)",
        COG_MEDIA_TYPE,
        ["data"],
    )
    lst_asset["bands"] = lst_bands
    lst_asset["proj:code"] = crs
    qa_asset = _asset(
        f"./{QA_FILENAME}",
        qa_path,
        "Valid observations per calendar month (COG)",
        COG_MEDIA_TYPE,
        # `quality` alongside `data`: the counts are the evidence behind every
        # surviving percentile, which is what a reader filtering on the role
        # is looking for.
        ["data", "quality"],
    )
    qa_asset["bands"] = qa_bands
    qa_asset["proj:code"] = crs
    qa_asset["description"] = QA_DESCRIPTION
    return {LST_ASSET_KEY: lst_asset, QA_ASSET_KEY: qa_asset}


def _collection_assets(thumbnail: Path, mirror: Path) -> dict[str, Any]:
    return {
        "thumbnail": _asset(
            f"./{THUMBNAIL_FILENAME}",
            thumbnail,
            "Preview of the composite, coloured by temperature",
            PNG_MEDIA_TYPE,
            ["thumbnail"],
        ),
        "items": _asset(
            f"./{MIRROR_FILENAME}",
            mirror,
            "Collection items as stac-geoparquet",
            PARQUET_MEDIA_TYPE,
            ["collection-mirror"],
        ),
    }


def _dump(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _check_no_clash(collection_dir: Path, item_id: str, bbox: list[float]) -> None:
    """Refuse to write over a tile that this one only shares a name with.

    A name states the north and west edges, so two tiles collide only when
    they share a corner and differ in size: a 1-degree tile and the 5-degree
    tile that starts at the same place. Rewriting the same tile is ordinary
    and stays allowed.
    """
    previous = read_items(collection_dir).get(item_id)
    if previous is None:
        return
    if [float(v) for v in previous["bbox"]] != bbox:
        msg = (
            f"{item_id} already names the tile {previous['bbox']} in "
            f"{collection_dir}; {bbox} shares its corner and would replace it"
        )
        raise ValueError(msg)


def write_catalog(
    out_dir: Path,
    lst,
    qa,
    meta: dict[str, Any],
    *,
    collection_id: str | None = None,
    host_name: str = DEFAULT_HOST_NAME,
    host_url: str = DEFAULT_HOST_URL,
    license_id: str = DEFAULT_LICENSE,
) -> Path:
    """Write one tile into the catalog and return the catalog's root directory.

    `meta` is the `part-meta.json` payload: the bbox, the CRS, the pixels per
    degree, and the composite window. Everything else follows from the arrays.

    `collection_id` defaults to the one the tile's own window earns, so two
    windows cannot collect into one collection by omission.

    The tile becomes an item. The collection, the root catalog, the thumbnail,
    and the item mirror are then rebuilt from every item on disk, so calling
    this for a second tile describes both rather than replacing the first.
    """
    collection_id = collection_id or collection_id_for_window(meta)
    provenance = catalog_provenance(meta, collection_id=collection_id)
    bbox, crs = provenance["bbox"], provenance["crs"]
    item_id = tile_id(bbox)
    updated = _now()

    root = Path(out_dir)
    collection_dir = root / collection_id
    item_dir = collection_dir / item_id

    _check_no_clash(collection_dir, item_id, bbox)
    item_dir.mkdir(parents=True, exist_ok=True)

    lst_path, qa_path = _write_cogs(item_dir, lst, qa, provenance)
    item = build_item(
        item_id,
        bbox,
        assets=_item_assets(lst_path, qa_path, crs),
        collection_id=collection_id,
        start=provenance["start"],
        end=provenance["end"],
        crs=crs,
        mask_rule=meta.get("mask_rule"),
    )
    _dump(item_dir / f"{item_id}.json", item)

    # Read every item back, this one included, so the collection describes the
    # whole tree rather than the tile this call happened to write.
    items = read_items(collection_dir)
    item_ids = sorted(items)
    ordered = [items[known] for known in item_ids]

    thumbnail = render_thumbnail(
        collection_dir / THUMBNAIL_FILENAME,
        [
            (known["bbox"], collection_dir / known["id"] / LST_FILENAME)
            for known in ordered
        ],
    )
    mirror = write_item_mirror(collection_dir / MIRROR_FILENAME, ordered)
    collection = build_collection(
        collection_id,
        ordered,
        assets=_collection_assets(thumbnail, mirror),
        host_name=host_name,
        host_url=host_url,
        license_id=license_id,
        updated=updated,
    )
    _dump(collection_dir / "collection.json", collection)
    _dump(root / "catalog.json", build_root_catalog(collection_id, updated=updated))

    start, end = (
        bound[:10] for bound in collection["extent"]["temporal"]["interval"][0]
    )
    (collection_dir / "README.md").write_text(
        _collection_readme(collection_id, item_ids, start, end, license_id)
    )
    (collection_dir / "AGENTS.md").write_text(_agents_md(collection_id, item_ids))
    (root / "README.md").write_text(_root_readme(collection_id))
    (root / "AGENTS.md").write_text(_agents_md(collection_id, item_ids))
    return root

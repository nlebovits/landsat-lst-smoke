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

Both COGs sit on the item, not on the collection. A Portolan collection carries
a raster asset itself only when it holds exactly one COG; two at collection
level is a conformance error, and modelling the tile as an item is also what
lets the 520-tile grid grow without a layout change.

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

from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE
from stac_window import (
    DEFAULT_COLLECTION,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
)

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
DEFAULT_COLLECTION_ID = "lst-p95-composite"

#: EPSG:4326 is the only coherent grid here: the shard plan is anchored to
#: whole degrees and sized in pixels per degree.
SUPPORTED_CRS = "EPSG:4326"

LST_ASSET_KEY = "lst_p95"
QA_ASSET_KEY = "qa_count"
LST_FILENAME = "lst_p95.tif"
QA_FILENAME = "qa_count.tif"
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


def tile_id(bbox) -> str:
    """The tile's name, from its north and west edges: `S30W065`.

    The same convention the 5-degree grid uses, so a catalog built from one
    slice of the plan sorts beside the rest.
    """
    _west, _south, _east, north = bbox
    west = bbox[0]
    ns = "N" if north >= 0 else "S"
    ew = "E" if west >= 0 else "W"
    return f"{ns}{abs(int(round(north))):02d}{ew}{abs(int(round(west))):03d}"


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


def _verify_cog(path: Path, count: int) -> None:
    """Fail loudly if the COG driver dropped anything the spec requires.

    Band tags, scale and offset survive `CreateCopy` today. They are also the
    whole point of the file, so a silent regression in GDAL would publish a
    raster no client can decode. Reopening costs one header read.
    """
    import rasterio

    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        if src.block_shapes[0] != (BLOCK_SIZE, BLOCK_SIZE):
            msg = f"{path.name}: internal tiles are {src.block_shapes[0]}"
            raise RuntimeError(msg)
        oversized = max(src.height, src.width) > BLOCK_SIZE
        if oversized and not src.overviews(1):
            msg = f"{path.name}: raster exceeds one tile but carries no overviews"
            raise RuntimeError(msg)
        for index in range(1, count + 1):
            missing = sorted(set(STATISTICS_KEYS) - set(src.tags(bidx=index)))
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
    _verify_cog(path, stack.shape[0])
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
    """
    encoding = read_cog_encoding(path)
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
        }
        if encoding["nodata"] is not None:
            band["nodata"] = encoding["nodata"]
        bands.append(band)
    return bands


# --------------------------------------------------------------------------
# The thumbnail. Required on every geospatial collection.
# --------------------------------------------------------------------------


def render_thumbnail(path: Path, dn, *, long_edge: int = 480) -> Path:
    """Render the composite as a small PNG, at the data's own aspect ratio.

    A thumbnail is a preview that helps someone recognise the data, so it shows
    Celsius rather than DN and leaves the nodata pixels blank. There is no
    basemap: fetching one would put a network call inside a test.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    celsius = np.where(dn == LST_NODATA_DN, np.nan, dn * LST_SCALE + LST_OFFSET)
    height, width = dn.shape
    scale = long_edge / max(height, width)
    figure = plt.figure(figsize=(width * scale / 100, height * scale / 100), dpi=100)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    axes.imshow(celsius, cmap="magma", interpolation="nearest")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100, transparent=True)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Asset bookkeeping: size and checksum, which Portolan asks for on every asset.
# --------------------------------------------------------------------------


def multihash_sha256(path: Path) -> str:
    """The file's SHA-256 as a hex multihash: code 0x12, length 0x20."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"1220{digest}"


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
    """A window bound as a full RFC 3339 instant."""
    if "T" in value:
        return value if value.endswith("Z") or "+" in value else f"{value}Z"
    return f"{value}T00:00:00Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
) -> dict[str, Any]:
    """The tile item: one footprint, one acquisition window, two COGs."""
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [FILE_EXTENSION, PROJECTION_EXTENSION],
        "id": item_id,
        "collection": collection_id,
        "geometry": bbox_geometry(bbox),
        "bbox": [float(v) for v in bbox],
        "properties": {
            "datetime": None,
            "start_datetime": _rfc3339(start),
            "end_datetime": _rfc3339(end),
            "proj:code": crs,
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


def _renders(bands: list[dict[str, Any]]) -> dict[str, Any]:
    """A draw-time colour ramp, so a client can render the COG from source.

    The rescale range is in the band's stored value space, matching the
    statistics beside it, because that is what a tiler reads off the pixels.
    """
    statistics = bands[0]["statistics"]
    return {
        LST_ASSET_KEY: {
            "title": "95th percentile land surface temperature",
            "assets": [LST_ASSET_KEY],
            "rescale": [[statistics["minimum"], statistics["maximum"]]],
            "colormap_name": "magma",
            "resampling": "average",
            "nodata": LST_NODATA_DN,
        }
    }


def build_collection(
    collection_id: str,
    bbox,
    *,
    item_id: str,
    assets: dict[str, dict[str, Any]],
    lst_bands: list[dict[str, Any]],
    start: str,
    end: str,
    host_name: str,
    host_url: str,
    license_id: str,
    updated: str,
) -> dict[str, Any]:
    """The collection: provenance, extent, the thumbnail, and the item mirror."""
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
        "extent": {
            "spatial": {"bbox": [[float(v) for v in bbox]]},
            "temporal": {"interval": [[_rfc3339(start), _rfc3339(end)]]},
        },
        "renders": _renders(lst_bands),
        "assets": assets,
        "links": [
            {"rel": "root", "href": "../catalog.json", "type": JSON_MEDIA_TYPE},
            {"rel": "parent", "href": "../catalog.json", "type": JSON_MEDIA_TYPE},
            {
                "rel": "item",
                "href": f"./{item_id}/{item_id}.json",
                "type": GEOJSON_MEDIA_TYPE,
                "title": f"Tile {item_id}",
            },
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


def _collection_readme(
    collection_id: str, item_id: str, start: str, end: str, license_id: str
) -> str:
    return (
        "# Landsat P95 Land Surface Temperature Composite\n\n"
        "The 95th percentile of clear-sky land surface temperature over "
        f"{start} to {end}, with the count of valid observations behind each "
        "pixel.\n\n"
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


def _agents_md(collection_id: str, item_id: str) -> str:
    return (
        "# Guidance for agents\n\n"
        "## Reading the data\n\n"
        f"Both assets are Cloud Optimized GeoTIFFs on the `{item_id}` item of "
        f"the `{collection_id}` collection. Read a window rather than the "
        "whole file: the internal tiles are "
        f"{BLOCK_SIZE} by {BLOCK_SIZE} pixels and the overviews let you draw "
        "the tile without touching full-resolution pixels.\n\n"
        "## Decoding temperature\n\n"
        f"{_decode_snippet(f'{item_id}/{LST_FILENAME}')}\n"
        f"DN {LST_NODATA_DN} is nodata. It marks a pixel where no observation "
        "survived cloud, shadow, snow, cirrus, and range masking, so treat it "
        "as absent rather than cold.\n\n"
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


def catalog_provenance(meta: dict[str, Any], *, collection_id: str) -> dict[str, Any]:
    """Everything a reader needs to know how this catalog was produced."""
    return {
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "collection_id": collection_id,
        "bbox": [float(v) for v in meta["bbox"]],
        "crs": meta.get("crs", SUPPORTED_CRS),
        "pixels_per_degree": int(meta["pixels_per_degree"]),
        "start": meta.get("start", DEFAULT_START),
        "end": meta.get("end", DEFAULT_END),
        "block_size": BLOCK_SIZE,
        "compression": COMPRESSION,
        "overview_resampling": OVERVIEW_RESAMPLING,
        "lst_scale": LST_SCALE,
        "lst_offset": LST_OFFSET,
        "lst_nodata": LST_NODATA_DN,
    }


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
        ["data"],
    )
    qa_asset["bands"] = qa_bands
    qa_asset["proj:code"] = crs
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


def write_catalog(
    out_dir: Path,
    lst,
    qa,
    meta: dict[str, Any],
    *,
    collection_id: str = DEFAULT_COLLECTION_ID,
    host_name: str = DEFAULT_HOST_NAME,
    host_url: str = DEFAULT_HOST_URL,
    license_id: str = DEFAULT_LICENSE,
) -> Path:
    """Write the whole catalog and return its root directory.

    `meta` is the `part-meta.json` payload: the bbox, the CRS, the pixels per
    degree, and the composite window. Everything else follows from the arrays.
    """
    provenance = catalog_provenance(meta, collection_id=collection_id)
    bbox, crs = provenance["bbox"], provenance["crs"]
    item_id = tile_id(bbox)
    updated = _now()

    root = Path(out_dir)
    collection_dir = root / collection_id
    item_dir = collection_dir / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    lst_path, qa_path = _write_cogs(item_dir, lst, qa, provenance)
    assets = _item_assets(lst_path, qa_path, crs)
    item = build_item(
        item_id,
        bbox,
        assets=assets,
        collection_id=collection_id,
        start=provenance["start"],
        end=provenance["end"],
        crs=crs,
    )
    _dump(item_dir / f"{item_id}.json", item)

    thumbnail = render_thumbnail(collection_dir / THUMBNAIL_FILENAME, lst)
    mirror = write_item_mirror(collection_dir / MIRROR_FILENAME, [item])
    collection = build_collection(
        collection_id,
        bbox,
        item_id=item_id,
        assets=_collection_assets(thumbnail, mirror),
        lst_bands=assets[LST_ASSET_KEY]["bands"],
        start=provenance["start"],
        end=provenance["end"],
        host_name=host_name,
        host_url=host_url,
        license_id=license_id,
        updated=updated,
    )
    _dump(collection_dir / "collection.json", collection)
    _dump(root / "catalog.json", build_root_catalog(collection_id, updated=updated))

    (collection_dir / "README.md").write_text(
        _collection_readme(
            collection_id, item_id, provenance["start"], provenance["end"], license_id
        )
    )
    (collection_dir / "AGENTS.md").write_text(_agents_md(collection_id, item_id))
    (root / "README.md").write_text(_root_readme(collection_id))
    (root / "AGENTS.md").write_text(_agents_md(collection_id, item_id))
    return root

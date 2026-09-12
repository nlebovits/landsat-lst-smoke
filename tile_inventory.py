# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["pyarrow>=16"]
# ///
"""Read one tile's scenes out of the precomputed inventory.

This is the runtime half. It opens a Parquet file, reads the row group that
holds one tile, and returns STAC item dicts the loader accepts. It opens no
catalogue, builds no land mask, and reads no row group belonging to another
tile.

The artifact is written by `usgs_inventory` with one row group per tile and
`tile_id` statistics on every group, so selecting a tile is a metadata lookup
followed by a single group read.

A missing, stale, or mismatched artifact stops the run here, before the fleet
starts. There is no Earth Search fallback: a silent fallback would turn a
configuration mistake into hundreds of machines quietly doing the slow thing.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The STAC extensions the loader looks for. `projection` is the one that
#: matters: `pystac` reports `proj:*` only when the item declares it.
STAC_EXTENSIONS = (
    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
    "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
)

GEOTIFF = "image/tiff; application=geotiff; profile=cloud-optimized"

#: The asset objects Earth Search publishes, minus the href. Copied field for
#: field, because `raster:bands` carries the nodata values the loader applies:
#: 0 for the thermal band, 1 for QA_PIXEL. Getting either wrong changes which
#: pixels the mask keeps.
ASSET_TEMPLATES = {
    "lwir11": {
        "type": GEOTIFF,
        "title": "Surface Temperature Band",
        "eo:bands": [
            {
                "name": "TIRS_B10",
                "common_name": "lwir11",
                "center_wavelength": 10.9,
                "full_width_half_max": 0.59,
            }
        ],
        "raster:bands": [
            {
                "nodata": 0,
                "data_type": "uint16",
                "spatial_resolution": 30,
                "unit": "kelvin",
                "scale": 0.00341802,
                "offset": 149,
            }
        ],
        "gsd": 100,
        "roles": ["data", "temperature"],
    },
    "qa_pixel": {
        "type": GEOTIFF,
        "title": "Pixel Quality Assessment Band",
        "raster:bands": [
            {
                "nodata": 1,
                "data_type": "uint16",
                "spatial_resolution": 30,
                "unit": "bit index",
            }
        ],
        "roles": ["cloud", "cloud-shadow", "snow-ice", "water-mask"],
    },
}

COLLECTION = "landsat-c2-l2"

#: Schema version of `tile_scene_inventory.parquet`. The reader owns it,
#: because the reader is what refuses an artifact it cannot interpret.
#: `usgs_inventory` stamps this value into every manifest it writes.
INVENTORY_SCHEMA_VERSION = 1


class InventoryError(RuntimeError):
    """The inventory cannot answer this run. Raised before any compute."""


# --------------------------------------------------------------------------
# Manifest.
# --------------------------------------------------------------------------


def read_manifest(path: Path | str) -> dict:
    """The manifest stored in the artifact's Parquet metadata."""
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        msg = (
            f"no inventory at {path}. Build it with:\n"
            f"  uv run land_tiles.py --out artifacts/land_tiles.parquet\n"
            f"  uv run usgs_inventory.py --out {path}"
        )
        raise InventoryError(msg)
    try:
        meta = pq.ParquetFile(path).schema_arrow.metadata or {}
        raw = meta.get(b"manifest")
    except Exception as exc:  # noqa: BLE001 - any reader failure is fatal here
        msg = f"cannot read {path} as Parquet: {exc}"
        raise InventoryError(msg) from exc
    if not raw:
        msg = (
            f"{path} carries no manifest. It was not written by "
            f"usgs_inventory, or it predates the manifest. Rebuild it."
        )
        raise InventoryError(msg)
    return json.loads(raw)


def check_manifest(
    manifest: dict,
    *,
    start: str,
    end: str,
    platforms: str,
    cloud_cover_lt: int,
    schema_version: int,
) -> None:
    """Refuse a run whose parameters the artifact does not cover.

    Every mismatch is a wrong answer that would look like a right one. A
    composite built from a 2020 inventory under a 2021 window produces a
    finished raster, priced and written, with the wrong scenes in it.

    Raises:
        InventoryError: naming the field, both values, and the fix.
    """
    problems = []
    if manifest.get("schema_version") != schema_version:
        problems.append(
            f"schema_version {manifest.get('schema_version')} != "
            f"{schema_version} expected by this code"
        )
    for field, want in (
        ("start", start),
        ("end", end),
        ("platforms", platforms),
        ("cloud_cover_lt", cloud_cover_lt),
    ):
        got = manifest.get(field)
        if got != want:
            problems.append(f"{field} {got!r} in the artifact, {want!r} requested")
    if problems:
        joined = "\n  ".join(problems)
        msg = (
            f"the inventory does not cover this run:\n  {joined}\n"
            f"Rebuild it with matching arguments, or pass the arguments the "
            f"artifact was built with. Artifact generated "
            f"{manifest.get('generated_utc')} from "
            f"{manifest.get('source', {}).get('last_modified')}."
        )
        raise InventoryError(msg)


def provenance(manifest: dict) -> dict:
    """The subset of the manifest that belongs in a run record."""
    source = manifest.get("source", {})
    return {
        "inventory_schema_version": manifest.get("schema_version"),
        "inventory_generated_utc": manifest.get("generated_utc"),
        "inventory_generator_commit": manifest.get("generator_commit"),
        "source_url": source.get("url"),
        "source_last_modified": source.get("last_modified"),
        "source_sha256": source.get("sha256"),
        "land_geometry_sha256": manifest.get("land_geometry_sha256"),
        "natural_earth_version": manifest.get("natural_earth_version"),
        "buffer_meters": manifest.get("buffer_meters"),
        "latitude_limit": manifest.get("latitude_limit"),
        "tile_count": manifest.get("tile_count"),
        "start": manifest.get("start"),
        "end": manifest.get("end"),
        "platforms": manifest.get("platforms"),
        "cloud_cover_lt": manifest.get("cloud_cover_lt"),
    }


# --------------------------------------------------------------------------
# Reading one tile.
# --------------------------------------------------------------------------


def row_groups_for_tile(parquet_file, tile_id: str) -> list[int]:
    """Row groups whose `tile_id` statistics can hold this tile.

    The writer sorts by `tile_id` and closes a group at every change, so this
    returns one group for a tile that is present and none for a tile that is
    absent. Reading the statistics costs no column data.
    """
    meta = parquet_file.metadata
    col = parquet_file.schema_arrow.names.index("tile_id")
    groups = []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(col).statistics
        if stats is None:
            groups.append(i)
        elif stats.min <= tile_id <= stats.max:
            groups.append(i)
    return groups


def thermal_rows_for_tile(parquet_file, tile_id: str) -> int:
    """How many of this tile's scenes carry a thermal band.

    126 of the 895 land tiles answer zero, and a run there stages every scene
    and composites nothing. S15W180 alone holds 4,274 such scenes, about
    350 GB of staging for an all-nodata output. `fleet_plan` asks this before
    it puts a machine on a tile.

    `thermal_href` is null exactly when `data_type` is `OLI_TIRS_L2SR`, over
    all 3,083,129 rows of the full artifact with no exceptions. USGS emits
    that product where the Collection 2 surface temperature algorithm has no
    usable emissivity, which is why every zero-thermal tile is an ocean tile
    holding a small island.

    The answer comes from the null-count statistics when a row group holds one
    tile, which is how `usgs_inventory` writes it. That reads no column data at
    all. A group without statistics, or one spanning more than this tile, falls
    back to reading the two columns it needs.
    """
    meta = parquet_file.metadata
    names = parquet_file.schema_arrow.names
    tile_col = names.index("tile_id")
    thermal_col = names.index("thermal_href")
    total = 0
    for i in row_groups_for_tile(parquet_file, tile_id):
        group = meta.row_group(i)
        tile_stats = group.column(tile_col).statistics
        stats = group.column(thermal_col).statistics
        exclusive = (
            tile_stats is not None and tile_stats.min == tile_stats.max == tile_id
        )
        if exclusive and stats is not None and stats.has_null_count:
            total += group.num_rows - stats.null_count
            continue
        table = parquet_file.read_row_groups([i], columns=["tile_id", "thermal_href"])
        tiles = table.column("tile_id").to_pylist()
        hrefs = table.column("thermal_href").to_pylist()
        total += sum(
            1
            for name, href in zip(tiles, hrefs, strict=True)
            if name == tile_id and href is not None
        )
    return total


def _tile_local_bbox(west, south, east, north, crossing, bounds):
    """A scene bbox trimmed to the tile's own longitude span.

    A footprint that wraps the antimeridian has `west > east` in the STAC
    convention, and `composite.block_depths` compares plain intervals.
    Trimming to the tile removes the wrap without changing which blocks the
    scene reaches, because every block lies inside the tile.
    """
    if not crossing:
        return (west, south, east, north)
    tile_west, _, tile_east, _ = bounds
    if tile_west >= 0:
        return (max(west, tile_west), south, 180.0, north)
    return (-180.0, south, min(east, tile_east), north)


def build_item(row: dict, *, collection: str = COLLECTION) -> dict:
    """One STAC item dict, from one inventory row.

    Carries exactly what `odc.stac.stac_load` reads: the projection triple,
    the two asset hrefs with their `raster:bands`, the scene id the loader
    groups on, and the acquisition time. A product with no thermal band gets
    no `lwir11` asset, which is what Earth Search returns for `L2SR`.
    """
    ring = [(row[f"corner_lon{n}"], row[f"corner_lat{n}"]) for n in range(4)]
    ring.append(ring[0])

    assets = {}
    if row["thermal_href"]:
        assets["lwir11"] = {
            **ASSET_TEMPLATES["lwir11"],
            "href": row["thermal_href"],
        }
    assets["qa_pixel"] = {**ASSET_TEMPLATES["qa_pixel"], "href": row["qa_href"]}

    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": list(STAC_EXTENSIONS),
        "id": row["item_id"],
        "collection": collection,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "bbox": [
            row["bbox_west"],
            row["bbox_south"],
            row["bbox_east"],
            row["bbox_north"],
        ],
        "properties": {
            "datetime": row["datetime"].isoformat().replace("+00:00", "Z"),
            "platform": row["platform"],
            "eo:cloud_cover": row["cloud_cover"],
            "landsat:scene_id": row["scene_id"],
            "landsat:wrs_path": f"{row['wrs_path']:03d}",
            "landsat:wrs_row": f"{row['wrs_row']:03d}",
            "landsat:collection_category": row["collection_category"],
            "proj:epsg": row["proj_epsg"],
            "proj:shape": [row["proj_shape_y"], row["proj_shape_x"]],
            "proj:transform": [
                30.0,
                0.0,
                row["proj_origin_x"],
                0.0,
                -30.0,
                row["proj_origin_y"],
            ],
        },
        "assets": assets,
        "links": [],
    }


def items_for_tile(
    path: Path | str, tile_id: str, *, bounds=None
) -> tuple[list[dict], list[tuple]]:
    """Every scene assigned to one tile, as item dicts and bboxes.

    Returns the same pair `shard_lst_p95` used to get from a STAC search, in
    the same acquisition order.

    Raises:
        InventoryError: if the tile holds no rows. The tile list already
            excludes ocean, so an empty tile is a build mistake or a tile the
            fleet should never have been asked to run.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    groups = row_groups_for_tile(pf, tile_id)
    if not groups:
        msg = (
            f"tile {tile_id} has no row group in {path}. Either it is not a "
            f"land tile in this artifact's tile list, or the artifact is "
            f"incomplete."
        )
        raise InventoryError(msg)

    table = pf.read_row_groups(groups)
    rows = [r for r in table.to_pylist() if r["tile_id"] == tile_id]
    if not rows:
        msg = f"tile {tile_id} is absent from {path}"
        raise InventoryError(msg)

    items = [build_item(r) for r in rows]
    boxes = [
        _tile_local_bbox(
            r["bbox_west"],
            r["bbox_south"],
            r["bbox_east"],
            r["bbox_north"],
            r["crosses_antimeridian"],
            bounds,
        )
        if bounds is not None
        else (r["bbox_west"], r["bbox_south"], r["bbox_east"], r["bbox_north"])
        for r in rows
    ]
    return items, boxes


def tile_ids(path: Path | str) -> list[str]:
    """Every tile the artifact holds scenes for, in order.

    Reads only the row-group statistics when each group holds one tile, which
    is how `usgs_inventory` writes it.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.names.index("tile_id")
    meta = pf.metadata
    seen = []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(col).statistics
        if stats is not None and stats.min == stats.max:
            seen.append(stats.min)
        else:
            seen.extend(
                pf.read_row_groups([i], columns=["tile_id"])
                .column("tile_id")
                .to_pylist()
            )
    out: list[str] = []
    for name in seen:
        if not out or out[-1] != name:
            out.append(name)
    return out

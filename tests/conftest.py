"""Fixtures shared by the inventory tests.

Two kinds of inventory appear here, and they answer different questions.

`slice_artifact` is a handful of real tiles cut out of the artifact
`usgs_inventory.py` builds, committed to the repository. It is what lets the
offline guarantee be tested rather than asserted: the full artifact is 167 MB
and gitignored, so every test that needed it used to skip, and on a clean
checkout that was eleven of them, including every socket-blocked check. Real
rows also carry the production column types. `pa.Table.from_pylist` infers
int64 where the writer forces int32, so a synthetic fixture cannot prove a
reader handles what the writer produces.

`synthetic_inventory` is built row by row from `make_row`. It is for the edge
cases the archive does not happen to contain, such as a product with no thermal
band or a footprint that wraps the antimeridian.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from land_tiles import tile_bounds  # noqa: E402
from tile_inventory import INVENTORY_SCHEMA_VERSION  # noqa: E402

#: The committed slice. Built by `tests/make_slice.py`.
SLICE_ARTIFACT = ROOT / "artifacts" / "inventory_slice.parquet"

#: The full build, which is gitignored. Tests that need scale rather than
#: correctness use it and skip when it is absent.
FULL_ARTIFACT = ROOT / "artifacts" / "tile_scene_inventory.parquet"

#: The tiles `make_slice.py` cuts. One dense mid-latitude tile, one temperate,
#: one equatorial, and one on the antimeridian.
SLICE_TILES = ("N05E010", "N40W075", "S15E175", "S30W065")

needs_full_artifact = pytest.mark.skipif(
    not FULL_ARTIFACT.exists(),
    reason="run usgs_inventory.py to build the full artifact",
)

WINDOW = {
    "start": "2021-01-01",
    "end": "2025-12-31T23:59:59Z",
    "platforms": "landsat-8,landsat-9",
    "cloud_cover_lt": 100,
}

MANIFEST = {
    "schema_version": INVENTORY_SCHEMA_VERSION,
    "generated_utc": "2026-09-09T00:00:00+00:00",
    "generator_commit": "abc123",
    "source": {
        "url": "https://example.invalid/LANDSAT_OT_C2_L2.parquet.gz",
        "last_modified": "Mon, 08 Sep 2026 10:05:47 GMT",
        "sha256": "0" * 64,
    },
    "collection": "landsat-c2-l2",
    "natural_earth_version": "ne_10m_land",
    "buffer_meters": 25000,
    "latitude_limit": 60,
    "land_geometry_sha256": "1" * 64,
    "tile_count": 3,
    **WINDOW,
}


def make_row(tile_id, n, *, thermal=True, crossing=False):
    """One inventory row, shaped exactly as `usgs_inventory` writes it."""
    stamp = datetime(2023, 6, n % 28 + 1, 12, 0, tzinfo=UTC)
    base = (
        "s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2023/"
        "227/081/LC08_L2SP_227081_20230601_20230610_02_T1/"
        "LC08_L2SP_227081_20230601_20230610_02_T1"
    )
    west, south, east, north = tile_bounds(tile_id)
    return {
        "tile_id": tile_id,
        "item_id": f"LC08_L2SP_2270{n:02d}_20230601_02_T1",
        "display_id": f"LC08_L2SP_2270{n:02d}_20230601_20230610_02_T1",
        "scene_id": f"LC82270{n:02d}2023152LGN00",
        "datetime": stamp,
        "platform": "landsat-8",
        "collection_category": "T1",
        "data_type": "OLI_TIRS_L2SP" if thermal else "OLI_TIRS_L2SR",
        "wrs_path": 227,
        "wrs_row": 81,
        "cloud_cover": 12.5,
        "bbox_west": 179.0 if crossing else west,
        "bbox_south": south,
        "bbox_east": -179.0 if crossing else east,
        "bbox_north": north,
        "crosses_antimeridian": crossing,
        "proj_epsg": 32620,
        "proj_shape_y": 7891,
        "proj_shape_x": 7831,
        "proj_origin_x": 608385.0,
        "proj_origin_y": -3300285.0,
        "thermal_href": f"{base}_ST_B10.TIF" if thermal else None,
        "qa_href": f"{base}_QA_PIXEL.TIF",
        **{
            f"corner_{axis}{i}": (west + i if axis == "lon" else south + i)
            for i in range(4)
            for axis in ("lon", "lat")
        },
    }


def write_inventory(path, rows, manifest=None):
    """Write rows as `usgs_inventory` does: one row group per tile, sorted.

    The row-group layout is what `row_groups_for_tile` prunes on, so a fixture
    that writes one group for everything would test a different reader.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(rows, key=lambda r: (r["tile_id"], r["datetime"]))
    table = pa.Table.from_pylist(rows)
    meta = {b"manifest": json.dumps(manifest or MANIFEST).encode()}
    schema = table.schema.with_metadata(meta)
    names = sorted({r["tile_id"] for r in rows})
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for name in names:
            chunk = pa.Table.from_pylist(
                [r for r in rows if r["tile_id"] == name], schema=schema
            )
            writer.write_table(chunk, row_group_size=chunk.num_rows)
    return path


@pytest.fixture
def synthetic_inventory(tmp_path):
    """Three tiles of hand-built rows, for cases the archive does not hold."""
    rows = (
        [make_row("S30W065", i) for i in range(5)]
        + [make_row("S30W060", i) for i in range(3)]
        + [make_row("S35W065", i) for i in range(4)]
    )
    return write_inventory(tmp_path / "inv.parquet", rows)


@pytest.fixture(scope="session")
def slice_artifact():
    """The committed slice of the real inventory.

    Session-scoped: every test that reads it only reads, and reopening a
    Parquet file per test buys nothing.
    """
    if not SLICE_ARTIFACT.exists():
        pytest.fail(
            f"{SLICE_ARTIFACT} is missing. It is committed to the repository; "
            f"rebuild it with `uv run tests/make_slice.py` from a full "
            f"artifact if it has been deleted."
        )
    return SLICE_ARTIFACT

"""The runtime read: pruning, manifest checks, and the item it builds.

These tests need no network. Two inventories appear. `synthetic_inventory` is
built row by row, so it can hold cases the archive does not, such as a product
with no thermal band. `slice_artifact` is real rows cut from a full build and
committed, so the reader is exercised against the column types and the values
the writer actually produces. Both come from `tests/conftest.py`.

The tests that need the full 167 MB build are the ones about scale, and only
those skip when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import (  # noqa: E402
    MANIFEST,
    SLICE_TILES,
    WINDOW,
    make_row,
    needs_full_artifact,
    write_inventory,
)
from land_tiles import tile_bounds  # noqa: E402
from tile_inventory import (  # noqa: E402
    ASSET_TEMPLATES,
    InventoryError,
    build_item,
    check_manifest,
    items_for_tile,
    provenance,
    read_manifest,
    row_groups_for_tile,
    tile_ids,
)
from usgs_inventory import INVENTORY_SCHEMA_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _check(**overrides):
    """`check_manifest` with the matching parameters, and any one changed."""
    window = {**WINDOW, "schema_version": INVENTORY_SCHEMA_VERSION, **overrides}
    return check_manifest(
        MANIFEST,
        start=str(window["start"]),
        end=str(window["end"]),
        platforms=str(window["platforms"]),
        cloud_cover_lt=int(window["cloud_cover_lt"]),
        schema_version=int(window["schema_version"]),
    )


@pytest.fixture
def small(tmp_path):
    """Three tiles of synthetic rows, written the way usgs_inventory writes."""
    rows = (
        [make_row("S30W065", i) for i in range(5)]
        + [make_row("S30W060", i) for i in range(3)]
        + [make_row("S35W065", i) for i in range(4)]
    )
    return write_inventory(tmp_path / "inv.parquet", rows)


# --------------------------------------------------------------------------
# Row-group pruning.
# --------------------------------------------------------------------------


class TestPruning:
    def test_one_row_group_per_tile(self, small):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(small)
        assert pf.metadata.num_row_groups == 3

    def test_a_tile_selects_exactly_one_group(self, small):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(small)
        for name in ("S30W060", "S30W065", "S35W065"):
            assert len(row_groups_for_tile(pf, name)) == 1

    def test_an_absent_tile_selects_none(self, small):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(small)
        assert row_groups_for_tile(pf, "N40W075") == []

    def test_reading_one_tile_returns_only_that_tile(self, small):
        items, _ = items_for_tile(small, "S30W065")
        assert len(items) == 5

    def test_absent_tile_raises_rather_than_returning_nothing(self, small):
        with pytest.raises(InventoryError, match="no row group"):
            items_for_tile(small, "N40W075")

    def test_tile_ids_reads_statistics_only(self, small):
        assert tile_ids(small) == ["S30W060", "S30W065", "S35W065"]

    @needs_full_artifact
    def test_single_tile_read_touches_a_fraction_of_the_file(self):
        """The point of the layout: one tile is not a scan of the file.

        This one needs the full build. A fraction is only meaningful against
        895 row groups; the committed slice has four, so a tile is a quarter of
        it by construction and the ratio would say nothing.
        """
        import pyarrow.parquet as pq

        from conftest import FULL_ARTIFACT

        pf = pq.ParquetFile(FULL_ARTIFACT)
        groups = row_groups_for_tile(pf, "S30W065")
        assert len(groups) == 1
        touched = sum(pf.metadata.row_group(g).total_byte_size for g in groups)
        whole = sum(
            pf.metadata.row_group(i).total_byte_size
            for i in range(pf.metadata.num_row_groups)
        )
        assert touched / whole < 0.01, f"one tile read {touched:,} of {whole:,} bytes"


# --------------------------------------------------------------------------
# The manifest gate.
# --------------------------------------------------------------------------


class TestManifest:
    def test_reads_the_embedded_manifest(self, small):
        assert read_manifest(small)["start"] == "2021-01-01"

    def test_missing_file_names_the_build_command(self, tmp_path):
        with pytest.raises(InventoryError, match="usgs_inventory.py"):
            read_manifest(tmp_path / "absent.parquet")

    def test_a_parquet_without_a_manifest_is_refused(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "bare.parquet"
        pq.write_table(pa.table({"tile_id": ["S30W065"]}), path)
        with pytest.raises(InventoryError, match="no manifest"):
            read_manifest(path)

    def test_an_unreadable_file_is_refused(self, tmp_path):
        path = tmp_path / "junk.parquet"
        path.write_bytes(b"not parquet at all")
        with pytest.raises(InventoryError, match="cannot read"):
            read_manifest(path)

    def test_matching_parameters_pass(self):
        _check()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("start", "2020-01-01"),
            ("end", "2025-01-01"),
            ("platforms", "landsat-8"),
            ("cloud_cover_lt", 50),
            ("schema_version", 99),
        ],
    )
    def test_any_mismatch_is_refused(self, field, value):
        with pytest.raises(InventoryError) as exc:
            _check(**{field: value})
        assert field in str(exc.value)
        assert "Rebuild" in str(exc.value)

    def test_provenance_names_the_source_and_the_land(self):
        p = provenance(MANIFEST)
        assert p["source_sha256"] == "0" * 64
        assert p["land_geometry_sha256"] == "1" * 64
        assert p["buffer_meters"] == 25000
        assert p["latitude_limit"] == 60
        assert p["inventory_generator_commit"] == "abc123"


# --------------------------------------------------------------------------
# The item it builds.
# --------------------------------------------------------------------------


class TestBuildItem:
    def test_carries_the_projection_triple(self):
        item = build_item(make_row("S30W065", 1))
        props = item["properties"]
        assert props["proj:epsg"] == 32620
        assert props["proj:shape"] == [7891, 7831]
        assert props["proj:transform"] == [30.0, 0.0, 608385.0, 0.0, -30.0, -3300285.0]

    def test_declares_the_projection_extension(self):
        """pystac reports proj:* only when the item declares the extension."""
        item = build_item(make_row("S30W065", 1))
        assert any("projection" in u for u in item["stac_extensions"])
        assert any("raster" in u for u in item["stac_extensions"])

    def test_both_bands_carry_their_nodata(self):
        """0 for the thermal band, 1 for QA_PIXEL. The mask depends on it."""
        item = build_item(make_row("S30W065", 1))
        assert item["assets"]["lwir11"]["raster:bands"][0]["nodata"] == 0
        assert item["assets"]["qa_pixel"]["raster:bands"][0]["nodata"] == 1

    def test_l2sr_has_no_thermal_asset(self):
        """Earth Search omits lwir11 for L2SR. So does this."""
        item = build_item(make_row("S30W065", 1, thermal=False))
        assert "lwir11" not in item["assets"]
        assert "qa_pixel" in item["assets"]

    def test_asset_media_type_marks_it_as_raster_data(self):
        for template in ASSET_TEMPLATES.values():
            assert str(template["type"]).startswith("image/tiff")

    def test_scene_id_is_present_for_grouping(self):
        """stac_load groups on landsat:scene_id."""
        item = build_item(make_row("S30W065", 1))
        assert item["properties"]["landsat:scene_id"].startswith("LC8")

    def test_datetime_is_utc_iso(self):
        item = build_item(make_row("S30W065", 1))
        assert item["properties"]["datetime"].endswith("Z")

    def test_geometry_ring_is_closed(self):
        item = build_item(make_row("S30W065", 1))
        ring = item["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]
        assert len(ring) == 5

    def test_pystac_accepts_it(self):
        pystac = pytest.importorskip("pystac")
        item = pystac.Item.from_dict(build_item(make_row("S30W065", 1)))
        assert item.id.startswith("LC08")
        assert item.datetime is not None


class TestAntimeridianBbox:
    def test_a_wrapping_bbox_is_trimmed_to_the_tile(self):
        """items_for_shard compares plain intervals, so the wrap has to go."""
        from tile_inventory import _tile_local_bbox

        east_side = _tile_local_bbox(
            179.0, -20.0, -179.0, -15.0, True, tile_bounds("S15E175")
        )
        assert east_side == (179.0, -20.0, 180.0, -15.0)

        west_side = _tile_local_bbox(
            179.0, -20.0, -179.0, -15.0, True, tile_bounds("S15W180")
        )
        assert west_side == (-180.0, -20.0, -179.0, -15.0)

    def test_a_normal_bbox_is_untouched(self):
        from tile_inventory import _tile_local_bbox

        plain = _tile_local_bbox(-64.0, -35.0, -62.0, -33.0, False, None)
        assert plain == (-64.0, -35.0, -62.0, -33.0)


# --------------------------------------------------------------------------
# The built artifact.
# --------------------------------------------------------------------------


class TestBuiltArtifact:
    """Against the committed slice, so these run on a clean checkout.

    They used to need the 167 MB build and skipped without it, which meant CI
    never checked any of them.
    """

    def test_manifest_records_the_window_and_the_land(self, slice_artifact):
        m = read_manifest(slice_artifact)
        assert m["start"] == "2021-01-01"
        assert m["end"] == "2025-12-31T23:59:59Z"
        assert m["platforms"] == "landsat-8,landsat-9"
        assert m["natural_earth_version"] == "ne_10m_land"
        assert m["buffer_meters"] == 25000
        assert m["latitude_limit"] == 60
        assert len(m["land_geometry_sha256"]) == 64
        assert len(m["source"]["sha256"]) == 64

    def test_snap_stayed_well_inside_the_limit(self, slice_artifact):
        from usgs_inventory import MAX_SNAP_METERS

        assert read_manifest(slice_artifact)["max_snap_meters"] < MAX_SNAP_METERS

    def test_every_row_group_holds_one_tile(self, slice_artifact):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(slice_artifact)
        col = pf.schema_arrow.names.index("tile_id")
        for i in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(i).column(col).statistics
            assert stats.min == stats.max

    def test_rows_are_sorted_by_tile_then_time(self, slice_artifact):
        items, _ = items_for_tile(slice_artifact, "S30W065")
        stamps = [i["properties"]["datetime"] for i in items]
        assert stamps == sorted(stamps)

    def test_the_slice_holds_every_tile_it_claims(self, slice_artifact):
        assert tile_ids(slice_artifact) == sorted(SLICE_TILES)

    def test_integer_columns_keep_the_writer_s_widths(self, slice_artifact):
        """Real rows, so the types are the writer's rather than inferred.

        `pa.Table.from_pylist` widens these to int64, which is why a synthetic
        fixture cannot stand in for this check.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(slice_artifact).schema_arrow
        for name in (
            "wrs_path",
            "wrs_row",
            "proj_epsg",
            "proj_shape_y",
            "proj_shape_x",
        ):
            assert schema.field(name).type == pa.int32(), name

    def test_a_real_tile_builds_loadable_items(self, slice_artifact):
        """The contract is the geobox odc-stac derives, not the raw property.

        `pystac` migrates `proj:epsg` to `proj:code` on read, exactly as it
        does for an Earth Search item, so the assertion is on the geobox.
        """
        pystac = pytest.importorskip("pystac")
        odc_stac = pytest.importorskip("odc.stac")
        from odc.stac._mdtools import extract_collection_metadata

        items, boxes = items_for_tile(
            slice_artifact, "S30W065", bounds=tile_bounds("S30W065")
        )
        assert len(items) == len(boxes) > 0
        for raw in items[:20]:
            assert raw["properties"]["proj:epsg"] > 32600
            item = pystac.Item.from_dict(raw)
            assert "qa_pixel" in item.assets
            parsed = odc_stac.parse_item(item, extract_collection_metadata(item))
            geobox = parsed.geoboxes()[0]
            assert geobox.crs.epsg == raw["properties"]["proj:epsg"]
            assert tuple(geobox.shape) == tuple(raw["properties"]["proj:shape"])
            assert geobox.transform.c == raw["properties"]["proj:transform"][2]

    def test_an_antimeridian_tile_reads_back(self, slice_artifact):
        """`S15E175` is in the slice because the seam is where bboxes wrap."""
        items, boxes = items_for_tile(
            slice_artifact, "S15E175", bounds=tile_bounds("S15E175")
        )
        assert len(items) == len(boxes) > 0
        for west, _, east, _ in boxes:
            assert west <= east, "a tile-local bbox must not wrap"


@needs_full_artifact
class TestFullArtifactScale:
    """Claims that only the whole build can support."""

    def test_every_land_tile_is_present(self):
        from conftest import FULL_ARTIFACT

        m = read_manifest(FULL_ARTIFACT)
        assert m["tile_count"] == m["tiles_with_scenes"]
        assert len(tile_ids(FULL_ARTIFACT)) == m["tile_count"]

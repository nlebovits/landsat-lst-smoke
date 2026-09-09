"""The land rule, and the grid it selects.

The tile list decides what the fleet buys, so every claim it rests on is
checked here: the Natural Earth release, the buffer, the latitude limit, and
the two defects the shared production method carries. Nothing in this file
asserts a tile count. The count is a measurement, and pinning it would turn a
corrected land rule into a test failure.

Only the geometry tests need the Natural Earth download. They are marked so a
network-free run still checks the grid, the naming, and the artifact format.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import land_tiles  # noqa: E402
from land_tiles import (  # noqa: E402
    COASTAL_BUFFER_METERS,
    LATITUDE_LIMIT,
    MAX_NATURAL_EARTH_SCALERANK,
    MERCATOR_X_MAX,
    NATURAL_EARTH_URL,
    TILE_SIZE_DEGREES,
    drop_placeholder_features,
    iter_grid,
    land_tiles_provenance,
    read_land_tiles,
    tile_bounds,
    tile_name,
    write_land_tiles,
)

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "land_tiles.parquet"

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="run land_tiles.py to build artifacts/"
)


# --------------------------------------------------------------------------
# The rule itself.
# --------------------------------------------------------------------------


class TestLandRule:
    def test_source_is_10m_not_110m(self):
        """The frozen 700-tile set came from 110m. The pixel mask uses 10m."""
        assert "10m" in NATURAL_EARTH_URL
        assert "110m" not in NATURAL_EARTH_URL
        assert NATURAL_EARTH_URL.endswith("ne_10m_land.zip")

    def test_buffer_is_25km(self):
        assert COASTAL_BUFFER_METERS == 25_000

    def test_latitude_limit_is_60(self):
        assert LATITUDE_LIMIT == 60

    def test_no_land_fraction_threshold(self):
        """The buffer is the rule. A second threshold would be a second rule."""
        source = Path(land_tiles.__file__).read_text()
        assert "land_fraction" not in source
        assert "fraction >" not in source


# --------------------------------------------------------------------------
# The grid.
# --------------------------------------------------------------------------


class TestGrid:
    def test_cell_count(self):
        """24 latitude bands by 72 longitude columns."""
        assert len(iter_grid()) == 1728

    def test_names_round_trip(self):
        for name, bounds in iter_grid():
            assert tile_bounds(name) == bounds

    def test_naming_matches_production(self):
        """`S30W065` is lat (-35, -30], lon [-65, -60), as FINDINGS records."""
        assert tile_name(-30, -65) == "S30W065"
        assert tile_bounds("S30W065") == (-65.0, -35.0, -60.0, -30.0)
        assert tile_name(40, -75) == "N40W075"
        assert tile_name(0, 5) == "N00E005"

    def test_grid_stops_at_the_latitude_limit(self):
        souths = [b[1] for _, b in iter_grid()]
        norths = [b[3] for _, b in iter_grid()]
        assert min(souths) == -LATITUDE_LIMIT
        assert max(norths) == LATITUDE_LIMIT

    def test_no_cell_crosses_the_antimeridian(self):
        """180 is a grid line, so west is always less than east."""
        for _, (west, _, east, _) in iter_grid():
            assert west < east
            assert -180.0 <= west < 180.0
            assert -175.0 <= east <= 180.0

    def test_the_seam_columns_both_exist(self):
        names = {name for name, _ in iter_grid()}
        assert "N00E175" in names
        assert "N00W180" in names
        assert tile_bounds("N00E175")[2] == 180.0
        assert tile_bounds("N00W180")[0] == -180.0

    def test_cells_tile_the_band_without_gaps(self):
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for _, b in iter_grid())
        assert area == pytest.approx(360 * 2 * LATITUDE_LIMIT)


# --------------------------------------------------------------------------
# The two corrections.
# --------------------------------------------------------------------------


class TestPlaceholderRemoval:
    def _frame(self, rows):
        gpd = pytest.importorskip("geopandas")

        geoms, ranks = [], []
        for rank, geom in rows:
            geoms.append(geom)
            ranks.append(rank)
        return gpd.GeoDataFrame({"scalerank": ranks}, geometry=geoms, crs=4326)

    def test_drops_the_null_island_record(self):
        """Natural Earth ships a 1 km square at 0, 0 with scalerank 100."""
        from shapely.geometry import box

        frame = self._frame(
            [(0.0, box(10, 10, 12, 12)), (100.0, box(-0.005, -0.004, 0.004, 0.005))]
        )
        kept = drop_placeholder_features(frame)
        assert len(kept) == 1
        assert kept.iloc[0]["scalerank"] == 0.0

    def test_keeps_records_with_no_scalerank(self):
        """One real Natural Earth feature carries NaN. It is land."""
        from shapely.geometry import box

        frame = self._frame([(float("nan"), box(10, 10, 12, 12))])
        assert len(drop_placeholder_features(frame)) == 1

    def test_drops_zero_area_geometry(self):
        from shapely.geometry import LineString, box

        frame = self._frame(
            [(0.0, box(10, 10, 12, 12)), (0.0, LineString([(0, 0), (1, 1)]))]
        )
        assert len(drop_placeholder_features(frame)) == 1

    def test_scalerank_limit_matches_natural_earth(self):
        assert MAX_NATURAL_EARTH_SCALERANK == 9


class TestAntimeridianBuffer:
    """A coastal buffer must not fold into a sliver that circles the planet."""

    def _series(self, polys):
        gpd = pytest.importorskip("geopandas")
        return gpd.GeoSeries(polys, crs="EPSG:3857")

    def test_a_seam_polygon_stays_inside_the_world(self):
        from shapely.geometry import box

        # A small island hard against the antimeridian, in EPSG:3857.
        island = box(MERCATOR_X_MAX - 5_000, 0, MERCATOR_X_MAX - 1_000, 4_000)
        out = land_tiles._buffer_without_wrapping(self._series([island]), 25_000)
        for geom in out:
            assert geom.bounds[0] >= -MERCATOR_X_MAX - 1e-6
            assert geom.bounds[2] <= MERCATOR_X_MAX + 1e-6

    def test_the_buffer_reaches_both_sides_of_the_seam(self):
        """25 km east of 180 belongs to the western column, not to nowhere."""
        from shapely.geometry import box

        island = box(MERCATOR_X_MAX - 5_000, 0, MERCATOR_X_MAX - 1_000, 4_000)
        out = land_tiles._buffer_without_wrapping(self._series([island]), 25_000)
        east = any(g.bounds[2] > MERCATOR_X_MAX - 1_000 for g in out)
        west = any(g.bounds[0] < -MERCATOR_X_MAX + 25_000 for g in out)
        assert east, "the eastern half of the buffer is missing"
        assert west, "the buffer did not wrap to the western column"

    def test_no_output_part_spans_more_than_half_the_world(self):
        from shapely.geometry import box

        island = box(MERCATOR_X_MAX - 5_000, 0, MERCATOR_X_MAX - 1_000, 4_000)
        out = land_tiles._buffer_without_wrapping(self._series([island]), 25_000)
        for geom in out:
            assert (geom.bounds[2] - geom.bounds[0]) < MERCATOR_X_MAX


# --------------------------------------------------------------------------
# The artifact.
# --------------------------------------------------------------------------


class TestArtifact:
    def test_round_trip(self, tmp_path):
        from shapely.geometry import box

        tiles = [
            {
                "tile_id": name,
                "west": b[0],
                "south": b[1],
                "east": b[2],
                "north": b[3],
                "geometry": box(*b).wkb,
            }
            for name, b in iter_grid()[:4]
        ]
        prov = land_tiles_provenance(tiles, land_checksum="abc")
        out = write_land_tiles(tmp_path / "t.parquet", tiles, prov)
        names, back = read_land_tiles(out)
        assert names == [t["tile_id"] for t in tiles]
        assert back["buffer_meters"] == str(COASTAL_BUFFER_METERS)
        assert back["natural_earth_version"] == "ne_10m_land"
        assert back["latitude_limit"] == str(LATITUDE_LIMIT)
        assert back["tile_size_degrees"] == str(TILE_SIZE_DEGREES)
        assert back["land_geometry_sha256"] == "abc"
        assert back["crs"] == "EPSG:4326"

    def test_geoparquet_metadata_is_present(self, tmp_path):
        import json

        import pyarrow.parquet as pq
        from shapely.geometry import box

        name, b = iter_grid()[0]
        tiles = [
            {
                "tile_id": name,
                "west": b[0],
                "south": b[1],
                "east": b[2],
                "north": b[3],
                "geometry": box(*b).wkb,
            }
        ]
        out = write_land_tiles(
            tmp_path / "t.parquet", tiles, land_tiles_provenance(tiles)
        )
        geo = json.loads(pq.ParquetFile(out).schema_arrow.metadata[b"geo"])
        assert geo["primary_column"] == "geometry"
        assert geo["columns"]["geometry"]["encoding"] == "WKB"
        assert geo["columns"]["geometry"]["crs"] == "EPSG:4326"


# --------------------------------------------------------------------------
# The built list, when it exists.
# --------------------------------------------------------------------------


@needs_artifact
class TestBuiltList:
    @pytest.fixture(scope="class")
    @classmethod
    def built(cls):
        return read_land_tiles(ARTIFACT)

    def test_every_tile_is_a_grid_cell(self, built):
        names, _ = built
        assert set(names) <= {n for n, _ in iter_grid()}

    def test_sorted_and_unique(self, built):
        names, _ = built
        assert names == sorted(names)
        assert len(names) == len(set(names))

    def test_provenance_records_the_rule(self, built):
        _, prov = built
        assert prov["natural_earth_version"] == "ne_10m_land"
        assert prov["buffer_meters"] == "25000"
        assert prov["buffer_crs"] == "EPSG:3857"
        assert prov["latitude_limit"] == "60"
        assert len(prov["land_geometry_sha256"]) == 64

    def test_count_is_recorded_not_asserted(self, built):
        names, prov = built
        assert int(prov["tile_count"]) == len(names)

    def test_the_known_ocean_artifacts_are_absent(self, built):
        """Null Island and the antimeridian slivers selected open ocean."""
        names, _ = built
        for ocean in ("N00E000", "N00W005", "S05E000", "S05W005"):
            assert ocean not in names, f"{ocean} is the Null Island disc"
        for ocean in ("N55W030", "N55W045", "S05E085", "S05W090"):
            assert ocean not in names, f"{ocean} came from a wrapped buffer"

    def test_real_antimeridian_land_is_present(self, built):
        """The Aleutians and Fiji straddle the seam and are land."""
        names, _ = built
        for seam in ("N55E175", "N55W180", "S15E175", "S15W180"):
            assert seam in names

    def test_nothing_outside_the_latitude_band(self, built):
        names, _ = built
        for name in names:
            _, south, _, north = tile_bounds(name)
            assert south >= -LATITUDE_LIMIT
            assert north <= LATITUDE_LIMIT

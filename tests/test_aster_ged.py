"""Where a GED cell lands, and what stops a run from reading the wrong one.

The mask's whole value rests on one thing: a cell covering the ground the
composite thinks it covers. A mosaic built one degree out still produces a
finished raster, still removes about the right number of pixels, and removes
the wrong ones. Nothing downstream can tell.

Two conventions decide the placement and both are documented rather than
guessed. The AG1km granule filename names its NORTHWEST corner, so
`AG1km.v003.33.-115.0010.h5` covers latitude [32, 33] and longitude
[-115, -114]. The grid is 1 degree square at 100 cells, so a cell is 0.01
degree and at 3,600 pixels per degree it is exactly 36 output pixels on a side.
Reading the corner as southwest, or the grid as centre-registered, moves the
mask by 100 cells or by half a cell respectively, and the tests below fail on
either.

The artifacts here are synthetic. Building the real one needs a NASA Earthdata
Login and several gigabytes of granules, which no test can have. The placement
arithmetic is what these check, and that does not need real observations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aster_ged  # noqa: E402
from conftest import write_numobs  # noqa: E402
from land_tiles import tile_bounds  # noqa: E402


class TestTheGranuleFilename:
    """The corner the name carries. Read it wrong and the mask is a degree out."""

    def test_the_user_guide_example_parses(self):
        assert aster_ged.granule_cell("AG1km.v003.33.-115.0010.h5") == (33, -115)

    def test_a_southern_and_eastern_granule_parses(self):
        assert aster_ged.granule_cell("AG1km.v003.-09.124.0010.h5") == (-9, 124)

    @pytest.mark.parametrize(
        "name",
        [
            "AG100.v003.33.-115.0001.h5",  # the 100 m product, 1000 x 1000
            "AG1kmB.v003.33.-115.0010.bin",  # the binary form
            "AG1km.v003.33.-115.0010.h5.tmp",  # a part file
            "readme.txt",
        ],
    )
    def test_anything_else_is_refused(self, name):
        assert aster_ged.granule_cell(name) is None


class TestCellPlacement:
    """Where a cell's northwest corner lands in the global array."""

    def test_the_northwest_corner_of_the_world_is_the_origin(self):
        assert aster_ged.cell_offset(60, -180, 60) == (0, 0)

    def test_the_southeast_corner_ends_one_granule_short_of_the_edge(self):
        rows, cols = aster_ged.mosaic_shape(60)
        row0, col0 = aster_ged.cell_offset(-59, 179, 60)
        assert (row0 + 100, col0 + 100) == (rows, cols)

    def test_the_equator_sits_halfway_down(self):
        rows, _ = aster_ged.mosaic_shape(60)
        assert aster_ged.cell_offset(0, 0, 60)[0] == rows // 2

    def test_a_degree_east_is_a_hundred_columns_east(self):
        a = aster_ged.cell_offset(10, 20, 60)
        b = aster_ged.cell_offset(10, 21, 60)
        assert b[1] - a[1] == aster_ged.CELLS_PER_DEGREE

    def test_a_degree_north_is_a_hundred_rows_up(self):
        a = aster_ged.cell_offset(10, 20, 60)
        b = aster_ged.cell_offset(11, 20, 60)
        assert a[0] - b[0] == aster_ged.CELLS_PER_DEGREE

    def test_the_mosaic_covers_the_band_and_no_more(self):
        assert aster_ged.mosaic_shape(60) == (12_000, 36_000)


class TestNumobsForBbox:
    """The only function a fleet instance calls."""

    @pytest.fixture(scope="class")
    def artifact(self, tmp_path_factory):
        # One gap over S15E175, which is the tile the committed inventory
        # slice holds as an all-L2SR footprint.
        return write_numobs(
            tmp_path_factory.mktemp("ged") / "numobs.tif",
            value=8,
            gaps=[(175.0, -20.0, 180.0, -15.0)],
        )

    def test_a_gap_tile_reads_all_zero(self, artifact):
        out = aster_ged.numobs_for_bbox(artifact, tile_bounds("S15E175"), (500, 500))
        assert set(np.unique(out).tolist()) == {0}

    def test_a_covered_tile_reads_the_fill_value(self, artifact):
        out = aster_ged.numobs_for_bbox(artifact, tile_bounds("S30W065"), (500, 500))
        assert set(np.unique(out).tolist()) == {8}

    def test_the_shape_is_the_shape_asked_for(self, artifact):
        out = aster_ged.numobs_for_bbox(artifact, tile_bounds("S30W065"), (18_000, 900))
        assert out.shape == (18_000, 900)
        assert out.dtype == np.uint8

    def test_one_cell_becomes_thirty_six_pixels_square(self, tmp_path):
        """0.01 degree at 3,600 px per degree, with no half-pixel shift.

        A centre-registered read would put the block at rows 18 to 53 rather
        than 0 to 35, and a bilinear one would smear its edges. Both stay
        invisible in a tile-wide statistic.
        """
        import masks

        rows, cols = aster_ged.mosaic_shape(5)
        mosaic = np.zeros((rows, cols), dtype="uint8")
        row0, col0 = aster_ged.cell_offset(0, 10, 5)
        mosaic[row0, col0] = 9
        manifest = aster_ged.build_manifest(
            {},
            lat_limit=5,
            buffer_meters=25_000,
            land_geometry_sha256="0" * 64,
            cell_count=0,
        )
        path = aster_ged.write_numobs(
            tmp_path / "one_cell.tif", mosaic, manifest, lat_limit=5
        )

        out = aster_ged.numobs_for_bbox(path, (10.0, -1.0, 11.0, 0.0), (3600, 3600))
        ys, xs = np.nonzero(out == 9)
        assert (ys.min(), ys.max()) == (0, 35)
        assert (xs.min(), xs.max()) == (0, 35)
        assert int((out == 9).sum()) == 36 * 36
        assert masks.GAP_NUMOBS == 0

    def test_a_bbox_outside_the_band_is_refused(self, artifact):
        # The mosaic stops at +/-60 degrees, like the tile grid. A silent
        # partial read would put the wrong cells under the northern rows.
        with pytest.raises(aster_ged.GedError, match="latitude span"):
            aster_ged.numobs_for_bbox(artifact, (0.0, 58.0, 5.0, 63.0), (500, 500))

    def test_a_missing_artifact_names_the_build_command(self, tmp_path):
        with pytest.raises(aster_ged.GedError, match="uv run aster_ged.py"):
            aster_ged.numobs_for_bbox(
                tmp_path / "absent.tif", tile_bounds("S30W065"), (500, 500)
            )


class TestTheManifest:
    """A mask built from one land geometry and a tile list from another."""

    def test_it_travels_inside_the_raster(self, numobs_artifact):
        manifest = aster_ged.read_manifest(numobs_artifact)
        assert manifest["schema_version"] == aster_ged.ASTER_GED_SCHEMA_VERSION
        assert manifest["collection"]["short_name"] == "AG1km"
        assert manifest["collection"]["version"] == "003"
        assert manifest["grid"]["cells_per_degree"] == 100

    def test_it_records_how_the_fill_was_mapped(self, numobs_artifact):
        # A reader of the raster alone cannot recover either conversion, and
        # both change which pixels the mask removes.
        encoding = aster_ged.read_manifest(numobs_artifact)["encoding"]
        assert encoding["source_fill"] == -9999
        assert encoding["fill_written_as"] == 0
        assert encoding["clipped_at"] == 255
        assert encoding["gap_rule"] == "numobs == 0"

    def test_a_matching_land_geometry_passes(self, numobs_artifact):
        from conftest import land_geometry_sha256

        aster_ged.check_manifest(
            aster_ged.read_manifest(numobs_artifact),
            land_geometry_sha256=land_geometry_sha256(),
        )

    def test_a_different_land_geometry_is_refused_by_name(self, numobs_artifact):
        manifest = aster_ged.read_manifest(numobs_artifact)
        with pytest.raises(aster_ged.GedError) as exc:
            aster_ged.check_manifest(manifest, land_geometry_sha256="f" * 64)
        message = str(exc.value)
        assert "land_geometry_sha256" in message
        assert "f" * 64 in message
        assert "land_tiles.py first" in message

    def test_a_stale_schema_is_refused(self, numobs_artifact):
        manifest = aster_ged.read_manifest(numobs_artifact) | {"schema_version": 0}
        with pytest.raises(aster_ged.GedError, match="schema_version 0"):
            aster_ged.check_manifest(manifest, land_geometry_sha256="0" * 64)

    def test_the_raster_digest_is_checked_when_the_path_is_given(
        self, numobs_artifact, tmp_path
    ):
        """The digest travels into every run record. Something has to read it.

        Without this the record quotes a number nothing verified, and a
        truncated or swapped mosaic passes the guard that runs before the first
        request.
        """
        from conftest import land_geometry_sha256

        manifest = aster_ged.read_manifest(numobs_artifact)
        digest = aster_ged._sha256(Path(numobs_artifact))
        manifest = manifest | {"raster_sha256": digest}
        aster_ged.check_manifest(
            manifest,
            land_geometry_sha256=land_geometry_sha256(),
            path=numobs_artifact,
        )

        wrong = manifest | {"raster_sha256": "e" * 64}
        with pytest.raises(aster_ged.GedError) as exc:
            aster_ged.check_manifest(
                wrong,
                land_geometry_sha256=land_geometry_sha256(),
                path=numobs_artifact,
            )
        assert "raster_sha256" in str(exc.value)
        assert digest in str(exc.value)

    def test_a_manifest_with_no_digest_still_passes(self, numobs_artifact):
        # The copy inside the raster leaves it empty, because a file cannot
        # contain its own digest. Only the sidecar carries one.
        from conftest import land_geometry_sha256

        manifest = aster_ged.read_manifest(numobs_artifact) | {"raster_sha256": ""}
        aster_ged.check_manifest(
            manifest,
            land_geometry_sha256=land_geometry_sha256(),
            path=numobs_artifact,
        )

    def test_a_missing_artifact_names_the_build_command(self, tmp_path):
        with pytest.raises(aster_ged.GedError, match="uv run aster_ged.py"):
            aster_ged.read_manifest(tmp_path / "absent.tif")

    def test_a_raster_without_a_manifest_is_refused(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin

        path = tmp_path / "bare.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype="uint8",
            count=1,
            height=4,
            width=4,
            crs="EPSG:4326",
            transform=from_origin(0, 0, 0.01, 0.01),
        ) as dst:
            dst.write(np.zeros((4, 4), dtype="uint8"), 1)
        with pytest.raises(aster_ged.GedError, match="carries no manifest"):
            aster_ged.read_manifest(path)

    def test_the_provenance_names_what_a_run_has_to_quote(self, numobs_artifact):
        record = aster_ged.provenance(aster_ged.read_manifest(numobs_artifact))
        for field in ("short_name", "version", "doi", "granule_count"):
            assert field in record


class TestBuildMosaic:
    """Placing granules, without a granule."""

    def test_a_cell_outside_the_band_is_skipped_rather_than_wrapped(self, tmp_path):
        # A granule at 70 degrees north has no row in a +/-60 mosaic. Writing
        # it anywhere would corrupt a real tile; wrapping would corrupt the
        # southern edge, which is the harder failure to see.
        mosaic, covered = aster_ged.build_mosaic(
            {(70, 0): tmp_path / "absent.h5"}, lat_limit=60
        )
        assert not mosaic.any()
        assert not covered.any()

    def test_a_cell_outside_the_columns_is_skipped_too(self, tmp_path):
        # An out-of-range column produces an empty numpy slice, which writes
        # nothing and raises nothing. The row was checked and the column was
        # not, so this is the half of the guard that was missing.
        mosaic, covered = aster_ged.build_mosaic(
            {(30, 180): tmp_path / "absent.h5"}, lat_limit=60
        )
        assert not mosaic.any()
        assert not covered.any()

    def test_the_manifest_names_what_was_placed_not_what_was_cached(self, tmp_path):
        """A skipped granule counted in the manifest is a manifest that lies.

        A cache filled by a build at another latitude limit holds cells this
        mosaic has no room for.
        """
        cache = {
            (30, 0): tmp_path / "inside.h5",
            (70, 0): tmp_path / "too_far_north.h5",
            (30, 180): tmp_path / "off_the_east_edge.h5",
        }
        placed = aster_ged.placeable_granules(cache, lat_limit=60)
        assert set(placed) == {(30, 0)}

    def test_a_cell_inside_the_band_lands_where_the_grid_says(self, tmp_path):
        # The negative cases above pass for a build that places nothing at all.
        # This is the one that fails if placement is broken.
        import h5py
        import numpy as np

        path = tmp_path / "AG1km.v003.30.-65.0010.h5"
        with h5py.File(path, "w") as fh:
            fh.create_dataset(
                f"{aster_ged.NUMOBS_GROUP}/NumObs",
                data=np.full((100, 100), 7, dtype="int16"),
            )
        mosaic, covered = aster_ged.build_mosaic({(30, -65): path}, lat_limit=60)
        row0, col0 = aster_ged.cell_offset(30, -65, 60)
        block = mosaic[row0 : row0 + 100, col0 : col0 + 100]
        assert (block == 7).all()
        assert int(covered.sum()) == 100 * 100
        assert int((mosaic == 7).sum()) == 100 * 100


class TestCoverageIsNotTheCount:
    """A cell with no granule is not a cell ASTER failed to see.

    MEASURED against CMR: ASTER GED AG1km v003 publishes no granule for 813 of
    the 14,941 one-degree cells the buffered land touches, and a search returns
    nothing for them rather than an empty granule. Landsat still holds surface
    temperature over that land. `N05W095` carries 615 thermal scenes of 617 and
    `N00E050` carries 1,031 of 1,034.

    Reading an absent granule as a gap dropped 34 whole tiles from a real fleet
    plan, among them the Solomons at 41 million land pixels, the Aleutians at
    31 million, and the Maldives at 30 million. So the mosaic records coverage
    beside the count, and a gap needs both.
    """

    @pytest.fixture(scope="class")
    def partial(self, tmp_path_factory):
        """A mosaic covering one degree cell of S30W065 and nothing else."""
        rows, cols = aster_ged.mosaic_shape(60)
        mosaic = np.zeros((rows, cols), dtype="uint8")
        covered = np.zeros((rows, cols), dtype="uint8")
        row0, col0 = aster_ged.cell_offset(-30, -65, 60)
        block = slice(row0, row0 + 100), slice(col0, col0 + 100)
        covered[block] = 1  # read, and the count stays 0: a real gap
        manifest = aster_ged.build_manifest(
            {},
            lat_limit=60,
            buffer_meters=25_000,
            land_geometry_sha256="0" * 64,
            cell_count=0,
        )
        return aster_ged.write_numobs(
            tmp_path_factory.mktemp("ged") / "partial.tif",
            mosaic,
            manifest,
            covered=covered,
        )

    def test_the_coverage_band_reads_back(self, partial):
        seen = aster_ged.coverage_for_bbox(
            partial, (-65.0, -31.0, -64.0, -30.0), (100, 100)
        )
        assert seen.all()

    def test_a_cell_with_no_granule_reads_as_uncovered(self, partial):
        seen = aster_ged.coverage_for_bbox(
            partial, (-60.0, -31.0, -59.0, -30.0), (100, 100)
        )
        assert not seen.any()

    def test_both_bands_come_back_from_one_read(self, partial):
        counts, covered = aster_ged.window_for_bbox(
            partial,
            (-65.0, -31.0, -64.0, -30.0),
            (100, 100),
            (aster_ged.NUMOBS_BAND, aster_ged.COVERAGE_BAND),
        )
        assert counts.shape == covered.shape == (100, 100)
        assert not counts.any()
        assert covered.all()

    def test_a_read_cell_at_zero_is_a_gap(self, partial):
        import masks

        gap = masks.emissivity_gap((-65.0, -31.0, -64.0, -30.0), 100, partial)
        assert gap.all()

    def test_an_unread_cell_is_not_a_gap(self, partial):
        import masks

        # Both cells hold a count of zero. Only the read one is evidence.
        gap = masks.emissivity_gap((-60.0, -31.0, -59.0, -30.0), 100, partial)
        assert not gap.any()

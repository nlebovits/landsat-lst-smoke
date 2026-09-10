"""Which pixels the product describes, and the two rules that decide.

`land_tiles.py` opens with the contract this module completes: one geometry
answers both "which tiles does the fleet run" and "which pixels carry a
temperature". A tile chosen from one geometry and masked with another produces
tiles that are entirely nodata, and pixels no tile ever visits. Until now only
the first half existed, and the pixel rule was the one written down but never
built.

Three defects these guard against.

The two rules diverging. A tile the tile list selected and the pixel mask
empties is not a coastal edge case; it is a machine spent on an all-nodata
raster. MEASURED here: at the GED grid of 0.01 degree, `N00E050` holds no land
cell at all, and at the run's 1/3600 degree it holds 1,852 land pixels. So a
coarse land test is not a safe proxy for the fine one, and anything that acts
on the coarse answer has to re-check.

The gap region standing in for the damage. USGS does not leave a gap cell
empty. It interpolates emissivity from the neighbours and retrieves a
temperature, and 89.53% of gap pixels carry one. Removing the region removes
701,839 valid pixels of S30W065 to remove 4,588 bad ones, and 524 of its 605
gap cells hold nothing bad at all. So the rule is the pair, and
`TestTheEmissivityRuleIsAPair` is the truth table that says neither half acts
alone.

`lst_p95` and `qa_count` disagreeing. A `qa_count` above zero beside a nodata
temperature says the pixel had observations and lost them to the reduction.
That is what happened under the emissivity rule, so it leaves the count alone.
Over water the pixel was never this product's subject, so both bands go.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import masks  # noqa: E402
from conftest import (  # noqa: E402
    FULL_LAND_GEOMETRY,
    needs_full_land_geometry,
    needs_land_geometry,
    write_numobs,
)
from land_tiles import read_land_tiles, tile_bounds  # noqa: E402
from lst_qa import LST_NODATA_DN  # noqa: E402

LAND_TILES = ROOT / "artifacts" / "land_tiles.parquet"

#: The committed manifest of the NumObs mosaic. The mosaic itself is 45.5 MB
#: and gitignored; this names the granules behind it and the geometry it was
#: built from.
GED_MANIFEST = ROOT / "artifacts" / "aster_numobs_manifest.json"

#: The buffered geometry's digest, committed so CI can check the tie without
#: the 16 MB file. `land_tiles.py --write-geometry` writes both.
RECORDED_GEOMETRY_SHA256 = ROOT / "artifacts" / "land_buffered_sha256.txt"

#: The tiles the committed geometry slice covers.
SLICE_TILES = ("N05E010", "N40W075", "S15E175", "S30W065")

#: Coarse enough that 895 tiles fit in a test, fine enough to see a coastline.
COARSE_PPD = 100

#: Interior South America. Every pixel is land, so the land rule removes
#: nothing and the emissivity rule is the only one that can act.
INLAND_TILE = "S30W065"


class TestTheTileGrid:
    def test_a_five_degree_tile_at_the_production_grid(self):
        assert masks.raster_shape(tile_bounds("S30W065"), 3600) == (18_000, 18_000)

    def test_the_transform_starts_at_the_north_west_corner(self):
        transform = masks.transform_for(tile_bounds("S30W065"), 3600)
        assert (transform.c, transform.f) == (-65.0, -30.0)
        assert transform.a == pytest.approx(1 / 3600)
        assert transform.e == pytest.approx(-1 / 3600)


@needs_land_geometry
class TestTheLandRule:
    def test_an_inland_tile_keeps_every_pixel(self, land_geometry):
        # S30W065 is interior South America. Nothing there is sea.
        land = masks.land_mask(tile_bounds("S30W065"), COARSE_PPD, land_geometry)
        assert land.all()

    def test_a_coastal_tile_keeps_some_and_drops_some(self, land_geometry):
        land = masks.land_mask(tile_bounds("N40W075"), COARSE_PPD, land_geometry)
        assert land.any()
        assert not land.all()

    def test_the_land_it_finds_is_where_the_land_is(self, land_geometry):
        """Two named points, one on each side of a coast.

        A mask that is inverted, transposed, or a degree out still returns a
        plausible land fraction. Only a point does.
        """
        ppd = 360
        west, _, _, north = bbox = tile_bounds("N40W075")
        land = masks.land_mask(bbox, ppd, land_geometry)

        def at(lon, lat):
            return bool(land[int((north - lat) * ppd), int((lon - west) * ppd)])

        assert at(-74.42, 39.36), "Atlantic City is land"
        assert at(-74.74, 39.99), "Trenton is land"
        assert not at(-71.00, 37.00), "the open Atlantic is not"
        assert not at(-70.50, 35.50), "nor is the deep Atlantic"

    def test_a_missing_geometry_names_the_command_that_writes_it(self, tmp_path):
        with pytest.raises(masks.MaskError, match="--write-geometry"):
            masks.land_mask(
                tile_bounds("S30W065"), COARSE_PPD, tmp_path / "absent.gpkg"
            )

    @needs_full_land_geometry
    def test_its_digest_is_the_one_the_tile_list_records(self):
        """The tie that makes the two rules one rule.

        `fleet_plan.check_mask_artifacts` refuses a plan where these differ.
        Only the full artifact answers it: `land_geometry_sha256` digests those
        bytes, and the committed slice is a different file.
        """
        _, provenance = read_land_tiles(LAND_TILES)
        assert (
            masks.geometry_checksum(FULL_LAND_GEOMETRY)
            == provenance["land_geometry_sha256"]
        )

    def test_every_artifact_records_the_same_geometry_digest(self):
        """The same tie, without the 16 MB file, so CI runs it.

        The geometry is gitignored, so the test above skips on every push and
        the one check binding the tile list, the inventory, and the mask
        artifacts to one geometry went untested. This checks the recorded
        digests against a committed constant instead of against the bytes. It
        cannot catch a corrupted geometry, and it does catch the failure that
        actually happens: one artifact rebuilt without the others.
        """
        recorded = RECORDED_GEOMETRY_SHA256.read_text().strip()
        assert len(recorded) == 64

        _, provenance = read_land_tiles(LAND_TILES)
        assert provenance["land_geometry_sha256"] == recorded

        manifest = json.loads(GED_MANIFEST.read_text())
        assert manifest["land_geometry_sha256"] == recorded

    @needs_full_land_geometry
    def test_the_recorded_digest_is_the_geometry_on_disk(self):
        # What the constant above cannot check, checked wherever the file is.
        recorded = RECORDED_GEOMETRY_SHA256.read_text().strip()
        assert masks.geometry_checksum(FULL_LAND_GEOMETRY) == recorded

    @needs_full_land_geometry
    def test_the_slice_gives_the_same_mask_as_the_full_geometry(self, land_geometry):
        """The committed fixture is a clip, not a simplification.

        `tests/make_land_slice.py` intersects the geometry with each tile's
        bbox, and `land_mask` rasterises only inside that bbox, so the two must
        agree pixel for pixel on the tiles the slice covers. A fixture that
        drifted from the artifact would test a mask the fleet does not run.
        """
        for tile in SLICE_TILES:
            bbox = tile_bounds(tile)
            full = masks.land_mask(bbox, 360, FULL_LAND_GEOMETRY)
            sliced = masks.land_mask(bbox, 360, land_geometry)
            assert np.array_equal(full, sliced), tile


@needs_land_geometry
class TestTheTwoRulesAgreeAboutTheTileList:
    """Every tile the fleet launches has to hold a pixel worth writing."""

    @needs_full_land_geometry
    @pytest.mark.timeout(300)
    def test_no_land_tile_is_emptied_by_the_land_rule(self):
        tiles, _ = read_land_tiles(LAND_TILES)
        assert len(tiles) == 895
        empty = [
            name
            for name in tiles
            if not masks.land_mask(
                tile_bounds(name), COARSE_PPD, FULL_LAND_GEOMETRY
            ).any()
        ]
        # MEASURED: exactly one, and it is a resolution artefact rather than a
        # divergence. N00E050 holds land thinner than a 0.01 degree cell, so a
        # cell-centre test misses it and the run's own grid does not.
        assert empty == ["N00E050"]
        fine = masks.land_mask(tile_bounds("N00E050"), 3600, FULL_LAND_GEOMETRY)
        assert int(fine.sum()) == 1_852

    @needs_full_land_geometry
    def test_a_tile_outside_the_list_is_all_sea(self):
        # N30W040 is mid-Atlantic and is not a land tile. If the pixel rule
        # kept anything there, the two rules would be selecting different
        # ground and the tile list would be the narrower of the two.
        tiles, _ = read_land_tiles(LAND_TILES)
        assert "N30W040" not in tiles
        land = masks.land_mask(tile_bounds("N30W040"), COARSE_PPD, FULL_LAND_GEOMETRY)
        assert not land.any()


@needs_land_geometry
class TestTheEmissivityRule:
    @pytest.fixture(scope="class")
    def gapped(self, tmp_path_factory):
        # A gap over the western half of S30W065, which is otherwise all land.
        return write_numobs(
            tmp_path_factory.mktemp("ged") / "numobs.tif",
            value=8,
            gaps=[(-65.0, -35.0, -62.5, -30.0)],
        )

    def test_a_zero_count_is_the_gap_and_nothing_else_is(self, gapped):
        gap = masks.emissivity_gap(
            tile_bounds("S30W065"), COARSE_PPD, gapped, buffer_cells=0
        )
        assert gap[:, :250].all()
        assert not gap[:, 250:].any()

    def test_the_region_grows_by_whole_cells(self, gapped):
        # The failures sit on the fringe of a gap, so the region grows by a
        # cell before the temperature test runs. At COARSE_PPD one pixel is one
        # cell, so one cell of growth is one column.
        grown = masks.emissivity_gap(tile_bounds("S30W065"), COARSE_PPD, gapped)
        assert grown[:, :251].all()
        assert not grown[:, 251:].any()

    def test_one_cell_grows_to_nine_on_the_production_grid(self, tmp_path):
        # A cell is 36 by 36 pixels at 3,600 per degree, and grown by one cell
        # it is 108 by 108. Growing in pixel space instead would widen the
        # region by a thirty-sixth of what the name says.
        one = write_numobs(
            tmp_path / "one.tif",
            value=8,
            gaps=[(-64.51, -30.51, -64.50, -30.50)],
        )
        bbox = (-65.0, -31.0, -64.0, -30.0)
        bare = masks.emissivity_gap(bbox, 3600, one, buffer_cells=0)
        grown = masks.emissivity_gap(bbox, 3600, one, buffer_cells=1)
        assert int(bare.sum()) == 36 * 36
        assert int(grown.sum()) == 108 * 108

    def test_a_cell_on_the_tile_edge_grows_only_inwards(self, tmp_path):
        # The tile's own corner cell. Two thirds of its grown region falls off
        # the tile, so 4 cells of the 9 survive the crop.
        corner = write_numobs(
            tmp_path / "corner.tif",
            value=8,
            gaps=[(-65.0, -30.01, -64.99, -30.0)],
        )
        bbox = (-65.0, -31.0, -64.0, -30.0)
        grown = masks.emissivity_gap(bbox, 3600, corner, buffer_cells=1)
        assert int(grown.sum()) == 72 * 72

    def test_a_gap_cell_outside_the_tile_still_buffers_in(self, tmp_path):
        # The margin is why the window is padded. Clipping the dilation at the
        # tile edge instead was two cells short on S30W065.
        outside = write_numobs(
            tmp_path / "outside.tif",
            value=8,
            gaps=[(-65.01, -30.51, -65.0, -30.50)],
        )
        bbox = (-65.0, -31.0, -64.0, -30.0)
        assert not masks.emissivity_gap(bbox, 3600, outside, buffer_cells=0).any()
        grown = masks.emissivity_gap(bbox, 3600, outside, buffer_cells=1)
        assert int(grown.sum()) == 36 * 108

    def test_the_thin_tiers_are_kept(self, tmp_path):
        # One and two observations are low confidence, not absence. Dropping
        # them would remove 6.9% of a measured tile against 0.22%.
        thin = write_numobs(tmp_path / "thin.tif", value=1)
        gap = masks.emissivity_gap(tile_bounds("S30W065"), COARSE_PPD, thin)
        assert not gap.any()

    def test_output_mask_keeps_the_gap_apart_from_the_water_rule(
        self, gapped, land_geometry
    ):
        # The gap is a region, not a removal. Only `apply_output_mask`
        # intersects it with a temperature, so `keep` is the water rule alone.
        bbox = tile_bounds("N40W075")
        land = masks.land_mask(bbox, COARSE_PPD, land_geometry)
        expected = masks.emissivity_gap(bbox, COARSE_PPD, gapped)
        keep, gap, _ = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=gapped, land_geometry_uri=land_geometry
        )
        assert np.array_equal(keep, land)
        assert np.array_equal(gap, expected)

    def test_the_counts_describe_the_masks_they_come_with(self, gapped, land_geometry):
        bbox = tile_bounds("S30W065")
        keep, gap, counts = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=gapped, land_geometry_uri=land_geometry
        )
        assert counts["pixels_total"] == keep.size
        assert counts["pixels_kept"] == int(keep.sum())
        assert counts["pixels_emissivity_gap"] == int(gap.sum())
        # This tile is all land, so the two gap counts coincide and the water
        # rule removes nothing.
        assert counts["pixels_water"] == 0
        assert (
            counts["pixels_emissivity_gap_on_land"] == counts["pixels_emissivity_gap"]
        )
        assert counts["pixels_kept"] == counts["pixels_total"]

    def test_the_counts_name_the_rule_that_will_run(self, gapped, land_geometry):
        _, _, counts = masks.output_mask(
            tile_bounds("S30W065"),
            COARSE_PPD,
            numobs_uri=gapped,
            land_geometry_uri=land_geometry,
        )
        assert counts["gap_buffer_cells"] == masks.GAP_BUFFER_CELLS
        assert counts["gap_hot_threshold_c"] == masks.GAP_HOT_THRESHOLD_C

    def test_land_with_no_granule_is_kept_and_counted(self, tmp_path, land_geometry):
        """A gap the artifact cannot demonstrate is not a gap.

        ASTER GED publishes no granule for 813 of the 14,941 land cells, and
        Landsat holds surface temperature over most of that land. Treating an
        absent granule as a gap dropped 34 tiles from a real fleet plan.
        """
        import numpy as np

        import aster_ged

        rows, cols = aster_ged.mosaic_shape(60)
        manifest = aster_ged.build_manifest(
            {},
            lat_limit=60,
            buffer_meters=25_000,
            land_geometry_sha256=__import__("conftest").land_geometry_sha256(),
            cell_count=0,
        )
        nothing_read = aster_ged.write_numobs(
            tmp_path / "unread.tif",
            np.zeros((rows, cols), dtype="uint8"),
            manifest,
            covered=np.zeros((rows, cols), dtype="uint8"),
        )
        bbox = tile_bounds(INLAND_TILE)
        keep, gap, counts = masks.output_mask(
            bbox,
            COARSE_PPD,
            numobs_uri=nothing_read,
            land_geometry_uri=land_geometry,
        )
        assert keep.all()
        assert not gap.any()
        assert counts["pixels_emissivity_gap"] == 0
        assert counts["pixels_land_unread"] == counts["pixels_total"]

    def test_water_and_gap_are_counted_apart_because_they_overlap(
        self, tmp_path, land_geometry
    ):
        # GED has no observation over sea either, so adding the two counts
        # double-counts. A summary that reported the sum would overstate what
        # the mask reaches on every coastal tile.
        everywhere = write_numobs(tmp_path / "all_gap.tif", value=0)
        bbox = tile_bounds("N40W075")
        keep, _, counts = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=everywhere, land_geometry_uri=land_geometry
        )
        assert counts["pixels_emissivity_gap"] == counts["pixels_total"]
        assert counts["pixels_emissivity_gap_on_land"] < counts["pixels_total"]
        assert (
            counts["pixels_water"] + counts["pixels_emissivity_gap"]
            > counts["pixels_total"]
        )
        # A tile of nothing but gap still keeps every land pixel. The region is
        # not the removal.
        assert counts["pixels_kept"] == int(keep.sum()) > 0


class TestApplyOutputMask:
    """Writing the mask into a finished tile."""

    @pytest.fixture
    def tile(self):
        rng = np.random.default_rng(0)
        lst = rng.integers(2, 9000, size=(20, 20), dtype="uint16")
        qa = rng.integers(0, 40, size=(12, 20, 20), dtype="uint8")
        keep = np.zeros((20, 20), dtype=bool)
        keep[5:15, 5:15] = True
        return lst, qa, keep

    def test_it_writes_nodata_outside_the_mask(self, tile):
        lst, qa, keep = tile
        masks.apply_output_mask(lst, qa, keep)
        assert (lst[~keep] == LST_NODATA_DN).all()

    def test_it_zeroes_the_monthly_counts_on_the_same_pixels(self, tile):
        lst, qa, keep = tile
        masks.apply_output_mask(lst, qa, keep)
        assert (qa[:, ~keep] == 0).all()
        assert np.array_equal(lst == LST_NODATA_DN, qa.sum(axis=0) == 0)

    def test_it_leaves_everything_inside_the_mask_alone(self, tile):
        lst, qa, keep = tile
        before_lst = lst.copy()
        before_qa = qa.copy()
        masks.apply_output_mask(lst, qa, keep)
        assert np.array_equal(lst[keep], before_lst[keep])
        assert np.array_equal(qa[:, keep], before_qa[:, keep])

    def test_it_reports_what_it_removed(self, tile):
        lst, qa, keep = tile
        before = int(((lst != LST_NODATA_DN) & ~keep).sum())
        counts = masks.apply_output_mask(lst, qa, keep)
        assert counts["valid_removed_by_mask"] == before
        assert counts["valid_removed_by_mask"] == 400 - 100

    def test_it_is_idempotent(self, tile):
        lst, qa, keep = tile
        masks.apply_output_mask(lst, qa, keep)
        after_lst = lst.copy()
        after_qa = qa.copy()
        second = masks.apply_output_mask(lst, qa, keep)
        assert np.array_equal(lst, after_lst)
        assert np.array_equal(qa, after_qa)
        assert second["valid_removed_by_mask"] == 0

    def test_a_mask_that_keeps_everything_changes_nothing(self, tile):
        lst, qa, _ = tile
        before_lst = lst.copy()
        before_qa = qa.copy()
        counts = masks.apply_output_mask(lst, qa, np.ones((20, 20), dtype=bool))
        assert np.array_equal(lst, before_lst)
        assert np.array_equal(qa, before_qa)
        assert counts["valid_removed_by_mask"] == 0

    def test_it_records_what_the_counts_describe(self, tile):
        lst, qa, keep = tile
        counts = masks.apply_output_mask(lst, qa, keep, scope="shards[0:64]")
        assert counts["scope"] == "shards[0:64]"


class TestTheEmissivityRuleIsAPair:
    """Neither half of the rule removes a pixel on its own.

    The gap region alone removes 701,839 valid pixels of S30W065 and 87% of
    the cells it removes carry nothing wrong. The threshold alone would delete
    ordinary hot ground. `nlebovits/landsat-lst` shipped the geometry alone,
    measured 2,799,286 pixels removed for 2,582 artifacts, and replaced it with
    this pair.
    """

    HOT = masks.gap_hot_dn()

    @pytest.fixture
    def tile(self):
        # Four quadrants of a 2 by 2 tile: hot-in-gap, hot-outside, cool-in-gap
        # and cool-outside, so one call covers the whole truth table.
        lst = np.array([[self.HOT, self.HOT], [4000, 4000]], dtype="uint16")
        qa = np.full((12, 2, 2), 3, dtype="uint8")
        keep = np.ones((2, 2), dtype=bool)
        gap = np.array([[True, False], [True, False]])
        return lst, qa, keep, gap

    def test_only_hot_inside_the_gap_is_removed(self, tile):
        lst, qa, keep, gap = tile
        masks.apply_output_mask(lst, qa, keep, gap)
        assert lst[0, 0] == LST_NODATA_DN
        assert lst[0, 1] == self.HOT
        assert lst[1, 0] == 4000
        assert lst[1, 1] == 4000

    def test_the_threshold_is_inclusive(self, tile):
        lst, qa, keep, gap = tile
        lst[0, 0] = self.HOT
        masks.apply_output_mask(lst, qa, keep, gap)
        assert lst[0, 0] == LST_NODATA_DN

    def test_a_pixel_a_step_below_the_threshold_survives(self, tile):
        lst, qa, keep, gap = tile
        lst[0, 0] = self.HOT - 1
        masks.apply_output_mask(lst, qa, keep, gap)
        assert lst[0, 0] == self.HOT - 1

    def test_the_gap_alone_removes_nothing(self, tile):
        lst, qa, keep, gap = tile
        lst[:] = 4000
        counts = masks.apply_output_mask(lst, qa, keep, gap)
        assert (lst == 4000).all()
        assert counts["valid_removed_by_emissivity"] == 0

    def test_the_threshold_alone_removes_nothing(self, tile):
        lst, qa, keep, _ = tile
        lst[:] = self.HOT
        counts = masks.apply_output_mask(lst, qa, keep, np.zeros((2, 2), dtype=bool))
        assert (lst == self.HOT).all()
        assert counts["valid_removed_by_emissivity"] == 0

    def test_nodata_passes_through(self, tile):
        # Nodata is 0, which is below every legal threshold, so a missing pixel
        # can never be read as a failed retrieval.
        lst, qa, keep, gap = tile
        lst[:] = LST_NODATA_DN
        counts = masks.apply_output_mask(lst, qa, keep, gap)
        assert counts["valid_removed_by_mask"] == 0

    def test_the_count_layer_survives_the_emissivity_rule(self, tile):
        # Zero observations is data. The count is the evidence behind the p95,
        # and it stays true whatever the retrieval did with the observations.
        lst, qa, keep, gap = tile
        masks.apply_output_mask(lst, qa, keep, gap)
        assert (qa == 3).all()

    def test_the_count_layer_does_not_survive_the_water_rule(self, tile):
        # A qa_count above zero beside a nodata temperature would say the pixel
        # had observations and lost them to the reduction. Over sea it had
        # none of this product's subject at all.
        lst, qa, keep, gap = tile
        keep[0, :] = False
        masks.apply_output_mask(lst, qa, keep, gap)
        assert (qa[:, 0, :] == 0).all()
        assert (qa[:, 1, :] == 3).all()

    def test_the_two_rules_are_counted_apart(self, tile):
        lst, qa, keep, gap = tile
        keep[1, :] = False
        counts = masks.apply_output_mask(lst, qa, keep, gap)
        assert counts["valid_removed_by_water"] == 2
        assert counts["valid_removed_by_emissivity"] == 1
        assert counts["valid_removed_by_mask"] == 3

    def test_a_threshold_below_the_floor_is_refused(self):
        with pytest.raises(masks.MaskError, match="below 50.0 C"):
            masks.gap_hot_dn(25.0)

    def test_the_floor_itself_is_allowed(self):
        assert masks.gap_hot_dn(masks.MIN_GAP_HOT_THRESHOLD_C) > 0

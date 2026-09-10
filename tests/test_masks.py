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

Masking that changes a temperature. Every pixel this removes was already nodata
or already outside the product's subject. Over an ASTER gap USGS wrote `ST_B10`
fill and `lst_qa.not_fill` rejected it, so `qa_count` is 0 there before the mask
runs. Over water the composite does hold values, and those are the ones the
mask removes.

`lst_p95` and `qa_count` disagreeing. A `qa_count` above zero beside a nodata
temperature says the pixel had observations and lost them to the reduction.
That is a different fact from a masked pixel, and the `qa_count` band exists to
keep the two apart.
"""

from __future__ import annotations

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
        gap = masks.emissivity_gap(tile_bounds("S30W065"), COARSE_PPD, gapped)
        assert gap[:, :250].all()
        assert not gap[:, 250:].any()

    def test_the_thin_tiers_are_kept(self, tmp_path):
        # One and two observations are low confidence, not absence. Dropping
        # them would remove 6.9% of a measured tile against 0.22%.
        thin = write_numobs(tmp_path / "thin.tif", value=1)
        gap = masks.emissivity_gap(tile_bounds("S30W065"), COARSE_PPD, thin)
        assert not gap.any()

    def test_output_mask_is_land_without_gap(self, gapped, land_geometry):
        bbox = tile_bounds("N40W075")
        land = masks.land_mask(bbox, COARSE_PPD, land_geometry)
        gap = masks.emissivity_gap(bbox, COARSE_PPD, gapped)
        keep, _ = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=gapped, land_geometry_uri=land_geometry
        )
        assert np.array_equal(keep, land & ~gap)

    def test_the_counts_describe_the_mask_they_come_with(self, gapped, land_geometry):
        bbox = tile_bounds("S30W065")
        keep, counts = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=gapped, land_geometry_uri=land_geometry
        )
        assert counts["pixels_total"] == keep.size
        assert counts["pixels_kept"] == int(keep.sum())
        # This tile is all land, so the gap is the only thing removing pixels
        # and the two gap counts coincide.
        assert counts["pixels_water"] == 0
        assert (
            counts["pixels_emissivity_gap_on_land"] == (counts["pixels_emissivity_gap"])
        )
        assert counts["pixels_kept"] + counts["pixels_emissivity_gap"] == keep.size

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
        keep, counts = masks.output_mask(
            bbox,
            COARSE_PPD,
            numobs_uri=nothing_read,
            land_geometry_uri=land_geometry,
        )
        assert keep.all()
        assert counts["pixels_emissivity_gap"] == 0
        assert counts["pixels_land_unread"] == counts["pixels_total"]

    def test_water_and_gap_are_counted_apart_because_they_overlap(
        self, tmp_path, land_geometry
    ):
        # GED has no observation over sea either, so adding the two counts
        # double-counts. A summary that reported the sum would overstate what
        # the mask removed on every coastal tile.
        everywhere = write_numobs(tmp_path / "all_gap.tif", value=0)
        bbox = tile_bounds("N40W075")
        keep, counts = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=everywhere, land_geometry_uri=land_geometry
        )
        assert counts["pixels_kept"] == 0
        assert counts["pixels_emissivity_gap"] == counts["pixels_total"]
        assert counts["pixels_emissivity_gap_on_land"] < counts["pixels_total"]
        assert (
            counts["pixels_water"] + counts["pixels_emissivity_gap"]
            > counts["pixels_total"]
        )


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

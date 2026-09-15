"""Which pixels the product describes, and the rule that decides.

`lst.land_tiles` opens with the contract this module completes: one geometry
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
gap cells hold nothing bad at all. An earlier build paired the region with a
70 C threshold instead; five tiles then showed the halves do not coincide, and
`TestTheEmissivityRegionRemovesNothing` holds that retirement in place.

`lst_p95` and `qa_count` disagreeing. A `qa_count` above zero beside a nodata
temperature says the pixel had observations and lost them to the reduction.
That is what the evidence rule in `lst_qa.supported_output` wants to say, so it
leaves the count alone. Over water the pixel was never this product's subject,
so both bands go.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from lst import masks
from lst.land_tiles import read_land_tiles, tile_bounds
from lst.lst_qa import LST_NODATA_DN

ROOT = Path(__file__).resolve().parent.parent

from conftest import (  # noqa: E402
    FULL_LAND_GEOMETRY,
    needs_full_land_geometry,
    needs_land_geometry,
    needs_strict_land_geometry,
    write_numobs,
)

LAND_TILES = ROOT / "artifacts" / "land_tiles.parquet"

#: The committed manifest of the NumObs mosaic. The mosaic itself is 45.5 MB
#: and gitignored; this names the granules behind it and the geometry it was
#: built from.
GED_MANIFEST = ROOT / "artifacts" / "aster_numobs_manifest.json"

#: The buffered geometry's digest, committed so CI can check the tie without
#: the 16 MB file. `lst-land-tiles --write-geometry` writes both.
RECORDED_GEOMETRY_SHA256 = ROOT / "artifacts" / "land_buffered_sha256.txt"

#: The unbuffered geometry's digest, committed for the same reason. A published
#: item names it in `processing:lineage`, so a reader who wants to know which
#: land a coverage count divided by has these bytes to compare against.
RECORDED_STRICT_SHA256 = ROOT / "artifacts" / "land_strict_sha256.txt"

#: The full 7.3 MB unbuffered geometry, gitignored beside the buffered one.
FULL_STRICT_GEOMETRY = ROOT / "artifacts" / "land_strict.gpkg"

#: The tiles the committed geometry slices cover. `tests/make_land_slice.py`
#: cuts both the buffered and the unbuffered geometry to these.
SLICE_TILES = (
    "N05E010",
    "N40W075",
    "S15E175",
    "S30W065",
    "S40W065",
    "S35W055",
)

#: Golfo San Matias and the Patagonian coast, where the 25 km processing buffer
#: is 38% of the mask. The tile the land counts are pinned on.
COASTAL_TILE = "S40W065"

#: Atlantic off Uruguay. The buffer reaches it and land does not, so the
#: processing mask is non-empty and land is zero.
ALL_SEA_TILE = "S35W055"

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

    def test_the_strict_geometry_has_a_recorded_digest_of_its_own(self):
        """A published item names it, so the bytes need a committed digest.

        `processing:lineage` states which land a coverage count divided by. A
        reader comparing that digest against nothing cannot check the claim, and
        the geometry itself is gitignored.
        """
        recorded = RECORDED_STRICT_SHA256.read_text().strip()
        assert len(recorded) == 64
        assert recorded != RECORDED_GEOMETRY_SHA256.read_text().strip()

    @pytest.mark.skipif(
        not FULL_STRICT_GEOMETRY.exists(),
        reason="run land_tiles.py --write-strict-geometry",
    )
    def test_the_recorded_strict_digest_is_the_geometry_on_disk(self):
        recorded = RECORDED_STRICT_SHA256.read_text().strip()
        assert masks.geometry_checksum(FULL_STRICT_GEOMETRY) == recorded

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


@needs_strict_land_geometry
@pytest.mark.timeout(300)
class TestLandIsNotTheProcessingMask:
    """The denominator of every published share of land.

    A run masks on Natural Earth land grown by 25 km, so that a coastal scene is
    not cut at the waterline. That mask reaches open sea. Calling its pixel count
    `lst:land_pixels` overstated the land of every coastal tile, and the fraction
    beside it understated coverage by the same factor.

    The counts here are pinned rather than related, because the defect this
    guards against passes every relational test. A rebuild that quietly returns
    the processing mask still sums, still contains, and still divides. Only a
    number says it changed.
    """

    def geometries(self, land_geometry, strict_land_geometry):
        return {
            "land_geometry_uri": land_geometry,
            "strict_land_geometry_uri": strict_land_geometry,
        }

    def test_the_land_count_is_pinned(self, land_geometry, strict_land_geometry):
        """MEASURED 2026-09-15 at 3600 pixels per degree on `S40W065`.

        Natural Earth 10m land, the placeholder record dropped, no buffer, a
        pixel counted where its centre falls inside a polygon. The same method
        at 25 km of Mercator buffer gives 73,254,945, which is what the item
        used to publish as land.

        FINDINGS.md and the pull request record one more measurement: an earlier
        session reported 47,741,196 for this tile, and that figure does not
        reproduce from Natural Earth 10m land on this grid. `all_touched=True`
        moves the count by 28,378, and `ne_10m_minor_islands` and
        `ne_10m_lakes` contribute no feature inside this box.
        """
        _, _, counts = masks.land_split(
            tile_bounds(COASTAL_TILE),
            3600,
            **self.geometries(land_geometry, strict_land_geometry),
        )
        assert counts["pixels_strict_land"] == 45_407_126
        assert counts["pixels_coastal_buffer"] == 27_847_819
        assert counts["pixels_processing_mask"] == 73_254_945

    def test_the_two_parts_sum_to_the_mask(self, land_geometry, strict_land_geometry):
        _, _, counts = masks.land_split(
            tile_bounds(COASTAL_TILE),
            3600,
            **self.geometries(land_geometry, strict_land_geometry),
        )
        assert (
            counts["pixels_strict_land"] + counts["pixels_coastal_buffer"]
            == counts["pixels_processing_mask"]
        )

    def test_land_lies_inside_the_processing_mask(
        self, land_geometry, strict_land_geometry
    ):
        strict, processing, _ = masks.land_split(
            tile_bounds(COASTAL_TILE),
            360,
            **self.geometries(land_geometry, strict_land_geometry),
        )
        assert strict.any()
        assert not strict.all()
        assert not (strict & ~processing).any()

    def test_an_inland_tile_splits_into_land_and_nothing(
        self, land_geometry, strict_land_geometry
    ):
        """Interior Argentina, where the buffer reaches no sea.

        The two counts coincide there, which is what makes the coastal numbers
        a measurement of the buffer rather than of the method.
        """
        _, _, counts = masks.land_split(
            tile_bounds(INLAND_TILE),
            COARSE_PPD,
            **self.geometries(land_geometry, strict_land_geometry),
        )
        assert counts["pixels_coastal_buffer"] == 0
        assert counts["pixels_strict_land"] == counts["pixels_processing_mask"]

    def test_a_tile_can_be_all_buffer_and_no_land(
        self, land_geometry, strict_land_geometry
    ):
        """`S35W055` is Atlantic that the 25 km buffer reaches and land does not.

        MEASURED 2026-09-15 at 3600 pixels per degree: 588,696 pixels of
        processing mask, 0 pixels of land. It is published, and it reported all
        588,696 as `lst:land_pixels`. The split has to survive the case, because
        a denominator of zero is what the first recount of the real catalog hit.
        """
        _, _, counts = masks.land_split(
            tile_bounds(ALL_SEA_TILE),
            3600,
            **self.geometries(land_geometry, strict_land_geometry),
        )
        assert counts["pixels_strict_land"] == 0
        assert counts["pixels_processing_mask"] == 588_696
        assert counts["pixels_coastal_buffer"] == 588_696

    def test_the_caller_can_hand_over_the_mask_it_holds(
        self, land_geometry, strict_land_geometry
    ):
        """`processing` saves rasterising the buffered geometry twice.

        A run already holds it from `output_mask`. Passing it has to give the
        same answer as letting `land_split` read the file, or the saving is a
        second opinion.
        """
        bbox = tile_bounds(COASTAL_TILE)
        processing = masks.land_mask(bbox, COARSE_PPD, land_geometry)
        _, handed, given = masks.land_split(
            bbox,
            COARSE_PPD,
            strict_land_geometry_uri=strict_land_geometry,
            processing=processing,
        )
        _, read, alone = masks.land_split(
            bbox, COARSE_PPD, **self.geometries(land_geometry, strict_land_geometry)
        )
        assert given == alone
        assert np.array_equal(handed, read)

    def test_the_gap_gets_its_own_numerator(
        self, land_geometry, strict_land_geometry, numobs_artifact
    ):
        """`ged_gap_fraction` divides by land, so its numerator must too.

        The gap over the processing mask counts sea, where ASTER has no clear-sky
        observation either. Dividing that by land would put the two halves of the
        fraction on different footprints.
        """
        bbox = tile_bounds(COASTAL_TILE)
        # A box over the coast, so the gap covers both land and sea inside the
        # processing mask. A gap wholly on land would make the two numerators
        # equal and prove nothing.
        gaps = ((-65.0, -43.0, -62.0, -41.0),)
        numobs = write_numobs(numobs_artifact.parent / "gap.tif", gaps=gaps)
        keep, gap, counts = masks.output_mask(
            bbox, COARSE_PPD, numobs_uri=numobs, land_geometry_uri=land_geometry
        )
        _, _, split = masks.land_split(
            bbox,
            COARSE_PPD,
            strict_land_geometry_uri=strict_land_geometry,
            processing=keep,
            gap=gap,
        )
        on_strict = split["pixels_emissivity_gap_on_strict_land"]
        assert on_strict <= counts["pixels_emissivity_gap_on_land"]

    def test_a_geometry_pair_that_cannot_both_be_true_is_refused(self, tmp_path):
        """Land outside the mask that is supposed to contain it.

        A buffer contains what it buffers. Losing that means the two files hold
        different Natural Earth releases, and every count built from them would
        compare two worlds. The failure is silent without this check: both masks
        rasterise, both counts look plausible, and their difference goes
        negative.
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box

        bbox = (0.0, 0.0, 1.0, 1.0)
        small = tmp_path / "small.gpkg"
        large = tmp_path / "large.gpkg"
        gpd.GeoDataFrame(geometry=[box(0.1, 0.1, 0.4, 0.4)], crs="EPSG:4326").to_file(
            small, driver="GPKG"
        )
        gpd.GeoDataFrame(geometry=[box(0.1, 0.1, 0.9, 0.9)], crs="EPSG:4326").to_file(
            large, driver="GPKG"
        )
        with pytest.raises(masks.MaskError, match="outside the processing mask"):
            masks.land_split(
                bbox,
                COARSE_PPD,
                land_geometry_uri=small,
                strict_land_geometry_uri=large,
            )


@needs_strict_land_geometry
class TestCountingValuesInsideAMask:
    """What `count_valid_within` answers, and what it refuses to answer."""

    def raster(self, path, values, nodata=0):
        import rasterio

        height, width = values.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=values.dtype,
            nodata=nodata,
        ) as dst:
            dst.write(values, 1)
        return path

    def test_each_region_is_counted_in_one_pass(self, tmp_path):
        values = np.zeros((8, 8), dtype="uint16")
        values[:4] = 300  # the top half carries a value
        path = self.raster(tmp_path / "v.tif", values)
        left = np.zeros((8, 8), dtype=bool)
        left[:, :4] = True
        counts = masks.count_valid_within(path, 0, left=left, right=~left)
        assert counts == {"total": 32, "left": 16, "right": 16}

    def test_the_regions_of_a_masked_tile_sum_to_the_total(self, tmp_path):
        values = np.full((8, 8), 300, dtype="uint16")
        values[6:] = 0  # nodata, outside every region
        path = self.raster(tmp_path / "v.tif", values)
        land = np.zeros((8, 8), dtype=bool)
        land[:3] = True
        coast = np.zeros((8, 8), dtype=bool)
        coast[3:6] = True
        counts = masks.count_valid_within(path, 0, land=land, coast=coast)
        assert counts["land"] + counts["coast"] == counts["total"] == 48

    def test_no_nodata_counts_every_pixel(self, tmp_path):
        values = np.zeros((4, 4), dtype="uint8")
        path = self.raster(tmp_path / "v.tif", values, nodata=None)
        whole = np.ones((4, 4), dtype=bool)
        assert masks.count_valid_within(path, None, all=whole) == {
            "total": 16,
            "all": 16,
        }

    def test_a_mask_on_the_wrong_grid_is_refused(self, tmp_path):
        """The failure this catches returns a number rather than an error.

        A recount derives the grid from the raster. If it ever derived it from
        somewhere else, the mask would still rasterise and still count, and the
        answer would describe different ground.
        """
        path = self.raster(tmp_path / "v.tif", np.zeros((8, 8), dtype="uint16"))
        with pytest.raises(masks.MaskError, match="different grid"):
            masks.count_valid_within(path, 0, land=np.ones((4, 4), dtype=bool))

    def test_total_cannot_name_a_region(self, tmp_path):
        path = self.raster(tmp_path / "v.tif", np.zeros((4, 4), dtype="uint16"))
        with pytest.raises(ValueError, match="cannot name a region"):
            masks.count_valid_within(path, 0, total=np.ones((4, 4), dtype=bool))


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

    def test_the_counts_name_the_region_they_report(self, gapped, land_geometry):
        _, _, counts = masks.output_mask(
            tile_bounds("S30W065"),
            COARSE_PPD,
            numobs_uri=gapped,
            land_geometry_uri=land_geometry,
        )
        assert counts["gap_buffer_cells"] == masks.GAP_BUFFER_CELLS
        # The withdrawn pair rule's threshold is gone from the counts, because
        # nothing applies it. `lst_qa.LST_OUTPUT_MAX_C` replaced it.
        assert "gap_hot_threshold_c" not in counts

    def test_land_with_no_granule_is_kept_and_counted(self, tmp_path, land_geometry):
        """A gap the artifact cannot demonstrate is not a gap.

        ASTER GED publishes no granule for 813 of the 14,941 land cells, and
        Landsat holds surface temperature over most of that land. Treating an
        absent granule as a gap dropped 34 tiles from a real fleet plan.
        """
        import numpy as np

        from lst import aster_ged

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

    def test_two_masks_in_sequence_remove_their_union(self, tile):
        """The two water rules reach this function as one `keep`.

        `composite.finalize_block` takes the union before calling, so this is
        the claim that makes the union safe to build: masking twice with two
        different planes leaves the same raster as masking once with both.
        Distinct from `test_it_is_idempotent`, which repeats one plane.
        """
        lst, qa, keep = tile
        observed = np.zeros(keep.shape, dtype=bool)
        observed[:, :5] = True  # a river the geometry does not know about

        once_lst, once_qa = lst.copy(), qa.copy()
        masks.apply_output_mask(once_lst, once_qa, keep & ~observed)

        masks.apply_output_mask(lst, qa, keep)
        masks.apply_output_mask(lst, qa, ~observed)

        assert np.array_equal(lst, once_lst)
        assert np.array_equal(qa, once_qa)
        assert (lst[~keep] == LST_NODATA_DN).all()
        assert (lst[observed] == LST_NODATA_DN).all()

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


class TestTheEmissivityRegionRemovesNothing:
    """The gap region is reported and never masked.

    An earlier build paired it with a 70 C threshold and removed the pixels
    where both held. MEASURED across five tiles, the halves do not coincide: on
    N30E075 all 207 pixels at or above 80 C fall outside the region and its
    one-cell buffer, so the pair reached none of them. The ceiling that
    replaced it is `lst_qa.LST_OUTPUT_MAX_C`, applied in `reduce_block` to every
    pixel wherever it sits. These tests hold the retirement in place.
    """

    #: The DN the withdrawn rule screened at, 70 C on the output encoding.
    HOT = 12_000

    @pytest.fixture
    def tile(self):
        # Four quadrants of a 2 by 2 tile: hot-in-gap, hot-outside, cool-in-gap
        # and cool-outside, the truth table the pair rule used to split.
        lst = np.array([[self.HOT, self.HOT], [4000, 4000]], dtype="uint16")
        qa = np.full((12, 2, 2), 3, dtype="uint8")
        keep = np.ones((2, 2), dtype=bool)
        return lst, qa, keep

    def test_the_mask_takes_no_gap_plane(self):
        import inspect

        names = inspect.signature(masks.apply_output_mask).parameters
        assert "gap" not in names
        assert "hot_dn" not in names

    def test_a_hot_pixel_over_land_survives(self, tile):
        lst, qa, keep = tile
        masks.apply_output_mask(lst, qa, keep)
        assert (lst == np.array([[self.HOT, self.HOT], [4000, 4000]])).all()

    def test_the_module_no_longer_owns_a_hot_threshold(self):
        assert not hasattr(masks, "GAP_HOT_THRESHOLD_C")
        assert not hasattr(masks, "MIN_GAP_HOT_THRESHOLD_C")
        assert not hasattr(masks, "gap_hot_dn")

    def test_the_water_rule_still_zeroes_the_count_layer(self, tile):
        # A qa_count above zero beside a nodata temperature would say the pixel
        # had observations and lost them to the reduction. Over sea it had
        # none of this product's subject at all.
        lst, qa, keep = tile
        keep[0, :] = False
        counts = masks.apply_output_mask(lst, qa, keep)
        assert (qa[:, 0, :] == 0).all()
        assert (qa[:, 1, :] == 3).all()
        assert counts["valid_removed_by_water"] == 2
        assert counts["valid_removed_by_mask"] == 2

    def test_nodata_passes_through(self, tile):
        lst, qa, keep = tile
        lst[:] = LST_NODATA_DN
        counts = masks.apply_output_mask(lst, qa, keep)
        assert counts["valid_removed_by_mask"] == 0

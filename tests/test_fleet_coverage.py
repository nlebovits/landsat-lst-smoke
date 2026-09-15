"""The coverage screen: what each land tile holds, before the fleet spends.

`tests/test_fleet_plan.py` covers what the plan launches. This covers the two
numbers a scheduler sorts on, and the one property both of them must keep: they
remove no tile. The emissivity rule takes a pixel for reading 70 C or hotter
inside the gap region, not for being in it, so a tile of nothing but gap cells
still publishes every ordinary temperature it holds.

Neither number predicts swath coverage. A swath is counted from valid
observations over the scene set the window selects, and the gap region is a
property of an emissivity mosaic built from other granules for another purpose.
A tile with no gap at all can still hold a WRS path that reaches no swath cell,
which is what `tile_prep.paths_without_a_swath` reports at run time.

Every geometry here is real and committed. `artifacts/land_strict_slice.gpkg`
is the unbuffered geometry clipped to `conftest.LAND_SLICE_TILES`, and
`masks.land_mask` rasterises inside one tile's bbox, so a clipped geometry
gives the same answer there as the whole world would.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fleet_plan  # noqa: E402
import masks  # noqa: E402
from land_tiles import tile_bounds  # noqa: E402
from tests.conftest import write_numobs  # noqa: E402

#: In the geometry slice, and mostly land.
LAND_TILE = "N05E010"
#: In the geometry slice. Golfo San Matias and the Patagonian coast, where the
#: 25 km processing buffer is 38% of the buffered mask, so strict and buffered
#: land differ by a wide margin here.
COAST_TILE = "S40W065"
#: In the geometry slice, and holds no strict land at all. Open Atlantic inside
#: the 25 km buffer. `masks.coverage` names the same tile for the same reason:
#: the published `S35W055` is 588,696 pixels of processing mask and 0 pixels of
#: land, and it once reported all 588,696 as land.
SEA_TILE = "S35W055"


@pytest.fixture
def screen(strict_land_geometry, numobs_artifact):
    def build(tile_id, numobs=None, ppd=fleet_plan.PLANNING_PIXELS_PER_DEGREE):
        return fleet_plan.coverage_row(
            tile_id,
            numobs_uri=numobs or numobs_artifact,
            strict_land_geometry_uri=strict_land_geometry,
            pixels_per_degree=ppd,
        )

    return build


class TestTheScreenMeasuresOneTile:
    def test_it_rasterises_at_the_planning_resolution(self, screen):
        """A 5 degree tile is 500 by 500 here and 18,000 by 18,000 at the
        compositing grid, so the screen costs 1/1296 of the pixels."""
        row = screen(LAND_TILE)
        assert fleet_plan.PLANNING_PIXELS_PER_DEGREE == 100
        assert row["planning_pixels"] == 500 * 500
        assert row["planning_pixels_per_degree"] == 100

    def test_it_names_the_tile_and_its_bbox(self, screen):
        row = screen(LAND_TILE)
        assert row["tile_id"] == LAND_TILE
        assert row["bbox"] == list(tile_bounds(LAND_TILE))

    def test_the_land_count_is_bounded_by_the_raster(self, screen):
        row = screen(LAND_TILE)
        assert 0 < row["strict_land_pixels"] <= row["planning_pixels"]

    def test_the_gap_count_is_bounded_by_the_land(self, screen):
        """The gap field counts gap over land, so it cannot exceed the land."""
        row = screen(LAND_TILE)
        assert 0 <= row["ged_gap_pixels_on_land"] <= row["strict_land_pixels"]

    @pytest.mark.parametrize("field", ["strict_land_share", "ged_gap_share"])
    def test_every_share_is_a_fraction(self, screen, field):
        assert 0.0 <= screen(LAND_TILE)[field] <= 1.0

    def test_each_share_is_its_count_over_its_own_denominator(self, screen):
        row = screen(LAND_TILE)
        assert row["strict_land_share"] == pytest.approx(
            row["strict_land_pixels"] / row["planning_pixels"]
        )
        assert row["ged_gap_share"] == pytest.approx(
            row["ged_gap_pixels_on_land"] / row["strict_land_pixels"]
        )

    def test_the_counts_are_ints_and_the_shares_floats(self, screen):
        row = screen(LAND_TILE)
        for field in ("strict_land_pixels", "ged_gap_pixels_on_land"):
            assert isinstance(row[field], int), field
        for field in ("strict_land_share", "ged_gap_share"):
            assert isinstance(row[field], float), field

    def test_it_counts_the_strict_geometry_and_not_the_buffered_one(
        self, screen, land_geometry, numobs_artifact
    ):
        """The buffer reaches 25 km out to sea, so it would rank an island
        chain like a continent. On this tile it is 38% of the buffered mask."""
        strict = screen(COAST_TILE)["strict_land_pixels"]
        buffered = masks.land_mask(
            tile_bounds(COAST_TILE),
            fleet_plan.PLANNING_PIXELS_PER_DEGREE,
            land_geometry,
        )
        assert strict < int(buffered.sum())

    def test_a_row_is_json_ready(self, screen):
        assert json.loads(json.dumps(screen(LAND_TILE)))


class TestTheGapShareIsMeasuredOverLand:
    """Over land is the only denominator that ranks tiles. GED has no
    observation over sea either, so a share of the whole tile would mostly
    measure how much sea the tile holds.
    """

    def test_a_mosaic_with_no_gap_reports_none(self, screen):
        assert screen(LAND_TILE)["ged_gap_share"] == 0.0

    def test_a_gap_over_the_whole_tile_reports_all_of_the_land(self, screen, tmp_path):
        numobs = write_numobs(tmp_path / "gap.tif", gaps=[tile_bounds(LAND_TILE)])
        row = screen(LAND_TILE, numobs=numobs)
        assert row["ged_gap_share"] == 1.0
        assert row["ged_gap_pixels_on_land"] == row["strict_land_pixels"]

    def test_a_tile_with_no_strict_land_carries_no_share_at_all(self, screen):
        """0/0 is not zero. `masks.coverage` settled this for the published
        item on the same geometry, and both fractions are absent there rather
        than invented. A scheduler that read 0.0 here would sort a cell of open
        sea beside a continent with no gap."""
        row = screen(SEA_TILE)
        assert row["strict_land_pixels"] == 0
        assert "ged_gap_share" not in row

    def test_the_counts_are_still_there_on_a_tile_with_no_land(self, screen):
        """The share goes, the evidence stays. `masks.coverage` keeps its
        counts on the same tile for the same reason."""
        row = screen(SEA_TILE)
        assert row["ged_gap_pixels_on_land"] == 0
        assert row["strict_land_share"] == 0.0
        assert row["planning_pixels"] == 500 * 500

    def test_a_gap_over_a_landless_tile_still_reports_no_share(self, screen, tmp_path):
        """The absence is about the denominator, not about the gap. A mosaic
        that is nothing but gap here still divides by no land."""
        numobs = write_numobs(tmp_path / "all.tif", gaps=[tile_bounds(SEA_TILE)])
        row = screen(SEA_TILE, numobs=numobs)
        assert "ged_gap_share" not in row
        assert row["ged_gap_pixels_on_land"] == 0

    def test_a_row_with_no_share_still_serialises(self, screen):
        assert "ged_gap_share" not in json.loads(json.dumps(screen(SEA_TILE)))

    def test_a_gap_over_the_sea_alone_does_not_move_the_share(self, screen, tmp_path):
        """The reason the denominator is land. A sea-only gap is not a fact
        about the temperatures this tile will publish."""
        west, south, east, north = tile_bounds(COAST_TILE)
        sea = (west, south, west + 1.0, south + 1.0)
        numobs = write_numobs(tmp_path / "sea.tif", gaps=[sea])
        assert screen(COAST_TILE, numobs=numobs)["ged_gap_share"] == pytest.approx(
            screen(COAST_TILE)["ged_gap_share"]
        )


class TestTheScreenCoversEveryLandTile:
    """A tile with no scene in this window is still a tile the schedule has to
    account for. A row that appears only when a tile is runnable would make the
    artifact's length depend on the window.
    """

    def plan(self, runnable=("A",), empty=(), no_thermal=()):
        return {
            "tiles": [
                {"tile_id": t, "scenes": 10, "thermal_scenes": 8} for t in runnable
            ],
            "tiles_without_scenes": list(empty),
            "tiles_without_thermal": list(no_thermal),
        }

    def rows(self, plan, strict, numobs):
        return fleet_plan.coverage_rows(
            plan,
            numobs_uri=numobs,
            strict_land_geometry_uri=strict,
            say=lambda *a, **k: None,
        )

    def test_every_tile_gets_a_row(self, strict_land_geometry, numobs_artifact):
        plan = self.plan(
            runnable=[LAND_TILE], empty=[COAST_TILE], no_thermal=["S30W065"]
        )
        rows = self.rows(plan, strict_land_geometry, numobs_artifact)
        assert len(rows) == 3
        assert {r["tile_id"] for r in rows} == {LAND_TILE, COAST_TILE, "S30W065"}

    def test_a_tile_the_fleet_skips_is_marked_rather_than_dropped(
        self, strict_land_geometry, numobs_artifact
    ):
        plan = self.plan(runnable=[LAND_TILE], empty=[COAST_TILE])
        rows = {
            r["tile_id"]: r
            for r in self.rows(plan, strict_land_geometry, numobs_artifact)
        }
        assert rows[LAND_TILE]["runnable"] is True
        assert rows[LAND_TILE]["scenes"] == 10
        assert rows[COAST_TILE]["runnable"] is False
        assert rows[COAST_TILE]["scenes"] is None
        assert rows[COAST_TILE]["thermal_scenes"] is None

    def test_a_skipped_tile_still_carries_its_coverage(
        self, strict_land_geometry, numobs_artifact
    ):
        """The whole point. A tile out of this window may be in the next one,
        and the scheduler wants its land count either way."""
        plan = self.plan(runnable=[], empty=[COAST_TILE])
        row = self.rows(plan, strict_land_geometry, numobs_artifact)[0]
        assert row["strict_land_pixels"] > 0

    def test_no_field_removes_a_tile(self, strict_land_geometry, tmp_path):
        """Scheduling information, not an exclusion gate. A tile whose every
        land pixel sits in the gap region still gets a row."""
        numobs = write_numobs(tmp_path / "all-gap.tif", gaps=[tile_bounds(LAND_TILE)])
        plan = self.plan(runnable=[LAND_TILE])
        rows = self.rows(plan, strict_land_geometry, numobs)
        assert len(rows) == 1
        assert rows[0]["ged_gap_share"] == 1.0
        assert rows[0]["runnable"] is True


class TestTheOutputIsDeterministic:
    """Two runs over the same three artifacts write the same bytes."""

    def plan(self):
        return {
            "tiles": [
                {"tile_id": t, "scenes": 10, "thermal_scenes": 8}
                for t in (COAST_TILE, LAND_TILE)
            ],
            "tiles_without_scenes": [],
            "tiles_without_thermal": [],
        }

    def rows(self, strict, numobs):
        return fleet_plan.coverage_rows(
            self.plan(),
            numobs_uri=numobs,
            strict_land_geometry_uri=strict,
            say=lambda *a, **k: None,
        )

    def test_the_rows_are_in_tile_order(self, strict_land_geometry, numobs_artifact):
        """The plan lists them the other way round."""
        rows = self.rows(strict_land_geometry, numobs_artifact)
        assert [r["tile_id"] for r in rows] == sorted(r["tile_id"] for r in rows)

    def test_two_runs_write_the_same_bytes(
        self, strict_land_geometry, numobs_artifact, tmp_path
    ):
        first = fleet_plan.write_coverage_rows(
            tmp_path / "a.jsonl", self.rows(strict_land_geometry, numobs_artifact)
        )
        second = fleet_plan.write_coverage_rows(
            tmp_path / "b.jsonl", self.rows(strict_land_geometry, numobs_artifact)
        )
        assert first.read_bytes() == second.read_bytes()

    def test_each_line_is_one_tile(
        self, strict_land_geometry, numobs_artifact, tmp_path
    ):
        rows = self.rows(strict_land_geometry, numobs_artifact)
        path = fleet_plan.write_coverage_rows(tmp_path / "c.jsonl", rows)
        lines = path.read_text().splitlines()
        assert len(lines) == len(rows)
        assert [json.loads(line)["tile_id"] for line in lines] == [
            r["tile_id"] for r in rows
        ]

    def test_the_file_ends_in_a_newline(
        self, strict_land_geometry, numobs_artifact, tmp_path
    ):
        rows = self.rows(strict_land_geometry, numobs_artifact)
        path = fleet_plan.write_coverage_rows(tmp_path / "d.jsonl", rows)
        assert path.read_text().endswith("\n")


class TestTheReportStatesWhatItDoesNotDo:
    def test_it_says_the_heavy_tiles_are_not_excluded(self):
        rows = [
            {"tile_id": "A", "strict_land_pixels": 100, "ged_gap_share": 0.9},
            {"tile_id": "B", "strict_land_pixels": 900, "ged_gap_share": 0.01},
        ]
        lines: list[str] = []
        fleet_plan.report_coverage(rows, say=lines.append)
        text = " ".join(lines)
        assert "not excluded" in text
        assert "1 tiles above 40%" in text

    def test_an_empty_screen_reports_nothing_rather_than_dividing_by_zero(self):
        lines: list[str] = []
        fleet_plan.report_coverage([], say=lines.append)
        assert lines == []

    def test_a_screen_of_nothing_but_landless_tiles_says_so(self):
        """No row carries a share, so there is no distribution to report."""
        rows = [{"tile_id": "A", "strict_land_pixels": 0}]
        lines: list[str] = []
        fleet_plan.report_coverage(rows, say=lines.append)
        assert "no gap share is defined" in " ".join(lines)

    def test_it_names_the_tiles_with_no_land_at_this_resolution(self):
        """MEASURED on the real plan: 36 of 895. They are in the tile list
        because the 25 km buffer reaches them, and a 100 pixel-per-degree cell
        is about 1.1 km, so an island under that size leaves no pixel."""
        rows = [
            {"tile_id": "A", "strict_land_pixels": 0},
            {"tile_id": "B", "strict_land_pixels": 500, "ged_gap_share": 0.0},
        ]
        lines: list[str] = []
        fleet_plan.report_coverage(rows, say=lines.append)
        assert "1 tiles hold no strict-land pixel" in " ".join(lines)

    def test_a_landless_tile_is_not_counted_as_a_tile_with_no_gap(self):
        """Counting it as zero would report a cell of open sea as a tile under
        5%, which is the reading the absent field exists to prevent."""
        rows = [
            {"tile_id": "A", "strict_land_pixels": 0},
            {"tile_id": "B", "strict_land_pixels": 500, "ged_gap_share": 0.9},
        ]
        lines: list[str] = []
        fleet_plan.report_coverage(rows, say=lines.append)
        text = " ".join(lines)
        assert "on the 1 tiles that hold land" in text
        assert "0% under 5%" in text

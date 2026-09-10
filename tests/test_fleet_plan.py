"""What the fleet launches, and what it must not.

One machine per tile, so every tile the plan keeps costs an instance. 126 of
the 895 land tiles would spend that instance staging scenes and writing an
all-nodata composite, because every scene they hold is an `OLI_TIRS_L2SR`
product and that product carries no `ST_B10`.

The two exclusions are different facts and the plan names them separately. A
tile with no row group has no scene in this window, which moves if the window
does. A tile with no thermal scene has nothing the pipeline can composite at
all, whatever the window.

Everything here reads `artifacts/inventory_slice.parquet`, which is committed
and holds `S15E175`: a real tile of 60 real `OLI_TIRS_L2SR` rows and no thermal
band. A synthetic row could assert the filter works. Only a real tile proves
the filter fires on what the archive actually returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fleet_plan  # noqa: E402
from tile_inventory import InventoryError, thermal_rows_for_tile  # noqa: E402

LAND_TILES = ROOT / "artifacts" / "land_tiles.parquet"

#: In the committed slice, and holds only L2SR products.
BARREN_TILE = "S15E175"
#: In the committed slice, and holds thermal scenes.
LIVE_TILES = ("N05E010", "N40W075", "S30W065")


@pytest.fixture
def plan(slice_artifact):
    return fleet_plan.build_plan(LAND_TILES, slice_artifact)


class TestThermalRowsForTile:
    """The count `fleet_plan` filters on, from Parquet statistics."""

    def test_a_tile_of_l2sr_alone_answers_zero(self, slice_artifact):
        import pyarrow.parquet as pq

        assert thermal_rows_for_tile(pq.ParquetFile(slice_artifact), BARREN_TILE) == 0

    @pytest.mark.parametrize("tile", LIVE_TILES)
    def test_a_tile_with_thermal_scenes_counts_them(self, slice_artifact, tile):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(slice_artifact)
        assert thermal_rows_for_tile(pf, tile) == 60

    def test_a_tile_that_is_absent_answers_zero(self, slice_artifact):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(slice_artifact)
        assert thermal_rows_for_tile(pf, "N55W005") == 0

    def test_it_agrees_with_reading_the_column(self, slice_artifact):
        # The fast path reads null-count statistics and no column data. If a
        # writer ever stops closing a row group at every tile change, the fast
        # path silently counts a neighbour's scenes, so the two paths are
        # compared against each other here.
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(slice_artifact)
        table = pf.read(columns=["tile_id", "thermal_href"])
        tiles = table.column("tile_id").to_pylist()
        hrefs = table.column("thermal_href").to_pylist()
        for tile in (*LIVE_TILES, BARREN_TILE):
            counted = sum(
                1
                for name, href in zip(tiles, hrefs, strict=True)
                if name == tile and href is not None
            )
            assert thermal_rows_for_tile(pf, tile) == counted, tile


class TestTheBarrenTilesComeOut:
    def test_a_tile_with_no_thermal_scene_is_not_launched(self, plan):
        assert BARREN_TILE in plan["tiles_without_thermal"]
        assert BARREN_TILE not in [t["tile_id"] for t in plan["tiles"]]

    def test_the_other_tiles_stay(self, plan):
        assert [t["tile_id"] for t in plan["tiles"]] == list(LIVE_TILES)
        assert plan["tile_count"] == len(LIVE_TILES)

    def test_a_barren_tile_is_named_rather_than_dropped_silently(self, plan):
        # An operator comparing 769 launched against 895 land tiles needs the
        # difference itemised, or the gap reads as a bug in the tile list.
        assert plan["tiles_without_thermal"] == [BARREN_TILE]
        assert plan["land_tile_count"] == 895

    def test_the_two_exclusions_stay_separate(self, plan):
        # No scene in the window, and no thermal band, are different facts with
        # different fixes. Merging them would tell an operator to widen a
        # window that would not help.
        assert BARREN_TILE not in plan["tiles_without_scenes"]
        assert not set(plan["tiles_without_thermal"]) & set(
            plan["tiles_without_scenes"]
        )

    def test_every_launched_tile_carries_its_thermal_count(self, plan):
        for entry in plan["tiles"]:
            assert entry["thermal_scenes"] > 0
            assert entry["thermal_scenes"] <= entry["scenes"]

    def test_a_partly_l2sr_tile_still_launches(self, plan):
        # Only zero comes out. A tile that is 90% L2SR still composites real
        # temperatures from the rest, and 28 of the 895 sit between 25% and
        # 100%. Dropping those would discard real coverage.
        entry = next(t for t in plan["tiles"] if t["tile_id"] == "S30W065")
        assert entry["thermal_scenes"] >= 1


class TestARunPointedAtABarrenTile:
    """`fleet_plan` filters these, so only a hand-typed tile arrives here.

    It exits 0 and writes the summary a driver keys on. A correct outcome
    reading as a dead machine is the distinction the barren-shard records exist
    to preserve, and the exit-code advice in `FINDINGS` says to key on
    `summary.json` rather than on status.
    """

    @pytest.fixture
    def run(self, slice_artifact, tmp_path):
        import json

        import shard_lst_p95

        code = shard_lst_p95.main(
            [
                "--tile",
                BARREN_TILE,
                "--inventory-uri",
                str(slice_artifact),
                "--workers",
                "2",
                "--threads-per-worker",
                "1",
                "--read-threads",
                "2",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        return code, summary, tmp_path / "out"

    def test_it_succeeds(self, run):
        code, _, _ = run
        assert code == 0

    def test_it_writes_a_summary_that_says_why(self, run):
        _, summary, _ = run
        assert summary["status"] == "no-thermal-coverage"
        assert summary["tile"] == BARREN_TILE
        assert summary["n_scenes"] == 0
        assert summary["n_scenes_inventory"] == summary["scenes_dropped_no_thermal"]

    def test_it_names_the_inventory_behind_the_answer(self, run):
        # A composite is reproducible only if the scene list is named, and so
        # is the claim that there was nothing to composite.
        _, summary, _ = run
        assert summary["inventory"]["source_sha256"]

    def test_it_writes_no_parts(self, run):
        # Nothing merged this tile, so nothing should look mergeable.
        _, _, out_dir = run
        assert not list(out_dir.glob("part-*.npz"))

    def test_it_reaches_no_cluster_and_no_bucket(self, run):
        # The check runs before staging and before the cluster, so a barren
        # tile costs no GET and no instance time beyond the inventory read.
        _, _, out_dir = run
        assert not (out_dir / "staging.json").exists()
        assert not (out_dir / "memory.csv").exists()


class TestNothingToLaunch:
    def test_an_inventory_of_l2sr_alone_stops_the_plan(self, tmp_path, slice_artifact):
        """A plan of zero tiles is a mistake, not a fleet of zero machines."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = pq.ParquetFile(slice_artifact)
        table = source.read()
        keep = pa.array(
            [name == BARREN_TILE for name in table.column("tile_id").to_pylist()]
        )
        only_barren = table.filter(keep).replace_schema_metadata(
            source.schema_arrow.metadata
        )
        path = tmp_path / "barren.parquet"
        pq.write_table(only_barren, path)

        with pytest.raises(InventoryError, match="OLI_TIRS_L2SR"):
            fleet_plan.build_plan(LAND_TILES, path)

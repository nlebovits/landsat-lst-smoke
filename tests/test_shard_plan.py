"""The shard plan is the one piece of geometry a wrong answer hides in.

FINDINGS.md publishes four numbers that come straight out of `plan_shards`:
1,296 shards for a full tile, 324 for a quarter, 324,000,000 px of coverage,
and 1.85 GB for one shard's time stack. Two bugs in this area reached a
four-instance run before rehearsal mode caught them, so each number is a test.
"""

import json
import sys
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shard_lst_p95 import (  # noqa: E402
    Shard,
    configure_read_env,
    items_for_shard,
    plan_shards,
    SHARD_FIXED_GIB,
    shard_bytes,
)

# S30W065, the tile the full run built.
FULL_TILE = (-65.0, -35.0, -60.0, -30.0)
# The north-west quarter, which shares its north and west edges with the full
# tile. plan_shards anchors row 0 to the bbox north edge and column 0 to its
# west edge, so only a quarter sharing both corners lines up with the whole.
QUARTER_TILE = (-65.0, -32.5, -62.5, -30.0)
PPD = 3600
SHARD = 512


class TestPlanShardsGeometry:
    def test_full_tile_raster_size(self):
        _, height, width = plan_shards(FULL_TILE, PPD, SHARD)
        assert (height, width) == (18_000, 18_000)

    def test_full_tile_shard_count(self):
        shards, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        assert len(shards) == 1_296

    def test_quarter_tile_is_one_quarter_of_the_full_tile(self):
        full, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        quarter, height, width = plan_shards(QUARTER_TILE, PPD, SHARD)
        assert (height, width) == (9_000, 9_000)
        assert len(quarter) == 324
        assert len(full) == 4 * len(quarter)

    def test_shards_cover_every_pixel_exactly_once(self):
        shards, height, width = plan_shards(FULL_TILE, PPD, SHARD)
        assert sum(s.ny * s.nx for s in shards) == 324_000_000 == height * width

    def test_no_two_shards_overlap(self):
        shards, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        seen = set()
        for s in shards:
            box = (s.y0, s.x0)
            assert box not in seen
            seen.add(box)
        assert len(seen) == len(shards)

    def test_edge_shards_are_smaller_rather_than_overhanging(self):
        # 18,000 is not a multiple of 512: 35 full rows of 512 plus a 80 px edge.
        shards, height, width = plan_shards(FULL_TILE, PPD, SHARD)
        assert 18_000 % SHARD == 80
        edges = [s for s in shards if s.ny != SHARD or s.nx != SHARD]
        assert edges, "a tile whose size is not a shard multiple must have edges"
        for s in shards:
            assert s.y0 + s.ny <= height
            assert s.x0 + s.nx <= width


class TestPlanShardsDeterminism:
    def test_the_plan_repeats(self):
        a, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        b, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        assert [s.bbox for s in a] == [s.bbox for s in b]

    def test_a_shard_covers_the_same_pixels_whichever_request_produced_it(self):
        """The same ground square comes out of two different requests.

        `--shard-slice` splits a tile across machines with no coordination
        beyond the slice index. That only holds if a shard index means the same
        pixels every time. Row 0 sits on the bbox north edge and column 0 on its
        west edge, and the production grid puts both on whole degrees.
        """
        full, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        quarter, _, _ = plan_shards(QUARTER_TILE, PPD, SHARD)
        # The quarter tile is the north-west corner of the full tile, whose
        # rows and columns therefore line up with the full tile's own.
        full_by_pixel = {(s.y0, s.x0): s.bbox for s in full}
        matched = 0
        for s in quarter:
            if (s.y0, s.x0) in full_by_pixel and s.ny == SHARD and s.nx == SHARD:
                assert full_by_pixel[(s.y0, s.x0)] == s.bbox
                matched += 1
        assert matched > 250, "expected most interior shards to line up"

    def test_slicing_the_plan_partitions_it(self):
        """The bug rehearsal caught: a slice must index the plan, not a
        filtered list, or the last slice comes up short and a tile gains gaps.
        """
        shards, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        quarter = len(shards) // 4
        slices = [
            shards[0:quarter],
            shards[quarter : 2 * quarter],
            shards[2 * quarter : 3 * quarter],
            shards[3 * quarter :],
        ]
        assert [len(s) for s in slices] == [324, 324, 324, 324]
        assert sum(s.ny * s.nx for sl in slices for s in sl) == 324_000_000

    def test_a_dropped_slice_leaves_a_countable_hole(self):
        shards, _, _ = plan_shards(FULL_TILE, PPD, SHARD)
        dropped = shards[3 * (len(shards) // 4) :]
        missing = sum(s.ny * s.nx for s in dropped)
        assert missing == 75_168_000


class TestItemsForShard:
    def test_a_scene_that_misses_the_shard_is_dropped(self):
        shard = Shard(0, 0, 0, 0, 512, 512, (-65.0, -30.15, -64.85, -30.0))
        far_away = (10.0, 10.0, 12.0, 12.0)
        assert items_for_shard(shard, [far_away]) == []

    def test_a_scene_that_covers_the_shard_is_kept(self):
        shard = Shard(0, 0, 0, 0, 512, 512, (-65.0, -30.15, -64.85, -30.0))
        covering = (-66.0, -31.0, -64.0, -29.0)
        assert items_for_shard(shard, [covering]) == [0]

    def test_touching_edges_do_not_count_as_overlap(self):
        shard = Shard(0, 0, 0, 0, 512, 512, (-65.0, -30.15, -64.85, -30.0))
        abutting_east = (-64.85, -30.15, -64.0, -30.0)
        assert items_for_shard(shard, [abutting_east]) == []

    def test_indices_come_back_in_input_order(self):
        shard = Shard(0, 0, 0, 0, 512, 512, (-65.0, -30.15, -64.85, -30.0))
        boxes = [
            (10.0, 10.0, 12.0, 12.0),
            (-66.0, -31.0, -64.0, -29.0),
            (20.0, 20.0, 22.0, 22.0),
            (-65.5, -30.5, -64.5, -29.5),
        ]
        assert items_for_shard(shard, boxes) == [1, 3]


class TestShardBytes:
    """The budget a fleet instance is sized from.

    These assertions used to encode a float32-only model that reported 0.39 GiB
    for a shard measured at 1.52. Reading the low number put 64 workers wanting
    97 GiB on a 128 GiB box, and they died at cluster start. So the anchor is
    now a measurement rather than a figure quoted from the document.
    """

    #: MEASURED by `measure_shard_memory.py --mode memory` at 512 px.
    #: The committed sweeps, read rather than copied, so the tests and the
    #: artifacts cannot drift. Two shard sizes, because the model claims the
    #: working set scales with the square of the edge and one size cannot show
    #: that. Regenerate with
    #: `measure_shard_memory.py --mode memory --shard N --out artifacts/shard_memory_N.json`.
    SWEEPS = {
        int(path.stem.rsplit("_", 1)[1]): json.loads(path.read_text())["rows"]
        for path in sorted(ARTIFACTS.glob("shard_memory_*.json"))
    }
    SWEEP = [(r["scenes"], r["peak_rss_gib"]) for r in SWEEPS[512]]
    POINTS = [
        (px, r["scenes"], r["peak_rss_gib"])
        for px, rows in SWEEPS.items()
        for r in rows
    ]

    @pytest.mark.parametrize(("scenes", "measured"), SWEEP)
    def test_it_never_under_predicts_the_measured_peak(self, scenes, measured):
        """The model may sit above the sample. It may not sit below it.

        Sampling RSS at 20 ms misses peaks, so the sweep is a lower bound on
        the real high-water mark and 300 scenes reads below 200. A model fitted
        through the middle of that would under-predict half the time, and
        under-predicting is what cost a fleet instance its workers.
        """
        assert shard_bytes(512, scenes) >= measured

    def test_it_stays_close_at_the_scene_counts_a_fleet_reads(self):
        """Conservative, not arbitrary. Slack here costs worker slots.

        Bounded over 400 scenes and up, which is the range a tile presents:
        199 to 971 per shard, with the densest tiles at the top. The lighter
        samples are where missed peaks dominate, and 300 reading below 200 is
        the proof that they do.
        """
        worst = max(shard_bytes(512, n) / m for n, m in self.SWEEP if n >= 400)
        assert worst < 1.25

    def test_it_counts_every_array_that_is_live_at_once(self):
        """Five arrays, 13 bytes per pixel-scene, not the decoded stack alone.

        dn uint16 and qa uint16 at 2 each, celsius float32 at 4, the valid mask
        at 1, and the copy `nanpercentile` partitions at 4. The measured slope
        is 12.7, so the accounting figure sits just above it.
        """
        arrays = shard_bytes(512, 1765) - SHARD_FIXED_GIB
        float32_only = 512 * 512 * 1765 * 4 / 1024**3
        assert arrays == pytest.approx(float32_only * 13 / 4, rel=1e-6)

    def test_the_arrays_scale_with_the_square_of_the_edge(self):
        """The fixed term does not scale, so compare the arrays alone."""
        big = shard_bytes(1024, 1765) - SHARD_FIXED_GIB
        small = shard_bytes(512, 1765) - SHARD_FIXED_GIB
        assert big == pytest.approx(4 * small)

    def test_the_measured_slope_is_the_same_at_both_shard_sizes(self):
        """The px-squared claim, against measurement rather than against itself.

        Comparing `shard_bytes` to `shard_bytes` proves only that the formula
        multiplies. If bytes per pixel-scene differed by edge, the whole reason
        a 360 px shard needs less memory than a 512 px one would be wrong, and
        that is what picks the instance.
        """
        slopes = {}
        for px, rows in self.SWEEPS.items():
            v = sorted((r["scenes"], r["peak_rss_gib"]) for r in rows)
            steps = [
                (v[i + 1][1] - v[i][1]) * 1024**3 / ((v[i + 1][0] - v[i][0]) * px * px)
                for i in range(len(v) - 1)
            ]
            slopes[px] = sum(steps) / len(steps)
        assert len(slopes) >= 2, "need sweeps at two shard sizes"
        assert max(slopes.values()) / min(slopes.values()) < 1.10, slopes

    @pytest.mark.parametrize(("shard_px", "scenes", "measured"), POINTS)
    def test_it_never_under_predicts_at_either_shard_size(
        self, shard_px, scenes, measured
    ):
        assert shard_bytes(shard_px, scenes) >= measured

    def test_a_quarter_tile_shard_no_longer_fits_a_small_worker(self):
        """1,765 scenes at 512 px is 6.3 GB, not the 1.85 GB long quoted.

        6.0 GB of that is array and the rest is per-worker overhead. The
        difference decides how many workers an instance can hold, which is what
        the failed run got wrong.
        """
        gib = shard_bytes(512, 1765)
        assert round(gib * 1024**3 / 1e9, 1) == 6.3
        assert gib > 1.8, "a 1.8 GiB worker limit cannot hold this shard"


class TestConfigureReadEnv:
    """Both the pipeline and measure_s3_requests.py call this one function.

    They used to set the read environment separately, and the measurement came
    out against settings the pipeline never used. Two of those settings move
    the request count directly.
    """

    def test_it_sets_what_the_request_count_depends_on(self, monkeypatch):
        for key in (
            "GDAL_DISABLE_READDIR_ON_OPEN",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES",
            "AWS_REQUEST_PAYER",
        ):
            monkeypatch.delenv(key, raising=False)
        configure_read_env("earth-search")
        import os

        assert os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        assert os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] == "YES"
        assert os.environ["AWS_REQUEST_PAYER"] == "requester"

    def test_planetary_computer_is_refused(self, monkeypatch):
        """It used to configure a read environment that could not read.

        Every href in the inventory is `s3://usgs-landsat`, which is
        requester-pays. Selecting planetary-computer skipped
        `AWS_REQUEST_PAYER` and left a run that failed on every scene. The flag
        selected a catalogue before the inventory replaced the per-tile search;
        now it selects a read environment, and there is only one.
        """
        import os

        monkeypatch.delenv("AWS_REQUEST_PAYER", raising=False)
        with pytest.raises(SystemExit, match="requester-pays"):
            configure_read_env("planetary-computer")
        assert "AWS_REQUEST_PAYER" not in os.environ

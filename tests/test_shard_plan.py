"""The shard plan is the one piece of geometry a wrong answer hides in.

FINDINGS.md publishes four numbers that come straight out of `plan_shards`:
1,296 shards for a full tile, 324 for a quarter, 324,000,000 px of coverage,
and 1.85 GB for one shard's time stack. Two bugs in this area reached a
four-instance run before rehearsal mode caught them, so each number is a test.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

#: The committed memory sweeps, keyed by filename. The shard edge and the
#: source come from inside each file rather than from its name, which used to
#: be parsed on the last underscore segment and made collection fail on any
#: suffix that is not a number.
SWEEPS_BY_NAME = {
    path.stem: json.loads(path.read_text())
    for path in sorted(ARTIFACTS.glob("shard_memory_*.json"))
}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shard_lst_p95 import (  # noqa: E402
    DEFAULT_SHARD_PX,
    SHARD_BYTES_PER_PIXEL_SCENE,
    SHARD_FIXED_GIB,
    Shard,
    client_bytes,
    configure_read_env,
    items_for_shard,
    plan_shards,
    shard_bytes,
    slice_demand,
    worker_memory_guard,
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
    for a shard measured at 1.42. Reading the low number put 64 workers wanting
    97 GiB on a 128 GiB box, and they died at cluster start. So the anchor is
    now a measurement rather than a figure quoted from the document.
    """

    #: MEASURED by `measure_shard_memory.py --mode memory`, read rather than
    #: copied, so the tests and the artifacts cannot drift.
    #:
    #: Four shard edges, because the model claims the working set scales with
    #: the square of the edge and one edge cannot show that. Two sources,
    #: because the synthetic fixture writes one untiled raster at the shard's
    #: own edge and the fleet reads tiled COGs far larger than the shard.
    #: Regenerate the synthetic pair with
    #: `measure_shard_memory.py --mode memory --shard N --out artifacts/shard_memory_N.json`,
    #: and a staged sweep by adding `--stage-dir` pointing at staged scenes.
    SWEEPS_BY_NAME = SWEEPS_BY_NAME
    SYNTHETIC = {
        s["shard_px"]: s["rows"]
        for s in SWEEPS_BY_NAME.values()
        if s["source"] == "synthetic"
    }
    SWEEP = [(r["scenes"], r["peak_rss_gib"]) for r in SYNTHETIC[512]]
    #: Every point of every sweep. The model has to bound all of them.
    POINTS = [
        (name, r["shard_px"], r["scenes"], r["peak_rss_gib"])
        for name, sweep in SWEEPS_BY_NAME.items()
        for r in sweep["rows"]
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

    def test_it_counts_more_than_the_five_named_arrays(self):
        """Five arrays come to 13, and 13 reads low at fleet depth.

        dn uint16 and qa uint16 at 2 each, celsius float32 at 4, the valid mask
        at 1, and the copy `nanpercentile` partitions at 4. A staged sweep over
        real COGs fits 13.68 at 360 px and 14.52 at 512, so the windowed read
        of a tiled source holds about two bytes the five do not name.
        """
        arrays = shard_bytes(512, 1765) - SHARD_FIXED_GIB
        float32_only = 512 * 512 * 1765 * 4 / 1024**3
        named_five = float32_only * 13 / 4
        assert arrays == pytest.approx(float32_only * 15 / 4, rel=1e-6)
        assert arrays > named_five, "the model has to exceed the named arrays"

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

        The slopes are the least-squares fits the sweeps record. The earlier
        estimator averaged consecutive differences, which agreed to 1.3% while
        the differences it averaged ran 10.9 to 16.4 bytes. Least squares uses
        every point once and puts the two synthetic edges 4.2% apart.

        Only the synthetic pair is compared. The staged sweeps read real
        scenes, so shard size and how much data falls in the shard move
        together, and their slopes range 11.2 to 14.8 across four edges. That
        scatter is a property of the fixture, not of the working set, which is
        why the staged sweeps are held to bounding the model rather than to
        agreeing with each other.
        """
        slopes = {
            s["shard_px"]: s["slope_bytes_per_pixel_scene"]
            for s in self.SWEEPS_BY_NAME.values()
            if s["source"] == "synthetic"
        }
        assert len(slopes) >= 2, "need synthetic sweeps at two shard sizes"
        assert max(slopes.values()) / min(slopes.values()) < 1.10, slopes

    @pytest.mark.parametrize("name", sorted(SWEEPS_BY_NAME))
    def test_the_committed_sweep_was_written_by_this_model(self, name):
        """An artifact cannot carry a prediction from a superseded model.

        `shard_memory_512.json` once declared 13 bytes per pixel-scene beside a
        `predicted_gib` column computed at 18, and it survived because no test
        read that column. The point of committing a sweep is that a
        re-measurement cannot leave the tests behind, and that only holds if
        every field in it is checked.
        """
        sweep = self.SWEEPS_BY_NAME[name]
        assert sweep["model_bytes_per_pixel_scene"] == SHARD_BYTES_PER_PIXEL_SCENE
        assert sweep["model_fixed_gib"] == SHARD_FIXED_GIB
        for row in sweep["rows"]:
            expected = round(shard_bytes(row["shard_px"], row["scenes"]), 3)
            assert row["predicted_gib"] == expected, row
            # The stored ratio comes from the unrounded pair. Both columns are
            # rounded to three places, so recomputing from them carries about
            # 0.001 of error on each at the smallest shard measured.
            assert row["ratio"] == pytest.approx(
                row["peak_rss_gib"] / expected, abs=0.005
            ), row

    @pytest.mark.parametrize(("name", "shard_px", "scenes", "measured"), POINTS)
    def test_it_never_under_predicts_any_measured_point(
        self, name, shard_px, scenes, measured
    ):
        """Every point of every sweep, synthetic and staged.

        The staged sweeps are the reason this is not the synthetic pair alone.
        A synthetic raster is written at the shard's own edge and read whole,
        which allocates no intermediate for a windowed read of a tiled COG. If
        the model only bounded that fixture it would be bounding the wrong
        read, and under-predicting is what cost a fleet instance its workers.
        """
        assert shard_bytes(shard_px, scenes) >= measured, name

    def test_a_quarter_tile_shard_no_longer_fits_a_small_worker(self):
        """1,765 scenes at 512 px is 7.2 GB, not the 1.85 GB long quoted.

        6.9 GB of that is array and the rest is per-worker overhead. The
        difference decides how many workers an instance can hold, which is what
        the failed run got wrong.
        """
        gib = shard_bytes(512, 1765)
        assert round(gib * 1024**3 / 1e9, 1) == 7.2
        assert gib > 1.8, "a 1.8 GiB worker limit cannot hold this shard"


class TestWorkerMemoryGuard:
    """The gate `shard_bytes` never had.

    Correcting the model did not stop the configuration that killed a
    `c6id.16xlarge`. It only printed a larger number on the way past.
    `staging.disk_guard` refuses a fetch that cannot finish, and this refuses a
    cluster that cannot fit, before either one spends anything.
    """

    #: The full tile the fleet writes, at 3600 px per degree over 5 degrees.
    TILE_PX = 18_000
    GIB = 1024**3
    #: MEASURED: the depths of shards[987:1051] of S30W065 at 360 px, the
    #: deepest 64-shard slice in the tile. One shard at 820 scenes and a median
    #: of 401, which is why the worst shard is not what 63 workers hold.
    DEEP_SLICE = [820] + [401] * 32 + [203] * 31

    def test_the_client_holds_sixteen_bytes_an_output_pixel(self):
        """uint16 of p95, twelve uint8 monthly counts, two bools of mask.

        4.8 GiB a tile. Both masks are built before staging so that a tile the
        water rule empties costs nothing, which makes them live for the whole
        run. At fourteen bytes 0.6 GiB of array on the largest tile sat outside
        the model, and the model is what a fleet instance is sized from.
        """
        assert client_bytes(self.TILE_PX, self.TILE_PX) == pytest.approx(4.8, abs=0.05)

    def test_emitting_the_pooled_baseline_adds_a_fourth_full_tile_array(self):
        """`--emit-pooled` allocates `pooled_out` at uint16, height by width.

        Two bytes a pixel is 0.6 GiB on the largest tile the fleet runs, which
        is the same figure that put the model 0.6 GiB low when it counted
        fourteen bytes instead of sixteen.
        """
        plain = client_bytes(self.TILE_PX, self.TILE_PX)
        with_pooled = client_bytes(self.TILE_PX, self.TILE_PX, emit_pooled=True)
        assert with_pooled - plain == pytest.approx(
            self.TILE_PX**2 * 2 / self.GIB, abs=0.01
        )
        assert with_pooled == pytest.approx(5.4, abs=0.05)

    def test_the_guard_refuses_a_run_the_pooled_array_pushes_over(self):
        # The whole point of counting it: a configuration that fits without the
        # flag and not with it has to be refused rather than printed.
        # 8 deep shards at 360 px want 13.9 GiB, and the client's own arrays
        # 4.8. That is 18.7 GiB, and 19.3 with the baseline.
        args = (360, [820] * 8, 8, self.TILE_PX, self.TILE_PX)
        machine = int(19.0 * self.GIB)
        assert worker_memory_guard(*args, total_bytes=machine) == pytest.approx(
            18.7, abs=0.1
        )
        with pytest.raises(SystemExit, match="19.3 GiB"):
            worker_memory_guard(*args, total_bytes=machine, emit_pooled=True)

    def test_an_empty_slice_still_counts_the_pooled_array(self):
        demand = worker_memory_guard(
            360,
            [],
            64,
            self.TILE_PX,
            self.TILE_PX,
            total_bytes=247 * self.GIB,
            emit_pooled=True,
        )
        assert demand == pytest.approx(
            client_bytes(self.TILE_PX, self.TILE_PX, emit_pooled=True)
        )

    def test_the_configuration_that_killed_an_instance_is_refused(self):
        """64 workers, 512 px, a quarter tile of scenes, on 128 GiB.

        The demand is about 97 GiB of workers on a box that also carries the
        client's arrays and 78 GB of staged page cache. The old model called
        the same shard 0.39 GiB and nothing objected.
        """
        with pytest.raises(SystemExit) as exc:
            worker_memory_guard(
                512,
                [1765] * 64,
                64,
                self.TILE_PX,
                self.TILE_PX,
                total_bytes=128 * self.GIB,
            )
        message = str(exc.value)
        # The operator's next decision is a smaller shard or fewer workers, so
        # the message has to carry the edge that fits and both escapes.
        assert "--shard" in message
        assert "--force" in message
        assert "128.0 GiB" in message

    def test_the_configuration_that_worked_is_allowed(self):
        """The m6id.16xlarge run: 64 workers, 360 px, the deep slice, 247 GiB."""
        demand = worker_memory_guard(
            360,
            self.DEEP_SLICE,
            64,
            self.TILE_PX,
            self.TILE_PX,
            total_bytes=247 * self.GIB,
        )
        expected = sum(shard_bytes(360, n) for n in self.DEEP_SLICE)
        assert demand == pytest.approx(expected + 4.8, abs=0.05)

    def test_it_sums_the_actual_depths_rather_than_the_worst(self):
        """The correction this guard needed, MEASURED at 2.77x.

        Multiplying the worst shard by the slot count reserved 102.6 GiB for a
        slice that peaked at 37.0. Summing the depths reserves 64.0, because
        one shard runs 820 scenes deep and the median runs 401.
        """
        summed = worker_memory_guard(
            360,
            self.DEEP_SLICE,
            64,
            self.TILE_PX,
            self.TILE_PX,
            total_bytes=247 * self.GIB,
        )
        worst_times_slots = 64 * shard_bytes(360, max(self.DEEP_SLICE))
        assert summed < worst_times_slots, "summing must reserve less"
        # Still above the 37.0 GiB the run measured, because a sampler cannot
        # prove the coincident peak it never caught.
        assert summed > 37.0

    def test_only_the_deepest_shards_up_to_the_slot_count_are_held(self):
        """Beyond the slots a slice queues, so shard 65 is not resident."""
        eight = worker_memory_guard(
            360, [820] * 64, 8, self.TILE_PX, self.TILE_PX, total_bytes=247 * self.GIB
        )
        assert eight == pytest.approx(8 * shard_bytes(360, 820) + 4.8, abs=0.05)

    def test_the_default_edge_fits_the_deepest_tile_in_the_fleet(self):
        """512 px refused four tiles on the machine the fleet runs.

        MEASURED against the full inventory at 64 workers and 256 GiB:
        N50E100 wanted 259.0 GiB at 512 px and 247.3 at 500. The other three
        were N50E090, N50E095 and N50E115, all within 2 GiB of the limit.

        1,028 scenes is the deepest shard the fleet holds, at 512 px. This
        prices that shard on the default edge and checks it leaves room, so a
        deeper inventory has somewhere to grow before tiles start failing at
        launch.
        """
        depths = [1028] * 64
        demand = slice_demand(DEFAULT_SHARD_PX, depths, 64, self.TILE_PX, self.TILE_PX)
        assert demand < 256.0
        assert slice_demand(512, depths, 64, self.TILE_PX, self.TILE_PX) > 256.0

    def test_the_default_edge_divides_a_tile_exactly(self):
        # 71 ragged shards at 512 px, none at 500. An edge shard is a smaller
        # unit of work in a plan whose memory budget is set by the full one.
        shards, height, width = plan_shards(FULL_TILE, PPD, DEFAULT_SHARD_PX)
        assert height == width == self.TILE_PX
        assert not [s for s in shards if s.ny != DEFAULT_SHARD_PX]
        assert not [s for s in shards if s.nx != DEFAULT_SHARD_PX]

    def test_the_reported_demand_is_the_one_the_guard_refuses_on(self):
        """The dry run used to print a verdict the guard disagreed with.

        `worst shard x slots` is the reading this guard replaced after it
        over-reserved by 2.77x. The dry run kept printing it, and only the
        superseded one carried a verdict. MEASURED on the real inventory at
        512 px and 64 workers: N45E100 read `OVER by 5.5 GiB` on a 256 GiB
        machine the guard accepts at 225.3 GiB, and eleven other tiles did the
        same.
        """
        depths = [1027] + [400] * 63
        demand = slice_demand(512, depths, 64, self.TILE_PX, self.TILE_PX)
        naive = shard_bytes(512, 1027) * 64 + client_bytes(self.TILE_PX, self.TILE_PX)
        assert demand < naive

        # A machine between the two verdicts proves they are one model now.
        # The guard returns the figure it would have refused on.
        between = int((demand + naive) / 2 * self.GIB)
        assert worker_memory_guard(
            512, depths, 64, self.TILE_PX, self.TILE_PX, total_bytes=between
        ) == pytest.approx(demand)

    def test_both_demands_count_the_pooled_baseline(self):
        depths = [820] * 8
        assert slice_demand(
            360, depths, 8, self.TILE_PX, self.TILE_PX, emit_pooled=True
        ) - slice_demand(360, depths, 8, self.TILE_PX, self.TILE_PX) == pytest.approx(
            self.TILE_PX**2 * 2 / self.GIB, abs=0.01
        )

    def test_an_empty_slice_demands_only_the_client_arrays(self):
        demand = worker_memory_guard(
            360, [], 64, self.TILE_PX, self.TILE_PX, total_bytes=247 * self.GIB
        )
        assert demand == pytest.approx(client_bytes(self.TILE_PX, self.TILE_PX))

    def test_the_edge_it_recommends_actually_fits(self):
        """A message that names an unusable escape is worse than none."""
        with pytest.raises(SystemExit) as exc:
            worker_memory_guard(
                512,
                [1765] * 64,
                64,
                self.TILE_PX,
                self.TILE_PX,
                total_bytes=128 * self.GIB,
            )
        named = re.search(r"--shard (\d+)", str(exc.value))
        assert named, "the message has to name an edge to try"
        edge = int(named.group(1))
        assert edge > 0
        assert worker_memory_guard(
            edge,
            [1765] * 64,
            64,
            self.TILE_PX,
            self.TILE_PX,
            total_bytes=128 * self.GIB,
        )

    def test_a_machine_with_no_room_for_the_client_still_refuses(self):
        """The client's arrays alone can exceed the box. No negative edge."""
        with pytest.raises(SystemExit) as exc:
            worker_memory_guard(
                512,
                [1765] * 64,
                64,
                self.TILE_PX,
                self.TILE_PX,
                total_bytes=2 * self.GIB,
            )
        assert "--shard 0" in str(exc.value)


class TestFitSlope:
    """A sweep that ran at one depth has no slope, and must say so.

    A staged sweep caps every requested count at the scenes on disk, so asking
    for 200 400 600 820 against a directory of 100 collapses to a single depth.
    Least squares then divides by zero, three minutes into a run on a machine
    that is being paid for by the hour.
    """

    def _rows(self, depths, shard_px=360):
        return [
            {"scenes": n, "shard_px": shard_px, "peak_rss_gib": 0.2 + n * 0.001}
            for n in depths
        ]

    def test_one_depth_yields_no_slope_rather_than_an_exception(self):
        import measure_shard_memory

        fit = measure_shard_memory.fit_slope(self._rows([100]))
        assert fit["slope_bytes_per_pixel_scene"] is None
        assert fit["intercept_gib"] is None

    def test_repeated_depths_yield_no_slope(self):
        import measure_shard_memory

        fit = measure_shard_memory.fit_slope(self._rows([100, 100, 100]))
        assert fit["slope_bytes_per_pixel_scene"] is None

    def test_two_depths_are_enough(self):
        import measure_shard_memory

        fit = measure_shard_memory.fit_slope(self._rows([100, 200]))
        assert fit["slope_bytes_per_pixel_scene"] > 0

    def test_it_recovers_a_slope_it_was_given(self):
        # 13 bytes per pixel-scene and a 0.25 GiB intercept, exactly.
        import measure_shard_memory

        px = 360
        rows = [
            {
                "scenes": n,
                "shard_px": px,
                "peak_rss_gib": px * px * n * 13 / 1024**3 + 0.25,
            }
            for n in (100, 200, 400, 800)
        ]
        fit = measure_shard_memory.fit_slope(rows)
        assert fit["slope_bytes_per_pixel_scene"] == pytest.approx(13, abs=0.01)
        assert fit["intercept_gib"] == pytest.approx(0.25, abs=0.01)


class TestTheDryRunChecksTheTargetMachine:
    """Planning happens on a laptop. The run happens on an instance.

    The guard in `main` reads the host it runs on, which is the right machine
    only when the run is already there. A dry run priced against this laptop
    would clear a configuration that an m6id refuses, or refuse one it would
    take. `--target-memory-gib` names the machine instead.
    """

    def _dry_run(self, slice_artifact, tmp_path, *extra):
        import shard_lst_p95

        return shard_lst_p95.main(
            [
                "--tile",
                "S30W065",
                "--inventory-uri",
                str(slice_artifact),
                "--shard",
                "512",
                "--workers",
                "64",
                "--threads-per-worker",
                "1",
                "--dry-run",
                "--search-in-dry-run",
                "--out-dir",
                str(tmp_path / "dry"),
                *extra,
            ]
        )

    def test_a_machine_that_cannot_hold_the_run_exits_two(
        self, slice_artifact, tmp_path, capsys
    ):
        # 2, not 1, so a driver can tell a configuration that does not fit from
        # a plan that failed to build.
        code = self._dry_run(slice_artifact, tmp_path, "--target-memory-gib", "1")
        assert code == 2
        out = capsys.readouterr().out
        assert "REFUSED on a 1 GiB machine" in out
        assert "--shard" in out

    def test_a_machine_with_room_exits_zero(self, slice_artifact, tmp_path, capsys):
        code = self._dry_run(slice_artifact, tmp_path, "--target-memory-gib", "4096")
        assert code == 0
        assert "fits" in capsys.readouterr().out

    def test_without_a_target_it_judges_nothing(self, slice_artifact, tmp_path, capsys):
        # The host planning a fleet run is not the host doing it, so a dry run
        # given no machine compares against none.
        code = self._dry_run(slice_artifact, tmp_path)
        assert code == 0
        out = capsys.readouterr().out
        assert "fits" not in out
        assert "REFUSED" not in out

    def test_the_budget_counts_the_client_arrays(
        self, slice_artifact, tmp_path, capsys
    ):
        # The client gathers into two full-tile arrays while the workers are
        # still allocating, so the machine holds both at once.
        self._dry_run(slice_artifact, tmp_path)
        out = capsys.readouterr().out
        assert f"+{client_bytes(18000, 18000):.1f} GiB of client output" in out

    def test_it_reports_the_slice_a_machine_runs_not_the_whole_tile(
        self, slice_artifact, tmp_path, capsys
    ):
        """The slice's worst shard is what that machine's memory holds.

        On S30W065 at 360 px the tile's worst shard is 820 scenes deep and
        `shards[0:64]` is 404. Reporting the tile figure overstates a light
        slice, and reporting a light slice as the tile understates every other
        machine in the fleet.
        """
        self._dry_run(slice_artifact, tmp_path, "--shard-slice", "0:8")
        out = capsys.readouterr().out
        assert "tile:" in out
        assert "what this machine holds" in out


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

"""The two corrections, each tested against the defect it exists to remove.

`destripe.py` carries claims that are cheap to state and expensive to get
wrong. Four of them decide whether the correction is honest:

The offset is one scalar per scene, so subtracting it cannot move any pixel
relative to its neighbour. If it could, the correction would be reshaping the
field rather than rebasing a scene.

The reference is a calendar-month median, not an annual one. An annual
reference absorbs the seasonal cycle and subtracting it removes the season
itself. `nlebovits/landsat-lst` measured that failure at 40.6 C down to 29.8 C,
spatial correlation 0.44.

A pixel one WRS path reaches must come out bit-identical to that path composited
alone. Feathering is supposed to touch the overlap and nothing else.

Per-scene values join the loaded stack on the time coordinate, never on
position. `odc.stac` sorts the time axis by datetime and de-striping thins it,
so positional labels address the wrong scenes. That defect killed all 35
composite shards of one production run in the sibling repository.

Nothing here reads S3 or opens a catalogue.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import destripe  # noqa: E402
from masks import transform_for  # noqa: E402

# A 10 x 1 degree box at 6 cells per degree: 6 rows, 60 columns. Small enough
# to assert on by hand, wide enough to carry an overlap with an interior.
BBOX = (-5.0, 0.0, 5.0, 1.0)
CELLS_PER_DEGREE = 6
HEIGHT, WIDTH = 6, 60
WEST, EAST = "001", "002"
#: Columns 24 to 35 are reached by both paths.
OVERLAP = slice(24, 36)


def grid_transform():
    return transform_for(BBOX, CELLS_PER_DEGREE)


def two_paths() -> dict[str, np.ndarray]:
    """West covers columns 0 to 35, east covers 24 to 59."""
    west = np.zeros((HEIGHT, WIDTH), dtype=bool)
    west[:, :36] = True
    east = np.zeros((HEIGHT, WIDTH), dtype=bool)
    east[:, 24:] = True
    return {WEST: west, EAST: east}


def scene_stack(n_scenes: int, months, bias=None, seed: int = 0):
    """A stack with a smooth spatial field, a seasonal cycle, and per-scene bias."""
    rng = np.random.default_rng(seed)
    months = np.asarray(months)
    field = (
        np.linspace(0.0, 6.0, WIDTH, dtype="float32")[None, :]
        + np.linspace(0.0, 3.0, HEIGHT, dtype="float32")[:, None]
    )
    season = 12.0 * np.cos((months - 1) * np.pi / 6.0)
    stack = np.empty((n_scenes, HEIGHT, WIDTH), dtype="float32")
    for s in range(n_scenes):
        stack[s] = 25.0 + field + season[s] + rng.normal(0.0, 0.2, (HEIGHT, WIDTH))
    if bias is not None:
        stack += np.asarray(bias, dtype="float32")[:, None, None]
    return stack


def offsets_of(stack, months):
    """Run the whole estimate in one block, the way one shard would."""
    planes, ref = destripe.month_climatology(stack, months)
    hist = np.zeros((stack.shape[0], destripe.N_ANOMALY_BINS), dtype="uint32")
    n_valid = np.zeros(stack.shape[0], dtype="int64")
    destripe.accumulate_anomaly(hist, n_valid, stack, months, planes, ref)
    return destripe.offsets_from_histograms(hist), n_valid


class TestTheOffsetIsOneScalarPerScene:
    def test_subtracting_it_moves_no_pixel_relative_to_its_neighbour(self):
        """Spatial structure survives, to well inside one output DN.

        The operation is exact in arithmetic. In float32 it moves the low bit,
        so the bound here is the encoding step (0.01 C) rather than zero. The
        measured residual is about four orders of magnitude below it.
        """
        months = np.array([1, 1, 7, 7])
        stack = scene_stack(4, months)
        before = np.diff(stack, axis=2)
        destripe.subtract_offsets(stack, np.array([3.0, -2.0, 0.5, -7.25]))
        moved = np.abs(before - np.diff(stack, axis=2)).max()
        assert moved < 1e-4
        assert moved < destripe.ANOMALY_BIN_C

    def test_it_shifts_every_pixel_of_the_scene_by_the_same_amount(self):
        stack = scene_stack(2, [1, 7])
        original = stack.copy()
        destripe.subtract_offsets(stack, np.array([4.0, -1.5]))
        assert np.allclose(original[0] - stack[0], 4.0)
        assert np.allclose(original[1] - stack[1], -1.5)


class TestTheReferenceIsSeasonal:
    def test_an_injected_per_scene_bias_comes_back(self):
        months = np.repeat(np.arange(1, 13), 4)
        bias = np.zeros(months.size)
        bias[5] = 6.0
        bias[17] = -4.0
        stack = scene_stack(months.size, months, bias=bias)
        offset, _ = offsets_of(stack, months)
        assert offset[5] == pytest.approx(6.0, abs=0.15)
        assert offset[17] == pytest.approx(-4.0, abs=0.15)

    def test_an_unbiased_scene_gets_an_offset_near_zero(self):
        months = np.repeat(np.arange(1, 13), 4)
        stack = scene_stack(months.size, months)
        offset, _ = offsets_of(stack, months)
        assert np.abs(offset).max() < 0.5

    def test_the_seasonal_amplitude_survives_the_correction(self):
        """The whole reason the reference is monthly and not annual."""
        months = np.repeat(np.arange(1, 13), 4)
        stack = scene_stack(months.size, months)
        amplitude = np.ptp(np.nanmean(stack, axis=(1, 2)))
        offset, _ = offsets_of(stack, months)
        corrected = destripe.subtract_offsets(stack.copy(), offset)
        after = np.ptp(np.nanmean(corrected, axis=(1, 2)))
        assert after == pytest.approx(amplitude, rel=0.05)

    def test_an_annual_reference_would_destroy_it(self):
        """Pin the failure the monthly reference exists to avoid."""
        months = np.repeat(np.arange(1, 13), 4)
        stack = scene_stack(months.size, months)
        amplitude = np.ptp(np.nanmean(stack, axis=(1, 2)))
        annual = np.nanmedian(stack, axis=0)
        annual_offset = np.nanmedian(stack - annual, axis=(1, 2))
        after = np.ptp(np.nanmean(stack - annual_offset[:, None, None], axis=(1, 2)))
        assert after < 0.1 * amplitude


class TestTheHistogramMedian:
    def test_it_matches_a_direct_median_to_one_bin(self):
        months = np.repeat(np.arange(1, 13), 4)
        rng = np.random.default_rng(7)
        stack = scene_stack(months.size, months, bias=rng.normal(0.0, 5.0, 48))
        planes, ref = destripe.month_climatology(stack, months)
        plane_of = {int(m): i for i, m in enumerate(planes)}
        direct = np.array(
            [
                np.nanmedian(stack[s] - ref[plane_of[int(months[s])]])
                for s in range(months.size)
            ]
        )
        offset, _ = offsets_of(stack, months)
        assert np.abs(offset - direct).max() <= destripe.ANOMALY_BIN_C

    def test_the_bin_range_covers_every_representable_anomaly(self):
        """No overflow bin exists, so nothing may fall outside the range."""
        from lst_qa import LST_VALID_MAX_C, LST_VALID_MIN_C

        widest = LST_VALID_MAX_C - LST_VALID_MIN_C
        assert destripe.ANOMALY_MIN_C <= -widest
        assert destripe.ANOMALY_MAX_C >= widest

    def test_splitting_the_tile_into_blocks_changes_nothing(self):
        """The histogram is what lets the estimate run in one pass.

        A spatial median does not decompose, which is why the sibling needs a
        second pass over every scene. If block splitting moved an offset, this
        design would be buying that pass back.
        """
        months = np.repeat(np.arange(1, 13), 4)
        rng = np.random.default_rng(3)
        stack = scene_stack(months.size, months, bias=rng.normal(0.0, 4.0, 48))
        whole, whole_n = offsets_of(stack, months)

        hist = np.zeros((months.size, destripe.N_ANOMALY_BINS), dtype="uint32")
        n_valid = np.zeros(months.size, dtype="int64")
        for x0 in range(0, WIDTH, 13):
            block = stack[:, :, x0 : x0 + 13]
            planes, ref = destripe.month_climatology(block, months)
            destripe.accumulate_anomaly(hist, n_valid, block, months, planes, ref)
        split = destripe.offsets_from_histograms(hist)

        assert np.array_equal(n_valid, whole_n)
        # The climatology itself is per pixel, so blocking it is exact. Only the
        # median of the pooled anomaly is read off bins.
        assert np.abs(split - whole).max() <= destripe.ANOMALY_BIN_C

    def test_a_scene_with_no_valid_pixel_gets_no_offset(self):
        months = np.array([1, 1, 7, 7])
        stack = scene_stack(4, months)
        stack[2] = np.nan
        offset, n_valid = offsets_of(stack, months)
        assert np.isnan(offset[2])
        assert n_valid[2] == 0


class TestRejection:
    def test_an_extreme_scene_is_discarded_not_clamped(self):
        offset = np.array([0.2, -73.0, 1.1])
        n_valid = np.array([9_000, 9_000, 9_000])
        keep = destripe.keep_mask(offset, n_valid, floor=500)
        assert list(keep) == [True, False, True]

    def test_a_sparse_scene_is_discarded(self):
        keep = destripe.keep_mask(
            np.array([0.1, 0.1]), np.array([9_000, 12]), floor=500
        )
        assert list(keep) == [True, False]

    def test_a_scene_with_no_offset_is_discarded(self):
        keep = destripe.keep_mask(
            np.array([np.nan, 0.1]), np.array([9_000, 9_000]), floor=500
        )
        assert list(keep) == [False, True]

    def test_the_cap_is_the_only_thing_that_decides_the_edge(self):
        offset = np.array([14.9, 15.0, 15.1])
        n_valid = np.full(3, 9_000)
        assert list(destripe.keep_mask(offset, n_valid, floor=1)) == [
            True,
            True,
            False,
        ]

    def test_the_prep_floor_replaces_the_native_one(self):
        """A coarse valid count cannot be scaled back into a native one."""
        assert destripe.DESTRIPE_MIN_PREP_SAMPLES < destripe.DESTRIPE_MIN_SCENE_PIXELS

    def test_the_diagnostics_report_what_rejection_removed(self):
        offset = np.array([0.5, -40.0, 1.0, np.nan])
        keep = destripe.keep_mask(offset, np.full(4, 9_000), floor=1)
        report = destripe.offset_diagnostics(offset, keep)
        assert report["n_scenes"] == 4
        assert report["n_kept"] == 2
        assert report["rejected_frac"] == 0.5


class TestTheSwath:
    def test_a_quad_keeps_the_ground_half_its_scenes_reached(self):
        count = np.zeros((HEIGHT, WIDTH), dtype="uint16")
        count[:, :10] = 8
        count[:, 10:20] = 3
        masks = destripe.swath_masks({(WEST, "030"): count}, {(WEST, "030"): 10})
        assert masks[WEST][:, :10].all()
        assert not masks[WEST][:, 10:].any()

    def test_a_path_unions_its_own_quads_and_nobody_elses(self):
        north = np.zeros((HEIGHT, WIDTH), dtype="uint16")
        north[:3, :20] = 4
        south = np.zeros((HEIGHT, WIDTH), dtype="uint16")
        south[3:, :20] = 4
        other = np.zeros((HEIGHT, WIDTH), dtype="uint16")
        other[:, 40:] = 4
        masks = destripe.swath_masks(
            {(WEST, "030"): north, (WEST, "031"): south, (EAST, "030"): other},
            {(WEST, "030"): 4, (WEST, "031"): 4, (EAST, "030"): 4},
        )
        assert masks[WEST][:, :20].all()
        assert not masks[WEST][:, 20:].any()
        assert not masks[EAST][:, :40].any()

    def test_a_quad_no_cell_retains_drops_out(self):
        count = np.zeros((HEIGHT, WIDTH), dtype="uint16")
        count[:, :5] = 1
        masks = destripe.swath_masks({(WEST, "030"): count}, {(WEST, "030"): 40})
        assert masks == {}


class TestWeights:
    def test_they_sum_to_one_wherever_a_path_covers(self):
        paths, weight, inside = destripe.path_weights(two_paths(), grid_transform())
        covered = inside.any(axis=0)
        total = weight.sum(axis=0)
        assert paths == (WEST, EAST)
        assert np.allclose(total[covered], 1.0, atol=1e-6)
        assert np.array_equal(total[~covered], np.zeros(int((~covered).sum())))

    def test_a_single_path_pixel_keeps_all_of_its_weight(self):
        _paths, weight, inside = destripe.path_weights(two_paths(), grid_transform())
        single = inside.sum(axis=0) == 1
        assert single.any()
        assert np.array_equal(
            weight[0][single & inside[0]], np.ones(int((single & inside[0]).sum()))
        )

    def test_each_weight_runs_from_one_to_zero_across_the_overlap(self):
        _paths, weight, _inside = destripe.path_weights(
            two_paths(), grid_transform(), factor=1
        )
        west = weight[0, 3, OVERLAP]
        east = weight[1, 3, OVERLAP]
        assert west[0] > 0.8
        assert west[-1] < 0.2
        assert east[0] < 0.2
        assert east[-1] > 0.8
        assert np.allclose(west + east, 1.0, atol=1e-6)

    def test_the_ramp_is_monotone_and_has_no_step(self):
        _paths, weight, _inside = destripe.path_weights(
            two_paths(), grid_transform(), factor=1
        )
        west = weight[0, 3, OVERLAP]
        assert np.all(np.diff(west) <= 1e-6)
        assert np.abs(np.diff(west)).max() < 0.25

    def test_three_covering_paths_blend_continuously(self):
        masks = two_paths()
        middle = np.zeros((HEIGHT, WIDTH), dtype=bool)
        middle[:, 20:40] = True
        masks["003"] = middle
        _paths, weight, inside = destripe.path_weights(masks, grid_transform())
        k = inside.sum(axis=0)
        assert (k == 3).any()
        assert np.allclose(weight.sum(axis=0)[k == 3], 1.0, atol=1e-6)
        assert (weight >= 0).all()

    def test_path_order_cannot_change_the_weights(self):
        forward = two_paths()
        reversed_ = {k: forward[k] for k in reversed(list(forward))}
        a = destripe.path_weights(forward, grid_transform())
        b = destripe.path_weights(reversed_, grid_transform())
        assert a[0] == b[0]
        assert np.array_equal(a[1], b[1])

    def test_no_path_means_no_weight_and_no_coverage(self):
        paths, weight, inside = destripe.path_weights({}, grid_transform())
        assert paths == ()
        assert weight.size == 0
        assert inside.size == 0

    def test_a_swath_touching_the_grid_edge_does_not_ramp_toward_it(self):
        """The grid border is not an acquisition edge.

        A swath read off the data is cut wherever the grid stops, so the border
        joins the polygon boundary and the cross-fade falls toward the corner of
        the raster instead of toward the place the path stops contributing. Two
        neighbouring tiles would then disagree along their shared border, which
        trades the WRS seam for a seam on the tile grid.
        """
        masks = two_paths()
        _paths, weight, _inside = destripe.path_weights(
            masks, grid_transform(), factor=1
        )
        # The overlap runs the full height, so every row carries the same ramp.
        # With the border in the boundary the first and last rows would measure
        # their distance to it instead, and fall toward 0.
        column = weight[0, :, OVERLAP.start + 4]
        assert np.ptp(column) < 0.01
        assert np.allclose(column, column[len(column) // 2], atol=0.01)
        # What remains is an endpoint effect, not a ramp: the clipped segment
        # stops one cell short of the grid, so a point in the outermost row is
        # half a cell past its end and measures a fraction further. Measured
        # here at 0.0009 of a weight, against 0.09 of one DN in the output.
        assert np.ptp(column) > 0.0

    def test_the_coarse_ramp_survives_a_corner(self):
        """Where the padded coarse mask meets the clipped boundary.

        `_block_any` grows a mask at the grid edge and `_boundaries` cuts that
        edge out of the linework. Each is right alone. This is the corner where
        both act at once.
        """
        height, width = 256, 256
        masks = {}
        west = np.zeros((height, width), dtype=bool)
        west[:, :160] = True
        east = np.zeros((height, width), dtype=bool)
        east[:, 96:] = True
        masks[WEST], masks[EAST] = west, east
        transform = transform_for((-5.0, 0.0, 5.0, 10.0), width // 10)

        _paths, weight, inside = destripe.path_weights(masks, transform, factor=4)
        covered = inside.any(axis=0)
        total = weight.sum(axis=0)
        assert np.allclose(total[covered], 1.0, atol=1e-5)
        assert (weight >= 0).all()
        # No cell falls to the equal-share tie-break, which is what the coarse
        # containment band used to produce along the whole boundary.
        multi = inside.sum(axis=0) >= 2
        assert not np.isclose(weight[0][multi], 0.5, atol=1e-6).all()

    def test_a_grid_too_small_to_coarsen_uses_the_exact_ramp(self):
        """Six rows cannot carry a ramp on one and a half cells."""
        coarse = destripe.path_weights(two_paths(), grid_transform(), factor=8)
        exact = destripe.path_weights(two_paths(), grid_transform(), factor=1)
        assert np.array_equal(coarse[1], exact[1])

    def test_the_coarse_ramp_tracks_the_exact_one(self):
        big_bbox = (-5.0, 0.0, 5.0, 5.0)
        height, width = 512, 1024
        cells = width // 10
        west = np.zeros((height, width), dtype=bool)
        west[:, : int(width * 0.6)] = True
        east = np.zeros((height, width), dtype=bool)
        east[:, int(width * 0.4) :] = True
        masks = {WEST: west, EAST: east}
        transform = transform_for(big_bbox, cells)

        coarse = destripe.path_weights(masks, transform, factor=4)
        exact = destripe.path_weights(masks, transform, factor=1)
        assert coarse[0] == exact[0]
        # Containment is exact whatever the factor, so these must agree exactly.
        assert np.array_equal(coarse[2], exact[2])
        single = exact[2].sum(axis=0) == 1
        assert np.array_equal(coarse[1][:, single], exact[1][:, single])
        covered = exact[2].any(axis=0)
        assert np.allclose(coarse[1].sum(axis=0)[covered], 1.0, atol=1e-5)
        assert np.abs(coarse[1] - exact[1])[:, covered].max() < 0.02


class TestTheFeatheredPercentile:
    def stack_and_labels(self, warm_offset: float = 4.0, n_scenes: int = 24):
        """Scenes alternating between two paths, the east path running warm.

        A scene observes only inside its own path's swath, which is what makes
        the contributing set change at the swath edge in the first place.

        The spread within a path is a deterministic ramp rather than noise. A
        percentile of a handful of random draws carries its own sampling
        scatter, and that scatter is larger than the seam these tests measure.
        """
        labels = np.array([WEST, EAST] * (n_scenes // 2))
        masks = two_paths()
        field = (30.0 + 0.02 * np.arange(WIDTH, dtype="float32"))[None, :] * np.ones(
            (HEIGHT, 1), dtype="float32"
        )
        stack = np.empty((n_scenes, HEIGHT, WIDTH), dtype="float32")
        rank = {WEST: 0, EAST: 0}
        for s in range(n_scenes):
            path = labels[s]
            stack[s] = field + 0.25 * rank[path]
            rank[path] += 1
            if path == EAST:
                stack[s] += warm_offset
            stack[s][~masks[path]] = np.nan
        return stack, labels

    def test_a_single_path_pixel_is_bit_identical_to_that_path_alone(self):
        stack, labels = self.stack_and_labels()
        paths, weight, inside = destripe.path_weights(two_paths(), grid_transform())
        out, _ = destripe.feathered_percentile(stack, labels, paths, weight)
        for j, path in enumerate(paths):
            single = (inside.sum(axis=0) == 1) & inside[j]
            if not single.any():
                continue
            alone = destripe.pooled_percentile(stack[labels == path])
            assert np.array_equal(out[single], alone[single])

    def test_permuting_the_paths_leaves_the_result_bit_identical(self):
        stack, labels = self.stack_and_labels()
        forward = two_paths()
        reversed_ = {k: forward[k] for k in reversed(list(forward))}
        a = destripe.path_weights(forward, grid_transform())
        b = destripe.path_weights(reversed_, grid_transform())
        out_a, _ = destripe.feathered_percentile(stack, labels, a[0], a[1])
        out_b, _ = destripe.feathered_percentile(stack, labels, b[0], b[1])
        assert np.array_equal(out_a, out_b)

    def test_it_removes_the_step_the_pooled_percentile_leaves(self):
        """The defect, and the fix, on one stack.

        The east path runs 6 C warm. Column 23 is reached by west alone and
        column 24 starts the overlap, so the pooled percentile jumps there: its
        sample set gains the warmer path's scenes all at once. That jump is the
        seam. The feathered value enters the overlap at the west estimate and
        leaves it at the east one, so it crosses the same ground continuously.
        """
        stack, labels = self.stack_and_labels(warm_offset=6.0)
        paths, weight, _inside = destripe.path_weights(
            two_paths(), grid_transform(), factor=1
        )
        pooled = destripe.pooled_percentile(stack)
        feathered, _ = destripe.feathered_percentile(stack, labels, paths, weight)

        # A seam is a jump between neighbouring columns, so that is the
        # measurement. Crossing the same 6 C over the twelve columns of the
        # overlap is not a seam, and averaging over the transect would score
        # the two fields the same.
        def worst_jump(field):
            return float(np.abs(np.diff(field[:, 1:-1], axis=1)).max())

        assert worst_jump(pooled) > 5.0
        assert worst_jump(feathered) < 0.6
        # The field still has to arrive at the same place on both sides.
        assert feathered[:, 0].mean() == pytest.approx(pooled[:, 0].mean())
        assert feathered[:, -1].mean() == pytest.approx(pooled[:, -1].mean())

    def test_a_path_that_observed_nothing_here_drops_out(self):
        stack, labels = self.stack_and_labels()
        stack[labels == EAST, :, 24:30] = np.nan
        paths, weight, _inside = destripe.path_weights(two_paths(), grid_transform())
        out, _ = destripe.feathered_percentile(stack, labels, paths, weight)
        west_alone = destripe.pooled_percentile(stack[labels == WEST])
        # Renormalising divides by the surviving weight, so this is the west
        # estimate to float32 rounding rather than bit for bit.
        assert np.allclose(out[:, 24:30], west_alone[:, 24:30], rtol=1e-6)

    def test_a_pixel_outside_every_swath_takes_the_pooled_percentile(self):
        """The swath is a threshold, so ground outside it still gets observed.

        A quad's swath is where at least half its scenes reached. A pixel one
        path reaches on a third of its passes is outside every swath and
        carries real temperatures. Returning nodata there would discard them
        while `qa_count` went on counting them.
        """
        stack, labels = self.stack_and_labels()
        masks = two_paths()
        masks[WEST][:, :4] = False
        masks[EAST][:, :4] = False
        paths, weight, _inside = destripe.path_weights(masks, grid_transform())
        out, pooled = destripe.feathered_percentile(stack, labels, paths, weight)

        assert not np.isnan(out[:, :4]).any()
        assert np.allclose(out[:, :4], destripe.pooled_percentile(stack)[:, :4])
        assert not np.isnan(out[:, 10:]).any()
        assert int(pooled.sum()) == out[:, :4].size
        assert pooled[:, :4].all()

    def test_a_pixel_nothing_observed_is_still_nodata(self):
        """The one remaining meaning of a nodata pixel in the composite."""
        stack, labels = self.stack_and_labels()
        stack[:, :, :4] = np.nan
        masks = two_paths()
        masks[WEST][:, :4] = False
        masks[EAST][:, :4] = False
        paths, weight, _inside = destripe.path_weights(masks, grid_transform())
        out, pooled = destripe.feathered_percentile(stack, labels, paths, weight)

        assert np.isnan(out[:, :4]).all()
        assert int(pooled.sum()) == 0

    def test_the_fallback_leaves_a_covered_pixel_bit_identical(self):
        """Rescuing the uncovered pixels must not move the feathered ones."""
        stack, labels = self.stack_and_labels()
        paths, weight, inside = destripe.path_weights(two_paths(), grid_transform())
        full, pooled = destripe.feathered_percentile(stack, labels, paths, weight)

        narrowed = two_paths()
        narrowed[WEST][:, :4] = False
        narrowed[EAST][:, :4] = False
        n_paths, n_weight, _ = destripe.path_weights(narrowed, grid_transform())
        cut, _ = destripe.feathered_percentile(stack, labels, n_paths, n_weight)

        covered = inside.any(axis=0)
        covered[:, :4] = False
        assert int(pooled.sum()) == 0
        assert np.array_equal(full[covered], cut[covered])


def resample_window_oracle(weight, inside, src_transform, dst_transform, shape_hw):
    """The eager window resample the shard path used, kept here as the oracle.

    Containment nearest, ramp bilinear, shares renormalised on the resampled
    containment. `composite.lazy_weights` does the same thing block by block
    on the workers, and this is what pins it.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    height, width = shape_hw
    n = weight.shape[0]
    out = np.zeros((n, height, width), dtype="float32")
    cover = np.zeros((n, height, width), dtype="uint8")
    for j in range(n):
        for source, destination, method in (
            (weight[j], out[j], Resampling.bilinear),
            (inside[j].astype("uint8"), cover[j], Resampling.nearest),
        ):
            reproject(
                source=source,
                destination=destination,
                src_transform=src_transform,
                src_crs=destripe.GRID_CRS,
                dst_transform=dst_transform,
                dst_crs=destripe.GRID_CRS,
                resampling=method,
            )
    covered = cover.astype(bool)
    out[~covered] = 0.0
    total = out.sum(axis=0)
    out /= np.where(total > 0, total, np.float32(1.0))
    k = covered.sum(axis=0)
    degenerate = (total <= 0) & (k > 0)
    if degenerate.any():
        for j in range(n):
            sel = degenerate & covered[j]
            out[j][sel] = 1.0 / k[sel]
    return out


def lazy_field(masks, dst_bbox, factor: int, chunk: int):
    """`composite.lazy_weights` over `dst_bbox` at `factor` x the swath grid."""
    import dask
    from odc.geo.geobox import GeoBox

    import composite

    paths, weight, inside = destripe.path_weights(masks, grid_transform())
    prep = destripe.Prep(
        tile="T",
        bbox=BBOX,
        pixels_per_degree=CELLS_PER_DEGREE,
        swath_factor=1,
        paths=paths,
        weight=weight,
        inside=inside,
        offset={},
        n_valid={},
        meta={},
    )
    w, s, e, n = dst_bbox
    ppd = CELLS_PER_DEGREE * factor
    shape_hw = (int(round((n - s) * ppd)), int(round((e - w) * ppd)))
    dst = GeoBox(shape_hw, transform_for(dst_bbox, ppd), destripe.GRID_CRS)
    lazy = composite.lazy_weights(prep, dst, chunk=chunk, dims=("y", "x"))
    assert lazy.chunks[0] == (len(paths),), "the path axis must be one chunk"
    assert all(c <= chunk for c in lazy.chunks[1] + lazy.chunks[2])
    with dask.config.set(scheduler="sync"):
        got = lazy.values
    want = resample_window_oracle(
        weight, inside, grid_transform(), dst.affine, shape_hw
    )
    return got, want


class TestTheLazyWeights:
    """`composite.lazy_weights` against the eager window resample it replaces."""

    def test_a_window_still_sums_to_one(self):
        got, _ = lazy_field(two_paths(), (-2.0, 0.0, 2.0, 1.0), 4, chunk=16)
        total = got.sum(axis=0)
        assert np.allclose(total[total > 0], 1.0, atol=1e-6)

    def test_it_matches_the_eager_resample_across_block_edges(self):
        """Blocks of 16 px over a 96 px field: five interior block edges."""
        got, want = lazy_field(two_paths(), (-2.0, 0.0, 2.0, 1.0), 4, chunk=16)
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 0.013

    def test_it_never_gives_weight_to_a_path_that_does_not_cover(self):
        masks = two_paths()
        masks[EAST][:] = False
        masks[EAST][:, 50:] = True
        got, want = lazy_field(masks, BBOX, 4, chunk=40)
        # Column 49 of the swath grid is columns 196 to 199 here, and the east
        # path does not reach it. Bilinear alone would leak a share across.
        assert not got[1, :, :196].any()
        assert not want[1, :, :196].any()


def item(scene_id: str, stamp: str, path: str, row: str = "030") -> dict:
    return {
        "properties": {
            "datetime": stamp,
            "landsat:scene_id": scene_id,
            "landsat:wrs_path": path,
            "landsat:wrs_row": row,
        }
    }


class TestJoiningOnTime:
    """Sub-second stamps throughout. A whole-second axis hides this bug class."""

    ITEMS = [
        item("A", "2021-01-05T14:02:11.123456Z", "228"),
        item("B", "2021-01-13T14:02:33.987654Z", "229"),
        item("C", "2021-02-06T14:02:12.500000Z", "228"),
    ]

    def test_labels_follow_the_stack_not_the_item_order(self):
        # The stack arrives sorted by datetime, and thinned: B was rejected.
        times = np.array(
            ["2021-02-06T14:02:12.500000", "2021-01-05T14:02:11.123456"],
            dtype="datetime64[ns]",
        )
        labels = destripe.align_to_time(
            self.ITEMS, times, value_of=destripe.path_of, what="WRS path", dtype=object
        )
        assert list(labels) == ["228", "228"]

    def test_a_step_no_item_accounts_for_is_refused(self):
        times = np.array(["2021-03-01T14:00:00.000000"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="no item"):
            destripe.align_to_time(
                self.ITEMS,
                times,
                value_of=destripe.path_of,
                what="WRS path",
                dtype=object,
            )

    def test_two_scenes_disagreeing_at_one_stamp_are_refused(self):
        clash = [
            item("A", "2021-01-05T14:02:11.123456Z", "228"),
            item("B", "2021-01-05T14:02:11.123456Z", "229"),
        ]
        times = np.array(["2021-01-05T14:02:11.123456"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="disagree"):
            destripe.align_to_time(
                clash, times, value_of=destripe.path_of, what="WRS path", dtype=object
            )

    def test_the_sub_second_part_of_a_stamp_survives_the_join(self):
        times = np.array(["2021-01-05T14:02:11.123456"], dtype="datetime64[ns]")
        labels = destripe.align_to_time(
            self.ITEMS, times, value_of=destripe.path_of, what="WRS path", dtype=object
        )
        assert list(labels) == ["228"]

    def test_the_quad_carries_both_halves_and_keeps_its_padding(self):
        assert destripe.quad_of(self.ITEMS[0]) == ("228", "030")
        assert destripe.path_of(item("X", "2021-01-01T00:00:00Z", "007")) == "007"


class TestTheLaneMedian:
    """`destripe.nan_median` is `np.nanmedian(axis=0)` to the bit, in a fast layout."""

    @pytest.mark.parametrize(
        ("depth", "shape", "share"),
        [
            (1, (5, 7), 1.0),
            (2, (5, 7), 0.5),
            (67, (24, 24), 0.7),  # odd depth, mixed odd and even counts
            (68, (24, 24), 0.7),  # even depth
            (300, (8, 8), 0.3),
        ],
    )
    def test_it_is_bit_identical_to_numpy(self, depth, shape, share):
        a = TestTheVectorisedPercentile.stack(depth, shape, share, seed=depth)
        a[:, 0, 0] = np.nan  # a pixel nothing observed
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            want = np.nanmedian(a, axis=0).astype("float32")
        got = destripe.nan_median(a)
        assert got.dtype == np.float32
        assert got.shape == want.shape
        np.testing.assert_array_equal(np.isnan(want), np.isnan(got))
        finite = np.isfinite(want)
        assert np.array_equal(want[finite].view("uint32"), got[finite].view("uint32"))

    def test_the_climatology_uses_it_and_matches_numpy(self):
        months = np.array([1, 1, 1, 7, 7, 7, 7, 12])
        stack = TestTheVectorisedPercentile.stack(8, (6, 6), 0.8, seed=8)
        planes, ref = destripe.month_climatology(stack, months)
        assert list(planes) == [1, 7, 12]
        for i, month in enumerate(planes):
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                want = np.nanmedian(stack[months == month], axis=0)
            np.testing.assert_array_equal(np.nan_to_num(ref[i]), np.nan_to_num(want))
            np.testing.assert_array_equal(np.isnan(ref[i]), np.isnan(want))

    def test_it_does_not_touch_its_input(self):
        a = TestTheVectorisedPercentile.stack(20, (4, 4), 0.9, seed=5)
        before = a.copy()
        destripe.nan_median(a)
        np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(before))

    def test_a_stack_with_no_scenes_is_all_nodata(self):
        got = destripe.nan_median(np.empty((0, 3, 4), dtype="float32"))
        assert got.shape == (3, 4)
        assert np.isnan(got).all()


class TestTheVectorisedPercentile:
    """`destripe.nan_percentile` is numpy's rule without numpy's per-pixel loop."""

    @staticmethod
    def stack(depth: int, shape, share: float, seed: int):
        rng = np.random.default_rng(seed)
        a = rng.normal(45.0, 6.0, (depth, *shape)).astype("float32")
        a[rng.random(a.shape) > share] = np.nan
        return a

    @pytest.mark.parametrize(
        ("depth", "shape", "share"),
        [(1, (5, 7), 1.0), (2, (5, 7), 0.5), (40, (24, 24), 0.6), (300, (8, 8), 0.3)],
    )
    def test_it_matches_numpy_on_a_block(self, depth, shape, share):
        a = self.stack(depth, shape, share, seed=depth)
        a[:, 0, 0] = np.nan  # a pixel nothing observed
        with np.errstate(all="ignore"):
            want = np.nanpercentile(a, 95, axis=0).astype("float32")
        got = destripe.nan_percentile(a, 95)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(np.isfinite(want), np.isfinite(got))
        finite = np.isfinite(want)
        assert np.abs(want[finite] - got[finite]).max() < 1e-4

    def test_it_matches_numpy_on_a_pixel_list(self):
        """The fallback reduces `celsius[:, mask]`, a `(n, k)` slice."""
        a = self.stack(50, (9, 9), 0.7, seed=3)
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:5, 1:8] = True
        with np.errstate(all="ignore"):
            want = np.nanpercentile(a[:, mask], 95, axis=0).astype("float32")
        got = destripe.nan_percentile(a[:, mask], 95)
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, atol=1e-4, equal_nan=True)

    def test_other_quantiles_follow_the_same_rule(self):
        a = self.stack(30, (6, 6), 0.8, seed=11)
        for q in (0.0, 50.0, 95.0, 100.0):
            with np.errstate(all="ignore"):
                want = np.nanpercentile(a, q, axis=0).astype("float32")
            got = destripe.nan_percentile(a, q)
            np.testing.assert_allclose(got, want, atol=1e-4, equal_nan=True)

    def test_it_does_not_touch_its_input(self):
        a = self.stack(20, (4, 4), 0.9, seed=5)
        before = a.copy()
        destripe.nan_percentile(a, 95)
        np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(before))

    def test_a_stack_with_no_scenes_is_all_nodata(self):
        """A block no scene reaches. numpy answers NaN and so must this."""
        empty = np.empty((0, 3, 4), dtype="float32")
        got = destripe.nan_percentile(empty, 95)
        assert got.shape == (3, 4)
        assert np.isnan(got).all()

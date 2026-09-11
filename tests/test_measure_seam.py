"""The four-arm seam comparison: what it reads, and what it measures.

`measure_seam.py` is the script that decides whether the two corrections work.
It had no tests, so its metric, its validity intersection, and the claim in its
own docstring were all unchecked.

The claim is the one that matters here. "Four composites from one load" was
false: the loop called `process_shard` per arm, and `process_shard` reads. Four
reads of the same objects cost four times what the comparison measures, and the
load is 94% of a shard. `shard_lst_p95.load_shard` and `reduce_shard` split it,
and `test_the_four_arms_come_from_one_read` is what keeps the claim true.

Nothing here reads S3. `odc.stac.stac_load` is replaced by a fixture, the way
`tests/test_pipeline_paths.py` replaces it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import measure_seam  # noqa: E402
import shard_lst_p95  # noqa: E402
from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE  # noqa: E402
from test_pipeline_paths import (  # noqa: E402
    AREA,
    N_TIME,
    NX,
    NY,
    build_stack,
    corrected_items,
    correction_for,
)


@pytest.fixture
def counted_load(monkeypatch):
    """The fake loader, plus how many times anything asked it to read."""
    import odc.stac
    import pystac

    dataset = build_stack()[0]
    calls = {"n": 0}

    def _load(*_args, **_kwargs):
        calls["n"] += 1
        return dataset.copy(deep=True)

    monkeypatch.setattr(odc.stac, "stac_load", _load)
    monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
    return calls


def shard():
    return shard_lst_p95.Shard(0, 0, 0, 0, NY, NX, AREA)


def run_arms(items, correction_of):
    """What `measure_seam.main` does between the load and the metric."""
    data, parsed, load_s = shard_lst_p95.load_shard(
        shard(), items, "EPSG:4326", 1 / 3600, 1
    )
    return {
        name: shard_lst_p95.reduce_shard(
            shard(), data, parsed, load_s, correction_of(debias, feather)
        )
        for name, debias, feather in measure_seam.ARMS
    }


class TestOneRead:
    """The script's own claim about what it costs."""

    def test_the_four_arms_come_from_one_read(self, counted_load):
        stack = build_stack()
        items = corrected_items(stack)
        run_arms(items, lambda d, f: correction_for(stack))
        assert counted_load["n"] == 1

    def test_four_separate_shards_would_have_read_four_times(self, counted_load):
        # The measurement the fix is against, so the assertion above cannot
        # pass because the fixture stopped being called at all.
        stack = build_stack()
        items = corrected_items(stack)
        for _ in range(4):
            shard_lst_p95.process_shard(
                shard(), items, "EPSG:4326", 1 / 3600, read_threads=1
            )
        assert counted_load["n"] == 4

    def test_reducing_twice_over_one_load_matches_reducing_once_each(
        self, counted_load
    ):
        """The contract the split must not break.

        `apply_to_stack` de-biases in place. An arm handed the previous arm's
        array would reduce a stack already shifted, and the difference would
        read as a correction that worked.
        """
        stack = build_stack()
        items = corrected_items(stack)

        shared = run_arms(items, lambda d, f: correction_for(stack))
        separate = {
            name: shard_lst_p95.process_shard(
                shard(),
                items,
                "EPSG:4326",
                1 / 3600,
                read_threads=1,
                correction=correction_for(stack),
            )
            for name, _d, _f in measure_seam.ARMS
        }
        for name in shared:
            np.testing.assert_array_equal(
                shared[name]["lst_p95"], separate[name]["lst_p95"], err_msg=name
            )

    def test_each_arm_sees_the_scene_count_the_load_carried(self, counted_load):
        stack = build_stack()
        items = corrected_items(stack)
        for out in run_arms(items, lambda d, f: correction_for(stack)).values():
            assert out["n_scenes"] == N_TIME


class TestTheMetric:
    """A seam is a jump that follows satellite geometry, not ground texture."""

    def edges(self, columns):
        weight = np.zeros((2, NY, NX), dtype="float32")
        weight[0, :, :columns] = 1.0
        weight[1, :, columns:] = 1.0
        return measure_seam.boundary_edges(weight)

    def test_a_shard_inside_one_swath_has_no_boundary(self):
        weight = np.ones((1, NY, NX), dtype="float32")
        down, right = measure_seam.boundary_edges(weight)
        assert not down.any()
        assert not right.any()
        field = np.zeros((NY, NX), dtype="float32")
        assert measure_seam.seam_step(field, down, right) is None

    def test_the_boundary_is_where_the_covering_set_changes(self):
        down, right = self.edges(2)
        # The two paths meet between column 1 and column 2.
        assert right[:, 1].all()
        assert not right[:, 0].any()
        assert not down.any()

    def test_a_planted_step_scores_above_the_texture_floor(self):
        down, right = self.edges(2)
        field = np.zeros((NY, NX), dtype="float32")
        field[:, 2:] = 6.0
        step = measure_seam.seam_step(field, down, right)
        assert step is not None
        assert step["on_boundary_mean_c"] == pytest.approx(6.0)
        assert step["elsewhere_median_c"] == pytest.approx(0.0)
        assert step["excess_c"] == pytest.approx(6.0)

    def test_texture_everywhere_is_not_a_seam(self):
        """A field with sharp edges everywhere has large steps everywhere.

        Subtracting the floor is what separates the artifact from the signal.
        Without it, agricultural ground would score as a seam.
        """
        down, right = self.edges(2)
        rng = np.random.default_rng(20260911)
        field = rng.uniform(0.0, 12.0, (NY, NX)).astype("float32")
        step = measure_seam.seam_step(field, down, right)
        assert step is not None
        assert abs(step["excess_c"]) < 6.0


class TestTheValidityIntersection:
    """No arm is scored on a pixel another arm lacks."""

    def fields(self):
        base = np.full((NY, NX), 30.0, dtype="float32")
        return {name: base.copy() for name, _d, _f in measure_seam.ARMS}

    def test_a_pixel_one_arm_lost_is_scored_for_none_of_them(self):
        down, right = np.zeros((NY - 1, NX), bool), np.ones((NY, NX - 1), bool)
        fields = self.fields()
        fields["both"][0, 0] = np.nan
        summary, _rows = measure_seam.compare(fields, down, right)
        assert summary["n_px"] == NY * NX
        assert summary["n_valid_px"] == NY * NX - 1

    def test_the_arms_are_compared_on_the_same_pixels(self):
        down, right = np.zeros((NY - 1, NX), bool), np.ones((NY, NX - 1), bool)
        fields = self.fields()
        # One arm reads warm everywhere it survives, and loses one pixel. The
        # mean has to be taken without that pixel for every arm, or the two
        # means differ for a reason that is not the correction.
        fields["pooled"][:] = 40.0
        fields["both"][0, 0] = np.nan
        _summary, rows = measure_seam.compare(fields, down, right)
        assert rows["pooled"]["mean_c"] == pytest.approx(40.0)
        assert rows["both"]["mean_c"] == pytest.approx(30.0)

    def test_an_arm_that_flattened_the_field_reports_it(self):
        """Removing the seam and the signal together is the failure to catch."""
        down, right = np.zeros((NY - 1, NX), bool), np.ones((NY, NX - 1), bool)
        fields = self.fields()
        rng = np.random.default_rng(7)
        fields["pooled"] = rng.uniform(10.0, 50.0, (NY, NX)).astype("float32")
        fields["both"][:] = float(fields["pooled"].mean())
        _summary, rows = measure_seam.compare(fields, down, right)
        assert rows["pooled"]["variance_retained"] == pytest.approx(1.0)
        assert rows["both"]["variance_retained"] == pytest.approx(0.0)


class TestDecoding:
    def test_nodata_becomes_nan_rather_than_a_temperature(self):
        # DN 0 decodes to -50 C under the encoding, which is a believable
        # value. Reading it as one would drag every mean here downward.
        dn = np.array([[LST_NODATA_DN, 4000]], dtype="uint16")
        out = measure_seam.decode(dn)
        assert np.isnan(out[0, 0])
        assert out[0, 1] == pytest.approx(4000 * LST_SCALE + LST_OFFSET)

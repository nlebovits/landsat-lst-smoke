"""The final-output validity rule, in the kernel that both engines run.

`tests/test_lst_qa.py` checks `supported_output` as a predicate, at the exact
bound. This checks that `composite.reduce_block` applies it, on the arrays the
graph and the fused path both hand it, and that it leaves everything else where
it was.

Four defects these guard against.

A P95 over four scenes shipping as a five-year percentile. The composite is a
95th percentile over five years, and a pixel Landsat saw four times carries an
order statistic over four scenes. MEASURED across five tiles, the floor of 5
costs between 0.0003% and 0.804% of valid land, so the tail it cuts is thin.

An impossible temperature shipping because it was possible per observation.
`lst_qa.in_trusted_range` bounds each observation to [-50, 80] C, and
`destripe.subtract_offsets` then moves the decoded value. MEASURED on N30E075,
207 published pixels reach 80 C or hotter that way. `TestThePhysicalBounds`
builds that exact sequence rather than testing the bound in isolation: every
observation it feeds the kernel passes the per-observation rule, and the
monthly counts are asserted to prove it.

The rule reaching pixels it has no business reaching. An ordinary
well-supported pixel has to encode to the DN it encoded to before this rule
existed, and `TestOrdinaryOutputIsUntouched` asserts that against
`lst_qa.encode_celsius` on the same values.

`qa_count` following the temperature out. The count is the only evidence a
consumer has for which rule removed a pixel, so the rule leaves it standing. A
nodata temperature beside a count of 4 says the floor reached it. A count of 0
beside nodata says the water rule did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import composite
from lst_qa import (
    LST_NODATA_DN,
    LST_OUTPUT_MAX_C,
    LST_OUTPUT_MIN_C,
    LST_SCALE,
    LST_VALID_MAX_C,
    LWIR_OFFSET_C,
    LWIR_SCALE,
    MIN_TOTAL_OBSERVATIONS,
    encode_celsius,
    to_celsius,
)

#: QA_PIXEL bit 6 is "clear", and nothing in bits 1 to 5 is set, so the
#: observation is usable. `tests/test_lst_qa.py` fixes that reading.
QA_CLEAR = 0b1000000

#: The block these tests reduce. Three pixels is enough to carry a sparse
#: pixel, a pixel exactly on the floor, and a well-supported one at once.
NY, NX = 1, 3

#: An ordinary clear-sky land temperature, far inside every bound.
WARM_C = 28.4


def raw_dn(celsius: float) -> int:
    """The source thermal DN that decodes nearest to `celsius`."""
    return int(round((celsius - LWIR_OFFSET_C) / LWIR_SCALE))


def decode(dn: int) -> float:
    """What one source DN decodes to, in Celsius, as the kernel decodes it."""
    return float(to_celsius(np.uint16(dn)))


def out_dn(celsius: float) -> int:
    """The output DN that `encode_celsius` writes for `celsius`."""
    return int(encode_celsius(np.array([celsius], dtype="float32"))[0])


#: The hottest source DN the per-observation rule accepts. The source grid is
#: 0.0034 C, so no DN decodes to exactly 80 C: the next one up decodes to
#: 80.0014 C and `in_trusted_range` rejects it. Every hot case below observes
#: this DN, so nothing it tests can be confused with the per-observation rule.
HOT_DN = max(d for d in range(59_000, 60_000) if decode(d) <= LST_VALID_MAX_C)
HOT_OBSERVED_C = decode(HOT_DN)


def reduce(lwir, qa, *, offset=None, month=None, feather=False):
    """`composite.reduce_block` on one block, pooled unless asked otherwise.

    The kernel takes its bands `(y, x, time)`, because `apply_ufunc` hands core
    dimensions last. The arrays here are built time-major, which is how they
    read, and moved on the way in.

    Returns:
        `(dn, counts)`. `dn` is `(y, x)` uint16 and `counts` is `(12, y, x)`
        uint8, moved back from the kernel's `(y, x, month)`.
    """
    n = lwir.shape[0]
    offset = np.zeros(n, dtype="float32") if offset is None else np.asarray(offset)
    month = np.ones(n, dtype="int16") if month is None else np.asarray(month)
    dn, counts, _fallback = composite.reduce_block(
        np.moveaxis(lwir, 0, -1),
        np.moveaxis(qa, 0, -1),
        offset,
        np.ones(n, dtype=bool),
        np.zeros(n, dtype="int16"),
        month,
        np.ones((NY, NX, 1), dtype="float32"),
        n_paths=1 if feather else 0,
        feather=feather,
        emit_pooled=False,
    )
    return dn, np.moveaxis(counts, -1, 0)


def stack(depths, dn=None):
    """A block where pixel `x` holds `depths[x]` clear observations of one DN.

    Every scene covers the whole block, because a scene that reaches no pixel
    of a block is cut before the decode and would shorten all three pixels at
    once. The scenes a pixel did not see carry the source fill value instead,
    which `lst_qa.not_fill` rejects per pixel.
    """
    dn = raw_dn(WARM_C) if dn is None else dn
    depth = max(depths)
    lwir = np.zeros((depth, NY, NX), dtype="uint16")
    qa = np.full((depth, NY, NX), QA_CLEAR, dtype="uint16")
    for x, count in enumerate(depths):
        lwir[:count, 0, x] = dn
    return lwir, qa


class TestTheObservationFloor:
    """`MIN_TOTAL_OBSERVATIONS`, applied to the summed monthly counts."""

    DEPTHS = (MIN_TOTAL_OBSERVATIONS - 1, MIN_TOTAL_OBSERVATIONS, 40)

    @pytest.fixture
    def reduced(self):
        return reduce(*stack(self.DEPTHS))

    def test_a_pixel_below_the_floor_becomes_nodata(self, reduced):
        dn, _ = reduced
        assert dn[0, 0] == LST_NODATA_DN

    def test_a_pixel_on_the_floor_survives(self, reduced):
        dn, _ = reduced
        assert dn[0, 1] == out_dn(decode(raw_dn(WARM_C)))

    def test_a_well_supported_pixel_survives(self, reduced):
        dn, _ = reduced
        assert dn[0, 2] == out_dn(decode(raw_dn(WARM_C)))

    def test_the_count_layer_survives_the_floor(self, reduced):
        # The regression that matters. Without the count, a consumer cannot
        # tell the pixel the floor screened from the pixel nothing observed.
        _, counts = reduced
        assert list(counts.sum(axis=0)[0]) == list(self.DEPTHS)

    def test_the_floor_reads_the_published_counts(self, reduced):
        # The rule sums `qa_count`, so a consumer recomputing it from the two
        # published bands reaches the answer the run reached.
        dn, counts = reduced
        total = counts.sum(axis=0, dtype="uint16")
        sparse = total < MIN_TOTAL_OBSERVATIONS
        assert (dn[sparse] == LST_NODATA_DN).all()
        assert (dn[~sparse] != LST_NODATA_DN).all()

    def test_a_pixel_nothing_observed_is_nodata_and_counts_zero(self):
        dn, counts = reduce(*stack((0, 8, 8)))
        assert dn[0, 0] == LST_NODATA_DN
        assert int(counts.sum(axis=0)[0, 0]) == 0

    def test_the_floor_counts_across_months_rather_than_within_one(self):
        # Four observations in January and one in June is five observations.
        # A per-month floor would drop this pixel; this rule keeps it.
        lwir, qa = stack((5, 5, 5))
        month = np.array([1, 1, 1, 1, 6], dtype="int16")
        dn, counts = reduce(lwir, qa, month=month)
        assert list(counts[:, 0, 0]) == [4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        assert dn[0, 0] != LST_NODATA_DN

    def test_the_floor_is_the_only_reason_these_pixels_went(self):
        # The temperature is ordinary, so nothing but the count can explain the
        # nodata. Without this, a broken decode would pass the class above.
        dn, _ = reduce(*stack(self.DEPTHS))
        assert out_dn(decode(raw_dn(WARM_C))) != LST_NODATA_DN
        assert dn[0, 0] == LST_NODATA_DN


class TestThePhysicalBounds:
    """`LST_OUTPUT_MIN_C` and `LST_OUTPUT_MAX_C`, on the output side.

    Each case drives the percentile past a bound with a scene offset, which is
    the mechanism that produced the published outliers. Every observation that
    enters the block is inside `[-50, 80]` C and passes `in_trusted_range`, and
    each test asserts the monthly counts to prove it, so a bound checked before
    the reduction cannot account for any removal here.
    """

    #: Deep enough that the observation floor never explains a nodata.
    DEPTH = 40

    #: How far past a bound a case reaches, in Celsius. Five DN steps, which is
    #: far above the float32 noise of the decode and the subtraction.
    MARGIN = 5 * LST_SCALE

    def reduce_shifted(self, observed_dn, target_c):
        """`DEPTH` observations of `observed_dn`, shifted to land on `target_c`.

        Returns `(dn, counts)` for the block.
        """
        offset = np.full(self.DEPTH, decode(observed_dn) - target_c, dtype="float32")
        lwir, qa = stack((self.DEPTH,) * NX, dn=observed_dn)
        return reduce(lwir, qa, offset=offset)

    def test_a_value_below_the_cold_bound_becomes_nodata(self):
        dn, counts = self.reduce_shifted(raw_dn(10.0), LST_OUTPUT_MIN_C - self.MARGIN)
        assert (counts.sum(axis=0) == self.DEPTH).all()
        assert (dn == LST_NODATA_DN).all()

    def test_a_value_inside_the_cold_bound_survives(self):
        target = LST_OUTPUT_MIN_C + self.MARGIN
        dn, counts = self.reduce_shifted(raw_dn(10.0), target)
        assert (counts.sum(axis=0) == self.DEPTH).all()
        assert (dn == out_dn(target)).all()

    def test_a_value_above_the_hot_bound_becomes_nodata(self):
        dn, counts = self.reduce_shifted(HOT_DN, LST_OUTPUT_MAX_C + self.MARGIN)
        assert (counts.sum(axis=0) == self.DEPTH).all()
        assert (dn == LST_NODATA_DN).all()

    def test_a_value_inside_the_hot_bound_survives(self):
        target = LST_OUTPUT_MAX_C - self.MARGIN
        dn, counts = self.reduce_shifted(HOT_DN, target)
        assert (counts.sum(axis=0) == self.DEPTH).all()
        assert (dn == out_dn(target)).all()

    def test_the_hottest_observation_is_below_the_bound_before_any_offset(self):
        # The premise of the two hot cases. Without an offset this block is
        # publishable, so the offset is what the ceiling is catching.
        lwir, qa = stack((self.DEPTH,) * NX, dn=HOT_DN)
        dn, counts = reduce(lwir, qa)
        assert HOT_OBSERVED_C <= LST_OUTPUT_MAX_C
        assert (counts.sum(axis=0) == self.DEPTH).all()
        assert (dn == out_dn(HOT_OBSERVED_C)).all()

    def test_the_hot_bound_needs_no_emissivity_gap(self):
        """Case 8 of the ticket, and it holds by construction.

        `reduce_block` takes no gap plane and reads no ASTER GED artifact, so
        an extreme-hot pixel goes wherever it sits. The withdrawn rule removed
        one only inside the gap region, and MEASURED on N30E075 all 207 pixels
        at or above 80 C fell outside that region and its buffer.
        """
        import inspect

        names = inspect.signature(composite.reduce_block).parameters
        assert "gap" not in names
        assert "hot_dn" not in names
        dn, _ = self.reduce_shifted(HOT_DN, 95.0)
        assert (dn == LST_NODATA_DN).all()

    def test_evidence_does_not_excuse_an_impossible_value(self):
        # 255 observations is above the median of every tile composited so far.
        offset = np.full(255, HOT_OBSERVED_C - 95.0, dtype="float32")
        lwir, qa = stack((255,) * NX, dn=HOT_DN)
        dn, counts = reduce(lwir, qa, offset=offset)
        assert (counts.sum(axis=0) == 255).all()
        assert (dn == LST_NODATA_DN).all()

    def test_the_count_layer_survives_the_bounds(self):
        _, counts = self.reduce_shifted(HOT_DN, 95.0)
        assert (counts.sum(axis=0) == self.DEPTH).all()


class TestOrdinaryOutputIsUntouched:
    """The rule has to be invisible to every pixel it does not remove."""

    def test_a_well_supported_block_encodes_as_it_did_before(self):
        rng = np.random.default_rng(3)
        wanted = rng.uniform(15.0, 45.0, size=NX)
        source = [raw_dn(float(c)) for c in wanted]
        lwir = np.zeros((60, NY, NX), dtype="uint16")
        qa = np.full((60, NY, NX), QA_CLEAR, dtype="uint16")
        for x, value in enumerate(source):
            lwir[:, 0, x] = value

        dn, counts = reduce(lwir, qa)
        # What the kernel wrote before this rule existed: the percentile of a
        # constant series is that constant, encoded.
        expected = encode_celsius(
            np.array([[decode(value) for value in source]], dtype="float32")
        )
        np.testing.assert_array_equal(dn, expected)
        assert (counts.sum(axis=0) == 60).all()
        assert (dn != LST_NODATA_DN).all()

    def test_an_all_fill_block_is_nodata_as_it_was(self):
        lwir = np.zeros((10, NY, NX), dtype="uint16")
        qa = np.full((10, NY, NX), QA_CLEAR, dtype="uint16")
        dn, counts = reduce(lwir, qa)
        assert (dn == LST_NODATA_DN).all()
        assert (counts == 0).all()

    def test_a_block_of_cloud_is_nodata_and_counts_zero(self):
        # Every observation present and every one rejected by QA. The floor
        # cannot be what removed these, and the answer is the same either way.
        lwir, qa = stack((20,) * NX)
        qa[:] = QA_CLEAR | 0b1000  # bit 3, cloud
        dn, counts = reduce(lwir, qa)
        assert (dn == LST_NODATA_DN).all()
        assert (counts == 0).all()

    def test_the_rule_runs_under_the_feathered_branch_too(self):
        # The same block through `destripe.feathered_percentile`, which is the
        # branch a production run takes. The rule sits after both branches.
        dn, counts = reduce(*stack((MIN_TOTAL_OBSERVATIONS - 1, 40, 40)), feather=True)
        assert dn[0, 0] == LST_NODATA_DN
        assert dn[0, 1] == out_dn(decode(raw_dn(WARM_C)))
        assert int(counts.sum(axis=0)[0, 0]) == MIN_TOTAL_OBSERVATIONS - 1


class TestTheRuleIsStatedOnce:
    """One implementation of the numeric rule, as `AGENTS.md` requires."""

    #: Deep enough that the observation floor never explains a nodata.
    DEPTH = 40

    def test_the_kernel_calls_the_predicate_rather_than_restating_it(self):
        import inspect

        source = inspect.getsource(composite.reduce_block)
        assert "supported_output(" in source
        for literal in ("MIN_TOTAL_OBSERVATIONS", "LST_OUTPUT_MIN_C", "80.0"):
            assert literal not in source

    def test_the_bounds_are_not_applied_to_the_diagnostic_pooled_band(self):
        """`--emit-pooled` exists to compare pooled against feathered.

        Filtering the comparison band would hide the difference it is there to
        show, so the rule reaches `lst_p95` alone. 95 C encodes perfectly well
        on the output grid, so the pooled band carries it and `lst_p95` does
        not.
        """
        offset = np.full(self.DEPTH, HOT_OBSERVED_C - 95.0, dtype="float32")
        lwir, qa = stack((self.DEPTH,) * NX, dn=HOT_DN)
        n = lwir.shape[0]
        dn, _counts, _fallback, pooled = composite.reduce_block(
            np.moveaxis(lwir, 0, -1),
            np.moveaxis(qa, 0, -1),
            offset,
            np.ones(n, dtype=bool),
            np.zeros(n, dtype="int16"),
            np.ones(n, dtype="int16"),
            np.ones((NY, NX, 1), dtype="float32"),
            n_paths=0,
            feather=False,
            emit_pooled=True,
        )
        assert out_dn(95.0) != LST_NODATA_DN
        assert (pooled == out_dn(95.0)).all()
        assert (dn == LST_NODATA_DN).all()

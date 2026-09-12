"""What counts as a usable LST observation, and what the encoder does with it.

Both P95 paths call `lst_qa`, so these tests are the contract. They cover the
four rules the earlier `nlebovits/landsat-lst` pipeline established and this
one had dropped: QA_PIXEL bits 1 to 5, the raw fill value, the physical range
in Celsius, and nodata rather than clipping on the way out.

Two cases exist because the obvious implementation passes without them. A test
of the combined bit mask alone cannot see a shifted or missing bit, so each bit
is also tested on its own. An exact `dn != 0` test cannot see the small nonzero
DN that reprojection leaves along a scene edge, so that value is tested by
itself as well.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lst_qa import (  # noqa: E402
    LST_MIN_TRUSTED_C,
    LST_MIN_TRUSTED_DN,
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    LST_VALID_MAX_C,
    LST_VALID_MIN_C,
    LWIR_FILL_DN,
    LWIR_OFFSET_C,
    LWIR_SCALE,
    QA_EXCLUDED_BIT_NUMBERS,
    QA_EXCLUDED_BITS,
    encode_celsius,
    in_trusted_range,
    masked_celsius,
    qa_clear,
    to_celsius,
)

# QA_PIXEL bit 6 is "clear" and bit 0 is "fill". Neither disqualifies a pixel,
# so 0b1000000 is a realistic clear-sky QA value that the mask has to keep.
QA_CLEAR = 0b1000000
# A DN that decodes to about 30 C: an ordinary clear-sky land observation.
DN_WARM = 45099


def dn_of(celsius: float) -> int:
    """The raw thermal DN whose decoded value is nearest `celsius`."""
    return int(round((celsius - LWIR_OFFSET_C) / LWIR_SCALE))


def decode(dn: int) -> float:
    return float(to_celsius(np.array([dn], dtype="uint16"))[0])


def one(dn, qa):
    """Mask a single pixel. Returns its Celsius value and its validity."""
    lst, valid = masked_celsius(
        np.array([dn], dtype="uint16"), np.array([qa], dtype="uint16")
    )
    return float(lst[0]), bool(valid[0])


class TestQaBits:
    """Each excluded bit, alone. The combined constant hides a shifted bit."""

    def test_the_mask_covers_exactly_bits_one_to_five(self):
        assert QA_EXCLUDED_BIT_NUMBERS == (1, 2, 3, 4, 5)
        assert QA_EXCLUDED_BITS == 0b111110
        assert QA_EXCLUDED_BITS == 62

    @pytest.mark.parametrize(
        ("bit", "meaning"),
        [
            (1, "dilated cloud"),
            (2, "cirrus"),
            (3, "cloud"),
            (4, "cloud shadow"),
            (5, "snow"),
        ],
    )
    def test_one_set_bit_rejects_a_good_observation(self, bit, meaning):
        value, valid = one(DN_WARM, QA_CLEAR | (1 << bit))
        assert np.isnan(value), f"bit {bit} ({meaning}) did not reject the pixel"
        assert valid is False

    def test_a_clear_qa_value_keeps_the_observation(self):
        value, valid = one(DN_WARM, QA_CLEAR)
        assert valid is True
        assert value == pytest.approx(decode(DN_WARM), abs=1e-4)

    def test_qa_zero_keeps_the_observation(self):
        _, valid = one(DN_WARM, 0)
        assert valid is True

    @pytest.mark.parametrize(
        "bits",
        [(1, 3), (2, 5), (3, 4), (1, 2, 3), (4, 5), (1, 2, 3, 4, 5)],
    )
    def test_combinations_of_excluded_bits_reject(self, bits):
        qa = QA_CLEAR
        for bit in bits:
            qa |= 1 << bit
        value, valid = one(DN_WARM, qa)
        assert np.isnan(value)
        assert valid is False

    @pytest.mark.parametrize("bit", [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    def test_bits_outside_one_to_five_do_not_reject(self, bit):
        # Bit 0 is fill, 6 is clear, 7 is water, and 8 upwards are confidence
        # pairs. This task masks none of them.
        _, valid = one(DN_WARM, 1 << bit)
        assert valid is True

    def test_qa_clear_is_elementwise(self):
        qa = np.array([QA_CLEAR, QA_CLEAR | 0b1000, 0, 0b100000], dtype="uint16")
        assert list(np.asarray(qa_clear(qa))) == [True, False, True, False]


class TestTheDecodeIsOneCopy:
    """`to_celsius` scales and offsets in place, and must not move a value.

    The block stack is the largest array the pipeline holds, so the decode
    keeps one float32 copy of it rather than three. The arithmetic is the same
    two float32 operations in the same order, so the result is bit-identical
    to the expression form, and this asserts that rather than trusting it.
    """

    @staticmethod
    def expression_form(dn):
        return dn.astype("float32") * np.float32(LWIR_SCALE) + np.float32(LWIR_OFFSET_C)

    @pytest.mark.parametrize("shape", [(1,), (7, 5), (4, 8, 8)])
    def test_it_is_bit_identical_to_the_expression_form(self, shape):
        rng = np.random.default_rng(sum(shape))
        dn = rng.integers(0, 65536, size=shape, dtype="uint16")
        got = to_celsius(dn)
        want = self.expression_form(dn)
        assert got.dtype == np.float32
        assert np.array_equal(got.view("uint32"), want.view("uint32"))

    def test_it_does_not_touch_its_input(self):
        dn = np.array([[100, 20000, 65535]], dtype="uint16")
        before = dn.copy()
        to_celsius(dn)
        np.testing.assert_array_equal(dn, before)

    def test_a_numpy_scalar_still_decodes(self):
        """`in_trusted_range(to_celsius(np.uint16(5)))` is asserted below."""
        assert float(to_celsius(np.uint16(5))) == float(
            self.expression_form(np.uint16(5))
        )

    def test_a_dataarray_decodes_to_the_same_bits(self):
        rng = np.random.default_rng(11)
        dn = rng.integers(0, 65536, size=(3, 4, 4), dtype="uint16")
        lazy = xr.DataArray(dn, dims=("time", "y", "x"))
        got = to_celsius(lazy)
        assert np.array_equal(
            got.values.view("uint32"), self.expression_form(dn).view("uint32")
        )
        np.testing.assert_array_equal(lazy.values, dn)


class TestFillAndPhysicalRange:
    def test_raw_fill_becomes_nan(self):
        value, valid = one(LWIR_FILL_DN, QA_CLEAR)
        assert np.isnan(value)
        assert valid is False

    def test_a_small_nonzero_dn_from_a_reprojected_edge_becomes_nan(self):
        # This is the case an exact `dn != 0` test cannot catch. Reprojection
        # interpolates between a valid DN and the fill beside it, so the edge
        # carries small nonzero values that decode near -124 C.
        for dn in (1, 2, 5, 37, 500):
            assert decode(dn) < LST_VALID_MIN_C
            value, valid = one(dn, QA_CLEAR)
            assert np.isnan(value), f"DN {dn} decodes to {decode(dn):.2f} C"
            assert valid is False

    def test_an_exact_fill_test_alone_would_miss_those_pixels(self):
        # Guards the reason the range check exists: without it, DN 5 survives.
        assert 5 != LWIR_FILL_DN
        assert not bool(np.asarray(in_trusted_range(to_celsius(np.uint16(5)))))

    def test_a_dn_above_the_hot_bound_becomes_nan(self):
        dn = dn_of(LST_VALID_MAX_C + 5.0)
        assert decode(dn) > LST_VALID_MAX_C
        value, valid = one(dn, QA_CLEAR)
        assert np.isnan(value)
        assert valid is False

    def test_the_cold_bound_is_inclusive(self):
        dn = min(d for d in range(1, 65536) if decode(d) >= LST_VALID_MIN_C)
        assert decode(dn) >= LST_VALID_MIN_C
        assert decode(dn - 1) < LST_VALID_MIN_C
        assert one(dn, QA_CLEAR)[1] is True
        assert one(dn - 1, QA_CLEAR)[1] is False

    def test_the_hot_bound_is_inclusive(self):
        dn = max(d for d in range(1, 65536) if decode(d) <= LST_VALID_MAX_C)
        assert decode(dn) <= LST_VALID_MAX_C
        assert decode(dn + 1) > LST_VALID_MAX_C
        assert one(dn, QA_CLEAR)[1] is True
        assert one(dn + 1, QA_CLEAR)[1] is False

    def test_an_ordinary_clear_sky_value_passes_through_unchanged(self):
        value, valid = one(DN_WARM, QA_CLEAR)
        assert valid is True
        assert LST_VALID_MIN_C <= value <= LST_VALID_MAX_C
        assert value == pytest.approx(30.0, abs=0.05)

    def test_non_finite_input_cannot_become_a_temperature(self):
        dn = np.array([np.nan, np.inf, -np.inf, float(DN_WARM)], dtype="float32")
        qa = np.full(4, QA_CLEAR, dtype="uint16")
        lst, valid = masked_celsius(dn, qa)
        assert list(np.asarray(valid)) == [False, False, False, True]
        assert np.isnan(lst[:3]).all()

    def test_the_valid_mask_and_the_nan_pattern_agree(self):
        rng = np.random.default_rng(0)
        dn = rng.integers(0, 65536, size=2000, dtype="uint16")
        qa = rng.integers(0, 4096, size=2000).astype("uint16")
        lst, valid = masked_celsius(dn, qa)
        assert np.array_equal(np.isfinite(lst), np.asarray(valid))


class TestReductionOrder:
    """The range filter has to run before the percentile, not after it."""

    def _stack(self):
        # Nine clear observations and three scene-edge artifacts, one pixel.
        good_c = [20.0, 22.0, 25.0, 28.0, 30.0, 33.0, 35.0, 38.0, 41.0]
        cold_dn = [1, 4, 300]  # all decode near -124 C
        dn = np.array(
            [[[dn_of(c)]] for c in good_c] + [[[d]] for d in cold_dn], dtype="uint16"
        )
        qa = np.full(dn.shape, QA_CLEAR, dtype="uint16")
        return dn, qa, good_c

    def test_p95_equals_the_percentile_of_the_valid_subset(self):
        dn, qa, good_c = self._stack()
        lst, _ = masked_celsius(dn, qa)
        p95 = np.nanpercentile(lst, 95, axis=0)

        kept = np.array([decode(dn_of(c)) for c in good_c], dtype="float32")
        assert p95[0, 0] == pytest.approx(float(np.percentile(kept, 95)), abs=1e-4)

    def test_filtering_after_the_percentile_gives_a_different_answer(self):
        # If the range check moved below nanpercentile, the cold samples would
        # still sit in the sample the percentile is drawn from. This asserts
        # the two orders disagree, so the test above cannot pass by accident.
        dn, qa, _ = self._stack()
        late = np.percentile(to_celsius(dn), 95, axis=0)

        lst, _ = masked_celsius(dn, qa)
        early = np.nanpercentile(lst, 95, axis=0)
        assert not np.isclose(float(late[0, 0]), float(early[0, 0]))

    def test_one_cold_sample_in_twenty_still_leaves_the_answer_clean(self):
        good = np.arange(20.0, 60.0, 2.0)  # 20 clear observations
        dn = np.array([[[dn_of(c)]] for c in good] + [[[3]]], dtype="uint16")
        qa = np.full(dn.shape, QA_CLEAR, dtype="uint16")
        lst, _ = masked_celsius(dn, qa)
        p95 = float(np.nanpercentile(lst, 95, axis=0)[0, 0])
        assert p95 == pytest.approx(
            float(np.percentile([decode(dn_of(c)) for c in good], 95)), abs=1e-3
        )

    def test_observation_counts_use_the_same_validity_rule(self):
        dn, qa, good_c = self._stack()
        _, valid = masked_celsius(dn, qa)
        assert int(valid.sum()) == len(good_c)


class TestEncoding:
    def test_an_all_invalid_time_series_encodes_to_nodata(self):
        dn = np.zeros((6, 2, 2), dtype="uint16")
        qa = np.full((6, 2, 2), QA_CLEAR, dtype="uint16")
        lst, valid = masked_celsius(dn, qa)
        assert int(valid.sum()) == 0
        with np.errstate(all="ignore"):
            p95 = np.nanpercentile(lst, 95, axis=0)
        out = encode_celsius(p95)
        assert out.dtype == np.uint16
        assert (out == LST_NODATA_DN).all()

    def test_observation_counts_are_zero_for_an_all_invalid_pixel(self):
        dn = np.zeros((6, 2, 2), dtype="uint16")
        qa = np.full((6, 2, 2), QA_CLEAR, dtype="uint16")
        _, valid = masked_celsius(dn, qa)
        assert list(valid.sum(axis=0).ravel()) == [0, 0, 0, 0]

    def test_nan_encodes_to_nodata(self):
        out = encode_celsius(np.array([np.nan], dtype="float32"))
        assert int(out[0]) == LST_NODATA_DN

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinity_encodes_to_nodata(self, value):
        out = encode_celsius(np.array([value], dtype="float32"))
        assert int(out[0]) == LST_NODATA_DN

    def test_an_out_of_range_cold_value_is_not_clipped_to_the_floor(self):
        out = int(encode_celsius(np.array([-124.15], dtype="float32"))[0])
        assert out == LST_NODATA_DN
        assert out != 1  # what clip(1, 65535) would have written

    def test_an_out_of_range_hot_value_is_not_clipped_to_the_ceiling(self):
        out = int(encode_celsius(np.array([700.0], dtype="float32"))[0])
        assert out == LST_NODATA_DN
        assert out != 65535

    def test_the_trusted_floor_matches_the_earlier_pipeline(self):
        assert LST_MIN_TRUSTED_DN == 2
        assert LST_MIN_TRUSTED_C == pytest.approx(
            LST_OFFSET + LST_MIN_TRUSTED_DN * LST_SCALE
        )
        assert LST_MIN_TRUSTED_C == pytest.approx(-49.98)

    def test_the_encoding_floor_collision_is_refused(self):
        # -49.99 C rounds to DN 1, which is reachable only from the floor and
        # marks a failed retrieval. DN 0 already means fill.
        assert int(encode_celsius(np.array([-49.99], dtype="float32"))[0]) == (
            LST_NODATA_DN
        )
        assert int(encode_celsius(np.array([-50.0], dtype="float32"))[0]) == (
            LST_NODATA_DN
        )

    def test_the_first_trusted_temperature_encodes(self):
        out = int(encode_celsius(np.array([LST_MIN_TRUSTED_C], dtype="float32"))[0])
        assert out == LST_MIN_TRUSTED_DN

    def test_an_ordinary_temperature_round_trips(self):
        out = encode_celsius(np.array([30.0, 45.5, 79.9], dtype="float32"))
        back = out.astype("float64") * LST_SCALE + LST_OFFSET
        assert list(back) == pytest.approx([30.0, 45.5, 79.9], abs=LST_SCALE)


class TestThePredicatesTakeDataArrays:
    """The predicates are one rule for numpy and xarray alike.

    `composite.reduce_block` hands numpy to `masked_celsius`; the predicates it
    is built from also answer a DataArray, which is what lets a test or a
    notebook ask the same question of a lazy stack.
    """

    def test_the_validity_rule_agrees_on_both(self):
        from lst_qa import valid_observation

        rng = np.random.default_rng(7)
        dn = rng.integers(0, 65536, size=(5, 8, 8), dtype="uint16")
        dn[0, 0, :] = 0  # a fill row
        dn[1, 1, :] = 3  # a reprojected-edge row
        qa = rng.integers(0, 4096, size=(5, 8, 8)).astype("uint16")
        _, eager = masked_celsius(dn.copy(), qa)
        lazy_dn = xr.DataArray(dn, dims=("time", "y", "x"))
        lazy_qa = xr.DataArray(qa, dims=("time", "y", "x"))
        lazy = valid_observation(lazy_dn, lazy_qa, to_celsius(lazy_dn))
        np.testing.assert_array_equal(eager, lazy.values)

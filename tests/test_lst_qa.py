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

import numpy as np
import pytest
import xarray as xr


from lst.lst_qa import (
    LST_MIN_TRUSTED_C,
    LST_MIN_TRUSTED_DN,
    LST_NODATA_DN,
    LST_OFFSET,
    LST_OUTPUT_MAX_C,
    LST_OUTPUT_MIN_C,
    LST_SCALE,
    LST_VALID_MAX_C,
    LST_VALID_MIN_C,
    LWIR_FILL_DN,
    LWIR_OFFSET_C,
    LWIR_SCALE,
    MIN_TOTAL_OBSERVATIONS,
    MIN_WATER_OBSERVATIONS,
    QA_EXCLUDED_BIT_NUMBERS,
    QA_EXCLUDED_BITS,
    QA_WATER_BIT,
    QA_WATER_BITS,
    WATER_MAX_C,
    WATER_SHARE_THRESHOLD,
    encode_celsius,
    in_trusted_range,
    masked_celsius,
    observed_water,
    qa_clear,
    qa_water,
    supported_output,
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
        # pairs. This task masks none of them. Bit 7 is read by `qa_water` and
        # still belongs here: the water rule counts the observation, and an
        # observation it never saw cannot be counted.
        _, valid = one(DN_WARM, 1 << bit)
        assert valid is True

    def test_qa_clear_is_elementwise(self):
        qa = np.array([QA_CLEAR, QA_CLEAR | 0b1000, 0, 0b100000], dtype="uint16")
        assert list(np.asarray(qa_clear(qa))) == [True, False, True, False]


class TestTheWaterBit:
    """QA_PIXEL bit 7, which classifies a pixel and disqualifies nothing."""

    def test_the_bit_is_seven(self):
        assert QA_WATER_BIT == 7
        assert QA_WATER_BITS == 0b10000000
        assert QA_WATER_BITS == 128

    def test_the_water_bit_is_not_an_excluded_bit(self):
        # The two masks must not overlap. A water observation that the clear
        # rule rejected would be missing from both sides of the share.
        assert QA_WATER_BIT not in QA_EXCLUDED_BIT_NUMBERS
        assert QA_WATER_BITS & QA_EXCLUDED_BITS == 0

    def test_it_reads_bit_seven_alone(self):
        qa = np.array(
            [QA_CLEAR, QA_CLEAR | QA_WATER_BITS, 0, QA_WATER_BITS], dtype="uint16"
        )
        assert list(np.asarray(qa_water(qa))) == [False, True, False, True]

    def test_a_cloudy_water_observation_still_sets_the_bit(self):
        # The predicate answers one question. Whether the observation counts
        # is `qa_clear`'s answer, and `reduce_block` takes both.
        qa = np.array([QA_CLEAR | QA_WATER_BITS | (1 << 3)], dtype="uint16")
        assert bool(np.asarray(qa_water(qa))[0]) is True
        assert bool(np.asarray(qa_clear(qa))[0]) is False

    def test_it_answers_a_dataarray(self):
        # The graph hands `reduce_block` numpy, but the predicates are written
        # to work on either, like every other rule in this module.
        qa = xr.DataArray(
            np.array([QA_CLEAR, QA_CLEAR | QA_WATER_BITS], dtype="uint16"),
            dims=("time",),
        )
        assert list(np.asarray(qa_water(qa))) == [False, True]


class TestTheObservedWaterClassification:
    """`observed_water`, the share rule, at and around its two thresholds."""

    #: Well above the floor, so only the share decides these.
    DEEP = 100

    def share(self, water, clear=DEEP):
        return bool(np.asarray(observed_water(np.array([water]), np.array([clear])))[0])

    def test_the_threshold_is_inclusive(self):
        on = int(np.ceil(WATER_SHARE_THRESHOLD * self.DEEP))
        assert self.share(on) is True
        assert self.share(on - 1) is False

    def test_all_water_is_water(self):
        assert self.share(self.DEEP) is True

    def test_no_water_is_not_water(self):
        assert self.share(0) is False

    def test_a_zero_denominator_is_unknown_rather_than_water(self):
        # The rule is a comparison, not a quotient, so `0 >= 0.75 * 0` is true
        # and only the floor keeps an unobserved pixel out. Without it every
        # pixel no scene reached would classify as water.
        assert self.share(0, clear=0) is False

    def test_a_pixel_below_the_floor_is_unknown(self):
        below = MIN_WATER_OBSERVATIONS - 1
        assert self.share(below, clear=below) is False

    def test_a_pixel_on_the_floor_is_decided(self):
        floor = MIN_WATER_OBSERVATIONS
        assert self.share(floor, clear=floor) is True

    def test_the_floor_matches_the_evidence_rule(self):
        # A pixel below `MIN_TOTAL_OBSERVATIONS` is already nodata, so a
        # different floor here would decide nothing and read as a second rule.
        assert MIN_WATER_OBSERVATIONS == MIN_TOTAL_OBSERVATIONS

    def test_the_threshold_clears_the_land_mode(self):
        """It has to sit above the land mode and at or below the water one.

        MEASURED over 17 cached blocks of N40W080: 93.8% of pixels sit below a
        share of 0.05 and 3.0% sit at or above 0.90. A threshold inside the
        land mode would delete ground, and one above the water mode would
        classify nothing.
        """
        assert 0.05 < WATER_SHARE_THRESHOLD <= 0.90

    def test_the_threshold_is_where_the_tropics_put_it(self):
        """The temperate blocks alone would take 0.75.

        MEASURED on a Kalimantan forest block, whose share ramps smoothly with
        no empty bin: 0.75 classifies 0.4599% of it and 0.90 classifies
        0.1111%, while Delaware Bay stays at 100.0000% and the Chesapeake
        falls only from 93.0602% to 93.0046%.
        """
        assert WATER_SHARE_THRESHOLD == 0.90

    def test_large_counters_do_not_wrap(self):
        # The counters are uint32 and the threshold is a Python float. Without
        # the float64 promotion the product is computed in the counter's own
        # type and a five-year stack of a busy pixel could overflow.
        clear = np.array([4_000_000_000], dtype="uint32")
        water = np.array([4_000_000_000], dtype="uint32")
        assert bool(np.asarray(observed_water(water, clear))[0]) is True

    def test_it_is_elementwise(self):
        # Written against the constant rather than a literal, so the case
        # either side of the threshold survives a change to it.
        on = int(np.ceil(WATER_SHARE_THRESHOLD * 100))
        water = np.array([0, on - 1, on, 100], dtype="uint32")
        clear = np.full(4, 100, dtype="uint32")
        got = list(np.asarray(observed_water(water, clear)))
        assert got == [False, False, True, True]

    def hot(self, celsius, water=DEEP, clear=DEEP):
        return bool(
            np.asarray(
                observed_water(
                    np.array([water]), np.array([clear]), np.array([celsius])
                )
            )[0]
        )

    def test_a_percentile_too_hot_to_be_water_is_not_water(self):
        # The defect this exists for. MEASURED in Center City Philadelphia:
        # 143 pixels reading 36 C to 56 C carried bit 7 on 86% to 100% of
        # their observations, so the share rule alone cannot reject them.
        assert self.share(self.DEEP) is True, "the share alone would classify it"
        assert self.hot(WATER_MAX_C + 1.0) is False

    def test_a_pixel_on_the_ceiling_is_still_water(self):
        assert self.hot(WATER_MAX_C) is True

    def test_a_cold_pixel_is_unaffected_by_the_ceiling(self):
        assert self.hot(WATER_MAX_C - 15.0) is True

    def test_a_nan_percentile_keeps_its_classification(self):
        # It carries no temperature to argue with, and it is nodata either
        # way. Written as `> max_c` rather than `<= max_c` for exactly this.
        assert self.hot(float("nan")) is True

    def test_the_ceiling_clears_open_water_with_room_to_spare(self):
        """The bound is calibrated against two sets with known truth.

        MEASURED at a share of 0.75: Delaware Bay runs at a P95 of 27.42 C and
        the Chesapeake at 28.69 C. Dropping the bound from 40 C to 34 C takes
        0.00% of Delaware Bay and 0.01% of the Chesapeake, and takes the Center
        City false positives from 1.33% of that box to 0.00%. Below 34 C the
        cost to open water starts: 32 C takes 0.15% of the Chesapeake.

        So the bound is not "above the hottest water ever seen". It is where
        the false positives end, and it clears the body of open water by more
        than 5 C.
        """
        assert WATER_MAX_C >= 32.0, "below this the bound starts cutting open water"
        assert WATER_MAX_C - 28.69 > 5.0, "no headroom over the Chesapeake median"

    def test_the_ceiling_is_below_the_urban_false_positives(self):
        # MEASURED in Center City: the 143 falsely classified pixels read
        # 36 C to 56 C. A bound above that range would let all of them back.
        assert WATER_MAX_C < 36.0

    def test_without_a_percentile_the_share_alone_decides(self):
        # The signature keeps `celsius` optional so the share rule can be
        # tested and measured on its own. No production path omits it.
        assert self.share(self.DEEP) is True

    def test_a_caller_can_ask_a_different_threshold(self):
        water, clear = np.array([50]), np.array([100])
        assert bool(np.asarray(observed_water(water, clear, threshold=0.5))[0])
        assert not bool(np.asarray(observed_water(water, clear, threshold=0.6))[0])


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
        from lst.lst_qa import valid_observation

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


class TestTheOutputSupportRule:
    """What `supported_output` publishes, and what it will not.

    The rule bounds the composite rather than an observation, so it asks two
    questions the per-observation rules cannot. Is there enough evidence behind
    this percentile, and is the number it produced a temperature. Both bounds
    are inclusive, and the nine cases below fix every edge.

    Why a second range check exists at all: `destripe.subtract_offsets` moves a
    decoded value after `in_trusted_range` has passed it, so the per-observation
    ceiling does not survive to the output. MEASURED on N30E075, 207 published
    pixels reach 80 C or hotter.
    """

    @staticmethod
    def ask(celsius, observations):
        """One pixel, as a bool."""
        return bool(
            np.asarray(
                supported_output(
                    np.array([celsius], dtype="float32"),
                    np.array([observations], dtype="uint16"),
                )
            )[0]
        )

    def test_the_floor_is_five_observations(self):
        assert MIN_TOTAL_OBSERVATIONS == 5

    def test_four_observations_are_not_enough(self):
        assert self.ask(28.4, 4) is False

    def test_five_observations_are_enough(self):
        assert self.ask(28.4, 5) is True

    def test_no_observation_at_all_is_not_enough(self):
        assert self.ask(28.4, 0) is False

    def test_a_step_below_the_cold_bound_is_refused(self):
        assert self.ask(LST_OUTPUT_MIN_C - LST_SCALE, 140) is False

    def test_the_cold_bound_is_inclusive(self):
        assert self.ask(LST_OUTPUT_MIN_C, 140) is True

    def test_a_step_above_the_hot_bound_is_refused(self):
        assert self.ask(LST_OUTPUT_MAX_C + LST_SCALE, 140) is False

    def test_the_hot_bound_is_inclusive(self):
        assert self.ask(LST_OUTPUT_MAX_C, 140) is True

    def test_an_ordinary_well_supported_pixel_survives(self):
        # The median tile carries between 56 and 160 observations, and a
        # hot-season p95 over land sits far inside both bounds.
        assert self.ask(28.4, 140) is True

    def test_nan_is_refused_however_many_observations_back_it(self):
        assert self.ask(np.nan, 3000) is False

    def test_the_two_halves_are_independent(self):
        # Evidence does not excuse an impossible value, and a possible value
        # does not excuse missing evidence.
        assert self.ask(120.0, 3000) is False
        assert self.ask(28.4, 1) is False

    def test_the_output_bounds_are_not_the_observation_bounds(self):
        # A guard against the two pairs being read as one. The cold bounds
        # differ by 30 C, and a per-observation value at -45 C is usable while
        # a composite at -45 C is not.
        assert LST_OUTPUT_MIN_C > LST_VALID_MIN_C
        assert self.ask(-45.0, 140) is False
        assert bool(np.asarray(in_trusted_range(np.float32(-45.0))))

    def test_it_is_elementwise(self):
        celsius = np.array([28.4, 28.4, -25.0, 95.0], dtype="float32")
        obs = np.array([140, 4, 140, 140], dtype="uint16")
        assert list(np.asarray(supported_output(celsius, obs))) == [
            True,
            False,
            False,
            False,
        ]

    def test_it_answers_a_dataarray_too(self):
        celsius = xr.DataArray(
            np.array([[28.4, 28.4]], dtype="float32"), dims=("y", "x")
        )
        obs = xr.DataArray(np.array([[140, 4]], dtype="uint16"), dims=("y", "x"))
        assert list(np.asarray(supported_output(celsius, obs)).ravel()) == [True, False]

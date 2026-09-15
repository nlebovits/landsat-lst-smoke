"""One definition of a usable LST observation.

`composite.reduce_block` calls `masked_celsius` on every block of the lazy
graph, and nothing else decides validity. The predicates below are that
definition, and they work on numpy arrays and DataArrays alike.

Most rules come from `nlebovits/landsat-lst` (`qa.py`, `encoding.py`, and the
composite validation in `pipeline.py`). Four of them matter:

QA_PIXEL bits 1 to 5 are masked: dilated cloud, cirrus, cloud, cloud shadow,
and snow. Bits 3 and 4 alone leave cloud edges and thin cirrus in the stack,
and those survive as per-scene warm and cool residuals along WRS footprints.

Raw DN 0 is the source fill value and decodes to -124.15 C.

The decoded Celsius value has to land inside [-50, 80]. Scene boundaries hold
more than exact fill: reprojection interpolates between a valid DN and the
DN 0 beside it, which produces small nonzero values that decode near -124 C. An
exact `!= 0` test cannot see them. The range check can.

Rejected values become NaN before the percentile runs, never after. A value
that reaches `nanpercentile` has already moved the answer, whatever the encoder
does with the result afterwards.

An unrepresentable composite becomes nodata, never a clipped DN. Clipping
-124 C to -49.99 C replaces a visible gap with a believable cold pixel.

A fifth rule is this repository's own. QA_PIXEL bit 7 marks an observation as
water, and the composite counts how often it fires rather than rejecting it.
`observed_water` turns that count into one decision per pixel, taken over the
whole window, and a pixel it calls water leaves the product. The rule exists
because the buffered land geometry cannot remove a river, and reaches 25 km
past the coast by design. Neither is a defect in the geometry. The geometry is
answering a different question.

The composite carries two rules of its own, and `supported_output` is both.
An observation that passed every rule above can still produce a percentile no
consumer should read. `destripe.subtract_offsets` runs after the range check
and moves the decoded value, so the per-observation bound does not survive to
the output; Delhi ships 207 pixels at or above 80 C. And a pixel observed four
times over five years has an order statistic over four scenes rather than a
five-year 95th percentile, however plausible the number looks.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Source constants: Landsat Collection 2 Level 2 surface temperature (ST_B10 /
# lwir11). celsius = dn * LWIR_SCALE + LWIR_OFFSET_K - KELVIN_ZERO_C.
# --------------------------------------------------------------------------

LWIR_SCALE = 0.00341802
LWIR_OFFSET_K = 149.0
KELVIN_ZERO_C = 273.15
#: Fused offset that takes a raw DN straight to Celsius.
LWIR_OFFSET_C = LWIR_OFFSET_K - KELVIN_ZERO_C  # -124.15
#: The source's fill value. Decodes to -124.15 C if it is left in.
LWIR_FILL_DN = 0

# --------------------------------------------------------------------------
# QA_PIXEL. Collection 2 bit assignments.
# --------------------------------------------------------------------------

QA_DILATED_CLOUD_BIT = 1
QA_CIRRUS_BIT = 2
QA_CLOUD_BIT = 3
QA_CLOUD_SHADOW_BIT = 4
QA_SNOW_BIT = 5

#: Every bit that disqualifies a pixel. Named for what it covers, because
#: `QA_CLOUD_BITS` would understate a mask that also drops shadow and snow.
QA_EXCLUDED_BIT_NUMBERS = (
    QA_DILATED_CLOUD_BIT,
    QA_CIRRUS_BIT,
    QA_CLOUD_BIT,
    QA_CLOUD_SHADOW_BIT,
    QA_SNOW_BIT,
)
QA_EXCLUDED_BITS = sum(1 << bit for bit in QA_EXCLUDED_BIT_NUMBERS)  # 0b111110

#: Water, which disqualifies no observation and classifies the pixel instead.
#: A water observation is a real retrieval of a real surface, so it belongs in
#: the record the classification reads. `observed_water` is what reads it.
QA_WATER_BIT = 7
QA_WATER_BITS = 1 << QA_WATER_BIT  # 0b10000000

# --------------------------------------------------------------------------
# Physical plausibility, in Celsius. Inclusive at both ends.
# --------------------------------------------------------------------------

LST_VALID_MIN_C = -50.0
LST_VALID_MAX_C = 80.0

# --------------------------------------------------------------------------
# Output encoding: celsius = dn * LST_SCALE + LST_OFFSET, DN 0 reserved.
# --------------------------------------------------------------------------

LST_SCALE = 0.01
LST_OFFSET = -50.0
LST_NODATA_DN = 0
LST_MIN_DN = 1
LST_MAX_DN = 65535

#: Lowest DN carrying a temperature worth publishing. DN 0 is fill, and DN 1 is
#: reachable only from a value sitting on the encoding floor at -49.99 C. A
#: hot-season P95 over land never gets that cold, so DN 1 marks a failed
#: retrieval rather than a temperature.
LST_MIN_TRUSTED_DN = 2
#: The bottom of the DN 2 bucket. Anything colder is recorded as missing.
LST_MIN_TRUSTED_C = LST_OFFSET + LST_MIN_TRUSTED_DN * LST_SCALE  # -49.98

# --------------------------------------------------------------------------
# Output plausibility. These bound the composite, not an observation, and
# `composite.reduce_block` applies them where the percentile and the monthly
# counts already sit together.
# --------------------------------------------------------------------------

#: Clear observations a pixel needs over the five years before its P95 is an
#: estimate rather than an order statistic over a handful of scenes.
#:
#: Empirical, and deliberately at the conservative end. MEASURED over the five
#: audited tiles, a floor of 5 removes 0.0003% of N30E075, 0.0005% of S30W065,
#: 0.0069% of N40W080, 0.597% of N00E110, and 0.804% of S25E030 of valid land.
#: Median total observations run from 56 on N00E110 to 160 on N40W080, so the
#: floor cuts a thin tail. Higher floors stop doing that: 50 removes 39% of
#: N00E110, whose median is 56.
MIN_TOTAL_OBSERVATIONS = 5

#: The coldest five-year hot-season P95 worth publishing over land.
#:
#: Sparse evidence, not cold ground, is what reaches below it. MEASURED: the
#: published N00E110 minimum is -39.82 C and the published S25E030 minimum is
#: -0.25 C, and the observation floor alone lifts them to 19.96 C and 9.84 C.
#: The bound is the second line, for a sparse pixel the floor lets through.
LST_OUTPUT_MIN_C = -20.0

#: The hottest land surface temperature this product will publish.
#:
#: The same number as `LST_VALID_MAX_C`, applied a second time because the
#: first application cannot hold. `destripe.subtract_offsets` shifts a decoded
#: value after `in_trusted_range` has passed it, and MEASURED on N30E075, 207
#: pixels reach the output at or above 80 C.
LST_OUTPUT_MAX_C = 80.0

# --------------------------------------------------------------------------
# Water, classified from the record rather than from a polygon. The buffered
# land geometry in `masks.py` reaches 25 km past the coast on purpose, and it
# knows nothing about a river. Both leave sea in the output. The stack does
# know: QA_PIXEL sets bit 7 over water, observation after observation.
# --------------------------------------------------------------------------

#: The share of a pixel's usable clear observations that must call it water.
#:
#: MEASURED over 17 cached 360 px blocks of `N40W080`, 2,199,809 observed
#: pixels with five-year stacks 569 to 765 scenes deep. The distribution has
#: two modes: 93.8% of pixels sit below 0.05 and 3.0% sit at or above 0.90.
#: The published temperature separates them. On the Philadelphia river block,
#: pixels below 0.05 have a median of 44.82 C and pixels at or above 0.90 have
#: a median of 28.92 C inside a 1.6 C interquartile band. The bands between run
#: monotonically down: 43.49 C over [0.25, 0.50), 39.24 C over [0.50, 0.75),
#: 33.87 C over [0.75, 0.90). A threshold of 0.25 reaches [0.05, 0.25), whose
#: median of 45.00 C is land, so it would delete ground.
#:
#: The temperate blocks would take 0.75, which is their emptiest bin. The
#: tropics decide otherwise. MEASURED on `N00E110`, a Kalimantan forest block
#: has no empty bin at all: the share ramps smoothly from 0 to 1 and the band
#: at [0.75, 0.90) reads 31.60 C against forest at 33.65 C, two degrees rather
#: than the eleven that separate the temperate river from its bank. Those
#: pixels are as likely canopy as stream.
#:
#: Raising the threshold costs almost nothing and halves that doubt, at a
#: bound of 34 C:
#:
#:     block                 truth             at 0.75    at 0.90
#:     Delaware Bay          all water        100.0000%  100.0000%
#:     Chesapeake            mostly water      93.0602%   93.0046%
#:     Delaware shoreline    mostly water      99.7207%   99.6998%
#:     Barito estuary        tropical water    12.0725%   12.0725%
#:     Kahayan river         tropical water     6.3156%    6.3156%
#:     Kalimantan forest     tropical land      0.4599%    0.1111%
#:     Sebangau peat         tropical land      0.0000%    0.0000%
#:     Rajasthan desert      arid land          0.0000%    0.0000%
#:     Center City           no water           0.0000%    0.0000%
#:
#: Open water is untouched and the ambiguous forest classifications fall four
#: times over. The rule errs towards publishing water rather than deleting
#: land, which is the cheaper of the two mistakes for a land product.
WATER_SHARE_THRESHOLD = 0.90

#: Usable clear observations a pixel needs before the share means anything.
#:
#: The same number as `MIN_TOTAL_OBSERVATIONS`, and for the same reason: a
#: pixel below it is already nodata, so a second floor would decide nothing.
#: It is stated separately because the two rules are separate, and because
#: without a floor the comparison form of `observed_water` would read `0 >= 0`
#: at an unobserved pixel and call the whole unobserved world water.
MIN_WATER_OBSERVATIONS = MIN_TOTAL_OBSERVATIONS

#: The hottest five-year P95 a pixel can hold and still be called water.
#:
#: The share rule trusts one QA bit, and over a dark roof that bit is wrong. A
#: reflectance test sets it, and it matches dark asphalt on almost every scene.
#: MEASURED in Center City Philadelphia, 6,264 pixels holding no water: 143 of
#: them read 36 C to 56 C and carry the flag on 86% to 100% of their 80 clear
#: observations. No threshold on the share excludes those, because their share
#: is the share of open sea.
#:
#: Water has a ceiling that asphalt does not, and the thermal band is the
#: better evidence about which surface this is.
#:
#: Calibrated against two sets with known truth. The share was 0.75 when this
#: sweep ran, which is the harder test: at 0.90 the share rule alone already
#: rejects some of what the bound had to catch.
#:
#:     bound    Center City masked    Delaware Bay kept    Chesapeake kept
#:     40 C                 1.33%              100.00%             93.07%
#:     36 C                 0.70%              100.00%             93.07%
#:     35 C                 0.26%              100.00%             93.07%
#:     34 C                 0.00%              100.00%             93.06%
#:     32 C                 0.00%              100.00%             92.92%
#:
#: 34 C is where the false positives end and before the cost to open water
#: starts. It takes 0.01% of the Chesapeake block and nothing from Delaware
#: Bay, against 32 C, which takes 0.15%.
#:
#: The calibration is temperate water, and the tropics have since been probed
#: against it. MEASURED on `N00E110`: the Barito estuary classifies at a median
#: of 33.18 C and the Kahayan river at 33.92 C, both under the bound, so the
#: warm-water fear it was written against is smaller than expected. The Kahayan
#: upper quartile of 34.74 C does cross it, and that slice of river keeps its
#: temperature rather than being called water. That is the conservative
#: direction on purpose. A warm pond published as land is a smaller error than
#: a warm roof deleted as water.
WATER_MAX_C = 34.0


# --------------------------------------------------------------------------
# Predicates. Each one works on a numpy array and on an xarray DataArray,
# eager or dask-backed, because both libraries answer the same operators.
# --------------------------------------------------------------------------


def qa_clear(qa_pixel):
    """True where QA_PIXEL bits 1 to 5 are all clear."""
    return (qa_pixel & QA_EXCLUDED_BITS) == 0


def qa_water(qa_pixel):
    """True where QA_PIXEL bit 7 calls the observation water.

    This rejects nothing. The observation still enters the percentile, and the
    count of how often it fires is what `observed_water` classifies the pixel
    on.
    """
    return (qa_pixel & QA_WATER_BITS) != 0


def not_fill(thermal_dn):
    """True where the raw thermal DN is not the source fill value."""
    return thermal_dn != LWIR_FILL_DN


def to_celsius(thermal_dn):
    """Decode a raw thermal DN to Celsius as float32.

    Cast first, then scale. Multiplying a uint16 array by a Python float
    produces float64 and doubles the stack for nothing.

    The scale and the offset are applied in place on the cast, so exactly one
    float32 copy of the stack exists at any moment. `astype` already made that
    copy and nothing else holds it. Bit-identical to `cast * scale + offset`:
    the same two float32 operations in the same order.

    numpy elides the temporaries of the expression form when the operand is a
    temporary of its own, so on a numpy stack this saves nothing. MEASURED at
    (820, 360, 360) uint16 on numpy 2.5.3: 405.4 MiB peak either way. It is
    the xarray path that pays, because a DataArray operation allocates a new
    array every time. MEASURED on the same block as a DataArray: 810.8 MiB
    before, 405.4 MiB after.
    """
    import numpy as np

    celsius = thermal_dn.astype("float32")
    celsius *= np.float32(LWIR_SCALE)
    celsius += np.float32(LWIR_OFFSET_C)
    return celsius


def in_trusted_range(celsius):
    """True where the decoded temperature is finite and physically possible."""
    import numpy as np

    return (
        np.isfinite(celsius)
        & (celsius >= LST_VALID_MIN_C)
        & (celsius <= LST_VALID_MAX_C)
    )


def valid_observation(thermal_dn, qa_pixel, celsius):
    """The full pre-percentile validity rule, in the order it has to run.

    Fill first, QA second, and the range check last, because the range check
    needs the decoded value. Nothing here looks at the reduction, so both paths
    can call it on whatever they hold.
    """
    return not_fill(thermal_dn) & qa_clear(qa_pixel) & in_trusted_range(celsius)


def encodable_dn(dn):
    """True where a rounded output DN is a temperature worth writing.

    NaN and infinity fail `isfinite`. Below `LST_MIN_TRUSTED_DN` is the
    encoding floor, which is a failed retrieval and not a cold pixel.
    """
    import numpy as np

    return np.isfinite(dn) & (dn >= LST_MIN_TRUSTED_DN) & (dn <= LST_MAX_DN)


def supported_output(celsius, total_observations):
    """True where the composite rests on enough evidence and is possible.

    Both bounds are inclusive: -20.00 C and 80.00 C are temperatures, and the
    pixel a step beyond either is not. `MIN_TOTAL_OBSERVATIONS` is the same, so
    a pixel observed exactly five times survives.

    `total_observations` is the sum of the twelve published `qa_count` bands, so
    a consumer reading the product can recompute this rule from the product.

    Args:
        celsius: the composite, float. NaN fails.
        total_observations: clear observations behind each pixel, integer.
    """
    import numpy as np

    return (
        np.isfinite(celsius)
        & (total_observations >= MIN_TOTAL_OBSERVATIONS)
        & (celsius >= LST_OUTPUT_MIN_C)
        & (celsius <= LST_OUTPUT_MAX_C)
    )


def observed_water(
    water_observations,
    clear_observations,
    celsius=None,
    *,
    threshold: float = WATER_SHARE_THRESHOLD,
    floor: int = MIN_WATER_OBSERVATIONS,
    max_c: float = WATER_MAX_C,
):
    """True where the pixel's own clear record says it is water.

    The rule is `water >= threshold * clear`, which is the share without the
    division. A pixel sitting exactly on the threshold is then decided by one
    comparison rather than by a rounded quotient, and the counters stay
    integers. The threshold is promoted to float64 first, so a uint32 counter
    cannot wrap on the way.

    A pixel below `floor` is unknown, not water. That is the rule that keeps an
    unobserved pixel out: with no floor the comparison reads `0 >= 0` and every
    pixel no scene reached would classify as water.

    `celsius` is the veto, and it is why the share is not the whole rule. The
    flag is one bit produced by a reflectance test, and over a dark roof or a
    rail yard that test fires on almost every scene. MEASURED in Center City
    Philadelphia: 143 pixels reading 36 C to 56 C carry the flag on 86% to
    100% of their 80 clear observations, so no threshold on the share can
    reach them. A five-year P95 above `max_c` is not water whatever the flag
    counted, and the thermal band is the better evidence about which it is.

    A pixel whose percentile is NaN keeps its classification. It carries no
    temperature to argue with, and it is nodata either way.

    Both counts are the whole window, unsaturated. They are not the published
    `qa_count`, which clips at 255 a month and carries no water flag, so this
    classification cannot be recomputed from the product.

    Args:
        water_observations: usable clear observations with QA_PIXEL bit 7 set.
        clear_observations: usable clear observations, the denominator.
        celsius: the composite this pixel would publish, float. Optional, and
            without it the share alone decides, which no production path does.
        threshold: the share at which the pixel becomes water, inclusive.
        floor: observations below which the pixel is unknown.
        max_c: the hottest percentile that can still be water.
    """
    import numpy as np

    clear = np.asarray(clear_observations)
    water = np.asarray(water_observations)
    wet = (clear >= floor) & (water >= np.float64(threshold) * clear)
    if celsius is None:
        return wet
    # `> max_c` rather than `<= max_c`, so a NaN percentile keeps its
    # classification instead of losing it to a comparison NaN always fails.
    return wet & ~(np.asarray(celsius) > np.float64(max_c))


# --------------------------------------------------------------------------
# numpy wrappers, called inside one block of the graph.
# --------------------------------------------------------------------------


def masked_celsius(thermal_dn, qa_pixel):
    """Decode one scene stack to Celsius, NaN wherever it is unusable.

    Returns the stack and the boolean validity mask that produced it, so the
    observation counts and the percentile are computed from one definition.
    The NaN is written in place, which keeps the working set at one float32
    copy of the stack.
    """
    import numpy as np

    valid = not_fill(thermal_dn) & qa_clear(qa_pixel)
    celsius = to_celsius(thermal_dn)
    valid &= in_trusted_range(celsius)
    celsius[~valid] = np.nan
    return celsius, valid


def encode_celsius(celsius):
    """Celsius to uint16 DN, with everything unrepresentable as nodata.

    `dn = (c - LST_OFFSET) / LST_SCALE`, rounded. Out-of-range values become
    `LST_NODATA_DN` rather than the nearest DN. A clipped -124 C arrives as a
    believable -49.99 C, which is worse than a gap.
    """
    import numpy as np

    dn = np.rint((celsius - LST_OFFSET) / LST_SCALE)
    return np.where(encodable_dn(dn), dn, LST_NODATA_DN).astype("uint16")

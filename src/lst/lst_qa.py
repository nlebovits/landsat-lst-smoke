"""One definition of a usable LST observation.

`composite.reduce_block` calls `masked_celsius` on every block of the lazy
graph, and nothing else decides validity. The predicates below are that
definition, and they work on numpy arrays and DataArrays alike.

The rules come from `nlebovits/landsat-lst` (`qa.py`, `encoding.py`, and the
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
# Predicates. Each one works on a numpy array and on an xarray DataArray,
# eager or dask-backed, because both libraries answer the same operators.
# --------------------------------------------------------------------------


def qa_clear(qa_pixel):
    """True where QA_PIXEL bits 1 to 5 are all clear."""
    return (qa_pixel & QA_EXCLUDED_BITS) == 0


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

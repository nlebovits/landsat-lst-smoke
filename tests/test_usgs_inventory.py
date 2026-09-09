"""The build's own rules: the guards that stop a wrong artifact shipping.

Everything here runs offline against hand-built Arrow tables. The parity of
these rules against the catalogue lives in `test_inventory_parity.py`, which is
marked `network`; this file is about what happens when an input is malformed,
which the catalogue cannot tell us.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from usgs_inventory import (  # noqa: E402
    MAX_SNAP_METERS,
    MONTH_BOUNDARY_GUARD_SECONDS,
    derive_projection,
    filter_to_window,
    stac_item_id,
    window_bounds,
)

CORNERS = (
    ("Upper Left", -64.5, -30.5),
    ("Upper Right", -63.5, -30.5),
    ("Lower Right", -63.5, -31.5),
    ("Lower Left", -64.5, -31.5),
)


def _table(rows=1, *, zone=20.0, projection="UTM", corner_override=None):
    """A minimal scan result, shaped as `derive_projection` reads it."""
    data = {
        "Display ID": pa.array(
            [f"LC08_L2SP_2270{i:02d}_20230601_20230610_02_T1" for i in range(rows)]
        ),
        "Product Map Projection L1": pa.array([projection] * rows),
        "UTM Zone": pa.array([zone] * rows, pa.float64()),
    }
    for name, lon, lat in CORNERS:
        data[f"Corner {name} Longitude"] = pa.array([lon] * rows, pa.float64())
        data[f"Corner {name} Latitude"] = pa.array([lat] * rows, pa.float64())
    if corner_override:
        column, value = corner_override
        data[column] = pa.array([value] * rows, pa.float64())
    return pa.table(data)


class TestProjectionRoundTrip:
    """Real corners in, the stored projection out, exactly.

    The slice carries both halves of the derivation: `corner_lon0..3` and
    `corner_lat0..3` are the inputs, and `proj_epsg`, `proj_shape_*` and
    `proj_origin_*` are what the build wrote from them. Re-deriving from the
    corners has to reproduce the stored values to the last bit, which is the
    claim `derive_projection` makes. A synthetic table cannot test this,
    because arbitrary corners do not sit on the 30 m product lattice.
    """

    @staticmethod
    def _bulk_shaped(slice_artifact):
        import pyarrow.parquet as pq

        rows = pq.read_table(slice_artifact).to_pylist()
        data = {
            "Display ID": pa.array([r["display_id"] for r in rows]),
            "Product Map Projection L1": pa.array(["UTM"] * len(rows)),
            "UTM Zone": pa.array(
                [float(r["proj_epsg"] - 32600) for r in rows], pa.float64()
            ),
        }
        for i, (name, _, _) in enumerate(CORNERS):
            data[f"Corner {name} Longitude"] = pa.array(
                [r[f"corner_lon{i}"] for r in rows], pa.float64()
            )
            data[f"Corner {name} Latitude"] = pa.array(
                [r[f"corner_lat{i}"] for r in rows], pa.float64()
            )
        return pa.table(data), rows

    def test_it_reproduces_every_stored_projection(self, slice_artifact):
        table, rows = self._bulk_shaped(slice_artifact)
        epsg, height, width, ulx, uly, drift = derive_projection(table)

        assert drift <= MAX_SNAP_METERS
        for i, row in enumerate(rows):
            assert int(epsg[i]) == row["proj_epsg"], row["display_id"]
            assert int(height[i]) == row["proj_shape_y"], row["display_id"]
            assert int(width[i]) == row["proj_shape_x"], row["display_id"]
            assert float(ulx[i]) == row["proj_origin_x"], row["display_id"]
            assert float(uly[i]) == row["proj_origin_y"], row["display_id"]


class TestProjectionGuards:
    @pytest.mark.parametrize(
        "column",
        ["Corner Upper Left Latitude", "Corner Lower Right Longitude"],
    )
    def test_a_null_corner_is_refused(self, column):
        """The defect this guard exists for.

        NaN survives the snap, and every comparison against NaN is False, so
        the `MAX_SNAP_METERS` check passed a row whose origin was NaN.
        `np.rint(nan).astype("int32")` is -2147483648, which is a shape the
        loader accepts and reads as garbage.
        """
        table = _table(corner_override=(column, float("nan")))
        with pytest.raises(ValueError, match="non-finite"):
            derive_projection(table)

    def test_a_null_utm_zone_is_refused(self):
        """`UTM Zone` is a DOUBLE, so a null casts to INT_MIN just as quietly."""
        with pytest.raises(ValueError, match="non-finite"):
            derive_projection(_table(zone=float("nan")))

    def test_a_non_utm_product_is_refused(self):
        with pytest.raises(ValueError, match="non-UTM"):
            derive_projection(_table(projection="PS"))

    def test_the_snap_limit_still_bites(self):
        """Corners well off the lattice have to stop the build.

        The guard is a quarter of a pixel. Arbitrary lon/lat lands anywhere in
        the 30 m cell, so this table is exactly the case the limit is for.
        """
        with pytest.raises(ValueError, match="product grid"):
            derive_projection(_table())


class TestWindow:
    def test_bounds_read_both_spellings(self):
        lo, hi = window_bounds("2021-01-01", "2025-12-31T23:59:59Z")
        assert str(lo).startswith("2021-01-01T00:00:00")
        assert str(hi).startswith("2025-12-31T23:59:59")

    def test_the_window_is_closed_at_both_ends(self):
        lo, hi = window_bounds()
        centre = np.array([lo, hi], dtype="datetime64[us]")
        assert filter_to_window(centre).all()

    def test_a_scene_a_second_outside_is_cut(self):
        """The reason the scan runs a day wide and this filter exists.

        `Date Acquired` is a date. A scene acquired across midnight on the
        first of the window has a date on one side and a centre on the other,
        and a STAC search compares the centre.
        """
        lo, hi = window_bounds()
        one_second = np.timedelta64(1, "s")
        centre = np.array([lo - one_second, hi + one_second], dtype="datetime64[us]")
        assert not filter_to_window(centre).any()

    def test_the_guard_is_wide_enough_for_the_measured_residual(self):
        """Sized from measure_scene_centre.py, not from a rounding argument.

        Truncating start and stop to whole seconds moves their midpoint by
        under a second; the definitional offset adds about 4 ms on top.
        """
        assert MONTH_BOUNDARY_GUARD_SECONDS >= 2.0


class TestItemId:
    def test_the_processing_date_is_dropped(self):
        assert (
            stac_item_id("LC09_L2SR_087074_20241231_20250102_02_T1")
            == "LC09_L2SR_087074_20241231_02_T1"
        )

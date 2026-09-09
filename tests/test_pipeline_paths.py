"""Both P95 paths, end to end, on one synthetic scene stack.

`lst_qa` holds the masking rule and `tests/test_lst_qa.py` tests it directly.
These tests check the wiring: that `shard_lst_p95.process_shard` and
`profile_lst_p95.build_graph` each call the rule, in the right order, and that
they encode the same stack to the same raster.

Neither test reads S3. `odc.stac.stac_load` is replaced by a fixture that
returns a dataset with known contents, so the reads, the CRS, and the STAC
items are out of the picture and only the mask, the reduction, and the encoder
are under test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import profile_lst_p95  # noqa: E402
import shard_lst_p95  # noqa: E402
from lst_qa import (  # noqa: E402
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    LWIR_FILL_DN,
    LWIR_OFFSET_C,
    LWIR_SCALE,
)

QA_CLEAR = 0b1000000
QA_CLOUD = QA_CLEAR | (1 << 3)
QA_CIRRUS = QA_CLEAR | (1 << 2)
QA_SNOW = QA_CLEAR | (1 << 5)

NY = NX = 4
N_TIME = 10


def dn_of(celsius: float) -> int:
    return int(round((celsius - LWIR_OFFSET_C) / LWIR_SCALE))


def build_stack():
    """Ten scenes over a 4x4 grid, with every rejection reason represented.

    Pixel (0, 0) is clear throughout. Pixel (0, 1) carries a cirrus-flagged
    scene and a snow-flagged one. Pixel (1, 0) carries the reprojected-edge
    artifact: a small nonzero DN that decodes near -124 C. Pixel (1, 1) is
    fill in every scene, so it has no valid observation at all.
    """
    warm = np.linspace(18.0, 44.0, N_TIME)
    dn = np.empty((N_TIME, NY, NX), dtype="uint16")
    for t, c in enumerate(warm):
        dn[t, :, :] = dn_of(float(c))
    qa = np.full((N_TIME, NY, NX), QA_CLEAR, dtype="uint16")

    qa[2, 0, 1] = QA_CIRRUS
    qa[7, 0, 1] = QA_SNOW
    qa[4, 2, 2] = QA_CLOUD

    dn[3, 1, 0] = 4  # reprojection interpolated against the fill: about -124 C
    dn[6, 1, 0] = 300  # same artifact, a larger nonzero value
    dn[:, 1, 1] = 0  # exact source fill, every scene

    # Spread the scenes across the calendar so the monthly climatology has
    # more than one bucket to fill.
    times = np.datetime64("2021-02-15", "ns") + np.arange(N_TIME) * np.timedelta64(
        46, "D"
    )
    return xr.Dataset(
        {
            "lwir11": (("time", "latitude", "longitude"), dn),
            "qa_pixel": (("time", "latitude", "longitude"), qa),
        },
        coords={"time": times},
    ), warm


@pytest.fixture
def stack():
    return build_stack()


@pytest.fixture
def fake_stac_load(monkeypatch, stack):
    """Replace the reader on both paths with the synthetic stack."""
    import odc.stac

    dataset = stack[0]

    def _load(*_args, **_kwargs):
        return dataset.copy(deep=True)

    monkeypatch.setattr(odc.stac, "stac_load", _load)
    return dataset


def expected_p95(warm, drop_indices=()):
    kept = [
        float(
            np.float32(dn_of(float(c))) * np.float32(LWIR_SCALE)
            + np.float32(LWIR_OFFSET_C)
        )
        for i, c in enumerate(warm)
        if i not in drop_indices
    ]
    return float(np.percentile(np.array(kept, dtype="float32"), 95))


def to_dn(celsius: float) -> int:
    return int(np.rint((celsius - LST_OFFSET) / LST_SCALE))


class TestShardPath:
    """`process_shard`: eager numpy, one shard, one worker."""

    def _run(self, monkeypatch, fake_stac_load):
        import pystac

        monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
        shard = shard_lst_p95.Shard(0, 0, 0, 0, NY, NX, (-60.0, -34.0, -59.9, -33.9))
        return shard_lst_p95.process_shard(
            shard, [{} for _ in range(N_TIME)], "EPSG:4326", 1 / 3600, read_threads=1
        )

    def test_a_clear_pixel_keeps_every_observation(
        self, monkeypatch, fake_stac_load, stack
    ):
        out = self._run(monkeypatch, fake_stac_load)
        _, warm = stack
        assert int(out["lst_p95"][0, 0]) == pytest.approx(
            to_dn(expected_p95(warm)), abs=1
        )
        assert int(out["qa_count"][:, 0, 0].sum()) == N_TIME

    def test_qa_flagged_scenes_leave_the_stack(
        self, monkeypatch, fake_stac_load, stack
    ):
        out = self._run(monkeypatch, fake_stac_load)
        _, warm = stack
        assert int(out["qa_count"][:, 0, 1].sum()) == N_TIME - 2
        assert int(out["lst_p95"][0, 1]) == pytest.approx(
            to_dn(expected_p95(warm, drop_indices={2, 7})), abs=1
        )

    def test_the_reprojected_edge_artifact_leaves_the_stack(
        self, monkeypatch, fake_stac_load, stack
    ):
        out = self._run(monkeypatch, fake_stac_load)
        _, warm = stack
        # Two of the ten samples decode near -124 C. Neither is exactly the
        # fill value, so only the range check can remove them.
        assert int(out["qa_count"][:, 1, 0].sum()) == N_TIME - 2
        assert int(out["lst_p95"][1, 0]) == pytest.approx(
            to_dn(expected_p95(warm, drop_indices={3, 6})), abs=1
        )
        decoded = int(out["lst_p95"][1, 0]) * LST_SCALE + LST_OFFSET
        assert decoded > 0.0

    def test_an_all_fill_pixel_becomes_nodata(self, monkeypatch, fake_stac_load):
        out = self._run(monkeypatch, fake_stac_load)
        assert int(out["lst_p95"][1, 1]) == LST_NODATA_DN
        assert int(out["qa_count"][:, 1, 1].sum()) == 0

    def test_every_finite_output_decodes_inside_the_trusted_range(
        self, monkeypatch, fake_stac_load
    ):
        out = self._run(monkeypatch, fake_stac_load)
        dn = out["lst_p95"]
        valid = dn != LST_NODATA_DN
        celsius = dn[valid].astype("float64") * LST_SCALE + LST_OFFSET
        assert celsius.min() >= -50.0
        assert celsius.max() <= 80.0

    def test_the_monthly_counts_never_exceed_the_scene_count(
        self, monkeypatch, fake_stac_load
    ):
        out = self._run(monkeypatch, fake_stac_load)
        assert out["qa_count"].shape == (12, NY, NX)
        assert int(out["qa_count"].sum(axis=0).max()) <= N_TIME

    def test_dropping_a_scene_with_no_thermal_band_changes_nothing(
        self, monkeypatch, stack
    ):
        """The claim `staging.drop_scenes_without_thermal` rests on.

        An `OLI_TIRS_L2SR` product carries `qa_pixel` and no `lwir11`, so
        `odc.stac.stac_load` fills the thermal band with the source fill value.
        `lst_qa.valid_observation` starts at `not_fill`, which makes the whole
        layer invalid before the percentile or the monthly counts see it.

        The filter is on by default, so this is what says the default is safe.
        Byte equality, not closeness: a fill layer that moved the answer at all
        would mean the mask is not doing what the module claims.
        """
        import odc.stac
        import pystac

        monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
        base = stack[0]

        n_extra = 3
        extra = base.isel(time=slice(0, n_extra)).copy(deep=True)
        extra["lwir11"][:] = LWIR_FILL_DN
        extra = extra.assign_coords(
            time=base["time"].values[:n_extra] + np.timedelta64(1, "D")
        )
        padded = xr.concat([base, extra], dim="time").sortby("time")

        shard = shard_lst_p95.Shard(0, 0, 0, 0, NY, NX, (-60.0, -34.0, -59.9, -33.9))
        out = {}
        for name, dataset in (("kept", padded), ("dropped", base)):
            monkeypatch.setattr(
                odc.stac,
                "stac_load",
                lambda *_a, _d=dataset, **_k: _d.copy(deep=True),
            )
            out[name] = shard_lst_p95.process_shard(
                shard,
                [{} for _ in range(dataset.sizes["time"])],
                "EPSG:4326",
                1 / 3600,
                read_threads=1,
            )

        assert np.array_equal(out["kept"]["lst_p95"], out["dropped"]["lst_p95"])
        assert np.array_equal(out["kept"]["qa_count"], out["dropped"]["qa_count"])
        # The layers were there and were counted. Otherwise this passes because
        # the fixture never added them.
        assert out["kept"]["n_scenes"] == out["dropped"]["n_scenes"] + n_extra


class TestArrayGraphPath:
    """`build_graph`: the lazy xarray graph the profiling harness measures."""

    def _run(self, fake_stac_load):
        lst_u16, qa_count, _ = profile_lst_p95.build_graph(
            [{} for _ in range(N_TIME)],
            (-60.0, -34.0, -59.9, -33.9),
            NX,
            "EPSG:4326",
            1 / 3600,
            time_chunk=N_TIME,
            load_chunk=NX,
        )
        return np.asarray(lst_u16.values), np.asarray(qa_count.values)

    def test_a_clear_pixel_keeps_every_observation(self, fake_stac_load, stack):
        lst, qa_count = self._run(fake_stac_load)
        _, warm = stack
        assert int(lst[0, 0]) == pytest.approx(to_dn(expected_p95(warm)), abs=1)
        assert int(qa_count[:, 0, 0].sum()) == N_TIME

    def test_qa_flagged_scenes_leave_the_stack(self, fake_stac_load, stack):
        lst, qa_count = self._run(fake_stac_load)
        _, warm = stack
        assert int(qa_count[:, 0, 1].sum()) == N_TIME - 2
        assert int(lst[0, 1]) == pytest.approx(
            to_dn(expected_p95(warm, drop_indices={2, 7})), abs=1
        )

    def test_the_reprojected_edge_artifact_leaves_the_stack(
        self, fake_stac_load, stack
    ):
        lst, qa_count = self._run(fake_stac_load)
        _, warm = stack
        assert int(qa_count[:, 1, 0].sum()) == N_TIME - 2
        assert int(lst[1, 0]) == pytest.approx(
            to_dn(expected_p95(warm, drop_indices={3, 6})), abs=1
        )

    def test_an_all_fill_pixel_becomes_nodata(self, fake_stac_load):
        lst, qa_count = self._run(fake_stac_load)
        assert int(lst[1, 1]) == LST_NODATA_DN
        assert int(qa_count[:, 1, 1].sum()) == 0

    def test_every_finite_output_decodes_inside_the_trusted_range(self, fake_stac_load):
        lst, _ = self._run(fake_stac_load)
        valid = lst != LST_NODATA_DN
        celsius = lst[valid].astype("float64") * LST_SCALE + LST_OFFSET
        assert celsius.min() >= -50.0
        assert celsius.max() <= 80.0


def test_the_two_paths_produce_the_same_raster(monkeypatch, fake_stac_load):
    import pystac

    monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
    shard = shard_lst_p95.Shard(0, 0, 0, 0, NY, NX, (-60.0, -34.0, -59.9, -33.9))
    sharded = shard_lst_p95.process_shard(
        shard, [{} for _ in range(N_TIME)], "EPSG:4326", 1 / 3600, read_threads=1
    )
    lst_u16, qa_count, _ = profile_lst_p95.build_graph(
        [{} for _ in range(N_TIME)],
        (-60.0, -34.0, -59.9, -33.9),
        NX,
        "EPSG:4326",
        1 / 3600,
        time_chunk=N_TIME,
        load_chunk=NX,
    )
    graph_lst = np.asarray(lst_u16.values)
    graph_qa = np.asarray(qa_count.values)

    # The nodata pattern has to be identical: it is the validity rule, encoded.
    np.testing.assert_array_equal(
        sharded["lst_p95"] == LST_NODATA_DN, graph_lst == LST_NODATA_DN
    )
    # The observation counts come from the same mask, so they match exactly.
    np.testing.assert_array_equal(sharded["qa_count"], graph_qa)
    # The temperatures agree to one DN, or 0.01 C. The two reductions differ
    # only in where they cast float32.
    np.testing.assert_allclose(
        sharded["lst_p95"].astype("int64"), graph_lst.astype("int64"), atol=1
    )

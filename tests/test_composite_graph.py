"""The lazy graph: its shape, its guards, and its pixels against a hand oracle.

Nothing here starts a cluster or reads S3. `odc.stac.load` is replaced by a
fixture that returns the ten-scene, 4 x 4 synthetic stack with every rejection
reason represented, chunked the way the loader chunks it, so only the graph,
the kernel, the corrections, and the encoder are under test.

Three guards matter more than the pixels, because they are what the earlier
array graph lacked:

* the time axis is one chunk, and a time-chunked input raises;
* the task count scales with the block count, not the scene count;
* the driver does no reprojection while building the graph.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import composite  # noqa: E402
import destripe  # noqa: E402
from lst_qa import (  # noqa: E402
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    LWIR_OFFSET_C,
    LWIR_SCALE,
    encode_celsius,
    masked_celsius,
)
from masks import transform_for  # noqa: E402

QA_CLEAR = 0b1000000
QA_CLOUD = QA_CLEAR | (1 << 3)
QA_CIRRUS = QA_CLEAR | (1 << 2)
QA_SNOW = QA_CLEAR | (1 << 5)

NY = NX = 4
N_TIME = 10
AREA = (-60.0, -34.0, -59.9, -33.9)
#: The synthetic grid: 40 px per degree, so the 0.1 degree AREA is 4 x 4.
PPD = NX * 10
RES = 1.0 / PPD
CORRECTED_PATHS = ("228", "229")


def dn_of(celsius: float) -> int:
    return int(round((celsius - LWIR_OFFSET_C) / LWIR_SCALE))


def to_dn(celsius: float) -> int:
    return int(np.rint((celsius - LST_OFFSET) / LST_SCALE))


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


def with_geobox(dataset, *, time_chunk=-1, chunk=NX):
    """Chunk the fixture the way the loader would, and give it a geobox."""
    from odc.geo import xr as gxr
    from odc.geo.geobox import GeoBox

    gbox = GeoBox((NY, NX), transform_for(AREA, PPD), "EPSG:4326")
    coords = gxr.xr_coords(gbox)
    out = dataset.copy(deep=True).chunk(
        {"time": time_chunk, "latitude": chunk, "longitude": chunk}
    )
    out = out.assign_coords(
        {k: v for k, v in coords.items() if k in ("latitude", "longitude")}
    )
    return gxr.assign_crs(out, "EPSG:4326")


@pytest.fixture
def stack():
    return build_stack()


@pytest.fixture
def fake_load(monkeypatch, stack):
    """Replace the loader with the synthetic stack, time in one chunk."""
    import odc.stac
    import pystac

    dataset = stack[0]
    monkeypatch.setattr(odc.stac, "load", lambda *_a, **_k: with_geobox(dataset))
    monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
    return dataset


def corrected_items(stack):
    """One item per scene of the fixture, alternating between two WRS paths."""
    times = stack[0]["time"].values
    return [
        {
            "properties": {
                "datetime": str(np.datetime_as_string(t, unit="us")) + "Z",
                "landsat:scene_id": f"SCENE{i:02d}",
                "landsat:wrs_path": CORRECTED_PATHS[i % 2],
                "landsat:wrs_row": "030",
            }
        }
        for i, t in enumerate(times)
    ]


def prep_for(*, uncovered_cols=0):
    """A prep artifact on the fixture's own grid, with one scene rejected.

    `uncovered_cols` pulls both swaths back off that many leading columns, so
    those pixels carry observations that no path's swath reaches and the
    pooled fallback has to fire.
    """
    west = np.zeros((NY, NX), dtype=bool)
    west[:, uncovered_cols:3] = True
    east = np.zeros((NY, NX), dtype=bool)
    east[:, max(1, uncovered_cols) :] = True
    paths, weight, inside = destripe.path_weights(
        {CORRECTED_PATHS[0]: west, CORRECTED_PATHS[1]: east},
        transform_for(AREA, PPD),
        factor=1,
    )
    offset = np.linspace(-2.0, 2.0, N_TIME)
    offset[4] = -40.0  # past the cap, so this scene has to be discarded
    return destripe.Prep(
        tile="T",
        bbox=AREA,
        pixels_per_degree=PPD,
        swath_factor=1,
        paths=paths,
        weight=weight,
        inside=inside,
        offset={f"SCENE{i:02d}": float(offset[i]) for i in range(N_TIME)},
        n_valid={f"SCENE{i:02d}": 9_000 for i in range(N_TIME)},
        meta={},
    )


def oracle(stack, items, prep, *, uncovered_cols=0):
    """The same kernels, called by hand on the eager stack."""
    dataset = stack[0]
    celsius, valid = masked_celsius(
        dataset["lwir11"].values.copy(), dataset["qa_pixel"].values.copy()
    )
    times = dataset["time"].values
    offset, keep, code, paths = composite.per_scene_vectors(
        items, times, prep, max_offset_c=destripe.DESTRIPE_MAX_OFFSET_C, debias=True
    )
    celsius[~keep] = np.nan
    valid[~keep] = False
    destripe.subtract_offsets(celsius, np.where(keep, offset, 0.0))
    west = np.zeros((NY, NX), dtype=bool)
    west[:, uncovered_cols:3] = True
    east = np.zeros((NY, NX), dtype=bool)
    east[:, max(1, uncovered_cols) :] = True
    _, weight, _ = destripe.path_weights(
        {CORRECTED_PATHS[0]: west, CORRECTED_PATHS[1]: east},
        transform_for(AREA, PPD),
        factor=1,
    )
    with np.errstate(all="ignore"):
        p95, fallback = destripe.feathered_percentile(
            celsius, code, tuple(range(len(paths))), weight
        )
    return encode_celsius(p95), valid, fallback


def graph(items, **kw):
    return composite.build_graph(
        items, AREA, crs="EPSG:4326", resolution=RES, chunk=NX, **kw
    )


def computed(out):
    with dask.config.set(scheduler="sync"):
        return out.compute()


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------


class TestTheTimeAxis:
    def test_it_is_one_chunk(self, fake_load):
        data = composite.open_stack(
            [{} for _ in range(N_TIME)], AREA, crs="EPSG:4326", resolution=RES, chunk=NX
        )
        assert data["lwir11"].chunks[0] == (N_TIME,)
        assert data["qa_pixel"].chunks[0] == (N_TIME,)

    def test_a_time_chunked_stack_is_refused(self, monkeypatch, stack):
        import odc.stac
        import pystac

        monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
        monkeypatch.setattr(
            odc.stac, "load", lambda *_a, **_k: with_geobox(stack[0], time_chunk=5)
        )
        with pytest.raises(RuntimeError, match="time axis in 2 chunks"):
            composite.open_stack(
                [{} for _ in range(N_TIME)],
                AREA,
                crs="EPSG:4326",
                resolution=RES,
                chunk=NX,
            )

    def test_apply_ufunc_refuses_a_chunked_core_dim(self, monkeypatch, stack):
        """The second line of defence, in case the first is ever removed."""
        import odc.stac
        import pystac

        monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
        monkeypatch.setattr(
            odc.stac, "load", lambda *_a, **_k: with_geobox(stack[0], time_chunk=5)
        )
        monkeypatch.setattr(composite, "open_stack", lambda *a, **k: odc.stac.load([]))
        with pytest.raises(ValueError, match="multiple chunks"):
            graph([{} for _ in range(N_TIME)])

    def test_quantile_is_not_used(self):
        source = (ROOT / "composite.py").read_text()
        assert ".quantile(" not in source


class TestTheGraphShape:
    """Fake items over a 36 x 36 block tile. No pixels, only the graph."""

    @staticmethod
    def fake_items(n: int, bbox):
        """Items with real footprints and absent files: the graph never reads."""
        import pystac

        w, s, e, no = bbox
        items = []
        for i in range(n):
            west = w + (e - w) * (i % 9) / 9
            south = s + (no - s) * (i // 9 % 9) / 9
            items.append(
                pystac.Item.from_dict(
                    fake_item(
                        f"FAKE{i:05d}",
                        (west, south, west + 1.7, south + 1.7),
                        f"2023-{1 + i % 12:02d}-{1 + i // 12 % 28:02d}T12:00:00Z",
                        f"{220 + i % 6:03d}",
                    )
                )
            )
        return items

    @pytest.mark.parametrize("n_scenes", [50, 500])
    def test_one_reduce_task_per_block_whatever_the_scene_count(self, n_scenes):
        bbox = (-65.0, -35.0, -60.0, -30.0)
        chunk = 500
        items = self.fake_items(n_scenes, bbox)
        t0 = time.perf_counter()
        out = composite.build_graph(
            items, bbox, crs="EPSG:4326", resolution=1 / 3600, chunk=chunk
        )
        build_s = time.perf_counter() - t0
        keys = list(dict(out["lst_p95"].__dask_graph__()))
        layer = [
            (k[0] if isinstance(k, tuple) else str(k)).rsplit("-", 1)[0] for k in keys
        ]
        reduce_tasks = sum(1 for name in layer if name == "reduce_block")
        assert out["lst_p95"].data.npartitions == 36 * 36
        assert reduce_tasks == 36 * 36, f"{reduce_tasks} reduce tasks"
        assert not any("rechunk" in name for name in layer)
        # Bounded per block: the reduce, two band reads, the output splits,
        # the transposes apply_gufunc adds, and nothing that grows with scenes.
        assert len(keys) <= 16 * 36 * 36 + 4 * n_scenes + 16
        # MEASURED here on every run; asserted loosely so a slow CI box passes.
        assert build_s < 60.0, f"graph build took {build_s:.1f}s"

    def test_the_total_grows_by_a_few_tasks_per_scene_not_per_block(self):
        bbox = (-65.0, -35.0, -60.0, -30.0)
        counts = {}
        for n in (50, 500):
            out = composite.build_graph(
                self.fake_items(n, bbox),
                bbox,
                crs="EPSG:4326",
                resolution=1 / 3600,
                chunk=500,
            )
            counts[n] = len(dict(out["lst_p95"].__dask_graph__()))
        per_scene = (counts[500] - counts[50]) / 450
        # odc opens each scene once per band, lazily. Anything beyond a handful
        # of tasks per added scene means a per-scene-per-block layer is back.
        assert per_scene <= 4, f"{per_scene:.1f} tasks per added scene"

    def test_the_driver_does_no_reprojection(self, monkeypatch):
        import rasterio.warp

        def refuse(*_a, **_k):
            raise AssertionError("reproject ran on the driver")

        monkeypatch.setattr(rasterio.warp, "reproject", refuse)
        bbox = (-65.0, -35.0, -60.0, -30.0)
        items = self.fake_items(60, bbox)
        prep = wide_prep(bbox)
        out = composite.build_graph(
            items, bbox, crs="EPSG:4326", resolution=1 / 3600, chunk=500, prep=prep
        )
        assert out.attrs["feather"] is True
        assert out["lst_p95"].data.npartitions == 36 * 36


def fake_item(scene_id, bbox, datetime, path):
    """A STAC item with a footprint and asset hrefs that no task will open."""
    from tile_inventory import ASSET_TEMPLATES

    west, south, east, north = bbox
    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": scene_id,
        "collection": "landsat-c2-l2",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "bbox": list(bbox),
        "properties": {
            "datetime": datetime,
            "landsat:scene_id": scene_id,
            "landsat:wrs_path": path,
            "landsat:wrs_row": "080",
            "proj:epsg": 4326,
            "proj:shape": [102, 102],
            "proj:transform": [1.7 / 102, 0.0, west, 0.0, -1.7 / 102, north],
        },
        "assets": {
            band: {
                **ASSET_TEMPLATES[band],
                "href": f"/nonexistent/{scene_id}_{band}.tif",
            }
            for band in ("lwir11", "qa_pixel")
        },
        "links": [],
    }


def wide_prep(bbox):
    """A prep artifact with six paths over the whole tile, on a coarse grid."""
    w, s, e, n = bbox
    padded = (w - 1, s - 1, e + 1, n + 1)
    h = wd = 70
    masks = {}
    for j in range(6):
        m = np.zeros((h, wd), dtype=bool)
        m[:, max(0, j * 12 - 4) : min(wd, j * 12 + 18)] = True
        masks[f"{220 + j:03d}"] = m
    paths, weight, inside = destripe.path_weights(
        masks, transform_for(padded, 10), factor=1
    )
    return destripe.Prep(
        tile="T",
        bbox=padded,
        pixels_per_degree=10,
        swath_factor=1,
        paths=paths,
        weight=weight,
        inside=inside,
        offset={f"FAKE{i:05d}": 0.0 for i in range(1000)},
        n_valid={f"FAKE{i:05d}": 9_000 for i in range(1000)},
        meta={},
    )


# --------------------------------------------------------------------------
# The pixels
# --------------------------------------------------------------------------


class TestPooled:
    """No prep artifact: the pooled percentile with every scene at baseline."""

    def _run(self):
        return computed(graph([{} for _ in range(N_TIME)]))

    def test_a_clear_pixel_keeps_every_observation(self, fake_load, stack):
        out = self._run()
        _, warm = stack
        assert int(out["lst_p95"][0, 0]) == pytest.approx(
            to_dn(expected_p95(warm)), abs=1
        )
        assert int(out["qa_count"][:, 0, 0].sum()) == N_TIME

    def test_qa_flagged_scenes_leave_the_stack(self, fake_load, stack):
        out = self._run()
        _, warm = stack
        assert int(out["qa_count"][:, 0, 1].sum()) == N_TIME - 2
        assert int(out["lst_p95"][0, 1]) == pytest.approx(
            to_dn(expected_p95(warm, drop_indices={2, 7})), abs=1
        )

    def test_the_reprojected_edge_artifact_leaves_the_stack(self, fake_load, stack):
        out = self._run()
        _, warm = stack
        assert int(out["qa_count"][:, 1, 0].sum()) == N_TIME - 2
        assert int(out["lst_p95"][1, 0]) == pytest.approx(
            to_dn(expected_p95(warm, drop_indices={3, 6})), abs=1
        )

    def test_an_all_fill_pixel_becomes_nodata(self, fake_load):
        out = self._run()
        assert int(out["lst_p95"][1, 1]) == LST_NODATA_DN
        assert int(out["qa_count"][:, 1, 1].sum()) == 0

    def test_every_finite_output_decodes_inside_the_trusted_range(self, fake_load):
        out = self._run()
        dn = out["lst_p95"].values
        valid = dn != LST_NODATA_DN
        celsius = dn[valid].astype("float64") * LST_SCALE + LST_OFFSET
        assert celsius.min() >= -50.0
        assert celsius.max() <= 80.0

    def test_the_outputs_have_the_shapes_and_dtypes_the_writer_expects(self, fake_load):
        out = self._run()
        assert out["lst_p95"].dtype == np.uint16
        assert out["qa_count"].dtype == np.uint8
        assert out["qa_count"].dims == ("month", "latitude", "longitude")
        assert out["qa_count"].shape == (12, NY, NX)
        assert out["fallback"].dtype == bool
        assert out.attrs["n_rejected"] == 0


class TestCorrected:
    """A prep artifact on: offsets, rejection, cross-fade, and the fallback."""

    def test_it_matches_the_kernels_called_by_hand(self, fake_load, stack):
        items = corrected_items(stack)
        prep = prep_for()
        out = computed(graph(items, prep=prep))
        want_dn, want_valid, want_fallback = oracle(stack, items, prep)
        np.testing.assert_array_equal(out["lst_p95"].values, want_dn)
        np.testing.assert_array_equal(out["fallback"].values, want_fallback)
        assert out.attrs["n_rejected"] == 1
        assert out.attrs["paths"] == list(CORRECTED_PATHS)

    def test_a_rejected_scene_leaves_the_monthly_counts(self, fake_load, stack):
        items = corrected_items(stack)
        out = computed(graph(items, prep=prep_for()))
        # Pixel (0, 0) is clear in every scene, so it loses exactly the one
        # that was discarded.
        assert int(out["qa_count"][:, 0, 0].sum()) == N_TIME - 1

    def test_the_fallback_fires_where_no_swath_reaches(self, fake_load, stack):
        items = corrected_items(stack)
        prep = prep_for(uncovered_cols=1)
        out = computed(graph(items, prep=prep))
        want_dn, _, want_fallback = oracle(stack, items, prep, uncovered_cols=1)
        assert int(out["fallback"].sum()) == NY
        np.testing.assert_array_equal(out["fallback"].values, want_fallback)
        np.testing.assert_array_equal(out["lst_p95"].values, want_dn)

    def test_nodata_exactly_where_nothing_was_observed(self, fake_load, stack):
        items = corrected_items(stack)
        out = computed(graph(items, prep=prep_for(uncovered_cols=1)))
        np.testing.assert_array_equal(
            out["lst_p95"].values == LST_NODATA_DN,
            out["qa_count"].values.sum(axis=0) == 0,
        )
        assert int(out["lst_p95"][1, 1]) == LST_NODATA_DN
        assert (out["lst_p95"].values != LST_NODATA_DN).sum() == NY * NX - 1

    def test_no_destripe_zeroes_the_offsets_and_keeps_every_scene(
        self, fake_load, stack
    ):
        items = corrected_items(stack)
        out = graph(items, prep=prep_for(), debias=False)
        assert out.attrs["n_rejected"] == 0
        assert out.attrs["debias"] is False

    def test_no_feather_composites_pooled(self, fake_load, stack):
        items = corrected_items(stack)
        out = computed(graph(items, prep=prep_for(), feather=False))
        assert out.attrs["feather"] is False
        # Every observed pixel took the pooled reduction.
        np.testing.assert_array_equal(
            out["fallback"].values, out["lst_p95"].values != LST_NODATA_DN
        )

    def test_emit_pooled_adds_a_fourth_output(self, fake_load, stack):
        items = corrected_items(stack)
        out = computed(graph(items, prep=prep_for(), emit_pooled=True))
        assert "lst_p95_pooled" in out
        assert out["lst_p95_pooled"].dtype == np.uint16
        pooled = out["lst_p95_pooled"].values
        assert (pooled != LST_NODATA_DN).sum() == NY * NX - 1


class TestTheFinish:
    """`staging_writes`: masks on the block, windows into the files, flags out."""

    def _finish(self, tmp_path, **masks):
        import rasterio

        out = graph([{} for _ in range(N_TIME)])
        counts, paths = composite.staging_writes(
            out, tmp_path, crs="EPSG:4326", dims=("latitude", "longitude"), **masks
        )
        with dask.config.set(scheduler="sync"):
            flags = composite.compute_all(counts)
        with rasterio.open(paths["lst_p95"]) as src:
            lst = src.read(1)
        with rasterio.open(paths["qa_count"]) as src:
            qa = src.read()
        return out, flags, lst, qa

    def test_unmasked_the_files_hold_the_graph_pixel_for_pixel(
        self, fake_load, tmp_path
    ):
        out, flags, lst, qa = self._finish(tmp_path)
        np.testing.assert_array_equal(lst, computed(out)["lst_p95"].values)
        np.testing.assert_array_equal(qa, computed(out)["qa_count"].values)
        assert flags["valid"] == int((lst != LST_NODATA_DN).sum())
        assert flags["removed_water"] == flags["removed_hot"] == flags["qa_zeroed"] == 0

    def test_water_removes_both_rasters_and_the_gap_removes_only_hot(
        self, fake_load, tmp_path
    ):
        keep = np.ones((NY, NX), dtype=bool)
        keep[3, :] = False  # the last row is sea
        gap = np.zeros((NY, NX), dtype=bool)
        gap[0, :] = True  # the first row is inside the ASTER gap region
        _, flags, lst, qa = self._finish(
            tmp_path,
            keep_mask=keep,
            gap_mask=gap,
            hot_dn=to_dn(30.0),  # the fixture's clear pixels reach 44 C
        )
        assert (lst[3, :] == LST_NODATA_DN).all()
        assert qa[:, 3, :].sum() == 0
        assert (lst[0, :] == LST_NODATA_DN).all()
        assert qa[:, 0, 0].sum() == N_TIME  # the gap rule leaves the counts alone
        # Row 3 is clear in every scene, so the water rule removes all of it.
        assert flags["removed_water"] == NX
        assert flags["removed_hot"] == NX
        assert flags["qa_zeroed"] == NX
        assert flags["valid"] == int((lst != LST_NODATA_DN).sum())

    def test_the_counts_are_one_dataarray(self, fake_load, tmp_path):
        out = graph([{} for _ in range(N_TIME)])
        counts, _ = composite.staging_writes(
            out, tmp_path, crs="EPSG:4326", dims=("latitude", "longitude")
        )
        assert isinstance(counts, xr.DataArray)
        assert counts.dims == ("flag",)
        assert list(counts["flag"].values) == list(composite.FLAGS)

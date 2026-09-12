"""The fused kernel against the graph engine, block for block and bit for bit.

`tests/test_block_plan.py` pins the plan, the pruning, and the submission with
a stub kernel. This pins the kernel: `composite.read_block` against the block
`odc.stac.load` produces, `composite.block_weights` against the lazy weight
field, and `composite.fused_block` against the whole graph, staging file for
staging file.

Nothing here starts a cluster or opens a socket. The graph side runs under the
synchronous scheduler; the driver-level comparison of the two engines lives in
`tests/test_composite_run.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import dask
import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import composite  # noqa: E402
import destripe  # noqa: E402
from masks import transform_for  # noqa: E402

#: A window wide enough that the 7 x 7 rehearsal walk leaves part of it
#: uncovered, so the fill path and the shallow blocks are exercised.
BBOX = (-60.0, -34.0, -59.2, -33.2)
PPD = 360
CHUNK = 72
N_SCENES = 20
RES = 1.0 / PPD


def tile_geobox(bbox=BBOX, pixels_per_degree=PPD, crs="EPSG:4326"):
    """The grid the driver builds for the plan, and `open_stack` lands on."""
    from odc.geo.geobox import GeoBox

    return GeoBox(
        composite.raster_shape(bbox, pixels_per_degree),
        transform_for(bbox, pixels_per_degree),
        crs,
    )


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    """One set of synthetic scenes and their footprints, shared by the module."""
    directory = tmp_path_factory.mktemp("fused-scenes")
    return composite.rehearsal_items(BBOX, N_SCENES, directory / "scenes")


@pytest.fixture(scope="module")
def loaded(scenes):
    """The graph engine's own stack over the same window, computed once."""
    items, _ = scenes
    data = composite.open_stack(
        items, BBOX, crs="EPSG:4326", resolution=RES, chunk=CHUNK
    )
    with dask.config.set(scheduler="sync"):
        return data.compute()


@pytest.fixture(scope="module")
def plan(scenes):
    items, boxes = scenes
    return composite.build_block_plan(items, boxes, tile_geobox(), CHUNK)


def prep_for(paths=("220", "221")):
    """A prep artifact over the window, with two overlapping swaths.

    Coarse enough that the reproject onto the run grid is a real resample, and
    one scene is offset past the cap so the rejection rule fires.
    """
    w, s, e, n = BBOX
    padded = (w - 0.2, s - 0.2, e + 0.2, n + 0.2)
    grid_ppd = 20
    height = int(round((padded[3] - padded[1]) * grid_ppd))
    width = int(round((padded[2] - padded[0]) * grid_ppd))
    west = np.zeros((height, width), dtype=bool)
    west[:, : int(width * 0.7)] = True
    east = np.zeros((height, width), dtype=bool)
    east[:, int(width * 0.4) :] = True
    names, weight, inside = destripe.path_weights(
        dict(zip(paths, (west, east), strict=True)),
        transform_for(padded, grid_ppd),
        factor=1,
    )
    offset = np.linspace(-3.0, 3.0, N_SCENES)
    offset[5] = -40.0  # past the cap: this scene has to be discarded
    return destripe.Prep(
        tile="T",
        bbox=padded,
        pixels_per_degree=grid_ppd,
        swath_factor=1,
        paths=names,
        weight=weight,
        inside=inside,
        offset={f"REHEARSAL{i:05d}": float(offset[i]) for i in range(N_SCENES)},
        n_valid={f"REHEARSAL{i:05d}": 9_000 for i in range(N_SCENES)},
        meta={},
    )


# --------------------------------------------------------------------------
# The grid the plan cuts, against the grid the loader builds
# --------------------------------------------------------------------------


class TestTheGrid:
    def test_the_planned_geobox_is_the_loaders_geobox(self, loaded):
        assert tile_geobox() == loaded.odc.geobox

    def test_the_blocks_are_the_dask_chunks(self, scenes, loaded, plan):
        items, _ = scenes
        data = composite.open_stack(
            items, BBOX, crs="EPSG:4326", resolution=RES, chunk=CHUNK
        )
        chunks_y, chunks_x = data["lwir11"].chunks[1], data["lwir11"].chunks[2]
        assert len(plan) == len(chunks_y) * len(chunks_x)
        for block in plan:
            assert block.shape == (chunks_y[block.row], chunks_x[block.col])
            assert block.geobox == loaded.odc.geobox[block.yslice, block.xslice]


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------


class TestReadBlock:
    def test_every_block_of_every_band_matches_the_loader(self, scenes, loaded, plan):
        """The plan's own subset, placed back on the tile's axis, is the block.

        A block reads only the scenes that reach it, so the comparison puts
        its planes back at their positions on the loaded stack and asserts
        that everything else there is the band's fill.
        """
        items, _ = scenes
        assert len(plan) == 16
        assert min(block.depth for block in plan) < N_SCENES, (
            "no block is pruned; widen the window"
        )
        for band, fill in (("lwir11", 0), ("qa_pixel", 1)):
            want_all = loaded[band].values
            for block in plan:
                subset = [items[i] for i in block.item_indices]
                got = composite.read_block(subset, block.geobox, band)
                want = np.moveaxis(want_all[:, block.yslice, block.xslice], 0, -1)
                np.testing.assert_array_equal(
                    got, want[..., list(block.item_indices)], err_msg=f"{band} {block}"
                )
                skipped = [i for i in range(N_SCENES) if i not in block.item_indices]
                assert (want[..., skipped] == fill).all(), f"{band} {block}"

    def test_the_whole_axis_reproduces_the_loaded_block_exactly(self, scenes, loaded):
        """Handed every item, the reader is the loader, fill values included.

        `qa_pixel` declares `nodata: 1` on its STAC asset while the file on
        disk declares 0, so the loaded qa plane is 1 where no scene reaches.
        The reader has to reproduce that, not a tidier 0, or the two engines
        drift on the first block a swath misses.

        `odc.stac.load` sorts its axis by the groupby key, which is the scene
        id, and the rehearsal's ids are in list order, so the item list is the
        loaded axis here and the comparison is positional. The plan orders a
        block's scenes by acquisition stamp instead, which is a different
        order and the same multiset: the percentile sorts, the monthly counts
        add, and each scene's offset travels with it, so the reduction cannot
        see the difference. The whole-graph tests below are what prove that.
        """
        items, _ = scenes
        np.testing.assert_array_equal(
            composite.item_times(items), loaded["time"].values
        )
        block = composite.build_block_plan(
            items, [i["bbox"] for i in items], tile_geobox(), CHUNK
        )[0]
        for band in ("lwir11", "qa_pixel"):
            got = composite.read_block(items, block.geobox, band)
            want = np.moveaxis(
                loaded[band].values[:, block.yslice, block.xslice], 0, -1
            )
            np.testing.assert_array_equal(got, want)
        lwir = composite.read_block(items, block.geobox, "lwir11")
        qa = composite.read_block(items, block.geobox, "qa_pixel")
        missing = lwir == 0
        assert missing.any(), "this block is fully covered; widen the window"
        assert (qa[missing] == 1).all()

    def test_no_items_is_an_empty_time_axis(self, plan):
        got = composite.read_block([], plan[0].geobox, "lwir11")
        assert got.shape == (CHUNK, CHUNK, 0)
        assert got.dtype == np.uint16


# --------------------------------------------------------------------------
# The weights
# --------------------------------------------------------------------------


class TestBlockWeights:
    def test_they_match_the_lazy_field_block_for_block(self, loaded, plan):
        prep = prep_for()
        lazy = composite.lazy_weights(
            prep,
            loaded.odc.geobox,
            chunk=CHUNK,
            dims=composite.spatial_dims("EPSG:4326"),
        )
        with dask.config.set(scheduler="sync"):
            want_all = np.asarray(lazy.values)
        for block in plan:
            got = composite.block_weights(prep, block.geobox)
            want = np.moveaxis(want_all[:, block.yslice, block.xslice], 0, -1)
            np.testing.assert_array_equal(got, want, err_msg=str(block))


# --------------------------------------------------------------------------
# The whole task
# --------------------------------------------------------------------------


def graph_run(
    tmp_path, items, *, prep=None, emit_pooled=False, feather=True, rule=None
):
    """The graph engine into its own directory, under the sync scheduler."""
    out = composite.build_graph(
        items,
        BBOX,
        crs="EPSG:4326",
        resolution=RES,
        chunk=CHUNK,
        prep=prep,
        feather=feather,
        emit_pooled=emit_pooled,
    )
    counts, paths = composite.staging_writes(
        out,
        tmp_path,
        crs="EPSG:4326",
        dims=composite.spatial_dims("EPSG:4326"),
        **(rule or {}),
    )
    with dask.config.set(scheduler="sync"):
        scalars = composite.compute_all(counts)
    return scalars, paths, out.attrs


def fused_run(
    tmp_path,
    items,
    boxes,
    *,
    prep=None,
    kernel_prep=...,
    emit_pooled=False,
    feather=True,
    rule=None,
):
    """The fused kernel over the plan, in this process, block by block.

    `kernel_prep` is what the tasks are handed, which the driver sets to None
    under `--no-feather` while the vectors still come from the real artifact.
    """
    from masks import gap_hot_dn

    kernel_prep = prep if kernel_prep is ... else kernel_prep
    rule = dict(rule or {})
    plan = composite.build_block_plan(items, boxes, tile_geobox(), CHUNK)
    vectors = composite.scene_vectors(items, prep)
    targets, paths = composite.staging_targets(
        tmp_path,
        shape=composite.raster_shape(BBOX, PPD),
        transform=transform_for(BBOX, PPD),
        crs="EPSG:4326",
        emit_pooled=emit_pooled,
    )
    keep_mask = rule.get("keep_mask")
    outputs = composite.BlockOutputs(
        targets=targets,
        keep=keep_mask,
        gap=rule.get("gap_mask"),
        hot_dn=(
            rule.get("hot_dn")
            if keep_mask is None or rule.get("hot_dn") is not None
            else gap_hot_dn()
        ),
    )
    results = [
        composite.fused_block(
            block,
            [items[i] for i in block.item_indices],
            composite.block_vectors(vectors, block),
            kernel_prep,
            outputs,
            emit_pooled=emit_pooled,
            feather=feather,
        )
        for block in plan
    ]
    scalars = {flag: sum(int(r[flag]) for r in results) for flag in composite.FLAGS}
    return scalars, paths, vectors, results


def read_all(paths):
    out = {}
    for name, path in paths.items():
        with rasterio.open(path) as src:
            out[name] = src.read()
    return out


def assert_same_files(got_paths, want_paths):
    want, got = read_all(want_paths), read_all(got_paths)
    assert set(got) == set(want)
    for name in want:
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)


class TestFusedBlock:
    @pytest.mark.parametrize("with_prep", [False, True])
    def test_the_staging_files_and_the_counts_match_the_graph(
        self, tmp_path, scenes, with_prep
    ):
        items, boxes = scenes
        prep = prep_for() if with_prep else None
        want_scalars, want_paths, want_attrs = graph_run(
            tmp_path / "graph", items, prep=prep
        )
        got_scalars, got_paths, vectors, results = fused_run(
            tmp_path / "fused", items, boxes, prep=prep
        )
        assert_same_files(got_paths, want_paths)
        assert got_scalars == want_scalars
        assert got_scalars["valid"] > 0
        assert list(vectors.paths) == want_attrs["paths"]
        # The prep run has to reach the feathered branch and drop the capped
        # scene, or this compares two pooled runs and proves nothing.
        assert want_attrs["feather"] is with_prep
        assert vectors.n_rejected == want_attrs["n_rejected"] == (1 if with_prep else 0)
        assert len(results) == 16
        assert all(r["block_s"] > 0 for r in results)
        assert sum(r["depth"] for r in results) < 16 * N_SCENES

    def test_the_pooled_baseline_matches_too(self, tmp_path, scenes):
        items, boxes = scenes
        prep = prep_for()
        want_scalars, want_paths, _ = graph_run(
            tmp_path / "graph", items, prep=prep, emit_pooled=True
        )
        got_scalars, got_paths, _, _ = fused_run(
            tmp_path / "fused", items, boxes, prep=prep, emit_pooled=True
        )
        assert "lst_p95_pooled" in got_paths
        assert_same_files(got_paths, want_paths)
        assert got_scalars == want_scalars

    def test_no_feather_takes_the_pooled_reduction(self, tmp_path, scenes):
        """A prep artifact with the cross-fade off: no weight warp, pooled p95.

        Two ways of saying it, because the driver uses the second: the kernel
        flag, and withholding the prep artifact from the tasks while the
        offsets and the rejection still come from it through the vectors.
        """
        items, boxes = scenes
        prep = prep_for()
        _, want_paths, want_attrs = graph_run(
            tmp_path / "graph", items, prep=prep, feather=False
        )
        assert want_attrs["feather"] is False
        _, flagged, _, results = fused_run(
            tmp_path / "flagged", items, boxes, prep=prep, feather=False
        )
        assert_same_files(flagged, want_paths)
        _, withheld, vectors, _ = fused_run(
            tmp_path / "withheld", items, boxes, prep=prep, kernel_prep=None
        )
        assert_same_files(withheld, want_paths)
        # The rejection still fired: the vectors came from the real artifact.
        assert vectors.n_rejected == want_attrs["n_rejected"] == 1
        assert all(r["weights_s"] < r["block_s"] for r in results)

    def test_the_output_mask_lands_on_the_same_pixels(self, tmp_path, scenes):
        items, boxes = scenes
        height, width = composite.raster_shape(BBOX, PPD)
        keep = np.ones((height, width), dtype=bool)
        keep[-CHUNK:, :] = False  # the bottom block row is sea
        gap = np.zeros((height, width), dtype=bool)
        gap[:CHUNK, :] = True  # the top block row is inside the ASTER gap
        rule = {"keep_mask": keep, "gap_mask": gap, "hot_dn": 5_000}
        want_scalars, want_paths, _ = graph_run(tmp_path / "graph", items, rule=rule)
        got_scalars, got_paths, _, _ = fused_run(
            tmp_path / "fused", items, boxes, rule=rule
        )
        assert_same_files(got_paths, want_paths)
        assert got_scalars == want_scalars
        assert got_scalars["removed_water"] > 0
        assert got_scalars["removed_hot"] > 0

    def test_a_block_no_scene_reaches_still_writes_its_nodata(self, tmp_path, scenes):
        """An empty plan entry owns its window and has to fill it."""
        items, boxes = scenes
        _, paths, _, results = fused_run(tmp_path / "fused", [], [], prep=None)
        assert len(results) == 16
        assert all(r["depth"] == 0 for r in results)
        with rasterio.open(paths["lst_p95"]) as src:
            assert (src.read(1) == 0).all()
        with rasterio.open(paths["qa_count"]) as src:
            assert (src.read() == 0).all()

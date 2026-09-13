"""The block plan, the per-block vectors, and the fused submission loop.

`tests/test_composite_graph.py` pins the lazy graph. This pins the path that
replaces its driver half: `composite.build_block_plan` cuts the tile into
blocks and gives each one the scenes whose footprint reaches it,
`composite.scene_vectors` builds the per-scene arrays without opening a stack,
and `composite.submit_blocks` sends one task per block at the workers.

The per-block kernel is not tested here. `submit_blocks` takes it as an
argument and every test passes a stub, so what is pinned is the plan, the
pruning, the placement, and the accounting.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import composite  # noqa: E402
import destripe  # noqa: E402
from masks import transform_for  # noqa: E402

#: A five degree tile at 3,600 px per degree: 18,000 px, 50 x 50 blocks of 360.
TILE_BBOX = (-65.0, -35.0, -60.0, -30.0)
PPD = 3600
CHUNK = 360


def geobox_for(bbox, pixels_per_degree=PPD, crs="EPSG:4326"):
    from odc.geo.geobox import GeoBox

    shape = composite.raster_shape(bbox, pixels_per_degree)
    return GeoBox(shape, transform_for(bbox, pixels_per_degree), crs)


def walk_items(n, bbox, *, edge=1.7, seed=0):
    """`n` items on a 7 x 7 walk across `bbox`, with shuffled acquisition times.

    The stamps are deliberately out of list order, so any test that depends on
    the plan's time ordering fails when the plan stops sorting.
    """
    w, s, e, no = bbox
    rng = np.random.default_rng(seed)
    minutes = rng.permutation(n)
    items, boxes = [], []
    for i in range(n):
        west = w + (e - w) * (i % 7) / 7 - 0.3
        south = s + (no - s) * (i // 7 % 7) / 7 - 0.3
        boxes.append((west, south, west + edge, south + edge))
        stamp = np.datetime64("2023-01-01T00:00:00", "s") + np.timedelta64(
            int(minutes[i]) * 37, "m"
        )
        items.append(
            {
                "properties": {
                    "datetime": f"{np.datetime_as_string(stamp, unit='s')}Z",
                    "landsat:scene_id": f"SCENE{i:05d}",
                    "landsat:wrs_path": f"{220 + i % 6:03d}",
                    "landsat:wrs_row": f"{80 + i % 3:03d}",
                }
            }
        )
    return items, boxes


def prep_for(items, *, rejected=()):
    """A prep artifact naming every item, with `rejected` past the offset cap."""
    paths = tuple(sorted({destripe.path_of(item) for item in items}))
    return destripe.Prep(
        tile="T",
        bbox=TILE_BBOX,
        pixels_per_degree=10,
        swath_factor=1,
        paths=paths,
        weight=np.zeros((len(paths), 4, 4), dtype="float32"),
        inside=np.ones((len(paths), 4, 4), dtype=bool),
        offset={
            destripe.scene_id_of(item): (-99.0 if i in rejected else 0.5)
            for i, item in enumerate(items)
        },
        n_valid={destripe.scene_id_of(item): 9_000 for item in items},
        meta={},
    )


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


class TestTheBlockGrid:
    def test_the_block_count_is_the_raster_shape_over_the_chunk(self):
        items, boxes = walk_items(40, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        height, width = composite.raster_shape(TILE_BBOX, PPD)
        rows, cols = -(-height // CHUNK), -(-width // CHUNK)
        assert (height, width) == (18_000, 18_000)
        assert len(plan) == rows * cols == 2_500
        assert [(b.row, b.col) for b in plan[:3]] == [(0, 0), (0, 1), (0, 2)]
        assert plan[-1].row == rows - 1
        assert plan[-1].col == cols - 1

    def test_the_windows_tile_the_raster_exactly(self):
        items, boxes = walk_items(20, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        painted = np.zeros((18_000, 18_000), dtype="uint8")
        for block in plan:
            painted[block.yslice, block.xslice] += 1
        assert painted.min() == painted.max() == 1

    def test_an_edge_block_is_partial_not_overhanging(self):
        """A tile whose edge is not a multiple of the chunk ends short."""
        bbox = (-65.0, -35.0, -64.0, -34.0)
        ppd = 1_000  # 1,000 px, so 360 leaves a 280 px edge block
        items, boxes = walk_items(12, bbox, edge=0.4)
        plan = composite.build_block_plan(items, boxes, geobox_for(bbox, ppd), CHUNK)
        assert composite.raster_shape(bbox, ppd) == (1_000, 1_000)
        assert len(plan) == 9
        last = plan[-1]
        assert last.shape == (280, 280)
        assert last.yslice == slice(720, 1_000)
        assert last.xslice == slice(720, 1_000)
        assert last.geobox.shape == (280, 280)
        assert plan[0].shape == (CHUNK, CHUNK)

    def test_a_wider_than_tall_raster_is_not_transposed(self):
        """`GeoBox.shape` prints x first and unpacks y first. Pin the order."""
        bbox = (-65.0, -35.0, -63.0, -34.0)  # 2 degrees wide, 1 degree tall
        ppd = 900  # 900 x 1,800 px, so 360 makes 3 block rows and 5 columns
        items, boxes = walk_items(10, bbox, edge=0.4)
        plan = composite.build_block_plan(items, boxes, geobox_for(bbox, ppd), CHUNK)
        assert composite.raster_shape(bbox, ppd) == (900, 1_800)
        assert len(plan) == 3 * 5
        assert plan[-1].yslice == slice(720, 900)
        assert plan[-1].xslice == slice(1_440, 1_800)
        assert plan[-1].shape == (180, 360)
        depths = composite.block_depths(bbox, ppd, CHUNK, boxes)
        assert depths.shape == (3, 5)
        assert np.array_equal(composite.plan_depths(plan, depths.shape), depths)

    def test_a_block_geobox_is_that_block_of_the_tile_geobox(self):
        items, boxes = walk_items(12, TILE_BBOX)
        full = geobox_for(TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, full, CHUNK)
        block = plan[7 * 50 + 3]
        assert block.geobox.crs == full.crs
        assert block.geobox.shape == (CHUNK, CHUNK)
        box = block.geobox.extent.boundingbox
        assert box.left == pytest.approx(-65.0 + 3 * CHUNK / PPD)
        assert box.top == pytest.approx(-30.0 - 7 * CHUNK / PPD)

    def test_a_projected_tile_geobox_is_refused(self):
        items, boxes = walk_items(4, TILE_BBOX)
        from odc.geo.geobox import GeoBox
        from rasterio.transform import from_origin

        utm = GeoBox(
            (1_000, 1_000), from_origin(500_000, 6_000_000, 30, 30), "EPSG:32720"
        )
        with pytest.raises(ValueError, match="compares"):
            composite.build_block_plan(items, boxes, utm, CHUNK)

    def test_a_footprint_list_of_the_wrong_length_is_refused(self):
        items, boxes = walk_items(6, TILE_BBOX)
        with pytest.raises(ValueError, match="footprints"):
            composite.build_block_plan(items, boxes[:-1], geobox_for(TILE_BBOX), CHUNK)


class TestTheItemIndices:
    @pytest.mark.parametrize("n_scenes", [49, 300])
    def test_the_counts_are_block_depths(self, n_scenes):
        items, boxes = walk_items(n_scenes, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        depths = composite.block_depths(TILE_BBOX, PPD, CHUNK, boxes)
        assert np.array_equal(composite.plan_depths(plan, depths.shape), depths)

    def test_an_index_names_a_scene_that_really_overlaps_the_block(self):
        items, boxes = walk_items(60, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        boxes = np.asarray(boxes, dtype="float64")
        for block in (plan[0], plan[1_275], plan[-1]):
            b = block.geobox.extent.boundingbox
            reach = {
                i
                for i, (w, s, e, n) in enumerate(boxes)
                if w < b.right and e > b.left and s < b.top and n > b.bottom
            }
            assert set(block.item_indices) == reach

    def test_the_indices_are_in_time_order(self):
        items, boxes = walk_items(80, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        stamps = composite.item_times(items)
        assert not np.array_equal(np.argsort(stamps, kind="stable"), np.arange(80))
        for block in plan:
            taken = stamps[list(block.item_indices)]
            assert list(taken) == sorted(taken), (block.row, block.col)

    def test_an_empty_block_stays_in_the_plan(self):
        """A block no scene reaches still owns its windows of the output."""
        items, boxes = walk_items(8, TILE_BBOX, edge=0.2)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        empty = [block for block in plan if block.depth == 0]
        assert len(plan) == 2_500
        assert empty, "the fixture is meant to leave blocks no scene reaches"
        assert all(block.item_indices == () for block in empty)

    def test_no_items_at_all_still_plans_every_block(self):
        plan = composite.build_block_plan([], [], geobox_for(TILE_BBOX), CHUNK)
        assert len(plan) == 2_500
        assert {block.depth for block in plan} == {0}

    def test_the_plan_prunes_the_time_axis(self):
        """The point of the plan: a block reads its own scenes, not the tile's."""
        items, boxes = walk_items(400, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        mean_depth = sum(block.depth for block in plan) / len(plan)
        assert mean_depth < 0.5 * len(items)


# --------------------------------------------------------------------------
# The per-scene vectors, and one block's subset of them
# --------------------------------------------------------------------------


class TestTheSceneVectors:
    def test_the_months_are_the_ones_xarray_gives(self):
        import xarray as xr

        items, _ = walk_items(30, TILE_BBOX)
        times = composite.item_times(items)
        expected = xr.DataArray(times, dims=("time",)).dt.month.values
        assert np.array_equal(composite.months_of(times), expected.astype("int8"))

    def test_the_axis_is_the_one_the_loader_would_build(self):
        items, _ = walk_items(20, TILE_BBOX)
        times = composite.item_times(items)
        assert times.dtype == np.dtype("M8[ns]")
        assert list(times) == [destripe.timestamp_of(item) for item in items]

    def test_it_matches_per_scene_vectors_on_the_same_axis(self):
        items, _ = walk_items(24, TILE_BBOX)
        prep = prep_for(items, rejected=(3, 11))
        times = composite.item_times(items)
        offset, keep, code, paths = composite.per_scene_vectors(
            items, times, prep, max_offset_c=destripe.DESTRIPE_MAX_OFFSET_C, debias=True
        )
        vectors = composite.scene_vectors(items, prep)
        assert np.array_equal(vectors.offset, offset)
        assert np.array_equal(vectors.keep, keep)
        assert np.array_equal(vectors.path_code, code)
        assert vectors.paths == paths
        assert vectors.n_rejected == 2

    def test_without_a_prep_every_scene_keeps_its_own_baseline(self):
        items, _ = walk_items(9, TILE_BBOX)
        vectors = composite.scene_vectors(items, None)
        assert len(vectors) == 9
        assert not vectors.offset.any()
        assert vectors.keep.all()
        assert (vectors.path_code == -1).all()
        assert vectors.paths == ()


class TestTheBlockSubset:
    def test_a_block_sees_only_its_own_scenes(self):
        items, boxes = walk_items(120, TILE_BBOX)
        prep = prep_for(items, rejected=(5, 6, 7))
        vectors = composite.scene_vectors(items, prep)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        block = max(plan, key=lambda b: b.depth)
        assert 0 < block.depth < len(items)
        cut = composite.block_vectors(vectors, block)
        assert len(cut) == block.depth
        for j, i in enumerate(block.item_indices):
            assert cut.times[j] == vectors.times[i]
            assert cut.offset[j] == vectors.offset[i]
            assert cut.keep[j] == vectors.keep[i]
            assert cut.path_code[j] == vectors.path_code[i]
            assert cut.month[j] == vectors.month[i]
        assert cut.paths == vectors.paths

    def test_the_block_time_coordinate_is_sorted_and_its_own(self):
        items, boxes = walk_items(120, TILE_BBOX)
        vectors = composite.scene_vectors(items, None)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        block = max(plan, key=lambda b: b.depth)
        cut = composite.block_vectors(vectors, block)
        assert list(cut.times) == sorted(cut.times)
        assert np.array_equal(cut.month, composite.months_of(cut.times))

    def test_a_bare_index_sequence_works_too(self):
        items, _ = walk_items(10, TILE_BBOX)
        vectors = composite.scene_vectors(items, None)
        cut = composite.block_vectors(vectors, (2, 5))
        assert list(cut.times) == [vectors.times[2], vectors.times[5]]

    def test_an_empty_block_gets_empty_vectors(self):
        items, _ = walk_items(10, TILE_BBOX)
        vectors = composite.scene_vectors(items, prep_for(items))
        cut = composite.block_vectors(vectors, ())
        assert len(cut) == 0
        assert cut.offset.dtype == vectors.offset.dtype
        assert cut.paths == vectors.paths


class TestTheTimestampJoin:
    """The pruned list joins on the stamp, exactly as the loaded stack did."""

    @staticmethod
    def colliding(n=6):
        """Items 1 and 2, whose footprints overlap, share one stamp."""
        items, boxes = walk_items(n, TILE_BBOX)
        items[2]["properties"]["datetime"] = items[1]["properties"]["datetime"]
        return items, boxes

    def test_two_items_on_one_stamp_still_raise(self):
        items, _ = self.colliding()
        with pytest.raises(ValueError, match="share the acquisition stamp"):
            composite.scene_vectors(items, prep_for(items))

    def test_the_pruned_list_raises_too(self):
        """A block's own list is the same rule, not a laxer one."""
        items, boxes = self.colliding()
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), CHUNK)
        block = next(b for b in plan if {1, 2} <= set(b.item_indices))
        pruned = [items[i] for i in block.item_indices]
        assert len(pruned) < len(items)
        with pytest.raises(ValueError, match="share the acquisition stamp"):
            composite.scene_vectors(pruned, prep_for(pruned))

    def test_a_stamp_no_item_accounts_for_still_raises(self):
        items, _ = walk_items(6, TILE_BBOX)
        times = composite.item_times(items)
        times[2] = np.datetime64("1999-01-01T00:00:00", "ns")
        with pytest.raises(ValueError, match="no item"):
            composite.per_scene_vectors(
                items,
                times,
                prep_for(items),
                max_offset_c=destripe.DESTRIPE_MAX_OFFSET_C,
                debias=True,
            )


# --------------------------------------------------------------------------
# The submission loop
# --------------------------------------------------------------------------


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class FakeClient:
    """Records every submit and runs the function inline."""

    def __init__(self):
        self.calls: list[dict] = []
        self.scattered: list[str] = []

    def scatter(self, data, workers=(), **_kw):
        self.scattered.extend(workers)
        return ("scattered", id(data))

    def submit(self, fn, *args, workers=None, **kwargs):
        self.calls.append({"args": args, "workers": workers, "kwargs": kwargs})
        return FakeFuture(fn(*args, **kwargs))


class FakeCluster:
    def __init__(self, n=4):
        self._worker_addresses = [f"unix:///w{i}.sock" for i in range(n)]


def counting_stub(block, items, vectors, prep, out, *, emit_pooled=False):
    """A stand-in for `fused_block`: counts what it was handed."""
    assert len(items) == len(vectors) == block.depth
    return {"valid": block.depth, "fallback": 1 if emit_pooled else 0}


def plan_and_vectors(n=90, bbox=TILE_BBOX, chunk=2_000):
    items, boxes = walk_items(n, bbox)
    plan = composite.build_block_plan(items, boxes, geobox_for(bbox), chunk)
    return items, composite.scene_vectors(items, None), plan


class TestSubmitBlocks:
    def test_one_task_per_block(self):
        items, vectors, plan = plan_and_vectors()
        client, cluster = FakeClient(), FakeCluster()
        totals = composite.submit_blocks(
            client, cluster, plan, counting_stub, items=items, vectors=vectors
        )
        assert len(client.calls) == len(plan)
        assert totals["valid"] == sum(block.depth for block in plan)

    def test_the_deepest_blocks_go_first(self):
        items, vectors, plan = plan_and_vectors()
        client, cluster = FakeClient(), FakeCluster()
        composite.submit_blocks(
            client, cluster, plan, counting_stub, items=items, vectors=vectors
        )
        depths = [call["args"][0].depth for call in client.calls]
        assert depths == sorted(depths, reverse=True)
        assert depths[0] > depths[-1], "the fixture is meant to vary in depth"

    def test_placement_is_round_robin_over_the_worker_addresses(self):
        items, vectors, plan = plan_and_vectors()
        client, cluster = FakeClient(), FakeCluster(n=4)
        composite.submit_blocks(
            client, cluster, plan, counting_stub, items=items, vectors=vectors
        )
        placed = [call["workers"][0] for call in client.calls]
        assert placed[:5] == [cluster._worker_addresses[i % 4] for i in range(5)]
        assert sorted(placed).count(cluster._worker_addresses[0]) == pytest.approx(
            len(plan) / 4, abs=1
        )

    def test_a_cluster_with_no_addresses_still_submits(self):
        items, vectors, plan = plan_and_vectors()
        client = FakeClient()
        totals = composite.submit_blocks(
            client,
            cluster=object(),
            plan=plan,
            fn=counting_stub,
            items=items,
            vectors=vectors,
        )
        assert len(client.calls) == len(plan)
        assert all(call["workers"] is None for call in client.calls)
        assert totals["valid"] == sum(block.depth for block in plan)

    def test_each_task_carries_only_its_own_block(self):
        items, vectors, plan = plan_and_vectors()
        client, cluster = FakeClient(), FakeCluster()
        composite.submit_blocks(
            client, cluster, plan, counting_stub, items=items, vectors=vectors
        )
        for call in client.calls:
            block, sent_items, sent_vectors = call["args"][:3]
            assert len(sent_items) == block.depth
            assert len(sent_vectors) == block.depth
            assert [destripe.scene_id_of(i) for i in sent_items] == [
                destripe.scene_id_of(items[j]) for j in block.item_indices
            ]

    def test_the_prep_goes_to_the_tasks_as_a_path_and_is_never_scattered(self):
        """One scatter per worker was lost on 64 workers and failed the run.

        MEASURED on the 8x8 window at 64 slots: `Scattered data scatter-0-76
        was lost before reaching a worker`. The tasks carry the artifact's
        path and each process reads it once, so nothing is scattered.
        """
        items, vectors, plan = plan_and_vectors()
        client, cluster = FakeClient(), FakeCluster(n=4)
        composite.submit_blocks(
            client,
            cluster,
            plan,
            counting_stub,
            items=items,
            vectors=vectors,
            prep="/prep/dir",
        )
        assert client.scattered == []
        assert all(call["args"][3] == "/prep/dir" for call in client.calls)

    def test_a_task_reads_the_prep_from_disk_once_per_process(self, monkeypatch):
        loads: list[str] = []

        def fake_load(path):
            loads.append(str(path))
            return object()

        monkeypatch.setattr(composite.destripe, "load_prep", fake_load)
        composite._prep_from_disk.cache_clear()
        first = composite.resolve_prep("/prep/one")
        second = composite.resolve_prep("/prep/one")
        assert first is second
        assert loads == ["/prep/one"]
        composite._prep_from_disk.cache_clear()

    def test_a_prep_object_passes_through_resolve_unchanged(self):
        items, _, _ = plan_and_vectors()
        prep = prep_for(items)
        assert composite.resolve_prep(prep) is prep
        assert composite.resolve_prep(None) is None

    def test_the_phase_marks_are_the_ones_the_summary_prints(self):
        items, vectors, plan = plan_and_vectors()
        marks: dict = {}
        composite.submit_blocks(
            FakeClient(),
            FakeCluster(),
            plan,
            counting_stub,
            items=items,
            vectors=vectors,
            marks=marks,
        )
        assert set(marks) == {"graph_build_s", "compute_s"}

    def test_emit_pooled_reaches_the_kernel(self):
        items, vectors, plan = plan_and_vectors()
        client = FakeClient()
        totals = composite.submit_blocks(
            client,
            FakeCluster(),
            plan,
            counting_stub,
            items=items,
            vectors=vectors,
            emit_pooled=True,
        )
        assert all(call["kwargs"] == {"emit_pooled": True} for call in client.calls)
        assert totals["fallback"] == len(plan)

    def test_what_a_submit_pickles_scales_with_the_block_not_the_tile(self):
        """MEASURED here: frisky's submit serialises its arguments per call."""
        items, vectors, plan = plan_and_vectors(n=400)
        client, cluster = FakeClient(), FakeCluster()
        composite.submit_blocks(
            client, cluster, plan, counting_stub, items=items, vectors=vectors
        )
        whole_axis = len(pickle.dumps((items, vectors)))
        biggest = max(len(pickle.dumps(call["args"][1:3])) for call in client.calls)
        deepest = max(block.depth for block in plan)
        assert biggest < whole_axis * (deepest / len(items)) * 1.5
        assert biggest < 0.5 * whole_axis


class TestTheBlockOutputs:
    def test_the_mask_planes_are_cut_to_the_block(self):
        items, boxes = walk_items(6, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), 6_000)
        keep = np.zeros((18_000, 18_000), dtype=bool)
        keep[6_100, 12_050] = True
        outputs = composite.BlockOutputs(
            targets={}, keep=keep, gap=np.ones((18_000, 18_000), dtype=bool), hot_dn=7
        )
        assert outputs.masked
        cut = outputs.for_block(plan[1 * 3 + 2])
        assert cut.keep.shape == cut.gap.shape == (6_000, 6_000)
        assert cut.keep[100, 50]
        assert cut.keep.sum() == 1
        assert cut.gap.all()
        assert cut.hot_dn == 7
        assert cut.targets is outputs.targets

    def test_an_unmasked_run_carries_no_planes(self):
        outputs = composite.BlockOutputs(targets={"lst_p95": ("p", None)})
        block = composite.BlockSpec(0, 0, None, slice(0, 4), slice(0, 4), ())
        assert not outputs.masked
        assert outputs.for_block(block) is outputs

    def test_submit_blocks_hands_each_task_its_own_window(self):
        items, vectors, plan = plan_and_vectors(chunk=6_000)
        keep = np.zeros((18_000, 18_000), dtype=bool)
        keep[:6_000] = True
        outputs = composite.BlockOutputs(targets={}, keep=keep)
        client = FakeClient()

        def seen(block, _items, _vectors, _prep, out, *, emit_pooled=False):
            return {"valid": int(out.keep.sum())}

        totals = composite.submit_blocks(
            client,
            FakeCluster(),
            plan,
            seen,
            items=items,
            vectors=vectors,
            out=outputs,
        )
        assert totals["valid"] == int(keep.sum())
        assert all(
            call["args"][4].keep.shape == (6_000, 6_000) for call in client.calls
        )

    def test_cutting_a_cut_window_again_returns_it_unchanged(self):
        """The driver cuts and the worker cuts, and one window has to survive.

        `submit_blocks` cuts before it submits and `fused_block` cuts again on
        the worker. A block's slices are absolute, so the second cut would
        index the tile's offset into a 6,000 px window and hand
        `finalize_block` an empty array.
        """
        items, boxes = walk_items(6, TILE_BBOX)
        plan = composite.build_block_plan(items, boxes, geobox_for(TILE_BBOX), 6_000)
        block = plan[1 * 3 + 2]
        keep = np.zeros((18_000, 18_000), dtype=bool)
        keep[6_100, 12_050] = True
        outputs = composite.BlockOutputs(
            targets={}, keep=keep, gap=np.ones((18_000, 18_000), dtype=bool), hot_dn=7
        )

        once = outputs.for_block(block)
        twice = once.for_block(block)

        assert twice is once
        assert twice.keep.shape == twice.gap.shape == (6_000, 6_000)
        assert twice.keep.sum() == 1
        assert twice.hot_dn == 7

    def test_the_window_a_submitted_task_cuts_is_the_block_not_an_empty_array(self):
        """The production pair, driver cut then worker cut, on a masked run.

        `counting_stub` never touches `out`, so the suite proved nothing about
        this pair until the 12-scene rehearsal raised on it.
        """
        items, vectors, plan = plan_and_vectors(chunk=6_000)
        keep = np.zeros((18_000, 18_000), dtype=bool)
        keep[:6_000] = True
        outputs = composite.BlockOutputs(targets={}, keep=keep)
        shapes = []

        def cuts_the_way_fused_block_does(
            block, _items, _vectors, _prep, out, *, emit_pooled=False
        ):
            if hasattr(out, "for_block"):
                out = out.for_block(block)
            shapes.append(out.keep.shape)
            return {"valid": int(out.keep.sum())}

        totals = composite.submit_blocks(
            FakeClient(),
            FakeCluster(),
            plan,
            cuts_the_way_fused_block_does,
            items=items,
            vectors=vectors,
            out=outputs,
        )

        assert shapes and all(shape == (6_000, 6_000) for shape in shapes)
        assert totals["valid"] == int(keep.sum())


# --------------------------------------------------------------------------
# The staging files both engines write into
# --------------------------------------------------------------------------


class TestStagingTargets:
    def test_it_creates_the_files_the_blocks_write_into(self, tmp_path):
        targets, paths = composite.staging_targets(
            tmp_path / "run",
            shape=(64, 48),
            transform=transform_for(TILE_BBOX, PPD),
            crs="EPSG:4326",
        )
        import rasterio

        assert set(targets) == set(paths) == {"lst_p95", "qa_count"}
        with rasterio.open(paths["lst_p95"]) as src:
            assert (src.count, src.dtypes[0], src.height, src.width) == (
                1,
                "uint16",
                64,
                48,
            )
            assert src.nodata == 0
        with rasterio.open(paths["qa_count"]) as src:
            assert (src.count, src.dtypes[0]) == (12, "uint8")
            assert src.nodata is None
        path, lock = targets["lst_p95"]
        assert path == str(paths["lst_p95"])
        assert isinstance(lock, composite.FileLock)

    def test_emit_pooled_adds_the_third_file(self, tmp_path):
        _, paths = composite.staging_targets(
            tmp_path / "run",
            shape=(8, 8),
            transform=transform_for(TILE_BBOX, PPD),
            crs="EPSG:4326",
            emit_pooled=True,
        )
        assert set(paths) == {"lst_p95", "qa_count", "lst_p95_pooled"}
        assert all(p.exists() for p in paths.values())


# --------------------------------------------------------------------------
# The driver's fused branch
# --------------------------------------------------------------------------


class TestTheDriverBranch:
    """`shard_lst_p95.run_fused`, with a stub in place of the block kernel."""

    @staticmethod
    def args_for(tmp_path):
        import shard_lst_p95

        # 5 degrees at 120 px per degree is 600 px, so 40 px blocks make 15x15.
        return shard_lst_p95.parse_args(
            [
                "--tile",
                "S30W065",
                "--engine",
                "fused",
                "--pixels-per-degree",
                "120",
                "--chunk",
                "40",
                "--out-dir",
                str(tmp_path),
            ]
        )

    def call(self, tmp_path, items, boxes, **overrides):
        import shard_lst_p95

        kwargs = {
            "client": FakeClient(),
            "cluster": FakeCluster(),
            "bbox": TILE_BBOX,
            "item_dicts": items,
            "item_bboxes": boxes,
            "prep": None,
            "keep_mask": None,
            "gap_mask": None,
            "marks": {},
            "say": lambda _line="": None,
        } | overrides
        return shard_lst_p95.run_fused(self.args_for(tmp_path), **kwargs), kwargs

    def test_it_refuses_cleanly_without_the_block_kernel(self, tmp_path, monkeypatch):
        monkeypatch.delattr(composite, "fused_block", raising=False)
        items, boxes = walk_items(6, TILE_BBOX)
        with pytest.raises(SystemExit, match="fused_block"):
            self.call(tmp_path, items, boxes)

    def test_it_plans_stages_and_submits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(composite, "fused_block", counting_stub, raising=False)
        items, boxes = walk_items(30, TILE_BBOX)
        client, marks, lines = FakeClient(), {}, []
        (scalars, paths, n_rejected, n_tasks), _ = self.call(
            tmp_path,
            items,
            boxes,
            client=client,
            prep=prep_for(items, rejected=(0, 4)),
            marks=marks,
            say=lines.append,
        )
        assert n_tasks == len(client.calls) == 225
        assert n_rejected == 2
        assert scalars["valid"] == sum(len(call["args"][1]) for call in client.calls)
        assert set(paths) == {"lst_p95", "qa_count"}
        assert all(p.exists() for p in paths.values())
        assert set(marks) == {"plan_s", "graph_build_s", "compute_s"}
        assert any(line.startswith("plan ") for line in lines)

    def test_the_mask_planes_reach_the_blocks_one_window_at_a_time(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(composite, "fused_block", counting_stub, raising=False)
        items, boxes = walk_items(12, TILE_BBOX)
        client = FakeClient()
        self.call(
            tmp_path,
            items,
            boxes,
            client=client,
            keep_mask=np.ones((600, 600), dtype=bool),
        )
        outs = [call["args"][4] for call in client.calls]
        assert all(out.keep.shape == (40, 40) for out in outs)
        assert all(out.hot_dn is not None for out in outs)

    def test_the_engine_flag_defaults_to_the_graph(self):
        import shard_lst_p95

        assert shard_lst_p95.parse_args(["--tile", "S30W065"]).engine == "graph"
        assert (
            shard_lst_p95.parse_args(["--tile", "S30W065", "--engine", "fused"]).engine
            == "fused"
        )

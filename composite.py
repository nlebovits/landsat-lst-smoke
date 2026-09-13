"""The p95 LST composite as one lazy dask-xarray graph per tile.

The stack is opened once with `odc.stac.load`, chunked in space and never in
time. A p95 over time needs every scene of a pixel in one block, so the time
axis is one chunk by construction and the reduction is a per-block operation
with no rechunk and no shuffle. `xr.apply_ufunc` with `dask="parallelized"`
refuses a time-chunked input, which is the guard the earlier array graph
lacked: it chunked time, `quantile` silently rechunked it back, and the
rechunk spilled 1.21 TiB.

Every numeric rule runs once, inside `reduce_block`, on one block:
`lst_qa.masked_celsius`, `destripe.subtract_offsets`,
`destripe.feathered_percentile` or `destripe.pooled_percentile`, the monthly
count, and `lst_qa.encode_celsius`. Those are the kernels the shard path
already pinned bit for bit. There is no second implementation of any of them.

The seam correction's weight field is a lazy array too. `odc.geo` reprojects
it from the swath grid onto the tile grid one block at a time, on the workers,
so the driver does no resampling before or during submission.

The outputs stream from the workers into two tiled staging GeoTIFFs: a second
blockwise kernel, `finalize_block`, masks each block, writes its windows under
a file lock, and returns per-pixel flags whose spatial sum is the one
DataArray the run computes. Nothing larger than one block returns to the
driver, and nothing is written to disk except those two files and the COGs
made from them.

    memory per block, DERIVED, to be checked against the first measured run:
        chunk^2 x n_scenes  x 4 B   the two uint16 bands dask holds
      + chunk^2 x n_present x 13 B  the decoded stack and its reduction
    360 px, 4,776 scenes, 820 present: 2.5 GB + 1.4 GB = 3.9 GB

That graph's one remaining cost scales with the tile's time axis rather than
with what a block reads. `odc.stac.load` puts two open tasks per item into it
whatever the block count, and hands every band-load task all of them, so the
driver's build MEASURED 15.5 s at 1,000 items and 60.5 s at 4,776 on the
64-worker 8x8 window run, serial, 19% of the 82 s wall. `build_block_plan`
carries the same information as a tuple of indices per block, from the
boolean matmul `block_depths` already runs, and `submit_blocks` sends one
task per block at the workers with only that block's scenes and vectors
aboard. MEASURED on the same shape and the real inventory footprints: the
plan, the vectors, and every block's subset together take 9 ms at 1,000 items
and 27 ms at 4,776, against 9.6 s and 62.9 s for `build_graph` on this
laptop. `--engine fused` selects that path; `--engine graph` is the default
and stays the tested one.

The task those blocks run is `fused_block`: one task that reads its own two
bands through `read_block`, resamples its own weights through
`block_weights`, and then calls the same `reduce_block` and the same
`finalize_block` the graph calls. It exists because the graph is three tasks
deep per block and frisky queues the two roots before the reduce, which idles
most of a wide cluster. Both engines write the same bytes;
`tests/test_composite_fused.py` asserts it block for block on the rehearsal
tile, `read_block` against `open_stack` and the staging files against each
other.
"""

from __future__ import annotations

import fcntl
import functools
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import destripe
import observe
from lst_qa import (
    LST_NODATA_DN,
    encode_celsius,
    masked_celsius,
)

#: Block edge in pixels. 360 divides an 18,000 px tile into 50 x 50 blocks
#: with no ragged edge. See the module docstring for the memory that sets it.
DEFAULT_CHUNK_PX = 360

#: Bytes per pixel-scene the loaded uint16 bands hold: lwir11 and qa_pixel.
LOADED_BYTES_PER_PIXEL_SCENE = 4

#: Bytes per pixel-scene of the decoded work on the scenes present in a block:
#: celsius float32, valid bool, the sort copy the percentile makes, and the two
#: uint16 subsets `reduce_block` cuts before decoding. MEASURED at 15 for the
#: shard path, whose loaded bands were already the subset; the 4 loaded bytes
#: above are counted at full depth instead.
PRESENT_BYTES_PER_PIXEL_SCENE = 13

#: Per-worker overhead outside the arrays, in GiB. From the shard measurement.
FIXED_GIB = 0.25

GIB = 1024.0**3

#: The tiled staging GeoTIFF the workers write windows into. Uncompressed, so
#: a 360 px window into a 512 px tile is a plain read-modify-write rather than
#: a decompress-recompress under the lock.
STAGING_BLOCK = 512


# --------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------


def spatial_dims(crs: str) -> tuple[str, str]:
    """The dimension names `odc.stac.load` gives this CRS.

    Getting these wrong is silent on the chunks argument, so they are derived
    from the CRS rather than assumed.
    """
    from odc.geo import CRS

    return ("y", "x") if CRS(crs).projected else ("latitude", "longitude")


def raster_shape(bbox, pixels_per_degree: int) -> tuple[int, int]:
    w, s, e, n = bbox
    return (
        int(round((n - s) * pixels_per_degree)),
        int(round((e - w) * pixels_per_degree)),
    )


def block_edges(height: int, width: int, chunk: int):
    """The pixel edges of the block grid, `(y_edges, x_edges)`.

    The blocks are the dask chunks, anchored the way the raster is, so a tile
    whose edge is not a multiple of `chunk` ends in a partial block rather
    than an overhanging one.
    """
    y_edges = np.arange(0, height + chunk, chunk).clip(max=height)
    x_edges = np.arange(0, width + chunk, chunk).clip(max=width)
    return y_edges, x_edges


def _overlap_masks(norths, souths, wests, easts, item_bboxes):
    """Which scenes reach each block row and each block column.

    Strict overlap in degrees, separably: a scene reaches a block when it
    reaches the block's row band and its column band. `block_depths` sums the
    product and `build_block_plan` reads the indices out of it, so both carry
    one overlap rule between them.

    Returns:
        `(rows, cols)` boolean, `(n_block_rows, n)` and `(n_block_cols, n)`.
    """
    boxes = np.asarray(item_bboxes, dtype="float64")  # (n, 4) w s e n
    rows = (boxes[:, 1][None, :] < norths[:, None]) & (
        boxes[:, 3][None, :] > souths[:, None]
    )
    cols = (boxes[:, 0][None, :] < easts[:, None]) & (
        boxes[:, 2][None, :] > wests[:, None]
    )
    return rows, cols


def block_depths(bbox, pixels_per_degree: int, chunk: int, item_bboxes):
    """How many scenes intersect each block, as a `(rows, cols)` array.

    The blocks are the dask chunks, anchored the way the raster is. The count
    feeds the memory model, the summary, and `build_block_plan`, which returns
    the indices behind these counts.
    """
    w, s, e, n = bbox
    height, width = raster_shape(bbox, pixels_per_degree)
    res = 1.0 / pixels_per_degree
    y_edges, x_edges = block_edges(height, width, chunk)
    norths = n - y_edges[:-1] * res
    souths = n - y_edges[1:] * res
    wests = w + x_edges[:-1] * res
    easts = w + x_edges[1:] * res
    if not len(item_bboxes):
        return np.zeros((len(norths), len(wests)), dtype="int64")
    rows, cols = _overlap_masks(norths, souths, wests, easts, item_bboxes)
    return (rows.astype("int64") @ cols.astype("int64").T).astype("int64")


def block_bytes(chunk: int, n_scenes: int, n_present: int) -> float:
    """Peak working set of one block, in GiB. DERIVED; see the module docstring."""
    loaded = chunk * chunk * n_scenes * LOADED_BYTES_PER_PIXEL_SCENE
    present = chunk * chunk * n_present * PRESENT_BYTES_PER_PIXEL_SCENE
    return (loaded + present) / GIB + FIXED_GIB


def memory_demand(chunk: int, n_scenes: int, depths, slots: int) -> float:
    """What `slots` concurrent blocks need at the deepest blocks, in GiB."""
    deepest = sorted(np.asarray(depths).ravel().tolist(), reverse=True)[:slots]
    return sum(block_bytes(chunk, n_scenes, d) for d in deepest)


def memory_guard(chunk, n_scenes, depths, slots, *, total_bytes=None) -> float:
    """Refuse a configuration that cannot fit, before the cluster starts.

    Returns the demand in GiB so the caller can report what it checked.

    Raises:
        SystemExit: naming the demand, the machine, and the escape.
    """
    if total_bytes is None:
        import psutil

        total_bytes = psutil.virtual_memory().total
    demand = memory_demand(chunk, n_scenes, depths, slots)
    total = total_bytes / GIB
    if demand <= total:
        return demand
    msg = (
        f"{slots} blocks at {chunk} px over {n_scenes:,} scenes need "
        f"{demand:.1f} GiB between them and this machine has {total:.1f} GiB. "
        f"Use a smaller --chunk, fewer --workers, or pass --force."
    )
    raise SystemExit(msg)


# --------------------------------------------------------------------------
# The block plan. One block, its grid, and the scenes that reach it.
#
# The lazy graph pays for the tile's whole time axis twice over: `odc.stac`
# builds two `open` tasks per item whatever the block count, and every
# band-load task takes every open handle as a dependency, so the driver's
# graph build ran 15.5 s at 1,000 items and 60.5 s at 4,776 on the 64-worker
# 8x8 window run, MEASURED, serial, 19% of the 82 s wall. A plan carries the
# same information as an integer per block: which scenes that block has to
# read. `block_depths` already computes the intersection as one boolean
# matmul, 9 ms for 2,500 blocks against 4,776 scenes, MEASURED; this reads the
# indices out of the same masks.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockSpec:
    """One block of the tile grid and the scenes that reach it.

    `yslice` and `xslice` are the block's pixel window into the full raster,
    `geobox` the same window as a grid the reader can load onto directly, and
    `item_indices` the positions in the run's item list of the scenes whose
    footprint overlaps the block, ordered by acquisition stamp.
    """

    row: int
    col: int
    geobox: Any
    yslice: slice
    xslice: slice
    item_indices: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.yslice.stop - self.yslice.start,
            self.xslice.stop - self.xslice.start,
        )

    @property
    def depth(self) -> int:
        """How many scenes this block reads. Its entry in `block_depths`."""
        return len(self.item_indices)


def item_times(items) -> np.ndarray:
    """The acquisition stamps of an item list, in the list's own order.

    `destripe.timestamp_of` parses the same field `odc.stac` puts on the time
    axis, so this is that axis without opening anything.
    """
    if not len(items):
        return np.empty(0, dtype="datetime64[ns]")
    return np.array([destripe.timestamp_of(item) for item in items], dtype="M8[ns]")


def time_order(items) -> np.ndarray:
    """The positions of `items` sorted by acquisition stamp, ties in list order.

    `odc.stac` sorts the loaded stack this way, and every per-scene vector is
    joined to it, so a plan that indexes in this order hands a block its
    scenes and its vectors in the same order the graph path would have.
    """
    return np.argsort(item_times(items), kind="stable")


def _geographic_block_edges(geobox, chunk: int):
    """The pixel edges and the degree edges of a geobox's block grid.

    Raises:
        ValueError: on a rotated or a projected geobox. The overlap rule is
            `block_depths`', which compares a scene's geographic bounding box
            against a block's, and only a north-up geographic grid has one
            block bounding box per row edge and column edge.
    """
    height, width = geobox.shape
    y_edges, x_edges = block_edges(int(height), int(width), chunk)
    a, b, c, d, e, f = tuple(geobox.transform)[:6]
    if b or d:
        msg = f"the tile geobox is rotated ({b}, {d}); the block plan needs north-up"
        raise ValueError(msg)
    if not geobox.crs.geographic:
        msg = (
            f"the tile geobox is on {geobox.crs}, and the block plan compares "
            f"scene footprints in degrees, as block_depths does. Composite on "
            f"a geographic grid."
        )
        raise ValueError(msg)
    return (
        y_edges,
        x_edges,
        f + e * y_edges[:-1],  # norths
        f + e * y_edges[1:],  # souths
        c + a * x_edges[:-1],  # wests
        c + a * x_edges[1:],  # easts
    )


def build_block_plan(items, item_bboxes, geobox_full, chunk: int) -> list[BlockSpec]:
    """One `BlockSpec` per block, carrying the scenes that reach that block.

    The overlap rule is `block_depths`': a scene reaches a block when their
    geographic bounding boxes strictly overlap. `len(spec.item_indices)`
    therefore equals that block's entry in `block_depths`, and the whole plan
    costs one boolean matmul plus one `flatnonzero` per block rather than a
    footprint test per block and scene.

    An empty block stays in the plan. It owns its windows of the output and
    still has to write nodata into them, which is what the lazy graph's
    all-fill block does today.

    Returns:
        The blocks in row-major order, so `plan[row * n_cols + col]` is the
        block at `(row, col)`.
    """
    if len(item_bboxes) != len(items):
        msg = (
            f"{len(items)} items against {len(item_bboxes)} footprints. The "
            f"plan indexes one list by the other's positions."
        )
        raise ValueError(msg)
    y_edges, x_edges, norths, souths, wests, easts = _geographic_block_edges(
        geobox_full, chunk
    )
    n_rows, n_cols = len(norths), len(wests)
    order = time_order(items)
    if len(items):
        boxes = np.asarray(item_bboxes, dtype="float64")[order]
        rows, cols = _overlap_masks(norths, souths, wests, easts, boxes)
    else:
        rows = np.zeros((n_rows, 0), dtype=bool)
        cols = np.zeros((n_cols, 0), dtype=bool)

    plan: list[BlockSpec] = []
    for row in range(n_rows):
        yslice = slice(int(y_edges[row]), int(y_edges[row + 1]))
        in_row = rows[row]
        for col in range(n_cols):
            xslice = slice(int(x_edges[col]), int(x_edges[col + 1]))
            # `in_row & cols[col]` is a mask over the time-sorted items, so
            # `order` maps it straight back to list positions in time order.
            hits = order[in_row & cols[col]]
            plan.append(
                BlockSpec(
                    row=row,
                    col=col,
                    geobox=geobox_full[yslice, xslice],
                    yslice=yslice,
                    xslice=xslice,
                    item_indices=tuple(int(i) for i in hits),
                )
            )
    return plan


def plan_depths(plan, shape=None) -> np.ndarray:
    """The plan's scene counts as the `(rows, cols)` array `block_depths` returns."""
    if shape is None:
        shape = (
            max((b.row for b in plan), default=-1) + 1,
            max((b.col for b in plan), default=-1) + 1,
        )
    depths = np.zeros(shape, dtype="int64")
    for block in plan:
        depths[block.row, block.col] = block.depth
    return depths


# --------------------------------------------------------------------------
# Per-scene vectors, and the subset one block sees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneVectors:
    """The per-scene arrays `reduce_block` takes, and the axis they sit on.

    `times` is the acquisition axis the other four were joined to, so a subset
    carries its own time coordinate with it and nothing downstream re-derives
    one. Every array is in the item list's order, which is why a `BlockSpec`'s
    time-ordered indices produce a time-ordered subset.
    """

    times: np.ndarray
    offset: np.ndarray
    keep: np.ndarray
    path_code: np.ndarray
    month: np.ndarray
    paths: tuple

    def __len__(self) -> int:
        return int(len(self.times))

    @property
    def n_rejected(self) -> int:
        return int((~self.keep).sum())

    def take(self, indices) -> SceneVectors:
        """The same five arrays at `indices`, in the order `indices` gives."""
        idx = np.asarray(indices, dtype="int64")
        return replace(
            self,
            times=self.times[idx],
            offset=self.offset[idx],
            keep=self.keep[idx],
            path_code=self.path_code[idx],
            month=self.month[idx],
        )


def months_of(times) -> np.ndarray:
    """The calendar month of each stamp, 1 to 12, as `int8`.

    The same integers `data["time"].dt.month` gives the graph path.
    """
    stamps = np.asarray(times, dtype="M8[ns]")
    return (stamps.astype("M8[M]").astype("int64") % 12 + 1).astype("int8")


def scene_vectors(
    items,
    prep=None,
    *,
    max_offset_c: float = destripe.DESTRIPE_MAX_OFFSET_C,
    debias: bool = True,
) -> SceneVectors:
    """The run's per-scene vectors, from the item list alone.

    The graph path reads the time axis off the loaded stack. There is no stack
    on the driver here, so the axis comes from `destripe.timestamp_of` and the
    join runs exactly as before: `per_scene_vectors` calls
    `destripe.align_to_time`, which still refuses a stamp two items disagree
    about and still refuses a step no item accounts for. Nothing joins on
    position.
    """
    times = item_times(items)
    offset, keep, path_code, paths = per_scene_vectors(
        items, times, prep, max_offset_c=max_offset_c, debias=debias
    )
    return SceneVectors(
        times=times,
        offset=offset,
        keep=keep,
        path_code=path_code,
        month=months_of(times),
        paths=paths,
    )


def block_vectors(vectors: SceneVectors, block) -> SceneVectors:
    """The per-scene vectors of one block, in time order.

    `block` is a `BlockSpec` or a sequence of indices into the run's item
    list. The result is what a block's kernel sees: its own scenes, its own
    time coordinate, and no trace of the tile's other thousands.
    """
    indices = getattr(block, "item_indices", block)
    return vectors.take(indices)


# --------------------------------------------------------------------------
# The stack
# --------------------------------------------------------------------------


def open_stack(items, bbox, *, crs: str, resolution: float, chunk: int):
    """The lazy `(time, y, x)` stack, time in one chunk, space in `chunk` blocks.

    `odc.stac.load` builds one read task per spatial block per band, and that
    task paints only the scenes that intersect the block. The default is one
    chunk per scene, which is the task explosion the failed graph had, so the
    time chunk is stated and then asserted.
    """
    import odc.stac
    import pystac

    ydim, xdim = spatial_dims(crs)
    parsed = [
        item if isinstance(item, pystac.Item) else pystac.Item.from_dict(item)
        for item in items
    ]
    data = odc.stac.load(
        parsed,
        bands=("lwir11", "qa_pixel"),
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        groupby="landsat:scene_id",
        chunks={"time": -1, ydim: chunk, xdim: chunk},
    )
    missing = {ydim, xdim} - set(data.dims)
    if missing:
        msg = f"expected spatial dims {ydim}/{xdim}, got {tuple(data.dims)}"
        raise RuntimeError(msg)
    n_time = data.sizes["time"]
    for band in ("lwir11", "qa_pixel"):
        chunks = data[band].chunks
        if chunks is not None and chunks[0] != (n_time,):
            msg = (
                f"{band} carries the time axis in {len(chunks[0])} chunks. A "
                f"percentile over time needs it in one, or the reduction "
                f"rechunks the whole tile."
            )
            raise RuntimeError(msg)
    return data


def per_scene_vectors(items, times, prep, *, max_offset_c, debias: bool):
    """One offset, one keep flag, and one path code per step of the time axis.

    Joined on the acquisition stamp through `destripe.align_to_time`, never on
    position. Without a prep artifact every scene keeps its own baseline, is
    kept, and carries path code -1, which matches no path.

    Returns:
        `(offset float32, keep bool, path_code int16, paths)`.
    """
    n = len(times)
    if prep is None:
        return (
            np.zeros(n, dtype="float32"),
            np.ones(n, dtype=bool),
            np.full(n, -1, dtype="int16"),
            (),
        )
    scene_id = destripe.align_to_time(
        items, times, value_of=destripe.scene_id_of, what="scene id", dtype=object
    )
    code_of = {path: j for j, path in enumerate(prep.paths)}
    path_code = destripe.align_to_time(
        items,
        times,
        value_of=lambda item: code_of.get(destripe.path_of(item), -1),
        what="WRS path",
        dtype="int16",
    )
    if debias:
        offset = np.array(
            [prep.offset.get(s, np.nan) for s in scene_id], dtype="float64"
        )
        n_valid = np.array([prep.n_valid.get(s, 0) for s in scene_id], dtype="int64")
        keep = destripe.keep_mask(
            offset,
            n_valid,
            floor=destripe.DESTRIPE_MIN_PREP_SAMPLES,
            max_offset_c=max_offset_c,
        )
    else:
        offset = np.zeros(n, dtype="float64")
        keep = np.ones(n, dtype=bool)
    return offset.astype("float32"), keep, path_code, tuple(prep.paths)


def lazy_weights(prep, geobox, *, chunk: int, dims):
    """The cross-fade weights on the tile grid, one warp task per path and block.

    `prep.weight` and `prep.inside` sit on the swath grid of the padded prep
    bbox. Containment is resampled nearest and the ramp bilinearly, then the
    shares are renormalised on the resampled containment, exactly as the
    shard's window resample did, but as lazy expressions on chunk-aligned
    planes. Bilinear alone would leak a small weight to a path that does not
    cover the pixel, and the leak would land on the swath edge.

    Returns a `(path, y, x)` float32 DataArray with `path` in one chunk and
    the spatial chunks of the stack.
    """
    import dask.array as da
    import xarray as xr
    from odc.geo import xr as gxr
    from odc.geo.geobox import GeoBox

    weight = np.asarray(prep.weight, dtype="float32")
    inside = np.asarray(prep.inside, dtype="uint8")
    n_paths, h, w = weight.shape
    src = GeoBox((h, w), destripe.prep_transform(prep), destripe.GRID_CRS)
    coords = gxr.xr_coords(src)

    def on_swath_grid(planes):
        arr = xr.DataArray(
            da.from_array(planes, chunks=(1, -1, -1)),
            dims=("path", *src.dims),
            coords=coords,
        )
        return gxr.assign_crs(arr, destripe.GRID_CRS)

    # odc-geo keeps `dst_nodata` distinguishable from data by moving any valid
    # pixel that equals it one ULP away. A zero weight is data, so the nodata
    # sentinels are NaN and 255, neither of which a weight or a flag can hold.
    ramp = gxr.xr_reproject(
        on_swath_grid(weight),
        geobox,
        resampling="bilinear",
        dst_nodata=np.nan,
        chunks=(chunk, chunk),
    ).fillna(np.float32(0.0))
    cover = gxr.xr_reproject(
        on_swath_grid(inside),
        geobox,
        resampling="nearest",
        dst_nodata=255,
        chunks=(chunk, chunk),
    )
    covered = cover == 1
    ramp = ramp.where(covered, np.float32(0.0))
    total = ramp.sum("path")
    share = ramp / total.where(total > 0, np.float32(1.0))
    k = covered.sum("path")
    degenerate = (total <= 0) & (k > 0)
    share = xr.where(
        degenerate & covered, np.float32(1.0) / k.where(k > 0, 1), share
    ).astype("float32")
    # `xr.where` orders the result's dims after its condition, which starts
    # with the spatial dims. The kernel wants the path axis first.
    share = share.transpose("path", *src.dims)
    share = share.rename(dict(zip(src.dims, dims, strict=True)))
    share = share.drop_vars([c for c in share.coords if c not in dims])
    return share.chunk({"path": -1})


# --------------------------------------------------------------------------
# The kernel. One block, every rule, once.
# --------------------------------------------------------------------------


def reduce_block(
    lwir,
    qa,
    offset,
    keep,
    path_code,
    month,
    weight,
    *,
    n_paths: int,
    feather: bool,
    emit_pooled: bool,
):
    """Mask, correct, reduce, count, and encode one block.

    `apply_ufunc` hands core dimensions last, so the bands arrive as
    `(y, x, time)` and the weights as `(y, x, path)`. They are viewed
    time-major here because that is the order every kernel below takes.

    Scenes that do not reach this block arrive as source fill. They are cut
    before the decode, so the float32 work runs at the block's own depth
    rather than the tile's.

    Returns:
        `lst_p95` uint16 `(y, x)`, `qa_count` uint8 `(y, x, 12)`, `fallback`
        bool `(y, x)`, and with `emit_pooled` a fourth uint16 `(y, x)`.
    """
    t0 = time.perf_counter_ns()
    lwir = np.moveaxis(lwir, -1, 0)
    qa = np.moveaxis(qa, -1, 0)
    weight = np.moveaxis(weight, -1, 0)
    keep = np.asarray(keep, dtype=bool)

    # One reduction over the whole stack. A Python loop over the scenes was
    # MEASURED at 0.79 s of a 2.15 s kernel on the 4,776-scene time axis; the
    # same test as one `any` is 0.06 s. `any` on an integer array is `!= 0`.
    present = lwir.any(axis=(1, 2))
    present &= keep
    lwir_p = np.ascontiguousarray(lwir[present])
    qa_p = np.ascontiguousarray(qa[present])
    offset_p = np.asarray(offset, dtype="float64")[present]
    code_p = np.asarray(path_code)[present]
    month_p = np.asarray(month)[present]

    celsius, valid = masked_celsius(lwir_p, qa_p)
    destripe.subtract_offsets(celsius, offset_p)

    ny, nx = lwir.shape[1:]
    # An all-fill pixel is a nodata pixel, not a warning: numpy says
    # "All-NaN slice encountered" once per block otherwise.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if feather and n_paths:
            p95, fallback = destripe.feathered_percentile(
                celsius, code_p, tuple(range(n_paths)), weight
            )
        else:
            p95 = destripe.pooled_percentile(celsius)
            fallback = np.isfinite(p95)
        pooled = (
            encode_celsius(destripe.pooled_percentile(celsius)) if emit_pooled else None
        )

    counts = np.zeros((12, ny, nx), dtype="uint8")
    for m in range(1, 13):
        sel = month_p == m
        if sel.any():
            counts[m - 1] = np.minimum(valid[sel].sum(axis=0), 255).astype("uint8")

    dn = encode_celsius(p95)
    _record_block_span(t0, int(present.sum()))
    outputs = (dn, np.moveaxis(counts, 0, -1), fallback)
    return (*outputs, pooled) if emit_pooled else outputs


def _record_block_span(t0_ns: int, n_present: int) -> None:
    """One span per block on the worker, where tracing is already on."""
    try:
        import frisky

        t1 = frisky.now_ns()
        frisky.record_span(
            "worker.exec.reduce_block",
            t1 - (time.perf_counter_ns() - t0_ns),
            t1,
            count=n_present,
        )
    except Exception:  # noqa: BLE001  instrumentation never fails the run
        return


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------


def build_graph(
    items,
    bbox,
    *,
    crs: str,
    resolution: float,
    chunk: int,
    prep=None,
    max_offset_c: float = destripe.DESTRIPE_MAX_OFFSET_C,
    debias: bool = True,
    feather: bool = True,
    emit_pooled: bool = False,
):
    """The whole composite as one lazy Dataset. Computes nothing.

    Returns a Dataset with `lst_p95` uint16 `(y, x)`, `qa_count` uint8
    `(month, y, x)`, `fallback` bool `(y, x)`, and with `emit_pooled` a
    `lst_p95_pooled` uint16 `(y, x)`. Its attrs carry `n_rejected` and
    `paths`, which are known before any compute.
    """
    import dask.array as da
    import xarray as xr

    ydim, xdim = spatial_dims(crs)
    data = open_stack(items, bbox, crs=crs, resolution=resolution, chunk=chunk)
    times = data["time"].values
    offset, keep, path_code, paths = per_scene_vectors(
        items, times, prep, max_offset_c=max_offset_c, debias=debias
    )
    month = data["time"].dt.month.values.astype("int8")

    use_feather = bool(feather and prep is not None and paths)
    if use_feather:
        weight = lazy_weights(prep, data.odc.geobox, chunk=chunk, dims=(ydim, xdim))
        weight = weight.assign_coords({ydim: data[ydim], xdim: data[xdim]})
    else:
        ny, nx = data.sizes[ydim], data.sizes[xdim]
        weight = xr.DataArray(
            da.zeros((1, ny, nx), chunks=(1, chunk, chunk), dtype="float32"),
            dims=("path", ydim, xdim),
            coords={ydim: data[ydim], xdim: data[xdim]},
        )

    vectors = [
        xr.DataArray(v, dims=("time",), coords={"time": data["time"]})
        for v in (offset, keep, path_code, month)
    ]
    output_core_dims = [[], ["month"], []] + ([[]] if emit_pooled else [])
    output_dtypes = [np.uint16, np.uint8, bool] + ([np.uint16] if emit_pooled else [])
    outputs = xr.apply_ufunc(
        reduce_block,
        data["lwir11"],
        data["qa_pixel"],
        *vectors,
        weight,
        input_core_dims=[
            ["time"],
            ["time"],
            ["time"],
            ["time"],
            ["time"],
            ["time"],
            ["path"],
        ],
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs={"output_sizes": {"month": 12}},
        kwargs={
            "n_paths": len(paths) if use_feather else 0,
            "feather": use_feather,
            "emit_pooled": emit_pooled,
        },
    )
    lst, counts, fallback = outputs[:3]
    counts = counts.transpose("month", ydim, xdim)
    lst = lst.astype("uint16")
    counts = counts.astype("uint8")

    out = xr.Dataset({"lst_p95": lst, "qa_count": counts, "fallback": fallback})
    if emit_pooled:
        out["lst_p95_pooled"] = outputs[3].astype("uint16")

    out.attrs["n_rejected"] = int((~keep).sum())
    out.attrs["n_scenes"] = int(len(times))
    out.attrs["paths"] = list(paths)
    out.attrs["feather"] = use_feather
    out.attrs["debias"] = bool(debias and prep is not None)
    out.attrs["bbox"] = tuple(bbox)
    out.attrs["pixels_per_degree"] = int(round(1.0 / resolution))
    return out


# --------------------------------------------------------------------------
# Finishing. One more blockwise kernel masks the block, writes its windows into
# the staging GeoTIFFs, and returns per-pixel flags. The spatial sum of those
# flags is the one DataArray the run computes: one collection, one call, every
# block decoded once, and nothing larger than a block back to the driver.
#
# How frisky runs it, MEASURED with frisky 0.7.2 on dask 2026.8.0: the dask
# arrays here carry no expression (`Array.expr` is None) and no records hook,
# so frisky's expression and records paths accept no collection at all and
# every compute runs as a stock dask graph through frisky's `Client.get`. The
# event log records that as `dask_expression_fallback` on each compute;
# `observe.collect` keeps the record, and it is the expected path here, not a
# defect. The tasks still run on frisky's workers with frisky's scheduler.
# --------------------------------------------------------------------------

#: What `finalize_block` counts per pixel, in the order of its `flag` axis.
FLAGS = ("fallback", "removed_water", "removed_hot", "qa_zeroed", "valid")


class FileLock:
    """A cross-process lock on one path, safe to pickle into a task.

    `finalize_block` serialises its window writes through this, and the
    workers are separate processes, so a threading lock would guard nothing.
    """

    def __init__(self, path):
        self.path = str(path)
        self._fh = None

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "a")  # noqa: SIM115  released in __exit__
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False

    acquire = __enter__
    release = __exit__


def file_statistics(path: Path, nodata) -> list[dict]:
    """Min, max, mean, population std, kept, and total per band, from a file.

    Read in tile rows, one band at a time, so the largest thing held is one
    strip of one band. This runs on the driver after the compute, over the
    staging file the workers wrote, and it is what puts the statistics on the
    COG. It is not in the graph on purpose: thirteen bands of min, max, mean,
    and std as lazy reductions added about eight tasks per band per block and
    a second reduction tree over blocks the compute already holds. A scan of a
    local file costs seconds and keeps the graph to the reduce and the write.
    """
    import rasterio
    from rasterio.windows import Window

    stats = []
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        total = int(src.height) * int(src.width)
        for band in range(1, src.count + 1):
            kept = 0
            total_sum = 0.0
            total_sq = 0.0
            minimum = np.inf
            maximum = -np.inf
            for row in range(0, src.height, STAGING_BLOCK):
                height = min(STAGING_BLOCK, src.height - row)
                strip = src.read(
                    band, window=Window.from_slices((row, row + height), (0, src.width))
                )
                values = strip if nodata is None else strip[strip != nodata]
                if values.size == 0:
                    continue
                wide = values.astype("float64")
                kept += int(wide.size)
                total_sum += float(wide.sum())
                total_sq += float((wide * wide).sum())
                minimum = min(minimum, float(wide.min()))
                maximum = max(maximum, float(wide.max()))
            if kept == 0:
                stats.append(
                    {
                        "min": 0.0,
                        "max": 0.0,
                        "mean": 0.0,
                        "std": 0.0,
                        "kept": 0,
                        "total": total,
                    }
                )
                continue
            mean = total_sum / kept
            var = max(total_sq / kept - mean * mean, 0.0)
            stats.append(
                {
                    "min": minimum,
                    "max": maximum,
                    "mean": mean,
                    "std": var**0.5,
                    "kept": kept,
                    "total": total,
                }
            )
    return stats


def _create_staging(path: Path, *, shape, count, dtype, nodata, transform, crs) -> None:
    """The empty tiled GeoTIFF the workers write their windows into."""
    import rasterio

    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": STAGING_BLOCK,
        "blockysize": STAGING_BLOCK,
        "compress": None,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path, "w", **profile):
        pass


def staging_targets(
    out_dir,
    *,
    shape,
    transform,
    crs: str,
    months: int = 12,
    emit_pooled: bool = False,
):
    """Create the staging GeoTIFFs the blocks write their windows into.

    The band count, the dtype, and the nodata of each file follow
    `reduce_block`'s outputs, so both engines write the same three files with
    the same header. One implementation, called once per run.

    Returns:
        `(targets, paths)`. `targets` maps a name to `(path, FileLock)`, which
        is what a worker's write needs and all it needs; `paths` maps the same
        names to the driver's `Path` objects for the COG copy and the cleanup.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [
        ("lst_p95", "uint16", LST_NODATA_DN, 1),
        ("qa_count", "uint8", None, int(months)),
    ]
    if emit_pooled:
        wanted.append(("lst_p95_pooled", "uint16", LST_NODATA_DN, 1))
    paths: dict = {}
    targets: dict = {}
    for name, dtype, nodata, count in wanted:
        path = out_dir / f"{name}.staging.tif"
        _create_staging(
            path,
            shape=shape,
            count=count,
            dtype=dtype,
            nodata=nodata,
            transform=transform,
            crs=crs,
        )
        paths[name] = path
        targets[name] = (str(path), FileLock(str(path) + ".lock"))
    return targets, paths


def _write_window(path: str, lock, stack, y0: int, x0: int) -> None:
    import rasterio
    from rasterio.windows import Window

    height, width = stack.shape[1:]
    with lock, rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path, "r+") as dst:
        dst.write(
            np.ascontiguousarray(stack),
            window=Window.from_slices((y0, y0 + height), (x0, x0 + width)),
        )


def finalize_block(
    lst,
    qa,
    fallback,
    keep,
    gap,
    rows,
    cols,
    *pooled,
    masked: bool,
    hot_dn,
    targets: dict,
):
    """Mask one block, write its windows, and return its per-pixel flags.

    `qa` arrives `(y, x, month)` because `month` is a core dimension; `keep`
    and `gap` arrive as the block's own planes, or as 0-d placeholders when
    the run is unmasked. `rows` and `cols` are the block's pixel indices,
    broadcast to its shape, so the window is read off their corners.

    The mask is `masks.apply_output_mask`, the one implementation of both
    rules, applied to the block in place. The per-pixel flags are read off
    its effect: what was valid before and is nodata after, split by which
    rule reached it.

    Returns `(y, x, flag)` uint8 in the order of `FLAGS`.
    """
    from masks import apply_output_mask

    lst = np.array(lst, dtype="uint16", copy=True)
    qa = np.ascontiguousarray(np.moveaxis(np.asarray(qa), -1, 0)).astype("uint8")
    height, width = lst.shape
    y0, x0 = int(np.asarray(rows).flat[0]), int(np.asarray(cols).flat[0])
    fallback = np.asarray(fallback, dtype=bool)

    valid_before = lst != LST_NODATA_DN
    qa_before = qa.any(axis=0)
    if masked:
        keep = np.broadcast_to(np.asarray(keep, dtype=bool), lst.shape)
        gap = np.broadcast_to(np.asarray(gap, dtype=bool), lst.shape)
        apply_output_mask(lst, qa, keep, gap, hot_dn=hot_dn)
    else:
        keep = np.ones(lst.shape, dtype=bool)
    valid_after = lst != LST_NODATA_DN
    water = ~keep

    path, lock = targets["lst_p95"]
    _write_window(path, lock, lst[None], y0, x0)
    path, lock = targets["qa_count"]
    _write_window(path, lock, qa, y0, x0)
    if pooled:
        pooled_dn = np.array(pooled[0], dtype="uint16", copy=True)
        pooled_dn[water] = LST_NODATA_DN
        path, lock = targets["lst_p95_pooled"]
        _write_window(path, lock, pooled_dn[None], y0, x0)

    flags = np.zeros((height, width, len(FLAGS)), dtype="uint8")
    flags[..., 0] = fallback
    flags[..., 1] = valid_before & water
    flags[..., 2] = valid_before & ~valid_after & keep
    flags[..., 3] = qa_before & water
    flags[..., 4] = valid_after
    return flags


def staging_writes(
    out,
    out_dir: Path,
    *,
    crs: str,
    dims,
    keep_mask=None,
    gap_mask=None,
    hot_dn=None,
):
    """The lazy finish of the graph: masks, window writes, and the flag counts.

    Returns `(counts, paths)`. `counts` is one lazy `(flag,)` DataArray whose
    compute masks every block, writes every window, and sums the flags. It is
    the only thing the run computes, so every block is decoded once.
    """
    import dask.array as da
    import xarray as xr

    from masks import gap_hot_dn, transform_for

    ydim, xdim = dims
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ny, nx = out.sizes[ydim], out.sizes[xdim]
    chunks_y, chunks_x = out["lst_p95"].chunks
    chunk = max(max(chunks_y), max(chunks_x))
    coords = {ydim: out[ydim], xdim: out[xdim]}
    transform = transform_for(out.attrs["bbox"], out.attrs["pixels_per_degree"])

    rows = xr.DataArray(
        da.arange(ny, chunks=chunks_y, dtype="int32"),
        dims=(ydim,),
        coords={ydim: out[ydim]},
    )
    cols = xr.DataArray(
        da.arange(nx, chunks=chunks_x, dtype="int32"),
        dims=(xdim,),
        coords={xdim: out[xdim]},
    )

    masked = keep_mask is not None
    if masked:
        keep = xr.DataArray(
            da.from_array(np.asarray(keep_mask, dtype=bool), chunks=(chunk, chunk)),
            dims=dims,
            coords=coords,
        )
        gap_plane = (
            np.zeros((ny, nx), dtype=bool)
            if gap_mask is None
            else np.asarray(gap_mask, dtype=bool)
        )
        gap = xr.DataArray(
            da.from_array(gap_plane, chunks=(chunk, chunk)), dims=dims, coords=coords
        )
        hot_dn = gap_hot_dn() if hot_dn is None else hot_dn
    else:
        keep = xr.DataArray(np.array(True))
        gap = xr.DataArray(np.array(False))

    targets, paths = staging_targets(
        out_dir,
        shape=(ny, nx),
        transform=transform,
        crs=crs,
        months=int(out.sizes["month"]),
        emit_pooled="lst_p95_pooled" in out,
    )
    extra = [out["lst_p95_pooled"]] if "lst_p95_pooled" in out else []

    flags = xr.apply_ufunc(
        finalize_block,
        out["lst_p95"],
        out["qa_count"],
        out["fallback"],
        keep,
        gap,
        rows,
        cols,
        *extra,
        input_core_dims=[[], ["month"], [], [], [], [], []] + [[] for _ in extra],
        output_core_dims=[["flag"]],
        dask="parallelized",
        output_dtypes=[np.uint8],
        dask_gufunc_kwargs={"output_sizes": {"flag": len(FLAGS)}},
        kwargs={"masked": masked, "hot_dn": hot_dn, "targets": targets},
    )
    counts = flags.sum([ydim, xdim]).assign_coords(flag=list(FLAGS))
    return counts, paths


def compute_all(counts) -> dict:
    """Compute the one `(flag,)` DataArray and name its entries."""
    values = np.asarray(counts.compute().values).astype("int64")
    return {flag: int(value) for flag, value in zip(FLAGS, values, strict=True)}


# --------------------------------------------------------------------------
# The fused submission path. One task per block, submitted by the driver,
# instead of one graph the scheduler unrolls into per-scene layers.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockOutputs:
    """Where a block writes and what it masks, cut to one block.

    `keep` and `gap` are the run's full-tile planes on the driver and one
    block's window of them on a worker. Cutting them here is a slice of a
    numpy array, not a computation: the alternative is shipping a whole
    18,000 px plane, 324 MB per band, to every worker to have it read one
    block of it.

    Two call sites cut, and `cut` is what keeps them from cutting twice.
    `submit_blocks` cuts on the driver so the wire carries one block's window
    rather than the tile's plane, and `fused_block` cuts on the worker so a
    direct call with the tile's planes still works. A block's slices are
    absolute, so a second cut indexes a 360 px window at the block's tile
    offset and yields an empty array: block (3, 3) reads `keep[1080:1440,
    1080:1440]` of a 360 px plane and gets `(0, 0)`. `finalize_block` then
    raises on the broadcast. MEASURED on the 12-scene rehearsal, 2026-09-13.
    The 8x8 instance benchmark missed it because it ran `--no-output-mask`,
    where `keep` is None and neither call site cuts at all.
    """

    targets: dict
    keep: Any = None
    gap: Any = None
    hot_dn: Any = None
    #: True once the planes are one block's window rather than the tile's.
    cut: bool = False

    @property
    def masked(self) -> bool:
        return self.keep is not None

    def for_block(self, block: BlockSpec) -> BlockOutputs:
        """This block's window of the mask planes, and the same write targets.

        Idempotent: cutting an already-cut window returns it unchanged, so the
        driver and the worker can both ask without agreeing which one did it.
        """
        if self.keep is None or self.cut:
            return self
        ys, xs = block.yslice, block.xslice
        return replace(
            self,
            keep=np.ascontiguousarray(np.asarray(self.keep, dtype=bool)[ys, xs]),
            gap=(
                None
                if self.gap is None
                else np.ascontiguousarray(np.asarray(self.gap, dtype=bool)[ys, xs])
            ),
            cut=True,
        )


@functools.lru_cache(maxsize=2)
def _prep_from_disk(path: str):
    """The prep artifact, read once per worker process and kept.

    Every block's task takes the same artifact, about 290 MB of swath weights
    on a 5 degree tile, and frisky's `submit` serialises its arguments on each
    call. Scattering one copy per worker was tried first: on 64 workers that
    is 18.6 GB through the scheduler, and one of the 64 scatters was lost
    before it reached its worker, which failed the run. MEASURED on the 8x8
    window at 64 slots, 2026-09-12. The workers share the driver's disk, so
    the task carries the path and each process reads the file once from page
    cache instead.
    """
    return destripe.load_prep(Path(path))


def resolve_prep(prep):
    """A prep object, from the object itself or from the path a task carries."""
    if isinstance(prep, (str, Path)):
        return _prep_from_disk(str(prep))
    return prep


def submit_blocks(
    client,
    cluster,
    plan,
    fn,
    *,
    items,
    vectors: SceneVectors,
    prep=None,
    out=None,
    emit_pooled: bool = False,
    marks: dict | None = None,
) -> dict:
    """Submit one `fn` per block, deepest first, round-robin over the workers.

    Each task is handed its own block's scenes and its own block's vectors,
    both already cut to `block.item_indices`, so nothing on the wire scales
    with the tile's time axis. The deepest blocks go first because they are
    the run's critical path: a block over 800 scenes cannot start late and
    still finish with the rest.

    Placement is `warm_workers`' pattern, `workers=[addr]` round-robin over
    `cluster._worker_addresses`. Frisky's scheduler would otherwise be free to
    pile the head of the queue, which is every deep block, onto one worker.

    `prep` goes to the tasks as given. Pass the artifact's directory rather
    than the loaded object: a path is a few bytes per submit and each worker
    reads the file once through `resolve_prep`, where the object itself would
    be pickled once per block.

    `observe.phase` marks stay the graph path's two: `graph_build` around the
    planning and the submission, `compute` around the wait, so the summary
    prints the same two figures against the same two names.

    Returns:
        The per-flag totals, summed over the blocks, in the shape
        `compute_all` returns.
    """
    addresses = list(getattr(cluster, "_worker_addresses", None) or [])
    deepest_first = sorted(plan, key=lambda block: -block.depth)

    futures = []
    with observe.phase("graph_build", count=len(plan), marks=marks):
        for n, block in enumerate(deepest_first):
            addr = addresses[n % len(addresses)] if addresses else None
            kwargs = {"emit_pooled": emit_pooled}
            if addr is not None:
                kwargs["workers"] = [addr]
            futures.append(
                client.submit(
                    fn,
                    block,
                    [items[i] for i in block.item_indices],
                    block_vectors(vectors, block),
                    prep,
                    out.for_block(block) if hasattr(out, "for_block") else out,
                    **kwargs,
                )
            )

    totals = dict.fromkeys(FLAGS, 0)
    with observe.phase("compute", count=len(plan), marks=marks):
        for future in futures:
            # Only the flags. `fused_block` returns its stage timings in the
            # same dict, and those are seconds, not pixels.
            result = future.result() or {}
            for flag in FLAGS:
                totals[flag] += int(result.get(flag, 0))
    return totals


# --------------------------------------------------------------------------
# The per-block kernel. One task reads its own two bands, resamples its own
# weights, reduces, and writes.
#
# The graph engine's shape is three tasks deep per block: two band loads and a
# reduce behind them. MEASURED on 64 single-threaded frisky workers with 64
# blocks at chunk 360, the 128 loads (4.2 s lwir11, 3.4 s qa_pixel, one core
# each, decode-bound) were pre-queued onto 54 workers up to 8 deep while 10
# workers got nothing, and the 64 reduces waited behind them. 665 of 3,521
# slot-seconds executed a task, 19%, and the 51 blocks whose two bands landed
# on different workers moved 259 MB each. Seven frisky placement environment
# knobs changed none of it.
#
# `fused_block` removes the queue rather than reordering it. One task per
# block, pinned by `submit_blocks`, so 64 blocks on 64 workers are 64
# concurrent tasks, no band array crosses the wire, and every worker holds one
# block at a time.
#
# Every numeric rule still runs exactly once, in the kernels this module
# already had. Nothing below reimplements a percentile, a mask, or a weight:
# `block_weights` calls `lazy_weights` on the block's own geobox, and
# `read_block` drives the same `odc.loader` reader `odc.stac.load` drives.
# --------------------------------------------------------------------------


def _band_config(asset: dict, band: str):
    """The band metadata and load parameters `odc.stac.load` resolves.

    `odc.stac` reads `raster:bands` off the asset and hands the result to
    `odc.loader.resolve_load_cfg`. Doing the same here is what makes the fused
    reader bit-identical to the graph's loader, including the part that looks
    wrong: `qa_pixel` declares `nodata: 1` in the item while the file on disk
    declares 0, so the loaded qa plane is filled with 1 outside a scene and
    every source 0 is remapped to 1. `lwir11` declares 0 and is filled with 0.
    `tests/test_composite_fused.py` pins both against `open_stack`.
    """
    from odc.loader import RasterBandMetadata, resolve_load_cfg

    raster = (asset.get("raster:bands") or [{}])[0]
    meta = RasterBandMetadata(
        data_type=raster.get("data_type"),
        nodata=raster.get("nodata"),
        units=str(raster.get("unit", "1")),
    )
    return meta, resolve_load_cfg({band: meta})[band]


def read_block(items: list[dict], geobox, band: str) -> np.ndarray:
    """`(y, x, n_items)` uint16 on the block geobox, one time step per item.

    Nearest resampling, reprojected into `geobox` by the same `odc.loader`
    reader `odc.stac.load` drives, so a block read here is bit-identical to
    the same block of `open_stack`. Where an item does not reach the block the
    plane keeps the band's declared fill: 0 for `lwir11`, 1 for `qa_pixel`
    (see `_band_config`). Item order is the time order given, and each item is
    one time step: `build_block_plan` hands one index per scene in acquisition
    order and a scene id belongs to one item.

    Nothing here reads more than the block. The warper opens the file and
    pulls only the source window that lands inside `geobox`. An item whose
    STAC bbox misses the block is skipped without an open, which is what
    `odc.stac`'s spatial binning does for the same item; its plane would be
    fill either way, so the skip is output-neutral. The footprint test is the
    bounding box rather than the geometry, so the skip is never wider than
    odc's, and `build_block_plan` has usually made it already.

    The loop runs inside one GDAL environment, entered through the same
    `RioDriver.restore_env` the loader's chunk task enters, so these reads see
    the GDAL configuration the graph's reads see. Opening a file with no
    ambient environment pushes and pops a GDAL config per open, MEASURED at
    19% of a 640-read block loop on the rehearsal scenes.
    """
    from odc.loader import (
        RasterSource,
        RioDriver,
        resolve_dst_nodata,
        resolve_src_nodata,
    )

    ny, nx = geobox.shape
    if not len(items):
        return np.zeros((ny, nx, 0), dtype="uint16")
    meta, cfg = _band_config(items[0]["assets"][band], band)
    dtype = np.dtype(cfg.dtype)
    fill = resolve_dst_nodata(dtype, cfg, resolve_src_nodata(cfg.fill_value, cfg))
    if fill is None:
        fill = dtype.type(0)
    stack = np.full((len(items), ny, nx), fill, dtype=dtype)
    west, south, east, north = geobox.geographic_extent.boundingbox
    driver = RioDriver()
    load_state = driver.new_load(geobox)
    try:
        with driver.restore_env(driver.capture_env(), load_state) as ctx:
            for i, item in enumerate(items):
                box = item.get("bbox")
                if box and not (
                    box[0] < east
                    and box[2] > west
                    and box[1] < north
                    and box[3] > south
                ):
                    continue
                href = str(item["assets"][band]["href"])
                source = RasterSource(href, band=1, meta=meta)
                roi, pix = driver.open(source, ctx).read(cfg, geobox)
                if pix.size:
                    # The destination starts at the fill value everywhere, and
                    # one item is one time step, so the nodata fuser odc would
                    # apply here is a plain copy.
                    stack[i][roi] = pix
    finally:
        driver.finalise_load(load_state)
    return np.moveaxis(stack, 0, -1)


def block_weights(prep, geobox) -> np.ndarray:
    """`(y, x, n_paths)` float32: `lazy_weights` on one block, computed here.

    The graph engine warps the swath grid into each block's own geobox, one
    dask task per block. This calls the same function on the same geobox with
    the block as its only chunk, so the resample, the renormalisation, and the
    degenerate case are the one implementation rather than a copy of it. The
    compute is synchronous: a worker thread must not start a scheduler.
    """
    ny, nx = geobox.shape
    share = lazy_weights(prep, geobox, chunk=max(ny, nx, 1), dims=geobox.dims)
    values = np.asarray(share.data.compute(scheduler="synchronous"), dtype="float32")
    return np.ascontiguousarray(np.moveaxis(values, 0, -1))


def fused_block(
    block: BlockSpec,
    items: list[dict],
    vectors: SceneVectors,
    prep,
    out: BlockOutputs,
    *,
    emit_pooled: bool = False,
    feather: bool = True,
) -> dict:
    """Read, weight, reduce, mask, and write one block. The whole task.

    `items` and `vectors` are already this block's own, cut by
    `submit_blocks` from `block.item_indices` and in acquisition order, so
    nothing here re-indexes them. `out` is already this block's window of the
    mask planes.

    Everything numeric is a call into the kernels the graph engine calls:
    `reduce_block` and `finalize_block`, unchanged.

    Returns:
        The per-flag counts `compute_all` sums, and the seconds each stage
        took. `submit_blocks` adds up the flags and leaves the timings.
    """
    t0_ns = time.perf_counter_ns()
    t0 = time.perf_counter()
    if hasattr(out, "for_block"):
        out = out.for_block(block)
    prep = resolve_prep(prep)
    lwir = read_block(items, block.geobox, "lwir11")
    qa = read_block(items, block.geobox, "qa_pixel")
    t_read = time.perf_counter()

    use_feather = bool(feather and prep is not None and len(vectors.paths))
    if use_feather:
        weight = block_weights(prep, block.geobox)
    else:
        weight = np.zeros((*block.shape, 1), dtype="float32")
    t_weight = time.perf_counter()

    outputs = reduce_block(
        lwir,
        qa,
        vectors.offset,
        vectors.keep,
        vectors.path_code,
        vectors.month,
        weight,
        n_paths=len(vectors.paths) if use_feather else 0,
        feather=use_feather,
        emit_pooled=emit_pooled,
    )
    del lwir, qa, weight
    t_reduce = time.perf_counter()

    flags = finalize_block(
        outputs[0],
        outputs[1],
        outputs[2],
        True if out.keep is None else out.keep,
        False if out.gap is None else out.gap,
        np.array([block.yslice.start]),
        np.array([block.xslice.start]),
        *outputs[3:],
        masked=out.masked,
        hot_dn=out.hot_dn,
        targets=out.targets,
    )
    t_end = time.perf_counter()
    _record_fused_span(t0_ns, len(items))
    counts = {flag: int(flags[..., i].sum()) for i, flag in enumerate(FLAGS)}
    return counts | {
        "row": block.row,
        "col": block.col,
        "depth": len(items),
        "read_s": t_read - t0,
        "weights_s": t_weight - t_read,
        "reduce_s": t_reduce - t_weight,
        "write_s": t_end - t_reduce,
        "block_s": t_end - t0,
    }


def _record_fused_span(t0_ns: int, n_items: int) -> None:
    """One span per fused task on the worker, beside `reduce_block`'s own."""
    try:
        import frisky

        t1 = frisky.now_ns()
        frisky.record_span(
            "worker.exec.fused_block",
            t1 - (time.perf_counter_ns() - t0_ns),
            t1,
            count=n_items,
        )
    except Exception:  # noqa: BLE001  instrumentation never fails the run
        return


def finish_staging(path: Path, *, scale, offset, descriptions, nodata) -> list[dict]:
    """Write the decoding rule and the statistics into a finished staging file.

    The windows are written by the workers, and a GeoTIFF's scales, offsets,
    descriptions, and band tags are header fields, so they go on afterwards in
    one `r+` open here. The COG copy then carries them across.

    Returns the per-band statistics it wrote, for the summary.
    """
    import rasterio
    from cog_catalog import statistics_tags

    statistics = file_statistics(path, nodata)
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path, "r+") as dst:
        count = dst.count
        if scale is not None and offset is not None:
            dst.scales = (scale,) * count
            dst.offsets = (offset,) * count
        if descriptions:
            dst.descriptions = tuple(descriptions)
        for index, stats in enumerate(statistics, start=1):
            dst.update_tags(index, **statistics_tags(**stats))
    return statistics


def cleanup_staging(paths) -> None:
    for path in paths.values():
        Path(path).unlink(missing_ok=True)
        Path(str(path) + ".lock").unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Rehearsal scenes. Synthetic pixels on disk, so a laptop runs the whole
# graph, the cluster, the writes, and the catalog for nothing.
# --------------------------------------------------------------------------


def item_for_files(scene_id: str, bands: dict, *, datetime: str, path: str, row: str):
    """A STAC item describing two band files already on disk.

    The projection comes from the thermal raster itself, so this describes a
    synthetic scene and a staged Landsat scene the same way. Every href is an
    absolute path, because `odc.stac` refuses a relative one.
    """
    import rasterio
    from tile_inventory import ASSET_TEMPLATES

    bands = {band: str(Path(href).resolve()) for band, href in bands.items()}
    from rasterio.warp import transform_bounds

    with rasterio.open(bands["lwir11"]) as ds:
        epsg = ds.crs.to_epsg()
        height, width = ds.height, ds.width
        transform = list(ds.transform)[:6]
        # A STAC bbox and geometry are geographic whatever the raster's CRS.
        # Copying UTM metres here made every scene miss every window.
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": scene_id,
        "collection": "landsat-c2-l2",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": datetime,
            "landsat:scene_id": scene_id,
            "landsat:wrs_path": path,
            "landsat:wrs_row": row,
            "proj:epsg": epsg,
            "proj:shape": [height, width],
            "proj:transform": transform,
        },
        "assets": {
            band: {**ASSET_TEMPLATES[band], "href": href}
            for band, href in bands.items()
        },
        "links": [],
    }


def rehearsal_items(bbox, n: int, directory: Path, *, seed: int = 0):
    """`n` synthetic scenes over `bbox`, as files and the items that name them.

    Each scene is a 1.7 degree square at 1/60 degree, placed on a 7 x 7 walk
    across the tile, with a warm field in `lwir11` and a clear `qa_pixel`. The
    grid is coarse so a full tile rehearses in seconds; `odc.stac` resamples it
    onto the run's grid the way it would a real scene.

    Returns:
        `(items, item_bboxes)`.
    """
    import rasterio
    from rasterio.transform import from_origin

    from lst_qa import LWIR_OFFSET_C, LWIR_SCALE

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, s, e, no = bbox
    res, edge = 1.0 / 60.0, 1.7
    px = int(round(edge / res))
    items, boxes = [], []
    for i in range(n):
        west = w + (e - w) * (i % 7) / 7 - 0.3
        south = s + (no - s) * (i // 7 % 7) / 7 - 0.3
        north = south + edge
        celsius = rng.normal(45.0, 6.0, (px, px)).astype("float32")
        dn = np.rint((celsius - LWIR_OFFSET_C) / LWIR_SCALE).astype("uint16")
        qa = np.full((px, px), 0b1000000, dtype="uint16")
        paths = {}
        for band, values in (("lwir11", dn), ("qa_pixel", qa)):
            path = directory / f"scene{i:04d}_{band}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=px,
                width=px,
                count=1,
                dtype="uint16",
                crs="EPSG:4326",
                transform=from_origin(west, north, res, res),
                nodata=0,
                tiled=True,
                blockxsize=64,
                blockysize=64,
            ) as ds:
                ds.write(values, 1)
            paths[band] = str(path)
        month = 1 + i % 12
        day = 1 + (i // 12) % 28
        items.append(
            item_for_files(
                f"REHEARSAL{i:05d}",
                paths,
                datetime=f"2023-{month:02d}-{day:02d}T12:{i % 60:02d}:00Z",
                path=f"{220 + i % 6:03d}",
                row=f"{80 + i % 3:03d}",
            )
        )
        boxes.append((west, south, west + edge, north))
    return items, boxes


def warm_worker() -> int:
    """Import the read stack on one worker before any real task runs there.

    A spawned worker starts empty. Its first task unpickles arguments that
    reference `odc.loader`, and that package's import is not re-entrant: when
    the deserialising thread and the executing thread both enter it, one of
    them sees `odc.loader._reader` half built and the task dies with
    `ImportError: cannot import name 'nodata_mask'`. MEASURED at about one run
    in 240 under the test suite. Importing once here, and waiting for it,
    leaves every later import a cache hit.
    """
    import os

    import odc.geo.xr  # noqa: F401
    import odc.stac  # noqa: F401
    import rasterio  # noqa: F401

    return os.getpid()


def warm_workers(client, cluster) -> int:
    """Run `warm_worker` on every worker and wait. Returns the process count.

    Targets each worker by address when the cluster exposes them, and
    otherwise floods the cluster with enough short tasks that every worker
    takes at least one.
    """
    addresses = list(getattr(cluster, "_worker_addresses", None) or [])
    if addresses:
        futures = [client.submit(warm_worker, workers=[addr]) for addr in addresses]
    else:
        futures = [client.submit(warm_worker, key=f"warm-{i}") for i in range(64)]
    return len({future.result() for future in futures})


__all__ = [
    "DEFAULT_CHUNK_PX",
    "BlockOutputs",
    "BlockSpec",
    "FileLock",
    "SceneVectors",
    "block_bytes",
    "block_depths",
    "block_edges",
    "block_vectors",
    "block_weights",
    "build_block_plan",
    "build_graph",
    "cleanup_staging",
    "compute_all",
    "finish_staging",
    "fused_block",
    "item_for_files",
    "item_times",
    "lazy_weights",
    "memory_demand",
    "memory_guard",
    "months_of",
    "open_stack",
    "per_scene_vectors",
    "plan_depths",
    "raster_shape",
    "read_block",
    "reduce_block",
    "rehearsal_items",
    "scene_vectors",
    "spatial_dims",
    "staging_targets",
    "staging_writes",
    "submit_blocks",
    "time_order",
]

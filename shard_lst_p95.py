# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2", "dask", "odc-stac", "pystac-client",
#   "planetary-computer", "xarray", "numpy", "geopandas",
#   "psutil", "rich", "boto3", "pyarrow>=16",
#   "rasterio", "shapely", "pyogrio",
#   "stac-geoparquet", "matplotlib",
# ]
# ///
"""Sharded p95 LST composite. One shard, one task, no shuffle.

The array-graph version does not scale. A p95 over a 9000x9000 area and 1765
scenes makes dask reorganise 572 GB of float32 from time-major read blocks into
space-major reduce blocks. Measured on EC2: `rechunk-merge` moved 182 of the
241 GiB shuffled, 76% of all transfer, and the run spilled 1.21 TiB against a
58 GB input. No block size avoids it, because the shuffle is inherent to
splitting one reduction across many workers.

This version splits the *problem* instead of the array. Each shard is a small
bbox processed entirely inside one worker: load, mask, reduce, encode, return.
Nothing crosses a worker boundary, so there is no rechunk and no shuffle.

    512 x 512 px x 1765 scenes x 15 bytes = 6.9 GB per shard

Fifteen bytes, not four. The decoded float32 stack is one of five arrays live
at once, which comes to 13, and a windowed read of a tiled COG holds about two
bytes more that the five do not name. `shard_bytes` counted only the decoded
stack until a fleet instance ran out of memory, and then counted 13 until a
staged sweep at fleet depth read 8% above it. The figure stays constant as the
area grows. A quarter tile is 324 shards and a full tile is 1,296. Frisky
schedules 250,000-400,000 tasks/s, so the task count is free.

    uv run shard_lst_p95.py --bbox=-62.5,-35.0,-60.0,-32.5 \
        --pixels-per-degree 3600 --shard 512 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import aster_ged
import destripe
import item_table
import masks
import staging
from aster_ged import DEFAULT_NUMOBS_URI
from cog_catalog import (
    DEFAULT_HOST_NAME,
    DEFAULT_HOST_URL,
    DEFAULT_LICENSE,
    MONTH_NAMES,
    catalog_provenance,
    check_catalog_inputs,
    collection_id_for_window,
    write_catalog,
)
from lst_qa import (
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    encode_celsius,
    masked_celsius,
)
from land_tiles import tile_bounds
from memory_sampler import MemorySampler
from stac_window import (
    DEFAULT_CLOUD_COVER_LT,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
)
from tile_inventory import (
    INVENTORY_SCHEMA_VERSION,
    check_manifest,
    items_for_tile,
    provenance,
    read_manifest,
)

#: Where the staged artifacts live on a VM unless the driver says otherwise.
DEFAULT_INVENTORY_URI = Path("artifacts/tile_scene_inventory.parquet")

#: Where scene objects are fetched to before the cluster starts. Overridden by
#: `LST_STAGE_DIR` and then by `--stage-dir`. The system temp directory is the
#: default because it is the one path that exists on every machine; a fleet
#: instance should point this at its NVMe mount instead.
DEFAULT_STAGE_DIR = Path(tempfile.gettempdir()) / "landsat-lst-stage"

#: The read environment still has a source, because requester-pays and the
#: region belong to the bucket the hrefs point at. It no longer selects a
#: catalogue: the items come from the inventory, and every href it writes is
#: `s3://usgs-landsat`.
#:
#: `planetary-computer` is listed and then refused. Keeping it in `choices`
#: means the run stops with a sentence that says why, rather than with
#: argparse's "invalid choice", which would read as a typo. Removing it
#: silently would be worse still: the flag used to select a catalogue, so a
#: command line that carries it is asking for something this path cannot do.
READ_SOURCES = ("earth-search", "planetary-computer")

#: The only source the sharded path can read. See `configure_read_env`.
SUPPORTED_READ_SOURCE = "earth-search"

#: Band order of the qa_count asset, and the report's column order. Defined
#: once in cog_catalog, because the COG band descriptions have to match.
MONTHS = MONTH_NAMES
GIB = 1024.0**3

#: How long `drive_shards` waits when neither staging nor the gather has
#: anything ready. A shard runs about a second, so this costs a fraction of a
#: core and bounds how long a freed shard sits unsubmitted.
POLL_INTERVAL_S = 0.005

#: The scene table, written beside the staged scenes rather than into the
#: output directory. `staging.cleanup` already removes that directory, and
#: 7.8 MB of scratch does not belong in an artifact that ships.
ITEM_TABLE_NAME = "item-table.json"


# --------------------------------------------------------------------------
# Read environment
# --------------------------------------------------------------------------


def configure_read_env(source: str = SUPPORTED_READ_SOURCE) -> None:
    """Set the GDAL and AWS variables that every S3 read path depends on.

    The number of HTTP requests GDAL issues is a function of these settings.
    `GDAL_DISABLE_READDIR_ON_OPEN` suppresses a directory listing on each open,
    and `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` collapses adjacent block reads into
    one request. A request count measured without them describes a different
    pipeline, so anything that reads scenes must call this first.

    Raises:
        SystemExit: for any source but `earth-search`. The inventory writes
            `s3://usgs-landsat` hrefs and that bucket is requester-pays, so a
            different source would skip `AWS_REQUEST_PAYER` and every read
            would fail on a bucket the run is entitled to read. This used to
            pass silently.
    """
    if source != SUPPORTED_READ_SOURCE:
        msg = (
            f"--source {source} cannot read this inventory. Every href it "
            f"holds is s3://usgs-landsat, which is requester-pays, and only "
            f"--source {SUPPORTED_READ_SOURCE} sets AWS_REQUEST_PAYER for it. "
            f"The flag selected a catalogue before the inventory replaced the "
            f"per-tile search. It now selects a read environment only. The "
            f"reference catalogue module is where a Planetary Computer query "
            f"belongs."
        )
        raise SystemExit(msg)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
    os.environ.setdefault(
        "AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-west-2")
    )


# --------------------------------------------------------------------------
# Shard geometry
# --------------------------------------------------------------------------


class Shard:
    """One tile of the output grid, addressed in pixels and in degrees."""

    __slots__ = ("row", "col", "y0", "x0", "ny", "nx", "bbox")

    def __init__(self, row, col, y0, x0, ny, nx, bbox):
        self.row, self.col = row, col
        self.y0, self.x0 = y0, x0
        self.ny, self.nx = ny, nx
        self.bbox = bbox

    def __repr__(self):
        return f"Shard(r{self.row} c{self.col} {self.ny}x{self.nx} {self.bbox})"


def plan_shards(
    bbox, pixels_per_degree: int, shard: int
) -> tuple[list[Shard], int, int]:
    """Cut the output grid into shard x shard pixel blocks.

    The grid is anchored to whole degrees, not to the bbox, so a shard lands on
    the same pixels no matter which request produced it. Edge shards are
    smaller rather than overhanging.
    """
    w, s, e, n = bbox
    res = 1.0 / pixels_per_degree
    height = int(round((n - s) * pixels_per_degree))
    width = int(round((e - w) * pixels_per_degree))

    shards = []
    for row, y0 in enumerate(range(0, height, shard)):
        ny = min(shard, height - y0)
        # Row 0 is the northern edge; latitude decreases as y grows.
        north = n - y0 * res
        south = north - ny * res
        for col, x0 in enumerate(range(0, width, shard)):
            nx = min(shard, width - x0)
            west = w + x0 * res
            east = west + nx * res
            shards.append(Shard(row, col, y0, x0, ny, nx, (west, south, east, north)))
    return shards, height, width


def items_for_shard(shard: Shard, item_bboxes) -> list[int]:
    """Indices of items whose footprint intersects this shard.

    A 512 px shard at 1/3600 degree is 0.14 degrees across; a Landsat scene is
    roughly 1.7. So most scenes miss most shards, and sending the whole item
    list to every task would waste both memory and reads.
    """
    w, s, e, n = shard.bbox
    return [
        i
        for i, (iw, isouth, ie, inorth) in enumerate(item_bboxes)
        if iw < e and ie > w and isouth < n and inorth > s
    ]


#: Bytes per pixel-scene that one shard holds at its peak. `process_shard` has
#: five arrays live at once, not the one an earlier version of this function
#: counted:
#:
#:     dn      uint16   2      the raw thermal stack
#:     qa      uint16   2      the QA stack
#:     celsius float32  4      the decoded stack
#:     valid   bool     1      the mask, kept for the monthly counts
#:     copy    float32  4      nanpercentile partitions a copy, not in place
#:
#: Those five sum to 13, and 13 under-predicts. Two bytes are not in the list.
#:
#: MEASURED by `measure_shard_memory.py --mode memory --stage-dir` against
#: 1,615 real staged scenes on an `m6id.16xlarge`, at the depths a fleet shard
#: carries. Least squares puts the slope at 13.68 bytes per pixel-scene at
#: 360 px and 14.52 at 512, and a 13-byte model reads low at 600 and 820
#: scenes on both edges, by up to 8%:
#:
#:     512 px, 820 scenes:  measured 3.08 GiB, 13-byte model 2.85
#:     360 px, 820 scenes:  measured 1.57 GiB, 13-byte model 1.54
#:
#: The surplus is the windowed read of a tiled COG: GDAL decodes whole blocks
#: and `odc.stac` assembles them into the target array, which the five named
#: arrays do not cover. It is not attributed to a specific allocation, because
#: nothing here has profiled one. So 15 is the five named arrays plus measured
#: read overhead, and it bounds every point of all six committed sweeps.
#:
#: The synthetic fixture is what made 13 look safe. It writes one untiled
#: raster at the shard's own edge and read it whole, so it never allocates
#: that intermediate, and it fits a slope of 12.7 to 13.2. A sweep that stops
#: below about 280 scenes agrees with 13 as well, because the fixed term still
#: covers the gap there. Every fleet shard runs deeper: 195 to 820.
#:
#: Each point runs in a fresh interpreter, and it has to. glibc does not return
#: freed arenas promptly, so measuring a second shard in the same process
#: reports the high-water mark of the first: two contaminated sweeps put the
#: slope at 17 and 18 and disagreed with each other by 18% at 700 scenes.
SHARD_BYTES_PER_PIXEL_SCENE = 15

#: Per-worker overhead outside the arrays, in GiB. From the same measurement.
SHARD_FIXED_GIB = 0.25


def shard_bytes(shard_px: int, n_scenes: int) -> float:
    """Peak working set for one shard, in GiB.

    Counting only the float32 stack understated this by 3.9x, and every memory
    decision in the pipeline read the low number: the dry-run budget, the shard
    size, and the worker count a fleet instance is launched with. A run
    configured from it put 64 workers wanting 97 GiB on a 128 GiB box that had
    just written 78 GB of staged scenes into page cache, and the workers died
    at cluster start.

    `FINDINGS.md` recorded the symptom before the model was fixed: memory ran
    at 94% of a 1.6 GiB limit that this function called 0.39 GiB.
    """
    arrays = shard_px * shard_px * n_scenes * SHARD_BYTES_PER_PIXEL_SCENE / GIB
    return arrays + SHARD_FIXED_GIB


#: Bytes the client holds per output pixel while it gathers. `lst_out` is
#: uint16 at height by width, `qa_out` is uint8 at 12 by height by width, and
#: the output mask is two bools at height by width. A full tile at 18,000 px
#: square is 4.8 GiB of it.
#:
#: Both masks are built before staging, not after the gather, so that a tile
#: the water rule empties costs nothing. They are therefore live for the whole
#: run and the budget has to name them. Fourteen bytes here put 0.6 GiB of
#: array outside the model on the largest tile the fleet runs.
CLIENT_BYTES_PER_OUTPUT_PIXEL = 2 + 12 + 2

#: What `--emit-pooled` adds: `pooled_out`, a fourth full-tile array at uint16.
#: Counted separately because it is off by default, and naming it in the sum
#: above would reserve 0.6 GiB on every run that does not ask for it.
POOLED_BYTES_PER_OUTPUT_PIXEL = 2


def client_bytes(width: int, height: int, *, emit_pooled: bool = False) -> float:
    """The full-tile arrays the client holds while it gathers, in GiB."""
    per_pixel = CLIENT_BYTES_PER_OUTPUT_PIXEL + (
        POOLED_BYTES_PER_OUTPUT_PIXEL if emit_pooled else 0
    )
    return width * height * per_pixel / GIB


def slice_demand(
    shard_px: int,
    depths,
    workers: int,
    width: int,
    height: int,
    *,
    emit_pooled: bool = False,
) -> float:
    """What one machine needs to run a slice, in GiB.

    The sum of the deepest `workers` shards plus the client's full-tile arrays.
    Only that many count: beyond them the slice queues rather than running
    wider, and the client gathers into its arrays while the workers are still
    allocating.

    `worker_memory_guard` refuses on this figure and the dry run reports it, so
    they are the same number by construction. They used to be two expressions,
    and the dry run's was `worst shard x slots`: the reading
    `worker_memory_guard` replaced after it over-reserved by 2.77x. On
    `N45E100` at 512 px the two read 261.5 GiB and 225.3 GiB, so a dry run
    printed `OVER by 5.5 GiB` about a run this guard accepts with 30 GiB to
    spare.
    """
    ordered = sorted(depths, reverse=True)[:workers]
    arrays = sum(shard_bytes(shard_px, n) for n in ordered)
    return arrays + client_bytes(width, height, emit_pooled=emit_pooled)


def worker_memory_guard(
    shard_px: int,
    depths,
    workers: int,
    width: int,
    height: int,
    *,
    total_bytes: int | None = None,
    emit_pooled: bool = False,
) -> float:
    """Refuse a configuration that cannot fit, before the cluster starts.

    `staging.disk_guard` refuses a fetch that cannot finish. This is the same
    guard on the other resource, and it was missing. `shard_bytes` was
    corrected after a `c6id.16xlarge` lost ten workers to coredumps, but the
    corrected number only ever reached a `print`. The same configuration would
    have launched again with a larger figure on the screen.

    `depths` is the scene count of every shard in this slice, and the demand is
    their sum plus the client's two full-tile arrays, because the client gathers
    into those while the workers are still allocating. Only the deepest
    `workers` shards count: beyond that the slice queues rather than running
    wider.

    Multiplying the worst shard by the slot count is the reading this replaced,
    and it over-reserved by 2.77x on the one slice that has been measured.
    MEASURED on an `m6id.16xlarge`, 64 shards of S30W065 at 360 px running 203
    to 820 scenes deep:

        64 x worst shard        102.6 GiB
        sum of actual depths     64.0 GiB
        simultaneous peak        37.0 GiB, sampled at 0.5 s

    The slice held one 820-scene shard and a median of 401, so the worst shard
    is not what 63 of the workers were holding. The remaining 1.73x is peak
    non-coincidence: the sum of each worker's own high-water mark came to
    48.6 GiB against 37.0 ever live at once. That headroom is deliberate,
    because a sampler cannot prove the coincident peak it never caught.

    Returns:
        The demand in GiB, so the caller can report what it checked.

    Raises:
        SystemExit: naming the demand, the machine, and both escapes. The
            operator's next decision is a smaller shard or fewer workers, and
            the message carries the edge that would fit.
    """
    if total_bytes is None:
        import psutil

        total_bytes = psutil.virtual_memory().total
    ordered = sorted(depths, reverse=True)[:workers]
    client = client_bytes(width, height, emit_pooled=emit_pooled)
    if not ordered:
        return client
    arrays = sum(shard_bytes(shard_px, n) for n in ordered)
    demand = slice_demand(
        shard_px, depths, workers, width, height, emit_pooled=emit_pooled
    )
    total = total_bytes / GIB
    if demand <= total:
        return demand
    # Solve for the edge whose arrays leave the fixed terms room. Reported
    # rather than applied, because shard size changes the output layout.
    scene_px = sum(ordered)
    room = total - client - len(ordered) * SHARD_FIXED_GIB
    fits = (
        int((room * GIB / (scene_px * SHARD_BYTES_PER_PIXEL_SCENE)) ** 0.5)
        if room > 0
        else 0
    )
    msg = (
        f"{len(ordered)} shards at {shard_px} px, {ordered[-1]:,} to "
        f"{ordered[0]:,} scenes deep, need {arrays:.1f} GiB between them, plus "
        f"{client:.2f} GiB of client output. That is {demand:.1f} GiB and this "
        f"machine has {total:.1f} GiB. Use --shard {fits} or smaller, drop "
        f"--workers, or pass --force to run it anyway. An undersized budget is "
        f"what killed a c6id.16xlarge mid-run."
    )
    raise SystemExit(msg)


# --------------------------------------------------------------------------
# The unit of work. Everything here happens inside one worker.
# --------------------------------------------------------------------------


def rehearse_shard(
    shard: Shard,
    item_dicts,
    crs: str,
    resolution: float,
    read_threads: int = 4,
    correction=None,
) -> dict:
    """Same contract as process_shard, with synthetic pixels and no S3.

    Exercises everything a real run does except the read: submit, the return
    payload, gather, assembly into the output raster, part writing and merge.
    Those are the paths that broke on billed instances, and all of them are
    testable on a laptop for nothing.
    """
    import numpy as np

    rng = np.random.default_rng(shard.row * 10007 + shard.col)
    n = max(len(item_dicts), 1)
    lst = rng.normal(45.0, 6.0, (shard.ny, shard.nx)).astype("float32")
    dn = encode_celsius(lst)
    qa = np.full((12, shard.ny, shard.nx), min(n // 12, 255), dtype="uint8")
    time.sleep(0.01)
    return {
        "row": shard.row,
        "col": shard.col,
        "y0": shard.y0,
        "x0": shard.x0,
        "lst_p95": dn,
        "qa_count": qa,
        "n_scenes": n,
        "n_scenes_kept": n,
        "n_rejected": 0,
        "n_pooled_fallback": 0,
        "load_s": 0.0,
        "reduce_s": 0.0,
    }


def load_shard(
    shard: Shard, item_dicts, crs: str, resolution: float, read_threads: int = 4
):
    """Read one shard's scenes. The only part of a shard that touches S3.

    Split out of `process_shard` so a caller can reduce one stack more than
    once. `measure_seam.py` composites four arms over the same pixels, and
    calling `process_shard` per arm read the same objects four times.

    The returned dataset holds the source DN at uint16, not decoded Celsius.
    `masked_celsius` allocates its own float32 array and never writes back, so
    each reduction can decode a fresh stack from this one. That matters because
    `destripe.apply_to_stack` de-biases in place: a second arm handed the first
    arm's array would reduce a stack already shifted.

    Returns:
        `(data, items, load_s)`. `items` are the parsed pystac objects, which
        the reduction needs to join per-scene values to the time axis.
    """
    import pystac
    from odc.geo import CRS
    from odc.stac import stac_load

    ydim, xdim = ("y", "x") if CRS(crs).projected else ("latitude", "longitude")

    # Items travel as plain dicts. pystac objects are heavier to pickle and we
    # send a different subset to every shard.
    items = [pystac.Item.from_dict(d) for d in item_dicts]

    # Measured on the 8-shard probe: loading was 94% of shard time (227 s of
    # 242 s) because chunks=None reads scenes one at a time on one thread,
    # leaving the other three idle. One chunk per scene lets the worker's own
    # threads read in parallel. The rechunk stays inside this process, so there
    # is still no cross-worker shuffle, which is the whole point of sharding.
    t0 = time.perf_counter()
    data = stac_load(
        items,
        bands=("lwir11", "qa_pixel"),
        crs=crs,
        resolution=resolution,
        bbox=shard.bbox,
        groupby="landsat:scene_id",
        chunks={"time": 1, ydim: -1, xdim: -1},
    ).compute(scheduler="threads", num_workers=read_threads)
    return data, items, time.perf_counter() - t0


def process_shard(
    shard: Shard,
    item_dicts,
    crs: str,
    resolution: float,
    read_threads: int = 4,
    correction=None,
) -> dict:
    """Load, mask, reduce and encode one shard. Returns small arrays only.

    Deliberately eager: no dask inside. The whole point is that this fits in
    memory, so a lazy graph would only reintroduce the rechunk we are avoiding.

    `correction` is what `destripe.shard_correction` cut out of the tile's prep
    artifact for this shard: one offset and one keep flag per item, and the
    cross-fade weights already resampled onto this shard's grid. Passing None
    composites the pooled percentile this repository built before either
    correction existed, which is what `--no-destripe --no-feather` asks for.

    Both corrections happen on the stack already in memory. The de-biasing is a
    scalar subtraction per scene, and the per-path percentiles reduce disjoint
    slices of the same array, so neither adds a read and neither adds a pass.

    This is `load_shard` then `reduce_shard`, and it keeps the signature and
    the return contract it has always had. Four test modules and
    `measure_shard_memory.py` call it directly.
    """
    return reduce_shard(
        shard,
        *load_shard(shard, item_dicts, crs, resolution, read_threads),
        correction=correction,
    )


def reduce_shard(shard: Shard, data, items, load_s: float, correction=None) -> dict:
    """Mask, reduce and encode one loaded stack. Reads nothing.

    Decodes its own Celsius stack from `data`, so calling it twice over one
    load gives two independent reductions. `destripe.apply_to_stack` writes NaN
    into that stack and subtracts the offsets in place, which is why the decode
    cannot be hoisted out with the read.
    """
    import numpy as np

    # One definition of a usable observation, shared with the array-graph path
    # in profile_lst_p95. It drops source fill, QA_PIXEL bits 1 to 5, and any
    # decoded value outside [-50, 80] C. The range check is what removes the
    # reprojected scene edges, where interpolation against the DN 0 fill leaves
    # small nonzero values that decode near -124 C and that an exact fill
    # comparison cannot see. All of it happens before the percentile, because a
    # value that reaches nanpercentile has already moved the answer.
    lst, valid = masked_celsius(data["lwir11"].values, data["qa_pixel"].values)

    t1 = time.perf_counter()
    n_rejected = 0
    n_pooled_fallback = 0
    pooled = None
    if correction is None:
        with np.errstate(all="ignore"):
            p95 = np.nanpercentile(lst, 95, axis=0)
    else:
        labels, n_rejected = destripe.apply_to_stack(
            lst, valid, items, data["time"].values, correction
        )
        if correction["emit_pooled"]:
            pooled = encode_celsius(destripe.pooled_percentile(lst))
        if correction["paths"]:
            p95, n_pooled_fallback = destripe.feathered_percentile(
                lst, labels, correction["paths"], correction["weight"]
            )
        else:
            p95 = destripe.pooled_percentile(lst)
            n_pooled_fallback = int(np.isfinite(p95).sum())
    t_reduce = time.perf_counter() - t1

    months = data["time"].dt.month.values
    qa_count = np.zeros((12, p95.shape[0], p95.shape[1]), dtype="uint8")
    for m in range(1, 13):
        sel = months == m
        if sel.any():
            qa_count[m - 1] = np.minimum(valid[sel].sum(axis=0), 255).astype("uint8")

    dn_out = encode_celsius(p95)
    out = {
        "row": shard.row,
        "col": shard.col,
        "y0": shard.y0,
        "x0": shard.x0,
        "lst_p95": dn_out,
        "qa_count": qa_count,
        # Scenes loaded, which is what this key has always meant and what the
        # memory model in `shard_bytes` is stated against. Survivors are a
        # separate key rather than a quieter redefinition of this one.
        "n_scenes": int(lst.shape[0]),
        "n_scenes_kept": int(lst.shape[0]) - n_rejected,
        "n_rejected": n_rejected,
        # Pixels the cross-fade could not describe, which took the pooled
        # percentile instead. A shard well inside one swath reports 0. A high
        # share says the swath definition missed ground the scenes did reach,
        # which is the number to watch per tile.
        "n_pooled_fallback": n_pooled_fallback,
        "load_s": load_s,
        "reduce_s": t_reduce,
    }
    if pooled is not None:
        out["lst_p95_pooled"] = pooled
    return out


def shard_task(
    shard: Shard,
    table_path,
    indices,
    crs: str,
    resolution: float,
    read_threads: int = 4,
    correction=None,
) -> dict:
    """What the client submits. Resolves the scene table inside the worker.

    `process_shard` still takes item dicts. Four test modules and
    `measure_shard_memory.py` call it directly, and the staged-parity test has
    to run the same function through two href regimes, so the unit of work
    keeps the signature it had. This wrapper is the only thing that changed
    about how it is reached.

    The saving is the argument list. A shard used to carry its own 509 item
    dicts, 423,128 B per task and 0.55 GB across a full tile. It now carries a
    path and a list of positions. MEASURED by `measure_submit_cost.py` against
    a real cluster: 5.3 to 6.1 ms per submit against 0.024 to 0.031 ms.
    """
    return process_shard(
        shard,
        item_table.select(table_path, indices),
        crs,
        resolution,
        read_threads,
        correction,
    )


def rehearse_task(
    shard: Shard,
    table_path,
    indices,
    crs: str,
    resolution: float,
    read_threads: int = 4,
    correction=None,
) -> dict:
    """The rehearsal counterpart. It reads no table, because it reads nothing.

    `rehearse_shard` only ever used `len(item_dicts)`, and `len(indices)` is the
    same number. A rehearsal never carries a correction, because `--rehearse`
    reads no prep file, but the signature matches so one driver submits both.
    """
    return rehearse_shard(shard, indices, crs, resolution, read_threads, correction)


def resolve_area(args):
    """The bbox this run covers, and the tile it belongs to.

    `--tile` is the production form: it fixes the bbox on the shared grid, so
    two machines given the same tile cut the same pixels. `--bbox` stays for
    dry runs and rehearsals, which plan shards without reading anything.

    Returns:
        The bbox as `(west, south, east, north)`, and the tile id or None.
    """
    if args.tile and args.bbox:
        raise SystemExit("pass --tile or --bbox, not both")
    if args.tile:
        return tile_bounds(args.tile), args.tile
    if not args.bbox:
        raise SystemExit("pass --tile (production) or --bbox (dry run)")
    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox needs west,south,east,north")
    return bbox, None


def load_tile_items(args, tile_id: str):
    """Every scene for one tile, from the precomputed inventory.

    Replaces the per-tile Earth Search query. The artifact is built once by
    `usgs_inventory`, staged for the fleet, and read here with a row-group
    lookup. Nothing in this function opens a catalogue.

    The manifest is checked before any read. A window or a filter the artifact
    does not cover stops the run here, which is the point: the alternative is
    a finished composite built from the wrong scenes.

    Returns:
        The item dicts, their tile-local bboxes, and the run provenance.
    """
    manifest = read_manifest(args.inventory_uri)
    check_manifest(
        manifest,
        start=args.start,
        end=args.end,
        platforms=args.platforms,
        cloud_cover_lt=args.cloud_cover_lt,
        schema_version=INVENTORY_SCHEMA_VERSION,
    )
    items, boxes = items_for_tile(
        args.inventory_uri, tile_id, bounds=tile_bounds(tile_id)
    )
    return items, boxes, provenance(manifest)


def stage_scenes_for(args, item_dicts, work_idx):
    """Fetch this slice's scene objects to local disk, or say why it did not.

    Runs after `--max-shards`, so a two-shard smoke run fetches what those two
    shards need rather than the whole slice, and before the cluster starts.

    That ordering was questioned and is now measured. Fetching beside the
    shards it feeds sounds free, because the fetch is network work and the
    shards are processor work. It is not: MEASURED on an `m6id.16xlarge`, the
    same 1,998 objects and 78.9 GiB stage in 91.9 s with the machine to
    themselves and in 358.7 s beside 64 busy workers. Staging spends its time
    on TLS, HTTP and the copy loop, all of which want a core, so 64 workers
    take 3.9x of it away. The overlap saved 220 s of wall clock on a 250-shard
    slice and paid 267 s for it.

    Returns:
        The staging report, or None when the run reads from S3. The report is
        this run's S3 line, counted rather than derived from a sampled
        requests-per-read.
    """
    if args.rehearse:
        print("stage         skipped: the rehearsal reads no objects")
        return None
    report = staging.stage_scenes(
        item_dicts,
        sorted({i for _, idx in work_idx for i in idx}),
        args.stage_dir,
        threads=args.stage_threads,
    )
    report_staging(report)
    return report


def report_staging(report) -> None:
    """The two console lines a staged run prints about what it fetched."""
    print(
        f"stage         {report['objects']:,} objects, "
        f"{report['bytes'] / GIB:.1f} GiB in {report['seconds']:.1f}s "
        f"-> {report['stage_dir']}"
    )
    reused = report.get("reused", 0)
    already = f", {reused:,} already staged" if reused else ""
    print(
        f"              {report['get_requests']:,} billable GETs, "
        f"{report['retries']} retries{already}"
    )


def drive_shards(
    client,
    fn,
    work_idx,
    table_path,
    crs,
    res,
    read_threads,
    *,
    assemble,
    marks,
    t0,
    correction_of=None,
):
    """Submit every shard, then assemble results as they return.

    Shards are submitted in batches rather than one at a time.
    `_AsCompleted.add` clears its completion cursor, so adding singly makes the
    next drain rescan every pending future and the gather quadratic in the
    shard count.

    Results are assembled one at a time and dropped. `client.gather` on every
    future at once held about 1 GB of results and aborted the process with a
    Rust panic across the PyO3 boundary at 90% completion, taking 290 finished
    shards with it.

    `correction_of(shard, indices)` is the seam correction for one shard, or
    None to composite pooled. It is called inside the submit comprehension so
    each shard's weight window is serialised and released rather than held for
    the whole tile. The weights are the one part of a task payload that cannot
    move into the table: they are cut to the shard's own window.

    Returns:
        The per-shard stats in completion order, and the seconds spent inside
        `client.submit`.
    """
    import frisky  # deferred, like main's: a dry run must not need it

    total = len(work_idx)
    stats: list[dict] = []
    completed = frisky.as_completed([], raise_errors=False)

    t_submit = time.perf_counter()
    marks["first_submit_s"] = t_submit - t0

    def task_args(shard, idx):
        """The submitted argument list, with the correction only if there is one.

        A run without `--tile-prep` submits what it always submitted. The
        correction is the one argument that cannot move into the scene table,
        because it is cut to the shard's own window.
        """
        base = (shard, table_path, idx, crs, res, read_threads)
        return base if correction_of is None else (*base, correction_of(shard, idx))

    completed.update(
        [client.submit(fn, *task_args(shard, idx)) for shard, idx in work_idx]
    )
    submit_s = time.perf_counter() - t_submit
    marks["last_submit_s"] = time.perf_counter() - t0

    done = 0
    #: Shards finished at the last checkpoint, so the line can report the rate
    #: over the last 25 rather than since the start. The cumulative figure hides
    #: page-cache warmup: the first wave of 64 shards runs about 59 s each and
    #: later ones about 11 s, so a running mean reads as a slow cluster for
    #: most of a run and as a rate for none of it.
    last_mark = (0, time.perf_counter())
    while done < total:
        for future in completed.next_batch(block=True):
            stats.append(assemble(future))
            done += 1
            marks.setdefault("first_result_s", time.perf_counter() - t0)
            if done % 25 == 0 or done == total:
                now = time.perf_counter()
                elapsed = now - t0 - marks["first_submit_s"]
                span = (now - last_mark[1]) / max(done - last_mark[0], 1)
                last_mark = (done, now)
                print(
                    f"  {done:4d}/{total}  {elapsed:6.1f}s  "
                    f"{elapsed / done:5.2f}s/shard mean  "
                    f"{span:5.2f}s/shard last 25"
                )
        if done < total and completed.is_empty():
            msg = f"{total - done} of {total} shards never returned a result"
            raise RuntimeError(msg)
    marks["last_result_s"] = time.perf_counter() - t0
    return stats, submit_s


def _target_verdict(args, demand_gib: float) -> str:
    """`  fits` or `  OVER by N GiB` against `--target-memory-gib`.

    A dry run plans for a machine that has not been launched, so the figure it
    checks against has to be named rather than read from the host. Without the
    flag there is nothing to compare and this adds nothing to the line.

    The caller passes the demand rather than its parts, because the dry run
    reports two different demands and only one of them is `worst x slots`. See
    `slice_demand`.
    """
    if not args.target_memory_gib:
        return ""
    over = demand_gib - args.target_memory_gib
    return f"   OVER by {over:.1f} GiB" if over > 0 else "   fits"


def no_thermal_coverage(args, tile_id, bbox, n_scenes, dropped, run_provenance) -> int:
    """Record a tile that holds no thermal scene, and succeed.

    126 of the 895 land tiles are like this, and every one is an ocean tile
    holding a small island. `thermal_href IS NULL` is exactly `OLI_TIRS_L2SR`
    across all 3,083,129 inventory rows, and USGS emits that product where the
    surface temperature algorithm has no usable emissivity. Compositing there
    is not a failure. There is nothing to composite.

    `fleet_plan.py` drops these tiles from the launch list, so a fleet never
    reaches this path. An operator naming the tile by hand does, and gets the
    same artifact a driver keys on. Writing nothing and exiting non-zero would
    make a correct outcome read as a dead machine, which is the distinction
    the barren-shard records exist to preserve.
    """
    summary = {
        "status": "no-thermal-coverage",
        "tile": tile_id,
        "bbox": bbox,
        "n_scenes_inventory": n_scenes,
        "n_scenes": 0,
        "scenes_dropped_no_thermal": dropped,
        "inventory": run_provenance,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        f"no thermal    all {n_scenes:,} scenes of {tile_id} are OLI_TIRS_L2SR "
        f"and carry no thermal band"
    )
    print("              nothing to composite; summary written, no parts")
    print(f"artifacts     {args.out_dir.resolve()}")
    return 0


def no_scene_survives_destriping(args, tile_id, prep, run_provenance) -> int:
    """Every scene of the tile failed the offset rule. Stop rather than ship.

    An empty composite is not a tile with no data. It is a tile whose whole
    scene list was found untrustworthy, and writing it as nodata would present
    that as an observation gap. `nlebovits/landsat-lst` takes the same position
    one level down: prefer honest omission over a questionable correction, and
    prefer a stopped run over an omission that looks like ground truth.

    The cap is the first thing to look at. It was calibrated on mid-latitude
    cropland at a 21.8% rejected share, and a tile at 100% is either somewhere
    that calibration does not describe or a tile whose prep pass went wrong.
    """
    summary = {
        "status": "no-scene-survives-destriping",
        "tile": tile_id,
        "n_scenes": len(prep.offset),
        "max_offset_c": args.max_offset_c,
        "offsets": prep.meta.get("offsets"),
        "inventory": run_provenance,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        f"destripe      all {len(prep.offset):,} scenes of {tile_id} fail the "
        f"{args.max_offset_c:g} C cap or the sparse floor"
    )
    print(f"              {prep.meta.get('offsets')}")
    print("              nothing to composite; summary written, no parts")
    return 1


def check_mask_inputs(args) -> dict | None:
    """Refuse a run whose output mask cannot be built, before it costs anything.

    Same reason the inventory manifest is checked before any read. A tile that
    composites for three hours and then cannot be masked has already bought the
    machine, and a tile written without the mask looks finished while carrying
    sea and ASTER emissivity gaps as temperatures.

    The two artifacts have to agree about the land. The NumObs mosaic was built
    over the cells one land geometry touches; rasterising a different geometry
    against it masks different ground on the two rules.

    The rehearsal is masked like any other run. Its pixels are synthetic but
    its bbox is not: `--tile` fixes it on the production grid, so the mask
    covers real ground and a rehearsal proves the assembly path the fleet uses
    rather than a shorter one. A laptop with no artifacts stops here with the
    command that writes them, which is the same answer a real run gets.

    Returns:
        The ASTER GED provenance for the run summary, or None under
        `--no-output-mask`.

    Raises:
        MaskError: if the land geometry is absent.
        GedError: if the NumObs artifact is absent, or was built from a
            different land geometry.
    """
    if args.no_output_mask:
        print("mask          skipped: --no-output-mask. Sea and ASTER gaps stay")
        return None
    if not args.land_geometry_uri.exists():
        msg = (
            f"no buffered land geometry at {args.land_geometry_uri}. Write it "
            f"with:\n  uv run land_tiles.py --out artifacts/land_tiles.parquet "
            f"--write-geometry {args.land_geometry_uri}"
        )
        raise masks.MaskError(msg)
    manifest = aster_ged.read_manifest(args.numobs_uri)
    aster_ged.check_manifest(
        manifest,
        land_geometry_sha256=masks.geometry_checksum(args.land_geometry_uri),
        path=args.numobs_uri,
    )
    print(
        f"mask          {manifest['granule_count']:,} ASTER GED granules, "
        f"{manifest['collection']['short_name']} v"
        f"{manifest['collection']['version']}"
    )
    return aster_ged.provenance(manifest)


def no_unmasked_pixels(args, tile_id, bbox, counts, ged_provenance) -> int:
    """Record a tile the water rule empties, and succeed.

    A tile whose bbox holds no land inside the buffered geometry has nothing to
    publish. `land_tiles.py` selects tiles from that same geometry, so a tile
    on the fleet's list never reaches here. An operator naming a bbox by hand
    does, and gets the artifact a driver already keys on, the way
    `no_thermal_coverage` does.

    The emissivity rule cannot reach this path. It removes a pixel only where
    the gap region and 70 C coincide, so a tile of nothing but gap cells still
    publishes every pixel that reads an ordinary temperature.

    `n_scenes` is null rather than 0. This runs before the search, so the count
    is unknown here, and writing zero would state a fact about the archive that
    nothing measured. `no_thermal_coverage` runs after the search and does
    write the real number.
    """
    summary = {
        "status": "no-unmasked-pixels",
        "tile": tile_id,
        "bbox": bbox,
        "n_scenes": None,
        "mask": counts
        | {
            "numobs_uri": str(args.numobs_uri),
            "land_geometry_uri": str(args.land_geometry_uri),
            "aster_ged": ged_provenance,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"no coverage   {tile_id} holds no land inside the buffered geometry")
    print("              nothing to publish; summary written, no parts")
    print(f"artifacts     {args.out_dir.resolve()}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sharded p95 LST composite: one shard, one task, no shuffle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tile",
        default=None,
        help="tile id such as S30W065. Sets the bbox from the production grid "
        "and selects this tile's rows in the inventory",
    )
    p.add_argument(
        "--bbox",
        default=None,
        help="west,south,east,north EPSG:4326 (use --bbox=...). Only for "
        "--rehearse and --dry-run; a real run needs --tile, because the "
        "inventory is addressed by tile",
    )
    p.add_argument(
        "--inventory-uri",
        type=Path,
        default=DEFAULT_INVENTORY_URI,
        help="the precomputed tile-scene inventory this run reads",
    )
    p.add_argument(
        "--land-tiles-uri",
        type=Path,
        default=Path("artifacts/land_tiles.parquet"),
        help="the authoritative land-tile list, for the driver",
    )
    p.add_argument(
        "--numobs-uri",
        type=Path,
        default=DEFAULT_NUMOBS_URI,
        help="ASTER GED clear-sky observation counts. A pixel whose count is "
        "zero has no emissivity, so Collection 2 never produced a surface "
        "temperature for it and no window ever will",
    )
    p.add_argument(
        "--land-geometry-uri",
        type=Path,
        default=masks.DEFAULT_LAND_GEOMETRY_URI,
        help="the buffered land geometry the pixel mask rasterises. The same "
        "geometry chose the tile list, and it travels as an artifact so that "
        "a run needs no network",
    )
    p.add_argument(
        "--no-output-mask",
        action="store_true",
        help="write every pixel the composite produced, including sea and "
        "ASTER emissivity gaps. Kept for measuring the mask against its "
        "absence; a published tile always carries it",
    )
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument("--shard", type=int, default=512, help="shard edge in pixels")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--source", choices=sorted(READ_SOURCES), default="earth-search")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=4)
    p.add_argument("--memory-limit-gib", type=float, default=13.0)
    p.add_argument(
        "--read-threads",
        type=int,
        default=4,
        help="threads used to read scenes inside one shard",
    )
    p.add_argument("--max-shards", type=int, default=None, help="cap, for smoke runs")
    p.add_argument(
        "--shard-slice",
        default=None,
        metavar="A:B",
        help="process only shards[A:B] of the plan. The plan is deterministic "
        "and anchored to whole degrees, so slice i on one machine and slice j "
        "on another cover the tile exactly once between them",
    )
    p.add_argument(
        "--rehearse",
        type=int,
        default=0,
        metavar="N",
        help="run the whole pipeline with N synthetic scenes and no S3 reads; "
        "proves submit, gather, assembly, part writing and merge for free",
    )
    p.add_argument(
        "--merge",
        nargs="+",
        default=None,
        metavar="DIR",
        help="assemble a finished tile from the part files written by "
        "--shard-slice runs, then exit",
    )
    p.add_argument("--out-dir", type=Path, default=Path("./shard-run"))
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=Path(os.environ.get("LST_STAGE_DIR", DEFAULT_STAGE_DIR)),
        help="local directory the scene objects are fetched into. Wants "
        "throughput as well as capacity, and now needs both at once: the "
        "fetch writes at up to 922 MB/s while the shards read the files it "
        "has already landed at about 358 MB/s. Pass --no-overlap to put the "
        "two back in sequence",
    )
    p.add_argument(
        "--stage-threads",
        type=int,
        default=None,
        help="threads in the fetch pool. Defaults to min(64, 4 x cores). The "
        "cap was chosen when staging ran on its own; beside compute each "
        "thread spends far more of its life blocked, so more of them keep the "
        "same bytes in flight for the same cores",
    )
    p.add_argument(
        "--keep-staged",
        action="store_true",
        help="leave the staged files behind. Steady state runs one tile per "
        "instance back to back, so the default removes them",
    )
    p.add_argument(
        "--keep-scenes-without-thermal",
        action="store_true",
        help="keep the L2SR products that carry no lwir11 band. They load as "
        "fill and reach neither the percentile nor the monthly counts, so "
        "the default drops them",
    )
    p.add_argument(
        "--no-catalog",
        action="store_true",
        help="skip the COGs and the STAC catalog that --merge writes, leaving "
        "only the .npy arrays",
    )
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="where the catalog lives; defaults to <out-dir>/catalog. Point "
        "every tile's merge at one path to collect them in one catalog",
    )
    p.add_argument(
        "--collection-id",
        default=None,
        help="the published collection id, which is also its directory name. "
        "Defaults to the window the parts were composited over, so "
        "2021-2025 gives lst-p95-2021-2025 and two windows cannot collect "
        "into one collection by omission",
    )
    p.add_argument(
        "--host-name",
        default=DEFAULT_HOST_NAME,
        help="the organization maintaining the published catalog",
    )
    p.add_argument(
        "--host-url",
        default=DEFAULT_HOST_URL,
        help="a page where the catalog maintainer can be reached",
    )
    p.add_argument(
        "--license",
        default=DEFAULT_LICENSE,
        help="SPDX identifier recorded in collection.json",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="run even if slots x read-threads oversubscribes the cores, or "
        "if the worker memory budget exceeds the machine. The memory model "
        "over-predicts by 6 to 14 percent, so an operator who knows that can "
        "spend the margin",
    )
    p.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="seconds between memory samples. The sampler runs in its own "
        "process and writes memory.csv beside the summary, so a run records "
        "the worker RSS that shard_bytes only predicts. A shard runs 40 to "
        "87 s, so the default takes about 120 samples of each one, and "
        "0.05 s produced the same peak from a 6.6x larger file",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="plan shards and print the budget; no cluster, no reads",
    )
    p.add_argument(
        "--target-memory-gib",
        type=float,
        default=None,
        help="RAM of the machine this run is planned for, in GiB. The dry run "
        "checks the worker budget against it and exits 2 if the run would be "
        "refused, so a configuration can be priced before an instance is "
        "launched. Without it the dry run reads no machine at all, because the "
        "host planning a fleet run is not the host doing it",
    )
    p.add_argument(
        "--search-in-dry-run",
        action="store_true",
        help="also hit STAC, to report real scenes per shard",
    )
    p.add_argument(
        "--tile-prep",
        type=Path,
        default=None,
        help="directory holding tile-prep.npz and tile-prep.json, written by "
        "tile_prep.py. Without it the run composites the pooled percentile "
        "and the WRS seam stays in the output",
    )
    p.add_argument(
        "--no-destripe",
        action="store_true",
        help="keep every scene at its own baseline. Reads the prep file for "
        "the swath geometry and ignores the offsets",
    )
    p.add_argument(
        "--no-feather",
        action="store_true",
        help="one pooled percentile instead of one per WRS path",
    )
    p.add_argument(
        "--max-offset-c",
        type=float,
        default=destripe.DESTRIPE_MAX_OFFSET_C,
        help="discard a scene whose absolute offset exceeds this. The prep "
        "file carries the estimate and this decides on it, so sweeping the cap "
        "costs a read of that file rather than another pass over the tile",
    )
    p.add_argument(
        "--emit-pooled",
        action="store_true",
        help="also write a pooled percentile beside the product, from the same "
        "load after the offsets are applied. It isolates the cross-fade and "
        "says nothing about the offsets; measure_seam.py separates those. It "
        "costs a second reduction over the stack, which is not free on an "
        "eager path",
    )
    args = p.parse_args(argv)
    if args.tile_prep is None and (args.no_destripe or args.no_feather):
        raise SystemExit(
            "--no-destripe and --no-feather turn off corrections that need "
            "--tile-prep to be on at all. Without --tile-prep the run is "
            "already pooled and un-de-striped."
        )
    # Refuse an unreadable source here rather than after the shard plan is
    # printed. `configure_read_env` runs late enough that a run could get a
    # full budget report before learning its source cannot read a scene.
    if args.source != SUPPORTED_READ_SOURCE:
        configure_read_env(args.source)
    return args


def mask_rule(args, counts, ged_provenance=None) -> dict | None:
    """The rule a part was masked under, for a merge to compare across parts.

    None under `--no-output-mask`, which is itself a rule a merge has to see:
    one unmasked part beside three masked ones is a raster no single rule
    describes.

    The inputs are named by identity, not by path. An absolute path on the
    machine that masked the part tells a reader of the published catalog
    nothing, and it carries the operator's home directory into a public file.
    The DOI and the two checksums say which artifact was used, which is the
    question a consumer and a merge both ask. `summary.json` keeps the paths,
    because an operator rerunning one slice does want them.
    """
    if counts is None:
        return None
    rule: dict = {
        "gap_buffer_cells": counts.get("gap_buffer_cells"),
        "gap_hot_threshold_c": counts.get("gap_hot_threshold_c"),
        "land_geometry_sha256": masks.geometry_checksum(args.land_geometry_uri),
    }
    if ged_provenance:
        rule["aster_ged"] = ged_provenance
    return rule


def load_tile_prep(args, tile_id: str, item_dicts, run_provenance):
    """The prep artifact, checked against the run that is about to use it.

    Returns None when the run was not given one, which composites pooled.

    Six checks, and each one guards a failure that produces a finished raster
    rather than an error. Compositing against another tile's offsets, another
    grid's weights, another scene list's estimates, or a smoke run's partial
    coverage all look ordinary in the output.

    `item_dicts` must be the list the shards will actually load, after
    `staging.drop_scenes_without_thermal`, because that is the list `tile_prep`
    hashed.

    Raises:
        SystemExit: on any mismatch, naming the command that rebuilds the file.
    """
    if args.tile_prep is None:
        return None
    prep = destripe.load_prep(args.tile_prep)
    rebuild = f"Rebuild it with tile_prep.py --tile {tile_id}."

    blocks = prep.meta.get("blocks") or {}
    if blocks.get("partial"):
        raise SystemExit(
            f"{args.tile_prep} was built from {blocks.get('run')} of "
            f"{blocks.get('with_scenes')} blocks holding scenes, under "
            f"--max-blocks {blocks.get('max_blocks')}. Each quad's swath was "
            f"counted over part of the tile and divided by all of its scenes, "
            f"so the swaths are too small and the offsets rest on too few "
            f"pixels. Rerun tile_prep.py without --max-blocks."
        )

    if prep.meta.get("schema_version") != tile_prep_schema_version():
        raise SystemExit(
            f"{args.tile_prep} carries schema version "
            f"{prep.meta.get('schema_version')} and this version reads "
            f"{tile_prep_schema_version()}. {rebuild}"
        )
    if prep.tile != tile_id:
        raise SystemExit(
            f"{args.tile_prep} was written for tile {prep.tile}, and this run "
            f"is building {tile_id}. {rebuild}"
        )
    if prep.pixels_per_degree != args.pixels_per_degree:
        raise SystemExit(
            f"{args.tile_prep} was built on a 1/{prep.pixels_per_degree} degree "
            f"grid and this run is on 1/{args.pixels_per_degree}. The weights "
            f"would land on the wrong ground. {rebuild}"
        )

    window = {
        "start": args.start,
        "end": args.end,
        "platforms": args.platforms,
        "cloud_cover_lt": args.cloud_cover_lt,
    }
    mine = destripe.scene_digest((destripe.scene_id_of(d) for d in item_dicts), window)
    if prep.digest != mine:
        raise SystemExit(
            f"{args.tile_prep} was fitted over a different scene set or a "
            f"different window: it carries digest {prep.digest or 'none'} and "
            f"this run computes {mine}. Its window was {prep.window} against "
            f"{window} here, over {len(prep.offset)} scenes against "
            f"{len(item_dicts)}. {rebuild}"
        )
    if prep.inventory and run_provenance and prep.inventory != run_provenance:
        raise SystemExit(
            f"{args.tile_prep} was built from a different inventory artifact:\n"
            f"  prep: {prep.inventory}\n"
            f"  run:  {run_provenance}\n"
            f"The digests match, so the same scenes are named, and a rebuilt "
            f"artifact can still move a footprint or a href. {rebuild}"
        )
    return prep


def tile_prep_schema_version() -> int:
    """Read lazily, because `tile_prep` imports this module."""
    import tile_prep

    return tile_prep.PREP_SCHEMA_VERSION


def shard_correction_for(args, prep, shard: Shard, item_dicts):
    """One shard's slice of the prep artifact, or None to composite pooled."""
    if prep is None:
        return None
    return destripe.shard_correction(
        prep,
        item_dicts,
        shard.bbox,
        (shard.ny, shard.nx),
        pixels_per_degree=args.pixels_per_degree,
        max_offset_c=args.max_offset_c,
        debias=not args.no_destripe,
        feather=not args.no_feather,
        emit_pooled=args.emit_pooled,
    )


def correction_rule(args, prep) -> dict | None:
    """The seam correction a part was built under, for a merge to compare.

    None means the pooled percentile with every scene at its own baseline,
    which is itself a rule a merge has to see. One un-de-striped part beside
    three de-striped ones is a raster carrying the seam in one corner only, and
    nothing in the pixels says which corner.
    """
    if prep is None:
        return None
    return {
        "prep_schema_version": prep.meta.get("schema_version"),
        # The scene set the offsets were fitted over. Without it two parts
        # built against different prep files compare equal on every parameter
        # and merge into one raster carrying two different corrections.
        "prep_scene_digest": prep.digest,
        "prep_window": prep.window,
        "prep_tile": prep.tile,
        "prep_bbox": list(prep.bbox),
        "prep_factor": prep.meta.get("prep_factor"),
        "swath_factor": prep.swath_factor,
        "weight_factor": prep.meta.get("weight_factor"),
        "swath_quad_share": prep.meta.get("swath_quad_share"),
        "anomaly_bin_c": prep.meta.get("anomaly_bin_c"),
        "min_offset_samples": prep.meta.get("min_offset_samples"),
        "max_offset_c": None if args.no_destripe else args.max_offset_c,
        "destripe": not args.no_destripe,
        "feather": not args.no_feather,
        "paths": list(prep.paths),
    }


def check_part_rules(metas) -> None:
    """Stop a merge whose parts were built under two different rules.

    Both rules decide pixel values, and neither leaves a mark a reader could
    find in the raster. Two machines that masked the same tile differently
    produce one output that no single rule describes, and one un-de-striped
    part beside three de-striped ones carries the WRS seam in one corner with
    nothing in the pixels to say which corner.

    Raises:
        SystemExit: if any part disagrees with another on either rule. The
            message names the rules and how to rebuild the odd slice.
    """
    for key, verb, remedy in (
        (
            "mask_rule",
            "masked",
            "Rerun the disagreeing slices with the same --numobs-uri, "
            "--land-geometry-uri, and --no-output-mask setting.",
        ),
        (
            "correction_rule",
            "corrected",
            "Rerun the disagreeing slices against the same --tile-prep, with "
            "the same --max-offset-c, --no-destripe, and --no-feather setting.",
        ),
    ):
        rules = {json.dumps(m.get(key), sort_keys=True) for m in metas}
        if len(rules) > 1:
            joined = "\n  ".join(sorted(rules))
            raise SystemExit(
                f"the parts were {verb} under {len(rules)} different rules, so "
                f"no one rule describes the merged tile:\n  {joined}\n{remedy}"
            )


def assemble_parts(parts, raster):
    """Every shard in every part file, laid into one tile's arrays.

    A part carries one `lst_` and one `qa_` key per shard it wrote, and a
    `pooled_` key as well when that slice ran `--emit-pooled`. The flag is
    per-run and `part-meta.json` records nothing about it, so the pooled
    accumulator is allocated on the first key that appears. A merge of parts
    that never used the flag then allocates nothing.

    Returns:
        `(lst, qa, pooled, seen, pooled_seen, n_shards)`. `pooled` is None when
        no part carried a baseline. `seen` and `pooled_seen` are what the
        coverage figures are taken from, and they can differ: slices running
        under different flags still merge, because a diagnostic raster is not
        a reason to refuse a product.
    """
    import numpy as np

    h, w = raster
    lst = np.zeros((h, w), dtype="uint16")
    qa = np.zeros((12, h, w), dtype="uint8")
    seen = np.zeros((h, w), dtype=bool)
    pooled = None
    pooled_seen = np.zeros((h, w), dtype=bool)
    n = 0

    for f in parts:
        with np.load(f) as z:
            for key in z.files:
                if not key.startswith("lst_"):
                    continue
                tag = key[4:]
                y0, x0 = (int(v) for v in tag.split("_"))
                a = z[key]
                q = z["qa_" + tag]
                lst[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a
                qa[:, y0 : y0 + q.shape[1], x0 : x0 + q.shape[2]] = q
                seen[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = True
                if "pooled_" + tag in z.files:
                    if pooled is None:
                        pooled = np.zeros((h, w), dtype="uint16")
                    b = z["pooled_" + tag]
                    pooled[y0 : y0 + b.shape[0], x0 : x0 + b.shape[1]] = b
                    pooled_seen[y0 : y0 + b.shape[0], x0 : x0 + b.shape[1]] = True
                n += 1
    return lst, qa, pooled, seen, pooled_seen, n


def merge_parts(dirs, out_dir: Path, args) -> int:
    """Assemble one tile from the parts written by --shard-slice runs.

    The merge applies no mask. Every part was masked by the machine that wrote
    it, over that machine's own slice, so the pixels arrive already screened.
    What the merge does check is that they were screened the same way: it reads
    every part's meta rather than the first, and stops when two disagree.

    The `.npy` arrays stay: the measurement scripts read them, and they are the
    cheapest way to reopen a merge. The COGs and the catalog beside them are
    what a client consumes.

    Whatever the catalog needs from `part-meta.json` is checked before the
    merge starts, so a run that cannot produce one says so in a second rather
    than after the arrays are assembled.
    """
    import numpy as np

    parts = sorted(f for d in dirs for f in Path(d).glob("part-*.npz"))
    if not parts:
        raise SystemExit(f"no part-*.npz under {dirs}")

    metas = {}
    for f in parts:
        meta_path = Path(f).parent / "part-meta.json"
        metas[meta_path] = json.loads(meta_path.read_text())
    check_part_rules(metas.values())

    meta = next(iter(metas.values()))
    if not args.no_catalog:
        try:
            check_catalog_inputs(meta)
        except ValueError as exc:
            raise SystemExit(f"cannot write a catalog for this tile: {exc}") from exc
    h, w = meta["raster"]
    lst, qa, pooled, seen, pooled_seen, n = assemble_parts(parts, meta["raster"])

    out_dir.mkdir(parents=True, exist_ok=True)
    covered = float(seen.mean())
    valid = lst != LST_NODATA_DN
    print(f"merged        {n} shards from {len(parts)} part files")
    print(f"raster        {w} x {h}   coverage {covered * 100:.2f}%")
    if covered < 1.0:
        missing = int((~seen).sum())
        print(f"WARNING       {missing:,} px never written; a slice is missing")
    if valid.any():
        cel = lst[valid].astype("float64") * LST_SCALE + LST_OFFSET
        print(
            f"LST p95       min {cel.min():.1f} C  mean {cel.mean():.1f} C  "
            f"max {cel.max():.1f} C  ({100 * valid.mean():.1f}% valid)"
        )
    np.save(out_dir / "lst_p95_dn.npy", lst)
    np.save(out_dir / "qa_count.npy", qa)
    pooled_coverage = None
    if pooled is not None:
        pooled_coverage = float(pooled_seen.mean())
        np.save(out_dir / "lst_p95_pooled_dn.npy", pooled)
        print(f"pooled        baseline written, coverage {pooled_coverage * 100:.2f}%")
        if pooled_coverage < covered:
            print(
                "WARNING       some slices ran without --emit-pooled; the "
                "baseline covers less ground than the product"
            )

    record: dict = {
        "shards": n,
        "parts": len(parts),
        "coverage": covered,
        "raster": [h, w],
        "meta": meta,
        # The rules every part agreed on, hoisted so a reader of the
        # merged tile does not have to open a part to find them.
        "mask_rule": meta.get("mask_rule"),
        # None here is a claim, not an absence: it says these pixels are the
        # pooled percentile with every scene at its own baseline.
        "correction_rule": meta.get("correction_rule"),
        # None when no slice ran `--emit-pooled`, which is the usual case. The
        # baseline is not a product, so a partial one is reported rather than
        # refused.
        "pooled_coverage": pooled_coverage,
    }
    # The record lands before the catalog, so a merge that took an hour is on
    # disk whatever the catalog writer then does.
    merge_json = out_dir / "merge.json"
    merge_json.write_text(json.dumps(record, indent=2, default=str))
    if not args.no_catalog:
        record["catalog"] = str(_write_catalog(out_dir, lst, qa, meta, args))
        merge_json.write_text(json.dumps(record, indent=2, default=str))
    print(f"artifacts     {out_dir.resolve()}")
    return 0 if covered == 1.0 else 2


def _write_catalog(out_dir: Path, lst, qa, meta: dict, args) -> Path:
    """Write the COGs and the STAC catalog for one merged tile.

    The catalog defaults to a directory beside the arrays, which is what a
    single tile wants. Several tiles pointed at one `--catalog-dir` land in
    one collection, an item each.
    """
    collection_id = args.collection_id or collection_id_for_window(meta)
    root = write_catalog(
        args.catalog_dir or out_dir / "catalog",
        lst,
        qa,
        meta,
        collection_id=collection_id,
        host_name=args.host_name,
        host_url=args.host_url,
        license_id=args.license,
    )
    provenance = catalog_provenance(meta, collection_id=collection_id)
    print(f"catalog       {root.resolve()}")
    print(
        f"encoding      lst_p95 uint16 scale {provenance['lst_scale']} "
        f"offset {provenance['lst_offset']} nodata {provenance['lst_nodata']}; "
        f"qa_count uint8 12 bands, no nodata"
    )
    return root


# The CLI entry point: plan, filter, submit, gather, write, and the merge and
# rehearse modes that short-circuit it. Each branch ends the run.
def main(argv=None) -> int:  # noqa: C901
    args = parse_args(argv)
    if args.merge:
        return merge_parts(args.merge, args.out_dir, args)
    bbox, tile_id = resolve_area(args)
    res = 1.0 / args.pixels_per_degree
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shards, height, width = plan_shards(bbox, args.pixels_per_degree, args.shard)
    # Shard slots and read threads MULTIPLY. 8 workers x 4 threads x 4 read
    # threads is 128 OS threads on 16 cores, which thrashes: measured 15 of 324
    # shards finished in 400 s, with the first wave of 64 all crawling at once.
    #
    # The reduce inside a shard is CPU-bound and wants a whole core. The reads
    # are I/O-bound and want oversubscription. So size slots to cores and let
    # read threads be the only multiplier.
    concurrency = args.workers * args.threads_per_worker

    print(f"bbox          {bbox}")
    print(f"grid          {args.crs} @ 1/{args.pixels_per_degree} deg")
    print(f"raster        {width} x {height} px  ({width * height / 1e6:.0f} Mpx)")
    print(
        f"shards        {len(shards)}  of {args.shard}x{args.shard} px "
        f"({max(s.row for s in shards) + 1} x {max(s.col for s in shards) + 1})"
    )

    if args.dry_run:
        lo, hi_slice = 0, len(shards)
        if args.shard_slice:
            a, _, b = args.shard_slice.partition(":")
            lo = int(a) if a else 0
            hi_slice = int(b) if b else len(shards)
            mine = shards[lo:hi_slice]
            px = sum(sh.ny * sh.nx for sh in mine)
            print(
                f"slice         shards[{lo}:{hi_slice}] -> {len(mine)} shards, "
                f"{px:,} px ({100 * px / (width * height):.1f}% of the tile)"
            )
            ys = [sh.y0 for sh in mine]
            xs = [sh.x0 for sh in mine]
            print(f"              rows {min(ys)}..{max(ys)}  cols {min(xs)}..{max(xs)}")
        edge = [s for s in shards if s.ny != args.shard or s.nx != args.shard]
        print(f"edge shards   {len(edge)} smaller than {args.shard} px")
        cover = sum(s.ny * s.nx for s in shards)
        assert cover == width * height, (
            f"shards cover {cover}, raster is {width * height}"
        )
        print(f"coverage      {cover:,} px == raster, no gaps or overlap")

        client_gib = client_bytes(width, height, emit_pooled=args.emit_pooled)
        print(
            f"\nnaive budget, assuming every shard sees every scene "
            f"(+{client_gib:.1f} GiB of client output):"
        )
        for n in (711, 1765, 3910):
            per = shard_bytes(args.shard, n)
            print(
                f"  at {n:>5} scenes: {per:5.2f} GiB per shard, "
                f"{per * concurrency + client_gib:6.1f} GiB across "
                f"{concurrency} slots"
                f"{_target_verdict(args, per * concurrency + client_gib)}"
            )

        refused = False
        if args.search_in_dry_run:
            if tile_id is None:
                raise SystemExit("--search-in-dry-run needs --tile")
            items, item_bboxes, _ = load_tile_items(args, tile_id)
            counts = [len(items_for_shard(sh, item_bboxes)) for sh in shards]
            # The slice is what one machine runs, and its worst shard is what
            # that machine's memory has to hold. Reporting the tile's worst
            # instead understates a light slice and overstates a heavy one: at
            # 360 px, S30W065 runs 404 scenes deep at shards[0:64] and 820 at
            # shards[987:1051].
            mine_counts = counts[lo:hi_slice] if args.shard_slice else counts
            counts.sort()
            print(f"\nactual scenes per shard (from {len(items)} total):")
            print(
                f"  tile:  min {counts[0]}  p50 {counts[len(counts) // 2]}  "
                f"p95 {counts[int(len(counts) * 0.95)]}  max {counts[-1]}"
            )
            worst = max(mine_counts) if mine_counts else 0
            if args.shard_slice:
                ordered = sorted(mine_counts)
                print(
                    f"  slice: min {ordered[0]}  p50 {ordered[len(ordered) // 2]}  "
                    f"max {worst}   <- what this machine holds"
                )
            per = shard_bytes(args.shard, worst)
            # The deepest `concurrency` shards, not the worst one repeated.
            # Real depths spread, so the two differ by up to 2.77x and only
            # this one is what `worker_memory_guard` refuses on.
            demand = slice_demand(
                args.shard,
                mine_counts,
                concurrency,
                width,
                height,
                emit_pooled=args.emit_pooled,
            )
            print(
                f"  worst shard: {per:.2f} GiB.  deepest {concurrency} shards "
                f"plus client output: {demand:.1f} GiB"
                f"{_target_verdict(args, demand)}"
            )
            print(
                f"  total shard-scene reads: {sum(counts):,} "
                f"vs {len(items) * len(shards):,} unfiltered "
                f"({len(items) * len(shards) / max(sum(counts), 1):.0f}x saved)"
            )
            if args.target_memory_gib:
                try:
                    worker_memory_guard(
                        args.shard,
                        mine_counts,
                        concurrency,
                        width,
                        height,
                        total_bytes=int(args.target_memory_gib * GIB),
                        emit_pooled=args.emit_pooled,
                    )
                except SystemExit as exc:
                    print(f"\nREFUSED on a {args.target_memory_gib:g} GiB machine:")
                    print(f"  {exc}")
                    refused = True

        (args.out_dir / "shards.json").write_text(
            json.dumps(
                [
                    {
                        "row": s.row,
                        "col": s.col,
                        "y0": s.y0,
                        "x0": s.x0,
                        "ny": s.ny,
                        "nx": s.nx,
                        "bbox": s.bbox,
                    }
                    for s in shards
                ],
                indent=2,
            )
        )
        print(f"\nplan written  {args.out_dir / 'shards.json'}")
        # 2, not 1, so a driver can tell "this configuration does not fit" from
        # a plan that failed to build at all.
        return 2 if refused else 0

    # ---------------- execute ----------------
    # Checked here, not before the dry run: planning a slice must never be
    # blocked by a runtime concurrency decision.
    total_threads = concurrency * args.read_threads
    cores = os.cpu_count() or 1
    print(
        f"concurrency   {concurrency} shard slots x {args.read_threads} read "
        f"threads = {total_threads} threads on {cores} cores"
    )
    if total_threads > cores * 6 and not args.force:
        raise SystemExit(
            f"{total_threads} threads on {cores} cores will thrash: slots and "
            f"read threads multiply. Try --workers {cores} "
            f"--threads-per-worker 1 --read-threads 4, or pass --force."
        )

    import numpy as np
    import psutil

    import frisky

    configure_read_env(args.source)
    os.environ.setdefault("FRISKY_TRACING_CAPACITY", "2000000")

    # Before the inventory read and before the first GET. The mask depends on
    # the tile's bbox and on two artifacts, and on nothing this run computes,
    # so a tile it empties can be recorded without staging a single object.
    ged_provenance = check_mask_inputs(args)
    keep = gap = mask_counts = None
    if ged_provenance is not None:
        keep, gap, mask_counts = masks.output_mask(
            bbox,
            args.pixels_per_degree,
            numobs_uri=args.numobs_uri,
            land_geometry_uri=args.land_geometry_uri,
        )
        print(
            f"              {mask_counts['pixels_kept'] / mask_counts['pixels_total']:.1%} "
            f"of the tile is land: {mask_counts['pixels_water']:,} px sea, "
            f"{mask_counts['pixels_emissivity_gap_on_land']:,} px inside the "
            f"ASTER gap region over land"
        )
        print(
            f"              the gap region removes only what reads "
            f"{masks.GAP_HOT_THRESHOLD_C:.0f} C or hotter, after the gather"
        )
        if not mask_counts["pixels_kept"]:
            return no_unmasked_pixels(args, tile_id, bbox, mask_counts, ged_provenance)

    t_search = time.perf_counter()
    if args.rehearse:
        w, so, e, no = bbox
        item_bboxes = [
            (
                w + (e - w) * (i % 7) / 7 - 0.3,
                so + (no - so) * (i // 7 % 7) / 7 - 0.3,
                w + (e - w) * (i % 7) / 7 + 0.6,
                so + (no - so) * (i // 7 % 7) / 7 + 0.6,
            )
            for i in range(args.rehearse)
        ]
        items = [{"id": f"fake-{i}"} for i in range(args.rehearse)]
        run_provenance = {"source": "rehearsal, synthetic items"}
    else:
        if tile_id is None:
            raise SystemExit(
                "a real run needs --tile: the inventory is addressed by tile. "
                "Use --bbox only with --dry-run or --rehearse."
            )
        items, item_bboxes, run_provenance = load_tile_items(args, tile_id)
    t_search = time.perf_counter() - t_search
    print(f"scenes        {len(items)} from the inventory in {t_search:.2f}s")
    if not items:
        print("no scenes matched")
        return 1
    # Both paths hand over plain dicts now. The inventory builds them and the
    # rehearsal fakes them, so nothing here converts a pystac object.
    item_dicts = items

    # L2SR products carry no thermal band, load as fill, and reach neither the
    # percentile nor the monthly counts. Dropping them is output-neutral and
    # buys back a layer on the time axis of every shard they touch, which is
    # what caps shard size at 94% of the worker memory limit. Skipped under
    # --rehearse, where the synthetic items carry no assets at all.
    dropped_no_thermal = 0
    if not args.rehearse and not args.keep_scenes_without_thermal:
        item_dicts, item_bboxes, dropped_no_thermal = (
            staging.drop_scenes_without_thermal(item_dicts, item_bboxes)
        )
        if dropped_no_thermal:
            print(
                f"              {dropped_no_thermal} of {len(items)} carry no "
                f"thermal band; dropped"
            )
        if not item_dicts:
            return no_thermal_coverage(
                args, tile_id, bbox, len(items), dropped_no_thermal, run_provenance
            )

    # After the thermal filter, because that is the list the shards will load
    # and the list `tile_prep` hashed. Before the first GET, like the mask and
    # the disk guard: a prep file for the wrong tile, grid, or scene set is a
    # run that finishes and looks ordinary, so it is refused here rather than
    # discovered in the pixels.
    prep = (
        None
        if args.rehearse
        else load_tile_prep(args, tile_id, item_dicts, run_provenance)
    )
    if prep is not None:
        kept = destripe.keep_mask(
            np.array([prep.offset.get(s, np.nan) for s in prep.offset]),
            np.array([prep.n_valid.get(s, 0) for s in prep.offset]),
            floor=destripe.DESTRIPE_MIN_PREP_SAMPLES,
            max_offset_c=args.max_offset_c,
        )
        print(
            f"prep          {len(prep.paths)} WRS paths, "
            f"{1.0 - kept.mean():.1%} of scenes rejected at "
            f"{args.max_offset_c:g} C"
        )
        print(
            f"              destripe {'off' if args.no_destripe else 'on'}, "
            f"feather {'off' if args.no_feather else 'on'}"
        )
        if not args.no_destripe and not kept.any():
            return no_scene_survives_destriping(args, tile_id, prep, run_provenance)
    elif not args.rehearse:
        print(
            "prep          none: pooled percentile, no scene offsets. The WRS "
            "seam stays in the output"
        )

    # Slice the PLAN, never the filtered list. Shards with no overlapping
    # scenes drop out of `work_idx`, so slicing after filtering shifts every index
    # and machines silently leave gaps. The rehearsal caught exactly that:
    # slice 972:1296 ran 288 shards, and the merge reported 1,440,000 px
    # never written.
    mine = shards
    if args.shard_slice:
        a, _, b = args.shard_slice.partition(":")
        lo = int(a) if a else 0
        hi = int(b) if b else len(shards)
        mine = shards[lo:hi]
        print(
            f"slice         shards[{lo}:{hi}] -> {len(mine)} of {len(shards)} planned"
        )

    # One pass over the plan, read twice. `work_idx` and `barren` partition
    # this slice, and staging needs the union of those indices, so all three
    # come from the same list rather than from three sweeps of the same test.
    per_shard = [items_for_shard(sh, item_bboxes) for sh in mine]
    work_idx = [(sh, idx) for sh, idx in zip(mine, per_shard, strict=True) if idx]
    # Shards with no overlapping scene are still this slice's responsibility.
    # Recording them as all-nodata keeps coverage complete, so the merge can
    # tell "no Landsat here" (ocean, edge) from "a machine died", which it
    # cannot do if they are simply absent.
    barren = [sh for sh, idx in zip(mine, per_shard, strict=True) if not idx]
    if barren:
        print(f"              {len(barren)} shards have no scenes; written as nodata")
    if args.max_shards:
        work_idx = work_idx[: args.max_shards]

    # Before the first GET, like the disk guard, because a configuration that
    # cannot fit should not buy its objects first. --force is the escape, and
    # the rehearsal skips it: rehearse_shard allocates nothing.
    memory_demand = None
    if work_idx and not args.rehearse and not args.force:
        memory_demand = worker_memory_guard(
            args.shard,
            [len(idx) for _, idx in work_idx],
            concurrency,
            width,
            height,
            emit_pooled=args.emit_pooled,
        )
        print(
            f"memory        {memory_demand:.1f} GiB demanded across "
            f"{concurrency} slots, fits"
        )

    counts = [len(idx) for _, idx in work_idx]
    print(
        f"shards        {len(work_idx)} with data, "
        f"scenes/shard min {min(counts)} "
        f"p50 {sorted(counts)[len(counts) // 2]} "
        f"p95 {sorted(counts)[int(len(counts) * 0.95)]} "
        f"max {max(counts)}"
    )
    print(
        f"worst shard   {shard_bytes(args.shard, max(counts)):.2f} GiB, "
        f"{shard_bytes(args.shard, max(counts)) * concurrency:.1f} GiB across "
        f"{concurrency} slots\n"
    )

    # One origin for every phase mark below, so the summary's numbers subtract.
    t0 = time.perf_counter()
    t0_wall = time.time()
    marks: dict[str, float | bool] = {}

    proc = psutil.Process()
    peak = {"rss": 0.0}

    # What the workers actually hold, sampled from outside them. `shard_bytes`
    # predicts this and nothing on a production run had ever measured it, so
    # the model was checked against its own output.
    #
    # It starts before staging as well as before the cluster, so the fetch has
    # a memory and network series too. Before this it was sampled by nothing.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sampler = MemorySampler(args.out_dir / "memory.csv", args.sample_interval)
    sampler.start()

    # Staging runs after --max-shards, so a smoke run over two shards fetches
    # the objects those two shards need and not the whole slice.
    #
    # It runs to completion before the cluster starts, and that ordering is
    # deliberate rather than historical. Overlapping the fetch with the shards
    # it feeds was tried and MEASURED as a loss: the same 1,998 objects and
    # 78.9 GiB that stage in 91.9 s on an idle machine take 358.7 s beside 64
    # busy workers, because staging is bound by processor time and not by the
    # network. See `FINDINGS.md`, "Staging beside compute is slower than
    # staging before it".
    marks["stage_start_s"] = time.perf_counter() - t0
    stage_report = stage_scenes_for(args, item_dicts, work_idx)
    marks["stage_end_s"] = time.perf_counter() - t0

    # After staging, so the table carries the staged hrefs.
    table_path = None
    table_report = None
    if not args.rehearse:
        args.stage_dir.mkdir(parents=True, exist_ok=True)
        table_report = item_table.write(args.stage_dir / ITEM_TABLE_NAME, item_dicts)
        table_path = table_report["path"]
        marks["table_write_s"] = table_report["seconds"]
        print(
            f"item table    {table_report['n_items']:,} scenes, "
            f"{table_report['bytes'] / 1e6:.1f} MB -> {table_path}"
        )

    t_cluster = time.perf_counter()
    cluster = frisky.LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=int(args.memory_limit_gib * GIB),
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    client = cluster.get_client()
    marks["cluster_start_s"] = time.perf_counter() - t_cluster
    dash = cluster.dashboard_address
    dash = dash if str(dash).startswith("http") else f"http://{dash}"
    print(f"dashboard     {dash}")

    lst_out = np.zeros((height, width), dtype="uint16")
    qa_out = np.zeros((12, height, width), dtype="uint8")
    # Built from identical scenes, identical worker code, and identical reads,
    # on the same load rather than a second run. It is taken after the offsets
    # are applied, so the difference between the two rasters is the cross-fade
    # alone. Separating the offsets from the cross-fade needs all four arms,
    # which is what measure_seam.py runs.
    pooled_out = np.zeros((height, width), dtype="uint16") if args.emit_pooled else None

    def assemble(future) -> dict:
        """One shard into the output arrays, then dropped."""
        try:
            res_d = future.result()
        except Exception as exc:  # a dead shard must not kill the tile
            return {"error": repr(exc)}
        y0, x0 = res_d["y0"], res_d["x0"]
        a = res_d["lst_p95"]
        lst_out[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a
        q = res_d["qa_count"]
        qa_out[:, y0 : y0 + q.shape[1], x0 : x0 + q.shape[2]] = q
        if pooled_out is not None and "lst_p95_pooled" in res_d:
            pooled = res_d["lst_p95_pooled"]
            pooled_out[y0 : y0 + pooled.shape[0], x0 : x0 + pooled.shape[1]] = pooled
            del pooled
        stat = {
            k: res_d.get(k)
            for k in (
                "row",
                "col",
                "n_scenes",
                "n_scenes_kept",
                "n_rejected",
                "n_pooled_fallback",
                "load_s",
                "reduce_s",
            )
        }
        del res_d, a, q
        peak["rss"] = max(peak["rss"], proc.memory_info().rss / GIB)
        return stat

    fn = rehearse_task if args.rehearse else shard_task
    try:
        stats, submit_s = drive_shards(
            client,
            fn,
            work_idx,
            table_path,
            args.crs,
            res,
            args.read_threads,
            assemble=assemble,
            marks=marks,
            t0=t0,
            correction_of=(
                None
                if prep is None
                else lambda sh, idx: shard_correction_for(
                    args, prep, sh, [item_dicts[i] for i in idx]
                )
            ),
        )
    except BaseException:
        # Without this the process hangs on a live cluster where before a
        # failure in this stretch simply killed it.
        cluster.close()
        sampler.stop()
        raise

    print(
        f"submitted     {len(work_idx)} shards in {submit_s:.1f}s of client time "
        f"({submit_s / max(len(work_idx), 1) * 1000:.2f} ms each)"
    )

    # `compute_s` spans the whole shard-processing window. It used to run from
    # the last submit to the last result, which was the same thing when every
    # shard was submitted before any returned, and both are recorded so a
    # before-and-after table can compare like with like.
    compute_s = marks["last_result_s"] - marks["first_submit_s"]
    marks |= {
        "t0_wall": t0_wall,
        "search_s": t_search,
        "submit_total_s": submit_s,
        "compute_s": compute_s,
        "compute_from_last_submit_s": marks["last_result_s"] - marks["last_submit_s"],
        "stage_s": marks["stage_end_s"] - marks["stage_start_s"],
        "wall_s": time.perf_counter() - t0,
    }
    print(
        f"phases        stage {marks['stage_s']:.1f}s, "
        f"submit {submit_s:.1f}s, compute {compute_s:.1f}s"
    )

    # Every shard has been gathered, so nothing reads the staged files again.
    # A failed run keeps them, which is what a rerun and a post-mortem both
    # want; the disk guard on the next run says so rather than filling up.
    if stage_report is not None and not args.keep_staged:
        staging.cleanup(args.stage_dir, owned=stage_report.get("owns_stage_dir", True))
    elif table_path is not None and not args.keep_staged:
        # A rehearsal writes the table and stages nothing, so `cleanup` never
        # runs and 7.8 MB would be left behind on every run.
        Path(table_path).unlink(missing_ok=True)

    # The mask goes on before anything is measured or written, so the summary
    # statistics, the part file, and a merge of parts from several machines all
    # describe the same product. `merge_parts` needs no mask of its own, and a
    # `--shard-slice` machine masks only its own slice: every pixel outside the
    # slice is already nodata and the mask only ever removes.
    if keep is not None and mask_counts is not None:
        mask_counts |= masks.apply_output_mask(
            lst_out,
            qa_out,
            keep,
            gap,
            scope="tile" if not args.shard_slice else f"shards[{args.shard_slice}]",
        )
        print(
            f"masked        {mask_counts['valid_removed_by_water']:,} px sea, "
            f"{mask_counts['valid_removed_by_emissivity']:,} px hot inside the "
            f"ASTER gap region"
        )

    # After the mask, not before. The mask allocates while the two full-tile
    # output arrays are live, and a sampler stopped above never sees the peak
    # that `--target-memory-gib` is checked against.
    sampler.stop()
    memory_peak = sampler.peak_between(0.0, time.monotonic())
    workers_gib = memory_peak.get("workers_rss_peak_mb", 0.0) / 1024
    tree_gib = memory_peak.get("tree_rss_peak_mb", 0.0) / 1024

    valid = lst_out != LST_NODATA_DN
    cel = (
        lst_out[valid].astype("float64") * LST_SCALE + LST_OFFSET
        if valid.any()
        else None
    )
    summary = {
        "bbox": bbox,
        "crs": args.crs,
        "pixels_per_degree": args.pixels_per_degree,
        "raster": [height, width],
        "shard_px": args.shard,
        "n_shards": len(work_idx),
        # What the run composited, after the L2SR filter. The inventory total
        # sits beside it, because the two differ by `scenes_dropped_no_thermal`
        # and a reader cannot tell which one a single figure means.
        "n_scenes": len(item_dicts),
        "n_scenes_inventory": len(items),
        "search_s": t_search,
        # Every phase, as seconds from one origin, so a before-and-after table
        # is two summaries subtracted rather than two logs scraped. The submit
        # time in particular used to be printed and then discarded.
        "phases": marks,
        "item_table": table_report,
        "compute_s": compute_s,
        "s_per_shard": compute_s / max(len(work_idx), 1),
        "client_rss_peak_gib": peak["rss"],
        # MEASURED across the worker processes, not predicted. `shard_bytes`
        # models the same quantity, so a run now says whether the model held.
        "workers_rss_peak_gib": workers_gib,
        "tree_rss_peak_gib": tree_gib,
        "memory_demand_gib": memory_demand,
        "valid_fraction": float(valid.mean()),
        # A driver needs one number, not a walk over shard_stats. A run that
        # loses shards still writes a summary and still writes its parts, so
        # without this the artifact of a half-finished tile looks finished.
        "n_shards_errored": sum(1 for s in stats if "error" in s),
        "shard_stats": stats,
        # Which inventory answered this run. A composite is only reproducible
        # if the scene list behind it is named, so this travels with the
        # numbers rather than beside them.
        "inventory": run_provenance,
        "tile": tile_id,
        # What this run actually put on the wire. `cost_report.py --s3-get-requests`
        # prices it directly, so the S3 line stops being derived from a
        # requests-per-read sampled on one laptop against three shards.
        "staging": stage_report,
        "scenes_dropped_no_thermal": dropped_no_thermal,
        # Which pixels the product describes at all, and what it cost to say
        # so. None under --no-output-mask, which writes a raster the mask never
        # touched.
        "mask": (
            None
            if mask_counts is None
            else mask_counts
            | {
                "numobs_uri": str(args.numobs_uri),
                "land_geometry_uri": str(args.land_geometry_uri),
                "aster_ged": ged_provenance,
            }
        ),
    }
    if cel is not None:
        summary |= {
            "min_c": float(cel.min()),
            "mean_c": float(cel.mean()),
            "max_c": float(cel.max()),
        }
        print(
            f"\nLST p95       min {cel.min():.1f} C  mean {cel.mean():.1f} C  "
            f"max {cel.max():.1f} C  ({100 * valid.mean():.1f}% valid)"
        )
    print(
        f"compute       {compute_s:.1f}s for {len(work_idx)} shards "
        f"({compute_s / max(len(work_idx), 1):.2f}s/shard of wall clock, not "
        f"per-shard duration)"
    )
    n_errored = sum(1 for s in stats if "error" in s)
    if n_errored:
        print(
            f"FAILED        {n_errored} of {len(work_idx)} shards errored; this "
            f"tile is incomplete"
        )
    print(f"client RSS    {peak['rss']:.2f} GiB peak")
    if workers_gib:
        # The model against the measurement, on every run. This is the check
        # that was missing when a MEASURED table carried `shard_bytes` output.
        against = ""
        if memory_demand:
            share = workers_gib / memory_demand
            against = f" against {memory_demand:.1f} GiB modelled ({share:.0%})"
        print(f"worker RSS    {workers_gib:.2f} GiB peak{against}")
    qa_mean = {MONTHS[i]: float(qa_out[i].mean()) for i in range(12)}
    summary["qa_count_per_month"] = qa_mean
    print("qa_count      " + "  ".join(f"{m} {v:.1f}" for m, v in qa_mean.items()))

    try:
        spans = frisky.query_spans(
            limit=2_000_000, dashboard_url=dash, request_timeout=60
        )
        summary["n_spans"] = len(spans)
        (args.out_dir / "spans.json").write_text(
            json.dumps(spans[:200000], default=str)
        )
    except Exception as exc:
        summary["span_error"] = repr(exc)
    cluster.close()

    # Write this slice as a part file so other machines' slices can be merged.
    import numpy as _np

    payload = {}
    for sh in mine:  # every planned shard in this slice, barren ones included
        tag = f"{sh.y0}_{sh.x0}"
        payload["lst_" + tag] = lst_out[sh.y0 : sh.y0 + sh.ny, sh.x0 : sh.x0 + sh.nx]
        payload["qa_" + tag] = qa_out[:, sh.y0 : sh.y0 + sh.ny, sh.x0 : sh.x0 + sh.nx]
        if pooled_out is not None:
            payload["pooled_" + tag] = pooled_out[
                sh.y0 : sh.y0 + sh.ny, sh.x0 : sh.x0 + sh.nx
            ]
    if payload:
        # numpy declares savez_compressed(**kwds: ArrayLike) alongside a bool
        # allow_pickle, so a dict of arrays collides with the named parameter.
        _np.savez_compressed(
            args.out_dir / "part-000.npz",
            **payload,  # ty: ignore[invalid-argument-type]
        )
        (args.out_dir / "part-meta.json").write_text(
            json.dumps(
                {
                    "raster": [height, width],
                    "bbox": bbox,
                    "crs": args.crs,
                    "pixels_per_degree": args.pixels_per_degree,
                    "shard_px": args.shard,
                    "n_shards": len(work_idx),
                    # What the mask did to this part. `merge_parts` compares
                    # it across parts, because two machines that masked the
                    # same tile differently produce one raster that no single
                    # rule describes.
                    "mask_rule": mask_rule(args, mask_counts, ged_provenance),
                    # What removed the WRS seam from this part, compared the
                    # same way and for the same reason. One un-de-striped part
                    # beside three de-striped ones carries the seam in one
                    # corner, and nothing in the pixels says which corner.
                    "correction_rule": correction_rule(args, prep),
                    # The merge turns these into the item's datetime interval,
                    # so a catalog states the window its pixels came from.
                    "start": args.start,
                    "end": args.end,
                },
                indent=2,
            )
        )
        print(f"part written  {args.out_dir / 'part-000.npz'} ({len(mine)} shards)")

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    # Beside the summary as well as inside it, because pricing a run reads only
    # this and `cost_report.py` should not have to know the summary's shape.
    if stage_report is not None:
        (args.out_dir / "staging.json").write_text(
            json.dumps(
                stage_report
                | {
                    "scenes_dropped_no_thermal": dropped_no_thermal,
                    # Beside the counts, because pricing a run reads only this
                    # file and whether the fetch overlapped the compute changes
                    # what its seconds mean.
                    "started_s": marks.get("stage_start_s"),
                    "ended_s": marks.get("stage_end_s"),
                    "overlapped": marks.get("overlapped", False),
                },
                indent=2,
            )
        )
    print(f"artifacts     {args.out_dir.resolve()}")
    # 3 for a tile that lost shards, 0 for one that did not. It used to return
    # 0 either way, so a run that gathered 1 shard of 64 reported success and
    # wrote a part file and a summary to match.
    #
    # This is the signal a fleet driver needs, and the panic that prompted
    # looking is not it. MEASURED: SIGABRT on four of eight workers mid-run,
    # which is what a non-unwinding panic does to a worker, and all 200 shards
    # still completed with no errors and exit 0. frisky reschedules the work.
    # What loses a tile quietly is a shard that raises.
    return 3 if n_errored else 0


if __name__ == "__main__":
    sys.exit(main())

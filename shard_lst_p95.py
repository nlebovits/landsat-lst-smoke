# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2", "dask", "odc-stac", "pystac-client",
#   "planetary-computer", "xarray", "numpy", "geopandas",
#   "psutil", "rich", "boto3", "pyarrow>=16",
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

import staging
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

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
GIB = 1024.0**3


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
#: uint16 at height by width, and `qa_out` is uint8 at 12 by height by width.
#: A full tile at 18,000 px square is 4.2 GiB of it.
CLIENT_BYTES_PER_OUTPUT_PIXEL = 2 + 12


def client_bytes(width: int, height: int) -> float:
    """The two full-tile arrays the client holds while it gathers, in GiB."""
    return width * height * CLIENT_BYTES_PER_OUTPUT_PIXEL / GIB


def worker_memory_guard(
    shard_px: int,
    depths,
    workers: int,
    width: int,
    height: int,
    *,
    total_bytes: int | None = None,
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
    if not ordered:
        return client_bytes(width, height)
    arrays = sum(shard_bytes(shard_px, n) for n in ordered)
    client = client_bytes(width, height)
    demand = arrays + client
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
    shard: Shard, item_dicts, crs: str, resolution: float, read_threads: int = 4
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
        "load_s": 0.0,
        "reduce_s": 0.0,
    }


def process_shard(
    shard: Shard, item_dicts, crs: str, resolution: float, read_threads: int = 4
) -> dict:
    """Load, mask, reduce and encode one shard. Returns small arrays only.

    Deliberately eager: no dask inside. The whole point is that this fits in
    memory, so a lazy graph would only reintroduce the rechunk we are avoiding.
    """
    import numpy as np
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
    t_load = time.perf_counter() - t0

    # One definition of a usable observation, shared with the array-graph path
    # in profile_lst_p95. It drops source fill, QA_PIXEL bits 1 to 5, and any
    # decoded value outside [-50, 80] C. The range check is what removes the
    # reprojected scene edges, where interpolation against the DN 0 fill leaves
    # small nonzero values that decode near -124 C and that an exact fill
    # comparison cannot see. All of it happens before the percentile, because a
    # value that reaches nanpercentile has already moved the answer.
    lst, valid = masked_celsius(data["lwir11"].values, data["qa_pixel"].values)

    t1 = time.perf_counter()
    with np.errstate(all="ignore"):
        p95 = np.nanpercentile(lst, 95, axis=0)
    t_reduce = time.perf_counter() - t1

    months = data["time"].dt.month.values
    qa_count = np.zeros((12, p95.shape[0], p95.shape[1]), dtype="uint8")
    for m in range(1, 13):
        sel = months == m
        if sel.any():
            qa_count[m - 1] = np.minimum(valid[sel].sum(axis=0), 255).astype("uint8")

    dn_out = encode_celsius(p95)
    return {
        "row": shard.row,
        "col": shard.col,
        "y0": shard.y0,
        "x0": shard.x0,
        "lst_p95": dn_out,
        "qa_count": qa_count,
        "n_scenes": int(lst.shape[0]),
        "load_s": t_load,
        "reduce_s": t_reduce,
    }


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
    shards need rather than the whole slice.

    Returns:
        The staging report, or None when the run reads from S3. The report is
        this run's S3 line, counted rather than derived from a sampled
        requests-per-read.
    """
    if args.rehearse:
        print("stage         skipped: the rehearsal reads no objects")
        return None
    if args.no_stage:
        print(
            "stage         skipped: --no-stage. Every shard reads from S3, and "
            "about 155 shards touch each scene"
        )
        return None
    report = staging.stage_scenes(
        item_dicts,
        sorted({i for _, idx in work_idx for i in idx}),
        args.stage_dir,
    )
    print(
        f"stage         {report['objects']:,} objects, "
        f"{report['bytes'] / GIB:.1f} GiB in {report['seconds']:.1f}s "
        f"-> {report['stage_dir']}"
    )
    print(
        f"              {report['get_requests']:,} billable GETs, "
        f"{report['retries']} retries"
    )
    return report


def _target_verdict(args, per_shard_gib, slots, client_gib) -> str:
    """`  fits` or `  OVER by N GiB` against `--target-memory-gib`.

    A dry run plans for a machine that has not been launched, so the figure it
    checks against has to be named rather than read from the host. Without the
    flag there is nothing to compare and this adds nothing to the line.
    """
    if not args.target_memory_gib:
        return ""
    demand = per_shard_gib * slots + client_gib
    over = demand - args.target_memory_gib
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
        help="local directory the scene objects are fetched into before the "
        "cluster starts. Wants throughput as well as capacity: the compute "
        "phase already reads about 358 MB/s and staging writes on top of it",
    )
    p.add_argument(
        "--no-stage",
        action="store_true",
        help="read every shard straight from S3, as the pipeline did before "
        "staging existed. About 155 shards touch each scene and each open "
        "costs 4.77 requests, so this is the expensive path and it is kept "
        "for measuring against",
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
        default=0.05,
        help="seconds between memory samples. The sampler runs in its own "
        "process and writes memory.csv beside the summary, so a run records "
        "the worker RSS that shard_bytes only predicts",
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
    args = p.parse_args(argv)
    # Refuse an unreadable source here rather than after the shard plan is
    # printed. `configure_read_env` runs late enough that a run could get a
    # full budget report before learning its source cannot read a scene.
    if args.source != SUPPORTED_READ_SOURCE:
        configure_read_env(args.source)
    return args


def merge_parts(dirs, out_dir: Path) -> int:
    """Assemble one tile from the parts written by --shard-slice runs."""
    import numpy as np

    parts = sorted(f for d in dirs for f in Path(d).glob("part-*.npz"))
    if not parts:
        raise SystemExit(f"no part-*.npz under {dirs}")

    meta = json.loads((Path(parts[0]).parent / "part-meta.json").read_text())
    h, w = meta["raster"]
    lst = np.zeros((h, w), dtype="uint16")
    qa = np.zeros((12, h, w), dtype="uint8")

    seen = np.zeros((h, w), dtype=bool)
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
                n += 1

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
    (out_dir / "merge.json").write_text(
        json.dumps(
            {
                "shards": n,
                "parts": len(parts),
                "coverage": covered,
                "raster": [h, w],
                "meta": meta,
            },
            indent=2,
            default=str,
        )
    )
    print(f"artifacts     {out_dir.resolve()}")
    return 0 if covered == 1.0 else 2


# The CLI entry point: plan, filter, submit, gather, write, and the merge and
# rehearse modes that short-circuit it. Each branch ends the run.
def main(argv=None) -> int:  # noqa: C901
    args = parse_args(argv)
    if args.merge:
        return merge_parts(args.merge, args.out_dir)
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

        client_gib = client_bytes(width, height)
        print(
            f"\nnaive budget, assuming every shard sees every scene "
            f"(+{client_gib:.1f} GiB of client output):"
        )
        for n in (711, 1765, 3910):
            per = shard_bytes(args.shard, n)
            print(
                f"  at {n:>5} scenes: {per:5.2f} GiB per shard, "
                f"{per * concurrency + client_gib:6.1f} GiB across "
                f"{concurrency} slots{_target_verdict(args, per, concurrency, client_gib)}"
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
            print(
                f"  worst shard: {per:.2f} GiB, "
                f"{per * concurrency + client_gib:.1f} GiB across "
                f"{concurrency} slots"
                f"{_target_verdict(args, per, concurrency, client_gib)}"
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

    # Slice the PLAN, never the filtered list. Shards with no overlapping
    # scenes drop out of `work`, so slicing after filtering shifts every index
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

    # One pass over the plan, read twice. `work` and `barren` partition this
    # slice, and staging needs the union of the indices in `work`, so all three
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
        )
        print(
            f"memory        {memory_demand:.1f} GiB demanded across "
            f"{concurrency} slots, fits"
        )

    # Staging runs after --max-shards, so a smoke run over two shards fetches
    # the objects those two shards need and not the whole slice.
    stage_report = stage_scenes_for(args, item_dicts, work_idx)

    work = [(sh, [item_dicts[i] for i in idx]) for sh, idx in work_idx]
    counts = [len(d) for _, d in work]
    print(
        f"shards        {len(work)} with data, "
        f"scenes/shard min {min(counts)} p50 {sorted(counts)[len(counts) // 2]} "
        f"max {max(counts)}"
    )
    print(
        f"worst shard   {shard_bytes(args.shard, max(counts)):.2f} GiB, "
        f"{shard_bytes(args.shard, max(counts)) * concurrency:.1f} GiB across "
        f"{concurrency} slots\n"
    )

    proc = psutil.Process()
    peak = {"rss": 0.0}

    # What the workers actually hold, sampled from outside them. `shard_bytes`
    # predicts this and nothing on a production run had ever measured it, so
    # the model was checked against its own output. The sampler starts before
    # the cluster, because worker RSS peaks while they are all allocating.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sampler = MemorySampler(args.out_dir / "memory.csv", args.sample_interval)
    sampler.start()

    cluster = frisky.LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=int(args.memory_limit_gib * GIB),
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    client = cluster.get_client()
    dash = cluster.dashboard_address
    dash = dash if str(dash).startswith("http") else f"http://{dash}"
    print(f"dashboard     {dash}")

    lst_out = np.zeros((height, width), dtype="uint16")
    qa_out = np.zeros((12, height, width), dtype="uint8")

    t0 = time.perf_counter()
    fn = rehearse_shard if args.rehearse else process_shard
    futures = [
        client.submit(fn, sh, d, args.crs, res, args.read_threads) for sh, d in work
    ]
    print(f"submitted     {len(futures)} shards in {time.perf_counter() - t0:.1f}s")

    done = 0
    stats = []
    t_compute = time.perf_counter()
    # Collect one at a time and drop each result after assembling it.
    # client.gather(futures) on all 324 at once held ~1 GB of results and
    # aborted the process with a Rust panic across the PyO3 boundary at 90%
    # ("panic in a function that cannot unwind"), losing the whole run.
    for fut in frisky.as_completed(list(futures), raise_errors=False):
        try:
            res_d = fut.result()
        except Exception as exc:  # a dead shard must not kill the tile
            stats.append({"error": repr(exc)})
            done += 1
            continue
        y0, x0 = res_d["y0"], res_d["x0"]
        a = res_d["lst_p95"]
        lst_out[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a
        q = res_d["qa_count"]
        qa_out[:, y0 : y0 + q.shape[1], x0 : x0 + q.shape[2]] = q
        stats.append(
            {k: res_d[k] for k in ("row", "col", "n_scenes", "load_s", "reduce_s")}
        )
        del res_d, a, q
        done += 1
        peak["rss"] = max(peak["rss"], proc.memory_info().rss / GIB)
        if done % 25 == 0 or done == len(futures):
            el = time.perf_counter() - t_compute
            print(
                f"  {done:4d}/{len(futures)}  {el:6.1f}s  "
                f"{el / done:5.2f}s/shard  client RSS {peak['rss']:.1f} GiB"
            )
    compute_s = time.perf_counter() - t_compute
    sampler.stop()
    memory_peak = sampler.peak_between(0.0, time.monotonic())
    workers_gib = memory_peak.get("workers_rss_peak_mb", 0.0) / 1024
    tree_gib = memory_peak.get("tree_rss_peak_mb", 0.0) / 1024

    # Every shard has been gathered, so nothing reads the staged files again.
    # A failed run keeps them, which is what a rerun and a post-mortem both
    # want; the disk guard on the next run says so rather than filling up.
    if stage_report is not None and not args.keep_staged:
        staging.cleanup(args.stage_dir, owned=stage_report.get("owns_stage_dir", True))

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
        "n_shards": len(work),
        # What the run composited, after the L2SR filter. The inventory total
        # sits beside it, because the two differ by `scenes_dropped_no_thermal`
        # and a reader cannot tell which one a single figure means.
        "n_scenes": len(item_dicts),
        "n_scenes_inventory": len(items),
        "search_s": t_search,
        "compute_s": compute_s,
        "s_per_shard": compute_s / max(len(work), 1),
        "client_rss_peak_gib": peak["rss"],
        # MEASURED across the worker processes, not predicted. `shard_bytes`
        # models the same quantity, so a run now says whether the model held.
        "workers_rss_peak_gib": workers_gib,
        "tree_rss_peak_gib": tree_gib,
        "memory_demand_gib": memory_demand,
        "valid_fraction": float(valid.mean()),
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
        f"compute       {compute_s:.1f}s for {len(work)} shards "
        f"({compute_s / max(len(work), 1):.2f}s each)"
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
                    "n_shards": len(work),
                },
                indent=2,
            )
        )
        print(
            f"part written  {args.out_dir / 'part-000.npz'} ({len(payload) // 2} shards)"
        )

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    # Beside the summary as well as inside it, because pricing a run reads only
    # this and `cost_report.py` should not have to know the summary's shape.
    if stage_report is not None:
        (args.out_dir / "staging.json").write_text(
            json.dumps(
                stage_report | {"scenes_dropped_no_thermal": dropped_no_thermal},
                indent=2,
            )
        )
    print(f"artifacts     {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

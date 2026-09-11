# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2", "dask", "odc-stac", "xarray", "numpy",
#   "pystac", "psutil", "boto3", "pyarrow>=16",
#   "rasterio", "shapely", "geopandas", "pyogrio",
# ]
# ///
"""One coarse pass over a tile, for everything a shard cannot work out alone.

A shard holds its own scenes and nothing else, so two quantities are out of its
reach. Both are properties of the whole tile.

**The per-scene offset.** De-striping compares a scene against a per-pixel
calendar-month climatology and subtracts its bulk deviation. If each shard fit
its own offset for a scene, the same scene would be shifted differently on
either side of a shard boundary, and the run would trade a WRS seam for a seam
on the shard grid.

**The swath, and the cross-fade weights on it.** Same argument one level up: a
weight field derived per shard lets two shards disagree about the same ground.

So this runs once per tile and writes one artifact the slices read. The cost is
one extra traversal, taken at `1/prep_factor` of the output resolution through
the source COGs' internal overviews.

    uv run tile_prep.py --tile S30W065 --prep-factor 4 --out-dir ./tile-prep

**One pass, not two.** `nlebovits/landsat-lst` needs two: a spatial median does
not decompose across blocks, so it computes the climatology in one phase and
re-reads every scene in a second to take each scene's median anomaly. Binning
the anomaly instead does decompose, and at a bin one output DN wide the median
read back off it is exact to the quantisation the product already carries. The
histogram is what buys the second pass back. See `destripe.accumulate_anomaly`.

**Where the swath comes from.** Not from the item footprints. The inventory ring
is the USGS product bounding rectangle, which FINDINGS.md measures at about 46%
more area than the imaged parallelogram Earth Search publishes. The seam sits at
the imaged edge. So each `(path, row)` quad's swath is the ground at least half
its scenes actually reached, counted off the valid observations themselves.

**The margin.** The grid runs past the tile by `--margin-deg`, using scenes this
tile already holds: a WRS scene is about 1.7 degrees across, so a scene touching
the tile reaches well beyond it. Without the margin every swath would be clipped
at the tile border, that border would become part of the swath boundary, and the
cross-fade would ramp toward the edge of the tile.

**What a tile boundary still does to this.** Both quantities are measured over
this grid and no wider, so a scene that two tiles share gets a different offset
in each: the median anomaly is taken over different ground. A quad's swath moves
for the same reason, because the inventory assigns only some of that quad's
scenes to each tile and the half-the-scenes threshold then has a different
denominator. Two merged tiles can therefore disagree along their shared border.
`nlebovits/landsat-lst` has the same limit. Nothing here fixes it, and
`FINDINGS.md` records it as open.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import destripe
import staging
from land_tiles import tile_bounds
from lst_qa import masked_celsius
from shard_lst_p95 import (
    DEFAULT_INVENTORY_URI,
    DEFAULT_STAGE_DIR,
    READ_SOURCES,
    configure_read_env,
    items_for_shard,
    plan_shards,
)
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

#: Version of the artifact this module writes. A slice refuses a prep file it
#: does not recognise rather than compositing against a layout it is guessing
#: at.
PREP_SCHEMA_VERSION = 1

#: Default resolution divisor for the pass. The offset is one scalar per scene,
#: and a constant has no resolution, so it does not need the output grid to
#: estimate. `nlebovits/landsat-lst` validated factor 2 against native offsets
#: at a median of 0.002 C and rejected factor 4 on its own grid at a maximum of
#: 0.546 C against a pre-registered 0.5 C gate. That gate was set on a different
#: grid and a different loader, so this repository owes its own sweep before the
#: default is defended rather than merely chosen.
DEFAULT_PREP_FACTOR = 4

#: How far the grid runs past the tile, in degrees.
DEFAULT_MARGIN_DEG = 1.0

#: Prep pixels on a block edge. The resident array is
#: `block**2 * scenes_in_block * 4` bytes, so this is the memory knob.
DEFAULT_BLOCK = 512

GIB = 1024.0**3


# --------------------------------------------------------------------------
# Instrumentation. frisky carries the stage timings, so `frisky observe
# prefixes` can attribute this pass without a second telemetry path.
# --------------------------------------------------------------------------


@contextmanager
def span(name: str, **keys):
    """Time one stage and emit a frisky span for it. Never fails the run."""
    t0 = time.perf_counter()
    start_ns = _now_ns()
    try:
        yield
    finally:
        _record(f"prep.{name}", start_ns, _now_ns(), keys)
        print(f"{name:<14}{time.perf_counter() - t0:8.1f}s")


def _now_ns() -> int:
    try:
        import frisky

        return int(frisky.now_ns())
    except Exception:
        return time.time_ns()


def _record(name: str, t0_ns: int, t1_ns: int, keys: dict) -> None:
    try:
        import frisky

        frisky.record_span(
            name, t0_ns, t1_ns, keys=[f"{k}={v}" for k, v in keys.items()] or None
        )
    except Exception:
        pass


# --------------------------------------------------------------------------
# The grid. Three resolutions, each an exact divisor of the output grid, so no
# block edge is ragged and no swath cell straddles two prep blocks.
# --------------------------------------------------------------------------


def prep_bbox(tile_bbox, margin_deg: float):
    """The tile, run out by a margin and snapped to whole degrees."""
    west, south, east, north = tile_bbox
    return (
        max(west - margin_deg, -180.0),
        max(south - margin_deg, -90.0),
        min(east + margin_deg, 180.0),
        min(north + margin_deg, 90.0),
    )


def check_grid(
    pixels_per_degree: int, prep_factor: int, swath_factor: int, block: int
) -> int:
    """The prep-to-swath ratio, refusing any combination that leaves a remainder.

    A ragged ratio would put a swath cell across two prep blocks, and the two
    blocks would then disagree about whether a path reached it.

    The block size is the third way to get one. `plan_shards` starts every
    block at an exact multiple of `block`, and `prep_block` returns its coarse
    origin as `block.y0 // ratio`. A block that is not a whole number of swath
    cells truncates that division, so neighbouring blocks write overlapping
    coarse rows and `accumulate` counts one cell twice while the far edge
    drifts. MEASURED at `--block 510 --swath-factor 16`: ratio 4, and block 1
    lands on coarse cell 127 where its true origin is 127.5.
    """
    problems = []
    for name, factor in (("prep", prep_factor), ("swath", swath_factor)):
        if pixels_per_degree % factor:
            problems.append(
                f"{name} factor {factor} does not divide {pixels_per_degree}"
            )
    prep_ppd = pixels_per_degree // prep_factor
    swath_ppd = pixels_per_degree // swath_factor
    if swath_ppd and prep_ppd % swath_ppd:
        problems.append(
            f"prep grid {prep_ppd} does not divide by swath grid {swath_ppd}"
        )
    ratio = prep_ppd // swath_ppd if swath_ppd else 0
    if ratio and block % ratio:
        problems.append(
            f"block {block} is not a whole number of swath cells: it spans "
            f"{block / ratio} cells at a ratio of {ratio}"
        )
    if problems:
        raise SystemExit("; ".join(problems))
    return ratio


def check_swath_grid(height: int, width: int, ratio: int) -> None:
    """Refuse a prep grid the swath grid cannot cover.

    `swath_shape` is the prep grid floored by the ratio, and `accumulate`
    clamps a block's coverage to it. A remainder therefore drops the last row
    or column of swath cells, and a path whose only cell is in there loses its
    swath without a word. MEASURED at `--margin-deg 0.125` on a 7 degree tile
    at prep factor 4: a height of 6,525 prep rows against a ratio of 2.

    The margin is what usually causes it, because the tile edges are whole
    degrees and the margin need not be.

    Raises:
        SystemExit: naming the remainder and the flag that moves it.
    """
    problems = [
        f"the prep grid is {size:,} {name} against a swath ratio of {ratio}, "
        f"leaving {size % ratio} that the swath grid would drop"
        for size, name in ((height, "rows"), (width, "columns"))
        if size % ratio
    ]
    if problems:
        raise SystemExit(
            "; ".join(problems) + ". Choose a --margin-deg whose pixels divide "
            "by the ratio, or a --swath-factor that divides the grid."
        )


def swath_transform(bbox, pixels_per_degree: int, swath_factor: int):
    from masks import transform_for

    return transform_for(bbox, pixels_per_degree // swath_factor)


def check_every_path_has_a_swath(item_dicts, paths) -> None:
    """Refuse a tile where some path reached no swath cell at all.

    `feathered_percentile` reduces one subset per path in `paths` and blends
    them. A scene whose path is missing from that list enters no subset, so it
    loads, costs a read, and contributes nothing. The pixels it observed still
    reach `qa_count`, and where another path covers them the composite is a
    value fitted without them. That is a wrong number rather than a missing
    one, and nothing in the raster marks it.

    A path drops out when every one of its quads stayed under
    `SWATH_QUAD_SHARE` on every cell. That is the swath definition failing to
    describe the path, not a fact about the ground, so the tile stops here
    rather than at the shards.

    Raises:
        SystemExit: naming the paths and the one flag that composites anyway.
    """
    absent = sorted({destripe.path_of(d) for d in item_dicts} - set(paths))
    if not absent:
        return
    raise SystemExit(
        f"{len(absent)} WRS paths reached no swath cell on this tile: "
        f"{', '.join(absent)}. Their scenes would load and contribute nothing "
        f"to any shard, against a swath share of {destripe.SWATH_QUAD_SHARE}. "
        f"Composite with --no-feather."
    )


def memory_model(block: int, scenes_per_block, n_scenes, n_quads, swath_shape, slots):
    """Every array this pass keeps resident, named, in GiB.

    The shard path refuses a configuration that will not fit, and this one has
    to do the same rather than print a figure and start reading. Five terms:

    - a worker's own block stack, `block**2 * scenes_in_block * 4` bytes, and
      one per slot
    - the histogram a worker builds for the scenes it saw, and which it ships
      whole to the driver. This is the term that surprises: at 26,000 bins it
      is 104 KB a scene, so a block seeing 2,000 scenes returns 208 MB
    - the results in flight, one per slot, because `as_completed` hands them
      over one at a time and each is freed after it is folded in
    - the driver's histogram accumulator over every scene of the tile
    - the driver's per-quad coverage counts on the swath grid
    """
    worst = max(scenes_per_block) if scenes_per_block else 0
    stack = block**2 * worst * 4 / GIB
    partial = worst * destripe.N_ANOMALY_BINS * 4 / GIB
    accumulator = n_scenes * destripe.N_ANOMALY_BINS * 4 / GIB
    coverage = n_quads * swath_shape[0] * swath_shape[1] * 2 / GIB
    return {
        "worker_block_stack_gib": stack,
        "worker_histogram_gib": partial,
        "workers_gib": (stack + partial) * slots,
        "results_in_flight_gib": partial * slots,
        "driver_histogram_gib": accumulator,
        "driver_coverage_gib": coverage,
        "total_gib": (stack + partial) * slots
        + partial * slots
        + accumulator
        + coverage,
    }


def memory_guard(model: dict, target_gib: float | None, force: bool) -> None:
    """Refuse a prep run that will not fit, before it buys its first object."""
    if not target_gib or model["total_gib"] <= target_gib:
        return
    message = (
        f"this prep needs {model['total_gib']:.1f} GiB and the machine has "
        f"{target_gib:.1f} GiB. Largest terms: workers "
        f"{model['workers_gib']:.1f}, results in flight "
        f"{model['results_in_flight_gib']:.1f}, driver histogram "
        f"{model['driver_histogram_gib']:.1f}, driver coverage "
        f"{model['driver_coverage_gib']:.1f}. Try a smaller --block, fewer "
        f"--workers, or a larger --prep-factor, or pass --force."
    )
    if not force:
        raise SystemExit(message)
    print(f"WARNING       {message}")


# --------------------------------------------------------------------------
# The unit of work. Everything here happens inside one worker.
# --------------------------------------------------------------------------


def prep_block(block, item_dicts, global_idx, crs, resolution, ratio, read_threads=4):
    """Read one block coarse, and return only what the tile reduction needs.

    Returns small arrays: the anomaly histogram and valid count for each scene
    the block saw, and each quad's per-cell coverage count on the swath grid.
    The block's own pixels are discarded here and never leave the worker.
    """
    import numpy as np
    import pystac
    from odc.geo import CRS
    from odc.stac import stac_load

    ydim, xdim = ("y", "x") if CRS(crs).projected else ("latitude", "longitude")
    items = [pystac.Item.from_dict(d) for d in item_dicts]
    index_of = {
        destripe.scene_id_of(d): g for d, g in zip(item_dicts, global_idx, strict=True)
    }

    t0 = time.perf_counter()
    data = stac_load(
        items,
        bands=("lwir11", "qa_pixel"),
        crs=crs,
        resolution=resolution,
        bbox=block.bbox,
        groupby="landsat:scene_id",
        chunks={"time": 1, ydim: -1, xdim: -1},
    ).compute(scheduler="threads", num_workers=read_threads)
    t_load = time.perf_counter() - t0

    celsius, valid = masked_celsius(data["lwir11"].values, data["qa_pixel"].values)
    times = data["time"].values
    months = data["time"].dt.month.values
    gidx = destripe.align_to_time(
        items,
        times,
        value_of=lambda item: index_of[destripe.scene_id_of(item)],
        what="scene index",
        dtype="int64",
    )
    quads = destripe.align_to_time(
        items, times, value_of=destripe.quad_of, what="WRS quad", dtype=object
    )

    t1 = time.perf_counter()
    planes, ref = destripe.month_climatology(celsius, months)
    n_scenes = celsius.shape[0]
    hist = np.zeros((n_scenes, destripe.N_ANOMALY_BINS), dtype="uint32")
    n_valid = np.zeros(n_scenes, dtype="int64")
    destripe.accumulate_anomaly(hist, n_valid, celsius, months, planes, ref)
    coverage = _quad_coverage(valid, quads, ratio)
    t_reduce = time.perf_counter() - t1

    return {
        "y0": block.y0 // ratio,
        "x0": block.x0 // ratio,
        "global_idx": gidx,
        "hist": hist,
        "n_valid": n_valid,
        "coverage": coverage,
        "n_scenes": n_scenes,
        "load_s": t_load,
        "reduce_s": t_reduce,
    }


def _quad_coverage(valid, quads, ratio: int):
    """How many scenes of each quad reached each swath cell of this block.

    A scene reaches a swath cell when any of the prep pixels under it carries a
    valid observation. Reaching, not covering: the question the swath asks is
    where the path contributed at all.
    """
    import numpy as np

    out: dict[tuple[str, str], np.ndarray] = {}
    for s in range(valid.shape[0]):
        reach = destripe._block_any(valid[s], ratio)
        quad = quads[s]
        if quad in out:
            out[quad] += reach.astype("uint16")
        else:
            out[quad] = reach.astype("uint16")
    return out


# --------------------------------------------------------------------------
# The tile reduction, in the driver.
# --------------------------------------------------------------------------


def accumulate(result, hist, n_valid, quad_count, swath_shape):
    """Fold one block's return value into the tile's accumulators, in place."""
    import numpy as np

    gidx = result["global_idx"]
    hist[gidx] += result["hist"]
    n_valid[gidx] += result["n_valid"]
    y0, x0 = result["y0"], result["x0"]
    for quad, reach in result["coverage"].items():
        if quad not in quad_count:
            quad_count[quad] = np.zeros(swath_shape, dtype="uint16")
        y1 = min(y0 + reach.shape[0], swath_shape[0])
        x1 = min(x0 + reach.shape[1], swath_shape[1])
        quad_count[quad][y0:y1, x0:x1] += reach[: y1 - y0, : x1 - x0]


def scene_table(item_dicts):
    """Scene ids, quads, and quad totals, in inventory order."""
    from collections import Counter

    quads = [destripe.quad_of(d) for d in item_dicts]
    return (
        [destripe.scene_id_of(d) for d in item_dicts],
        quads,
        dict(Counter(quads)),
    )


def write_artifact(out_dir: Path, payload: dict, meta: dict) -> None:
    """The two files a slice reads. Numbers in the `.npz`, provenance in JSON."""
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "tile-prep.npz", **payload)
    (out_dir / "tile-prep.json").write_text(json.dumps(meta, indent=2, default=str))


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="One coarse pass per tile: scene offsets and swath weights."
    )
    p.add_argument("--tile", required=True, help="tile id, e.g. S30W065")
    p.add_argument("--inventory-uri", type=Path, default=DEFAULT_INVENTORY_URI)
    p.add_argument("--out-dir", type=Path, default=Path("./tile-prep"))
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument("--prep-factor", type=int, default=DEFAULT_PREP_FACTOR)
    p.add_argument("--swath-factor", type=int, default=destripe.SWATH_FACTOR)
    p.add_argument("--weight-factor", type=int, default=destripe.WEIGHT_FACTOR)
    p.add_argument("--margin-deg", type=float, default=DEFAULT_MARGIN_DEG)
    p.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--source", choices=sorted(READ_SOURCES), default="earth-search")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--read-threads", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=None, help="cap, for smoke runs")
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    p.add_argument("--no-stage", action="store_true")
    p.add_argument(
        "--target-memory-gib",
        type=float,
        default=None,
        help="RAM of the machine this pass runs on. Without it nothing is "
        "checked, because the host planning a fleet run is not the host doing "
        "it. With it, a configuration that will not fit stops here",
    )
    p.add_argument(
        "--force", action="store_true", help="run past the memory guard anyway"
    )
    p.add_argument(
        "--max-offset-c",
        type=float,
        default=destripe.DESTRIPE_MAX_OFFSET_C,
        help="reported only. The cap is applied when a slice reads this file, "
        "so sweeping it costs a read rather than this whole pass.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def load_items(args):
    """Every scene of the tile, with the manifest checked before any read."""
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
        args.inventory_uri, args.tile, bounds=tile_bounds(args.tile)
    )
    items, boxes, dropped = staging.drop_scenes_without_thermal(items, boxes)
    return items, boxes, dropped, provenance(manifest)


def run_blocks(args, work, item_dicts, resolution, ratio, on_result) -> list:
    """Submit every block to a frisky cluster and fold results as they land."""
    import frisky

    cluster = frisky.LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
    )
    client = cluster.get_client()
    stats = []
    try:
        futures = [
            client.submit(
                prep_block,
                block,
                [item_dicts[i] for i in idx],
                list(idx),
                args.crs,
                resolution,
                ratio,
                args.read_threads,
            )
            for block, idx in work
        ]
        # One at a time, like `shard_lst_p95`: gathering the whole list at once
        # panicked the Rust scheduler there, and each result here is large.
        for future in frisky.as_completed(list(futures), raise_errors=False):
            result = future.result()
            on_result(result)
            stats.append(
                {
                    "n_scenes": result["n_scenes"],
                    "load_s": round(result["load_s"], 2),
                    "reduce_s": round(result["reduce_s"], 2),
                }
            )
            del result
    finally:
        client.close()
        cluster.close()
    return stats


def main(argv=None) -> int:  # noqa: C901
    import numpy as np

    args = parse_args(argv)
    ratio = check_grid(
        args.pixels_per_degree, args.prep_factor, args.swath_factor, args.block
    )
    bbox = prep_bbox(tile_bounds(args.tile), args.margin_deg)
    prep_ppd = args.pixels_per_degree // args.prep_factor
    swath_ppd = args.pixels_per_degree // args.swath_factor
    resolution = 1.0 / prep_ppd

    blocks, height, width = plan_shards(bbox, prep_ppd, args.block)
    check_swath_grid(height, width, ratio)
    swath_shape = (height // ratio, width // ratio)
    print(f"tile          {args.tile}  margin {args.margin_deg} deg -> {bbox}")
    print(
        f"prep grid     1/{prep_ppd} deg  {width} x {height} px, {len(blocks)} blocks"
    )
    print(f"swath grid    1/{swath_ppd} deg  {swath_shape[1]} x {swath_shape[0]} px")

    items, boxes, dropped, run_provenance = load_items(args)
    print(f"scenes        {len(items)} from the inventory, {dropped} with no thermal")
    if not items:
        print("no scenes matched")
        return 1

    scene_ids, quads, quad_scenes = scene_table(items)
    window = {
        "start": args.start,
        "end": args.end,
        "platforms": args.platforms,
        "cloud_cover_lt": args.cloud_cover_lt,
    }
    per_block = [items_for_shard(b, boxes) for b in blocks]
    work = [(b, idx) for b, idx in zip(blocks, per_block, strict=True) if idx]
    # Blocks with no scene were never going to contribute coverage, so they are
    # not what `--max-blocks` takes away. This is the denominator the artifact
    # records, and the one the truncation is measured against.
    with_scenes = len(work)
    if args.max_blocks:
        work = work[: args.max_blocks]
    partial = len(work) < with_scenes
    if partial:
        print(
            f"WARNING       --max-blocks runs {len(work)} of {with_scenes} "
            f"blocks with scenes. Every quad's swath is counted over part of "
            f"the tile and divided by all of its scenes, so the swaths come "
            f"out small. A slice refuses this artifact."
        )
    slots = args.workers * args.threads_per_worker
    model = memory_model(
        args.block,
        [len(idx) for _, idx in work],
        len(items),
        len(quad_scenes),
        swath_shape,
        slots,
    )
    print(
        f"blocks        {len(work)} with scenes, worst holds "
        f"{model['worker_block_stack_gib']:.2f} GiB of pixels and returns "
        f"{model['worker_histogram_gib']:.2f} GiB of histogram"
    )
    print(
        f"memory        {model['total_gib']:.1f} GiB across {slots} slots: "
        f"workers {model['workers_gib']:.1f}, in flight "
        f"{model['results_in_flight_gib']:.1f}, driver "
        f"{model['driver_histogram_gib'] + model['driver_coverage_gib']:.1f} "
        f"({destripe.N_ANOMALY_BINS} bins of {destripe.ANOMALY_BIN_C} C)"
    )
    memory_guard(model, args.target_memory_gib, args.force)
    if args.dry_run:
        return 0

    configure_read_env(args.source)
    if not args.no_stage:
        with span("stage"):
            report = staging.stage_scenes(
                items, sorted({i for _, idx in work for i in idx}), args.stage_dir
            )
            print(
                f"              {report['objects']:,} objects, "
                f"{report['bytes'] / GIB:.1f} GiB, "
                f"{report['get_requests']:,} billable GETs"
            )

    hist = np.zeros((len(items), destripe.N_ANOMALY_BINS), dtype="uint32")
    n_valid = np.zeros(len(items), dtype="int64")
    quad_count: dict[tuple[str, str], np.ndarray] = {}

    with span("blocks", n=len(work)):
        stats = run_blocks(
            args,
            work,
            items,
            resolution,
            ratio,
            lambda r: accumulate(r, hist, n_valid, quad_count, swath_shape),
        )

    with span("offsets", scenes=len(items)):
        offset = destripe.offsets_from_histograms(hist)
        keep = destripe.keep_mask(
            offset,
            n_valid,
            floor=destripe.DESTRIPE_MIN_PREP_SAMPLES,
            max_offset_c=args.max_offset_c,
        )
        diagnostics = destripe.offset_diagnostics(offset, keep)
        print(f"              {diagnostics}")

    with span("geometry", quads=len(quad_count)):
        masks = destripe.swath_masks(quad_count, quad_scenes)
        paths, weight, inside = destripe.path_weights(
            masks,
            swath_transform(bbox, args.pixels_per_degree, args.swath_factor),
            factor=args.weight_factor,
        )
        covered = inside.any(axis=0) if inside.size else np.zeros(swath_shape, bool)
        print(
            f"              {len(paths)} paths, "
            f"{covered.mean():.1%} of the prep grid covered, "
            f"{(inside.sum(axis=0) >= 2).mean():.1%} reached by two or more"
        )
        check_every_path_has_a_swath(items, paths)

    write_artifact(
        args.out_dir,
        {
            "scene_ids": np.array(scene_ids),
            "offset": offset,
            "n_valid": n_valid,
            "paths": np.array(paths),
            "weight": weight,
            "inside": inside,
        },
        {
            "schema_version": PREP_SCHEMA_VERSION,
            # The scene set these offsets were fitted over. A slice recomputes
            # it from its own item list and refuses a prep file that does not
            # match, because two prep files built from different scene lists
            # under identical settings are otherwise indistinguishable.
            "scene_digest": destripe.scene_digest(scene_ids, window),
            "tile": args.tile,
            "bbox": list(bbox),
            "pixels_per_degree": args.pixels_per_degree,
            "prep_factor": args.prep_factor,
            "swath_factor": args.swath_factor,
            "weight_factor": args.weight_factor,
            "margin_deg": args.margin_deg,
            "anomaly_bin_c": destripe.ANOMALY_BIN_C,
            "min_offset_samples": destripe.DESTRIPE_MIN_PREP_SAMPLES,
            "max_offset_c_reported": args.max_offset_c,
            "swath_quad_share": destripe.SWATH_QUAD_SHARE,
            "paths": list(paths),
            "n_scenes": len(items),
            "scenes_without_thermal": dropped,
            "offsets": diagnostics,
            # `planned` counted every block, barren ones included, so an
            # ordinary run showed `run` below it and the pair said nothing
            # about truncation. `with_scenes` is the number that was going to
            # run, and `partial` is the flag a slice refuses on.
            "blocks": {
                "planned": len(blocks),
                "with_scenes": with_scenes,
                "run": len(stats),
                "max_blocks": args.max_blocks,
                "partial": partial,
            },
            "block_stats": stats[:200],
            "inventory": run_provenance,
            "window": window,
        },
    )
    print(f"written       {args.out_dir / 'tile-prep.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

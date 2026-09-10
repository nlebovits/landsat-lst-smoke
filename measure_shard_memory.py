# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "odc-stac", "pystac", "xarray", "numpy", "rioxarray", "rasterio",
#   "dask", "pyarrow>=16",
# ]
# ///
"""What one shard costs in memory, and what shard size costs in compute.

Two figures decide how a fleet instance is configured, and both were argued
rather than measured until a `c6id.16xlarge` lost ten worker processes to
coredumps three minutes into a run.

`shard_bytes` returned `px * px * scenes * 4`, the decoded float32 stack alone.
`process_shard` holds five arrays at once, and the fifth is easy to miss
because numpy allocates it rather than the pipeline:

    dn      uint16   2      the raw thermal stack
    qa      uint16   2      the QA stack
    celsius float32  4      the decoded stack
    valid   bool     1      the mask, kept for the monthly counts
    copy    float32  4      nanpercentile partitions a copy, not in place

Thirteen bytes per pixel-scene, not four. The old model reported 0.39 GiB for a
shard this measures at 1.42, so a dry-run budget of 25 GiB across 64 slots
described a real demand of 97 on a 128 GiB box.

Thirteen is the accounting figure and it sits above the measurement, which is
the direction to be wrong in. Least squares over the committed sweeps puts the
slope at 12.67 bytes at 512 px and 13.20 at 360, and the model bounds all
twelve points.

`frisky` had the right number the whole time. A full-tile run reported
`memory 95.93 GiB / 102.40 GiB (94%)` across 64 workers, which is 1.50 GiB
each. It was read as a tuning result rather than as a refutation of the model.
So `--mode memory` samples RSS directly, and the answer has to agree with both
frisky and the model.

    uv run measure_shard_memory.py --mode memory \\
        --out artifacts/shard_memory.json

Shard size is the other figure. It used to be a request-cost lever worth 2.8x,
and staging removed that: one GET per object regardless of shard size. What
remains is a memory decision, because memory falls with the square of the edge
while compute does not. `--mode timing` prices the compute side.

It needs real scenes. A synthetic raster generated at the shard's own size
makes GDAL read a whole file per scene, and reports a 4.7x cliff at 512 px that
does not exist on the 7800 x 7900 tiled COGs the pipeline actually reads. Stage
a handful of real scenes first and point `--stage-dir` at them.

    uv run shard_lst_p95.py --tile S30W065 --max-shards 1 \\
        --stage-dir /tmp/stage --keep-staged --out-dir /tmp/out
    uv run measure_shard_memory.py --mode timing --stage-dir /tmp/stage \\
        --out artifacts/shard_timing.json

Cost: `--mode memory` touches no network and no object store. `--mode timing`
reads only what is already staged.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shard_lst_p95 import (  # noqa: E402
    GIB,
    SHARD_BYTES_PER_PIXEL_SCENE,
    SHARD_FIXED_GIB,
    Shard,
    configure_read_env,
    process_shard,
    shard_bytes,
)
from tile_inventory import ASSET_TEMPLATES  # noqa: E402

#: The grid the production tiles use.
PIXELS_PER_DEGREE = 3600

#: A corner inside `S30W065`, so a synthetic scene and a real one cover the
#: same ground and the two modes stay comparable.
WEST, NORTH = -64.0, -33.98

#: DN 40000 decodes to 12.57 C, inside the trusted range, so every pixel counts
#: as valid and the arrays reach their full size.
SYNTHETIC_THERMAL_DN = 40000

#: QA_PIXEL bit 6. None of the excluded bits 1 to 5 are set.
SYNTHETIC_QA_CLEAR = 0b1000000

#: Shard edges to sweep. 360 divides 3600 exactly, so its shards land on whole
#: degrees with no partial edge; 512 is what the full-tile run used.
DEFAULT_EDGES = (256, 360, 448, 512)


def sample_peak_rss(stop: threading.Event, peak: dict) -> None:
    """Poll RSS while a shard runs.

    `resource.getrusage` reports a high-water mark for the process, but only
    after the fact and not per call. Polling /proc gives the shape of the curve
    and catches a transient that ru_maxrss would attribute to an earlier shard.
    """
    while not stop.is_set():
        try:
            with open("/proc/self/statm") as fh:
                peak["rss"] = max(peak["rss"], int(fh.read().split()[1]) * 4096)
        except OSError:
            return
        time.sleep(0.02)


def item_from_files(scene_id: str, bands: dict[str, str]) -> dict:
    """A STAC item describing files already on disk.

    The projection fields come from the thermal raster itself rather than from
    the inventory, so this reads a synthetic tile and a staged Landsat scene
    the same way.

    The assets reuse `tile_inventory.ASSET_TEMPLATES`. Their `raster:bands`
    carries the dtype and nodata, and without it `odc.stac` loads both bands as
    float32 and the QA mask fails on a bitwise operation against a float. That
    would also measure the wrong arrays, which is the point of this script.

    Every href is resolved to an absolute path. `odc.stac.parse_item` raises
    `Can not determine absolute path for asset` on a relative one, and both
    `--work-dir` and `--stage-dir` default to or accept relative paths, so the
    regenerate command in the tests failed before this line existed.
    `staging.stage_scenes` writes absolute hrefs for the same reason.
    """
    import rasterio

    bands = {band: str(Path(href).resolve()) for band, href in bands.items()}

    with rasterio.open(bands["lwir11"]) as ds:
        epsg = ds.crs.to_epsg()
        height, width = ds.height, ds.width
        transform = list(ds.transform)[:6]
        west, south, east, north = ds.bounds

    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": scene_id,
        "collection": "landsat-c2-l2",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    (west, south),
                    (east, south),
                    (east, north),
                    (west, north),
                    (west, south),
                ]
            ],
        },
        "bbox": [west, south, east, north],
        "properties": {
            "datetime": "2023-06-15T12:00:00Z",
            "landsat:scene_id": scene_id,
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


def synthetic_items(n_scenes: int, edge: int, work: Path) -> list[dict]:
    """`n_scenes` items over one shard, written as real GeoTIFFs on disk.

    EPSG:4326 at the output grid's own resolution, so nothing reprojects and
    the measurement is of the mask and the reduction rather than of a warp.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    res = 1.0 / PIXELS_PER_DEGREE
    work.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n_scenes):
        paths = {}
        for band, value in (
            ("lwir11", SYNTHETIC_THERMAL_DN),
            ("qa_pixel", SYNTHETIC_QA_CLEAR),
        ):
            path = work / f"{band}_{i}.TIF"
            if not path.exists():
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=edge,
                    width=edge,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:4326",
                    transform=from_origin(WEST, NORTH, res, res),
                ) as ds:
                    ds.write(np.full((edge, edge), value, "uint16"), 1)
            paths[band] = str(path)
        items.append(item_from_files(f"SYNTH{i:05d}", paths))
    return items


def shard_at(edge: int) -> Shard:
    """One shard anchored at the same corner, whatever its edge."""
    res = 1.0 / PIXELS_PER_DEGREE
    return Shard(
        0, 0, 0, 0, edge, edge, (WEST, NORTH - edge * res, WEST + edge * res, NORTH)
    )


def run_once(shard: Shard, items, read_threads: int) -> tuple[dict, float, float]:
    """One `process_shard`, with peak RSS sampled alongside it."""
    peak = {"rss": 0}
    stop = threading.Event()
    watcher = threading.Thread(target=sample_peak_rss, args=(stop, peak), daemon=True)
    watcher.start()
    t0 = time.perf_counter()
    out = process_shard(
        shard, items, "EPSG:4326", 1.0 / PIXELS_PER_DEGREE, read_threads
    )
    elapsed = time.perf_counter() - t0
    stop.set()
    watcher.join(timeout=1)
    return out, elapsed, peak["rss"] / GIB


def _one_point(shard_px, n_scenes, work, read_threads, out, stage_dir=None):
    """One scene count, in a process that has run nothing else.

    This has to be a fresh interpreter. glibc does not return freed arenas to
    the kernel promptly, so a second shard measured in the same process reports
    the high-water mark of the first. Two sweeps over the same six points
    disagreed by 18% at 700 scenes until each ran on its own.

    `stage_dir` selects the source. Staged scenes are the production read
    pattern: real tiled COGs much larger than the shard, read through a window.
    The synthetic fixture writes one untiled raster at the shard's own edge,
    which couples shard size to source layout. That coupling is why
    `--mode timing` refuses the fixture, and the same objection applies here.
    """
    items = (
        staged_items(stage_dir, n_scenes)
        if stage_dir is not None
        else synthetic_items(n_scenes, shard_px, work)
    )
    _, elapsed, rss = run_once(shard_at(shard_px), items, read_threads)
    out.put((rss, elapsed, len(items)))


def fit_slope(rows) -> dict:
    """Bytes per pixel-scene and the fixed term, by least squares.

    The mean of consecutive differences was the earlier estimator and it hides
    the scatter it averages: at 360 px those differences run 9.2 to 16.4 bytes
    while their mean lands within 1.3% of the 512 px mean. Least squares uses
    every point once and the reported spread says how much to trust it.
    """
    xs = [r["scenes"] * r["shard_px"] ** 2 for r in rows]
    ys = [r["peak_rss_gib"] * GIB for r in rows]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denom = n * sxx - sx * sx
    if denom == 0:
        # Every point at the same depth. A sweep against a stage directory
        # holding fewer scenes than its smallest requested count collapses to
        # one x, and a line through one x has no slope.
        return {
            "slope_bytes_per_pixel_scene": None,
            "intercept_gib": None,
            "step_slope_min": None,
            "step_slope_max": None,
        }
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    pairs = sorted(zip(xs, ys, strict=True))
    steps = [
        (b[1] - a[1]) / (b[0] - a[0])
        for a, b in zip(pairs, pairs[1:], strict=False)
        if b[0] != a[0]
    ]
    return {
        "slope_bytes_per_pixel_scene": round(slope, 2),
        "intercept_gib": round(intercept / GIB, 3),
        "step_slope_min": round(min(steps), 2) if steps else None,
        "step_slope_max": round(max(steps), 2) if steps else None,
    }


def measure_memory(args) -> dict:
    """Peak RSS against `shard_bytes`, across scene counts.

    The model has to track the measurement as scenes rise, not just match at
    one point, because the fleet reads shards from 199 to 971 scenes deep.

    `--stage-dir` swaps the synthetic fixture for real staged COGs. Scene
    counts then cap at what is staged, so a sweep asking for more points than
    the directory holds collapses to the counts it can serve.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    stage_dir = Path(args.stage_dir) if args.stage_dir else None
    source = "staged" if stage_dir else "synthetic"
    work = Path(args.work_dir) / f"synthetic-{args.shard}"
    if stage_dir is not None:
        available = len(staged_items(stage_dir, 1 << 30))
        scenes = sorted({min(n, available) for n in args.scenes})
        if len(scenes) < 2:
            # Two depths at least, or there is no slope to fit. Caught here
            # rather than in the fit, because the fix is to stage more scenes
            # or ask for shallower ones and neither is obvious from a
            # ZeroDivisionError.
            msg = (
                f"{stage_dir} holds {available} scenes and --scenes "
                f"{' '.join(str(n) for n in args.scenes)} caps to {scenes}. "
                f"A slope needs at least two depths. Stage more scenes, or "
                f"sweep below {available}, for example --scenes "
                f"{available // 4} {available // 2} {available}."
            )
            raise SystemExit(msg)
        print(f"  {available} staged scenes, sweeping {scenes}", flush=True)
    else:
        scenes = list(args.scenes)

    rows = []
    for n in scenes:
        if stage_dir is None:
            synthetic_items(n, args.shard, work)  # write the rasters here
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_one_point,
            args=(args.shard, n, work, args.read_threads, queue, stage_dir),
        )
        proc.start()
        rss, elapsed, n_used = queue.get()
        proc.join()
        predicted = shard_bytes(args.shard, n_used)
        rows.append(
            {
                "scenes": n_used,
                "shard_px": args.shard,
                "peak_rss_gib": round(rss, 3),
                "predicted_gib": round(predicted, 3),
                "ratio": round(rss / predicted, 3),
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"  {n_used:>5} scenes   measured {rss:5.2f} GiB   "
            f"model {predicted:5.2f} GiB"
            f"   ratio {rss / predicted:5.2f}   {elapsed:6.1f}s",
            flush=True,
        )
    worst = max(abs(r["ratio"] - 1.0) for r in rows)
    return {
        "mode": "memory",
        "source": source,
        "shard_px": args.shard,
        "rows": rows,
        "worst_ratio_error": round(worst, 3),
        "under_predicted": [r["scenes"] for r in rows if r["ratio"] > 1.0],
        # Written from the imported constants rather than typed here. An
        # earlier sweep recorded 13 beside a `predicted_gib` column computed at
        # 18, and no test read that column.
        "model_bytes_per_pixel_scene": SHARD_BYTES_PER_PIXEL_SCENE,
        "model_fixed_gib": SHARD_FIXED_GIB,
        **fit_slope(rows),
        "ru_maxrss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB, 3
        ),
    }


def staged_items(stage_dir: Path, n_scenes: int) -> list[dict]:
    """Items pointing at scenes a staged run left on disk.

    Real Landsat COGs, so the read pattern is the production one. A synthetic
    raster generated at the shard's own edge is not: it makes the shard size
    and the source layout move together, which reports a cliff that is an
    artifact of the fixture.
    """
    scenes = sorted(p for p in stage_dir.iterdir() if p.is_dir())
    if not scenes:
        msg = (
            f"{stage_dir} holds no staged scenes. Run shard_lst_p95.py with "
            f"--stage-dir {stage_dir} --keep-staged first."
        )
        raise SystemExit(msg)
    items = []
    for scene in scenes[:n_scenes]:
        bands = {p.stem.split(".")[0]: str(p) for p in scene.glob("*.TIF")}
        if "lwir11" in bands and "qa_pixel" in bands:
            items.append(item_from_files(scene.name, bands))
    if not items:
        msg = f"{stage_dir} holds scene directories but no lwir11/qa_pixel pair"
        raise SystemExit(msg)
    return items


def measure_timing(args) -> dict:
    """Seconds per megapixel-scene, across shard edges, on real scenes.

    Larger shards amortise the fixed per-shard cost over more pixels, so this
    rises as the edge falls. How far it rises is what prices the memory
    headroom a smaller shard buys.
    """
    items = staged_items(Path(args.stage_dir), args.max_scenes)
    print(f"  {len(items)} staged scenes", flush=True)
    rows = []
    for edge in args.edges:
        times = [
            run_once(shard_at(edge), items, args.read_threads)[1]
            for _ in range(args.repeats)
        ]
        elapsed = sum(times) / len(times)
        mpx_scene = edge * edge * len(items) / 1e6
        rows.append(
            {
                "shard_px": edge,
                "shards_per_tile": round((18000 / edge) ** 2),
                "seconds_per_shard": round(elapsed, 3),
                "seconds_per_mpx_scene": round(elapsed / mpx_scene, 4),
            }
        )
        print(
            f"  {edge:>4} px   {elapsed:6.2f}s/shard   "
            f"{elapsed / mpx_scene:7.4f} s/Mpx-scene",
            flush=True,
        )
    base = next(r for r in rows if r["shard_px"] == max(args.edges))
    for row in rows:
        row["penalty_vs_largest"] = round(
            row["seconds_per_mpx_scene"] / base["seconds_per_mpx_scene"], 3
        )
    return {
        "mode": "timing",
        "n_scenes": len(items),
        "repeats": args.repeats,
        "rows": rows,
        "note": (
            "An upper bound on the penalty. Fixed per-shard cost is amortised "
            "over the scenes in the shard, and a fleet shard carries 199 to "
            "971 rather than the handful measured here."
        ),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=("memory", "timing"), default="memory")
    p.add_argument("--shard", type=int, default=512, help="edge, for --mode memory")
    p.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        # The six counts the committed sweeps hold, so the regenerate command
        # in the tests reproduces the artifact rather than a shorter sweep.
        default=(100, 200, 300, 404, 500, 700),
        help="scene counts to sweep, for --mode memory",
    )
    p.add_argument(
        "--edges",
        type=int,
        nargs="+",
        default=DEFAULT_EDGES,
        help="shard edges to sweep, for --mode timing",
    )
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="staged scenes to read. Required for --mode timing. Optional for "
        "--mode memory, where it replaces the synthetic fixture with real "
        "tiled COGs and caps the sweep at the scenes on disk",
    )
    p.add_argument("--max-scenes", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--read-threads", type=int, default=4)
    p.add_argument("--work-dir", type=Path, default=Path("./shard-memory-work"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.mode == "timing" and args.stage_dir is None:
        p.error("--mode timing needs --stage-dir pointing at staged scenes")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_read_env("earth-search")
    if not hasattr(os, "sysconf"):
        print("this measurement reads /proc and needs Linux")
        return 1

    print(f"mode          {args.mode}")
    report = measure_memory(args) if args.mode == "memory" else measure_timing(args)

    if args.mode == "memory":
        if report["slope_bytes_per_pixel_scene"] is None:
            print("\nslope         not fitted: every point ran at one depth")
        else:
            print(
                f"\nslope         {report['slope_bytes_per_pixel_scene']} "
                f"B/px-scene by least squares, steps "
                f"{report['step_slope_min']} to {report['step_slope_max']}"
            )
            print(f"intercept     {report['intercept_gib']} GiB")
        under = report["under_predicted"]
        print(
            f"model         {report['model_bytes_per_pixel_scene']} B/px-scene "
            f"+ {report['model_fixed_gib']} GiB, worst ratio error "
            f"{report['worst_ratio_error']:.1%}"
        )
        print(
            f"              under-predicts at {under}"
            if under
            else "              bounds every point"
        )
        print("frisky reported 1.50 GiB per worker on the full-tile run.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"written       {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

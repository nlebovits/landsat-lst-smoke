# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2", "dask", "odc-stac", "pystac-client",
#   "planetary-computer", "xarray", "numpy", "geopandas",
#   "psutil", "rich",
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

    512 x 512 px x 1765 scenes x 4 bytes = 1.85 GB per shard

That fits in one worker with room to spare, and it stays constant as the area
grows. A quarter tile is 324 shards; a full tile is 1,296. Frisky schedules
250,000-400,000 tasks/s, so the task count is free.

    uv run shard_lst_p95.py --bbox=-62.5,-35.0,-60.0,-32.5 \
        --pixels-per-degree 3600 --shard 512 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

STAC_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
STAC_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
SOURCES = {
    "earth-search": STAC_EARTH_SEARCH,
    "planetary-computer": STAC_PLANETARY_COMPUTER,
}
COLLECTION = "landsat-c2-l2"

LWIR_SCALE = 0.00341802
LWIR_OFFSET_C = 149.0 - 273.15
LWIR_FILL_DN = 0
QA_CLOUD_BITS = 0b11000

LST_SCALE, LST_OFFSET = 0.01, -50.0
LST_NODATA_DN, LST_MIN_DN, LST_MAX_DN = 0, 1, 65535
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GIB = 1024.0**3


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


def plan_shards(bbox, pixels_per_degree: int, shard: int) -> tuple[list[Shard], int, int]:
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
        i for i, (iw, isouth, ie, inorth) in enumerate(item_bboxes)
        if iw < e and ie > w and isouth < n and inorth > s
    ]


def shard_bytes(shard_px: int, n_scenes: int) -> float:
    """Peak float32 working set for one shard's complete time stack, in GiB."""
    return shard_px * shard_px * n_scenes * 4 / GIB


# --------------------------------------------------------------------------
# The unit of work. Everything here happens inside one worker.
# --------------------------------------------------------------------------


def rehearse_shard(shard: Shard, item_dicts, crs: str, resolution: float,
                   read_threads: int = 4) -> dict:
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
    dn = np.rint((lst - LST_OFFSET) / LST_SCALE)
    dn[(dn < LST_MIN_DN) | (dn > LST_MAX_DN)] = LST_NODATA_DN
    qa = np.full((12, shard.ny, shard.nx), min(n // 12, 255), dtype="uint8")
    time.sleep(0.01)
    return {
        "row": shard.row, "col": shard.col, "y0": shard.y0, "x0": shard.x0,
        "lst_p95": dn.astype("uint16"), "qa_count": qa,
        "n_scenes": n, "load_s": 0.0, "reduce_s": 0.0,
    }


def process_shard(shard: Shard, item_dicts, crs: str, resolution: float,
                  read_threads: int = 4) -> dict:
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

    dn = data["lwir11"].values
    qa = data["qa_pixel"].values
    valid = ((qa & QA_CLOUD_BITS) == 0) & (dn != LWIR_FILL_DN)
    del qa

    lst = dn.astype("float32") * np.float32(LWIR_SCALE) + np.float32(LWIR_OFFSET_C)
    del dn
    lst[~valid] = np.nan

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

    dn_out = np.rint((p95 - LST_OFFSET) / LST_SCALE)
    bad = ~np.isfinite(dn_out) | (dn_out < LST_MIN_DN) | (dn_out > LST_MAX_DN)
    dn_out[bad] = LST_NODATA_DN
    return {
        "row": shard.row, "col": shard.col,
        "y0": shard.y0, "x0": shard.x0,
        "lst_p95": dn_out.astype("uint16"),
        "qa_count": qa_count,
        "n_scenes": int(lst.shape[0]),
        "load_s": t_load, "reduce_s": t_reduce,
    }


def search_items(args, bbox):
    """STAC search once, in the client. Returns items and their bboxes."""
    import pystac_client

    query = {"eo:cloud_cover": {"lt": args.cloud_cover_lt}}
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()]
    if plats and args.platforms.strip().lower() != "all":
        query["platform"] = {"in": plats}
    cat = pystac_client.Client.open(SOURCES[args.source])
    items = list(cat.search(
        collections=[COLLECTION], bbox=bbox,
        datetime=f"{args.start}/{args.end}", query=query).items())
    return items, [tuple(i.bbox) for i in items]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sharded p95 LST composite: one shard, one task, no shuffle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bbox", required=True, help="west,south,east,north EPSG:4326 (use --bbox=...)")
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument("--shard", type=int, default=512, help="shard edge in pixels")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--cloud-cover-lt", type=int, default=100)
    p.add_argument("--platforms", default="landsat-8,landsat-9")
    p.add_argument("--source", choices=sorted(SOURCES), default="earth-search")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=4)
    p.add_argument("--memory-limit-gib", type=float, default=13.0)
    p.add_argument("--read-threads", type=int, default=4,
                   help="threads used to read scenes inside one shard")
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
        "--rehearse", type=int, default=0, metavar="N",
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
    p.add_argument("--force", action="store_true",
                   help="run even if slots x read-threads oversubscribes the cores")
    p.add_argument("--dry-run", action="store_true",
                   help="plan shards and print the budget; no cluster, no reads")
    p.add_argument("--search-in-dry-run", action="store_true",
                   help="also hit STAC, to report real scenes per shard")
    return p.parse_args(argv)


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
                lst[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
                qa[:, y0:y0 + q.shape[1], x0:x0 + q.shape[2]] = q
                seen[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = True
                n += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    covered = float(seen.mean())
    valid = lst != LST_NODATA_DN
    print(f"merged        {n} shards from {len(parts)} part files")
    print(f"raster        {w} x {h}   coverage {covered*100:.2f}%")
    if covered < 1.0:
        missing = int((~seen).sum())
        print(f"WARNING       {missing:,} px never written; a slice is missing")
    if valid.any():
        cel = lst[valid].astype("float64") * LST_SCALE + LST_OFFSET
        print(f"LST p95       min {cel.min():.1f} C  mean {cel.mean():.1f} C  "
              f"max {cel.max():.1f} C  ({100*valid.mean():.1f}% valid)")
    np.save(out_dir / "lst_p95_dn.npy", lst)
    np.save(out_dir / "qa_count.npy", qa)
    (out_dir / "merge.json").write_text(json.dumps(
        {"shards": n, "parts": len(parts), "coverage": covered,
         "raster": [h, w], "meta": meta}, indent=2, default=str))
    print(f"artifacts     {out_dir.resolve()}")
    return 0 if covered == 1.0 else 2


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.merge:
        return merge_parts(args.merge, args.out_dir)
    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox needs west,south,east,north")
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
    print(f"raster        {width} x {height} px  ({width*height/1e6:.0f} Mpx)")
    print(f"shards        {len(shards)}  of {args.shard}x{args.shard} px "
          f"({max(s.row for s in shards)+1} x {max(s.col for s in shards)+1})")

    if args.dry_run:
        if args.shard_slice:
            a, _, b = args.shard_slice.partition(":")
            lo = int(a) if a else 0
            hi = int(b) if b else len(shards)
            mine = shards[lo:hi]
            px = sum(sh.ny * sh.nx for sh in mine)
            print(f"slice         shards[{lo}:{hi}] -> {len(mine)} shards, "
                  f"{px:,} px ({100*px/(width*height):.1f}% of the tile)")
            ys = [sh.y0 for sh in mine]; xs = [sh.x0 for sh in mine]
            print(f"              rows {min(ys)}..{max(ys)}  cols {min(xs)}..{max(xs)}")
        edge = [s for s in shards if s.ny != args.shard or s.nx != args.shard]
        print(f"edge shards   {len(edge)} smaller than {args.shard} px")
        cover = sum(s.ny * s.nx for s in shards)
        assert cover == width * height, f"shards cover {cover}, raster is {width*height}"
        print(f"coverage      {cover:,} px == raster, no gaps or overlap")

        print("\nnaive budget, assuming every shard sees every scene:")
        for n in (711, 1765, 3910):
            per = shard_bytes(args.shard, n)
            print(f"  at {n:>5} scenes: {per:5.2f} GiB per shard, "
                  f"{per*concurrency:6.1f} GiB across {concurrency} slots")

        if args.search_in_dry_run:
            items, item_bboxes = search_items(args, bbox)
            counts = [len(items_for_shard(sh, item_bboxes)) for sh in shards]
            counts.sort()
            hi = counts[-1]
            print(f"\nactual scenes per shard (from {len(items)} total):")
            print(f"  min {counts[0]}  p50 {counts[len(counts)//2]}  "
                  f"p95 {counts[int(len(counts)*0.95)]}  max {hi}")
            per = shard_bytes(args.shard, hi)
            print(f"  worst shard: {per:.2f} GiB, "
                  f"{per*concurrency:.1f} GiB across {concurrency} slots")
            print(f"  total shard-scene reads: {sum(counts):,} "
                  f"vs {len(items)*len(shards):,} unfiltered "
                  f"({len(items)*len(shards)/max(sum(counts),1):.0f}x saved)")

        (args.out_dir / "shards.json").write_text(json.dumps(
            [{"row": s.row, "col": s.col, "y0": s.y0, "x0": s.x0,
              "ny": s.ny, "nx": s.nx, "bbox": s.bbox} for s in shards], indent=2))
        print(f"\nplan written  {args.out_dir/'shards.json'}")
        return 0

    # ---------------- execute ----------------
    # Checked here, not before the dry run: planning a slice must never be
    # blocked by a runtime concurrency decision.
    total_threads = concurrency * args.read_threads
    cores = os.cpu_count() or 1
    print(f"concurrency   {concurrency} shard slots x {args.read_threads} read "
          f"threads = {total_threads} threads on {cores} cores")
    if total_threads > cores * 6 and not args.force:
        raise SystemExit(
            f"{total_threads} threads on {cores} cores will thrash: slots and "
            f"read threads multiply. Try --workers {cores} "
            f"--threads-per-worker 1 --read-threads 4, or pass --force."
        )

    import numpy as np
    import psutil

    import frisky

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    if args.source == "earth-search":
        os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
        os.environ.setdefault("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-west-2"))
    os.environ.setdefault("FRISKY_TRACING_CAPACITY", "2000000")

    t_search = time.perf_counter()
    if args.rehearse:
        w, so, e, no = bbox
        item_bboxes = [
            (w + (e - w) * (i % 7) / 7 - 0.3, so + (no - so) * (i // 7 % 7) / 7 - 0.3,
             w + (e - w) * (i % 7) / 7 + 0.6, so + (no - so) * (i // 7 % 7) / 7 + 0.6)
            for i in range(args.rehearse)
        ]
        items = [{"id": f"fake-{i}"} for i in range(args.rehearse)]
    else:
        items, item_bboxes = search_items(args, bbox)
    t_search = time.perf_counter() - t_search
    print(f"scenes        {len(items)} in {t_search:.1f}s")
    if not items:
        print("no scenes matched")
        return 1
    if args.source == "planetary-computer" and not args.rehearse:
        import planetary_computer

        for it in items:
            planetary_computer.sign_inplace(it)
    item_dicts = items if args.rehearse else [it.to_dict() for it in items]

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
        print(f"slice         shards[{lo}:{hi}] -> {len(mine)} of "
              f"{len(shards)} planned")

    work = []
    for sh in mine:
        idx = items_for_shard(sh, item_bboxes)
        if idx:
            work.append((sh, [item_dicts[i] for i in idx]))
    # Shards with no overlapping scene are still this slice's responsibility.
    # Recording them as all-nodata keeps coverage complete, so the merge can
    # tell "no Landsat here" (ocean, edge) from "a machine died", which it
    # cannot do if they are simply absent.
    barren = [sh for sh in mine if not items_for_shard(sh, item_bboxes)]
    if barren:
        print(f"              {len(barren)} shards have no scenes; "
              f"written as nodata")
    if args.max_shards:
        work = work[: args.max_shards]
    counts = [len(d) for _, d in work]
    print(f"shards        {len(work)} with data, "
          f"scenes/shard min {min(counts)} p50 {sorted(counts)[len(counts)//2]} "
          f"max {max(counts)}")
    print(f"worst shard   {shard_bytes(args.shard, max(counts)):.2f} GiB, "
          f"{shard_bytes(args.shard, max(counts))*concurrency:.1f} GiB across "
          f"{concurrency} slots\n")

    proc = psutil.Process()
    peak = {"rss": 0.0}

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
        client.submit(fn, sh, d, args.crs, res, args.read_threads)
        for sh, d in work
    ]
    print(f"submitted     {len(futures)} shards in {time.perf_counter()-t0:.1f}s")

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
        lst_out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
        q = res_d["qa_count"]
        qa_out[:, y0:y0 + q.shape[1], x0:x0 + q.shape[2]] = q
        stats.append({k: res_d[k] for k in ("row", "col", "n_scenes", "load_s", "reduce_s")})
        del res_d, a, q
        done += 1
        peak["rss"] = max(peak["rss"], proc.memory_info().rss / GIB)
        if done % 25 == 0 or done == len(futures):
            el = time.perf_counter() - t_compute
            print(f"  {done:4d}/{len(futures)}  {el:6.1f}s  "
                  f"{el/done:5.2f}s/shard  client RSS {peak['rss']:.1f} GiB")
    compute_s = time.perf_counter() - t_compute

    valid = lst_out != LST_NODATA_DN
    cel = lst_out[valid].astype("float64") * LST_SCALE + LST_OFFSET if valid.any() else None
    summary = {
        "bbox": bbox, "crs": args.crs, "pixels_per_degree": args.pixels_per_degree,
        "raster": [height, width], "shard_px": args.shard,
        "n_shards": len(work), "n_scenes": len(items),
        "search_s": t_search, "compute_s": compute_s,
        "s_per_shard": compute_s / max(len(work), 1),
        "client_rss_peak_gib": peak["rss"],
        "valid_fraction": float(valid.mean()),
        "shard_stats": stats,
    }
    if cel is not None:
        summary |= {"min_c": float(cel.min()), "mean_c": float(cel.mean()),
                    "max_c": float(cel.max())}
        print(f"\nLST p95       min {cel.min():.1f} C  mean {cel.mean():.1f} C  "
              f"max {cel.max():.1f} C  ({100*valid.mean():.1f}% valid)")
    print(f"compute       {compute_s:.1f}s for {len(work)} shards "
          f"({compute_s/max(len(work),1):.2f}s each)")
    print(f"client RSS    {peak['rss']:.2f} GiB peak")
    qa_mean = {MONTHS[i]: float(qa_out[i].mean()) for i in range(12)}
    summary["qa_count_per_month"] = qa_mean
    print("qa_count      " + "  ".join(f"{m} {v:.1f}" for m, v in qa_mean.items()))

    try:
        spans = frisky.query_spans(limit=2_000_000, dashboard_url=dash, request_timeout=60)
        summary["n_spans"] = len(spans)
        (args.out_dir / "spans.json").write_text(json.dumps(spans[:200000], default=str))
    except Exception as exc:
        summary["span_error"] = repr(exc)
    cluster.close()

    # Write this slice as a part file so other machines' slices can be merged.
    import numpy as _np

    payload = {}
    for sh in mine:  # every planned shard in this slice, barren ones included
        tag = f"{sh.y0}_{sh.x0}"
        payload["lst_" + tag] = lst_out[sh.y0:sh.y0 + sh.ny, sh.x0:sh.x0 + sh.nx]
        payload["qa_" + tag] = qa_out[:, sh.y0:sh.y0 + sh.ny, sh.x0:sh.x0 + sh.nx]
    if payload:
        _np.savez_compressed(args.out_dir / "part-000.npz", **payload)
        (args.out_dir / "part-meta.json").write_text(json.dumps(
            {"raster": [height, width], "bbox": bbox, "crs": args.crs,
             "pixels_per_degree": args.pixels_per_degree,
             "shard_px": args.shard, "n_shards": len(work)}, indent=2))
        print(f"part written  {args.out_dir/'part-000.npz'} ({len(payload)//2} shards)")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"artifacts     {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

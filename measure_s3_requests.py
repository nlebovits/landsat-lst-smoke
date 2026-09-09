# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["odc-stac", "pystac-client", "xarray", "numpy", "rioxarray"]
# ///
"""Count the S3 GET requests one shard actually issues. Bounded and cheap.

Requester-pays charges scale with request count, not bytes, and nothing in this
repo has ever measured that count. Every requests-per-read figure quoted so far
has been a guess. This measures it.

Method: GDAL's curl layer logs each HTTP request when CPL_CURL_VERBOSE=YES.
Counting the "> GET " lines gives the exact number of range requests, including
the header reads GDAL makes on open. No proxy, no bucket-owner logging, no
inference.

Two things this has to get right, or the number describes a different pipeline:

* The GDAL and AWS settings must be the ones the real run uses. They come from
  `shard_lst_p95.configure_read_env`, not from a copy. `GDAL_DISABLE_READDIR_ON_OPEN`
  and `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` both change the request count
  directly, and without `AWS_REQUEST_PAYER` the reads return 403.
* rasterio installs a CPL error handler, so GDAL's curl output never reaches
  stderr at all. It arrives as `rasterio._err` log records shaped
  `CURL_INFO_HEADER_OUT: GET ...`. Both `redirect_stderr` and a file-descriptor
  redirect capture nothing here. This attaches a log handler instead.

    uv run measure_s3_requests.py --shards 3

Cost: a handful of shards, seconds of compute, a few thousand GETs. Run it on
one instance in-region. The count is what matters, not the wall clock.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from stac_window import DEFAULT_END, DEFAULT_START

sys.path.insert(0, str(Path(__file__).resolve().parent))


@contextlib.contextmanager
def capture_gdal_log():
    """Collect the curl records GDAL emits through rasterio's error handler.

    rasterio routes CPL messages into Python logging, so `CPL_CURL_VERBOSE`
    output lands on the `rasterio._err` logger rather than on stderr. Records
    arrive from every read thread, and `list.append` is atomic, so no lock is
    needed.
    """
    lines: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    logger = logging.getLogger("rasterio")
    handler = Collect()
    saved_level, saved_disabled = logger.level, logger.disabled
    logger.setLevel(logging.DEBUG)
    logger.disabled = False
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved_level)
        logger.disabled = saved_disabled


def parse_log(text: str) -> dict:
    """Turn one curl verbose log into the counts that matter.

    One `CURL_INFO_HEADER_OUT: GET ` is one billable S3 request. The header
    block travels inside a single record, so the Range line is counted anywhere
    in the text rather than at the start of a line.
    """
    return {
        "get_requests": len(re.findall(r"CURL_INFO_HEADER_OUT: GET ", text)),
        "range_requests": len(re.findall(r"Range: bytes=", text)),
        "connections": len(re.findall(r"Connected to ", text)),
        "http_200": len(re.findall(r"CURL_INFO_HEADER_IN: HTTP/[\d.]+ 200", text)),
        "http_206": len(re.findall(r"CURL_INFO_HEADER_IN: HTTP/[\d.]+ 206", text)),
        "http_4xx": len(re.findall(r"CURL_INFO_HEADER_IN: HTTP/[\d.]+ 4\d\d", text)),
        "http_5xx": len(re.findall(r"CURL_INFO_HEADER_IN: HTTP/[\d.]+ 5\d\d", text)),
        "requester_charged": len(re.findall(r"x-amz-request-charged: requester", text)),
        "retries": len(re.findall(r"Retrying again in|HTTP error code: \d+", text)),
    }


def count_requests(
    shard, item_dicts, crs, resolution, read_threads, log_path: Path | None
) -> dict:
    """Load one shard with curl verbose on, and count what crossed the wire.

    The load call mirrors `shard_lst_p95.process_shard` exactly. Any difference
    here would measure a pipeline that does not exist.
    """
    import pystac
    from odc.geo import CRS
    from odc.stac import stac_load

    ydim, xdim = ("y", "x") if CRS(crs).projected else ("latitude", "longitude")
    items = [pystac.Item.from_dict(d) for d in item_dicts]

    t0 = time.perf_counter()
    with capture_gdal_log() as lines:
        data = stac_load(
            items,
            bands=("lwir11", "qa_pixel"),
            crs=crs,
            resolution=resolution,
            bbox=shard.bbox,
            groupby="landsat:scene_id",
            chunks={"time": 1, ydim: -1, xdim: -1},
        ).compute(scheduler="threads", num_workers=read_threads)
        shape = data["lwir11"].values.shape
    wall = time.perf_counter() - t0

    text = "\n".join(lines)
    if log_path is not None:
        with gzip.open(log_path, "wt", encoding="utf-8") as fh:
            fh.write(text)

    counts = parse_log(text)
    n = len(items)
    band_reads = n * 2
    return {
        "row": shard.row,
        "col": shard.col,
        "n_scenes": n,
        "pixels": [int(shape[1]), int(shape[2])],
        "log_records": len(lines),
        **counts,
        "wall_s": wall,
        "requests_per_scene": counts["get_requests"] / max(n, 1),
        "requests_per_band_read": counts["get_requests"] / max(band_reads, 1),
        # Requests per million output pixels covered, per scene. Requests per
        # band-read rises with shard size by construction, so it cannot answer
        # whether a larger read block cuts the total. This can.
        "requests_per_scene_megapixel": counts["get_requests"]
        / max(n * shape[1] * shape[2] / 1e6, 1e-9),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbox", default="-62.5,-35.0,-60.0,-32.5")
    ap.add_argument("--pixels-per-degree", type=int, default=3600)
    ap.add_argument("--crs", default="EPSG:4326")
    ap.add_argument(
        "--shard",
        type=int,
        default=512,
        help="shard edge in pixels; this is the read size knob",
    )
    ap.add_argument("--shards", type=int, default=3, help="how many to measure")
    ap.add_argument(
        "--read-threads",
        type=int,
        default=4,
        help="must match shard_lst_p95 --read-threads",
    )
    ap.add_argument("--source", default="earth-search")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--cloud-cover-lt", type=int, default=100)
    ap.add_argument("--platforms", default="landsat-8,landsat-9")
    ap.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="cap scenes per shard, sampled evenly across the "
        "shard's items. The reported figure is a ratio per "
        "band-read, so a cap lowers the spend without changing "
        "what is measured",
    )
    ap.add_argument(
        "--keep-logs",
        action="store_true",
        help="write the gzipped curl log beside --out, as evidence",
    )
    ap.add_argument("--out", type=Path, default=Path("s3-requests.json"))
    args = ap.parse_args()

    from shard_lst_p95 import configure_read_env, items_for_shard, plan_shards

    # The catalogue query lives in stac_reference now. This is a measurement
    # tool, so it describes what the old runtime did; the runtime itself reads
    # the precomputed inventory and opens no catalogue.
    from stac_reference import search_items

    # Same settings as the real run, and curl verbose on top of them.
    configure_read_env(args.source)
    os.environ["CPL_CURL_VERBOSE"] = "YES"
    os.environ["CPL_DEBUG"] = "ON"

    bbox = tuple(float(v) for v in args.bbox.split(","))
    shards, h, w = plan_shards(bbox, args.pixels_per_degree, args.shard)

    items, boxes = search_items(
        bbox,
        start=args.start,
        end=args.end,
        platforms=args.platforms,
        cloud_cover_lt=args.cloud_cover_lt,
        source=args.source,
    )
    dicts = [i.to_dict() for i in items]
    print(
        f"scenes {len(items)}   shards planned {len(shards)}   "
        f"shard {args.shard}px   threads {args.read_threads}"
    )

    # Pick shards spread across the plan, skipping barren ones.
    picks, step = [], max(len(shards) // (args.shards + 1), 1)
    for i in range(0, len(shards), step):
        idx = items_for_shard(shards[i], boxes)
        if idx:
            if args.max_scenes and len(idx) > args.max_scenes:
                step_i = len(idx) / args.max_scenes
                idx = [idx[int(k * step_i)] for k in range(args.max_scenes)]
            picks.append((shards[i], [dicts[j] for j in idx]))
        if len(picks) == args.shards:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res = 1.0 / args.pixels_per_degree
    out = []
    for sh, d in picks:
        log_path = None
        if args.keep_logs:
            log_path = args.out.with_suffix(f".r{sh.row}c{sh.col}.log.gz")
        r = count_requests(sh, d, args.crs, res, args.read_threads, log_path)
        out.append(r)
        print(
            f"  shard r{r['row']:>2} c{r['col']:<2} {r['n_scenes']:>4} scenes  "
            f"{r['get_requests']:>7,} GETs  "
            f"{r['requests_per_band_read']:5.2f} per band-read  "
            f"{r['wall_s']:6.1f}s"
        )
        if r["http_4xx"] or r["http_5xx"] or r["retries"]:
            print(
                f"    WARNING {r['http_4xx']} 4xx, {r['http_5xx']} 5xx, "
                f"{r['retries']} retries; the count includes failed requests"
            )

    if not out:
        print("no shard had any scene; nothing measured")
        return 1

    if sum(x["get_requests"] for x in out) == 0:
        print(
            "\nZERO GETs captured. The log was not intercepted, or the reads "
            "never happened. Do not feed this into cost_report.py."
        )
        return 1

    per = [x["requests_per_band_read"] for x in out]
    permp = [x["requests_per_scene_megapixel"] for x in out]
    summary = {
        "shard_px": args.shard,
        "max_scenes": args.max_scenes,
        "read_threads": args.read_threads,
        "shards_measured": len(out),
        "scenes_searched": len(items),
        "requests_per_band_read_min": min(per),
        "requests_per_band_read_mean": sum(per) / len(per),
        "requests_per_band_read_max": max(per),
        "requests_per_scene_megapixel_mean": sum(permp) / len(permp),
        "clean": all(
            x["http_4xx"] == 0 and x["http_5xx"] == 0 and x["retries"] == 0 for x in out
        ),
        "detail": out,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(
        f"\nrequests per band-read: min {min(per):.2f} "
        f"mean {sum(per) / len(per):.2f} max {max(per):.2f}"
    )
    print(
        f"requests per scene-megapixel: mean {sum(permp) / len(permp):.2f} "
        f"(compare shard sizes on this, not on the line above)"
    )
    print(f"written {args.out}")
    print("\nFeed the mean into cost_report.py --requests-per-read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

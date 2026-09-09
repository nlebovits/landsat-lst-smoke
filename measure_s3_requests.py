# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["odc-stac", "pystac-client", "xarray", "numpy", "rioxarray"]
# ///
"""Count the S3 GET requests one shard actually issues. Bounded and cheap.

Requester-pays charges scale with request count, not bytes, and nothing in this
repo has ever measured that count. Every requests-per-read figure quoted so far
has been a guess. This measures it.

Method: GDAL's curl layer logs each HTTP request when CPL_CURL_VERBOSE=YES.
Counting the "> GET " lines on stderr gives the exact number of range requests,
including the header reads GDAL makes on open. No proxy, no bucket-owner
logging, no inference.

    uv run measure_s3_requests.py --shards 3

Cost: a handful of shards, seconds of compute, a few thousand GETs. Run it on
one instance in-region; the count is what matters, not the wall clock.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def count_requests(shard, item_dicts, crs, resolution) -> dict:
    """Load one shard with curl verbose on, and count what crossed the wire."""
    import numpy as np
    import pystac
    from odc.geo import CRS
    from odc.stac import stac_load

    os.environ["CPL_CURL_VERBOSE"] = "YES"
    os.environ["CPL_DEBUG"] = "ON"

    ydim, xdim = ("y", "x") if CRS(crs).projected else ("latitude", "longitude")
    items = [pystac.Item.from_dict(d) for d in item_dicts]

    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stderr(buf):
        data = stac_load(
            items, bands=("lwir11", "qa_pixel"), crs=crs, resolution=resolution,
            bbox=shard.bbox, groupby="landsat:scene_id",
            chunks={"time": 1, ydim: -1, xdim: -1},
        ).compute(scheduler="threads", num_workers=4)
        _ = data["lwir11"].values.shape
    wall = time.perf_counter() - t0
    log = buf.getvalue()

    gets = len(re.findall(r"^> GET ", log, re.M))
    ranges = len(re.findall(r"^> Range: ", log, re.M))
    conns = len(re.findall(r"Connected to ", log))
    return {
        "row": shard.row, "col": shard.col,
        "n_scenes": len(items),
        "get_requests": gets,
        "range_requests": ranges,
        "connections": conns,
        "wall_s": wall,
        "requests_per_scene": gets / max(len(items), 1),
        "requests_per_band_read": gets / max(len(items) * 2, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbox", default="-62.5,-35.0,-60.0,-32.5")
    ap.add_argument("--pixels-per-degree", type=int, default=3600)
    ap.add_argument("--crs", default="EPSG:4326")
    ap.add_argument("--shard", type=int, default=512)
    ap.add_argument("--shards", type=int, default=3, help="how many to measure")
    ap.add_argument("--load-chunk", type=int, default=None,
                    help="unused here; the point is to measure before tuning")
    ap.add_argument("--out", type=Path, default=Path("s3-requests.json"))
    args = ap.parse_args()

    from shard_lst_p95 import items_for_shard, plan_shards, search_items

    bbox = tuple(float(v) for v in args.bbox.split(","))
    shards, h, w = plan_shards(bbox, args.pixels_per_degree, args.shard)

    class A:  # search_items expects an argparse-ish object
        source = "earth-search"; collection = None
        cloud_cover_lt = 100; platforms = "landsat-8,landsat-9"
        start = "2020-01-01"; end = "2025-01-01"
    items, boxes = search_items(A(), bbox)
    dicts = [i.to_dict() for i in items]
    print(f"scenes {len(items)}   shards planned {len(shards)}")

    # Pick shards spread across the plan, skipping barren ones.
    picks, step = [], max(len(shards) // (args.shards + 1), 1)
    for i in range(0, len(shards), step):
        idx = items_for_shard(shards[i], boxes)
        if idx:
            picks.append((shards[i], [dicts[j] for j in idx]))
        if len(picks) == args.shards:
            break

    res = 1.0 / args.pixels_per_degree
    out = []
    for sh, d in picks:
        r = count_requests(sh, d, args.crs, res)
        out.append(r)
        print(f"  shard r{r['row']:>2} c{r['col']:<2} {r['n_scenes']:>4} scenes  "
              f"{r['get_requests']:>7,} GETs  {r['requests_per_band_read']:5.2f} per band-read  "
              f"{r['wall_s']:6.1f}s")

    if out:
        per = [x["requests_per_band_read"] for x in out]
        summary = {
            "shards_measured": len(out),
            "requests_per_band_read_min": min(per),
            "requests_per_band_read_mean": sum(per) / len(per),
            "requests_per_band_read_max": max(per),
            "detail": out,
        }
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nrequests per band-read: min {min(per):.2f} "
              f"mean {sum(per)/len(per):.2f} max {max(per):.2f}")
        print(f"written {args.out}")
        print("\nFeed the mean into cost_report.py --requests-per-read. "
              "Until then, S3 request cost is UNKNOWN, not estimated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2", "dask", "odc-stac", "xarray", "numpy",
#   "pystac", "psutil", "boto3", "pyarrow>=16",
#   "rasterio", "shapely", "geopandas", "pyogrio",
# ]
# ///
"""What the two seam corrections do, measured on one shard that straddles a swath.

Four composites from one load: pooled, de-striped only, feathered only, and
both. Same scenes, same worker code, same reads, so every difference between
the four rasters is the correction and nothing else. One `stac_load`, one warp,
and four reductions over it. The Celsius decode repeats per arm, because
de-biasing writes into the stack it corrects.

    uv run measure_seam.py --tile S30W065 --tile-prep ./tile-prep \\
        --out-dir ./seam

Pick the shard with `--shard-index`, or let `--auto` choose the planned shard
with the most swath boundary inside it. `FINDINGS.md` records that every
comparison so far sat inside a WRS footprint and measured the interior case, so
choosing the boundary case is the point of this script.

**The metric.** A seam is a jump between neighbouring pixels that follows
satellite geometry rather than ground. So the measurement is the mean absolute
step across the pixel edges that cross a swath boundary, minus the median
absolute step over every other edge. Subtracting that floor is what separates a
seam from ordinary texture: a field with sharp field boundaries has large steps
everywhere, and only the excess at the swath edge is the artifact.

Reported beside it is the share of spatial variance retained. A correction that
flattens the field removes the seam and the signal together, and
`nlebovits/landsat-lst` measured exactly that failure at 32% of variance lost
from scene-wise scale matching. The pair of numbers is the result; neither one
alone is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import destripe
import staging
from land_tiles import tile_bounds
from masks import transform_for
from shard_lst_p95 import (
    DEFAULT_INVENTORY_URI,
    DEFAULT_STAGE_DIR,
    configure_read_env,
    items_for_shard,
    load_shard,
    load_tile_items,
    plan_shards,
    reduce_shard,
)
from stac_window import (
    DEFAULT_CLOUD_COVER_LT,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
)

#: The four arms, as `(name, debias, feather)`.
ARMS = (
    ("pooled", False, False),
    ("destriped", True, False),
    ("feathered", False, True),
    ("both", True, True),
)


def boundary_edges(weight):
    """Pixel edges that cross a swath boundary, as two boolean masks.

    An edge crosses a boundary when the set of paths reaching one side differs
    from the set reaching the other. That is exactly where the pooled composite
    changes its sample set, and so exactly where it can step.
    """
    covering = weight > 0
    down = (covering[:, 1:, :] != covering[:, :-1, :]).any(axis=0)
    right = (covering[:, :, 1:] != covering[:, :, :-1]).any(axis=0)
    return down, right


def seam_step(field, down, right):
    """Mean step across swath boundaries, above the step everywhere else.

    Returns None when the shard holds no boundary, which means it is the
    interior case and this measurement does not apply to it.
    """
    import numpy as np

    d = np.abs(np.diff(field, axis=0))
    r = np.abs(np.diff(field, axis=1))
    finite_d = np.isfinite(d)
    finite_r = np.isfinite(r)

    on = np.concatenate([d[down & finite_d], r[right & finite_r]])
    off = np.concatenate([d[~down & finite_d], r[~right & finite_r]])
    if on.size == 0 or off.size == 0:
        return None
    return {
        "on_boundary_mean_c": float(on.mean()),
        "elsewhere_median_c": float(np.median(off)),
        "excess_c": float(on.mean() - np.median(off)),
        "n_boundary_edges": int(on.size),
    }


def compare(fields: dict, down, right) -> tuple[dict, dict]:
    """Every arm against the pooled baseline, on the pixels all four share.

    De-striping discards scenes, so a pixel the pooled arm resolves can be
    nodata once the offsets are applied. Every number here is computed on the
    intersection, which makes the four arms comparable and makes none of them
    the raster that arm would ship on its own. `n_valid_px` against the shard
    size says how much that cost.

    Returns `(summary, rows)`. They are returned apart rather than nested so the
    caller can iterate the rows without unpacking a dict of mixed value types.
    """
    import numpy as np

    valid = np.ones_like(fields["pooled"], dtype=bool)
    for field in fields.values():
        valid &= np.isfinite(field)

    base = seam_step(np.where(valid, fields["pooled"], np.nan), down, right)
    base_var = float(np.var(fields["pooled"][valid])) if valid.any() else 0.0

    rows: dict[str, dict] = {}
    for name, field in fields.items():
        step = seam_step(np.where(valid, field, np.nan), down, right)
        variance = float(np.var(field[valid])) if valid.any() else 0.0
        row: dict = {
            "seam": step,
            "variance_c2": round(variance, 4),
            "variance_retained": round(variance / base_var, 4) if base_var else None,
            "mean_c": round(float(field[valid].mean()), 3) if valid.any() else None,
        }
        if base and step and base["excess_c"] > 0:
            row["seam_removed"] = round(1.0 - step["excess_c"] / base["excess_c"], 4)
        rows[name] = row

    summary = {
        "n_valid_px": int(valid.sum()),
        "n_px": int(valid.size),
        "valid_share": round(float(valid.mean()), 4),
        "pooled_variance_c2": round(base_var, 4),
        "pooled_seam": base,
    }
    return summary, rows


def pick_shard(shards, prep):
    """The planned shard with the most swath boundary in it.

    Counted once on the prep file's own swath grid, then read per shard by
    slicing that one array. Resampling every shard's weights to answer this
    would be two `reproject` calls per path for each of 1,296 shards, which
    takes minutes to answer a question one array already holds.
    """
    import numpy as np
    from rasterio.transform import rowcol

    if prep.inside.size == 0:
        raise SystemExit("this prep file found no WRS path with a swath.")
    down, right = boundary_edges(prep.inside.astype("float32"))
    edge = np.zeros(prep.inside.shape[1:], dtype="uint8")
    edge[:-1, :] |= down
    edge[:, :-1] |= right
    integral = np.pad(edge.astype("int64").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    transform = destripe.prep_transform(prep)
    height, width = edge.shape

    def count(shard):
        west, south, east, north = shard.bbox
        r0, c0 = rowcol(transform, west, north, op=math.floor)
        r1, c1 = rowcol(transform, east, south, op=math.ceil)
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(r1), height), min(int(c1), width)
        if r1 <= r0 or c1 <= c0:
            return 0
        return int(
            integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        )

    best, best_edges = None, -1
    for shard in shards:
        edges = count(shard)
        if edges > best_edges:
            best, best_edges = shard, edges
    if best is None or best_edges <= 0:
        raise SystemExit(
            "no planned shard of this tile holds a swath boundary. Every shard "
            "is the interior case, and there is no seam here to measure."
        )
    print(
        f"chosen        shard row {best.row} col {best.col}, "
        f"{best_edges} swath cells on a boundary"
    )
    return best


def decode(dn):
    """DN back to Celsius, with nodata as NaN."""
    import numpy as np

    from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE

    out = dn.astype("float32") * LST_SCALE + LST_OFFSET
    return np.where(dn == LST_NODATA_DN, np.nan, out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tile", required=True)
    p.add_argument("--tile-prep", type=Path, required=True)
    p.add_argument("--inventory-uri", type=Path, default=DEFAULT_INVENTORY_URI)
    p.add_argument("--out-dir", type=Path, default=Path("./seam"))
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument("--shard", type=int, default=512)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--auto", action="store_true", help="pick the seamiest shard")
    p.add_argument("--max-offset-c", type=float, default=destripe.DESTRIPE_MAX_OFFSET_C)
    p.add_argument("--read-threads", type=int, default=4)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--source", default="earth-search")
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    args = p.parse_args(argv)
    if args.shard_index is None and not args.auto:
        raise SystemExit("pass --shard-index N or --auto")
    return args


def main(argv=None) -> int:
    import numpy as np

    args = parse_args(argv)
    prep = destripe.load_prep(args.tile_prep)
    if prep.tile != args.tile:
        raise SystemExit(f"{args.tile_prep} was written for tile {prep.tile}")
    if not prep.paths:
        raise SystemExit(
            f"{args.tile_prep} found no WRS path with a swath. There is no "
            f"boundary in this tile and nothing here to measure."
        )

    bbox = tile_bounds(args.tile)
    shards, _h, _w = plan_shards(bbox, args.pixels_per_degree, args.shard)
    items, boxes, run_provenance = load_tile_items(args, args.tile)

    def weight_of(shard):
        return destripe.weights_for_window(
            prep.weight,
            prep.inside,
            destripe.prep_transform(prep),
            transform_for(shard.bbox, args.pixels_per_degree),
            (shard.ny, shard.nx),
        )

    shard = pick_shard(shards, prep) if args.auto else shards[args.shard_index]
    idx = items_for_shard(shard, boxes)
    if not idx:
        raise SystemExit(f"shard row {shard.row} col {shard.col} holds no scene")
    item_dicts = [items[i] for i in idx]
    print(f"shard         {shard.bbox}  {len(item_dicts)} scenes")

    configure_read_env(args.source)
    report = staging.stage_scenes(items, idx, args.stage_dir)
    print(
        f"stage         {report['objects']:,} objects, "
        f"{report['get_requests']:,} billable GETs"
    )

    # One read for all four arms. Reading per arm made the comparison cost four
    # times what it measures, because the load is 94% of a shard. The decode
    # still repeats: `destripe.apply_to_stack` de-biases in place, so an arm
    # handed the previous arm's array would reduce a stack already shifted.
    # Decoding from the uint16 source holds one float32 stack at a time, where
    # keeping four pristine copies would hold four.
    data, items, load_s = load_shard(
        shard,
        item_dicts,
        args.crs,
        1.0 / args.pixels_per_degree,
        args.read_threads,
    )
    print(f"load          {load_s:.2f}s for {len(items)} scenes, once for four arms")

    fields = {}
    timings = {}
    for name, debias, feather in ARMS:
        correction = destripe.shard_correction(
            prep,
            item_dicts,
            shard.bbox,
            (shard.ny, shard.nx),
            pixels_per_degree=args.pixels_per_degree,
            max_offset_c=args.max_offset_c,
            debias=debias,
            feather=feather,
        )
        t0 = time.perf_counter()
        out = reduce_shard(shard, data, items, load_s, correction)
        timings[name] = round(time.perf_counter() - t0, 2)
        fields[name] = decode(out["lst_p95"])
        print(
            f"{name:<14}{timings[name]:7.2f}s  "
            f"{out['n_scenes']} scenes loaded, "
            f"{out['n_scenes_kept']} kept, {out['n_rejected']} rejected, "
            f"{out['n_pooled_fallback']} px pooled"
        )

    down, right = boundary_edges(weight_of(shard))
    summary, rows = compare(fields, down, right)
    result = summary | {
        "arms": rows,
        "tile": args.tile,
        "shard": {"row": shard.row, "col": shard.col, "bbox": list(shard.bbox)},
        "n_scenes": len(item_dicts),
        "max_offset_c": args.max_offset_c,
        "seconds": timings,
        "prep": prep.meta.get("offsets"),
        "inventory": run_provenance,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seam.json").write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(args.out_dir / "seam_fields.npz", **fields)

    print()
    print(f"{'arm':<12}{'seam excess':>13}{'removed':>10}{'variance kept':>15}")
    for name, row in rows.items():
        excess = row["seam"]["excess_c"] if row["seam"] else float("nan")
        removed = row.get("seam_removed")
        kept = row["variance_retained"]
        print(
            f"{name:<12}"
            f"{excess:>12.3f}C"
            f"{'-' if removed is None else format(removed, '.1%'):>10}"
            f"{'-' if kept is None else format(kept, '.1%'):>15}"
        )
    print(f"\nwritten       {args.out_dir / 'seam.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

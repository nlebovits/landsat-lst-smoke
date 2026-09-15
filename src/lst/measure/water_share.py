# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "numpy", "odc-stac", "odc-geo", "rasterio", "pyarrow", "pyproj", "shapely",
#   "geopandas", "pyogrio", "xarray", "dask",
# ]
# ///
"""What share of a pixel's clear observations call it water, measured.

`lst_qa.observed_water` decides a pixel from two counters that
`composite.reduce_block` accumulates over the whole five-year stack: how many
observations were usable and clear, and how many of those carried QA_PIXEL bit
7. This script is where the threshold between them comes from.

It rebuilds one block of a tile from its scenes, the way
`composite.fused_block` does, and reports the distribution of that share. It
reads only the block's window from each covering scene, so a block costs
megabytes where a tile costs 400 GB, and it caches the raw reads so a second
question about the same block needs no second read.

The counters here are the ones the kernel keeps. Validity comes from
`lst_qa.masked_celsius`, so a change to the QA rule moves this measurement and
the product together.

    uv run lst-measure-water-share --tile N40W080 --lat 38.00 --lon -76.25 \
        --label sea --out artifacts/water_share.json

A block already cached by an earlier run, or by `lst-tiles/block_probe.py`,
is read straight from disk and needs no credentials:

    uv run lst-measure-water-share --from-cache ../lst-tiles/block_K.npz.raw.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lst import composite
from lst import destripe
from lst import lst_qa
from lst.lst_qa import QA_WATER_BITS, masked_celsius

#: Thresholds the report always states, so two blocks are read off one table.
REPORTED_THRESHOLDS = (0.25, 0.50, 0.75, 0.90)

#: Width of a histogram bin of the water share.
SHARE_BIN = 0.05

#: Rows of a block decoded at once. The decode holds one float32 copy of the
#: chunk's whole time axis, so this trades peak memory against loop overhead:
#: 60 rows of a 765-scene 360 px block is about 66 MB.
DECODE_ROWS = 60


def water_counts(lwir, qa, keep, *, rows: int = DECODE_ROWS):
    """`(water_observations, clear_observations)` for one block, as uint32.

    The two counters `composite.reduce_block` keeps, computed the same way and
    on the same definition of a usable observation. Both are unsaturated: the
    published `qa_count` clips at 255 a month and drops the water flag, so the
    share cannot be read back off it.

    Args:
        lwir: `(y, x, time)` uint16 of raw thermal DN.
        qa: `(y, x, time)` uint16 of QA_PIXEL.
        keep: `(time,)` bool, the destripe rejection from
            `composite.scene_vectors`. A rejected scene is not in the
            percentile, so it is not in the share either.
        rows: how many rows to decode at once.

    Returns:
        Two `(y, x)` uint32 arrays, the numerator and the denominator.
    """
    height, width = lwir.shape[0], lwir.shape[1]
    keep = np.asarray(keep, dtype=bool)
    water = np.zeros((height, width), dtype="uint32")
    clear = np.zeros((height, width), dtype="uint32")
    for y in range(0, height, rows):
        stop = min(y + rows, height)
        band = np.moveaxis(lwir[y:stop], -1, 0)[keep]
        flags = np.moveaxis(qa[y:stop], -1, 0)[keep]
        _celsius, valid = masked_celsius(np.ascontiguousarray(band), flags)
        clear[y:stop] = valid.sum(axis=0, dtype="uint32")
        water[y:stop] = (valid & ((flags & QA_WATER_BITS) != 0)).sum(
            axis=0, dtype="uint32"
        )
    return water, clear


def share_of(water, clear):
    """The water share, and where it is defined.

    Returns `(share, known)`. `share` is 0.0 where nothing observed the pixel,
    and `known` says where that 0.0 means "no water" rather than "no evidence".
    """
    known = np.asarray(clear) > 0
    share = np.zeros(known.shape, dtype="float64")
    np.divide(water, clear, out=share, where=known)
    return share, known


def histogram(share, known, *, width: float = SHARE_BIN):
    """Counts of the water share in equal bins over [0, 1]."""
    edges = np.arange(0.0, 1.0 + width / 2, width)
    counts, _ = np.histogram(share[known], bins=edges)
    return [
        {"low": round(float(edges[i]), 4), "high": round(float(edges[i + 1]), 4)}
        | {"pixels": int(counts[i])}
        for i in range(counts.size)
    ]


def summarise(water, clear, dn=None, *, label: str = "block") -> dict:
    """Everything one block says about the threshold.

    Args:
        water: the numerator from `water_counts`.
        clear: the denominator from `water_counts`.
        dn: the block's published `lst_p95` DN, if it is known. With it the
            report states how many published values each threshold withdraws,
            and what temperature each share band holds.
        label: what the block is, carried into the record.
    """
    share, known = share_of(water, clear)
    quantiles = (1, 5, 25, 50, 75, 90, 95, 99)
    record = {
        "label": label,
        "pixels": int(share.size),
        "pixels_unobserved": int((~known).sum()),
        "observations": {
            "median": int(np.median(clear[known])) if known.any() else 0,
            "quantiles": {
                str(q): int(v)
                for q, v in zip(
                    quantiles, np.percentile(clear[known], quantiles), strict=True
                )
            }
            if known.any()
            else {},
        },
        "share_quantiles": {
            str(q): round(float(v), 6)
            for q, v in zip(
                quantiles, np.percentile(share[known], quantiles), strict=True
            )
        }
        if known.any()
        else {},
        "histogram": histogram(share, known),
        "classified": {},
    }
    valid = None if dn is None else (np.asarray(dn) != lst_qa.LST_NODATA_DN)
    for threshold in REPORTED_THRESHOLDS:
        hit = known & (share >= threshold)
        block = {
            "pixels": int(hit.sum()),
            "share_of_block": round(float(hit.mean()), 6),
        }
        if valid is not None:
            block["published_valid_withdrawn"] = int((hit & valid).sum())
        record["classified"][str(threshold)] = block
    if valid is not None:
        record["published_valid"] = int(valid.sum())
        record["temperature_by_share"] = _temperature_bands(share, known, dn, valid)
    return record


def _temperature_bands(share, known, dn, valid) -> list[dict]:
    """The published temperature inside each band of the water share.

    This is the physical check on the threshold. A band whose temperature sits
    inside the land distribution is land, whatever its share says, and a
    threshold that reaches it deletes ground.
    """
    celsius = np.asarray(dn, dtype="float64") * lst_qa.LST_SCALE + lst_qa.LST_OFFSET
    bands = (
        (0.0, 0.05),
        (0.05, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 0.9),
        (0.9, 1.1),
    )
    out = []
    for low, high in bands:
        sel = known & valid & (share >= low) & (share < high)
        if not sel.any():
            continue
        p25, median, p75 = np.percentile(celsius[sel], (25, 50, 75))
        out.append(
            {
                "low": low,
                "high": min(high, 1.0),
                "pixels": int(sel.sum()),
                "p25_c": round(float(p25), 2),
                "median_c": round(float(median), 2),
                "p75_c": round(float(p75), 2),
            }
        )
    return out


def block_holding(plan, lat: float, lon: float, bbox, pixels_per_degree: int):
    """The `BlockSpec` a coordinate falls in.

    Rounds rather than truncates. `(-75.20 + 80) * 3600` is 17279.999999 in
    binary floating point, and truncation puts the probe in the block next
    door.
    """
    west, _south, _east, north = bbox
    row = int(round((north - lat) * pixels_per_degree))
    col = int(round((lon - west) * pixels_per_degree))
    for block in plan:
        if (
            block.yslice.start <= row < block.yslice.stop
            and block.xslice.start <= col < block.xslice.stop
        ):
            return block
    msg = f"no block of this tile holds {lat}, {lon}"
    raise SystemExit(msg)


def parallel_read(items, geobox, band: str, *, workers: int):
    """`composite.read_block` one scene at a time, in threads, stacked in order.

    Each scene's plane is independent of every other, so the kernel called per
    item and stacked gives the array the loop inside it gives. The loop is
    latency bound rather than bandwidth bound, and a five-year block is
    hundreds of remote opens: serially that is tens of minutes, and the whole
    point of a block probe is that it costs megabytes and minutes.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(index: int):
        return index, composite.read_block([items[index]], geobox, band)[..., 0]

    # Keyed by index rather than a pre-filled list of None. The list form
    # types as `list[None]`, which `np.stack` refuses, and a worker that
    # returned nothing would have stacked a None into the array instead of
    # raising. A missing key is a KeyError at the point the gap appears.
    planes: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, plane in pool.map(one, range(len(items))):
            planes[index] = plane
    print(f"read          {band} {len(items)} scenes", flush=True)
    return np.stack([planes[i] for i in range(len(items))], axis=-1)


def read_raw(items, geobox, cache: Path | None, *, workers: int):
    """The block's two bands, from `cache` if it holds them.

    Reading a five-year block is thousands of ranged GETs against a
    requester-pays bucket. Caching them makes a second question about the same
    block free, and makes the measurement repeatable without credentials.
    """
    if cache is not None and cache.exists():
        held = np.load(cache)
        print(f"cached        {cache}", flush=True)
        return held["lwir"], held["qa"]
    lwir = parallel_read(items, geobox, "lwir11", workers=workers)
    qa = parallel_read(items, geobox, "qa_pixel", workers=workers)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lwir=lwir, qa=qa)
        print(f"cached to     {cache}", flush=True)
    return lwir, qa


def probe(args) -> dict:
    """Rebuild one block from its scenes and measure it."""
    from odc.geo.geobox import GeoBox

    from lst import masks
    from lst import shard_lst_p95
    from lst import staging
    from lst.land_tiles import tile_bounds
    from lst.tile_inventory import items_for_tile

    shard_lst_p95.configure_read_env()
    bbox = tile_bounds(args.tile)
    height, width = composite.raster_shape(bbox, args.pixels_per_degree)
    geobox = GeoBox(
        (height, width), masks.transform_for(bbox, args.pixels_per_degree), "EPSG:4326"
    )
    items, boxes = items_for_tile(args.inventory, args.tile, bounds=bbox)
    # The same cut the run makes before it builds a graph. An `OLI_TIRS_L2SR`
    # product carries no `lwir11` asset at all, so a reader asked for one
    # raises rather than returning fill.
    items, boxes, dropped = staging.drop_scenes_without_thermal(items, boxes)
    if dropped:
        print(f"dropped       {dropped} scenes with no thermal band", flush=True)
    prep = destripe.load_prep(args.tile_prep)
    plan = composite.build_block_plan(items, boxes, geobox, args.chunk)
    vectors = composite.scene_vectors(items, prep)
    block = block_holding(plan, args.lat, args.lon, bbox, args.pixels_per_degree)
    print(
        f"block         {args.tile} r{block.row} c{block.col} "
        f"rows {block.yslice.start}-{block.yslice.stop} "
        f"cols {block.xslice.start}-{block.xslice.stop} "
        f"scenes {block.depth}",
        flush=True,
    )

    scenes = [items[i] for i in block.item_indices]
    theirs = composite.block_vectors(vectors, block)
    lwir, qa = read_raw(scenes, block.geobox, args.cache, workers=args.workers)
    water, clear = water_counts(lwir, qa, theirs.keep)

    weight = composite.block_weights(prep, block.geobox)
    dn = composite.reduce_block(
        lwir,
        qa,
        theirs.offset,
        theirs.keep,
        theirs.path_code,
        theirs.month,
        weight,
        n_paths=len(theirs.paths),
        feather=True,
        emit_pooled=False,
    )[0]
    if args.save_block is not None:
        # Everything a replay needs to run this block again without reading a
        # scene: the per-scene vectors, the weights, and what the kernel said.
        args.save_block.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_block,
            dn=np.asarray(dn),
            offset=theirs.offset,
            keep=theirs.keep,
            path_code=theirs.path_code,
            month=theirs.month,
            weight=weight,
            times=theirs.times.astype("datetime64[s]").astype("int64"),
            scene_ids=np.array([destripe.scene_id_of(i) for i in scenes]),
            yslice=np.array([block.yslice.start, block.yslice.stop]),
            xslice=np.array([block.xslice.start, block.xslice.stop]),
        )
        print(f"block to      {args.save_block}", flush=True)

    record = summarise(water, clear, dn, label=args.label or args.tile)
    record["tile"] = args.tile
    record["lat"] = args.lat
    record["lon"] = args.lon
    record["block"] = {
        "row": int(block.row),
        "col": int(block.col),
        "rows": [int(block.yslice.start), int(block.yslice.stop)],
        "cols": [int(block.xslice.start), int(block.xslice.stop)],
        "scenes": int(block.depth),
    }
    return record


def from_cache(args) -> dict:
    """Measure a block whose raw reads are already on disk.

    Takes the `.raw.npz` an earlier probe wrote. A sibling `.npz` holding the
    block's `keep` vector and its published `dn` is used when it is beside it,
    which is what `lst-tiles/block_probe.py` writes.
    """
    held = np.load(args.from_cache)
    lwir, qa = held["lwir"], held["qa"]
    name = str(args.from_cache)
    beside = Path(name[: -len(".raw.npz")]) if name.endswith(".raw.npz") else None
    keep, dn = np.ones(lwir.shape[-1], dtype=bool), None
    if beside is not None and beside.exists():
        block = np.load(beside, allow_pickle=True)
        keep, dn = block["keep"], block["dn"]
        print(f"block         {beside}", flush=True)
    water, clear = water_counts(lwir, qa, keep)
    record = summarise(water, clear, dn, label=args.label or Path(name).stem)
    record["source"] = name
    return record


def report(record: dict) -> None:
    """The record as a table, because a threshold is chosen by eye first."""
    print(
        f"\n{record['label']}  {record['pixels']:,} px, "
        f"{record['pixels_unobserved']:,} unobserved"
    )
    print(f"observations  median {record['observations']['median']}")
    if record.get("share_quantiles"):
        line = "  ".join(
            f"q{q}={v:.4f}"
            for q, v in sorted(
                record["share_quantiles"].items(), key=lambda kv: int(kv[0])
            )
        )
        print(f"share         {line}")
    print("histogram")
    for row in record["histogram"]:
        if row["pixels"]:
            print(f"  [{row['low']:.2f},{row['high']:.2f})  {row['pixels']:>9,}")
    print("classified")
    for threshold, block in record["classified"].items():
        line = (
            f"  >= {threshold}  {block['pixels']:>9,} px  {block['share_of_block']:.4%}"
        )
        if "published_valid_withdrawn" in block:
            line += f"  withdraws {block['published_valid_withdrawn']:,} published"
        print(line)
    for row in record.get("temperature_by_share", []):
        print(
            f"  [{row['low']:.2f},{row['high']:.2f})  {row['pixels']:>8,} px  "
            f"p25 {row['p25_c']:6.2f}  median {row['median_c']:6.2f}  "
            f"p75 {row['p75_c']:6.2f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default="N40W080")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--chunk", type=int, default=360)
    parser.add_argument("--pixels-per-degree", type=int, default=3600)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="measure a .raw.npz an earlier probe wrote, reading no scenes",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="where to write or read this block's raw reads",
    )
    parser.add_argument(
        "--inventory", type=Path, default=Path("artifacts/tile_scene_inventory.parquet")
    )
    parser.add_argument("--tile-prep", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="threads issuing the scene reads",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--save-block",
        type=Path,
        default=None,
        help="write this block's vectors, weights, and DN, so a replay can "
        "rerun the kernel on it without reading a scene",
    )
    args = parser.parse_args(argv)
    if args.from_cache is None and (args.lat is None or args.lon is None):
        parser.error("--lat and --lon are required unless --from-cache is given")
    if args.from_cache is None and args.tile_prep is None:
        parser.error("--tile-prep is required to rebuild a block")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    record = from_cache(args) if args.from_cache is not None else probe(args)
    report(record)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwritten       {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

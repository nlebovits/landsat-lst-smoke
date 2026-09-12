# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "odc-stac", "odc-geo", "pystac", "pystac-client", "xarray", "dask",
#   "numpy", "rasterio", "matplotlib",
# ]
# ///
"""What the QA and nodata change does to one real window of the archive.

Runs one fixed window twice over the same scenes, the same bounds, and the same
resolution. The first pass applies the mask this repository shipped before:
QA_PIXEL bits 3 and 4, an exact `dn != 0` fill test, and an encoder whose floor
is DN 1. The second is `composite.build_graph`, the graph the pipeline runs,
which applies `lst_qa`: QA_PIXEL bits 1 to 5, the fill test, a `[-50, 80]` C
plausibility range before the percentile, and an encoder whose floor is
`LST_MIN_TRUSTED_DN`.

Only the mask changes between the two passes. The window stays on the window
the earlier measurements used, so the year change cannot be confused with the
masking change. Run the year change as its own dry run and report its scene
count separately.

    uv run compare_qa_masks.py --max-scenes 120 --out ./qa-parity

The window is one block, and the graph runs under the synchronous scheduler in
this process, so the comparison stays one process and no cluster starts. Both
rasters are written as GeoTIFFs on the window's own grid beside the JSON
report, so they open in QGIS next to a published COG.

The scene cap is the spend control. It samples evenly across the window's
items, so a capped run still crosses WRS boundaries and still holds every
rejection reason. Requester-pays reads cost money, and the window is read
twice, once raw for the legacy pass and once by the graph: 120 scenes over a
512 px window is about 2,300 GET requests between them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import destripe
from lst_qa import (
    LST_MAX_DN,
    LST_MIN_DN,
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    LST_VALID_MAX_C,
    LST_VALID_MIN_C,
    LWIR_FILL_DN,
    QA_CIRRUS_BIT,
    QA_CLOUD_BIT,
    QA_CLOUD_SHADOW_BIT,
    QA_DILATED_CLOUD_BIT,
    QA_SNOW_BIT,
    in_trusted_range,
    masked_celsius,
    qa_clear,
    to_celsius,
)

#: The mask this repository applied before the change: cloud and cloud shadow.
LEGACY_QA_BITS = 0b11000


def legacy_masked_celsius(thermal_dn, qa_pixel):
    """The pre-change rule, kept here so the comparison runs both in one process."""
    valid = ((qa_pixel & LEGACY_QA_BITS) == 0) & (thermal_dn != LWIR_FILL_DN)
    celsius = to_celsius(thermal_dn)
    celsius[~valid] = np.nan
    return celsius, valid


def legacy_encode(celsius):
    """The pre-change encoder: same nodata policy, but a DN 1 floor."""
    dn = np.rint((celsius - LST_OFFSET) / LST_SCALE)
    bad = ~np.isfinite(dn) | (dn < LST_MIN_DN) | (dn > LST_MAX_DN)
    dn[bad] = LST_NODATA_DN
    return dn.astype("uint16")


def p95_and_counts(celsius, valid):
    with np.errstate(all="ignore"):
        p95 = np.nanpercentile(celsius, 95, axis=0)
    return p95, valid.sum(axis=0).astype("int32")


def rejection_census(thermal_dn, qa_pixel) -> dict:
    """How many samples each rule removes, in the order the rules run.

    The per-bit counts are unconditional: a sample flagged both cloud and
    shadow appears under both. The sequential counts are what each step removes
    from what reached it, so they sum to the total rejected.
    """
    total = int(thermal_dn.size)
    is_fill = thermal_dn == LWIR_FILL_DN
    clear = np.asarray(qa_clear(qa_pixel))

    survived_fill = ~is_fill
    qa_removed = survived_fill & ~clear
    survived_qa = survived_fill & clear

    celsius = to_celsius(thermal_dn)
    finite = np.isfinite(celsius)
    non_finite_removed = survived_qa & ~finite
    below = survived_qa & finite & (celsius < LST_VALID_MIN_C)
    above = survived_qa & finite & (celsius > LST_VALID_MAX_C)

    per_bit = {
        "dilated_cloud_bit_1": QA_DILATED_CLOUD_BIT,
        "cirrus_bit_2": QA_CIRRUS_BIT,
        "cloud_bit_3": QA_CLOUD_BIT,
        "cloud_shadow_bit_4": QA_CLOUD_SHADOW_BIT,
        "snow_bit_5": QA_SNOW_BIT,
    }
    census = {
        "samples": total,
        "rejected_raw_fill": int(is_fill.sum()),
        "rejected_qa_bits_1_to_5": int(qa_removed.sum()),
        "rejected_non_finite": int(non_finite_removed.sum()),
        "rejected_below_minus_50_c": int(below.sum()),
        "rejected_above_80_c": int(above.sum()),
        "qa_bits_set_anywhere": {
            name: int(((qa_pixel >> bit) & 1).astype(bool).sum())
            for name, bit in per_bit.items()
        },
        "legacy_qa_bits_3_and_4_removed": int(
            (survived_fill & ((qa_pixel & LEGACY_QA_BITS) != 0)).sum()
        ),
    }
    census["rejected_total"] = (
        census["rejected_raw_fill"]
        + census["rejected_qa_bits_1_to_5"]
        + census["rejected_non_finite"]
        + census["rejected_below_minus_50_c"]
        + census["rejected_above_80_c"]
    )
    census["kept"] = total - census["rejected_total"]
    return census


def decode(dn):
    return dn.astype("float64") * LST_SCALE + LST_OFFSET


def compare(old_dn, new_dn) -> dict:
    """Everything the report has to state about the two output rasters."""
    old_valid = old_dn != LST_NODATA_DN
    new_valid = new_dn != LST_NODATA_DN
    changed = old_dn != new_dn
    both = old_valid & new_valid

    diff = decode(new_dn[both]) - decode(old_dn[both])
    out = {
        "pixels": int(old_dn.size),
        "valid_pixels_before": int(old_valid.sum()),
        "valid_pixels_after": int(new_valid.sum()),
        "pixels_changed": int(changed.sum()),
        "pixels_changed_fraction": float(changed.mean()),
        "valid_to_nodata": int((old_valid & ~new_valid).sum()),
        "nodata_to_valid": int((~old_valid & new_valid).sum()),
        "both_valid": int(both.sum()),
    }
    if diff.size:
        out["p95_difference_c"] = {
            "min": float(diff.min()),
            "p50": float(np.percentile(diff, 50)),
            "p95": float(np.percentile(diff, 95)),
            "p99": float(np.percentile(diff, 99)),
            "max": float(diff.max()),
            "mean": float(diff.mean()),
        }
    for label, dn, mask in (
        ("before", old_dn, old_valid),
        ("after", new_dn, new_valid),
    ):
        if mask.any():
            celsius = decode(dn[mask])
            out[f"{label}_celsius"] = {
                "min": float(celsius.min()),
                "max": float(celsius.max()),
                "mean": float(celsius.mean()),
                "outside_trusted_range": int(
                    ((celsius < LST_VALID_MIN_C) | (celsius > LST_VALID_MAX_C)).sum()
                ),
            }
    return out


def write_images(out_dir: Path, old_dn, new_dn) -> None:
    """Three panels: before, after, and the difference in Celsius.

    Diagnostic only. It shows whether the changed pixels line up with WRS scene
    boundaries. The numbers in the JSON report are the measurement.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = (old_dn != LST_NODATA_DN) & (new_dn != LST_NODATA_DN)
    diff = np.where(both, decode(new_dn) - decode(old_dn), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    for ax, dn, title in (
        (axes[0], old_dn, "before: QA bits 3-4"),
        (axes[1], new_dn, "after: QA bits 1-5 + range"),
    ):
        celsius = np.where(dn != LST_NODATA_DN, decode(dn), np.nan)
        im = ax.imshow(celsius, cmap="inferno")
        ax.set_title(f"{title}\nP95 LST, C")
        fig.colorbar(im, ax=ax, shrink=0.8)
    limit = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title("after minus before, C")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    # 80 dpi keeps the panel under the 500 KB ceiling the repository
    # enforces on committed files, and the WRS-scale structure this is for
    # is still legible.
    fig.savefig(out_dir / "qa_difference.png", dpi=80)
    plt.close(fig)


def write_raster(path: Path, dn, bbox, crs: str) -> None:
    """One encoded raster of the window, as a GeoTIFF.

    This used to leave two `.npy` arrays, which carry no grid, no nodata and no
    decoding rule, so nothing but this script could read them back. A GeoTIFF
    carries all three and opens beside a published COG.
    """
    import rasterio
    from rasterio.transform import from_bounds

    height, width = dn.shape
    west, south, east, north = bbox
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=from_bounds(west, south, east, north, width, height),
        nodata=LST_NODATA_DN,
        tiled=True,
        compress="deflate",
    ) as dst:
        dst.scales = (LST_SCALE,)
        dst.offsets = (LST_OFFSET,)
        dst.write(dn, 1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 512 px square at 3,600 px/degree, the ground FINDINGS.md records this
    # comparison over. Holding the bbox holds the comparison comparable.
    p.add_argument("--bbox", default="-62.5,-32.642222,-62.357778,-32.5")
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="block edge in pixels. The default covers the window in one "
        "block, so the graph reduces it in a single task",
    )
    # Deliberately the old window. The masking change has to be measured with
    # every other input held fixed, the year window included.
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--cloud-cover-lt", type=int, default=100)
    p.add_argument("--platforms", default="landsat-8,landsat-9")
    p.add_argument("--source", default="earth-search")
    p.add_argument(
        "--max-scenes",
        type=int,
        default=120,
        help="cap, sampled evenly across the window's items; the spend control",
    )
    p.add_argument("--out", type=Path, default=Path("./qa-parity"))
    p.add_argument("--no-image", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import dask

    import composite
    from shard_lst_p95 import configure_read_env

    # The catalogue query lives in stac_reference now. This is a measurement
    # tool, so it describes what the old runtime did; the runtime itself reads
    # the precomputed inventory and opens no catalogue.
    from stac_reference import search_items

    configure_read_env(args.source)
    bbox = tuple(float(v) for v in args.bbox.split(","))
    resolution = 1.0 / args.pixels_per_degree
    height, width = composite.raster_shape(bbox, args.pixels_per_degree)

    items, _ = search_items(
        bbox,
        start=args.start,
        end=args.end,
        platforms=args.platforms,
        cloud_cover_lt=args.cloud_cover_lt,
        source=args.source,
    )
    if not items:
        raise SystemExit(f"no scene intersects {bbox}")
    n_intersecting = len(items)
    if args.max_scenes and n_intersecting > args.max_scenes:
        pick = np.linspace(0, n_intersecting - 1, args.max_scenes).round().astype(int)
        items = [items[i] for i in sorted(set(pick.tolist()))]
    item_dicts = [item.to_dict() for item in items]

    print(f"window        {height} x {width} px {bbox}")
    print(f"period        {args.start}/{args.end}  (held fixed across both passes)")
    print(f"scenes        {len(item_dicts)} loaded of {n_intersecting} intersecting")

    # The synchronous scheduler throughout: one process, one block, no
    # cluster, and a traceback that points at the kernel that raised.
    t0 = time.perf_counter()
    with dask.config.set(scheduler="sync"):
        stack = composite.open_stack(
            item_dicts,
            bbox,
            crs=args.crs,
            resolution=resolution,
            chunk=args.chunk,
        ).compute()
    dn = stack["lwir11"].values
    qa = stack["qa_pixel"].values
    print(f"loaded        {dn.shape} in {time.perf_counter() - t0:.1f}s")

    census = rejection_census(dn, qa)

    old_c, old_valid = legacy_masked_celsius(dn.copy(), qa)
    old_p95, old_counts = p95_and_counts(old_c, old_valid)
    old_dn = legacy_encode(old_p95)

    # The second pass is the pipeline itself, not a restatement of it. With no
    # prep artifact the graph composites the pooled percentile, which is what
    # the legacy pass computes too, so the mask is the only difference left.
    t1 = time.perf_counter()
    graph = composite.build_graph(
        item_dicts, bbox, crs=args.crs, resolution=resolution, chunk=args.chunk
    )
    with dask.config.set(scheduler="sync"):
        shipped = graph.compute()
    new_dn = shipped["lst_p95"].values
    new_counts = shipped["qa_count"].values.sum(axis=0, dtype="int32")
    print(f"graph         {new_dn.shape} in {time.perf_counter() - t1:.1f}s")

    report = {
        "inputs": {
            "bbox": list(bbox),
            "window_pixels": [height, width],
            "chunk": args.chunk,
            "crs": args.crs,
            "pixels_per_degree": args.pixels_per_degree,
            "resolution_deg": resolution,
            "start": args.start,
            "end": args.end,
            "platforms": args.platforms,
            "cloud_cover_lt": args.cloud_cover_lt,
            "source": args.source,
            "n_scenes_loaded": int(dn.shape[0]),
            "n_scenes_intersecting": n_intersecting,
        },
        "rejections": census,
        "valid_observations": {
            "before": int(old_valid.sum()),
            "after": int(new_counts.sum()),
            "removed": int(old_valid.sum() - new_counts.sum()),
            "observation_count_pixels_changed": int((old_counts != new_counts).sum()),
            "pixels_with_zero_observations_before": int((old_counts == 0).sum()),
            "pixels_with_zero_observations_after": int((new_counts == 0).sum()),
        },
        "output": compare(old_dn, new_dn),
    }
    # The encoded raster cannot tell a value the encoder refused from ground
    # nothing observed, and that distinction is what the nodata change is
    # about. So the two shipped kernels the graph calls run once more over the
    # stack already in memory, for this one count and nothing else.
    shipped_c, _ = masked_celsius(dn.copy(), qa)
    shipped_p95 = destripe.pooled_percentile(shipped_c)
    lst_finite = np.asarray(in_trusted_range(shipped_p95))
    report["output"]["p95_outside_trusted_range_before_encoding"] = int(
        (np.isfinite(shipped_p95) & ~lst_finite).sum()
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_raster(args.out / "lst_p95_before.tif", old_dn, bbox, args.crs)
    write_raster(args.out / "lst_p95_after.tif", new_dn, bbox, args.crs)
    (args.out / "qa_parity.json").write_text(json.dumps(report, indent=2))
    if not args.no_image:
        write_images(args.out, old_dn, new_dn)

    print(json.dumps(report, indent=2))
    print(f"artifacts     {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

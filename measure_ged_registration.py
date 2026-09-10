# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "numpy", "rasterio", "geopandas", "shapely", "pyogrio", "pyarrow>=16",
# ]
# ///
"""Check the ASTER GED mask against a composite this repository already built.

`FINDINGS.md` labels every figure MEASURED or DERIVED, and the ASTER GED
section is DERIVED. It says so in as many words: the L2SR product type follows
a static ancillary input, ASTER GED is the input the Collection 2 algorithm
reads, and a check against GED is the step that would make the claim measured.
This script is that step.

It needs no new data. `fulltile/` holds the finished S30W065 composite, 18,000
by 18,000 at 3,600 pixels per degree over `[-65, -35, -60, -30]`, with its
monthly observation counts beside it. Everything below reads those two arrays
and the NumObs artifact.

Three measurements, and each one can fail in a way that matters.

Registration. The mask's whole value rests on a cell landing where the
composite thinks it lands. The granule filename names its northwest corner, so
the placement is documented rather than guessed, but a documented convention
read wrongly produces a mask that is confidently one degree out. Shifting the
cell assignment by up to two cells on each axis and re-counting the agreement
turns that into a curve: right registration peaks sharply at zero and falls
away on both sides.

Coverage. Where GED holds no observation, USGS wrote `ST_B10` fill, and
`lst_qa.not_fill` rejected it before the percentile. So a gap pixel's monthly
counts must already sum to zero. That share is the artifact checked against the
archive itself, with no external oracle: the composite was built months before
this mask existed and knows nothing about it.

The tile-level claim. If the L2SR product type really follows ASTER coverage,
the tiles that hold nothing but L2SR should hold little or no land with
emissivity. This is the check `FINDINGS.md:987` names.

    uv run measure_ged_registration.py \\
        --raster fulltile/tile/lst_p95_dn.npy \\
        --tile S30W065 --out artifacts/ged_registration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aster_ged
import masks
from land_tiles import tile_bounds
from lst_qa import LST_NODATA_DN

#: How far the registration scan looks, in GED cells, on each axis.
SHIFT_CELLS = 2

#: The observation-count tiers the cross-tab reports. The top one is open.
TIERS = (0, 1, 2, 3)


def shifted_bbox(bbox, dy_cells: int, dx_cells: int):
    """The tile's bbox moved by whole GED cells, for the registration scan."""
    step = 1.0 / aster_ged.CELLS_PER_DEGREE
    west, south, east, north = bbox
    return (
        west + dx_cells * step,
        south - dy_cells * step,
        east + dx_cells * step,
        north - dy_cells * step,
    )


def registration_scan(bbox, shape, missing, numobs_uri) -> dict:
    """Agreement between missing pixels and NumObs zero, over a shift grid.

    Args:
        bbox: The tile's own bounds.
        shape: `(height, width)` of the composite.
        missing: Boolean array, True where the composite wrote nodata.
        numobs_uri: The NumObs artifact.

    Returns:
        The share at every shift, the best shift, and the share at zero. A
        peak anywhere but `(0, 0)` means the mosaic is misplaced.
    """
    import numpy as np

    n_missing = int(missing.sum())
    shares = {}
    for dy in range(-SHIFT_CELLS, SHIFT_CELLS + 1):
        for dx in range(-SHIFT_CELLS, SHIFT_CELLS + 1):
            counts = aster_ged.numobs_for_bbox(
                numobs_uri, shifted_bbox(bbox, dy, dx), shape
            )
            hit = int(np.count_nonzero(missing & (counts == masks.GAP_NUMOBS)))
            shares[f"{dy},{dx}"] = hit / n_missing if n_missing else 0.0
    best = max(shares, key=lambda key: shares[key])
    return {
        "missing_pixels": n_missing,
        "share_by_shift": shares,
        "best_shift": best,
        "share_at_zero": shares["0,0"],
        "registered": best == "0,0",
    }


def tier_crosstab(counts, lst, any_observation, land) -> list[dict]:
    """Every output pixel binned by the observation count of its GED cell.

    Args:
        counts: NumObs on the composite's grid.
        lst: The encoded composite.
        any_observation: True where the monthly counts sum above zero.
        land: The land mask, so a sea pixel is not read as a gap.

    Returns:
        One row per tier, over land only.
    """
    import numpy as np

    valid = lst != LST_NODATA_DN
    rows = []
    edges = [*[(t, counts == t) for t in TIERS], (TIERS[-1] + 1, counts > TIERS[-1])]
    for tier, selector in edges:
        sel = selector & land
        n = int(np.count_nonzero(sel))
        n_valid = int(np.count_nonzero(sel & valid))
        n_seen = int(np.count_nonzero(sel & any_observation))
        rows.append(
            {
                "numobs": tier if tier <= TIERS[-1] else f">{TIERS[-1]}",
                "land_pixels": n,
                "valid": n_valid,
                "missing": n - n_valid,
                "with_any_observation": n_seen,
                # For the gap tier this is the claim. USGS wrote fill there and
                # `not_fill` rejected it, so a gap pixel cannot have been
                # counted in any month. Anything but 0.0 means the mask and the
                # archive disagree about where the gaps are.
                "share_with_any_observation": (n_seen / n) if n else 0.0,
            }
        )
    return rows


def tile_level_claim(inventory_uri, land_tiles_uri, numobs_uri, land_geometry_uri):
    """Land with emissivity, for the tiles that hold only L2SR products.

    `thermal_href IS NULL` is exactly `data_type = 'OLI_TIRS_L2SR'`, and USGS
    writes that product where the surface temperature algorithm has no usable
    emissivity. If that is the mechanism, these tiles hold almost no land that
    ASTER GED covers.
    """
    import pyarrow.parquet as pq

    from land_tiles import read_land_tiles
    from tile_inventory import thermal_rows_for_tile

    import numpy as np

    tiles, _ = read_land_tiles(land_tiles_uri)
    pf = pq.ParquetFile(inventory_uri)
    cells = aster_ged.CELLS_PER_DEGREE
    rows = []
    for name in tiles:
        if thermal_rows_for_tile(pf, name):
            continue
        bbox = tile_bounds(name)
        shape = masks.raster_shape(bbox, cells)
        land = masks.land_mask(bbox, cells, land_geometry_uri)
        counts, covered = aster_ged.window_for_bbox(
            numobs_uri,
            bbox,
            shape,
            (aster_ged.NUMOBS_BAND, aster_ged.COVERAGE_BAND),
        )
        # Emissivity, not "not proven absent". `masks.output_mask` keeps a cell
        # with no granule, which is right for publishing and wrong for
        # measuring: it would count an unpublished cell as covered.
        read = covered > 0
        n_land = int(np.count_nonzero(land))
        n_read = int(np.count_nonzero(land & read))
        n_emis = int(np.count_nonzero(land & read & (counts > 0)))
        rows.append(
            {
                "tile_id": name,
                "land_cells": n_land,
                "land_cells_with_granule": n_read,
                "land_cells_with_emissivity": n_emis,
                "share_with_emissivity": (n_emis / n_land) if n_land else 0.0,
            }
        )
    rows.sort(key=lambda row: row["share_with_emissivity"], reverse=True)
    over = sum(1 for row in rows if row["share_with_emissivity"] > 0.01)
    ungranuled = sum(1 for row in rows if row["land_cells_with_granule"] == 0)
    return {
        "tiles_without_thermal": len(rows),
        "tiles_over_1pct_emissivity": over,
        # ASTER GED publishes a granule only where it holds data, so a tile
        # with none supports the claim by a different route from a tile whose
        # granules all read zero. The two are counted apart.
        "tiles_with_no_granule": ungranuled,
        "worst_twenty": rows[:20],
    }


#: Hot-tail threshold, in Celsius. A P95 of a hot season over land does not
#: reach it. Every pixel above it is a retrieval that failed upward.
HOT_C = 70.0


def temperature_census(counts, covered, lst, land) -> dict:
    """What the pixels in each observation tier actually say.

    The tier cross-tab counts pixels. This reads their temperatures, which is
    the reason the mask exists. A gap pixel that carries a plausible value
    costs nothing to keep. A gap pixel that carries 90 C is a failed retrieval
    wearing the shape of a measurement, and no consumer can tell.

    Returns:
        One row per tier with the percentiles and the hot-tail share, plus the
        tile-wide figures the enrichment is computed against.
    """
    import numpy as np

    from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE

    usable = (lst != LST_NODATA_DN) & land & (covered > 0)
    celsius = lst.astype("float64") * LST_SCALE + LST_OFFSET
    base = celsius[usable]
    hot_total = int((base >= HOT_C).sum())
    base_rate = hot_total / base.size if base.size else 0.0

    rows = []
    edges = [*[(t, counts == t) for t in TIERS], (TIERS[-1] + 1, counts > TIERS[-1])]
    for tier, selector in edges:
        sel = selector & usable
        n = int(np.count_nonzero(sel))
        if not n:
            rows.append({"numobs": tier, "valid": 0})
            continue
        values = celsius[sel]
        hot = int((values >= HOT_C).sum())
        rows.append(
            {
                "numobs": tier if tier <= TIERS[-1] else f">{TIERS[-1]}",
                "valid": n,
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "p999": float(np.percentile(values, 99.9)),
                "max": float(values.max()),
                "hot": hot,
                "hot_rate": hot / n,
                "hot_enrichment": (hot / n) / base_rate if base_rate else 0.0,
                "share_of_hot_tail": hot / hot_total if hot_total else 0.0,
            }
        )
    return {
        "hot_threshold_c": HOT_C,
        "valid": int(base.size),
        "hot": hot_total,
        "hot_rate": base_rate,
        "p50": float(np.percentile(base, 50)) if base.size else 0.0,
        "p99": float(np.percentile(base, 99)) if base.size else 0.0,
        "p999": float(np.percentile(base, 99.9)) if base.size else 0.0,
        "max": float(base.max()) if base.size else 0.0,
        "tiers": rows,
    }


def dilate_cells(flag, pixels_per_cell: int):
    """Widen a cell-grid boolean by one cell, then upsample to the tile grid.

    Dilating on the cell grid rather than the pixel grid is what makes "one
    cell" mean one cell. At 3,600 pixels per degree a cell is 36 pixels, so a
    pixel-grid dilation of one would widen the mask by a thirty-sixth of what
    the name says.
    """
    import numpy as np

    padded = np.pad(flag, 1, mode="edge")
    out = np.zeros_like(flag)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= padded[dy : dy + flag.shape[0], dx : dx + flag.shape[1]]
    return np.repeat(np.repeat(out, pixels_per_cell, axis=0), pixels_per_cell, axis=1)


def rule_table(numobs_uri, bbox, shape, lst, land, pixels_per_degree: int) -> list:
    """Four candidate rules, priced on the same tile.

    Each row is what the rule costs and what it buys: ordinary pixels removed
    against hot-tail pixels removed. The shipped rule is the first. The others
    are here because a threshold nobody priced is a threshold nobody chose.
    """
    import numpy as np

    from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE

    cells = aster_ged.CELLS_PER_DEGREE
    per_cell = pixels_per_degree // cells
    cell_shape = masks.raster_shape(bbox, cells)
    cell_counts, cell_covered = aster_ged.window_for_bbox(
        numobs_uri, bbox, cell_shape, (aster_ged.NUMOBS_BAND, aster_ged.COVERAGE_BAND)
    )
    read = cell_covered > 0

    celsius = lst.astype("float64") * LST_SCALE + LST_OFFSET
    valid = (lst != LST_NODATA_DN) & land
    hot = valid & (celsius >= HOT_C)
    n_valid = int(np.count_nonzero(valid))
    n_hot = int(np.count_nonzero(hot))

    rules = [
        ("numobs == 0", read & (cell_counts == 0), False),
        ("numobs == 0, 1-cell buffer", read & (cell_counts == 0), True),
        ("numobs <= 2", read & (cell_counts <= 2), False),
        ("numobs <= 2, 1-cell buffer", read & (cell_counts <= 2), True),
    ]
    rows = []
    for name, flag, buffer in rules:
        if buffer:
            drop = dilate_cells(flag, per_cell)
        else:
            drop = np.repeat(np.repeat(flag, per_cell, axis=0), per_cell, axis=1)
        removed = int(np.count_nonzero(valid & drop))
        hot_removed = int(np.count_nonzero(hot & drop))
        rows.append(
            {
                "rule": name,
                "valid_removed": removed,
                "valid_removed_share": removed / n_valid if n_valid else 0.0,
                "hot_removed": hot_removed,
                "hot_removed_share": hot_removed / n_hot if n_hot else 0.0,
                "hot_left": n_hot - hot_removed,
            }
        )
    return rows


def any_observation_mask(qa_path, shape):
    """True where the monthly counts sum above zero, read a month at a time.

    The array is 12 by 18,000 by 18,000, which is 3.9 GB. Memory-mapping it and
    reducing month by month holds one 324 MB plane at a time instead.
    """
    import numpy as np

    qa = np.load(qa_path, mmap_mode="r")
    seen = np.zeros(shape, dtype=bool)
    for month in range(qa.shape[0]):
        seen |= np.asarray(qa[month]) > 0
    return seen


def main(argv=None) -> int:  # noqa: C901 - one report, one branch per section
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raster", type=Path, default=Path("fulltile/tile/lst_p95_dn.npy"))
    p.add_argument("--qa", type=Path, default=Path("fulltile/tile/qa_count.npy"))
    p.add_argument("--tile", default="S30W065")
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--numobs-uri", type=Path, default=aster_ged.DEFAULT_NUMOBS_URI)
    p.add_argument(
        "--land-geometry-uri", type=Path, default=masks.DEFAULT_LAND_GEOMETRY_URI
    )
    p.add_argument("--inventory-uri", type=Path, default=None)
    p.add_argument(
        "--land-tiles-uri", type=Path, default=Path("artifacts/land_tiles.parquet")
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/ged_registration.json"))
    args = p.parse_args(argv)

    if not args.raster.exists():
        print(f"no composite at {args.raster}. Run a full tile first.")
        return 1

    bbox = tile_bounds(args.tile)
    lst = np.load(args.raster, mmap_mode="r")
    shape = tuple(lst.shape)
    expected = masks.raster_shape(bbox, args.pixels_per_degree)
    if shape != expected:
        print(
            f"{args.raster} is {shape}; {args.tile} at {args.pixels_per_degree} "
            f"px/deg is {expected}"
        )
        return 1
    lst = np.asarray(lst)
    print(f"composite     {args.tile} {shape[1]:,} x {shape[0]:,} from {args.raster}")

    missing = lst == LST_NODATA_DN
    counts, covered = aster_ged.window_for_bbox(
        args.numobs_uri,
        bbox,
        shape,
        (aster_ged.NUMOBS_BAND, aster_ged.COVERAGE_BAND),
    )
    land = masks.land_mask(bbox, args.pixels_per_degree, args.land_geometry_uri)

    report = {
        "tile": args.tile,
        "bbox": list(bbox),
        "raster": list(shape),
        "pixels_per_degree": args.pixels_per_degree,
        "source_raster": str(args.raster),
        "aster_ged": aster_ged.provenance(aster_ged.read_manifest(args.numobs_uri)),
        "land_pixels": int(np.count_nonzero(land)),
        "valid_pixels": int(np.count_nonzero(lst != LST_NODATA_DN)),
        "missing_pixels": int(np.count_nonzero(missing)),
    }

    report["registration"] = registration_scan(bbox, shape, missing, args.numobs_uri)
    reg = report["registration"]
    print(
        f"registration  best shift {reg['best_shift']} cells, "
        f"{reg['share_at_zero']:.2%} agreement at zero"
    )
    ordered = sorted(reg["share_by_shift"].items(), key=lambda kv: -kv[1])[:3]
    for shift, share in ordered:
        print(f"              ({shift}) {share:.2%}")

    if args.qa.exists():
        seen = any_observation_mask(args.qa, shape)
        report["tiers"] = tier_crosstab(counts, lst, seen, land)
        print("tiers         numobs  land px      valid    missing  any obs")
        for row in report["tiers"]:
            print(
                f"              {str(row['numobs']):>6}  {row['land_pixels']:>10,}  "
                f"{row['valid']:>9,}  {row['missing']:>9,}  "
                f"{row['share_with_any_observation']:.4%}"
            )
    else:
        print(f"tiers         skipped: no monthly counts at {args.qa}")

    report["temperatures"] = temperature_census(counts, covered, lst, land)
    census = report["temperatures"]
    print(
        f"tile          p50 {census['p50']:.1f} C  p99 {census['p99']:.1f} C  "
        f"p99.9 {census['p999']:.1f} C  max {census['max']:.1f} C"
    )
    print(
        f"              {census['hot']:,} px at or above {HOT_C:.0f} C "
        f"({census['hot_rate']:.6%} of valid)"
    )
    print(
        "temperatures  numobs        valid     p50     p99   p99.9     max   "
        ">=70C  enrich  %tail"
    )
    for row in census["tiers"]:
        if not row["valid"]:
            continue
        print(
            f"              {str(row['numobs']):>6} {row['valid']:>12,} "
            f"{row['p50']:>7.1f} {row['p99']:>7.1f} {row['p999']:>7.1f} "
            f"{row['max']:>7.1f} {row['hot']:>7,} {row['hot_enrichment']:>6.0f}x "
            f"{row['share_of_hot_tail']:>6.1%}"
        )

    report["rules"] = rule_table(
        args.numobs_uri, bbox, shape, lst, land, args.pixels_per_degree
    )
    print(
        "rules         rule                          valid removed        "
        "hot removed   hot left"
    )
    for row in report["rules"]:
        print(
            f"              {row['rule']:<28} {row['valid_removed']:>10,} "
            f"({row['valid_removed_share']:>7.4%})  {row['hot_removed']:>6,} "
            f"({row['hot_removed_share']:>6.2%})  {row['hot_left']:>8,}"
        )

    if args.inventory_uri:
        report["tile_level"] = tile_level_claim(
            args.inventory_uri,
            args.land_tiles_uri,
            args.numobs_uri,
            args.land_geometry_uri,
        )
        level = report["tile_level"]
        print(
            f"tile level    {level['tiles_over_1pct_emissivity']} of "
            f"{level['tiles_without_thermal']} L2SR-only tiles hold more than "
            f"1% land with emissivity"
        )
        print(
            f"              {level['tiles_with_no_granule']} of them have no "
            f"ASTER GED granule over land at all"
        )
    else:
        print("tile level    skipped: pass --inventory-uri to check the 126 tiles")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"written       {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

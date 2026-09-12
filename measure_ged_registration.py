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

It needs no new data. A finished S30W065 run publishes the two COGs this
reads, `lst_p95.tif` and `qa_count.tif`, 18,000 by 18,000 at 3,600 pixels per
degree over `[-65, -35, -60, -30]`. Everything below reads those two rasters
and the NumObs artifact.

Three measurements, and each one can fail in a way that matters.

Registration. The mask's whole value rests on a cell landing where the
composite thinks it lands. The granule filename names its northwest corner, so
the placement is documented rather than guessed, but a documented convention
read wrongly produces a mask that is confidently one degree out. Shifting the
cell assignment by up to two cells on each axis and re-counting the agreement
turns that into a curve: right registration peaks sharply at zero and falls
away on both sides.

Coverage. A gap cell is not a hole. USGS interpolates GED emissivity from the
neighbouring cells and retrieves a temperature anyway, so most gap pixels carry
a value: MEASURED at 89.53% on S30W065. The hard holes are the minority, and
they are what the registration scan lands on. A share near zero would mean the
gap really is a hole, and the tile says otherwise. This is the artifact checked
against the archive itself, with no external oracle: the composite was built
months before this mask existed and knows nothing about it.

The tile-level claim. If the L2SR product type really follows ASTER coverage,
the tiles that hold nothing but L2SR should hold little or no land with
emissivity. This is the check `FINDINGS.md:987` names.

    uv run measure_ged_registration.py \\
        --raster run/lst_p95.tif --qa run/qa_count.tif \\
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

#: Where a `--no-catalog` run leaves the two COGs. A catalog run puts them
#: under `<out-dir>/catalog/<collection>/<tile>/`, so point `--raster` and
#: `--qa` there instead.
DEFAULT_RASTER = Path("run/lst_p95.tif")
DEFAULT_QA = Path("run/qa_count.tif")

#: Rows per window when the temperature band is read. A whole-band read holds
#: the decoded tile twice, once inside GDAL and once in the array it hands
#: back; a strip at a time holds it once and 66 MB besides.
WINDOW_ROWS = 1024

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
                # For the gap tier this is the number that killed the
                # whole-cell rule. USGS interpolates emissivity across a gap
                # cell rather than leaving it empty, so most gap pixels do
                # carry a retrieval: 0.8953 on S30W065. Watch it for drift, not
                # for zero. A share near zero would mean the gap is a hard hole
                # and the geometry alone would remove almost nothing.
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
    over = sum(1 for row in rows if float(row["share_with_emissivity"]) > 0.01)
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


#: Hot-tail threshold, in Celsius. A screen, not a physical ceiling.
#:
#: It marks where this tile's artifact population separates, and one tile is
#: all that calibrated it. It carries no claim about the hottest land surface,
#: because it never acts alone: a pixel above it outside the gap region
#: survives, and 503 such pixels do survive on S30W065. `masks.py` owns the
#: value the fleet applies, and `nlebovits/landsat-lst` calibrated the same
#: number the same way in `config.py:193-210`.
#:
#: Check the tail against 70 C on the next tile that carries a substantial gap
#: population.
HOT_C = masks.GAP_HOT_THRESHOLD_C


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


#: The rule `masks.py` applies. Named here so the table marks which row ships
#: rather than leaving a reader to infer it from the order.
SHIPPED_RULE = "numobs == 0, 1-cell buffer AND >= 70 C"


def _cell_region(numobs_uri, bbox, pixels_per_degree: int, pad: int):
    """The GED counts and coverage for a tile, padded, on the cell grid.

    The same reader the mask uses, so the table prices the rule that ships
    rather than a second implementation of it. The padding matters: a gap cell
    just outside the tile buffers into it, and an earlier pass that clipped the
    dilation at the tile edge was two cells short on S30W065.
    """
    return aster_ged.cell_window_for_bbox(
        numobs_uri,
        bbox,
        pad_cells=pad,
        bands=(aster_ged.NUMOBS_BAND, aster_ged.COVERAGE_BAND),
    )


def gap_cell_census(numobs_uri, bbox, lst, land, pixels_per_degree: int) -> dict:
    """How the hot pixels sit inside the gap cells, cell by cell.

    This is the measurement that chooses the rule. A whole-cell rule is only
    defensible if a gap cell is bad as a cell. It is not: most gap cells hold
    no hot pixel at all, and the ones that do hold a thin scatter rather than a
    bad block. That is why the shipped rule is a pair and not the geometry.
    """
    import numpy as np

    from lst_qa import LST_NODATA_DN

    per_cell = aster_ged.cells_per_pixel_block(pixels_per_degree)
    counts, covered = _cell_region(numobs_uri, bbox, pixels_per_degree, 0)
    gap = (covered > 0) & (counts == 0)

    hot_px = (lst >= masks.gap_hot_dn()) & (lst != LST_NODATA_DN) & land
    valid_px = (lst != LST_NODATA_DN) & land
    rows, cols = gap.shape

    def per_cell_sum(flag):
        return flag.reshape(rows, per_cell, cols, per_cell).sum(
            axis=(1, 3), dtype="int64"
        )

    hot_c = per_cell_sum(hot_px)
    valid_c = per_cell_sum(valid_px)

    with_hot = gap & (hot_c > 0)
    n_gap = int(np.count_nonzero(gap))
    n_with = int(np.count_nonzero(with_hot))
    valid_in_with = int(valid_c[with_hot].sum())
    hot_in_with = int(hot_c[with_hot].sum())
    return {
        "pixels_per_cell": per_cell * per_cell,
        "gap_cells": n_gap,
        "gap_cells_with_a_hot_pixel": n_with,
        "gap_cells_with_none": n_gap - n_with,
        "share_of_gap_cells_with_none": (n_gap - n_with) / n_gap if n_gap else 0.0,
        "valid_pixels_in_gap_cells": int(valid_c[gap].sum()),
        "hot_pixels_in_gap_cells": int(hot_c[gap].sum()),
        # Inside a cell that does hold a hot pixel, what share of the cell is
        # hot. A whole-cell rule removes the other share for nothing.
        "hot_share_of_cells_with_a_hot_pixel": (
            hot_in_with / valid_in_with if valid_in_with else 0.0
        ),
        "worst_cell_hot_pixels": int(hot_c[gap].max()) if n_gap else 0,
    }


def rule_table(numobs_uri, bbox, shape, lst, land, pixels_per_degree: int) -> list:
    """Six candidate rules, priced on the same tile.

    Each row is what the rule costs and what it buys: ordinary pixels removed
    against hot-tail pixels removed. The first four vary the geometry alone.
    The last two intersect the geometry with the temperature, which is what
    `masks.py` applies, and `SHIPPED_RULE` marks the one that ships.

    The first four are here because a threshold nobody priced is a threshold
    nobody chose, and because the geometry alone was shipped once, in this
    project and in `nlebovits/landsat-lst`, on the strength of the hot column
    read without the column beside it.
    """
    import numpy as np

    from lst_qa import LST_NODATA_DN

    pad = 1
    cell_counts, cell_covered = _cell_region(numobs_uri, bbox, pixels_per_degree, pad)
    read = cell_covered > 0

    hot_dn = masks.gap_hot_dn()
    valid = (lst != LST_NODATA_DN) & land
    hot = valid & (lst >= hot_dn)
    n_valid = int(np.count_nonzero(valid))
    n_hot = int(np.count_nonzero(hot))

    def to_pixels(cells):
        return aster_ged.cells_to_pixels(cells[pad:-pad, pad:-pad], lst.shape)

    specs = [
        ("numobs == 0", 0, 0, False),
        ("numobs == 0, 1-cell buffer", 0, 1, False),
        ("numobs <= 2", 2, 0, False),
        ("numobs <= 2, 1-cell buffer", 2, 1, False),
        ("numobs == 0 AND >= 70 C", 0, 0, True),
        (SHIPPED_RULE, 0, 1, True),
    ]
    rows = []
    for name, tier, buffer_cells, conjoin in specs:
        cells = read & (cell_counts <= tier)
        drop = to_pixels(aster_ged.dilate_cells(cells, buffer_cells))
        if conjoin:
            drop &= lst >= hot_dn
        removed = int(np.count_nonzero(valid & drop))
        hot_removed = int(np.count_nonzero(hot & drop))
        del drop
        rows.append(
            {
                "rule": name,
                "shipped": name == SHIPPED_RULE,
                "valid_removed": removed,
                "valid_removed_share": removed / n_valid if n_valid else 0.0,
                "hot_removed": hot_removed,
                "hot_removed_share": hot_removed / n_hot if n_hot else 0.0,
                "hot_left": n_hot - hot_removed,
            }
        )
    return rows


def raster_size(path) -> tuple[int, int]:
    """`(height, width)` from a raster's header, without decoding a pixel."""
    import rasterio

    with rasterio.open(path) as src:
        return (src.height, src.width)


def read_temperature(path):
    """The encoded composite, read window by window out of its COG.

    `src.read(1)` would decode the whole band into a second array on top of
    the one it returns. Filling one array a strip at a time holds the tile
    once, which is what the memory map this replaced did.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        out = np.empty((src.height, src.width), dtype=src.dtypes[0])
        for row in range(0, src.height, WINDOW_ROWS):
            rows = min(WINDOW_ROWS, src.height - row)
            window = Window.from_slices(slice(row, row + rows), slice(0, src.width))
            out[row : row + rows] = src.read(1, window=window)
    return out


def any_observation_mask(qa_path, shape):
    """True where the monthly counts sum above zero, read a band at a time.

    The asset is 12 uint8 bands of 18,000 by 18,000, which is 3.9 GB. Reducing
    one band at a time holds one 324 MB plane, the way the memory map did.
    """
    import numpy as np
    import rasterio

    seen = np.zeros(shape, dtype=bool)
    with rasterio.open(qa_path) as src:
        for band in range(1, src.count + 1):
            seen |= src.read(band) > 0
    return seen


def main(argv=None) -> int:  # noqa: C901 - one report, one branch per section
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--raster",
        type=Path,
        default=DEFAULT_RASTER,
        help="the published lst_p95.tif of a finished run",
    )
    p.add_argument(
        "--qa",
        type=Path,
        default=DEFAULT_QA,
        help="the published qa_count.tif beside it, 12 uint8 bands",
    )
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
    # The header first, so a raster of the wrong shape costs a header read
    # rather than a full decode of the tile.
    shape = raster_size(args.raster)
    expected = masks.raster_shape(bbox, args.pixels_per_degree)
    if shape != expected:
        print(
            f"{args.raster} is {shape}; {args.tile} at {args.pixels_per_degree} "
            f"px/deg is {expected}"
        )
        return 1
    lst = read_temperature(args.raster)
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

    report["gap_cells"] = gap_cell_census(
        args.numobs_uri, bbox, lst, land, args.pixels_per_degree
    )
    cells = report["gap_cells"]
    print(
        f"gap cells     {cells['gap_cells']:,} cells of "
        f"{cells['pixels_per_cell']:,} px, "
        f"{cells['gap_cells_with_none']:,} of them "
        f"({cells['share_of_gap_cells_with_none']:.1%}) hold no hot pixel"
    )
    print(
        f"              in the {cells['gap_cells_with_a_hot_pixel']:,} that do, "
        f"hot is {cells['hot_share_of_cells_with_a_hot_pixel']:.2%} of the "
        f"cell; worst holds {cells['worst_cell_hot_pixels']:,}"
    )
    print("              the cell is not bad, so the rule is not the cell")

    report["rules"] = rule_table(
        args.numobs_uri, bbox, shape, lst, land, args.pixels_per_degree
    )
    print(
        "rules         rule                                    valid removed  "
        "      hot removed   hot left"
    )
    for row in report["rules"]:
        mark = "*" if row["shipped"] else " "
        print(
            f"            {mark} {row['rule']:<38} {row['valid_removed']:>10,} "
            f"({row['valid_removed_share']:>7.4%})  {row['hot_removed']:>6,} "
            f"({row['hot_removed_share']:>6.2%})  {row['hot_left']:>8,}"
        )
    print("              * the rule masks.py applies")

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

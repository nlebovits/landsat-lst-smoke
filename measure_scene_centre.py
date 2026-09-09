# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "duckdb>=1.0", "pyarrow>=16", "numpy", "pystac-client",
# ]
# ///
"""How far the computed scene centre sits from the one Earth Search publishes.

`usgs_inventory` has one tolerance. It computes the acquisition centre as the
midpoint of the bulk file's `Start Time` and `Stop Time`, while Earth Search
publishes the scene centre from the product metadata. `MONTH_BOUNDARY_GUARD_
SECONDS` is sized from the gap between them, so the gap has to be measured
rather than argued.

An earlier version of this repository argued it instead, and got two things
wrong. It said the bulk file truncates start and stop to whole seconds. Most of
it does, but not all:

    L8 2021        231,144 scenes    0% whole-second
    L8 2022        234,181 scenes   53% whole-second
    L8 2023-2025 and all L9         100% whole-second

It then quoted a 1.117 s residual, which no truncation argument allows:
truncating both timestamps moves their midpoint by strictly under a second.
Measuring the two regimes apart resolves it. Truncation contributes under a
second, as the argument says, and the rest is a separate constant offset that
the earlier figure did not distinguish.

Landsat 8 2021 is what makes the separation possible. It carries microseconds
on both timestamps, so the midpoint is exact and any residual against Earth
Search is the definitional difference alone. `--platform landsat-8 --year 2021`
is the default for that reason.

Run it with `--all-years` to see both regimes in one sample, which is what the
build faces.

    uv run measure_scene_centre.py --out artifacts/scene_centre_offset.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from stac_window import DEFAULT_CLOUD_COVER_LT

#: How many scenes to compare. Earth Search takes 50 ids per request, so this
#: is 8 requests at the default.
DEFAULT_SAMPLE = 400

#: Landsat platform number to the STAC `platform` string, as in usgs_inventory.
SATELLITE_BY_PLATFORM = {"landsat-8": 8, "landsat-9": 9}


def sample_scenes(
    parquet_path: Path | str,
    *,
    platform: str = "landsat-8",
    year: int | None = 2021,
    sample: int = DEFAULT_SAMPLE,
    seed: int = 0,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
) -> list[dict]:
    """Scenes with their start, stop, and computed centre.

    Sampled across the whole selection rather than the first rows, because the
    bulk file is written in acquisition order and the leading rows share a
    path, a row, and a day.

    Returns:
        One dict per scene, carrying the display id, the STAC item id, the two
        timestamps, and whether both carry sub-second precision.
    """
    import duckdb

    from usgs_inventory import stac_item_id

    sat = SATELLITE_BY_PLATFORM[platform]
    year_clause = (
        f"AND substr(\"Date Acquired\", 1, 4) = '{year}'" if year is not None else ""
    )
    sql = f"""
    SELECT "Display ID", "Start Time", "Stop Time"
    FROM read_parquet(?)
    WHERE "Satellite" = {sat}
      AND "Scene Cloud Cover L1" < {cloud_cover_lt}
      {year_clause}
    """
    con = duckdb.connect()
    try:
        rows = con.execute(sql, [str(parquet_path)]).fetchall()
    finally:
        con.close()

    if not rows:
        msg = f"no {platform} scenes in {year} in {parquet_path}"
        raise ValueError(msg)

    rng = random.Random(seed)
    picked = rng.sample(rows, min(sample, len(rows)))
    out = []
    for display, start, stop in picked:
        out.append(
            {
                "display_id": display,
                "item_id": stac_item_id(display),
                "start": start,
                "stop": stop,
                "sub_second": "." in start and "." in stop,
            }
        )
    return out


def compare(scenes: list[dict], *, source: str = "earth-search") -> dict:
    """Compare each computed centre against the catalogue's published one.

    Returns:
        The per-scene gaps in seconds and the summary the guard is sized from.
    """
    import numpy as np

    from stac_reference import fetch_items_by_id

    found = fetch_items_by_id([s["item_id"] for s in scenes], source=source)

    gaps = []
    for scene in scenes:
        item = found.get(scene["item_id"])
        if item is None:
            continue
        start = np.datetime64(scene["start"].replace(" ", "T"), "us")
        stop = np.datetime64(scene["stop"].replace(" ", "T"), "us")
        centre = start + (stop - start) // 2
        published = np.datetime64(
            item["properties"]["datetime"].replace("Z", "").rstrip(), "us"
        )
        gap = float((centre - published).astype("timedelta64[us]").astype("int64"))
        gaps.append(
            {
                "item_id": scene["item_id"],
                "sub_second": scene["sub_second"],
                "gap_seconds": gap / 1e6,
                "duration_seconds": float(
                    (stop - start).astype("timedelta64[us]").astype("int64")
                )
                / 1e6,
            }
        )

    if not gaps:
        msg = "Earth Search returned none of the sampled items"
        raise RuntimeError(msg)

    values = np.array([g["gap_seconds"] for g in gaps])
    exact = np.array([g["sub_second"] for g in gaps])
    summary = {
        "compared": len(gaps),
        "not_found": len(scenes) - len(gaps),
        "worst_abs_seconds": float(np.abs(values).max()),
        "median_abs_seconds": float(np.median(np.abs(values))),
        "mean_signed_seconds": float(values.mean()),
        "sub_second_rows": int(exact.sum()),
    }
    if exact.any():
        # The number the guard is sized from. No truncation in these rows, so
        # the residual is the definitional difference on its own.
        summary["worst_abs_seconds_sub_second_rows"] = float(
            np.abs(values[exact]).max()
        )
    if (~exact).any():
        summary["worst_abs_seconds_whole_second_rows"] = float(
            np.abs(values[~exact]).max()
        )
    return {"summary": summary, "gaps": gaps}


def _bulk_path(cache_dir: Path | str) -> Path:
    """The newest cached bulk download.

    Newest rather than first alphabetically: the cache name carries a digest of
    the publication, not a date, so sorting by name picks an arbitrary one and
    a stale file would silently answer for the current build.
    """
    found = sorted(
        Path(cache_dir).glob("LANDSAT_OT_C2_L2-*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        msg = f"no cached USGS bulk file in {cache_dir}. Run usgs_inventory.py first."
        raise FileNotFoundError(msg)
    return found[-1]


def main(argv=None) -> int:
    from usgs_inventory import DEFAULT_CACHE_DIR, MONTH_BOUNDARY_GUARD_SECONDS

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--platform", default="landsat-8", choices=sorted(SATELLITE_BY_PLATFORM)
    )
    p.add_argument(
        "--year",
        type=int,
        default=2021,
        help="restrict to one year. 2021 on landsat-8 is the sub-second block",
    )
    p.add_argument("--all-years", action="store_true", help="ignore --year")
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    bulk = _bulk_path(args.cache_dir)
    scenes = sample_scenes(
        bulk,
        platform=args.platform,
        year=None if args.all_years else args.year,
        sample=args.sample,
        seed=args.seed,
    )
    result = compare(scenes)
    s = result["summary"]

    print(f"source        {bulk.name}")
    print(
        f"selection     {args.platform}, {'all years' if args.all_years else args.year}"
    )
    print(
        f"compared      {s['compared']} scenes, {s['not_found']} not in the catalogue"
    )
    print(f"sub-second    {s['sub_second_rows']} of {s['compared']} rows")
    print(f"worst gap     {s['worst_abs_seconds']:.6f} s")
    print(f"median gap    {s['median_abs_seconds']:.6f} s")
    print(f"mean signed   {s['mean_signed_seconds']:+.6f} s")
    if "worst_abs_seconds_sub_second_rows" in s:
        print(
            f"  no rounding {s['worst_abs_seconds_sub_second_rows']:.6f} s "
            f"(sub-second rows, the definitional gap alone)"
        )
    if "worst_abs_seconds_whole_second_rows" in s:
        print(
            f"  truncated   {s['worst_abs_seconds_whole_second_rows']:.6f} s "
            f"(whole-second rows)"
        )
    margin = MONTH_BOUNDARY_GUARD_SECONDS / max(s["worst_abs_seconds"], 1e-9)
    print(
        f"guard         {MONTH_BOUNDARY_GUARD_SECONDS} s, "
        f"{margin:.0f}x the worst gap measured here"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"written       {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

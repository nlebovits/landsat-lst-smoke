# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["geopandas", "shapely", "pyogrio", "pandas"]
# ///
"""Cut the committed land-geometry fixtures out of the full geometries.

`land_tiles.py --write-geometry` writes 16 MB of buffered Natural Earth, which
the large-file hook refuses and which no repository wants in its history. The
inventory has the same shape of problem and the same answer: the full artifact
is gitignored and a slice of real rows is committed beside it.

Two geometries, two slices. The buffered one is the processing mask a run
writes through. The unbuffered one, cut by `--strict`, is what a published
`lst:land_pixels` counts, and the two differ by up to 25 km of sea off every
coastline.

The slices are clipped to `conftest.LAND_SLICE_TILES`, which is the four tiles
`tests/make_slice.py` cuts plus `S40W065`. Clipping rather than selecting is
what makes them small: one South American polygon spans a continent, and
`masks.land_mask` only ever rasterises inside one tile's bbox, so the clipped
geometry produces the same mask there. `tests/test_masks.py` asserts that
rather than assuming it.

A slice cannot answer the checksum tie. `land_geometry_sha256` digests the bytes
of the full file, so the test that compares it against `land_tiles.parquet`
needs the real artifact and skips without it.

    uv run tests/make_land_slice.py
    uv run tests/make_land_slice.py --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from land_tiles import tile_bounds  # noqa: E402

#: The tiles both geometry slices hold. Kept in step with
#: `tests/conftest.LAND_SLICE_TILES`, which names the same five.
SLICE_TILES = ("N05E010", "N40W075", "S15E175", "S30W065", "S40W065")


def build(source: Path, out: Path, tiles=SLICE_TILES) -> Path:
    """Write the buffered geometry clipped to `tiles`."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    land = gpd.read_file(source)
    parts = []
    for name in tiles:
        cell = box(*tile_bounds(name))
        hit = land.iloc[land.sindex.query(cell, predicate="intersects")]
        clipped = hit.geometry.intersection(cell)
        parts.append(
            gpd.GeoDataFrame(geometry=clipped[~clipped.is_empty], crs=land.crs)
        )
    sliced = gpd.GeoDataFrame(pd.concat(parts).reset_index(drop=True), crs=land.crs)
    out.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_file(out, driver="GPKG")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--strict",
        action="store_true",
        help="cut the unbuffered geometry instead of the buffered one. This is "
        "the land a published coverage property divides by",
    )
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    stem = "land_strict" if args.strict else "land_buffered"
    flag = "--write-strict-geometry" if args.strict else "--write-geometry"
    source = args.source or ROOT / "artifacts" / f"{stem}.gpkg"
    out = args.out or ROOT / "artifacts" / f"{stem}_slice.gpkg"

    if not source.exists():
        print(
            f"no geometry at {source}. Write it with:\n"
            f"  uv run land_tiles.py --out artifacts/land_tiles.parquet "
            f"{flag} {source}"
        )
        return 1
    written = build(source, out)
    print(f"written       {written} ({written.stat().st_size / 1024:.0f} KB)")
    print(f"tiles         {', '.join(SLICE_TILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

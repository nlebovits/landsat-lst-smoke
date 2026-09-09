# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["pyarrow>=16"]
# ///
"""Cut a few real tiles out of the full inventory, for the test suite.

The full artifact is 167 MB and gitignored, so the tests that prove the runtime
reads it offline used to skip on every clean checkout. This writes a slice
small enough to commit and real enough to be worth testing against: real hrefs,
real geoboxes, real acquisition times, and the column types the production
writer forces rather than the ones `from_pylist` would infer.

It preserves the layout the reader depends on. One row group per tile, sorted
by tile then time, with the manifest embedded under the same key. The manifest
is copied from the source and its counts rewritten to describe the slice, so a
test cannot read a tile count of 895 from a file holding four tiles.

    uv run tests/make_slice.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import FULL_ARTIFACT, SLICE_ARTIFACT, SLICE_TILES  # noqa: E402

#: Rows to keep per tile. Enough that `items_for_shard` has something to filter
#: and the offline tests can assert a realistic count, small enough to commit.
ROWS_PER_TILE = 60


def build_slice(
    source: Path,
    out: Path,
    tiles=SLICE_TILES,
    rows_per_tile: int = ROWS_PER_TILE,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tile_inventory import read_manifest, row_groups_for_tile

    pf = pq.ParquetFile(source)
    manifest = read_manifest(source)

    chunks = []
    for name in sorted(tiles):
        groups = row_groups_for_tile(pf, name)
        if not groups:
            msg = f"tile {name} is absent from {source}"
            raise SystemExit(msg)
        table = pf.read_row_groups(groups)
        # A boolean mask rather than `pyarrow.compute.equal`, which is
        # generated at import time and so has no static signature to check.
        # `Table.filter` keeps the column types the writer chose, which is the
        # reason for slicing a real artifact instead of building rows by hand.
        mask = pa.array([v == name for v in table.column("tile_id").to_pylist()])
        chunks.append(table.filter(mask).slice(0, rows_per_tile))

    manifest = manifest | {
        "tile_count": len(chunks),
        "tiles_with_scenes": len(chunks),
        "tile_scene_rows": sum(c.num_rows for c in chunks),
        "row_groups": len(chunks),
        "slice_of": str(source.name),
        "slice_note": (
            "a few tiles cut from a full build for the test suite. The window "
            "and land fields are the source's own, so manifest checks behave "
            "exactly as they do against the full artifact."
        ),
    }
    meta = {b"manifest": json.dumps(manifest, indent=2).encode()}
    schema = chunks[0].schema.with_metadata(meta)

    out.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        out,
        schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=["tile_id", "platform", "collection_category", "data_type"],
        write_statistics=True,
    ) as writer:
        for chunk in chunks:
            writer.write_table(
                chunk.replace_schema_metadata(meta), row_group_size=chunk.num_rows
            )
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=FULL_ARTIFACT)
    p.add_argument("--out", type=Path, default=SLICE_ARTIFACT)
    p.add_argument("--rows-per-tile", type=int, default=ROWS_PER_TILE)
    args = p.parse_args(argv)

    if not args.source.exists():
        msg = f"no inventory at {args.source}. Build it with usgs_inventory.py."
        raise SystemExit(msg)

    manifest = build_slice(args.source, args.out, rows_per_tile=args.rows_per_tile)
    print(f"tiles         {manifest['tile_count']}: {', '.join(sorted(SLICE_TILES))}")
    print(f"rows          {manifest['tile_scene_rows']}")
    print(f"written       {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

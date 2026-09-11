# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["frisky>=0.7.2", "dask", "pyarrow>=16", "duckdb>=1.0", "psutil"]
# ///
"""What one `client.submit` costs, against the payload it carries.

A full tile spent about 373.7 s submitting 1,296 shards
(`fulltile/part*/slice.log`), and the working explanation was that pickling a
shard's 509 STAC item dicts is expensive. It is not: `pickle.dumps` of that list
takes 0.9 ms, so 1,296 of them is 1.2 s, or 0.3% of the phase. The cost is the
payload moving through the client, the scheduler, and the worker, once for every
shard that shares the same scenes.

This measures the three shapes the pipeline could submit, against a real
cluster, so the claim rests on a number rather than on an argument:

* a list of item dicts, which is what the pipeline did
* a scattered table and a list of positions
* a table path and a list of positions, which is what it does now

It reads no object and opens no bucket. The item dicts come from the committed
inventory slice and are multiplied to full-tile depth by rewriting their ids, so
that pickle's memo cannot dedup what a real tile would not.

    uv run measure_submit_cost.py --out artifacts/submit_cost.json

Cost: nothing. A laptop, a few minutes, no S3.

The laptop's figure is smaller than the instance's for the same reason every
figure here is: the instance carries a deeper shard and 64 workers asking one
scheduler for work. The ratio between the rows is the result, not the absolute
seconds.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

#: Scenes one shard reads, at the p50 the full-tile run measured at 512 px.
DEFAULT_SCENES = 509

#: Scenes in the whole tile's table, as that run's `scenes` line reported.
DEFAULT_TABLE = 3910

#: Shards in a full tile at 512 px, for the extrapolation column.
FULL_TILE_SHARDS = 1296


def take_dicts(payload):
    """The shape the pipeline used to submit."""
    return len(payload)


def take_scattered(table, indices):
    """The shape `client.scatter` would give."""
    return len([table[i] for i in indices])


def take_path(path, indices):
    """The shape the pipeline submits now."""
    import item_table

    return len(item_table.select(path, indices))


def grown_table(n: int, inventory, tile: str) -> list[dict]:
    """`n` distinct item dicts, built from the committed slice.

    The ids and hrefs are rewritten per copy. A plain `deepcopy` shares the
    immutable strings and lets pickle's memo collapse them, which understates
    the payload of a real tile by about 40%.
    """
    from tile_inventory import items_for_tile

    from land_tiles import tile_bounds

    items, _ = items_for_tile(inventory, tile, bounds=tile_bounds(tile))
    if not items:
        raise SystemExit(f"{inventory} holds no scenes for {tile}")
    out: list[dict] = []
    while len(out) < n:
        for item in items:
            grown = copy.deepcopy(item)
            grown["id"] = f"{grown['id']}-{len(out)}"
            for band, asset in grown.get("assets", {}).items():
                asset["href"] = f"{asset['href']}-{band}-{len(out)}"
            out.append(grown)
            if len(out) == n:
                break
    return out


def time_submits(client, make_call, reps: int, expected: int) -> float:
    """Mean seconds inside `client.submit`, with every result checked."""
    futures = []
    t0 = time.perf_counter()
    for _ in range(reps):
        futures.append(client.submit(*make_call()))
    per_call = (time.perf_counter() - t0) / reps
    for future in futures:
        got = future.result()
        if got != expected:
            msg = f"a task returned {got}, not {expected}"
            raise SystemExit(msg)
    return per_call


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inventory-uri",
        type=Path,
        default=ROOT / "artifacts" / "inventory_slice.parquet",
        help="the committed slice is enough; this reads no object",
    )
    p.add_argument("--tile", default="S30W065")
    p.add_argument(
        "--scenes",
        type=int,
        default=DEFAULT_SCENES,
        help="scenes one shard reads. The full tile measured a p50 of 509",
    )
    p.add_argument(
        "--table",
        type=int,
        default=DEFAULT_TABLE,
        help="scenes in the tile's table. The full tile held 3,910",
    )
    p.add_argument("--reps", type=int, default=40, help="submits per shape")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    import frisky

    import item_table

    table = grown_table(args.table, args.inventory_uri, args.tile)
    indices = list(range(min(args.scenes, len(table))))
    subset = [table[i] for i in indices]
    stage = Path(args.out.parent if args.out else ROOT) / "submit-cost-table.json"
    report = item_table.write(stage, table)
    print(
        f"table         {report['n_items']:,} scenes, "
        f"{report['bytes'] / 1e6:.2f} MB, written in {report['seconds']:.2f}s"
    )
    print(f"shard payload {len(subset):,} scenes\n")

    cluster = frisky.LocalCluster(
        n_workers=args.workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    client = cluster.get_client()
    rows = []
    try:
        rows.append(
            (
                "item dicts in the task",
                time_submits(
                    client, lambda: (take_dicts, subset), args.reps, len(indices)
                ),
            )
        )
        t0 = time.perf_counter()
        # `scatter` on a list scatters each element, so the table is wrapped to
        # travel as one key. `broadcast=True` raises NotImplementedError here.
        scattered = client.scatter([table])[0]
        scatter_s = time.perf_counter() - t0
        rows.append(
            (
                "scattered table and indices",
                time_submits(
                    client,
                    lambda: (take_scattered, scattered, indices),
                    args.reps,
                    len(indices),
                ),
            )
        )
        rows.append(
            (
                "table path and indices",
                time_submits(
                    client,
                    lambda: (take_path, report["path"], indices),
                    args.reps,
                    len(indices),
                ),
            )
        )
    finally:
        cluster.close()
        stage.unlink(missing_ok=True)

    print(f"{'submit payload':<30}{'per submit':>13}{'x 1,296 shards':>18}")
    for label, per in rows:
        print(f"{label:<30}{per * 1000:10.3f} ms{per * FULL_TILE_SHARDS:15.1f} s")
    baseline = rows[0][1]
    print(
        f"\nthe pipeline's shape costs {baseline / rows[-1][1]:.0f}x what it "
        f"submits now, and scatter takes {scatter_s * 1000:.0f} ms once"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "tile": args.tile,
                    "scenes_per_shard": len(indices),
                    "scenes_in_table": report["n_items"],
                    "table_bytes": report["bytes"],
                    "workers": args.workers,
                    "reps": args.reps,
                    "scatter_s": scatter_s,
                    "full_tile_shards": FULL_TILE_SHARDS,
                    "per_submit_s": {label: per for label, per in rows},
                    "full_tile_s": {
                        label: per * FULL_TILE_SHARDS for label, per in rows
                    },
                },
                indent=2,
            )
        )
        print(f"written       {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

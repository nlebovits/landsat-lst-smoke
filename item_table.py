"""The scene table every shard reads, written once and loaded once per worker.

`client.submit` used to carry a shard's own list of STAC item dicts. A shard at
512 px sees a p50 of 509 scenes, which pickle to 423,128 B, so a full tile
pushed 0.55 GB of task payload at the scheduler to describe 3,910 items that
pickle to 3.2 MB between them. MEASURED by `measure_submit_cost.py` against a
real `frisky.LocalCluster`, that costs 5.3 to 6.1 ms per submit against 0.024
to 0.031 ms for a path and a list of positions, about 200x.

The cost was never the pickling. `pickle.dumps` of 509 item dicts takes 0.9 ms,
so 1,296 of them is about 1.2 s against a measured phase of 373.7 s. It is the
payload itself moving through the client, the scheduler, and the worker, once
per shard that shares it.

Why a file rather than `client.scatter`. Two reasons, both measured:

* `Client.scatter(broadcast=True)` raises `NotImplementedError` in frisky
  0.7.2. Plain `scatter` places exactly one replica and the scheduler
  replicates on demand, which leaves a window at cluster start where the only
  copy is on one worker. Scattered data has no recompute path, so losing that
  worker loses the run. `tests/test_run_survives_worker_death.py` kills workers
  in exactly that window and requires exit 0.
* A worker restarted after a death re-reads this file from local disk and
  carries on. There is no scheduler state to lose and nothing to transfer.

JSON, not pickle. Pickle is smaller here (3.2 MB against 7.89 MB at
3,910 items), the parse costs 45 ms once per worker process, and the item dicts
are already JSON-native: `tile_inventory.build_item` writes nothing that needs
a `default=`. So what the round trip does is checkable, which is not a claim
anyone can make about a pickle, and a post-mortem can read the file.

The round trip changes one thing, and it is the whole of it. MEASURED over a
real item: `build_item` writes each geometry corner as a tuple and JSON has no
tuple, so `geometry.coordinates[0][n]` comes back as a two-element list. Every
number is identical, GeoJSON coordinates are arrays anyway, and
`pystac.Item.from_dict` reads the two the same. Nothing about the ring reaches
the pixels either: `odc.stac.stac_load` composites from the asset hrefs and the
`proj:` triple. `tests/test_no_stac_at_runtime.py` checks the arrays rather
than trusting this paragraph.

The table lives beside the staged scenes rather than in the output directory.
`staging.cleanup` already removes that directory, and 7.8 MB of scratch does
not belong in an artifact that ships.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

#: Parsed tables, keyed by the file they came from and its identity on disk.
#: A worker process runs many shards and every one of them wants the same
#: table, so it is parsed once. Keying on `(path, mtime_ns, size)` rather than
#: on the path alone means a stale entry cannot outlive the file that produced
#: it, which matters because the tests run several pipelines in one process.
_CACHE: dict[tuple[str, int, int], list[dict]] = {}

#: Two worker threads in one process reach `load` together on the first shard.
#: Without this they both parse, and the second discards its work.
_LOCK = threading.Lock()


def write(path, item_dicts) -> dict:
    """Write the table the workers will read, and describe what was written.

    Returns:
        A report: the path, the item count, the bytes on disk, and the seconds
        it took. It goes in the run summary, because a worker that cannot
        parse this file fails every shard and the size is the first thing to
        look at.
    """
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload = json.dumps(item_dicts)
    path.write_text(payload)
    return {
        "path": str(path.resolve()),
        "n_items": len(item_dicts),
        "bytes": len(payload),
        "seconds": time.perf_counter() - t0,
    }


def load(path) -> list[dict]:
    """The table, parsed once per worker process.

    Raises:
        FileNotFoundError: if the table is gone. Loud is right: the alternative
            is every shard failing with a different error from inside the
            loader.
    """
    path = str(Path(path).resolve())
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size)
    with _LOCK:
        table = _CACHE.get(key)
        if table is None:
            table = json.loads(Path(path).read_text())
            # One table per process. A second entry would only appear if the
            # file changed under a running worker, and holding the old one
            # buys nothing.
            _CACHE.clear()
            _CACHE[key] = table
    return table


def select(path, indices) -> list[dict]:
    """The items one shard needs, by position in the table."""
    table = load(path)
    return [table[i] for i in indices]

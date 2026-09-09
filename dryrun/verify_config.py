"""Prove the recommended quarter-tile config builds, before spending anything.

Graph only: no cluster, no reads, no cost. Run this before any instance.
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profile_lst_p95 import build_graph
from stac_window import DEFAULT_END, DEFAULT_START, items_cache_path

BBOX = (-62.5, -35.0, -60.0, -32.5)  # quarter of S30W065
items = pickle.loads(items_cache_path(BBOX, DEFAULT_START, DEFAULT_END).read_bytes())
CRS, RES = "EPSG:4326", 1 / 3600  # the actual grid
LOAD, RED, TC = 2048, 1024, 50

t0 = time.perf_counter()
lst, qa, _ = build_graph(items, BBOX, RED, CRS, RES, TC, load_chunk=LOAD)
build = time.perf_counter() - t0
raw = sum(len(o.__dask_graph__()) for o in (lst, qa))

print(f"scenes        {len(items)}")
print(f"grid          {CRS} @ 1/3600 deg")
print(
    f"lst_p95       shape {lst.shape}  dtype {lst.dtype}  chunks {lst.chunks[0][0]} px"
)
print(f"qa_count      shape {qa.shape}  dtype {qa.dtype}")
print(f"graph build   {build:.1f}s   raw tasks {raw:,}")
read_block_mb = LOAD * LOAD * TC * 4 / 1e6
print(f"read block    {LOAD}x{LOAD}x{TC} float32 = {read_block_mb:.0f} MB")
for slots in (16, 32):
    print(
        f"  x{slots} concurrent reads = {read_block_mb * slots / 1000:.1f} GB in flight"
    )
assert lst.shape == (9000, 9000), f"expected 9000x9000, got {lst.shape}"
assert str(lst.dtype) == "uint16", f"expected uint16, got {lst.dtype}"
assert qa.shape[0] == 12, f"expected 12 qa bands, got {qa.shape[0]}"
print("\nOK: shape, dtype and band count all correct. Safe to run.")

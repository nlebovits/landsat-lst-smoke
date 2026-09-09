"""Build the quarter-tile graph locally. No cluster, no reads, no cost.

Exercises exactly the three things that killed session 4: argparse handling of
a negative bbox, graph construction, and graph_stats (count + dask.optimize).
"""

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
CACHE = Path("/tmp/qt_items.pkl")
BBOX = (-62.5, -35.0, -60.0, -32.5)


def t(label):
    class T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *a):
            self.dt = time.perf_counter() - self.t0
            print(f"  {label:28s} {self.dt:8.2f}s", flush=True)

    return T()


import pystac_client

if CACHE.exists():
    items = pickle.loads(CACHE.read_bytes())
    print(f"items: {len(items)} (cached)")
else:
    with t("stac_search"):
        cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
        items = list(
            cat.search(
                collections=["landsat-c2-l2"],
                bbox=BBOX,
                datetime="2020-01-01/2025-01-01",
                query={
                    "eo:cloud_cover": {"lt": 100},
                    "platform": {"in": ["landsat-8", "landsat-9"]},
                },
            ).items()
        )
    CACHE.write_bytes(pickle.dumps(items))
    print(f"items: {len(items)}")

sys.path.insert(0, str(Path.cwd()))
from profile_lst_p95 import build_graph

LOAD = int(os.environ.get("LOAD", 512))
CHUNK = int(os.environ.get("CHUNK", 256))
TC = int(os.environ.get("TC", 10))
print(f"\nload_chunk={LOAD} chunk={CHUNK} time_chunk={TC}  scenes={len(items)}")

with t("build_graph") as b:
    lst_u16, qa_count, _ = build_graph(
        items, BBOX, CHUNK, "epsg:3857", 30, TC, load_chunk=LOAD
    )
print(f"  shape {lst_u16.shape}  chunks {lst_u16.chunks}")

# Count without optimize: how expensive is materialising the graph alone?
with t("materialise + count") as c:
    n = 0
    for obj in (lst_u16, qa_count):
        g = obj.__dask_graph__()
        n += len(g)
print(f"  raw tasks {n:,}")

# The suspect.
import dask

with t("dask.optimize") as o:
    opt = dask.optimize(lst_u16, qa_count)
with t("count optimized") as c2:
    m = sum(len(x.__dask_graph__()) for x in opt)
print(f"  fused tasks {m:,}  fusion {n / m:.2f}x")

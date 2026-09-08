"""Quarter tile on the CORRECT grid, sweeping dask's array.chunk-size."""
import os, pickle, sys, time
from pathlib import Path
import dask
sys.path.insert(0, "/home/nissim/Documents/dev/radiant-earth/landsat-lst-smoke")
from profile_lst_p95 import build_graph

items = pickle.loads(Path("/tmp/qt_items.pkl").read_bytes())
BBOX = (-62.5, -35.0, -60.0, -32.5)
CRS, RES = "EPSG:4326", 1/3600      # the actual tile grid

print(f"scenes {len(items)}   grid {CRS} @ 1/3600 deg")
print(f"{'chunk-size':>12}{'build s':>10}{'optimize s':>12}{'raw tasks':>12}{'fused':>12}{'red.block':>11}{'blockMB':>9}")
for cs in ("128MiB", "512MiB", "1GiB"):
    with dask.config.set({"array.chunk-size": cs}):
        t0 = time.perf_counter()
        lst, qa, _ = build_graph(items, BBOX, 512, CRS, RES, 10, load_chunk=1024)
        tb = time.perf_counter() - t0
        blk = lst.chunks[0][0]
        raw = sum(len(o.__dask_graph__()) for o in (lst, qa))
        t0 = time.perf_counter()
        opt = dask.optimize(lst, qa)
        to = time.perf_counter() - t0
        fused = sum(len(o.__dask_graph__()) for o in opt)
        mb = blk*blk*len(items)*4/1e6
        print(f"{cs:>12}{tb:>10.1f}{to:>12.1f}{raw:>12,d}{fused:>12,d}{blk:>11d}{mb:>9.0f}")
        print(f"             shape {lst.shape}")

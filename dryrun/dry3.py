"""What actually controls task count at quarter-tile scale?"""
import pickle, sys, time
from pathlib import Path
import dask
sys.path.insert(0, "/home/nissim/Documents/dev/radiant-earth/landsat-lst-smoke")
from profile_lst_p95 import build_graph

items = pickle.loads(Path("/tmp/qt_items.pkl").read_bytes())
BBOX = (-62.5, -35.0, -60.0, -32.5)
CRS, RES = "EPSG:4326", 1/3600
print(f"scenes {len(items)}  grid {CRS} @ 1/3600\n")
print(f"{'load':>6}{'reduce':>8}{'tchunk':>8}{'build s':>9}{'raw tasks':>12}{'blocks':>8}")
for load, red, tc in ((1024,512,10),(1024,512,50),(2048,512,10),(2048,512,50),(2048,1024,50)):
    t0=time.perf_counter()
    lst, qa, _ = build_graph(items, BBOX, red, CRS, RES, tc, load_chunk=load)
    tb=time.perf_counter()-t0
    raw = sum(len(o.__dask_graph__()) for o in (lst, qa))
    import math
    blocks = math.ceil(9000/load)**2
    print(f"{load:>6}{red:>8}{tc:>8}{tb:>9.1f}{raw:>12,d}{blocks:>8d}")

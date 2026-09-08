"""Is dask overriding the reduce chunk? Test at department and tile scale."""
import numpy as np, xarray as xr, dask, dask.array as da

print(f"dask array.chunk-size default: {dask.config.get('array.chunk-size')}")
print()
print(f"{'scenes':>8}{'set chunk':>11}{'after .chunk()':>16}{'after quantile':>16}{'block MB':>10}")
for scenes in (711, 1765, 3910):
    for c in (256, 512):
        x = xr.DataArray(
            da.random.random((scenes, 4096, 4096), chunks=(10, 1024, 1024)).astype("float32"),
            dims=("time", "y", "x"))
        xc = x.chunk({"y": c, "x": c})
        after_chunk = xc.chunks[1][0]
        q = xc.quantile(0.95, dim="time")
        after_q = q.chunks[0][0]
        mb = after_q*after_q*scenes*4/1e6
        flag = "" if after_q == c else "  <-- OVERRIDDEN"
        print(f"{scenes:>8}{c:>11}{after_chunk:>16}{after_q:>16}{mb:>10.0f}{flag}")

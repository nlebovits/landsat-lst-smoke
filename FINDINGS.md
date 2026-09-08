# Profiling a Landsat p95 LST composite

Where the time and memory actually go when you build a 95th-percentile Land
Surface Temperature composite for Pergamino, Buenos Aires, from five years of
Landsat Collection 2 Level 2 scenes.

Session `b5dd543f-2974-40c1-9f4c-5892e87c8d48`, 2026-09-08.
All measurements below come from runs in this session. Nothing is estimated
unless it says so.

## Headline

The full job runs in **93.6 seconds of compute, 139.3 seconds end to end**, for
711 scenes on a single `m6i.4xlarge` in `us-west-2`, reading `s3://usgs-landsat`
in-region.

```
uv run profile_lst_p95.py \
  --source earth-search \
  --load-chunk 1024 --chunk 512 \
  --workers 8 --threads-per-worker 4 \
  --memory-limit-gib 6
```

Peak memory is 31.8 GiB. Output is `min 32.3 C, mean 46.2 C, max 55.9 C`, with
every pixel valid.

## Study area and data

| | |
|---|---|
| Boundary | `pergamino_dept.gpkg`, Pergamino department |
| bbox, EPSG:4326 | `(-60.942796, -34.17645991, -60.138771, -33.54020044)` |
| Output grid | EPSG:3857 at 30 m, 2985 x 2845 px, 8.48 Mpx per scene |
| Window | 2020-01-01 to 2025-01-01 |
| Cloud filter | `eo:cloud_cover < 100` |
| Scenes | **711** via Earth Search, **673** via Planetary Computer |
| Volume | 8.7 MB per scene, about 6.1 GB per full run |

The two catalogues disagree on the same query. Earth Search returns 38 more
scenes than Planetary Computer. Neither is obviously wrong, but do not compare
scene counts across them.

## Output encoding

| Asset | Dtype | Scale | Offset | Nodata | Units |
|---|---|---|---|---|---|
| `lst_p95` | uint16, 1 band | 0.01 | -50.0 | 0 | celsius |
| `qa_count` | uint8, 12 bands (Jan..Dec) | — | — | none | count |

Decode with `celsius = dn * 0.01 + (-50.0)`. DN 1 to 65535 covers -49.99 C to
605.35 C in 0.01 C steps. Pergamino lands near DN 7000 to 9500, well inside
range. One edge case: a true -50.00 C encodes to DN 0 and becomes
indistinguishable from nodata. That never occurs here.

`qa_count` carries no nodata by design. A zero means no valid observation
survived masking for that month, which is different from a masked pixel, and
keeping the distinction visible is the point.

## Six corrections to the original workflow

Each of these is a defect in the starting snippet, found by measurement.

**1. No spatial chunking.** `chunks={"time": 10}` leaves the spatial dimensions
whole. `quantile` collapses the time axis into a single chunk, so one block
becomes 2985 x 2845 x 711 float32 values, or 22.1 GiB. The job cannot run.
Chunk spatially.

**2. DN 0 is the fill value.** The `lwir11` band uses 0 for fill. Left unmasked
it decodes to -124 C and drags the percentile down. Mask `dn != 0` alongside
the QA bits.

**3. `.where()` on the whole Dataset.** Masking the Dataset rather than the band
also masks `qa_pixel` and promotes it to float. Mask `lwir11` only, and cast to
float32 explicitly so the stack never silently becomes float64.

**4. No platform filter.** At `cloud_cover < 100` the search returns 913 scenes,
and **240 of them are Landsat 7**, which has no `lwir11` asset. Its thermal band
is `lwir` (ST_B6), and it has carried SLC-off scan gaps since 2003. Those 240
scenes load as nodata while still costing graph size. Filter to
`platform in ["landsat-8", "landsat-9"]`. The original `cloud_cover < 20` filter
hid this.

**5. `max` is not `p95`.** The snippet computes `max(dim="time")`. A percentile
is a different and more expensive reduction, because it forces the time axis
into one chunk.

**6. One chunk size for two jobs.** Reads and the percentile want opposite block
sizes. Separating them is the single largest optimisation found here, and it is
covered below.

## Local findings

Measured on a 16-core laptop reading Planetary Computer over a domestic link.

### The pipeline is I/O bound, overwhelmingly

Time inside `worker.exec.call`, broken down by task prefix:

| task | share of execution |
|---|---|
| `lwir11` read | 78.3% |
| `qa_pixel` read | 19.0% |
| `custom_nanquantile` (the p95 itself) | 1.7% |
| everything else | 1.0% |

`worker.exec.gil` accounts for 0.1% of execution. The workers hold the GIL
essentially never, so threads are nearly free and CPU is not the constraint.
The percentile you are asking for costs 1.7%. Reading the COGs costs 97.3%.

### The link, not the reader stack, was the ceiling

A bare `urllib` thread pool with no geospatial code in the path reached
5.53 MB/s at 32 concurrent range requests. The full odc pipeline reached 5.2 to
5.8 MB/s. The reader stack was already at the ceiling, and no tuning could beat
the connection.

| concurrency | MB/s | latency p50 |
|---|---|---|
| 1 | 0.97 | 709 ms |
| 16 | 4.87 | 2,839 ms |
| 32 | 5.53 | 4,400 ms |
| 64 | 5.42 | 8,108 ms |
| 128 | 9.30 | 11,988 ms |

### Load big, reduce small

Reads want large blocks. The source is UTM and the destination is not, so every
destination block needs a skewed source window plus an edge halo. Neighbouring
blocks then re-fetch the same source tiles. The percentile wants small blocks,
because each task holds `chunk * chunk * n_scenes` float32 values at once.

Decoupling the two settled it. At 24 scenes locally:

| read / reduce | compute | bytes moved |
|---|---|---|
| 256 / 256 | 107.4 s | ~591 MB |
| 512 / 512 | 60.7 s | ~334 MB |
| 1024 / 1024 | 48.1 s | ~236 MB |
| **1024 / 256** | **40.5 s** | **211 MB** |

Against the original 4 workers x 2 threads at chunk 256, that is **175.3 s down
to 40.5 s, a 4.3x speedup**, achieved by moving 2.8x less data rather than by
moving it faster. Throughput never improved. When you are bandwidth-limited,
transfer less.

## Moving to EC2

Round-trip latency measured from the laptop:

| endpoint | TCP RTT |
|---|---|
| Planetary Computer blob, `landsateuwest` | 7 to 21 ms |
| AWS S3, `us-west-2` | 153 to 254 ms |
| AWS S3, `eu-central-1` | 47 ms |

Running from Europe against `us-west-2` would raise latency 12x while raising
bandwidth 23x. Little's Law then says a gigabit link needs 33 concurrent streams
at 1 MB requests, or roughly 509 at 64 KB requests. Blocking readers give one
request per thread, so that regime would need an async reader.

**Running in-region removes the problem instead.** At sub-millisecond RTT the
requirement drops to between 1 and 11 streams, and ordinary threads cover it.
An async reader such as `async-tiff` is not worth building for this workload
once compute sits next to the data. `odc.loader` does expose `register_driver`
and a `ReaderDriver` protocol, and `stac_load` accepts `driver=`, so the plug
point exists if a future workload needs it.

### EC2 results

All runs: 711 scenes, `m6i.4xlarge` (16 vCPU, 64 GiB), `us-west-2`,
`s3://usgs-landsat` requester-pays, read chunk 1024.

| run | workers x threads | reduce | compute | total | MB/s | peak RSS | tasks |
|---|---|---|---|---|---|---|---|
| chunk 128 | 4 x 8 | 128 | 230.5 s | 309.5 s | 27.0 | 35.46 GiB | 527,721 |
| chunk 256 | 4 x 8 | 256 | 144.9 s | 196.5 s | 42.5 | 30.69 GiB | 144,833 |
| chunk 512 | 4 x 8 | 512 | 132.8 s | 157.3 s | 46.3 | 28.36 GiB | 44,157 |
| 1 worker | 1 x 32 | 512 | 190.0 s | 213.3 s | 32.2 | 30.98 GiB | 44,157 |
| 2 workers | 2 x 16 | 512 | 160.4 s | 183.5 s | 38.4 | 30.94 GiB | 44,157 |
| control A | 4 x 8 | 512 | 124.0 s | 168.6 s | 49.5 | 28.22 GiB | 44,157 |
| **8 workers** | **8 x 4** | 512 | **93.6 s** | **139.3 s** | **65.9** | 31.83 GiB | 44,157 |
| control B | 4 x 8 | 512 | 112.6 s | 154.8 s | 54.6 | 30.20 GiB | 44,157 |

Every run produced identical output.

### Task count governs memory, not per-task size

Chunk 128 was expected to lower memory, because each task holds a quarter as
much data. It raised memory instead, from 28.36 GiB to 35.46 GiB, and ran 1.7x
slower. The reason is pinning. At read chunk 1024, a 128 reduce block means each
input block feeds 64 outputs rather than 4, so the large input blocks stay
resident far longer. **Larger reduce blocks release memory sooner.**

### More processes, not more cores

Total threads were pinned at 32 across the topology runs. Only the process split
moved, on the same 16 vCPUs.

| workers x threads | CPU used | compute |
|---|---|---|
| 1 x 32 | 22% | 190.0 s |
| 2 x 16 | 33% | 160.4 s |
| 4 x 8 | 44 to 55% | 112.6 to 124.0 s |
| 8 x 4 | **76%** | **93.6 s** |

CPU utilisation more than tripled and wall time halved, with no change in core
count or thread count. The cores were idle all along. A single process could not
feed them, because sends serialised behind one queue and memory concentrated
into one spill manager.

Consolidating to one worker eliminated worker-to-worker transfer completely,
which is confirmed by zero `tcp.recv.queue`, `tcp.send.queue`, and
`worker.transfer.recv` spans. It still lost, and `worker.exec.call` rose from
1,728 s to 2,777 s for identical work. Fewer processes is worse here even when
the shuffle is free.

### The real cost is transfer queueing

`frisky observe transfers` on the 2-worker run:

```
Transfers  730 messages, 1,197 keys
  bytes    10.69 GiB logical -> 9.58 GiB wire
  wire     12.8s at 857.70 MiB/s per stream
  queue    send 6.2m + writer 21.3s + recv 1.7m

 send queue   369.1 s   118.9% of worker capacity
 recv queue   101.5 s    32.7%
 wire+read     12.8 s     4.1%
```

The wire moves 9.58 GiB in 12.8 seconds. The network is fast. The cost is 470
seconds of queueing, and the send queue alone exceeds one worker's entire
wall-clock capacity. Adding workers adds parallel transfer channels, which is
why more processes help.

Frisky's scheduler is not involved. The job runs 44,157 tasks in 132.8 s, or
332 tasks per second, against a published capacity of 250,000 to 400,000. At
3 microseconds per task the scheduler accounts for 0.13 s, or 0.10% of compute.

Spilling appeared in the low-worker runs and shows up in `frisky observe
overview`: 2.02 GiB spilled at 1 worker, 6.68 GiB at 2 workers, with disk
reaching 478 MB/s write and 314 MB/s read.

## Run-to-run variance is 9.6%

Two identical control runs differed by 11.4 s on a mean of 118.3 s. That is the
noise floor, and it changes how the other numbers should be read.

- The 8-worker gain, 24.7 s, is 2.2x the noise. It is real, and CPU utilisation
  and effective parallelism corroborate it mechanistically.
- The chunk 256 versus 512 difference, 12.1 s or 8.4%, sits **inside** the noise
  floor. Do not treat 512 as proven better than 256 on this evidence.
- The chunk 128 versus 256 difference, 59%, is far outside the noise and holds.

Any single-run comparison in this work carries roughly ten percent uncertainty.

## Instrumentation

`profile_lst_p95.py` is a standalone PEP 723 script. It reads no cache and
writes no raster. Nine stages are timed: imports, boundary, cluster start, STAC
search, asset signing, graph build, graph optimise, compute, stats, report.

Each stage records wall time, CPU time, RSS before and after, `ru_maxrss`
high-water, and optionally the Python heap peak. A separate sampler process
polls at 100 ms for client RSS, per-worker RSS, system memory, swap, and network
and disk counters. The sampler runs in its own process because the client holds
the GIL through graph construction, and an in-thread sampler drifts exactly when
the interesting thing is happening.

Frisky spans supplied every causal finding: the GIL fraction, the read-versus-
compute split, effective parallelism, spilling, and the transfer queue
breakdown. The psutil sampler supplied throughput and memory. Neither alone
would have been enough.

Use the `frisky observe` CLI rather than parsing `summary.json` by hand. The
funnel is `overview`, then `transfers` or `prefixes` depending on what the
overview says, then `stragglers` and `timeline`. Every command accepts a saved
spans file, so analysis continues after the cluster is gone. Hand-aggregating
span buckets mixes scheduler traffic with worker shuffle and produces wrong
conclusions, which happened repeatedly here before the CLI settled it.

Span collection is not free. It cost 15.8% of the run at 1.8M spans and 5.0% at
500k. Cap `--span-limit` around 500k.

## Practical notes

Frisky's API has sharp edges worth knowing:

- `memory_limit` is an integer count of **bytes, per worker**. The string
  `"4GB"` is not parsed.
- `processes` defaults to `False`, the opposite of `distributed.LocalCluster`.
- Workers start with `spawn`, and they read `FRISKY_TRACING_CAPACITY` from the
  environment, so set it before the cluster is constructed.
- Get the client from `cluster.get_client()`. `frisky.Client(cluster)` raises.
- There is no public span context manager. Build one on `record_span` and
  `now_ns`.
- Use `query_spans`, not `get_spans`. With `processes=True` the worker spans
  live in the worker processes.

`odc.loader.configure_rio(client=...)` accepts `client` and discards it, so GDAL
settings must reach spawned workers through `os.environ`.

Planetary Computer SAS tokens last about an hour and are signed at search time.
The script signs immediately before building the graph, so the token clock
starts at compute rather than at search. Earth Search needs no signing, because
`s3://` hrefs authenticate per request from ambient AWS credentials, with
`AWS_REQUEST_PAYER=requester`.

## What is not settled

- **16 workers is untested.** The worker-count curve had not turned at 8, and
  CPU was at 76%.
- **A larger instance is untested.** More total threads is the next lever now
  that process splitting has been exhausted.
- **Chunk 512 versus 256 is unresolved**, as noted above.
- **This tunes one department at 711 scenes.** A different area or scene count
  will move the optimum, because the memory term scales with both.

## Scaling to a full grid tile

The grid is defined in the sibling `landsat-lst` repo: 5 degree tiles on an
EPSG:4326 grid at 3600 px per degree, named `N40W075` / `S30W065`, with a
south-exclusive north-inclusive convention. Pergamino falls entirely inside
**S30W065**, lat (-35, -30], lon [-65, -60).

| | department | quarter tile | full tile |
|---|---|---|---|
| raster | 2985 x 2845 | 9000 x 9000 | 18,000 x 18,000 |
| pixels | 8.5 Mpx | 81 Mpx | 324 Mpx |
| scenes | 711 | **1,765** | **3,910** |
| WRS path/rows | 6 | 13 | 25 |
| read volume | 6.1 GB | ~58 GB | ~233 GB |

Scene counts are measured, not estimated.

### RETRACTED: the memory model was wrong

An earlier version of this document claimed rechunk memory is
`chunk² × scenes × 4 × 2`, that it scales linearly with scene count, and that
reduce chunk 512 would need 327 GiB at 3,910 scenes. **All of that is wrong**,
and a local dry run costing nothing showed why.

**`--chunk` never controlled the reduce block.** Dask normalises every block to
its `array.chunk-size` target, 128 MiB by default, during the rechunk that
`quantile` forces:

| scenes | chunk requested | block after `.chunk()` | block after `quantile` | size |
|---|---|---|---|---|
| 711 | 256 | 256 | **217** | 134 MB |
| 711 | 512 | 512 | **217** | 134 MB |
| 1,765 | 256 or 512 | — | **137** | 133 MB |
| 3,910 | 256 or 512 | — | **92** | 132 MB |

At 711 scenes, chunk 256 and chunk 512 produce an **identical** 217-pixel reduce
block. The block auto-shrinks as scene count rises, holding per-block memory
near constant. That is why measured peak RSS barely moved across chunk 128, 256
and 512: 35.5, 30.7 and 28.4 GiB.

So reduce chunk 512 is **not** infeasible at tile scale, and the optimum does
not move with scene count for the reason given. `--chunk` does still change the
*intermediate* rechunk, and therefore task count, which is what the measured
wall-time differences reflect. The numbers in the table below are real
measurements. The causal story attached to them was invented.

### What actually stalled the quarter tile

The observed fact stands: at 1,765 scenes, eight workers each reached 11 GB,
96 GiB in total, **before a single byte of imagery was read**. The explanation
offered at the time, that read columns pin memory in proportion to scene count,
was a guess made without instrumentation.

A local dry run, no cluster and no reads, gives the real answer:

```
scenes 1765, load_chunk 512, EPSG:3857 @ 30 m
  build_graph        72.68 s
  raw tasks       6,278,518
  dask.optimize     139.63 s
  fused tasks     4,270,657
```

**212 seconds of single-threaded client work before anything is dispatched**,
and 6.3 million tasks against an estimate of 115 thousand, a 50x miss. The
estimate assumed `--chunk 256` set the reduce block; it did not, so the block
came out at 137 px and the block count with it.

This is why frisky could not help. Graph construction and `dask.optimize` run
in the client, before any task reaches the scheduler, so there are no worker
spans to read. The harness also pipes stdout through `grep`, which
block-buffers, so no stage line printed until the run ended. Three EC2 runs were
diagnosed by watching CPU percentage and RSS and guessing.

### Two harness defects this exposed

**`graph_stats` does not scale.** It runs `dask.optimize` over the whole graph.
That costs a few seconds at 44k tasks. At quarter-tile scale, 6.3M raw tasks, it
measured 139.6 s on its own and no imagery was read in that time. `--no-graph-stats`
now disables it, and it should be off above roughly 200k tasks. Because this
stage sits between graph build and compute, **the quarter-tile stall cannot be
attributed to the pipeline rather than to this instrumentation.** No successful
quarter-tile run was obtained.

**Frisky workers survive `pkill` by script name.** They spawn through
multiprocessing, so their command line is a bare `-c`. Killing a run with
`pkill -f profile_lst_p95` leaves the workers behind. Eight orphans held 88 GB
across two restarts here and contaminated a memory reading, which produced a
wrong conclusion until the process list was checked. Match on the venv path and
kill by PID:

```bash
ps -eo pid,rss,args --sort=-rss \
  | awk '$2 > 200000 && /environments-v2/ {print $1}' \
  | xargs -r kill -9
```

### A third error, found the same way

The quarter-tile runs used `--crs epsg:3857 --resolution 30`, carried over from
the department work. The grid is **EPSG:4326 at 1/3600 degree**. Web Mercator
inflates area by `1/cos(lat)`, so the raster came out 11160 x 9278 rather than
9000 x 9000: 104 Mpx instead of 81, on the wrong projection. Any tile run must
pass the grid CRS explicitly.

### The lesson that cost the most

Every one of these was reproducible locally, with no cluster, no reads and no
money: the argparse failure, the 6.3M task graph, the 139.6 s optimize, the
chunk override, and the wrong CRS. Build the graph on a laptop first. It takes
about two minutes and costs nothing.

## Cost

Four EC2 sessions, eight completed full-scale runs plus the quarter-tile attempts, about **$3.35** total.

## Files

| path | contents |
|---|---|
| `profile_lst_p95.py` | the profiling harness |
| `sweep_throughput.py` | configuration sweep driver |
| `ec2-results/` | eight EC2 runs: stages, memory series, graph stats, frisky reports |
| `smoke/`, `smoke2/`, `split/`, `sweep/` | local runs with full frisky spans and traces |
| `full/` | the 673-scene local run, ended early by a network change |

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

### The quarter tile does not run, and the reason is architectural

Two configurations were run against the real grid on a 128 GiB box. Neither
completed. `frisky observe` gave the reason, from the live dashboard.

**Attempt 1**, `load 2048 / chunk 1024 / time_chunk 50`, chosen to minimise
task count at 260k:

```
state    waiting 136,056   processing 1,581   memory 21,423   queued 3,695
Worker I/O   produced 1.85 TiB   spilled 14,313 objects (360.12 GiB)
blocked  ('lwir11-...', 29, 2, 0)  waiting on 50  blocker ('open-lwir11-...', 34) Queued
```

Every read task waited on 50 dependencies, one per scene in the time chunk, and
each block was 839 MB. **Minimising task count was the wrong objective.** Frisky
schedules 250,000-400,000 tasks/s, so even 1.4M tasks is about 5 seconds of
scheduling. Working set is the constraint, and `time_chunk 50` multiplied it.

**Attempt 2**, `load 1024 / chunk 512 / time_chunk 10`, the shape validated at
711 scenes. It read at 69 MB/s with a 3.0 GiB peak, against 0 MB/s and 100.8 GiB
for attempt 1. It still stalled:

```
state    waiting 503,696   processing 17,121   memory 167,209
memory   57.07 GiB / 104.00 GiB
spilled  1.21 TiB   unspilled 373.95 GiB   network recv 225.07 GiB
```

`frisky observe transfers` isolates the cause:

```
Transfers  96,229 messages, 112,461 keys
  bytes    241.70 GiB logical -> 162.59 GiB wire
  costs    queue 26.8m, cpu 31.6m, wire 11.8m

Prefix           Msgs     Logical      Wire      Queue      CPU
rechunk-merge  19,893  182.62 GiB  120.77 GiB   12.3m    23.8m
```

**`rechunk-merge` is 76% of all shuffled bytes.** The p95 forces a reorganisation
of `81 Mpx × 1765 scenes × 4 bytes` = **572 GB of float32** from time-major read
blocks into space-major reduce blocks. That is a half-terabyte all-to-all
shuffle, and no block size avoids it. 1.21 TiB spilled for a 58 GB input.

### The fix is to shard, not to tune, and it works

A 512 x 512 shard holds `512² × 1765 × 4` = 1.85 GB for its complete time stack.
That fits in one worker with no shuffle at all. `shard_lst_p95.py` submits one
shard as one task: load, mask, reduce, encode, return. No dask array spans a
worker boundary, so there is nothing to rechunk and nothing to shuffle.

**Measured, quarter tile of S30W065, `r6i.4xlarge`, 16 vCPU:**

```
raster      9000 x 9000 px      shards   324 of 512 px
scenes      1765                search   18.2 s
compute     1113.9 s (18.6 min)          3.44 s/shard wall
client RSS  5.09 GiB peak
LST p95     min -17.0 C  mean 44.6 C  max 75.9 C  (100.0% valid)
qa_count    Jan 18.9 ... Jun 12.2  Jul 10.5 ... Dec 16.1
```

`qa_count` is the correctness check: all twelve months populated, summer above
winter for a southern-hemisphere site.

Frisky on the completed run, against the array attempt on the same problem:

| | array graph | **sharded** |
|---|---|---|
| spilled | 1.21 TiB | **0 B** |
| worker-to-worker transfer | 225.07 GiB | **0 B** |
| task results pinned | 167,209 | 324, released as gathered |
| memory peak | 57 GiB / 104 | **26.5 GiB / 96 (28%)** |
| completed | never | **324 / 324** |

### Confirmed on 64 vCPU

The same quarter tile on a `c6i.16xlarge`, 64 slots x 4 read threads:

```
324/324   264.8 s   0.82 s/shard   client RSS 5.1 GiB
LST p95   min -17.0 C  mean 44.6 C  max 75.9 C  (100.0% valid)
frisky    memory 95.93 GiB / 102.40 GiB (94%)  spilled 0 B  network recv 0 B
```

**1113.9 s to 264.8 s, a 4.2x speedup on 4x the cores**, and the output matches
the 16-core run exactly on min, mean, max and valid fraction. Throughput rose
from 167 to 358 MB/s and CPU sat at 92% user, 2% idle.

Memory ran at 94% of the configured limit, so `--memory-limit-gib 1.6` across 64
workers is the practical floor at 512 px shards and ~711 scenes. Above 64 cores
the shard edge should shrink: at 384 px a shard needs 0.39 GiB rather than 0.69,
which is what keeps the concurrent working set inside RAM.

One diagnostic to read correctly: 64 `frisky_worker_sigterm_dump` entries appear
at the end of a clean run. That is one per worker at `cluster.close()`, not a
memory kill. `pressure_fraction` near 1.0 in those dumps is a snapshot at
shutdown.

### There is no inefficiency left on this instance

```
wall-clock 1187.2 s   workers 16   tasks 324
compute    18,740.7 s      scheduler 18.5 s
worker.exec.call 18,699.8 s      worker.exec.gil 2.1 ms
```

Slot utilisation is 15.8 of 16, or **99%**. The GIL costs 2.1 milliseconds.
Scheduler overhead is 18.5 s of 1,187. `observe stragglers` reports every worker
within 0 s of the median. Nothing spills, nothing transfers.

**The only remaining lever is more cores**, and because the work is CPU-bound the
cost per tile stays flat while wall time falls:

| instance | vCPU | full tile wall | cost |
|---|---|---|---|
| `r6i.4xlarge` | 16 | ~74 min | ~$1.25 |
| **`c6i.16xlarge`** | **64** | **~18 min** (measured 4x 264.8 s) | **~$0.80** |
| 4 x `c6i.16xlarge` | 256 | **~4.4 min** | ~$0.80 |

A full tile is 1,296 shards, exactly 4x the quarter, because per-shard cost is
bounded by revisit rate rather than tile size. Roughly 520 land tiles between
60 N and 60 S puts a global five-year composite near **$650**, plus about $0.09
per tile in requester-pays GET charges.

### Two failures worth keeping

**Concurrency knobs multiply.** `--workers 8 --threads-per-worker 4` gives 32
shard slots, and `--read-threads 4` multiplies that to 128 OS threads on 16
cores. Measured: 15 of 324 shards in 400 s, the first wave of 64 all crawling.
Sizing slots to cores instead gave 15 shards in 105 s, 3.8x faster. The script
now refuses to start above `cores * 6` threads.

**`client.gather` on every future at once aborts the process.** With 324 futures
and ~1 GB of results resident, frisky 0.7.2 raised a Rust panic across the PyO3
boundary at 90% completion, which cannot unwind and so aborts rather than
raising. 290 shards had completed with zero errors and all of it was lost.
Collecting with `frisky.as_completed` and assembling one result at a time
completed 324 of 324 with no panics.

### Orphaned workers contaminated two measurements

Frisky workers spawn with a bare `-c` command line, so `pkill` by script name
misses them, and an RSS threshold misses the small ones. Orphans survived into
a later run twice. The second time, 26.6 GB of dead cluster made a healthy run
look like it was failing, and I read the wrong dashboard port before noticing
two schedulers were listening. Kill by process age relative to the current run,
and confirm against `ss -tlnp` that only one dashboard is up.

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

Five EC2 sessions, eight completed department-scale runs, one completed 200-scene quarter-tile smoke run, and two quarter-tile attempts that did not complete. About **$4.45** total.

## Files

| path | contents |
|---|---|
| `profile_lst_p95.py` | the profiling harness |
| `sweep_throughput.py` | configuration sweep driver |
| `ec2-results/` | eight EC2 runs: stages, memory series, graph stats, frisky reports |
| `smoke/`, `smoke2/`, `split/`, `sweep/` | local runs with full frisky spans and traces |
| `full/` | the 673-scene local run, ended early by a network change |

## Splitting one tile across machines

The shard plan is deterministic and anchored to whole degrees, so a shard covers
the same pixels regardless of which request produced it. That makes a tile
splittable across machines with no coordination beyond the slice index.

```bash
# four machines, one tile
uv run shard_lst_p95.py --bbox=-65,-35,-60,-30 --pixels-per-degree 3600 \
    --shard-slice 0:324   --out-dir ./part0     # machine 0
uv run shard_lst_p95.py ... --shard-slice 324:648  --out-dir ./part1
uv run shard_lst_p95.py ... --shard-slice 648:972  --out-dir ./part2
uv run shard_lst_p95.py ... --shard-slice 972:1296 --out-dir ./part3

# then anywhere
uv run shard_lst_p95.py --merge part0 part1 part2 part3 --out-dir ./tile
```

Verified locally: four slices of a full tile cover **324,000,000 px exactly**,
with no gaps and no overlap, and a merge round-trip reproduces the source array
bit for bit. Merging an incomplete set reports the missing pixel count and exits
non-zero rather than writing a quietly wrong raster.

### Wall clock per tile

Cost is flat, because the work is CPU-bound and you are buying core-hours either
way. Only wall time changes.

| hardware for one tile | compute | cost |
|---|---|---|
| 1 x `c6i.16xlarge` (64 vCPU) | ~20 min | $0.88 |
| 2 x `c6i.16xlarge` | ~10 min | $0.88 |
| 4 x `c6i.16xlarge` | **~5 min** | $0.89 |

Add roughly 160 s for boot and dependency install on a cold instance, and 40 s
for the STAC search. Keeping an instance warm removes the first; caching the
item list removes most of the second.

Note that `c6i` beats the `r6i` used for the measurements on both axes: the peak
was 26.5 GiB of 96, so the memory-optimised instance was renting RAM the job
never touched.

## Rehearsal mode: everything except the read, for free

The expensive lesson of this session is that almost every failure was findable
on a laptop, and I kept finding them on billed instances instead: an argparse
flag, a CRS/resolution pairing, a silently ignored chunk argument, wrong
spatial dim names, a 6.3M-task graph, threads multiplying to 128 on 16 cores.

`--rehearse N` runs the entire pipeline with N synthetic scenes and no S3 at
all. It exercises shard planning, item filtering, cluster startup, submission,
gather, assembly into the output raster, part writing and merge. The only thing
it does not cover is read throughput, which is the one question that genuinely
needs the cloud.

```bash
# a full tile across four machines, rehearsed in about 30 seconds
for i in 0 1 2 3; do
  uv run shard_lst_p95.py --bbox=-65,-35,-60,-30 --pixels-per-degree 3600 \
    --shard-slice $((i*324)):$(((i+1)*324)) --rehearse 900 --out-dir part$i
done
uv run shard_lst_p95.py --merge part0 part1 part2 part3 --out-dir tile
```

### It immediately found two bugs that would have cost a four-instance run

**`--shard-slice` sliced the filtered list, not the plan.** Shards with no
overlapping scene are dropped before slicing, so indices shift and the last
slice came up 36 shards short. Machines would have left gaps in a real tile,
where ocean and edge shards legitimately have no scenes. The slice now applies
to the plan, which is deterministic and anchored to whole degrees.

**A barren shard was skipped rather than written.** That makes "no Landsat
coverage here" indistinguishable from "a machine died", which defeats the whole
point of the coverage check. Barren shards are now written as nodata, so a
complete set of slices always reports 100% coverage and a genuinely missing
slice is caught.

Verified after both fixes: four slices of an 18,000 x 18,000 tile merge to
**100.00% coverage, 1,296 of 1,296 shards**, and dropping one slice still
reports `75,168,000 px never written`.

### What rehearsal cannot tell you

Read throughput, and therefore wall time and cost. Everything else — geometry,
coverage, slicing, assembly, memory shape of the merge (peak 8.2 GB for a full
tile), and the concurrency guard — is answerable before an instance exists.

## Full tile, four machines: 4.8 minutes

`S30W065`, 18,000 x 18,000 px, 3,910 scenes, 1,296 shards across four
`c6i.16xlarge` (64 vCPU each), one `--shard-slice` per machine.

| slice | shards | compute | per shard |
|---|---|---|---|
| 0 | 324 | 273.5 s | 0.84 s |
| 1 | 324 | 289.7 s | 0.89 s |
| 2 | 324 | 284.6 s | 0.88 s |
| 3 | 324 | 247.0 s | 0.76 s |

**4.8 minutes wall**, set by the slowest slice. Zero errors on all four. The
merge assembled 1,296 of 1,296 shards at **100.00% coverage** in 13.3 s, peaking
at 8.26 GB.

```
LST p95   min -49.7 C   mean 45.8 C   max 90.6 C   (100.0% valid)
```

### The tails are the thinly-observed pixels, and only those

| band | share of tile | mean observations per pixel |
|---|---|---|
| 0-60 C | **99.967%** | 173.3 |
| below -20 C or above 75 C | **0.027%** | **6.0** |

p1 is 29.7 C, p50 46.4 C, p99 53.3 C. The extremes sit on pixels with about six
observations against 173 elsewhere, where a 95th percentile carries no
information. `qa_count` is the band that lets a consumer drop them, which is
why it is written without a nodata value.

### Cost, audited

An earlier version of this section claimed about **$10** for the full-tile run,
with 55 minutes of overhead against 4.8 minutes of compute. **Both figures were
wrong.** They came from assuming each instance ran roughly an hour. AWS says
otherwise:

| instance | launch (UTC) | terminate (UTC) | lifetime |
|---|---|---|---|
| `i-01140eb2b3090f95a` | 06:07:33 | 06:18:15 | **642 s** |
| `i-0190abda85788d281` | 06:07:33 | 06:18:15 | 642 s |
| `i-062dc4088e6d06202` | 06:07:33 | 06:18:15 | 642 s |
| `i-072d85a880e92be10` | 06:07:33 | 06:18:15 | 642 s |

`c6i.16xlarge`, `ami-04678417fc39d7171`, us-west-2b, Linux on-demand, shared
tenancy. **0.7133 instance-hours**, and EC2 bills Linux per second.

| item | basis | cost |
|---|---|---|
| EC2 | 0.7133 ih x $2.72/hr | **$1.94** |
| EBS | 4 x 150 GB gp3 x 0.178 h x ($0.08/730) | $0.012 |
| Public IPv4 | 4 x 0.178 h x $0.005 | $0.004 |
| S3 GET, requester pays | 605,617 reads x 2 bands x R req x $0.0004/1000 | **$0.97 - $3.88** |
| S3 to EC2 transfer, same region | | $0.00 |
| **total** | | **$2.92 - $5.83** |

Measured: instance lifetimes, and the 605,617 shard-scene reads from the shard
plan. Estimated: the $2.72/hr list rate, which the Pricing API would not confirm
for this SSO role (`AccessDenied`), though spot at $0.89-$1.03 is consistent
with it; and R, the requests GDAL issues per windowed read, bracketed at 2 to 8
rather than guessed. Cost Explorer is also `AccessDenied`, so no billed figure
was available to check against.

**Overhead is 2.2x, not 13x.** 10.7 minutes of lifetime against 4.8 minutes of
compute leaves 5.9 minutes for boot, dependency install, the STAC search, the
part write and the span query. The earlier claim that `savez_compressed` alone
took 15-20 minutes is impossible inside a 642-second lifetime.

**Quarter tile**, one instance, 562 s: $0.43 EC2 plus $0.24-$0.96 S3 =
**$0.67 - $1.39**.

### Global, 520 land tiles

| | on-demand | spot (~$0.95/hr) |
|---|---|---|
| EC2 | $1,010 | $353 |
| S3 GET | $505 - $2,020 | same |
| **total** | **$1,515 - $3,030** | **$858 - $2,373** |

Materially worse than the ~$610 quoted before, because **S3 request charges were
omitted entirely and are comparable to or larger than compute**. On spot they
dominate.

That changes what to optimise. Request count is now a cost term, not only a
latency term, so **larger `--load-chunk` values pay twice**: fewer, larger reads
cut both wall clock and the GET bill. This is the opposite of the earlier
conclusion here, and it follows from the line item that was missing.

Remaining overhead is still worth removing, but it is minutes rather than tens
of minutes: every machine repeats the same STAC search over 3,910 items, boot
plus dependency install is ~160 s, and the part write is single-threaded
`savez_compressed` over ~1.1 GB.

## How to price a run, so this does not recur

The $10 error came from inferring instance lifetime from how long the work felt,
while polling loops returned instantly and made wall clock run far ahead of that
sense. The fix is to never infer it. AWS records it exactly:

```bash
aws ec2 describe-instances --region us-west-2 \
  --filters "Name=tag:purpose,Values=<tag>" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,StateTransitionReason]' \
  --output text
```

`LaunchTime` and the timestamp inside `StateTransitionReason` give the lifetime
to the second, and terminated instances stay queryable for about an hour. Linux
on-demand bills per second past a 60-second minimum, so the cost is
`instance_seconds / 3600 x rate`, with no hourly rounding.

Then price every line, not just EC2:

| line | why it is easy to miss |
|---|---|
| EC2 | the only one usually remembered |
| EBS | small here, but scales with volume size x lifetime |
| Public IPv4 | $0.005/hr per address since 2024, per instance |
| **S3 GET** | **requester-pays; at this read pattern it rivals or beats EC2** |
| data transfer | free S3 to EC2 in-region, not free across regions |

Separate measured from estimated in the write-up. Here the lifetimes and read
counts were measured; the hourly rate and requests-per-read were estimated, and
the Pricing API and Cost Explorer were both `AccessDenied` for this SSO role, so
neither could confirm the total.

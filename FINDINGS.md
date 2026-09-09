# Profiling a p95 composite of land surface temperature

Where the time and the memory go in a p95 composite of Land Surface
Temperature (LST). The pipeline reads five years of Landsat Collection 2 Level 2
scenes. The study area covers Pergamino, Buenos Aires, and the 5 degree grid
tile that contains it.

Session `b5dd543f-2974-40c1-9f4c-5892e87c8d48`, 2026-09-08. Every measurement
here comes from a run in this session. A label marks each figure as MEASURED,
DERIVED, or UNKNOWN. The `Corrections` section at the end records what earlier
versions of this document got wrong.

## Headline

Four `c6i.16xlarge` instances built the full tile `S30W065` in **4.8 minutes of
wall clock**. The tile spans 18,000 x 18,000 px over 3,910 scenes, cut into
1,296 shards of 512 px.

| slice | shards | compute | per shard |
|---|---|---|---|
| 0 | 324 | 273.5 s | 0.84 s |
| 1 | 324 | 289.7 s | 0.89 s |
| 2 | 324 | 284.6 s | 0.88 s |
| 3 | 324 | 247.0 s | 0.76 s |

The slowest slice fixes the wall clock. All four slices finished with zero
errors. The merge assembled 1,296 of 1,296 shards at **100.00% coverage** in
13.3 s, peaking at 8.26 GB.

```
LST p95   min -49.7 C   mean 45.8 C   max 90.6 C   (100.0% valid)
```

Each instance lived 642 s, which puts the fleet at **$1.94 of EC2 time**. S3
requester-pays charges add **$2.31**, so the tile cost **$4.28**. Nobody counted
the requests behind that second figure until the third version of this document.
They turned out to be the larger line. See `Cost`.

## What to run

```bash
# one machine per slice
uv run shard_lst_p95.py --bbox=-65,-35,-60,-30 \
    --pixels-per-degree 3600 \
    --shard-slice 0:324 --out-dir ./part0
uv run shard_lst_p95.py ... --shard-slice 324:648  --out-dir ./part1
uv run shard_lst_p95.py ... --shard-slice 648:972  --out-dir ./part2
uv run shard_lst_p95.py ... --shard-slice 972:1296 --out-dir ./part3

# then anywhere
uv run shard_lst_p95.py --merge part0 part1 part2 part3 --out-dir ./tile
```

The shard plan is deterministic and anchors to whole degrees, so a shard covers
the same pixels whichever request produced it. The machines need no coordination
beyond the slice index.

Rehearse the same fleet on a laptop first. `--rehearse N` substitutes N
synthetic scenes and touches no object store:

```bash
for i in 0 1 2 3; do
  uv run shard_lst_p95.py --bbox=-65,-35,-60,-30 \
    --pixels-per-degree 3600 \
    --shard-slice $((i*324)):$(((i+1)*324)) \
    --rehearse 900 --out-dir part$i
done
uv run shard_lst_p95.py --merge part0 part1 part2 part3 --out-dir tile
```

## Study area, grid, and data

| | |
|---|---|
| Boundary | `pergamino_dept.gpkg`, Pergamino department |
| bbox, EPSG:4326 | `(-60.942796, -34.17645991, -60.138771, -33.54020044)` |
| Department grid | EPSG:3857 at 30 m, 2985 x 2845 px, 8.48 Mpx per scene |
| Window | 2020-01-01 to 2025-01-01 |
| Cloud filter | `eo:cloud_cover < 100` |
| Scenes | **711** via Earth Search, **673** via Planetary Computer |
| Volume | 8.7 MB per scene, about 6.1 GB per department run |

Earth Search returns 38 more scenes than Planetary Computer for the same query.
The difference has no known cause in either catalogue. Treat the two scene
counts as incomparable.

The sibling `landsat-lst` repository defines the production grid. A tile spans
5 degrees on an EPSG:4326 grid at 3600 px per degree. Names look like `N40W075`
and `S30W065`. The convention excludes the south edge and includes the north
edge. Pergamino falls inside `S30W065`, lat (-35, -30], lon [-65, -60).

The next table counts Worldwide Reference System (WRS) path and row
combinations in its fourth row. A catalogue search produced every scene count in
it. None of them is an estimate.

| | department | quarter tile | full tile |
|---|---|---|---|
| raster | 2985 x 2845 | 9000 x 9000 | 18,000 x 18,000 |
| pixels | 8.5 Mpx | 81 Mpx | 324 Mpx |
| scenes | 711 | **1,765** | **3,910** |
| WRS path/rows | 6 | 13 | 25 |
| read volume | 6.1 GB | ~58 GB | ~233 GB |

## Output encoding

| Asset | Dtype | Scale | Offset | Nodata | Units |
|---|---|---|---|---|---|
| `lst_p95` | uint16, 1 band | 0.01 | -50.0 | 0 | celsius |
| `qa_count` | uint8, 12 bands (Jan..Dec) | n/a | n/a | none | count |

Decode with `celsius = dn * 0.01 + (-50.0)`. Digital numbers 1 to 65535 cover
-49.99 C to 605.35 C in 0.01 C steps. Pergamino lands near digital number 7000
to 9500, well inside range. One edge case exists. A true -50.00 C encodes to
digital number 0, which then looks identical to nodata. That value never occurs
here.

`qa_count` has no nodata value, by design. A zero means that no valid
observation survived masking for that month, which differs from a masked pixel.
Keeping the two apart is the point of the band.

## Architecture: shard, do not tune

### The array graph fails at tile scale

Two array-graph configurations ran against the real grid on a 128 GiB box.
Neither finished. `frisky observe` gave the reason from the live dashboard.

The first attempt used `load 2048 / chunk 1024 / time_chunk 50`, chosen to
minimise task count at 260k. It produced 1.85 TiB of worker output, spilled
14,313 objects (360.12 GiB), and read 0 MB/s at a 100.8 GiB peak. Every read
task waited on 50 dependencies, one per scene in the time chunk, and each block
came to 839 MB. **Minimising task count was the wrong objective.** Frisky
schedules 250,000 to 400,000 tasks per second, so even 1.4M tasks amount to
about 5 seconds of scheduling. The working set is the constraint, and
`time_chunk 50` multiplied it.

The second attempt used `load 1024 / chunk 512 / time_chunk 10`, the shape that
worked at 711 scenes. It read at 69 MB/s with a 3.0 GiB peak. It stalled
anyway:

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

**`rechunk-merge` accounts for 76% of all shuffled bytes.** The p95 forces a
reorganisation of `81 Mpx x 1765 scenes x 4 bytes`, or **572 GB of float32**,
from time-major read blocks into space-major reduce blocks. That is a
half-terabyte all-to-all shuffle, and no block size avoids it. The run spilled
1.21 TiB for a 58 GB input.

The client stalls before the cluster even starts. A local dry run, with no
cluster and no reads, shows where:

```
scenes 1765, load_chunk 512, EPSG:3857 @ 30 m
  build_graph        72.68 s
  raw tasks       6,278,518
  dask.optimize     139.63 s
  fused tasks     4,270,657
```

**212 seconds of single-threaded client work precede any dispatch**, over a
graph of 6.3 million tasks. An earlier estimate put the count at 115 thousand,
a 50x miss. Frisky cannot show any of this, because graph construction and
`dask.optimize` both run in the client, ahead of the scheduler, so no worker
span exists to read.

### The shard plan

A 512 x 512 shard needs `512² x 1765 x 4` = 1.85 GB for its complete time stack.
One worker holds that, and no array crosses a worker boundary, so nothing
rechunks and nothing shuffles.

`shard_lst_p95.py` submits one shard as one task. The task loads, masks,
reduces, encodes, and returns.

| | array graph | **sharded** |
|---|---|---|
| spilled | 1.21 TiB | **0 B** |
| worker-to-worker transfer | 225.07 GiB | **0 B** |
| task results pinned | 167,209 | 324, released as gathered |
| memory peak | 57 GiB / 104 | **26.5 GiB / 96 (28%)** |
| completed | never | **324 / 324** |

### Measured results

Every row below ran the sharded pipeline to completion.

| run | instance | slots | shards | scenes | compute | per shard |
|---|---|---|---|---|---|---|
| quarter tile | `r6i.4xlarge` | 16 | 324 | 1,765 | 1,113.9 s | 3.44 s |
| quarter tile | `c6i.16xlarge` | 64 | 324 | 1,765 | 264.8 s | 0.82 s |
| full tile | 4 x `c6i.16xlarge` | 256 | 1,296 | 3,910 | 289.7 s | 0.88 s |

The two quarter-tile runs produce identical output:

```
LST p95   min -17.0 C  mean 44.6 C  max 75.9 C  (100.0% valid)
qa_count  Jan 18.9 ... Jun 12.2  Jul 10.5 ... Dec 16.1
```

`qa_count` is the correctness check. Every one of the twelve months has
observations, and summer exceeds winter for a southern-hemisphere site.

Four times the cores cut the quarter tile from 1,113.9 s to 264.8 s, a 4.2x
speedup. Throughput rose from 167 to 358 MB/s. Processor time came to 92% user and 2%
idle. Client Resident Set Size (RSS) peaked at 5.1 GiB on both runs.

```
frisky    memory 95.93 GiB / 102.40 GiB (94%)  spilled 0 B  network recv 0 B
```

Memory ran at 94% of the configured limit, so `--memory-limit-gib 1.6` across 64
workers is the practical floor at 512 px shards and 1,765 scenes. Past 64 cores
the shard edge should shrink. A 384 px shard needs 0.39 GiB rather than 0.69,
which is what keeps the concurrent working set inside RAM.

One diagnostic invites a misreading. A clean 64-worker run ends with 64
`frisky_worker_sigterm_dump` entries, one per worker at `cluster.close()`. None
of them means a worker stopped under memory pressure. The `pressure_fraction` near 1.0 in those dumps
is a snapshot taken at shutdown.

### This instance has no headroom left

```
wall-clock 1187.2 s   workers 16   tasks 324
compute    18,740.7 s      scheduler 18.5 s
worker.exec.call 18,699.8 s      worker.exec.gil 2.1 ms
```

Slot utilisation reaches 15.8 of 16, or **99%**. The Global Interpreter
Lock (GIL) costs 2.1 milliseconds. Scheduler overhead comes to 18.5 s out of 1,187.
`observe stragglers` reports every worker within 0 s of the median. The run
spills nothing and transfers nothing.

**More cores is the only lever left.** Processor time bounds the work, so the
compute cost per tile does not change as wall time falls. A full tile comes to
1,296 shards, four times the quarter tile, because it covers four times the
area. The compute cost of one shard does not change with tile size either.
Revisit rate sets how many scenes a shard reads.

| hardware for one tile | compute | projected compute cost |
|---|---|---|
| 1 x `c6i.16xlarge` (64 vCPU) | ~20 min | $0.88 |
| 2 x `c6i.16xlarge` | ~10 min | $0.88 |
| 4 x `c6i.16xlarge` | ~5 min | $0.89 |

Those three figures project from compute time alone. They exclude boot, install,
the catalogue search, and every S3 request charge. The measured fleet cost $1.94
of EC2 time and $4.28 in total. Shard size, not instance count, moves the larger
half. See `Cost`.

`c6i` outperformed the `r6i` of the early measurements on both axes. The peak was
26.5 GiB of 96, so the memory-optimised instance rented RAM the job never
touched.

### The tails are the thinly observed pixels

| band | share of tile | mean observations per pixel |
|---|---|---|
| 0-60 C | **99.967%** | 173.3 |
| under -20 C or over 75 C | **0.027%** | **6.0** |

p1 is 29.7 C, p50 is 46.4 C, and p99 is 53.3 C. The extreme values fall on
pixels with about six observations, against 173 elsewhere, where the p95 conveys
no information. `qa_count` is the band that lets a consumer drop them, which is
why the writer emits it without a nodata value.

## Tuning results that still hold

These come from the department run at 711 scenes, before the shard design. The
measurements stand. Where the shard design supersedes a conclusion, the text
says so.

The department headline: **93.6 s of compute, 139.3 s end to end** on one
`m6i.4xlarge` in `us-west-2`, reading `s3://usgs-landsat` in region. Peak memory
was 31.8 GiB, and every output pixel came out valid.

```
uv run profile_lst_p95.py \
  --source earth-search \
  --load-chunk 1024 --chunk 512 \
  --workers 8 --threads-per-worker 4 \
  --memory-limit-gib 6
```

### Reads dominate the pipeline

Time inside `worker.exec.call`, by task prefix:

| task | share of execution |
|---|---|
| `lwir11` read | 78.3% |
| `qa_pixel` read | 19.0% |
| `custom_nanquantile` (the p95 itself) | 1.7% |
| everything else | 1.0% |

`worker.exec.gil` accounts for 0.1% of execution. The workers seldom occupy the
GIL, so threads come close to free and the processor is not the constraint. The
percentile costs 1.7%. Decoding the cloud-optimised GeoTIFFs costs 97.3%.

The domestic link, not the reader stack, set the local ceiling. A bare `urllib`
thread pool with no geospatial code in the path measured 5.53 MB/s at 32
concurrent range requests. The full `odc` pipeline measured 5.2 to 5.8 MB/s.

| concurrency | MB/s | latency p50 |
|---|---|---|
| 1 | 0.97 | 709 ms |
| 16 | 4.87 | 2,839 ms |
| 32 | 5.53 | 4,400 ms |
| 64 | 5.42 | 8,108 ms |
| 128 | 9.30 | 11,988 ms |

### Load big blocks

Large blocks speed up reads. The source uses Universal Transverse
Mercator (UTM) and the destination does not, so every destination block needs a
skewed source window plus an edge halo. Neighbouring blocks then re-fetch the
same source tiles.

At 24 scenes locally:

| read / reduce | compute | bytes moved |
|---|---|---|
| 256 / 256 | 107.4 s | ~591 MB |
| 512 / 512 | 60.7 s | ~334 MB |
| 1024 / 1024 | 48.1 s | ~236 MB |
| **1024 / 256** | **40.5 s** | **211 MB** |

Against the original 4 workers x 2 threads at chunk 256, that is **175.3 s down
to 40.5 s, a 4.3x speedup**. Moving 2.8x less data produced it. Throughput never
improved. When bandwidth binds the job, move less. The reduce half of that
recommendation no longer applies: `--chunk` never controlled the reduce block,
and the shard design has no reduce block. See `Corrections`.

Round Trip Time (RTT) from the laptop:

| endpoint | TCP RTT |
|---|---|
| Planetary Computer blob, `landsateuwest` | 7 to 21 ms |
| AWS S3, `us-west-2` | 153 to 254 ms |
| AWS S3, `eu-central-1` | 47 ms |

Running from Europe against `us-west-2` raises latency 12x while raising
bandwidth 23x. Little's Law then puts a gigabit link at 33 concurrent streams
for 1 MB requests, or about 509 for 64 KB requests. Blocking readers give one
request per thread, so that regime needs an async reader. **Running in region
removes the problem instead.** At sub-millisecond RTT the requirement drops to
between 1 and 11 streams, and ordinary threads cover it. `odc.loader` exposes
`register_driver` and a `ReaderDriver` protocol if a future workload ever needs
an async path.

### More processes, not more cores

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

Every run produced identical output. The topology runs pinned total threads at
32. Only the process split moved, on the same 16 vCPUs.

| workers x threads | CPU used | compute |
|---|---|---|
| 1 x 32 | 22% | 190.0 s |
| 2 x 16 | 33% | 160.4 s |
| 4 x 8 | 44 to 55% | 112.6 to 124.0 s |
| 8 x 4 | **76%** | **93.6 s** |

Processor use more than tripled and wall time halved, with no change in core
count or thread count. The cores had idled all along. One process could not feed
them, because sends serialised behind one queue and memory concentrated into one
spill manager.

Consolidating to one worker eliminated worker-to-worker transfer, which zero
`tcp.recv.queue`, `tcp.send.queue`, and `worker.transfer.recv` spans confirm. It
still lost, and `worker.exec.call` rose from 1,728 s to 2,777 s for identical
work. Fewer processes performs worse here even when the shuffle costs nothing.
On the 2-worker run `frisky observe transfers` shows the wire moving 9.58 GiB in
12.8 s while queueing costs 470 s. The send queue alone came to 118.9% of one
worker's wall-clock capacity. Adding workers adds parallel transfer channels.

The shard design supersedes this result. A sharded run transfers 0 B, so the
queue disappears rather than shrinking.

### Run-to-run variance is 9.6 percent

The control runs differed by 11.4 s on a mean of 118.3 s. That is the noise
floor, and it governs how to read every other number here.

- The 8-worker gain of 24.7 s is 2.2x the noise. It holds, and processor
  utilisation corroborates it mechanistically.
- The chunk 256 versus 512 difference, 12.1 s or 8.4%, falls **inside** the
  noise floor. This evidence does not make 512 better than 256.
- The chunk 128 versus 256 difference of 59% falls far outside the noise and
  holds.

Any single-run comparison in this work has about ten percent uncertainty.

## Cost

### Full-tile figures

MEASURED:

| | |
|---|---|
| instances | 4 x `c6i.16xlarge`, `ami-04678417fc39d7171`, us-west-2b |
| launch / stop (UTC) | 06:07:33 / 06:18:15 |
| lifetime | **642 s each, 2,568 instance-seconds** |
| compute, four slices | 273.5 + 289.7 + 284.6 + 247.0 = **1,094.8 s** |
| catalogue search, per machine | **38.1 s** over 3,910 items |
| shard-scene reads, full tile | **605,617** |
| non-compute residual, per machine | **368 s** (boot, install, search, part write, spans) |

DERIVED, from the pinned us-west-2 Linux on-demand list rate. Linux bills per
second past a 60-second minimum, so `cost = instance_seconds / 3600 x rate`. The
Elastic Block Store (EBS) line covers four 150 GB root volumes.

| line | formula | cost |
|---|---|---|
| S3 GET requests | 605,617 x 2 x 4.77 / 1000 x $0.0004 | **$2.31** |
| EC2, full-tile fleet | 2,568 s / 3600 x $2.72 | **$1.94** |
| Public IPv4 | 0.7133 instance-hours x $0.005 | $0.014 |
| EBS | 4 x 150 GB x 642 s / 2,628,000 x $0.08 | $0.012 |
| S3 to EC2 transfer, same region | | $0.00 |
| **total** | | **$4.28** |

The S3 line is the largest one, and it exceeds the compute. Earlier versions of
this document guessed at it instead of counting it.

The $2.72/hr rate now has a **VERIFIED** label. It matches the AWS public price
list, which needs no credentials. Cost Explorer still returns `AccessDenied`, so
no billed figure exists to check against. Every total here is arithmetic over a
verified rate.

### The S3 request count, measured

MEASURED by `measure_s3_requests.py`, which counts the requests GDAL puts on the
wire. It sampled three shards across the quarter-tile plan, 60 scenes each, and
drew evenly from each shard's item list.

| shard edge | requests per band read | mean | requests per scene-megapixel |
|---|---|---|---|
| 512 px | 4.60, 4.70, 5.00 | **4.77** | **36.37** |
| 1024 px | 6.50, 7.00, 7.00 | **6.83** | **13.03** |

Every run came back clean. All 4,176 responses were `206 Partial Content`, every
one carried `x-amz-request-charged: requester`, and nothing retried or failed. A
retry would inflate the count, so the script counts 4xx, 5xx, and retries, and
marks a run that has any.

The number an earlier version called "not 2, not 4, and not 8" is **4.77** at the
512 px shard the full tile ran on. Across the tile that is 605,617 reads x 2
bands x 4.77, or **5,777,586 GETs**, and **$2.31**.

That closes the lower bound. The full tile cost **$4.28**. S3 requests are 54% of
it and EC2 is 45%.

#### The measurement ran on a laptop

It ran on the laptop against `us-west-2`, not in region. The count follows from
the GDAL settings and the block layout, not from latency, and the clean 206 tally
rules out retries adding to it. The reads call
`shard_lst_p95.configure_read_env`, so the settings match the pipeline exactly.
An in-region repeat would cost about $0.05 and would confirm the figure rather
than move it.

```bash
uv run measure_s3_requests.py --shards 3 --max-scenes 60 --shard 512
uv run measure_s3_requests.py --shards 3 --max-scenes 60 --shard 1024
```

`--max-scenes` samples evenly across each shard's items rather than taking the
first N, because how many blocks a scene touches depends on where its footprint
falls on the shard. The output is a ratio per band read, so the cap lowers the
spend without changing what the run measures.

### Steady state across 520 tiles

The four-machine run is a validation shape. It paid boot, install, and the
38.1 s catalogue search four times over to buy wall clock. Projecting 2,568
instance-seconds per tile across 520 tiles would overstate the total.

Steady state runs one tile per instance, back to back, so boot and install
amortise away:

```
per tile = compute 1094.8 s + search 38.1 s + tail
tail = part write + span query, bracketed 60-240 s
     = 1,193 to 1,373 s  =  0.331 to 0.381 instance-hours
```

| 520 land tiles, **EC2 only** | |
|---|---|
| on-demand @ $2.72/hr | **$469 - $539** |
| spot @ ~$0.95/hr | **$164 - $188** |

S3 requests add $2.31 per tile at the 512 px shard and $0.87 at 1024 px. Those
charges do not amortise, because they scale with reads rather than with instance
time.

| 520 land tiles | shard 512 | shard 1024 |
|---|---|---|
| EC2, on-demand @ $2.72/hr | $469 - $539 | $469 - $539 |
| EC2, spot @ ~$0.95/hr | $164 - $188 | $164 - $188 |
| S3 GET requests | **$1,202** | **$453** |

At the shard size the full tile ran on, S3 requests cost more than twice the
on-demand compute. On spot they cost more than six times it. Shard size is the
largest cost lever in this pipeline. "Corrections to earlier versions of this
document" prices it and states what it costs in memory.

The whole session, across five EC2 sessions, eight completed department runs,
one 200-scene quarter-tile smoke run, and two quarter-tile attempts that never
finished, cost about **$4.45**.

### How to price a run

`cost_report.py` produces the report deterministically. It reads lifetimes from
the EC2 API, labels every figure MEASURED, DERIVED, or UNKNOWN, and prints the
formula behind each derived line. It **refuses to price S3 without a measured
requests-per-read**, reporting a lower bound instead of a total. It enforces two
rules, both of which this session learned the expensive way.

- **Runtime comes from `LaunchTime` and `StateTransitionReason`, never from
  elapsed feel.** Polling loops return instantly, so wall clock runs far ahead
  of any sense of it. That distortion produced a wrong cost figure once, and it
  made progress look stalled three times during the run.
- **An estimate never appears as a billed cost.** This document pins every rate
  to a published list price.

AWS records the lifetime exactly, and terminated instances stay queryable for
about an hour:

```bash
aws ec2 describe-instances --region us-west-2 \
  --filters "Name=tag:purpose,Values=<tag>" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,StateTransitionReason]' \
  --output text
```

After that hour both fields disappear and the API can no longer supply the
lifetime. **Run the report immediately after teardown**, or capture the instance
records before stopping the instances. The fleet in this document aged out
first, so `--recorded` takes the lifetimes directly:

```bash
./cost_report.py --tag purpose=lst-benchmark --region us-west-2 \
    --profile radiant-earth --ebs-gb 150 \
    --recorded c6i.16xlarge:4:642 \
    --shard-scene-reads 605617 --requests-per-read 4.77
```

That command produced the $4.28 in the preceding table. An earlier version of
the script exited when the API returned no instances, which blocked the S3 line
as well. It now labels the EC2 lines OMITTED and prices what it can. Terminated
instances also stop reporting `BlockDeviceMappings`, so EBS reads as zero unless
the caller passes `--ebs-gb`.

Then price every line, not only EC2:

| line | why it is easy to miss |
|---|---|
| EC2 | the only one anyone remembers |
| EBS | small here, but it scales with volume size and with lifetime |
| Public IPv4 | $0.005/hr per address since 2024, per instance |
| **S3 GET** | requester-pays, and the largest line at a 512 px shard |
| data transfer | free from S3 to EC2 in region, and not free across regions |

### Establishing rates without credentials

`pricing:GetProducts` and `ce:GetCostAndUsage` both return `AccessDenied` for
this SSO role, so nobody has ever checked anything in this repository against a
bill. The EC2 rate escapes that limit. AWS publishes on-demand rates as a public
JSON
file that needs no auth:

```
https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/
  ec2-ondemand-without-sec-sel/US%20West%20(Oregon)/Linux/index.json
```

The server gzips it despite the `.json` name. It lists 1,322 instance types and
confirms `c6i.16xlarge` at **$2.72/hr**, matching the pinned value.
`cost_report.py` fetches it at run time and prints
`VERIFIED from AWS public price list`. It falls back to the pinned table when
the fetch fails, and it says so. The equivalent S3, EBS, and IPv4 endpoints use
different URL shapes, and nobody has located them. Those three rates remain
published but unverified, and together they came to under 1% of this run.

Reconciling against a real bill needs the one extra permission.
`ce:GetCostAndUsage` is read-only and cheap to grant. With it, tag every
instance with a run id, wait 24 to 48 h for Cost Explorer to settle, then
compare the derived figure against the billed one:

```bash
aws ce get-cost-and-usage --time-period Start=<day> End=<day+1> \
  --granularity DAILY --metrics UnblendedCost UsageQuantity \
  --group-by Type=DIMENSION,Key=SERVICE
```

That converts list-price arithmetic into a verified model.

## Defects, and how measurement exposed them

### Defects in the original workflow

Measurement found each of these in the starting snippet.

**1. No spatial chunking.** `chunks={"time": 10}` leaves the spatial dimensions
whole. `quantile` collapses the time axis into one chunk, so one block becomes
2985 x 2845 x 711 float32 values, or 22.1 GiB. The job cannot run. Chunk
spatially.

**2. Digital number 0 is the fill value.** The `lwir11` band uses 0 for fill.
Unmasked, it decodes to -124 C and drags the percentile down. Mask `dn != 0`
alongside the QA bits.

**3. `.where()` applied to the entire Dataset.** Masking the Dataset rather than
the band also masks `qa_pixel` and promotes it to float. Mask `lwir11` alone, and cast to
float32 explicitly so the stack never becomes float64 without warning.

**4. No platform filter.** At `cloud_cover < 100` the search returns 913 scenes,
and **240 of them are Landsat 7**, which has no `lwir11` asset. Its thermal band
is `lwir` (ST_B6), and its Scan Line Corrector (SLC) has left scan gaps since
2003. Those 240 scenes load as nodata while still enlarging the graph. Filter to
`platform in ["landsat-8", "landsat-9"]`. The original `cloud_cover < 20` filter
hid this.

**5. `max` is not `p95`.** The snippet computes `max(dim="time")`. A percentile
is a different and more expensive reduction, because it forces the time axis
into one chunk.

**6. One chunk size for two jobs.** Large blocks speed up reads. Small blocks
speed up the percentile. Separating the two was the largest optimisation at
department scale.

### Rehearsal mode finds them on a laptop

Almost every failure in this session was findable on a laptop, and this session
kept finding them on billed instances instead. The list covers an argparse flag,
a projection and resolution pairing, an ignored chunk argument, wrong spatial
dimension names, a 6.3M-task graph, and threads multiplying to 128 on 16 cores.

`--rehearse N` runs the entire pipeline with N synthetic scenes and no object
store. It exercises shard planning, item filtering, cluster startup, submission,
gather, assembly into the output raster, part writing, and merge. Read
throughput is the one thing it cannot cover, and that is the one question that
needs the cloud.

It found two bugs that would have wasted a four-instance run.

**`--shard-slice` sliced the filtered list, not the plan.** The pipeline drops
shards with no overlapping scene before slicing, so indices shift and the last
slice came up 36 shards short. Real tiles would have carried gaps wherever ocean
and edge shards have no scenes. The slice now applies to the plan.

**The writer skipped a barren shard instead of writing it.** That makes "no
Landsat coverage here" indistinguishable from "a machine stopped", which defeats
the coverage check. The writer now emits barren shards as nodata.

After both fixes, four slices of an 18,000 x 18,000 tile cover
**324,000,000 px exactly**, each pixel once. They merge to **100.00% coverage,
1,296 of 1,296 shards**, and dropping one slice reports
`75,168,000 px never written`. A merge round-trip reproduces the source array
bit for bit, and an incomplete merge exits non-zero. The merge peaks at 8.2 GB.

Build the graph on a laptop first. It takes about two minutes and needs no
instance.

### Sharp edges in the harness

**`graph_stats` does not scale.** It runs `dask.optimize` over the whole graph,
which costs a few seconds at 44k tasks and 139.6 s at 6.3M, reading no imagery
in that time. `--no-graph-stats` disables it, and it should stay off past about
200k tasks. Because the stage sits between graph build and compute, nobody can
separate the pipeline from this instrumentation as the cause of the quarter-tile
stall. The array path never completed a quarter tile. The sharded path completed
one twice.

**Frisky workers survive `pkill` by script name.** They spawn through
multiprocessing, so their command line shows a bare `-c`. An RSS threshold
misses the small ones. Orphans contaminated two measurements here. Eight of them
held 88 GB across two restarts and produced a wrong conclusion. Later, 26.6 GB
of dead cluster made a healthy run look like a failing one, with two schedulers
listening on different dashboard ports. Match on the venv path, stop by process
id, and confirm against `ss -tlnp` that one dashboard listens:

```bash
ps -eo pid,rss,args --sort=-rss \
  | awk '$2 > 200000 && /environments-v2/ {print $1}' \
  | xargs -r kill -9
```

**Concurrency knobs multiply.** `--workers 8 --threads-per-worker 4` gives 32
shard slots, and `--read-threads 4` multiplies that to 128 OS threads on 16
cores. Measured: 15 of 324 shards in 400 s, with the first wave of 64 crawling.
Sizing slots to cores gave 15 shards in 105 s, 3.8x faster. The script now
refuses to start past `cores * 6` threads.

**`client.gather` on every future at once aborts the process.** With 324 futures
and about 1 GB of results resident, frisky 0.7.2 raised a Rust panic across the
PyO3 boundary at 90% completion. A panic cannot unwind, so the process aborts
rather than raising, and 290 completed shards went with it. `frisky.as_completed`
with one result assembled at a time completed 324 of 324 with no panics.

**The wrong projection produced the wrong raster.** The quarter-tile runs used
`--crs epsg:3857 --resolution 30`, carried over from the department work. The
grid is **EPSG:4326 at 1/3600 degree**. Web Mercator inflates area by
`1/cos(lat)`, so the raster came out 11160 x 9278 rather than 9000 x 9000: 104
Mpx instead of 81, on the wrong projection. Pass the grid CRS explicitly.

**The harness pipes stdout through `grep`, which block-buffers.** No stage line
printed until the run ended, so three EC2 runs went diagnosed by watching
processor percentage and RSS, and by guessing.

### Defects that would have produced a wrong number

Both sat in `measure_s3_requests.py`, and neither would have announced itself.

**The measurement did not use the pipeline's read settings.** The script set
`CPL_CURL_VERBOSE` and nothing else. The real run also sets
`GDAL_DISABLE_READDIR_ON_OPEN`, `GDAL_HTTP_MULTIRANGE`,
`GDAL_HTTP_MERGE_CONSECUTIVE_RANGES`, `VSI_CACHE`, and `AWS_REQUEST_PAYER`. The
first suppresses a directory listing on every open and the third collapses
adjacent block reads into one request, so both move the count directly. Without
the last, every read returns 403. Both scripts now call
`shard_lst_p95.configure_read_env`, so the measurement cannot drift from the
pipeline it describes.

**`redirect_stderr` captured nothing.** rasterio installs a CPL error handler,
so GDAL's curl output never reaches stderr. It arrives as `rasterio._err` log
records shaped `CURL_INFO_HEADER_OUT: GET ...`, and redirecting file descriptor
2 does not catch them either. The script attaches a log handler instead. Left
alone it would have reported 0 GETs, and a zero reads like a pipeline that
issues no requests rather than like a broken counter. The script now exits
non-zero on a zero count instead of writing one to a file that `cost_report.py`
would price.

### Sharp edges in the cluster library

- `memory_limit` takes an integer count of **bytes, per worker**. It does not
  parse the string `"4GB"`.
- `processes` defaults to `False`, the opposite of `distributed.LocalCluster`.
- Workers start with `spawn` and read `FRISKY_TRACING_CAPACITY` from the
  environment, so set it before constructing the cluster.
- Get the client from `cluster.get_client()`. `frisky.Client(cluster)` raises.
- No public span context manager exists. Build one on `record_span` and `now_ns`.
- Use `query_spans`, not `get_spans`. With `processes=True` the worker spans
  live in the worker processes.
- `odc.loader.configure_rio(client=...)` accepts `client` and discards it, so
  GDAL settings must reach spawned workers through `os.environ`.
- Planetary Computer Shared Access Signature (SAS) tokens last about an hour,
  and the catalogue signs them at search time. The script signs immediately
  before building the graph, so the token clock starts at compute rather than at
  search. Earth Search needs no signing, because `s3://` hrefs authenticate per
  request from ambient AWS credentials with `AWS_REQUEST_PAYER=requester`.

### Instrumentation

`profile_lst_p95.py` is a standalone PEP 723 script that reads no cache and
writes no raster. It times nine stages, from imports through compute to the
report, recording wall time, processor time, RSS, and `ru_maxrss` for each. A
separate sampler process polls at 100 ms for client and per-worker RSS, system
memory, swap, and the network and disk counters. The sampler runs in its own
process because the client occupies the GIL through graph construction, and an
in-thread sampler drifts at exactly the interesting moment.

Frisky spans supplied every causal finding here. The psutil sampler supplied
throughput and memory. Both sources were necessary.

Use the `frisky observe` CLI rather than parsing `summary.json` by hand. The
funnel runs `overview`, then `transfers` or `prefixes`, then `stragglers` and
`timeline`. Every command accepts a saved spans file, so analysis continues
after the cluster goes away. Hand-aggregating span buckets mixes scheduler
traffic with worker shuffle and produced wrong conclusions here more than once.

Span collection is not free. It cost 15.8% of the run at 1.8M spans and 5.0% at
500k. Cap `--span-limit` around 500k.

## What is not settled

- **Nobody has run a tile at the 1024 px shard.** The 2.8x request saving rests
  on three shards of a read measurement. The 90.6 GiB working set it implies has
  no run behind it, against a measured 26.5 GiB peak at 512 px.
- **The request count came from the laptop, not from in region.** Every response
  was a clean 206, so latency added nothing to the count, but no in-region repeat
  exists. It would cost about $0.05.
- **The request count covers one area at one date range.** Requests per band read
  follows from where the shard window falls on the internal block grid, so a
  different grid origin or resolution can move it.
- **Nobody has run past 64 cores per instance.** The prediction that a 384 px
  shard edge keeps the working set inside RAM has no measurement behind it.
- **Chunk 512 versus 256 remains unresolved** at department scale. The
  difference falls inside the 9.6% noise floor.
- **The department tuning covers one department at 711 scenes.** A different
  area or scene count moves the optimum, because the memory term scales with
  both.

## Corrections to earlier versions of this document

Every entry here is a claim an earlier version stated as fact. Each one shares
one mistake: it presented an estimate as a measurement.

**The full tile cost $10.70.** Wrong. That figure assumed each instance ran an
hour. Each ran 642 s. The measured fleet cost $1.94 of EC2 time.

**S3 GET charges add about $0.09 per tile, which puts a global composite near
$650.** Wrong, and withdrawn. Both numbers came from an invented
requests-per-read multiplier. The same multiplier produced a global figure of
"$1,515 - $3,030". Do not use any of the three. `measure_s3_requests.py` has
since counted the requests: 4.77 per band read at a 512 px shard, or $2.31 per
tile.

**Larger `--load-chunk` pays twice, because request count is now a cost
driver.** Withdrawn as written, and now reinstated on a measurement instead of
the guess that produced it. The knob is `--shard`. The sharded pipeline has no
`--load-chunk`, and an earlier version of `measure_s3_requests.py` carried a
`--load-chunk` flag that it never used.

Doubling the shard edge from 512 px to 1024 px raises requests per band read
from 4.77 to 6.83. It cuts requests per scene-megapixel from 36.37 to 13.03.
Across a full tile that is 2,177,623 GETs against 5,777,586, or $0.87 against
$2.31.

Read the two ratios in that order. Requests per band read has to rise with shard
size, because each read covers four times the area. Requests per scene-megapixel
is the ratio that prices a tile, and it falls by 2.8x.

The larger shard is not free. The dry run puts the worst 1024 px shard at
2.83 GiB and the 32-slot working set at 90.6 GiB against 128 GiB of RAM, where
the 512 px plan measured a 26.5 GiB peak. The saving has a price, and no run
has yet
proven at tile scale.

**Rechunk memory is `chunk² x scenes x 4 x 2`, it scales linearly with scene
count, and reduce chunk 512 needs 327 GiB at 3,910 scenes.** Wrong. A local dry
run costing nothing showed why. `--chunk` never controlled the reduce block.
Dask normalises every block to its `array.chunk-size` target, 128 MiB by
default, during the rechunk that `quantile` forces:

| scenes | chunk requested | block after `.chunk()` | block after `quantile` | size |
|---|---|---|---|---|
| 711 | 256 | 256 | **217** | 134 MB |
| 711 | 512 | 512 | **217** | 134 MB |
| 1,765 | 256 or 512 | n/a | **137** | 133 MB |
| 3,910 | 256 or 512 | n/a | **92** | 132 MB |

At 711 scenes, chunk 256 and chunk 512 produce an **identical** 217-pixel reduce
block. The block auto-shrinks as scene count rises, which holds per-block memory
near constant. Measured peak RSS moved little across chunk 128, 256, and 512:
35.5, 30.7, and 28.4 GiB. Reduce chunk 512 remains feasible at
tile scale. `--chunk` does change the *intermediate* rechunk, and so the task
count, which is what the measured wall-time differences reflect. Those runs
produced real numbers. The explanation attached to them did not survive.

**Read columns pin memory in proportion to scene count. That is what drove eight
workers to 11 GB each before any read.** A guess, made without
instrumentation. The observation itself stands. At 1,765 scenes the eight
workers together held 96 GiB before any imagery arrived. The cause turned out to be 212
seconds of single-threaded client work in graph build and `dask.optimize`, over
a graph of 6.3 million tasks.

## Files

| path | contents |
|---|---|
| `shard_lst_p95.py` | the sharded pipeline, the slicer, and the merge |
| `profile_lst_p95.py` | the array-graph profiling harness |
| `sweep_throughput.py` | configuration sweep driver |
| `cost_report.py` | the labelled, deterministic cost report |
| `measure_s3_requests.py` | counts the S3 GET requests one shard issues |
| `s3-requests/` | the request measurement: both shard sizes, and the priced tile |
| `dryrun/` | local graph-build runs, no cluster and no reads |
| `ec2-results/` | eight department-scale runs: stages, memory series, frisky reports |
| `fulltile/` | the full-tile run (part files and merged raster gitignored) |
| `smoke/`, `smoke2/`, `split/`, `sweep/` | local runs with full frisky spans and traces |
| `full/`, `full.log` | the 673-scene local run, ended early by a network change |

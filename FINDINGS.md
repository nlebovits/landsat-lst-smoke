# Profiling a p95 composite of land surface temperature

Where the time, the memory, and the money go in a p95 composite of Land Surface
Temperature (LST) over five years of Landsat Collection 2 Level 2 scenes.
Session `b5dd543f-2974-40c1-9f4c-5892e87c8d48`, 2026-09-08. A label marks each
figure MEASURED or DERIVED. `Corrections` records what earlier versions of this
document got wrong.

## Headline

Four `c6i.16xlarge` instances built the full tile `S30W065`, 18,000 x 18,000 px
over 3,910 scenes, in **4.8 minutes of wall clock** for **$4.28**.

| slice | shards | compute | per shard |
|---|---|---|---|
| 0 | 324 | 273.5 s | 0.84 s |
| 1 | 324 | 289.7 s | 0.89 s |
| 2 | 324 | 284.6 s | 0.88 s |
| 3 | 324 | 247.0 s | 0.76 s |

The slowest slice fixes the wall clock. Zero errors on all four. The merge
assembled 1,296 of 1,296 shards at **100.00% coverage** in 13.3 s, peaking at
8.26 GB.

```
LST p95   min -49.7 C   mean 45.8 C   max 90.6 C   (100.0% valid)
```

Read the two ends of that range as evidence of a defect, not as temperatures.
The run used a mask that let cirrus, dilated cloud, and snow through, and an
encoder whose floor was DN 1. Under the mask the pipeline now applies, -49.7 C
sits below the trusted minimum and 90.6 C sits above any land skin temperature,
so both become nodata. The figures stand as what was measured. They are not what
the pipeline would write today.

S3 requester-pays requests are 54% of that $4.28 and EC2 is 45%. Shard size, not
instance count, moves the larger half.

## What to run

Build the artifacts once, on a laptop, before any instance starts:

```bash
# the authoritative tile list, from the pixel mask's own land geometry
uv run land_tiles.py --out artifacts/land_tiles.parquet

# the scene inventory, from the USGS bulk metadata Parquet
uv run usgs_inventory.py \
    --land-tiles artifacts/land_tiles.parquet \
    --out artifacts/tile_scene_inventory.parquet

# what the fleet will launch, and the checks that gate it
uv run fleet_plan.py --out artifacts/fleet_plan.json
```

`usgs_inventory.py` reuses an unchanged USGS download. Pass `--refresh` to
discard the cache and fetch the file the service is serving now.

The figures this document quotes about the land rule and the acquisition time
come from measurement scripts, which no build runs:

```bash
# what each defect in the shared land method selects
uv run measure_land_defects.py --out artifacts/land_defects.json

# how far the computed scene centre sits from the published one
uv run measure_scene_centre.py --all-years \
    --out artifacts/scene_centre_offset.json
```

Then stage both Parquet files where the fleet can read them, and run one
machine per tile:

```bash
# one machine per slice
uv run shard_lst_p95.py --tile S30W065 \
    --inventory-uri artifacts/tile_scene_inventory.parquet \
    --pixels-per-degree 3600 \
    --shard-slice 0:324 --out-dir ./part0
uv run shard_lst_p95.py ... --shard-slice 324:648  --out-dir ./part1
uv run shard_lst_p95.py ... --shard-slice 648:972  --out-dir ./part2
uv run shard_lst_p95.py ... --shard-slice 972:1296 --out-dir ./part3

# then anywhere
uv run shard_lst_p95.py --merge part0 part1 part2 part3 --out-dir ./tile
```

The shard plan is deterministic and anchors to whole degrees, so a shard covers
the same pixels whichever request produced it. Machines need no coordination
beyond the slice index. Rehearse the same fleet on a laptop first, where
`--rehearse N` substitutes N synthetic scenes and touches no object store:

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
| Boundary | `pergamino_dept.gpkg`, Pergamino department, Buenos Aires |
| bbox, EPSG:4326 | `(-60.942796, -34.17645991, -60.138771, -33.54020044)` |
| Department grid | EPSG:3857 at 30 m, 2985 x 2845 px, 8.48 Mpx per scene |
| Window, as measured | 2020-01-01 to 2025-01-01 |
| Window, current default | 2021-01-01 to 2025-12-31T23:59:59Z |
| Cloud filter | `eo:cloud_cover < 100` |
| Scenes | **711** via Earth Search, **673** via Planetary Computer |
| Volume | 8.7 MB per scene, about 6.1 GB per department run |

Earth Search returns 38 more scenes than Planetary Computer for the same query.
The difference has no known cause. Treat the two counts as incomparable.

Every measurement in this document ran against the first window. The scripts now
default to the second one, which is the five-year composite the asset spec asks
for. So every scene count, timing, request count, and cost below describes
2020-2024, not 2021-2025, and none has been rerun. `Corrections` explains the
change. Label any figure measured against the new window where it appears.

The sibling `landsat-lst` repository defines the production grid. A tile spans
5 degrees on EPSG:4326 at 3600 px per degree, under a name such as `N40W075` or
`S30W065`. The convention excludes the south edge and includes the north.
Pergamino falls inside `S30W065`, lat (-35, -30], lon [-65, -60). The fourth row
counts Worldwide Reference System (WRS) path and row combinations. A catalogue
search produced every scene count, so none is an estimate.

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
-49.99 C to 605.35 C in 0.01 C steps, and Pergamino lands near 7000 to 9500. A
true -50.00 C would encode to digital number 0 and look identical to nodata,
which never occurs here. `qa_count` has no nodata value by design. A zero means
that masking removed every observation for that month, which differs from a
masked pixel. Keeping the two apart is the point of the band.

## Architecture: shard, do not tune

### The catalogue is read once for the whole fleet

Every tile VM used to open Earth Search and page the same catalogue. The answer
never differed, and the cost was 38.1 s of instance time plus one dependency on
a public service per machine. At 895 tiles that is 895 chances for a run to
fail after it has already paid for its instance.

The catalogue work now happens once, on a laptop, and the fleet reads the
result. The source is the USGS Landsat Bulk Metadata Service file for OLI/TIRS
Collection 2 Level 2, `LANDSAT_OT_C2_L2.parquet.gz`, updated daily.

| | Earth Search, per tile | bulk Parquet, once |
|---|---|---|
| transfer | 100 items per request, 22.4 KB each | 433 MB, one download |
| for the whole band | about 42 GB of JSON | 433 MB |
| wall clock | 38.1 s x 895 machines | 58 s, on a laptop |
| runtime dependency | 895, one per machine | none |

Both inventories describe the same archive. Over 2021 to 2025 inside +/-60
degrees, with `eo:cloud_cover < 100` and Landsat 8 and 9, the bulk file yields
**1,460,446** scenes. `tests/test_inventory_parity.py` enumerates the two sets
over a bounded region and finds no item on either side that the other lacks.

The bulk file has no STAC assets and no `proj:*` fields. Each one is an
exact function of columns it does carry, and each rule is checked against Earth
Search across both platforms, both hemispheres, eight or more UTM zones, the
antimeridian, and all five years:

| field | rule | agreement |
|---|---|---|
| `proj:epsg` | `32600 + UTM Zone` | exact |
| `proj:shape` | corner envelope over 30 m, plus one | exact |
| `proj:transform` | corner envelope, half a pixel out, snapped | exact |
| `lwir11` href | from `Display ID`, null for `L2SR` | exact |
| `qa_pixel` href | from `Display ID` | exact |
| `landsat:scene_id` | `Landsat Scene Identifier` | exact |
| `eo:cloud_cover` | `Scene Cloud Cover L1` | exact |
| `datetime` | centre of the acquisition | within 0.89 s |

Two of those rules are worth stating plainly, because both are easy to get
wrong in a way that produces a plausible answer.

USGS writes **every** scene in a northern UTM zone. A southern scene uses a
negative northing rather than the 10,000,000 m false northing, so choosing
`327xx` by latitude sign is wrong. Measured on a stratified sample, it was
wrong for 112 of 400 scenes.

The transform is **exact, not close**. The corner columns hold five decimal
places, about 1 m, so a reprojected corner lands about a metre from the true
origin. Landsat Level 2 products sit on a 30 m lattice offset by half a pixel,
so snapping to that lattice recovers the origin exactly. Across all 1,460,446
scenes the largest snap moved a corner **0.68 m**, against a 15 m half-pixel.
`MAX_SNAP_METERS` stops the build at 7.5 m, so the reconstruction is checked on
every row rather than argued for.

The one tolerance is the acquisition time, and stating it correctly took
measurement. The centre here is the midpoint of the bulk file's acquisition
start and stop. Earth Search publishes the scene centre from the product
metadata. A definitional gap separates the two, and rounding in the source
widens it; `measure_scene_centre.py` reports each apart from the other.

Timestamp precision in the bulk file is mixed, which is what makes that
possible. Landsat 8 2021 carries microseconds on both timestamps, Landsat 8
2022 is 53% whole-second, and everything later is whole-second. On the
untruncated rows the midpoint is exact, so the residual against Earth Search is
the definitional difference by itself: a systematic **4.24 ms** over 401
scenes, with the median equal to the worst case to four decimal places. On the
truncated rows, truncating both timestamps moves their midpoint by strictly
under a second, and the worst of 199 was **0.89 s**.

So the bound is a second and change, derived rather than assumed. An earlier
version of this document asserted 1.117 s and attributed all of it to
truncation, which no truncation argument allows, and it described the whole
file as whole-second when a sixth of the window is not.

The runtime reads only the month, so the tolerance matters only at a month
boundary. `MONTH_BOUNDARY_GUARD_SECONDS` is 30 s, 34 times the worst case
measured. Over the window, 4 scenes fall within 2 s of a month boundary, 7
within 5 s, and 52 within 30 s. `usgs_inventory` fetches every scene inside the
band from Earth Search and writes its exact time. Those are the only catalogue
requests the whole build makes, and widening the band from 2 s to 30 s bought
the margin for 48 more of them on one laptop.

### What a single-tile read costs

The artifact is 167 MB, 3,083,129 tile-scene rows, sorted by tile and then by
acquisition time, Zstandard-compressed, with one row group per tile and
`tile_id` statistics on every group. A VM specifies its tile. The reader
compares that name against the statistics and reads one group.

| tile | scenes | row groups | uncompressed read | median |
|---|---|---|---|---|
| `S30W065` | 4,776 | 1 of 895 | 2.80 MB of 1,781 MB (0.157%) | 123 ms |
| `N40W075` | 2,935 | 1 of 895 | 1.72 MB of 1,781 MB (0.097%) | 87 ms |
| `N05E010` | 4,187 | 1 of 895 | 2.46 MB of 1,781 MB (0.138%) | 114 ms |

Against the 38.1 s Earth Search baseline that is about 310x, but the wall clock
is not the point. `tests/test_no_stac_at_runtime.py` blocks every socket in the
process and runs the read to completion, which is the property worth having.

There is no fallback. A missing, unreadable, or mismatched artifact raises
before the cluster starts, and the driver refuses to launch. A silent fallback
would turn one wrong argument into 895 machines quietly doing the slow thing,
and every other test here would still pass.

### The array graph fails at tile scale

Two array-graph configurations ran against the real grid on a 128 GiB box, and
neither finished.

`load 2048 / chunk 1024 / time_chunk 50` minimised task count at 260k. It
produced 1.85 TiB of worker output, spilled 14,313 objects (360.12 GiB), and
read 0 MB/s at a 100.8 GiB peak. Every read task waited on 50 dependencies, one
per scene in the time chunk, at 839 MB per block. **Minimising task count was
the wrong objective.** Frisky schedules 250,000 to 400,000 tasks per second, so
even 1.4M tasks amount to about 5 seconds of scheduling. The working set is the
constraint, and `time_chunk 50` multiplied it.

`load 1024 / chunk 512 / time_chunk 10`, the shape that worked at 711 scenes,
read at 69 MB/s with a 3.0 GiB peak and stalled anyway:

```
state    waiting 503,696   processing 17,121   memory 167,209
memory   57.07 GiB / 104.00 GiB
spilled  1.21 TiB   unspilled 373.95 GiB   network recv 225.07 GiB

Transfers  96,229 messages, 112,461 keys
  bytes    241.70 GiB logical -> 162.59 GiB wire
  costs    queue 26.8m, cpu 31.6m, wire 11.8m

Prefix           Msgs     Logical      Wire      Queue      CPU
rechunk-merge  19,893  182.62 GiB  120.77 GiB   12.3m    23.8m
```

**`rechunk-merge` accounts for 76% of all shuffled bytes.** The p95 forces a
reorganisation of `81 Mpx x 1765 scenes x 4 bytes`, or **572 GB of float32**,
from time-major read blocks into space-major reduce blocks. No block size avoids
that half-terabyte all-to-all shuffle. The run spilled 1.21 TiB for a 58 GB
input.

The client stalls before the cluster starts. A local dry run, no cluster and no
reads, shows where:

```
scenes 1765, load_chunk 512, EPSG:3857 @ 30 m
  build_graph        72.68 s
  raw tasks       6,278,518
  dask.optimize     139.63 s
  fused tasks     4,270,657
```

**212 seconds of single-threaded client work precede any dispatch**, over 6.3
million tasks against an estimate of 115 thousand, a 50x miss. Frisky cannot
show it, because both stages run in the client ahead of the scheduler, so no
worker span exists to read.

### The shard plan

A 512 x 512 shard needs `512² x 1765 x 4` = 1.85 GB for its complete time stack.
One worker holds that, and no array crosses a worker boundary, so nothing
rechunks and nothing shuffles. `shard_lst_p95.py` submits one shard as one task
that loads, masks, reduces, encodes, and returns.

| | array graph | **sharded** |
|---|---|---|
| spilled | 1.21 TiB | **0 B** |
| worker-to-worker transfer | 225.07 GiB | **0 B** |
| task results pinned | 167,209 | 324, released as gathered |
| memory peak | 57 GiB / 104 | **26.5 GiB / 96 (28%)** |
| completed | never | **324 / 324** |

### Measured results

| run | instance | slots | shards | scenes | compute | per shard |
|---|---|---|---|---|---|---|
| quarter tile | `r6i.4xlarge` | 16 | 324 | 1,765 | 1,113.9 s | 3.44 s |
| quarter tile | `c6i.16xlarge` | 64 | 324 | 1,765 | 264.8 s | 0.82 s |
| full tile | 4 x `c6i.16xlarge` | 256 | 1,296 | 3,910 | 289.7 s | 0.88 s |

Four times the cores cut the quarter tile 4.2x, from 1,113.9 s to 264.8 s.
Throughput rose from 167 to 358 MB/s, and processor time came to 92% user.

Client Resident Set Size (RSS) peaked at 5.1 GiB on both runs. The output
matched exactly:

```
LST p95   min -17.0 C  mean 44.6 C  max 75.9 C  (100.0% valid)
qa_count  Jan 18.9 ... Jun 12.2  Jul 10.5 ... Dec 16.1
frisky    memory 95.93 GiB / 102.40 GiB (94%)  spilled 0 B  network recv 0 B
```

`qa_count` is the correctness check: every one of the twelve months has
observations, and summer exceeds winter for a southern-hemisphere site.

Memory ran at 94% of the configured limit, so `--memory-limit-gib 1.6` across 64
workers is the practical floor at 512 px shards and 1,765 scenes. Past 64 cores
the shard edge should shrink, because a 384 px shard needs 0.39 GiB rather than
0.69. One diagnostic invites a misreading. A clean 64-worker run ends with 64
`frisky_worker_sigterm_dump` entries, one per worker at `cluster.close()`. None
of them means a worker stopped under memory pressure.

### This instance has no headroom left

```
wall-clock 1187.2 s   workers 16   tasks 324
compute    18,740.7 s      scheduler 18.5 s
worker.exec.call 18,699.8 s      worker.exec.gil 2.1 ms
```

Slot utilisation reaches 15.8 of 16, or **99%**. The Global Interpreter
Lock (GIL) costs 2.1 milliseconds and the scheduler 18.5 s of 1,187. `observe
stragglers` reports every worker within 0 s of the median. The run spills 0 B
and transfers 0 B.

**More cores is the only lever left for wall clock.** A full tile comes to 1,296
shards, four times the quarter tile, because it covers four times the area.
Revisit rate, not tile size, sets how many scenes a shard reads, so the compute
cost per tile does not change as wall time falls.

| hardware for one tile | compute | projected compute cost |
|---|---|---|
| 1 x `c6i.16xlarge` (64 vCPU) | ~20 min | $0.88 |
| 2 x `c6i.16xlarge` | ~10 min | $0.88 |
| 4 x `c6i.16xlarge` | ~5 min | $0.89 |

Those three figures project from compute time alone, excluding boot, install,
the catalogue search, and every S3 request charge. The measured fleet cost $1.94
of EC2 time and $4.28 in total. `c6i` outperformed the `r6i` of the early
measurements on both axes, because the 26.5 GiB peak of 96 means the
memory-optimised instance rented RAM the job never touched.

### The tails are the thinly observed pixels

| band | share of tile | mean observations per pixel |
|---|---|---|
| 0-60 C | **99.967%** | 173.3 |
| under -20 C or over 75 C | **0.027%** | **6.0** |

p1 is 29.7 C, p50 is 46.4 C, and p99 is 53.3 C. The extremes fall on pixels with
about six observations against 173 elsewhere, where the p95 conveys nothing.
`qa_count` lets a consumer drop them, so the writer emits it without a nodata
value.

## Tuning results that still hold

The department run at 711 scenes preceded the shard design. Its measurements
stand. The shard design supersedes most of the conclusions drawn from them.

Reads dominate. `lwir11` and `qa_pixel` take 97.3% of `worker.exec.call` and the
percentile takes 1.7%, with `worker.exec.gil` at 0.1%. The domestic link, not
the reader stack, set the local ceiling: a bare `urllib` thread pool with no
geospatial code in the path measured 5.53 MB/s, against 5.2 to 5.8 MB/s for the
full `odc` pipeline. Larger read blocks cut wall time 4.3x at 24 scenes, from
175.3 s to 40.5 s, by moving 2.8x less data rather than by moving it faster. The
reduce half of that recommendation no longer applies, because `--chunk` never
controlled the reduce block and the shard design has no reduce block. See
`Corrections`. Round-trip latency from the laptop runs 7 to 21 ms to the
Planetary Computer blob against 153 to 254 ms to `us-west-2`, so every
production run belongs in region.

All runs below: 711 scenes, `m6i.4xlarge` (16 vCPU, 64 GiB), `us-west-2`,
`s3://usgs-landsat` requester-pays, read chunk 1024, identical output.

| run | workers x threads | reduce | compute | total | MB/s | peak RSS | tasks | CPU used |
|---|---|---|---|---|---|---|---|---|
| chunk 128 | 4 x 8 | 128 | 230.5 s | 309.5 s | 27.0 | 35.46 GiB | 527,721 | |
| chunk 256 | 4 x 8 | 256 | 144.9 s | 196.5 s | 42.5 | 30.69 GiB | 144,833 | |
| chunk 512 | 4 x 8 | 512 | 132.8 s | 157.3 s | 46.3 | 28.36 GiB | 44,157 | |
| 1 worker | 1 x 32 | 512 | 190.0 s | 213.3 s | 32.2 | 30.98 GiB | 44,157 | 22% |
| 2 workers | 2 x 16 | 512 | 160.4 s | 183.5 s | 38.4 | 30.94 GiB | 44,157 | 33% |
| control A | 4 x 8 | 512 | 124.0 s | 168.6 s | 49.5 | 28.22 GiB | 44,157 | 44 to 55% |
| **8 workers** | **8 x 4** | 512 | **93.6 s** | **139.3 s** | **65.9** | 31.83 GiB | 44,157 | **76%** |
| control B | 4 x 8 | 512 | 112.6 s | 154.8 s | 54.6 | 30.20 GiB | 44,157 | 44 to 55% |

Total threads stayed at 32 on the same 16 vCPUs, so only the process split
moved. Processor use more than tripled and wall time halved. One process could
not feed the cores, because sends serialised behind one queue and memory
concentrated into one spill manager. The shard design supersedes the result,
because a sharded run transfers 0 B.

**Run-to-run variance is 9.6%.** The control runs differed by 11.4 s on a mean
of 118.3 s. The 8-worker gain of 24.7 s is 2.2x that noise and holds. The
chunk 256 versus 512 difference of 12.1 s falls inside it, so this evidence does
not make 512 better than 256. Any single-run comparison here has about ten
percent uncertainty.

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
| non-compute residual, per machine | **368 s** (boot, install, search, write, spans) |

DERIVED from the pinned us-west-2 Linux on-demand list rate. Linux bills per
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

The S3 line exceeds the compute. Earlier versions of this document guessed at it
instead of counting it. The $2.72/hr rate now has a **VERIFIED** label against
the AWS public price list. Cost Explorer still returns `AccessDenied`, so every
total here is arithmetic over a verified rate rather than a bill.

### The S3 request count, measured

MEASURED by `measure_s3_requests.py`, which counts the requests GDAL puts on the
wire across three shards of the quarter-tile plan, 60 scenes each, drawn evenly
from each shard's item list.

| shard edge | requests per band read | mean | requests per scene-megapixel |
|---|---|---|---|
| 512 px | 4.60, 4.70, 5.00 | **4.77** | **36.37** |
| 1024 px | 6.50, 7.00, 7.00 | **6.83** | **13.03** |

All 4,176 responses were `206 Partial Content`, every one came back with
`x-amz-request-charged: requester`, and nothing retried or failed. A retry would
inflate the count, so the script counts 4xx, 5xx, and retries, and marks any run
that has them. At the 512 px shard the full tile ran on, 605,617 reads x 2 bands
x 4.77 gives **5,777,586 GETs** and **$2.31**.

```bash
uv run measure_s3_requests.py --shards 3 --max-scenes 60 --shard 512
uv run measure_s3_requests.py --shards 3 --max-scenes 60 --shard 1024
```

It ran on the laptop against `us-west-2`, not in region. The count follows from
the GDAL settings and the block layout rather than from latency, and the clean
206 tally rules out retries. The reads call `shard_lst_p95.configure_read_env`,
so the settings match the pipeline. An in-region repeat would cost about $0.05
and would confirm the figure rather than move it. `--max-scenes` samples evenly
across each shard's items instead of taking the first N, because how many blocks
a scene touches depends on where its footprint falls on the shard.

### Steady state across 895 tiles

The tile count is **895**, generated rather than assumed. `land_tiles.py`
intersects the 5-degree grid inside +/-60 degrees with Natural Earth 10m land
buffered by 25 km, which is the geometry the pixel mask uses. Earlier versions
of this document priced 520 tiles. That figure has no derivation anywhere in
either repository, and `Corrections` withdraws it.

The four-machine run validates rather than produces. It paid boot and install
four times over to buy wall clock, so projecting 2,568 instance-seconds per
tile would overstate the total. Steady state runs one tile per instance, back
to back. The 38.1 s catalogue search is gone: a VM now reads its tile out of a
precomputed inventory in 123 ms.

```
per tile = compute 1094.8 s + inventory read 0.12 s + tail
tail = part write + span query, bracketed 60-240 s
     = 1,155 to 1,335 s  =  0.321 to 0.371 instance-hours
```

| 895 land tiles | shard 512 | shard 1024 |
|---|---|---|
| EC2, on-demand @ $2.72/hr | $781 - $903 | $781 - $903 |
| EC2, spot @ ~$0.95/hr | $273 - $315 | $273 - $315 |
| S3 GET requests | **$2,067** | **$779** |

Read the EC2 rows as DERIVED. Measurement supplies the per-tile compute and the
tile count. The tail is a bracket. The S3 rows multiply the measured per-tile
request cost, $2.31 at a 512 px shard and $0.87 at 1024 px, by the same tile
count.

Removing the per-tile search saves 38.1 s x 895 tiles, or 9.5 instance-hours.
That is $26 on-demand and $9 spot, against an S3 line of $2,067. The search was
never the money. It was 895 dependencies on a public service, one per machine,
each able to fail a run that had already paid for its instance.

S3 charges do not amortise, because they scale with reads rather than with
instance time. At the shard size the full tile ran on they cost more than twice
the on-demand compute, and more than six times the spot compute. Shard size is
the largest cost lever in this pipeline, and `Corrections` prices what it costs
in memory.

The whole session, across five EC2 sessions, eight completed department runs,
one 200-scene quarter-tile smoke run, and two quarter-tile attempts that never
finished, cost about **$4.45**.

### How to price a run

`cost_report.py` reads lifetimes from the EC2 API, labels every figure MEASURED,
DERIVED, or UNKNOWN, prints the formula behind each derived line, and **refuses
to price S3 without a measured requests-per-read**. It enforces two rules, both
of which this session learned the expensive way.

- **Runtime comes from `LaunchTime` and `StateTransitionReason`, never from
  elapsed feel.** Polling loops return instantly, so wall clock runs far ahead
  of any sense of it. That distortion produced a wrong cost figure once, and it
  made progress look stalled three times during the run.
- **An estimate never appears as a billed cost.** This document pins every rate
  to a published list price.

```bash
aws ec2 describe-instances --region us-west-2 \
  --filters "Name=tag:purpose,Values=<tag>" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,StateTransitionReason]' \
  --output text
```

Terminated instances stay queryable for about an hour. After that both fields
disappear and the API can no longer supply the lifetime, so **run the report
immediately after teardown**. The fleet here aged out first, so `--recorded`
takes the lifetimes directly, and this command produced the $4.28 total:

```bash
./cost_report.py --tag purpose=lst-benchmark --region us-west-2 \
    --profile radiant-earth --ebs-gb 150 \
    --recorded c6i.16xlarge:4:642 \
    --shard-scene-reads 605617 --requests-per-read 4.77
```

An earlier version exited when the API returned no instances, which blocked the
S3 line as well. It now labels the EC2 lines OMITTED and prices what it can.
Terminated instances also stop reporting `BlockDeviceMappings`, so EBS reads as
zero unless the caller passes `--ebs-gb`. Price every line, not only EC2: EBS
scales with volume size and lifetime, Public IPv4 costs $0.005/hr per address
since 2024, S3 GET is requester-pays and the largest line at a 512 px shard, and
transfer is free from S3 to EC2 in region but not across regions.

### Establishing rates without credentials

`pricing:GetProducts` and `ce:GetCostAndUsage` both return `AccessDenied` for
this SSO role, so nobody has ever checked anything here against a bill. The EC2
rate escapes that limit, because AWS publishes on-demand rates as a public JSON
file that needs no auth:

```
https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/
  ec2-ondemand-without-sec-sel/US%20West%20(Oregon)/Linux/index.json
```

The server gzips it despite the `.json` name. It lists 1,322 instance types and
confirms `c6i.16xlarge` at **$2.72/hr**, matching the pinned value.
`cost_report.py` fetches it at run time and prints
`VERIFIED from AWS public price list`. When the fetch fails it falls back to the
pinned table, and says so. The equivalent S3, EBS, and IPv4 endpoints use
different URL shapes that nobody has located, so those three rates remain
published but unverified. Together they came to under 1% of this run.

Reconciling against a real bill needs one read-only permission. With
`ce:GetCostAndUsage`, tag every instance with a run id, wait 24 to 48 h for Cost
Explorer to settle, then compare the derived figure against the billed one:

```bash
aws ce get-cost-and-usage --time-period Start=<day> End=<day+1> \
  --granularity DAILY --metrics UnblendedCost UsageQuantity \
  --group-by Type=DIMENSION,Key=SERVICE
```

## Defects, and how measurement exposed them

### Defects in the original workflow

**1. No spatial chunking.** `chunks={"time": 10}` leaves the spatial dimensions
whole, and `quantile` collapses the time axis into one chunk, so one block
becomes 2985 x 2845 x 711 float32 values, or 22.1 GiB. The job cannot run.

**2. Digital number 0 is the fill value.** The `lwir11` band uses 0 for fill.
Unmasked, it decodes to -124 C and drags the percentile down. Mask `dn != 0`
alongside the QA bits.

**3. `.where()` applied to the entire Dataset.** That also masks `qa_pixel` and
promotes it to float. Mask `lwir11` alone and cast to float32 explicitly, so the
stack never becomes float64 without warning.

**4. No platform filter.** At `cloud_cover < 100` the search returns 913 scenes,
and **240 of them are Landsat 7**, which has no `lwir11` asset. Its thermal band
is `lwir` (ST_B6), and its Scan Line Corrector (SLC) has left scan gaps since
2003. Those scenes load as nodata while still enlarging the graph. Filter to
`platform in ["landsat-8", "landsat-9"]`. The original `cloud_cover < 20` hid it.

**5. `max` is not `p95`.** A percentile is a different and more expensive
reduction, because it forces the time axis into one chunk.

**6. One chunk size for two jobs.** Large blocks speed up reads and small blocks
speed up the percentile. Separating them was the largest department-scale win.

### The pixel mask was narrower than the mask it came from

`nlebovits/landsat-lst` masks QA_PIXEL bits 1 to 5: dilated cloud, cirrus,
cloud, cloud shadow, and snow. This repository masked bits 3 and 4, which is
cloud and cloud shadow alone. The narrower mask dropped the rules below.

**Cirrus and dilated cloud stayed in the stack.** Thin cloud and cloud edges
leave per-scene warm and cool residuals behind, and a residual that follows a
scene footprint is what produces striping in a composite.

**An exact fill test cannot see a reprojected scene edge.** The raw band uses
DN 0 for fill, and `dn != 0` removes it. Reprojection then interpolates between
a valid DN and the fill beside it, which leaves small nonzero values along the
edge. DN 5 decodes to -124.13 C. Nothing about it equals the fill value.
Rejecting a decoded Celsius value outside `[-50, 80]` is what removes it.

**The filter has to run before the percentile.** A range check on the finished
P95 comes too late. The invalid samples were in the sample the percentile was
drawn from, so they have already moved the answer, whatever the encoder does
next.

Invalid values are masked, never clipped. Clipping -124 C into range writes
-49.99 C, which reads as a real, cold, believable pixel. A gap does not.
`LST_MIN_TRUSTED_DN` is 2 for the same reason: DN 0 means fill and DN 1 is
reachable only from the encoding floor, so DN 1 marks a failed retrieval.

`lst_qa.py` now holds the rule and both P95 paths call it. `shard_lst_p95.py`
wraps it in numpy and `profile_lst_p95.py` wraps it in xarray. The predicates
are the same objects in both.

### One shard, before and after the QA change

MEASURED, 2026-09-09. `compare_qa_masks.py` ran shard r0 c0 of the quarter tile
twice in one process, over one loaded stack. Everything except the mask was
held fixed, the year window included, so nothing here is contaminated by the
2021-2025 change.

| input | value |
|---|---|
| shard | r0 c0, 512 x 512 px, bbox `(-62.5, -32.642222, -62.357778, -32.5)` |
| grid | EPSG:4326 at 1/3600 degree |
| window | 2020-01-01 to 2025-01-01, the window every other figure used |
| scenes | 120 loaded, sampled evenly from the 527 that intersect the shard, of 1,765 in the bbox |
| source | Earth Search, `landsat-8,landsat-9`, `eo:cloud_cover < 100` |

Rejections over 31,457,280 samples, counted in the order the rules run:

| rule | samples removed |
|---|---|
| raw thermal DN 0 | 9,422,215 |
| QA_PIXEL bits 1 to 5 | 8,979,475 |
| non-finite after decoding | 0 |
| below -50 C | 42 |
| above 80 C | 0 |
| **kept** | **13,055,548** |

The 42 samples below -50 C are the reprojected scene edge. They are 0.0001% of
the stack and no exact fill test reaches them. QA bits 3 and 4 alone removed
8,337,415 of the non-fill samples, so bits 1, 2, and 5 add 642,060. Counting
each bit on its own across the whole stack: cloud 7,458,684, cirrus 3,264,696,
cloud shadow 878,731, snow 527,085, dilated cloud 345,940.

| output | before | after |
|---|---|---|
| valid observations in the stack | 13,697,650 | 13,055,548 |
| valid output pixels of 262,144 | 262,144 | 262,144 |
| pixels changed | 247,922 (94.57%) | |
| pixels moving valid to nodata | 0 | |
| pixels moving nodata to valid | 0 | |
| P95 min, max | 37.18 C, 52.53 C | 37.40 C, 52.57 C |
| finite outputs outside [-50, 80] C | 0 | 0 |

Pixelwise difference, after minus before, over the 262,144 pixels valid in
both: min -2.28 C, P50 +0.14 C, P95 +0.63 C, P99 +0.92 C, max +2.15 C, mean
+0.20 C. The composite warms, which is what removing cool cloud contamination
from a ninety-fifth percentile does.

`qa-parity/qa_difference.png` holds the two rasters and their difference. This
shard sits inside a WRS footprint, so it shows field-shaped differences and no
scene boundary. A shard chosen on a footprint edge would show the boundary
case. The image is diagnostic. The table is the measurement.

The year change was measured on its own. A dry run of the same quarter tile
over 2021-01-01 to 2025-12-31T23:59:59Z returns **1,984 scenes** against the
1,765 the 2020-2024 window returned. Scenes per shard become min 199, P50 536,
P95 797, max 798, which puts the worst 512 px shard at 0.78 GiB and 24.9 GiB
across 32 slots. Do not compare that run's output with a 2020-2024 run as
though only the mask had changed.

### The land mask selected open ocean, twice

Generating the tile list from the pixel mask's own geometry exposed two defects
in that geometry. Both are in the shared method, so both reach the pixel mask
in `nlebovits/landsat-lst` as well as the tile list here.

**Natural Earth includes a placeholder at Null Island.** `ne_10m_land` holds one
record with `scalerank` 100, a square about 1 km on a side centred on longitude
0, latitude 0. Natural Earth documents `scalerank` over 0 to 9, and no land
exists there. Buffered by 25 km it becomes a disc in the Gulf of Guinea
spanning 0.2291 degrees in each direction. The disc touches four grid cells and
three of them are open ocean: `N00E000`, `N00W005`, and `N05E000`. The fourth,
`N05W005`, holds the coast of Côte d'Ivoire and stays in the list on its own
merits. `drop_placeholder_features` removes the record by `scalerank`, which is
a property of the record rather than of its position.

**Buffering across the antimeridian wraps the longitude.** The buffer runs in
EPSG:3857, where x is linear in longitude and the world ends at 20,037,508 m. A
25 km buffer around a coastline touching the antimeridian pushes vertices past
that edge, and reprojecting them to EPSG:4326 wraps 180.22 degrees back to
-179.78. The ring then holds vertices at both edges of the world and reads as a
polygon spanning every longitude.

Nineteen parts of `ne_10m_land` touch the seam. The ones inside the latitude
band turn into slivers that circle the planet. Their sources are the Aleutians
near 52 degrees north, an island near 9 degrees south, and Fiji between 16 and
19 degrees south. Between them they selected 68 open-ocean cells, banded at the
latitudes those sources occupy:

| source latitude | ocean cells |
|---|---|
| 50 to 55 north | 13 |
| 10 to 5 south | 26 |
| 20 to 15 south | 29 |

`make_valid`, which the shared method already applies, does not repair any of
it. The wrap is present before `make_valid` runs.

`_buffer_without_wrapping` shifts the seam-touching parts a half world east,
buffers them there, and cuts the result at the seam. Mercator x is linear in
longitude, so the translation is exact and the buffer distance never changes.

The effect on the tile count, from `measure_land_defects.py`, which builds the
list four ways from one Natural Earth release and one buffer:

| land rule | tiles inside +/-60 |
|---|---|
| the frozen 700-tile set, Natural Earth 110m, no buffer | 700 |
| Natural Earth 10m, 25 km buffer, both defects present | 966 |
| after removing the antimeridian slivers | 898 |
| after removing the Null Island placeholder instead | 963 |
| after removing both, which is production | **895** |

The slivers account for 68 cells and the placeholder for 3. Each cell belongs
to one defect or to the other, so 71 come out altogether.

Landsat coverage checks that in one direction only. A cell no scene ever
reaches lies outside the archive, so counting empty cells catches a land rule
that has gone wrong. The reverse inference fails, because Landsat images open
water: 49 of the 71 removed cells contain no scene, and the remaining 22
contain scenes while still being ocean. `N05E000` is the clearest case. It
covers the Bight of Benin, five degrees of water west of the Niger delta, and
1,979 scenes cross it. Coverage confirms the corrections rather than deciding
them. The decisive figure is the other one: of the 895 tiles that remain, every
one holds scenes.

The pixel mask still has both defects. A 25 km disc of ocean at Null Island
and two globe-circling slivers of ocean are marked as land, so any pixel inside
them is composited rather than masked. Fixing that belongs in the repository
that defines the mask.

**The 25 km buffer is a Mercator buffer.** EPSG:3857 inflates distance by
`1/cos(lat)`, so 25 km of Mercator is 25 km on the ground at the equator and
about 12.5 km at 60 degrees. That is the production rule and this tile list
keeps it, because a tile list built on a different buffer than the pixel mask
would select tiles the mask then blanks. `land_tiles.parquet` records
`buffer_is_mercator`, so the artifact states the choice.

### Rehearsal mode finds them on a laptop

Almost every failure here was findable on a laptop, and this session kept
finding them on billed instances instead. The list covers an argparse flag, a
projection and resolution pairing, an ignored chunk argument, wrong spatial
dimension names, a 6.3M-task graph, and threads multiplying to 128 on 16
cores.

`--rehearse N` runs the entire pipeline with N synthetic scenes and no object
store, exercising shard planning, item filtering, cluster startup, submission,
gather, assembly, part writing, and merge. Read throughput is the one thing it
cannot cover, and that is the one question that needs the cloud. It found two
bugs that would have wasted a four-instance run.

**`--shard-slice` sliced the filtered list, not the plan.** The pipeline drops
shards with no overlapping scene before slicing, so indices shift and the last
slice came up 36 shards short. Real tiles would have carried gaps wherever ocean
and edge shards have no scenes. The slice now applies to the plan.

**The writer skipped a barren shard instead of writing it.** That makes "no
Landsat coverage here" indistinguishable from "a machine stopped", which defeats
the coverage check. The writer now emits barren shards as nodata.

After both fixes, four slices of an 18,000 x 18,000 tile cover
**324,000,000 px exactly**, each pixel once, and merge to **100.00% coverage,
1,296 of 1,296 shards**. Dropping one slice reports
`75,168,000 px never written`. A merge round-trip reproduces the source array
bit for bit, an incomplete merge exits non-zero, and the merge peaks at 8.2 GB.

### Sharp edges in the harness

**`graph_stats` does not scale.** It runs `dask.optimize` over the whole graph,
costing a few seconds at 44k tasks and 139.6 s at 6.3M while reading no imagery.
`--no-graph-stats` disables it, and it should stay off past about 200k tasks.
Because the stage sits between graph build and compute, nobody can separate the
pipeline from this instrumentation as the cause of the quarter-tile stall. The
array path never completed a quarter tile. The sharded path completed one twice.

**Frisky workers survive `pkill` by script name.** They spawn through
multiprocessing, so their command line shows a bare `-c`. An RSS threshold
misses the small ones. Orphans contaminated two measurements here. Eight held
88 GB across two restarts and produced a wrong conclusion. Later, 26.6 GB of
dead cluster made a healthy run look like a failing one, with two schedulers on
different dashboard ports. Match on the venv path, stop by process id, and
confirm against `ss -tlnp` that one dashboard listens:

```bash
ps -eo pid,rss,args --sort=-rss \
  | awk '$2 > 200000 && /environments-v2/ {print $1}' \
  | xargs -r kill -9
```

**Concurrency knobs multiply.** `--workers 8 --threads-per-worker 4` gives 32
shard slots, and `--read-threads 4` multiplies that to 128 OS threads on 16
cores: 15 of 324 shards in 400 s, with the first wave of 64 crawling. Sizing
slots to cores gave 15 shards in 105 s, 3.8x faster. The script now refuses to
start past `cores * 6` threads.

**`client.gather` on every future at once aborts the process.** With 324 futures
and about 1 GB of results resident, frisky 0.7.2 raised a Rust panic across the
PyO3 boundary at 90% completion. A panic cannot unwind, so the process aborts
rather than raising, and 290 completed shards went with it. `frisky.as_completed`
with one result assembled at a time completed 324 of 324 with no panics.

**The wrong projection produced the wrong raster.** The quarter-tile runs used
`--crs epsg:3857 --resolution 30`, carried over from the department work, but
the grid is **EPSG:4326 at 1/3600 degree**. Web Mercator inflates area by
`1/cos(lat)`, so the raster came out 11160 x 9278 rather than 9000 x 9000: 104
Mpx instead of 81, on the wrong projection. Pass the grid CRS explicitly.

**The harness pipes stdout through `grep`, which block-buffers.** No stage line
printed until the run ended, so three EC2 runs went diagnosed by watching
processor percentage and RSS, and by guessing.

### Defects that would have produced a wrong number

Both sat in `measure_s3_requests.py`, and neither would have announced itself.

**The measurement did not use the pipeline's read settings.** The script set
`CPL_CURL_VERBOSE` and nothing else, while the real run also sets
`GDAL_DISABLE_READDIR_ON_OPEN`, `GDAL_HTTP_MULTIRANGE`,
`GDAL_HTTP_MERGE_CONSECUTIVE_RANGES`, `VSI_CACHE`, and `AWS_REQUEST_PAYER`. The
first suppresses a directory listing on every open and the third collapses
adjacent block reads into one request, so both move the count directly. Without
the last, every read returns 403. Both scripts now call
`shard_lst_p95.configure_read_env`.

**`redirect_stderr` captured nothing.** rasterio installs a CPL error handler,
so GDAL's curl output never reaches stderr. It arrives as `rasterio._err` log
records shaped `CURL_INFO_HEADER_OUT: GET ...`, and redirecting file descriptor
2 does not catch them either. The script attaches a log handler instead. Left
alone it would have reported 0 GETs. A zero reads like a pipeline that issues no
requests rather than like a broken counter, so the script now exits non-zero on
a zero count.

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
- Planetary Computer Shared Access Signature (SAS) tokens last about an hour and
  the catalogue signs them at search time, so the script signs immediately
  before building the graph. Earth Search needs no signing, because `s3://`
  hrefs authenticate per request from ambient AWS credentials with
  `AWS_REQUEST_PAYER=requester`.
- Span collection cost 15.8% of the run at 1.8M spans and 5.0% at 500k. Cap
  `--span-limit` around 500k. Use the `frisky observe` funnel (`overview`, then
  `transfers` or `prefixes`, then `stragglers` and `timeline`) rather than
  aggregating span buckets by hand, which mixes scheduler traffic with worker
  shuffle and produced wrong conclusions here more than once.

## What is not settled

- **No fleet has run against the precomputed inventory.** Every parity check
  passes, including a fixed shard loaded from both paths to an identical P95
  raster, but the largest run through the new path is six scenes.
- **The 895 tiles have never been priced against a real run.** The per-tile
  compute is measured and the tile count is measured. Their product is not.
- **The pixel mask still carries both land defects.** The tile list here drops
  the Null Island placeholder and the antimeridian slivers. The mask in
  `nlebovits/landsat-lst` does not, so a pixel inside either is composited
  rather than masked.
- **Landsat 7 is out of scope and untested.** The pipeline runs Landsat 8 and
  9, and the bulk file covers OLI/TIRS only. Adding Landsat 7 needs a second
  bulk file and a different thermal band.
- **Nobody has run a tile at the 1024 px shard.** The 2.8x request saving rests
  on three shards of a read measurement, and the 90.6 GiB working set it implies
  has no run behind it, against a measured 26.5 GiB peak at 512 px.
- **The request count came from the laptop, not from in region.** Every response
  was a clean 206, so latency added nothing, but no in-region repeat exists. It
  would cost about $0.05.
- **The request count covers one area at one date range.** Requests per band
  read follows from where the shard window falls on the internal block grid, so
  a different grid origin or resolution can move it.
- **Nobody has run past 64 cores per instance.** The prediction that a 384 px
  shard edge keeps the working set inside RAM has no measurement behind it.
- **Chunk 512 versus 256 remains unresolved** at department scale, inside the
  9.6% noise floor.
- **The department tuning covers one department at 711 scenes.** A different
  area or scene count moves the optimum, because the memory term scales with
  both.
- **This pipeline does not destripe.** The QA and nodata rules match
  `nlebovits/landsat-lst` at the pixel level. They are not the destriping
  algorithm. Scene-offset correction, the monthly climatology it fits, the
  ASTER GED gap handling, and the temporal sampling rule are all absent here,
  and the P95 itself is unchanged. A tighter mask removes some of what feeds
  scene-edge artifacts. It does not make this composite equal to the production
  one.
- **The QA comparison covers one 512 px shard at 120 scenes.** That shard sits
  inside a WRS footprint, so it measures the interior case and not the boundary
  case.
- **No 2021-2025 composite has been built.** Only its scene count is measured.

## Corrections to earlier versions of this document

Every entry is a claim an earlier version stated as fact. Each shares one
mistake: it presented an estimate as a measurement.

**Steady state across 520 tiles, at $469 to $539 of on-demand EC2 and $1,202
of S3.** Withdrawn. The 520 has no derivation in this repository or in
`nlebovits/landsat-lst`, and nothing reproduces it. The tile list is now
generated: 895 cells inside +/-60 degrees intersect Natural Earth 10m land
buffered by 25 km, which is the geometry the pixel mask uses. The nearest
figure with a derivation behind it was the 700-tile frozen set in
`landsat_lst.tiling`, built from Natural Earth 110m without a buffer. Re-priced
at 895 tiles, on-demand EC2 is $781 to $903 and S3 is $2,067 at a 512 px shard.
The per-tile rates are unchanged and still measured; only the multiplier moved.

**Every tile VM pays a 38.1 s catalogue search.** No longer true, and it was
never the expensive part. Removing it saves 9.5 instance-hours across 895
tiles, or $26 on-demand. It removes 895 runtime dependencies on a public
service, which is the reason to do it. The inventory is now built once from the
USGS bulk metadata Parquet and read in 123 ms per tile.

**The full tile cost $10.70.** Wrong. That assumed each instance ran an hour.
Each ran 642 s, so the fleet cost $1.94 of EC2 time.

**QA_PIXEL bits 3 and 4 are the mask.** Wrong. The mask this workflow was
ported from covers bits 1 to 5, and it also drops any decoded value outside
[-50, 80] C before the percentile runs. Bits 3 and 4 alone leave cirrus,
dilated cloud, and snow in the stack, and an exact `dn != 0` test leaves the
reprojected scene edge in it. On one 512 px shard over 120 scenes the wider
rule removes 642,102 more observations and moves 94.57% of the output pixels,
by a median of 0.14 C. The timings, scene counts, request counts, and costs
published above were all computed under the narrow mask.

**The window is 2020-01-01 to 2025-01-01.** Wrong twice over. The asset spec
asks for a five-year composite covering 2021 through 2025. That window included
all of 2020, which is outside it, and excluded all of 2025, which is inside it.
A STAC `datetime` range is closed at both ends, so the end also needed a time of
day: a bare `2025-12-31` drops every scene acquired that day. The default is now
`2021-01-01/2025-12-31T23:59:59Z`, held in `stac_window.py` and read by every
entry point. Cached STAC item lists are named after the query that produced
them, window included, so a 2020-2024 cache cannot answer a 2021-2025 request.
No measurement in this document has been rerun against the new window.

**S3 GET charges add about $0.09 per tile, which puts a global composite near
$650.** Wrong, and withdrawn, along with the "$1,515 - $3,030" global figure the
same invented multiplier produced. Use none of the three.
`measure_s3_requests.py` has since counted 4.77 requests per band read at a
512 px shard, or $2.31 per tile.

**Larger `--load-chunk` pays twice, because request count is now a cost
driver.** Withdrawn as written and reinstated on a measurement. The knob is
`--shard`: the sharded pipeline has no `--load-chunk`. Doubling the shard edge
from 512 px to 1024 px raises requests per band read from 4.77 to 6.83, and cuts
requests per scene-megapixel from 36.37 to 13.03. Across a full tile that is
2,177,623 GETs against 5,777,586, or $0.87 against $2.31. Read the ratios in
that order: requests per band read has to rise, because each read covers four
times the area, while requests per scene-megapixel is what prices a tile and it
falls 2.8x. The saving has a price. The dry run puts the worst 1024 px shard at
2.83 GiB and the 32-slot working set at 90.6 GiB against 128 GiB of RAM, where
the 512 px plan measured a 26.5 GiB peak, and no run has proven it at tile
scale.

**Rechunk memory is `chunk² x scenes x 4 x 2`, it scales linearly with scene
count, and reduce chunk 512 needs 327 GiB at 3,910 scenes.** Wrong. `--chunk`
never controlled the reduce block. Dask normalises every block to its
`array.chunk-size` target, 128 MiB by default, during the rechunk that
`quantile` forces:

| scenes | chunk requested | block after `.chunk()` | block after `quantile` | size |
|---|---|---|---|---|
| 711 | 256 | 256 | **217** | 134 MB |
| 711 | 512 | 512 | **217** | 134 MB |
| 1,765 | 256 or 512 | n/a | **137** | 133 MB |
| 3,910 | 256 or 512 | n/a | **92** | 132 MB |

Chunk 256 and chunk 512 produce an **identical** 217-pixel reduce block at 711
scenes, and the block auto-shrinks as scene count rises, holding per-block
memory near constant. Measured peak RSS moved little across chunk 128, 256, and
512: 35.5, 30.7, and 28.4 GiB. Reduce chunk 512 remains feasible at tile scale.
`--chunk` does change the *intermediate* rechunk, and so the task count, which
is what the measured wall-time differences reflect. Those runs produced real
numbers. The explanation attached to them did not survive.

**Read columns pin memory in proportion to scene count. That is what drove eight
workers to 11 GB each before any read.** A guess, made without instrumentation.
The observation itself stands. At 1,765 scenes the eight workers together held
96 GiB before any imagery arrived. The cause was 212 seconds of single-threaded client
work in graph build and `dask.optimize`, over 6.3 million tasks.

## Files

| path | contents |
|---|---|
| `shard_lst_p95.py` | the sharded pipeline, the slicer, and the merge |
| `profile_lst_p95.py` | the array-graph profiling harness |
| `lst_qa.py` | the QA, fill, range, and nodata rules both P95 paths call |
| `stac_window.py` | the composite window, and the cache identity it fixes |
| `land_tiles.py` | the buffered land geometry and the generated tile list |
| `usgs_inventory.py` | the precompute stage: USGS bulk metadata to one artifact |
| `tile_inventory.py` | the runtime read of one tile, from one row group |
| `fleet_plan.py` | the driver, and the checks that run before the fleet does |
| `stac_reference.py` | Earth Search, kept only as a parity oracle |
| `artifacts/` | `land_tiles.parquet`, the inventory, and their manifests |
| `compare_qa_masks.py` | one shard, run under both masks, in one process |
| `qa-parity/` | that comparison, with both rasters and the difference image |
| `sweep_throughput.py` | configuration sweep driver |
| `cost_report.py` | the labelled, deterministic cost report |
| `measure_s3_requests.py` | counts the S3 GET requests one shard issues |
| `s3-requests/` | the request measurement: both shard sizes, and the priced tile |
| `dryrun/` | local graph-build runs, no cluster and no reads |
| `ec2-results/` | eight department-scale runs: stages, memory series, frisky reports |
| `fulltile/` | the full-tile run (part files and merged raster gitignored) |
| `smoke/`, `smoke2/`, `split/`, `sweep/` | local runs with full frisky spans and traces |
| `full/`, `full.log` | the 673-scene local run, ended early by a network change |

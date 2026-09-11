# Profiling a p95 composite of land surface temperature

Where the time, the memory, and the money go in a p95 composite of Land Surface
Temperature (LST) over five years of Landsat Collection 2 Level 2 scenes.
Session `b5dd543f-2974-40c1-9f4c-5892e87c8d48`, 2026-09-08. A label marks each
figure MEASURED or DERIVED. `Corrections` records what earlier versions of this
document got wrong.

## How to read this document

This is the measurement record, not the design record. Every figure was true of
the run that produced it, and the configuration around it has moved since. The
merged pull requests are the current record: #6 precomputed the inventory, #7
made staging the default, and #8 replaced the emissivity rule. Where a PR and
this document disagree, read the PR.

- **No section here is a recommended configuration.** The `Headline` run is four
  unstaged `c6i.16xlarge` at a 512 px shard under the old QA mask. Steady state
  is one staged `m6id.16xlarge` a tile at `--shard 360`. See `What to run`.
- **No figure here is a per-tile price.** The $4.28 below bought one tile on
  four machines that read every shard from S3. A staged tile is $1.68 to $1.87
  and 26 to 29 minutes on one instance. See `Cost`.

## Headline

Four `c6i.16xlarge` instances built the full tile `S30W065`, 18,000 x 18,000 px
over 3,910 scenes, in **4.8 minutes of wall clock** for **$4.28**.

Read every number in this section as one tile on four machines, reading every
shard from S3, at a 512 px shard, on 2026-09-08. The 4.8 minutes is the slowest
of four parallel slices, not the time one machine takes. The $4.28 is the whole
tile, not a per-machine or per-hour rate. Staging, the 360 px shard, and the
current mask all postdate it, and `What to run` has the configuration that
replaced it.

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

S3 requester-pays requests are 54% of that $4.28 and EC2 is 45%. That run read
every shard straight from S3, which opens each scene about 155 times at 4.77
requests an open. Staging fetches each object once instead, which takes the
per-tile S3 line from **$2.31 to $0.0031** and the 769-tile total from
**$2,389 - $2,494 to $1,290 - $1,436**. It costs a staging phase, an
`m6id.16xlarge` in place of a `c6i.16xlarge`, and a 360 px shard. One staged run
has completed on an instance, over 64 shards of one tile. No fleet has run
staged. See `Cost` and `What is not settled`.

## What to run

Build the artifacts once, on a laptop, before any instance starts:

```bash
# the authoritative tile list, and the geometry the pixel mask rasterises
uv run land_tiles.py --out artifacts/land_tiles.parquet \
    --write-geometry artifacts/land_buffered.gpkg

# the scene inventory, from the USGS bulk metadata Parquet
uv run usgs_inventory.py \
    --land-tiles artifacts/land_tiles.parquet \
    --out artifacts/tile_scene_inventory.parquet

# ASTER GED observation counts, for the emissivity half of the mask
uv run python -c \
    "import earthaccess; earthaccess.login(persist=True)"
uv run aster_ged.py --out artifacts/aster_numobs.tif

# what the fleet will launch, and the checks that gate it
uv run fleet_plan.py --out artifacts/fleet_plan.json
```

`land_tiles.py --write-geometry` copies the buffered polygons out of the cache
byte for byte, so their digest is the `land_geometry_sha256` the tile list
already records. The tile list, the inventory, and the ASTER GED mosaic each
record that same digest, and every gate compares them.

At 16 MB the geometry is gitignored, like the inventory before it, and
`tests/make_land_slice.py` cuts the committed fixture: the same polygons
clipped to the four tiles `tests/make_slice.py` covers, 148 KB.
`masks.land_mask` rasterises only inside one tile's bbox, so the clip gives the
same mask there, and `test_masks.py` asserts that pixel for pixel rather than
assuming it. The checksum tie and the sweep over all 895 tiles both read the
full file, and both skip without it.

`aster_ged.py` is the only step that needs a NASA Earthdata Login, and it needs
it once. The fleet reads the GeoTIFF it writes and opens no connection.

`usgs_inventory.py` reuses an unchanged USGS download. Pass `--refresh` to
discard the cache and fetch the file the service is serving now.

The figures this document quotes about the land rule and the acquisition time
come from measurement scripts, which no build runs:

```bash
# what each defect in the shared land method selects
uv run measure_land_defects.py --out artifacts/land_defects.json

# the ASTER GED mask against a composite built before it existed
uv run measure_ged_registration.py \
    --raster fulltile/tile/lst_p95_dn.npy --tile S30W065 \
    --out artifacts/ged_registration.json

# how far the computed scene centre sits from the published one
uv run measure_scene_centre.py --all-years \
    --out artifacts/scene_centre_offset.json
```

Then stage both Parquet files where the fleet can read them. Steady state runs
one `m6id.16xlarge` per tile:

```bash
uv run shard_lst_p95.py --tile S30W065 \
    --inventory-uri artifacts/tile_scene_inventory.parquet \
    --pixels-per-degree 3600 \
    --shard 360 \
    --stage-dir /mnt/nvme/stage \
    --out-dir ./tile
```

`--shard` and `--stage-dir` are requirements here, not preferences. `--shard`
defaults to 512, and at 512 px the memory guard refuses a tile's deepest slice
on anything under 256 GiB. `--stage-dir` defaults to the system temp directory, which on an
instance is the root EBS volume, and gp3 tops out at 1,000 MB/s against a
measured staging rate of 922 MB/s.

Splitting a tile across machines buys wall clock and pays for it in staging,
because each slice stages nearly the whole tile. A quarter of the tile's area
touches about 2,004 of its 4,776 scenes, so four machines stage 157 GB each,
628 GB against the 375 GB one instance writes. Split when wall clock is worth
that:

```bash
# one machine per slice, four slices of the 2,500-shard plan at 360 px
uv run shard_lst_p95.py --tile S30W065 ... --shard 360 \
    --shard-slice 0:625 --out-dir ./part0
uv run shard_lst_p95.py ... --shard-slice 625:1250  --out-dir ./part1
uv run shard_lst_p95.py ... --shard-slice 1250:1875 --out-dir ./part2
uv run shard_lst_p95.py ... --shard-slice 1875:2500 --out-dir ./part3

# then anywhere
uv run shard_lst_p95.py --merge part0 part1 part2 part3 --out-dir ./tile
```

Staging is on by default. `--no-stage` reads every shard from S3 instead, which
is the path `measure_s3_requests.py` prices and the one that costs 739 requests
per object, or $2.31 a tile against $0.0031. It exists to measure against.
Do not pair it with `--shard 360`: the small shard is chosen for memory once
staging has removed the request cost, and unstaged it multiplies that cost
instead. See `Shard size is a memory decision once staging is on`.

The merge writes `lst_p95_dn.npy` and `qa_count.npy` for the analysis scripts,
and `tile/catalog/` for everyone else: two COGs on a STAC item, inside a
Portolan collection with its thumbnail, its item mirror, and its two Markdown
documents. Pass `--no-catalog` to skip the rasters and keep only the arrays.
Check the result with the Portolan validator:

```bash
uv tool install rashid
rashid check ./tile/catalog --all
```

Every tile becomes one item of one catalog. Point each tile's merge at the
same `--catalog-dir`, and each run adds its item, then rebuilds the
collection, the thumbnail, and the item mirror from every item on disk:

```bash
uv run shard_lst_p95.py --merge part0 --out-dir ./tile-a \
  --catalog-dir ./catalog
uv run shard_lst_p95.py --merge part1 --out-dir ./tile-b \
  --catalog-dir ./catalog
```

A tile is named for its north and west edges, fraction included, so the
5-degree grid reads `S30W065` and a half-degree tile reads `S32.5W062.5`. Two
tiles can then never take one directory.

Whatever the catalog needs from `part-meta.json` is checked before the merge
starts. A run that cannot produce a catalog says so in a second instead of
after the arrays are assembled, and `merge.json` reaches disk before the
catalog writer runs, so an hour of merging is recorded either way.

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

A `lst_p95` nodata now carries three meanings, and the product does not
distinguish them:

| meaning | rule | can a wider window fix it |
|---|---|---|
| no usable observation | every scene was cloudy, or the pixel is off every footprint | yes |
| water | outside the buffered land geometry | no |
| failed emissivity retrieval | 70 C or hotter in an ASTER GED gap cell, or one cell from it | no |

The last two are the output mask, and `masks.py` applies both. Both remove
values the composite held, and they treat `qa_count` differently.

The water rule zeroes `qa_count` with the temperature, so the two bands cannot
disagree. A count above zero beside a nodata pixel would say the pixel had
observations and lost them to the reduction, and over sea it was never this
product's subject at all.

The emissivity rule leaves `qa_count` alone. Zero observations is data, the
count is the evidence behind every surviving p95, and it remains correct
whatever the retrieval did with the observations it counted.

A consumer that needs the three apart has the tile's `summary.json`, which
counts each of them, and the mask's own inputs, which are named in it.

The published item states the rules instead of the counts. `processing:lineage`
names both output rules, the one-cell buffer, and the 70 C threshold, and
`sci:publications` cites the ASTER GED DOI. The inputs are identified by
checksum rather than by path: an absolute path on the machine that masked a part
tells a reader of the catalog nothing, and it would carry the operator's home
directory into a public file. `summary.json` keeps the paths, because an operator
rerunning one slice does want them. Because a raster cannot contain its own
digest, that field is empty whenever the sidecar holding it is absent, and an
empty string still reads as a checksum a consumer could compare against, so the
writer omits the digest instead of publishing a blank one.

The generated `AGENTS.md` used to say a nodata pixel meant no observation
survived cloud, shadow, snow, cirrus, and range masking. Over the ocean that was
false, and inside an ASTER gap it was false the other way: the pixel had clear
observations and the retrieval failed. Both documents now state all three
meanings, and the README quotes what the emissivity rule removed on S30W065.

The merge writes this encoding into two Cloud Optimized GeoTIFFs, so a reader
gets the rule from the file rather than from this table. `lst_p95.tif` records
the scale and the offset in its band metadata, which QGIS, `gdalinfo`, and
rioxarray all read:

```python
import rioxarray

da = rioxarray.open_rasterio("lst_p95.tif", masked=True)

# The file states its own decoding rule.
scale, offset = da.rio.scales[0], da.rio.offsets[0]  # 0.01, -50.0
celsius = da * scale + offset
```

Both files belong to a Portolan catalog written beside the merged arrays. The
internal tiles are 512 by 512 pixels. Internal overviews let a client draw the
tile without reading full-resolution pixels. Each band records its minimum,
maximum, mean, standard deviation, and valid percent in the header, not in an
`.aux.xml` sidecar. A sidecar is a second file, and a range request over the
raster returns none of it. `cog_catalog.py` writes all of this, and
`tests/test_cog_catalog.py` refuses a tree that `rashid`, the Portolan
validator, reports an error on.

The writer reopens every COG it produces and checks the block size, the
overviews, the statistics, and the decoding rule against what it asked for. A
scale the driver dropped is the one failure that leaves a file which reads as
valid and decodes to nonsense, so it is checked rather than assumed.

The STAC says the same thing through the extensions that already define it. The
item declares raster v2.0.0 and render v2.0.0, and the band states the decoding
rule as `raster:scale` and `raster:offset`, beside the `unit`, the `nodata`, and
the statistics. No `lst:`-prefixed property restates any of it. Portolan reuses
an established extension wherever one applies, and every field here has a
registered home.

`statistics`, `nodata`, and the render's `rescale` are all the stored digital
numbers. That is the domain the COG header reports, and the one a reader meets
before it applies `raster:scale`. Decoding them in the STAC would invite a
client that honours the scale to apply it twice. `unit` names what a pixel means
once decoded, which is the one field describing the far side of the transform.

A collection id includes the window its pixels came from, so 2021-2025 writes
`lst-p95-2021-2025`. The id is also the directory name. A shared id would put a
second window's tiles in the first window's collection, and the later merge would
then widen the temporal extent over pixels nobody asked about. So the id comes
from `part-meta.json` rather than from a default, and `--collection-id` overrides
it.

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
| wall clock | 38.1 s x 895 machines | 32 s, on a laptop, 5.3 GB peak |
| runtime dependency | 895, one per machine | none |

Both inventories describe the same archive. Over 2021 to 2025 inside +/-60
degrees, with `eo:cloud_cover < 100` and Landsat 8 and 9, the bulk file yields
**1,460,446** scenes, of which **1,457,559** reach a land tile and go into the
artifact. `tests/test_inventory_parity.py` enumerates the two sets over three
bounded regions, one of them across the antimeridian, and every item Earth
Search returns is present in the inventory.

The reverse does not hold, and the reason is worth stating. The bulk file's
corner columns describe the **product bounding rectangle**. Earth Search
publishes the **imaged parallelogram**, which the rectangle contains and
exceeds by about 46% of its area. `derive_projection` wants the rectangle and
reconstructs `proj:shape` from it exactly. Tile assignment gets a superset:
measured over 670 non-crossing scenes in three regions, the rectangle produces
**7.7% more tile-scene pairs** than the published footprints do, and the worst
single scene gained two tiles.

Those extra scenes are read and contribute nodata over the tile, so they cost
S3 requests and change no output pixel. No column in the bulk file gives the
imaged footprint, so matching Earth Search exactly would mean modelling the
scene rotation rather than reading it. A superset errs on the safe side:
nothing is missed, because the containment runs one way.

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
so snapping to that lattice recovers the origin exactly. Across the 1,457,559
scenes the artifact holds, the largest snap moved a corner **0.68 m**, against
a 15 m half-pixel.
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

A 512 x 512 shard needs `512² x 1765 x 15` = 6.9 GB at its peak, plus the
0.25 GiB fixed term. One worker holds that, and no array crosses a worker
boundary, so nothing rechunks and nothing shuffles.

Fifteen bytes per pixel-scene, not the four an earlier version of this section
claimed. `process_shard` holds five arrays at once and decoding allocates
transients on top of them. See `The memory model was four times low`, which is
what a fleet instance found out the expensive way. `shard_lst_p95.py` submits
one shard as one task that loads, masks, reduces, encodes, and returns.

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
workers is the practical floor at 512 px shards and 1,765 scenes.

Read that 94% as the measurement it is. `95.93 GiB / 64` is **1.50 GiB per
worker**, and the budget function of the day called the same shard 0.39 GiB.
frisky had the right answer, and nothing in this document reconciled it against
the budget function. One diagnostic invites a misreading. A clean 64-worker run ends with 64
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

**More cores is the only lever left inside compute.** Of the 700 s around it,
the submission half was recoverable and the staging half was not. A full tile comes to 1,296
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

### Submission carried the scene list, once per shard

Compute is saturated. Staging and submission, on either side of it, were not.
The issue proposed fixing both. Submission held. Staging did not.

**The submission cost is the payload, not the pickling.** The issue attributed
about 399 s to "roughly 660,000 dict serializations". MEASURED on the committed
inventory slice, that is not where the time goes:

| | |
|---|---|
| `pickle.dumps` of 509 item dicts | 0.904 ms |
| the same, across 1,296 shards | **1.2 s of 373.7 s** |
| `cloudpickle.dumps(process_shard)` as `__main__` | 3,313 B, 0.023 ms |

Serialisation is 0.3% of the phase. The payload itself is the cost. 509 item
dicts pickle to 423,128 B, so a full tile pushed **0.55 GB** at the scheduler to
describe 3,910 items that pickle to 3.2 MB between them. The amplification is
171.

MEASURED by `measure_submit_cost.py` against a real `frisky.LocalCluster`, four
workers, a 3,910-item table, 40 submits per shape, three runs on an idle laptop:

| submit payload | per submit |
|---|---|
| 509 item dicts, as the pipeline did | **5.3 to 6.1 ms** |
| a scattered table and a list of positions | 0.025 to 0.026 ms |
| **a table path and a list of positions** | **0.024 to 0.031 ms** |

MEASURED on an `m6id.16xlarge` over 250 real shards: **0.14 ms each**, against
the 0.31 s per submit the committed full-tile slices imply. The phase is gone.

`Client.scatter(broadcast=True)` raises `NotImplementedError` in frisky 0.7.2.
Plain `scatter` places one replica and scattered data has no recompute path, so
a worker killed before it replicates loses the run, which is the window
`tests/test_run_survives_worker_death.py` fires in. The table goes to a file in
the stage directory instead and every worker parses it once, 7.89 MB and about
45 ms. A restarted worker re-reads it from local disk.

### Staging beside compute is slower than staging before it

The second half of the issue proposed overlapping the fetch with the shards it
feeds, on the premise that staging is network-bound and compute is
processor-bound. It was built, run, and **falsified**.

MEASURED on an `m6id.16xlarge`, us-west-2, the identical workload in both rows:

| 1,998 objects, 78.9 GiB | wall |
|---|---|
| staging with the machine to itself | **91.9 s** = 922 MB/s |
| staging beside 64 busy workers | **358.7 s** = 236 MB/s |

**3.9x.** Not a scale artifact and not the 922 MB/s figure failing: same
objects, same bytes, same instance type. Staging spends its time on TLS, HTTP
and the copy loop, and all three want a core. At 25 Gbps the instance was using
30% of its network at 922 MB/s, so the fetch was never network-bound in the
first place, and the premise the overlap rested on was wrong.

The arithmetic on a 250-shard slice, with 256 fetch threads and
`--read-threads 1`, which is the overlap at its best:

```
overlapped   449 s   (stage 358.7 s, compute 264.7 s, overlap 219.9 s)
serial       357 s   (stage  91.9 s + compute 264.7 s)
```

The overlap pays 267 s of extra staging to save 220 s of wall clock. It loses
by about 90 s, and it loses by more as a tile gets deeper, because the fetch
slows in proportion to how busy the workers are. The implementation is removed
rather than kept behind a flag: a second ordering that is slower in every
measured case is a maintenance cost with no case to answer for it.

`worker.paused` does not appear at all in that run, against 4,463 s in the
64-core quarter tile, once workers are sized to one thread each and given a real
memory limit. `worker.exec.deserialize` falls to 30.7 ms across 128 tasks from
48.5 s, which is the item table: workers no longer unpack 400 item dicts per
task.

### A first wave is not a rate

The 64-shard wave that opens a run reports about **59 s per shard**. The next
39 shards report **6.6 s**, and a staged shard settles near **11 s**. The
difference is page-cache warmup on first touch. The effect was recorded once in
a pull request body and then walked into twice from this document, so it is
named here.

The progress line was part of the trap. It printed a running mean, which hides
the decay: it reads as a slow cluster for most of a run and as a rate for none
of it. It now prints the rate over the last 25 shards beside the mean, so the
number that misleads has been replaced rather than annotated.

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

### The staged path, measured on an instance

MEASURED on one `m6id.16xlarge` in us-west-2, 64 shards of `S30W065` at a
360 px shard and 64 workers:

| | |
|---|---|
| objects staged | **1,998** for 999 scenes, two bands each |
| GETs | **1,998**, zero retries |
| staged volume | **78.9 GiB in 91.9 s = 922 MB/s** |
| per scene | 84.8 MB, against the 78.5 MB mean the staged volumes below use |
| compute | 65.9 s of wall clock for 64 shards |
| composite | min 14.3 C, mean 47.7 C, max 65.2 C |
| machine memory | 247 GiB |

One GET per object, on the wire, at fleet width.

Staging also moves compute. MEASURED in PR #7, commit `c771f27`, over four
shards run twice in one process: **11 s staged against 21 s unstaged**, so
reading local files runs about twice as fast as `/vsis3` for the same pixels.
That is four shards, not a tile, and no staged tile has been timed end to end.

DERIVED from `shard_bytes`, and printed by the same run: a worst shard of
**0.88 GiB**, or 56.6 GiB across 64 slots. That is the model's own output. An
earlier version of this section put it in the table above and then cited it as
evidence for the model at 64 workers, which is circular. The coefficient has
risen since, so the same slice now budgets 0.98 GiB a shard and 59.7 GiB summed.
That run measured only the client process, because `shard_lst_p95.py` sampled
`psutil.Process()` and none of its children.

It samples the children now. `memory_sampler.py` polls the client and every
worker, and each run writes `memory.csv` beside its summary along with a
`workers_rss_peak_gib`. One instance run has since reported it, 35.99 GiB across
64 workers, and frisky's own spans agree at 35.87. The other independent check is
frisky's `memory 95.93 GiB / 102.40 GiB (94%)` across 64 workers on the
quarter-tile run, which is 1.50 GiB each against 1.73 from the model at 512 px
and 404 scenes.

What the superseded model would have predicted for the same configuration is
14 GiB, which is the configuration that killed an earlier instance.

The staging rate is the figure this run existed to produce, and it came in
below the 1.0 to 3.0 GB/s the cost section had assumed. Reading whole objects
over a 1 ms round trip is bounded by something other than the 200 ms round trip
that bounds the laptop, and 922 MB/s is what that something costs.

### Staging: fetch each object once

The request count is not a property of the reader. It is a property of the shard
grid, and it decomposes into two factors that multiply.

A 512 px shard at 3600 px per degree covers about 209 km². A Landsat scene
covers 185 x 180 km, or 33,300 km². So about 159 shards touch each scene, and
the plan agrees: 605,617 shard-scene reads over 3,910 scenes is **154.9 opens
per object**. Each open costs the 4.77 GETs above, because a fresh worker
process shares no cache with the one next to it.

```
154.9 opens x 4.77 GETs = 739 requests per object
```

The bytes were never the problem either. MEASURED on one shard of three real
scenes: the windowed reads pull 0.342 MB per object, so 155 shards pull 53.0 MB,
against 33.6 MB for the whole object. **Staging moves 0.63x the bytes**, because
the overlapping windows fetch the same blocks again for every shard that touches
them. In-region S3 to EC2 transfer is $0.00 in either case. S3 bills the shape
of an access pattern, not its volume, and this shape is the worst available:
many small random reads of files read 155 times each.

`staging.py` fetches each object once, with one `get_object` per object, and
writes it to local disk. The item hrefs then point at that copy. The 155 reads
still happen. They stop being billable.

| per full tile | reading from S3 | staged |
|---|---|---|
| objects fetched | 605,617 x 2 reads | 3,910 x 2 = **7,820** |
| GETs | 5,777,586 | **7,820** |
| bytes moved | 1.00x | **0.63x** |
| S3 cost | $2.31 | **$0.0031** |

### The staged path, checked against the bucket

Before any fleet ran, one shard of `S30W065` ran twice over the same three real
scenes: once from `s3://usgs-landsat`, once from staged local files, and then a
third time through a `frisky` cluster to exercise the submit path. MEASURED:

| | reading from S3 | staged |
|---|---|---|
| GETs | **30** for 6 band reads | **6** for 6 objects |
| per band read | **5.00** | 1.00 |
| responses | 30 x `206`, 30 `x-amz-request-charged: requester` | |
| 4xx, 5xx, retries | 0, 0, 0 | 0, 0, 0 |
| bytes | 2.05 MB windowed | 201.41 MB whole |
| `lst_p95`, `qa_count` | **byte-identical between the two** | |

5.00 requests per band read sits at the top of the 4.60 to 5.00 range
`measure_s3_requests.py` found, so the 4.77 mean holds. One shard at 5.00 is
775x rather than 739x; the ratio in this document keeps the measured mean.

The check also found a defect the unit tests could not. The disk guard reserved
52 MB for a thermal band, and `ST_B10` runs to 93.6 MB, so it under-reserved by
43% on the largest scenes. The constants are now measured, at the top of the
range rather than the mean.

The `NotGeoreferencedWarning` that `odc-stac` raises during the load appears on
both paths, which places it in the reader rather than in staging.

Two `botocore` defaults also had to go, and a six-object run cannot show either.
`retries` defaults to `legacy`, which retries a 500 or a 503 up to five times
inside `get_object`, so a throttled run would report fewer GETs than it paid
for and the counted S3 line would understate the bill. `max_pool_connections`
defaults to 10 against a fetch pool of up to 64, so 54 threads would queue on a
connection rather than pull an object. Retrying now belongs to `staging.py`,
where `MAX_ATTEMPTS` bounds it and `staging.json` records it.

Each of the three details below has a test, because a broken one produces a
correct composite and a larger bill.

- **`get_object`, not `download_file`.** The transfer manager splits anything
  over 8 MB into several ranged GETs, which would give back three quarters of
  the saving and change nothing else a test would notice.
- **No LIST and no HEAD.** Both are billed. Every key comes from the item href,
  and the disk guard runs off a per-band size estimate rather than a HEAD.
- **The manifest deduplicates before fetching.** One scene appearing in 155
  shards has to produce two objects, not 310.

Staging counts its own requests, one per object plus any retry, so a staged run
prices itself. `cost_report.py --s3-get-requests` reads that total from
`staging.json` and skips the `reads x bands x requests-per-read` derivation
entirely. No total then rests on the softest figure in this document.

Staging costs disk, and the measured object sizes make it much larger than the
first estimate. Steady state runs one whole tile per instance, which is 4,776
scenes on `S30W065` and 3,445 on a mean tile:

| per instance | scenes | staged | guard reserves |
|---|---|---|---|
| a quarter-tile slice, 324 shards | 2,004 | 157 GB | 242 GB |
| a mean tile, 1,296 shards | 3,445 | 270 GB | 416 GB |
| `S30W065`, 1,296 shards | 4,776 | 375 GB | 577 GB |

That rules out gp3, which tops out at 1,000 MB/s and would put the staging
phase alone at 270 s. `--stage-dir` has to name an NVMe mount; its default is
the system temp directory, which on these instances is the root volume.

Disk is not the binding constraint, though. **Memory is**, and the shard edge
decides which boxes hold the deepest slice a tile presents:

| config | needs | of RAM | 769 tiles |
|---|---|---|---|
| `c6id.16xlarge` 128 GiB, 512 px | 140.0 GiB | 109% | — |
| `c6id.16xlarge` 128 GiB, 360 px | 71.4 GiB | 56% | $1,072 |
| `m6id.16xlarge` 256 GiB, 512 px | 140.0 GiB | 55% | $1,156 |
| **`m6id.16xlarge` 256 GiB, 360 px** | **71.4 GiB** | **28%** | **$1,262** |

DERIVED, and printed by the dry run: the 15-byte model summed over the 64 shard
depths of a deep slice of S30W065, plus 4.83 GiB of client output arrays. The
depths are measured, from the inventory. The slices are `shards[987:1051]` at
360 px, 203 to 820 scenes deep, and `shards[690:754]` at 512 px, 198 to 802. The
failed run was a `c6id.16xlarge` at 512 px.

**Use `--shard 360`.** A 512 px shard needs 140.0 GiB, so a 128 GiB box refuses
it and only the `m6id.16xlarge` runs it. At 360 px both boxes hold the slice, so
memory no longer rules out the `c6id.16xlarge` and the $190 it saves. That row
became affordable when the guard began summing a slice's shard depths instead
of multiplying its worst shard by the slot count. No run has tested a 128 GiB box
since, and the `m6id.16xlarge` is the box the one staged run measured, so it
stays the recorded configuration.

### The memory model was four times low

The budget function counted one array:

```python
return shard_px * shard_px * n_scenes * 4 / GIB   # the float32 stack
```

`process_shard` holds five at once, and the fifth belongs to numpy rather than
to the pipeline:

| array | bytes per pixel-scene |
|---|---|
| `dn` uint16 | 2 |
| `qa` uint16 | 2 |
| `celsius` float32 | 4 |
| `valid` bool | 1 |
| the copy `nanpercentile` partitions | 4 |
| **named total** | **13** |
| the windowed read of a tiled COG | **~2, measured** |
| **model** | **15** |

Thirteen is the arithmetic and it under-predicts. MEASURED by
`measure_shard_memory.py --mode memory --stage-dir` against 1,615 real staged
scenes on an `m6id.16xlarge`, at the depths a fleet shard runs:

| scenes | 200 | 400 | 600 | 820 |
|---|---|---|---|---|
| 360 px, GiB | 0.53 | 0.87 | 1.20 | 1.62 |
| 512 px, GiB | 0.86 | 1.57 | 2.28 | **3.05** |

Least squares puts the slope at **14.47** bytes per pixel-scene at 360 px and
**14.44** at 512, a ratio of 1.002. A 13-byte model reads low at 600 and 820
scenes on both edges, worst by 7% at 512 px and 820 scenes, where it predicts
2.85 GiB against 3.05 measured.

The sweep ran on two instances. One fit 13.68 and 14.52, the other 14.47 and
14.44, so one edge's slope moves about 6% between runs while the pair stays
near 14.5. On the second run the two edges agree to 0.2%, which measures the
px-squared scaling on the source a fleet reads. The synthetic fixture could
only assert it.

The surplus is the read. GDAL decodes whole blocks out of a tiled source and
`odc.stac` assembles them into the target array, and the five named arrays do
not cover that intermediate. Which allocation holds it is unverified, because
this document profiles none. So the model is the five named arrays plus
measured read overhead, and 15 bounds all 36 points of the eight committed
sweeps.

Erring high is the safe direction: over-reserving costs worker slots an
operator can add back, and under-reserving cost a fleet instance its workers.

**What made 13 look safe for a week.** Two things, and both are properties of
how it was measured rather than of the pipeline. The synthetic fixture writes
one untiled raster at the shard's own edge and reads it whole, so it never
allocates the intermediate, and it fits 12.68 to 12.97. And a staged sweep that
stops shallow agrees with 13 as well, because the 0.25 GiB fixed term still
covers the gap below about 280 scenes. The first staged sweep reached 100
scenes. Every fleet shard runs 195 to 820.

Both edges' slopes are least-squares fits, and the earlier figures of 12.73 and
12.89 were not. They came from averaging consecutive differences, which agreed
to 1.3% while the differences being averaged ran 10.9 to 16.4 bytes.

#### The synthetic fixture is not the read the fleet does

`--mode memory` writes one untiled raster at the shard's own edge. The fleet
reads a window out of a 7800 x 7900 tiled COG, which allocates an intermediate
the fixture never needs. `--mode timing` already refuses that fixture for
exactly this reason, and the objection applies to memory as well.

MEASURED against 100 real staged scenes of `S30W065`, at four shard edges:

| source | 256 px | 360 px | 512 px | 1024 px |
|---|---|---|---|---|
| synthetic | | 12.97 | 12.68 | |
| staged COGs | 14.48 | 13.59 | 13.25 | 10.47 |

The model at 15 bytes plus 0.25 GiB bounds all 36 points of the eight committed
sweeps, staged and synthetic, and under-predicts none of them. It is the slope
that is not tightly determined: these shallow staged fits sit 0.9 to 1.2 bytes
below the deep staged sweeps at the same two edges, and the 1024 px figure sits
below every other. Real scenes make shard edge and how much data falls in the
shard move together, which is the same confound the timing mode documents in
the other direction.

So the fixture understates the slope by 10% at 360 px and 12% at 512, measured
against the deep sweeps, and the margin absorbs it. Read these four as the
shallow end they are. They stop at 100 scenes. Every fleet shard runs 195 to
820, a range only the deep sweep above reaches.

Each point runs in a fresh interpreter, and it has to. glibc does not return
freed arenas to the kernel promptly, so a second shard measured in the same
process reports the high-water mark of the first. A contaminated sweep put the slope at 17,
another at 18, and they disagreed by 18% at 700 scenes,
which is how an 18 reached this document for an hour.

What it cost before the fix: a `c6id.16xlarge` reported a 25 GiB budget across
64 slots for a real demand of 97, and lost ten worker processes to coredumps
three minutes into the run. Nothing said `MemoryError`.

#### The model now refuses a run it cannot fit

Correcting `shard_bytes` did not stop that configuration. The corrected figure
went to a `print` and to no comparison, so the same command would have printed
a larger number and launched anyway.

`worker_memory_guard` refuses it, and it runs before the first GET the way
`staging.disk_guard` does, so a configuration that cannot fit does not buy its
objects first. The demand is the sum of the slice's shard depths, deepest first
up to the slot count, plus the client's arrays, which are `uint16` of p95,
twelve `uint8` monthly counts, and the two boolean masks: 16 bytes an output
pixel, or 4.83 GiB for an 18,000 px tile. The refusal states the demand, the
machine's total, and the shard edge that would fit. `--force` spends the margin.

It summed nothing at first. It multiplied the worst shard by the slot count,
which on the deep slice of S30W065 at 360 px over-reserves by **3.1x**. The two
budget rows are recomputed here against the 15-byte model now in the code, and
the third is the measurement they are checked against:

| | GiB |
|---|---|
| 64 x worst shard, plus client | 115.5 |
| sum of the 64 actual depths, plus client | **71.4** |
| simultaneous peak, sampled at 0.5 s | **37.0** |

The slice runs 203 to 820 scenes deep with a median of 401, so the worst shard
is not what the other 63 workers hold. Summing removes 44 of the 78 GiB of
over-reservation and needs no new measurement, because `work_idx` already
carries every depth.

The remaining 1.9x is peak non-coincidence. The per-process column of
`memory.csv` puts the sum of each worker's own high-water mark at 48.6 GiB
against 37.0 ever live at once, so the workers do not peak together. That
headroom stays.

#### Frisky agrees with the sampler

The whole reason to add `memory_sampler.py` was that nothing measured worker
RSS. frisky measured it all along. `frisky observe overview` on the spans file
the run already writes:

```
perf    wall-clock 92.3s   workers 64   tasks 64   spans 7871
memory  35.87 GiB / 832.00 GiB (4% peak)   spilled 0 B   unspilled 0 B
```

35.87 GiB against the sampler's 35.99, a 0.3% difference, from two independent
instruments. And `spilled 0 B` says the workers never came near the limit,
which no external sampler can report.

That is the cross-check the model needed, and it was one command away for the
whole branch. `observe overview`, `workers`, `stragglers` and `prefixes` all
read a spans file offline, so they work after the instance is gone.
`observe detail` needs a live dashboard URL and cannot.

#### A shard runs about a minute, not 1.4 s

`compute 87.1s for 64 shards (1.36s each)` reads as a per-shard duration and is
not one. It is wall clock over shard count, and all 64 shards run at once.
frisky's `worker.exec.call` spans give the real distribution:

| per-shard seconds | min 39.5 | p50 59.9 | max 86.9 |
|---|---|---|---|

Worth stating because a 2.2x spread across shards of 203 to 820 scenes is the
imbalance a fleet driver would want to see, and because the misreading briefly
justified a change to the sampling interval that the measurement then refuted:
a shard runs long enough that 0.5 s samples it about 120 times, and 0.05 s
found the same peak from a file 6.6x larger. The default stays at 0.5 s and the
progress line now says which figure it prints.

That guard reads the host it runs on, which is the right machine only once the
run is already there. Planning happens somewhere else, so `--dry-run` takes
`--target-memory-gib` and checks the budget against the machine the run is
headed for. It exits 2 when the configuration would be refused, which prices a
fleet before an instance exists:

```bash
uv run shard_lst_p95.py --tile S30W065 --shard 512 --workers 64 \
    --shard-slice 690:754 --target-memory-gib 128 \
    --dry-run --search-in-dry-run
```

```
  slice: min 198  p50 412  max 802   <- what this machine holds
  worst shard: 3.19 GiB, 820.7 GiB across 256 slots   OVER by 692.7 GiB
REFUSED on a 128 GiB machine:
  64 shards at 512 px, 198 to 802 scenes deep, need 135.2 GiB between them,
  plus 4.83 GiB of client output. That is 140.0 GiB and this machine has
  128.0 GiB. Use --shard 485 or smaller, drop --workers, or pass --force.
```

`--workers 64` is 64 worker processes at the default four threads each, so the
naive line counts 256 slots. The guard counts the 64 shards the slice holds.

That is the configuration that killed the `c6id.16xlarge`, refused from a
laptop before an instance starts.

#### A slice is not the tile, and the light one was measured

The dry run reports the slice's worst shard as well as the tile's. One machine
runs one slice, so that slice's worst shard sets the memory it needs. On S30W065 at
360 px the two differ by a factor of two:

| | scenes per shard |
|---|---|
| `shards[0:64]`, the slice the `m6id` ran | min 199, p50 397, **max 404** |
| `shards[987:1051]`, the deepest slice | min 203, p50 401, **max 820** |
| whole tile | min 195, p50 408, p95 802, **max 820** |

So the instance run that produced the figures above took a slice at half the
tile's worst depth. A fleet machine at 360 px faces 71.4 GiB, not the 59.7 the
lighter slice needs, and that is the number to size an instance from.
`m6id.16xlarge` holds it with 3.6x to spare. A 512 px shard at the same depth
needs 140.0 GiB, which fits that box and not a 128 GiB one.

The number was already in this document. The quarter-tile run printed
`memory 95.93 GiB / 102.40 GiB (94%)` across 64 workers, which is 1.50 GiB
each, against a budget function reporting 0.39. It was written up as a tuning
result.

### Shard size is a memory decision once staging is on

It used to be a request-cost lever worth 2.8x. Staging removed that: one GET
per object whatever the shard edge. What remains is memory, which falls with
the square of the edge, against compute, which does not.

Unstaged the lever is still there, and a smaller shard pulls it the wrong way.
MEASURED from the inventory by `--dry-run --search-in-dry-run` on `S30W065`: the
2,500-shard plan at 360 px makes **1,264,988** shard-scene reads against
**690,659** for the 1,296-shard plan at 512 px. That is 265 opens per object
against 145, and unstaged every open is billed and pays a fresh round trip. So a
360 px shard with `--no-stage` takes the memory-optimised edge and the unstaged
request bill together, which is the worst of the four pairings.

MEASURED by `measure_shard_memory.py --mode timing` over ten staged scenes:

| shard | s/Mpx-scene | vs 512 px | worst shard at 820 scenes |
|---|---|---|---|
| 256 px | 2.768 | +36% | 1.00 GiB |
| **360 px** | **2.277** | **+12%** | **1.73 GiB** |
| 448 px | 2.056 | +1% | 2.55 GiB |
| 512 px | 2.030 | — | 3.25 GiB |

The seconds are measured. The memory column is the 15-byte model at the deepest
shard the inventory gives S30W065, which is 820 scenes.

Read the +12% as a staged figure. A local open costs a file handle and a header
parse, so spreading that over fewer pixels moves the total by 12%. An unstaged
open costs a round trip and about four ranged GETs, and the 360 px plan issues
1.83x as many of them.

Read the penalty as an upper bound. These ran at ten scenes per shard, where
the fixed per-shard cost is amortised over the least work; a fleet shard
carries 195 to 820. The measurement needs real scenes, because a synthetic
raster generated at the shard's own edge makes shard size and source layout
move together and reports a 4.7x cliff at 512 px that does not exist.

360 divides 3600 exactly, so its shards align to whole degrees with no partial
edge, 50 x 50 to a tile.

### Scenes with no thermal band

An `OLI_TIRS_L2SR` product carries `QA_PIXEL` and no `ST_B10`, so
`tile_inventory.build_item` writes it with no `lwir11` asset, which is what
Earth Search returns for it. Those scenes reach `process_shard`, load as fill,
and contribute nothing: `lst_qa.valid_observation` begins at `not_fill`, so the
whole layer is invalid before the percentile or the monthly counts see it.

Dropping them is output-neutral, and `test_pipeline_paths.py` asserts the
stronger claim rather than the plausible one: `lst_p95` and `qa_count` come back
byte-identical with the fill layers present and absent. The filter is on by
default because of that test, and `--keep-scenes-without-thermal` turns it off.

MEASURED over the full artifact, `177,254` of `3,083,129` tile-scene rows carry
no thermal band, or **5.75%**. By distinct scene it is `96,220` of `1,457,559`,
or **6.60%**.

```sql
SELECT count(*) FILTER (thermal_href IS NULL), count(*)
FROM 'artifacts/tile_scene_inventory.parquet'
```

#### Product type and thermal band select the same rows

`thermal_href IS NULL` and `data_type = 'OLI_TIRS_L2SR'` match row for row over
all 3,083,129 rows, with no exception in either direction:

| data_type | thermal | rows |
|---|---|---|
| `OLI_TIRS_L2SP` | present | 2,905,875 |
| `OLI_TIRS_L2SR` | absent | 177,254 |

#### It follows the footprint, not the date

Grouped by WRS path and row, 95.4% of footprints are all one product or all the
other:

| WRS path/row | count |
|---|---|
| always L2SP | 6,999 |
| always L2SR | 769 |
| mixed | 376 |
| **total** | **8,144** |

The 376 mixed footprints hold a flat L2SR rate by year: 4.3, 5.4, 5.0, 4.2 and
5.2 percent for 2021 through 2025. Platform makes no difference either, at
5.8% for landsat-8 against 5.7% for landsat-9.

A static ancillary input explains the 95.4%. A change in the processing system
does not, because it would show in the years. Nor does a sensor difference,
because it would show in the platforms.

#### DERIVED: ASTER GED

The Collection 2 algorithm for surface temperature reads land emissivity from
ASTER GED, which covers no ocean. A footprint with too little usable emissivity
gets no ST band, and USGS writes the product as L2SR instead.

#### Measured against ASTER GED

MEASURED. `measure_ged_registration.py` cross-tabs the 126 zero-thermal tiles
against ASTER GED coverage, using the artifact `aster_ged.py` builds from
14,128 granules.

| statement | tiles |
|---|---|
| L2SR-only tiles | 126 |
| more than 1% of their land has ASTER emissivity | **0** |
| ASTER GED publishes no granule over their land | **126** |

Not one of them. LP DAAC publishes an AG1km granule only where ASTER GED has
data, and for all 126 tiles it publishes none. The product type follows ASTER
coverage, and that is no longer inference.

The mechanism remains DERIVED. USGS reads emissivity from ASTER GED and
produces no surface temperature without it, which explains the correlation, and
this document has not read the algorithm.

#### Severity across the land tiles

| L2SR share | tiles |
|---|---|
| none | 500 |
| under 5% | 203 |
| 5% to 25% | 38 |
| 25% to under 100% | 28 |
| **100%** | **126** |

Every tile at the top of that range is an island or a coast. The worst twenty:

```
N40W025 99.9%  Azores          S50E165 82.6%  NZ subantarctic
S05E165 99.5%  Vanuatu         S30E155 73.9%  Coral Sea
N20W115 99.3%  Revillagigedo   N10E070 71.4%  Maldives
N15W110 99.0%  Revillagigedo   N55W180 70.5%  Aleutians
S20E155 90.7%  Coral Sea       S10E055 66.4%  Seychelles
S10E165 87.0%  Solomons        N25W165 59.6%  NW Hawaii
N15E140 83.7%  Marianas        S10E115 50.5%  Indonesia
N10E145 83.7%  Marianas        S15E055 49.8%  Seychelles
```

#### Gaps inside a scene are invisible here

**Durban does not appear in this signal.** Tile `S30E030` holds 1,012 scenes
and zero L2SR rows.

The archive splits ASTER coverage loss into two kinds and the product type sees
only one:

| kind | how it appears | visible in the inventory |
|---|---|---|
| whole footprint | no ST band, product is L2SR | yes, exactly |
| within a scene | ST band present, gap pixels written as fill | no |

`lst_qa.not_fill` tests `thermal_dn != 0`, so a gap pixel inside an L2SP
product is already excluded before the percentile. Those pixels are not wrong.
Their time axis is thin.

The output mask is what handles the second kind, and it reads ASTER GED's own
observation count rather than inferring the gap from the composite. See "The
output mask" below.

`qa_count` cannot do that job alone, and the reason is worth stating.
`masked_celsius` returns `not_fill & qa_clear`, so a zero there merges the
ASTER gap with cloud. A pixel that was cloudy on every pass and a pixel that
never had emissivity look identical, and they call for opposite advice: widen
the window, or stop. The mask tells them apart by reading a static dataset
rather than by reading the composite.

What that is worth depends on which path runs. Unstaged, each dropped row saves
739 requests, and the global line falls by **$105**. Staged, it saves one GET
per row and the global line falls by 7 cents. The reason to keep the filter is
the memory: each such scene adds one layer to the time axis of every shard it
touches, and that axis is what pins the run at 94% of the worker memory limit.

The filter is defined in `staging.py`, not in `usgs_inventory.py`.
`test_inventory_parity.py` asserts the artifact matches Earth Search item for
item, and Earth Search returns these products. Filtering at build time would
break that parity and discard the evidence for it.

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

**126 of the 895 land tiles have no scene with a thermal band.** Only 769 do.
The rest carry `OLI_TIRS_L2SR` products alone, so a run there would boot,
stage, compute, and write an all-nodata composite. Every figure below counts
769.

`fleet_plan.py` filters them, and the filter costs nothing. The inventory holds
one row group per tile, so `thermal_href` null-count statistics answer the
question from Parquet metadata with no column data read at all. The plan names
them in `tiles_without_thermal`, beside the existing `tiles_without_scenes`,
because the two are different facts with different fixes: one moves if the
window moves, and the other does not. Skipping them saves about $189 and three
hours of fleet time. S15W180 alone holds 4,274 such scenes, about 350 GB of
staging for nothing.

Only a tile at zero comes out. A tile that is 90% L2SR still composites real
temperatures from the rest, and 28 of the 895 sit between 25% and 100%.

A run pointed at one of the 126 by hand writes a `summary.json` with status
`no-thermal-coverage` and exits 0. Nothing to composite is a correct outcome,
not a dead machine, and the advice below is to key a driver on `summary.json`
rather than on exit status. Writing no artifact and exiting non-zero would have
made that impossible for exactly these tiles.

The global figures scale on `tile_scene_rows`, the **2,905,875** thermal-carrying
tile-scene pairs, and not on 895 copies of `S30W065`. That tile holds
4,776 scenes against a mean of 3,445, so pricing the globe from it overstates
the S3 line by 13%. `Corrections` withdraws the $2,067 that did.

A mean tile holds 3,445 scenes. MEASURED on an `m6id.16xlarge` in us-west-2,
staging moves **922 MB/s**, so a mean tile writes about 278 GB in **302 s**.
That is below the 1.0 to 3.0 GB/s this document assumed before the run, and it
adds about $100 across the fleet. The staged column prices `m6id.16xlarge` at
$3.7968/hr and a 360 px shard, which is the configuration that fits the memory
a worst-case shard needs.

Only **769** of the 895 land tiles hold a scene with a thermal band. The other
126 would boot, stage, compute, and write an all-nodata composite, so both
columns run 769 and the unstaged column is restated on the same basis.

| 769 land tiles | reading from S3 | **staged** |
|---|---|---|
| instance | `c6i.16xlarge` @ $2.72/hr | `m6id.16xlarge` @ $3.7968/hr |
| shard | 512 px | 360 px |
| EC2, on-demand | $671 - $776 | $1,288 - $1,434 |
| EC2, spot | $234 - $271 (~$0.95/hr) | $375 - $418 (~$1.10/hr) |
| S3 GET requests | **$1,718** | **$2.32** |
| **total, on-demand** | **$2,389 - $2,494** | **$1,290 - $1,436** |
| **total, spot** | **$1,952 - $1,989** | **$377 - $420** |

Read the EC2 rows as DERIVED. Measurement supplies the per-tile compute and the
tile count. The tail is a bracket, and the staged column adds the 302 s a mean
tile takes to fetch at the measured 922 MB/s, and prices the disk it writes to.
The S3 rows are arithmetic over `tile_scene_rows`: 154.9 opens x 2 bands x 4.77 GETs unstaged,
against 2 GETs staged.

Per tile, the staged column is **$1.68 to $1.87** and **26 to 29 minutes** on
one `m6id.16xlarge`. Divide either total by 769; the minutes follow from the
EC2 line at $3.7968/hr. Quote these two for one tile. The headline's $4.28 and
4.8 minutes describe four unstaged machines working on one tile together.

Staging cuts the total by **1.7x to 1.9x on demand and 4.6x to 5.3x on spot**.
EC2 rises, on a larger instance, at a smaller shard, and for longer, and still
rises by far less than the requests it removes.

One term in the staged column stays a bracket. Per-tile compute is scaled from
the 512 px full-tile run by the measured 12% penalty, because one 64-shard wave
cannot be extrapolated: 25 shards took 59.3 s and the next 39 took 6.6,
which is page-cache warmup on the staged files rather than a rate.

Removing the per-tile search saves 38.1 s x 895 tiles, or 9.5 instance-hours.
That is $26 on-demand and $9 spot, against an S3 line of $1,718. The search was
never the money. It was 895 dependencies on a public service, one per machine,
each able to fail a run that had already paid for its instance.

S3 charges do not amortise, because they scale with reads rather than with
instance time. Unstaged they cost more than twice the on-demand compute and more
than six times the spot compute. Shard size moves them by a factor of 2.8, and
staging moves them by a factor of 738, so shard size is no longer the lever
worth spending memory on.

The department-scale phase, across five EC2 sessions, eight completed
department runs, one 200-scene quarter-tile smoke run, and two quarter-tile
attempts that never finished, cost about **$4.45**. The full-tile fleet cost
**$4.28** on top of it. The staged and memory-sweep instances are unpriced.

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

### The output mask

A pixel over the sea, and a pixel where ASTER GED records no emissivity, are
pixels this product has nothing to say about. Both stay that way at any window
length, because the sea is the sea and the emissivity dataset is fixed.
`masks.py` applies the two rules and `shard_lst_p95.main` runs it once, over
the assembled tile.

Water came first. `land_tiles.py` opens with the contract: one geometry answers
both "which tiles does the fleet run" and "which pixels hold a temperature".
Only the first half existed. That docstring described the pixel rule, nothing
built it, and every coastal tile published sea as temperature.

Emissivity is the second rule. Collection 2 Level-2 Surface Temperature reads
an emissivity value for each pixel from ASTER GED, which USGS built from
clear-sky ASTER scenes acquired 2000 to 2008. Where ASTER never caught clear
sky, GED records no emissivity. `aster_ged.py` mosaics GED's own
observation-count layer, where a count of zero marks that by definition rather
than by a guess at a fill value.

What USGS does there is the whole difficulty. Rather than leave the pixel
empty, it interpolates emissivity from the neighbouring cells and retrieves a
temperature anyway, and some of those retrievals fail upward. So the rule is a
pair. A pixel comes out where its cell reports zero observations, or lies one
cell from such a cell, AND the pixel reads 70 C or hotter.

#### The gap region and the damage are different sets

A gap cell records where ASTER missed the ground. It says nothing about the
retrieval that consumed the interpolated emissivity, so the geometry alone
removes far more than it should.

MEASURED on the published S30W065 composite, which predates this mask by months
and records nothing about it:

| statement | value |
|---|---|
| gap cells on the tile | 605 |
| of those, with no pixel at or above 70 C | 524, or 86.6% |
| in the 81 that do, the hot share of the cell | 4.77% |
| `NumObs == 0` land pixels holding a temperature | 89.53% |
| valid pixels the geometry alone removes | 701,839, or 0.2167% |
| bad pixels it removes | 4,588 |

153 ordinary pixels lost for each bad one. The failures are a thin scatter
inside a minority of cells, and the mask was applied at the resolution of the
cell.

I looked for a static predictor of the 81 bad cells and found none. 602 of the
605 gap cells are partly holed, so hole geometry selects everything, and the
cells with no hot pixel average more missing pixels (140) than the cells with
one (108). Temperature is the only thing that separates them.

DERIVED, for the 89.53%: USGS resamples ASTER GED from 1 km to the 30 m product
grid, so a 30 m pixel inside a zero cell can still take emissivity from its
neighbours. A zero cell becomes a hole only where the zero region is wider than
that neighbourhood. This document has not checked either step against the
algorithm.

The registration scan counts the hard holes, which are the minority of a gap
cell. So the scan measures precision and not recall. `NumObs == 0` explains
where 99.71% of the tile's missing pixels are, and predicts nothing about
whether one pixel is missing.

#### An earlier build applied this rule, and withdrew it

`nlebovits/landsat-lst`, the project this repository took its land rule from,
shipped the geometry alone in its #116. It measured 2,799,286 pixels removed on
S30W065 to catch 2,582 artifacts, and replaced it with the pair. Its
`config.py` carries the same threshold and the same one-cell buffer, with a
hard floor at 50 C so no configuration can turn the threshold into a plain
ceiling.

An earlier version of this document made the same mistake for the same reason.
It read the hot column without the column beside it. The table below prints
both, and marks the row `masks.py` applies.

#### Six rules, priced on one tile

MEASURED on S30W065 by `measure_ged_registration.py`, which writes the same
rows to `artifacts/ged_registration.json`:

| rule | valid removed | share | hot removed | hot left |
|---|---|---|---|---|
| `numobs == 0` | 701,839 | 0.2167% | 4,588, 77.30% | 1,347 |
| `numobs == 0`, 1-cell buffer | 2,799,906 | 0.8644% | 5,432, 91.52% | 503 |
| `numobs <= 2` | 22,308,570 | 6.8871% | 5,228, 88.09% | 707 |
| `numobs <= 2`, 1-cell buffer | 36,914,431 | 11.3962% | 5,834, 98.30% | 101 |
| `numobs == 0` AND `>= 70 C` | 4,588 | 0.0014% | 4,588, 77.30% | 1,347 |
| **`numobs == 0`, 1-cell buffer AND `>= 70 C`** | **5,432** | **0.0017%** | **5,432, 91.52%** | **503** |

Read the last row against the second. Identical hot tails, 2,794,474 ordinary
pixels apart. The buffer costs almost nothing once the temperature test bounds
what it can reach. It stays at one cell because it takes 844 more artifacts
and no ordinary pixels.

The `numobs <= 2` rows are the reason the thin tiers stay. One and two
observations are low confidence, not absence, and buying the last of the tail
there costs 11.4% of the raster unconditionally.

The water rule removes values the composite held, because Landsat retrieves a
real temperature over water.

The emissivity rule removes values too. 5,432 of them on S30W065, every one at
70 C or hotter inside the gap region. An earlier version of this document said
the rule removed only pixels that were already nodata, on the reading that USGS
wrote `ST_B10` fill across a gap. It writes fill across the hard holes alone,
which are 10.5% of a gap cell's pixels.

`qa_count` follows the water rule alone. Zero observations is data, and the
count layer stays the evidence behind every surviving p95. Over water the pixel
was never this product's subject, so both bands go.

#### The mask goes on once, in the client

The run builds it before staging and applies it after the gather. It depends on
the tile's bbox and two artifacts, and on nothing the run computes, so a tile it
empties costs no staged object and no cluster. It goes on before the summary
statistics, the part file, and any merge, so a `--shard-slice` machine masks its
own slice and `merge_parts` needs no rule of its own. Masking at merge instead
would leave a single-machine run unmasked and would let two machines' parts
disagree.

Not per shard. Rasterising the geometry inside `process_shard` would repeat the
same work in each of 1,296 shards, inside the processes with the least memory
to spare.

Both masks are resident for the whole run, so the client budget accounts for
them. `CLIENT_BYTES_PER_OUTPUT_PIXEL` moves from 14 to 16: `lst_out` at uint16,
`qa_out` at 12 uint8, one bool of water rule, and one bool of gap region. On
the largest tile that is 0.6 GiB the model used to omit, and a fleet instance
is sized from that model.

The memory sampler stops after the mask, not before it. A sampler stopped
earlier never observed the arrays the mask allocates while the two full-tile
outputs are live, which is the peak `--target-memory-gib` is checked against.
`apply_output_mask` also reduces `qa_out` with `np.any` rather than a sum in
int64. The sum built a 2.6 GiB array on the largest tile, beside a 4.8 GiB
budget, for a question with a boolean answer.

#### A coarse land test is not a safe proxy for a fine one

MEASURED, over all 895 land tiles against the committed geometry. At the GED
grid of 0.01 degree, one tile comes back with no land cell:

| tile | land at 0.01 deg | land at 1/3600 deg |
|---|---|---|
| `N00E050` | 0 cells | 1,852 px |

A cell counts as land when land covers its centre, and `N00E050` contains land
thinner than a cell. An earlier `fleet_plan` screened tiles at the GED grid and
had to re-check anything it emptied at the run's own resolution, or it would
have lost a real tile silently on the first fleet it planned.

That screen is gone with the drop list it fed. The measurement stays here
because the trap does not: any future test that asks a coarse grid a question
about a fine one meets `N00E050` again.

#### The download the build makes

MEASURED against `ne_10m_land` buffered by 25 km: the geometry meets **14,941**
of the 43,200 one-degree cells inside +/-60 degrees. AG1km v003 holds 24,873
granules globally, so restricting the fetch to land avoids most of it.

AG1km, not AG100. The ASTER GED User Guide V3 gives AG100 as 1000 by 1000 cells
per degree and AG1km as 100 by 100, so AG1km's cell measures 0.01 degree exactly
and its granule is already the grid this mask reads. AG100 is about a hundred
times the download and would then need decimating to the same answer.

Reading a granule converts it twice, and the manifest records both conversions,
because a reader of the raster alone cannot recover either. The source count is
int16 and the artifact is uint8, so the read clips at 255. The rule tests
`== 0`, and clipping a large count cannot move a pixel. The source fill of -9999
becomes 0, so a cell with no observation reads as gap.

#### An absent granule is not a gap

MEASURED against CMR: ASTER GED AG1km v003 publishes no granule for **813** of
the 14,941 land cells, and a search returns nothing for them rather than an
empty granule. LP DAAC publishes only where the dataset has data.

That is tempting to read as a gap, and it is wrong. Landsat holds surface
temperature over most of that land:

| tile | scenes | with a thermal band | L2SR share |
|---|---|---|---|
| `N05W095` | 617 | 615 | 0.3% |
| `N00E050` | 1,034 | 1,031 | 0.3% |
| `N55W175` | 2,129 | 1,793 | 15.8% |

The first build treated an absent granule and a zero count as the same value,
and `fleet_plan` dropped **34 tiles** on it, among them the Solomons at 41
million land pixels, the Aleutians at 31 million, and the Maldives at 30
million. Every one of those tiles has thermal scenes and a real composite.

So the artifact gained a second band. Band 1 is the count, band 2 is 1 exactly
where the build read a granule, and `masks.emissivity_gap` needs both: a pixel
is a gap when its cell was read AND its count is zero. A cell with no granule
keeps its pixels, and `output_mask` counts them as `pixels_land_unread` so a
tile resting on absent granules says so.

With that rule the emissivity screen drops no tile at all, and the fleet plan
returns to 895 land tiles minus the 126 L2SR-only ones: **769 to launch**.

#### Registration

The granule filename specifies its NORTHWEST corner, per the user guide:
`AG1km.v003.33.-115.0010.h5` covers latitude [32, 33] and longitude
[-115, -114]. Reading it as the southwest corner moves the mask 100 cells.

MEASURED in `tests/test_aster_ged.py`: one 0.01 degree cell written alone comes
back at rows 0 to 35 and columns 0 to 35 of the tile, 1,296 pixels, at 3,600
pixels per degree. Exact block replication, no half-pixel shift. A
centre-registered read would put it at rows 18 to 53, which no tile-wide
statistic would show.

MEASURED against the real artifact and the published S30W065 composite. Shifting
the cell assignment and re-counting how many of the tile's missing pixels land
on a zero cell:

| shift, cells | agreement |
|---|---|
| 0, 0 | **99.71%** |
| 1, 0 | 58.56% |
| -1, 0 | 58.23% |
| 0, 1 | 54.93% |
| 0, -1 | 56.68% |
| 2, 2 | 16.40% |

The peak at zero settles the placement. An external analysis of the same tile,
built on AG100 rather than AG1km and by different code, reported 99.65% at zero
against 57.7% and 58.8% one cell either way. Different products, different
code, agreement to within 0.1 points.

The granule's `Geolocation/Latitude` array is a linspace of 100 samples from
33.0 to 32.0 inclusive, so its spacing is 1/99 degree rather than 0.01. Read as
cell centres it would put the grid half a cell outside latitude 32 to 33. The
scan resolves that in favour of 100 equal 0.01 degree cells, because a
half-cell error is 18 output pixels and a whole-cell scan cannot see it.

#### Two reasons a tile is not launched

`fleet_plan` names two, and they are different facts with different fixes.

| list | what it means | does a wider window help |
|---|---|---|
| `tiles_without_scenes` | no row group in this window | yes |
| `tiles_without_thermal` | every scene is `OLI_TIRS_L2SR` | no |

There was a third, `tiles_without_emissivity`, for a tile whose every land
pixel sat inside a gap. It is gone. The pixel rule removes a gap pixel for
reading 70 C or hotter, not for being a gap pixel, so such a tile still
publishes every ordinary temperature in it. The list had dropped no tile on
the real plan.

`fleet_plan` still reads the mosaic. It checks that the tile list, the
inventory, and the two mask artifacts refer to one land geometry, and it
records the pixel rule in `emissivity_rule` so a finished tile is checkable
against the plan that launched it.

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

### A worker that aborts, and a shard that does not arrive

frisky aborts a worker process on a Rust panic it cannot unwind. One
`m6id.16xlarge` run of the deep slice emitted five of them and the next run of
the same configuration emitted none, so it is intermittent.

**It fires at cluster start, not at teardown.** The console puts all five
between the `worst shard` print and the `dashboard` print:

```
worst shard   1.54 GiB, 98.3 GiB across 64 slots
thread '<unnamed>' (38475) panicked at library/core/src/panicking.rs:225:5:
panic in a function that cannot unwind
thread caused non-unwinding panic. aborting.
    [x5]
dashboard     http://127.0.0.1:45491
```

frisky 0.7.2 is the newest release on PyPI and publishes no repository URL, so
there is no upgrade and nowhere to report it. The extension is a stripped
release build.

**It costs nothing, MEASURED.** `SIGABRT` on four of eight workers mid-run is
what a non-unwinding panic does to a worker, and the run finished all 200
shards with no errors and exit 0. frisky reschedules a dead worker's task.
`tests/test_run_survives_worker_death.py` is that experiment.

**What does lose a tile quietly is a shard that raises.** `process_shard`
exceptions are caught per shard so one bad shard cannot kill the tile, which is
right, and then `main` returned 0 regardless, which was not. A run that
gathered one shard of 64 reported success and wrote a part file and a summary
to match. A driver reading the exit code, or reading the summary without
walking `shard_stats`, would call that tile done.

So the run now answers in three places at once:

| shards lost | console | `summary.json` | exit |
|---|---|---|---|
| none | nothing | `n_shards_errored: 0` | 0 |
| some | `FAILED n of m shards errored` | `n_shards_errored: n` | **3** |

Exit 3 still writes the summary and the part file, because the per-shard errors
are the post-mortem. A driver may now key on the exit code, and `summary.json`
records the same count for one that would rather read a file.

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
  raster. The largest run through the new path is one `m6id.16xlarge`, which
  staged 999 scenes and computed 64 shards of one tile.
- **The 895 tiles have never been priced against a real run.** The per-tile
  compute is measured and the tile count is measured. Their product is not.
- **One staged run has completed, over 64 shards of one tile.** It measured
  the staging rate and wrote a composite. It sampled no worker RSS, so it
  confirmed nothing about the memory model: the 56.6 GiB it printed is
  `shard_bytes` output. A later instance run sampled 35.99 GiB across 64
  workers. Neither ran a whole tile, and no fleet has run at all. Three
  `c6id.16xlarge` attempts came before it and produced no composite: the first
  wrote its results to a serial console that AWS discards on termination, the
  second stopped on a missing `pyarrow`, and the third lost its workers to the
  memory model below.
- **A worker aborting is survivable, and the panic is not the risk.** This
  entry used to say frisky aborts workers at teardown and leaves the exit code
  unknown. Both halves were wrong, and the section below has the measurements.
- **The memory model's slope on real COGs is not tightly determined.** Thirty-six
  points across four shard edges and two sources, and the model at 15 bytes
  bounds every one. The synthetic sweeps at 360 and 512 px fit 12.97 and 12.68
  bytes per pixel-scene. The shallow staged sweeps on 100 real scenes fit 14.48,
  13.59, 13.25 and 10.47 at 256, 360, 512 and 1024 px, against 14.47 and 14.44
  from the deep staged sweeps. Real scenes make shard edge and data coverage move
  together, so the scatter is partly the fixture. Only the two deep sweeps reach
  the 195 to 820 scenes a fleet shard runs, and they cover two edges of the four.
  Every sweep is EPSG:4326, so nothing reprojects. A UTM source warping into the
  output grid could hold arrays this does not count.
- **No fleet run has measured worker RSS yet.** `memory_sampler.py` is wired
  into `shard_lst_p95.py` and writes `memory.csv` and `workers_rss_peak_gib` on
  every run. One 64-shard instance run has reported it, 35.99 GiB across 64
  workers, which frisky's spans put at 35.87. No whole tile and no fleet has been
  measured, and the other check on the model at 64 workers is frisky's 1.50 GiB a
  worker on the quarter-tile run.
- **The staging rate rests on one slice of one instance.** An `m6id.16xlarge`
  staged 1,998 objects, 78.9 GiB in 91.9 s, or 922 MB/s. A mean tile writes about
  278 GB, or 302 s, at that rate. No other instance type and no whole tile has
  staged, and the fleet's staging term is that one measurement scaled.
- **The per-tile scene count in `Cost` is not the inventory's.** The 1,094.8 s
  of compute was measured against the 3,910 scenes Earth Search returned.
  The artifact assigns 4,776 to `S30W065` and 3,445 to a mean tile, so the
  compute term rests on a scene list that no longer matches the one a fleet
  would read.
- **The staged disk requirement is estimated per object, not checked.** The
  guard reserves 95 MB for a thermal band and 10 MB for a QA band, from HEADs
  over 30 scenes per platform. HEAD is billable, so nothing checks the real
  size before fetching. A slice of larger-than-average scenes falls back on the
  in-flight free-space floor.
- **The pixel mask still carries both land defects.** The tile list here drops
  the Null Island placeholder and the antimeridian slivers. The mask in
  `nlebovits/landsat-lst` does not, so a pixel inside either is composited
  rather than masked.
- **The 7.7% extra tile-scene pairs have not been priced.** The rectangle
  overhang is measured on 670 scenes in three regions, and the extra reads it
  implies scale with tile-boundary geometry rather than with scene count. No
  run has paid for them, so the S3 line in `Cost` describes the scene list a
  catalogue search returns rather than the larger one in the artifact.
- **The scene-centre offset is measured on Landsat 8 only.** 4.24 ms is
  systematic across 401 scenes of Landsat 8 2021, which is the block that
  carries microsecond timestamps. Landsat 9 publishes none, so the same
  separation cannot be run on it, and the offset is assumed to hold there.
  The 30 s guard covers 7,000 times the measured value either way.
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
- **Nobody has run past 64 cores per instance.**
- **Chunk 512 versus 256 remains unresolved** at department scale, inside the
  9.6% noise floor.
- **The department tuning covers one department at 711 scenes.** A different
  area or scene count moves the optimum, because the memory term scales with
  both.
- **The seam corrections are built and have never run on real pixels.**
  `destripe.py` defines both rules, `tile_prep.py` estimates what they need once
  per tile, and `shard_lst_p95.py` applies them. 87 unit tests cover the
  numerics and the wiring, including the claim the whole design rests on:
  splitting the tile into blocks does not move a scene offset, so the estimate
  costs one source traversal rather than the two `nlebovits/landsat-lst` pays.
  Every one of those tests runs against a synthetic stack. No tile has been
  prepped, no shard has been composited with a correction on, and
  `measure_seam.py` has produced no numbers. Until it does, the seam removal
  and the variance retained quoted in the docstrings are the sibling
  repository's measurements on its own grid, not this one's.
- **One assumption in `destripe.py` no synthetic loader can check.** Every
  per-scene value reaches a loaded stack by matching
  `destripe.timestamp_of(item)` against the `time` coordinate `odc.stac`
  produced. The unit tests replace `stac_load` with a fixture built to return
  those stamps, so they assert the assumption back at themselves.
  `tests/test_time_axis_join.py` checks it against real scenes, and it is
  `s3`-marked and has not been run. If odc-stac ever stops taking the item
  datetime as its group timestamp, `align_to_time` raises and every shard of
  the tile stops at once. That is the right failure and it is still a failure.
- **The swath comes from the data, and that is a departure.**
  `nlebovits/landsat-lst` rasterises the imaged parallelogram Earth Search
  publishes. This repository reads the USGS bulk metadata, whose corner columns
  describe the product bounding rectangle and exceed the imaged area by about
  46%, so rasterising the ring would put the cross-fade tens of kilometres off
  the seam. Counting valid observations per `(path, row)` quad answers the same
  question from the pixels instead. Comparing the two definitions on real ground
  is still owed.
- **Neighbouring tiles can disagree about both an offset and a swath.** Both
  are measured over one tile plus a 1 degree margin and no wider. So a scene that two
  tiles share gets a different offset in each, because the median anomaly is
  taken over different ground. A quad's swath moves for a second
  reason: the inventory assigns only some of that quad's scenes to each tile,
  so the half-the-scenes threshold has a different denominator. The margin
  makes the swath edges inside a tile real acquisition edges. It does not make
  two tiles agree about an edge near their shared border.
  `nlebovits/landsat-lst` has the same limit, and no merged pair of tiles has
  been inspected.
- **The prep resolution factor has no measurement behind it.** It defaults to
  4. `nlebovits/landsat-lst` validated factor 2 at a median offset error of
  0.002 C and rejected factor 4 at a maximum of 0.546 C against a
  pre-registered 0.5 C gate, on a different grid and a different loader. This
  repository owes its own sweep.
- **The sparse floor is a placeholder with a citation that does not fit it.**
  `DESTRIPE_MIN_PREP_SAMPLES = 200` comes from `nlebovits/landsat-lst`, where
  it screened a factor-2 grid over a 5 degree tile. `tile_prep` estimates on a
  factor-4 grid over the tile plus a margin, which holds roughly a fifth as
  many pixels per scene, so 200 screens a different thing here. It needs what the
  15 C cap got there, which is a sweep of the rejected share against the floor
  on a real tile.
- **The 15 C offset cap was calibrated somewhere else.** One mid-latitude
  agricultural AOI, at a 21.8% rejected share. A humid tropical tile may not
  behave that way, and the rejected share is the number to watch per tile.
- **The shard memory model has not been re-measured with the corrections on.**
  `SHARD_BYTES_PER_PIXEL_SCENE = 15` should hold or fall on a fully covered
  shard: the per-path percentile partitions each path's subset in place, where
  the pooled one partitions a copy of the whole stack. The pooled fallback adds
  to it. It reduces the uncovered pixels alone, so it costs 4 bytes per
  pixel-scene times the uncovered share: zero inside one swath, a second
  whole-stack copy on a shard the swaths miss entirely. All of this is
  arithmetic, and `worker_memory_guard` refuses runs on the constant.
- **How much ground the swaths miss is unmeasured.** A quad's swath is the
  ground where at least half its scenes produced a valid observation. A pixel
  one path sees on a third of its passes falls outside every swath and still
  carries temperatures. `feathered_percentile` composites those pixels
  pooled, and `process_shard` counts them as `n_pooled_fallback`. Whether that
  share is a rounding error or a third of a tile depends on cloud, and no tile
  has been prepped. Watch it beside the rejected share. A high one says the
  swath definition described less ground than the scenes cover, so the
  cross-fade describes less of the tile than the composite implies.
- **A capped prep run counted its swaths against the wrong denominator.**
  `--max-blocks` truncates the work list before any coverage accumulates, and
  `swath_masks` still divides each quad's per-cell count by every scene the
  inventory gave that quad. So a smoke run measured coverage over part of the
  tile, compared it against all of the tile's scenes, and wrote the small
  swaths that follow as an ordinary artifact. Every one of `load_tile_prep`'s
  refusals passed it. The meta recorded `{"planned": len(blocks), "run":
  len(stats)}`, which could not signal this either: `planned` counted the
  barren blocks that were never going to run, so an uncapped run also showed
  `run` below `planned`. The pair is now `with_scenes` against `run`, beside
  `max_blocks` and a `partial` flag, and a slice refuses a partial artifact.
- **The prep memory model is arithmetic, not a measurement.**
  `tile_prep.memory_model` names five resident terms and `--target-memory-gib`
  refuses a run that exceeds them, the way `worker_memory_guard` does for a
  shard. The shard constant was calibrated against six committed sweeps. This
  one counts array shapes and has never been checked against an RSS series. The
  term worth watching is the histogram a block returns whole to the driver, at
  104 KB a scene, so a block seeing 2,000 scenes hands back 208 MB.
- **The mask has been measured on one tile.** S30W065 is interior South
  America, entirely land, with no coastline for the water rule to cut and a
  0.24% gap share. A coastal or tropical tile would exercise both rules
  harder, and none has been checked.
- **The rule leaves 503 hot pixels on S30W065.** Each is in a cell with
  observations, so the gap test excludes it. Extending the geometry to
  `numobs <= 2` would take the tail to 98.30% and remove 11.4% of the tile, at
  which point the rule stops being a screen. The 503 stay.
- **The 70 C threshold rests on one tile.** It marks where this tile's artifact
  population separates, and nothing published bounds it. It never acts alone,
  so it makes no claim about the hottest land surface: a pixel above it outside
  a gap cell is kept. Check the tail against 70 C on the next tile with a
  substantial gap population.
- **No composite has been built with the mask on.** Every masked run so far is
  a rehearsal, which fills its shards with synthetic pixels. The mask covers
  real ground in those runs, and the temperatures under it are not real.
- **The QA comparison covers one 512 px shard at 120 scenes.** That shard sits
  inside a WRS footprint, so it measures the interior case and not the boundary
  case.
- **No 2021-2025 composite has been built.** Only its scene count is measured.

## Corrections to earlier versions of this document

Every entry is a claim an earlier version stated as fact. Each shares one
mistake: it presented an estimate as a measurement.

**Masking never changes a temperature.** The claim was that USGS wrote a gap
pixel as `ST_B10` fill, `lst_qa.not_fill` rejected it, and `qa_count` already
stood at zero, so the emissivity rule only recorded what the raster already
knew. USGS interpolates emissivity across a gap cell instead, and 89.53% of gap
pixels carry a retrieval. The rule removed 701,839 real temperatures on
S30W065. The measurement that disproves this was in the same document, twelve
lines below the claim.

**`numobs == 0` is the rule.** It was priced on its hot column alone: 77.30% of
the tile's pixels at or above 70 C, for 0.2167% of the valid ones. The column
beside it says 701,839 pixels, and 524 of the 605 gap cells contain nothing
wrong.
The rule is now the conjunction of the grown gap region and 70 C, which reaches
more of the tail for 5,432 pixels.

**A tile whose land is all gap publishes nothing.** `fleet_plan` carried a
third drop list on that reading. Such a tile publishes every ordinary
temperature it holds, so the list is gone. It had dropped no tile on the real
plan.

**S3 GET requests cost $2,067 across 895 tiles.** That multiplied 895 by the
$2.31 measured on `S30W065`, which carries 4,776 scenes against a mean of 3,445.
The global line scales on `tile_scene_rows`. Over the 3,083,129 tile-scene pairs
the inventory holds it comes to $1,822. Over the 2,905,875 with a thermal band,
which is what the 769 launched tiles read, it comes to **$1,718**, and that is
the figure `Cost` uses. The dense tile was the one that had been run, and the
arithmetic used it as the mean without saying so.

**Shard size is the largest cost lever in this pipeline.** True when written and
superseded. Moving from a 512 px shard to 1024 px cuts requests 2.8x and costs
four times the memory per shard. Staging cuts them 738x and costs disk, which is
cheaper than memory and does not cap the shard plan. Shard size is a memory
decision once staging is on. Unstaged it is still a cost lever, and a 360 px
shard makes it worse, at 265 opens per object against 145.

**The antimeridian slivers selected 45 open-ocean cells, and the Null Island
placeholder selected four: `N00E000`, `N00W005`, `S05E000`, `S05W005`.** Both
wrong, and the table in the same section already disagreed with the first.
`measure_land_defects.py` builds the tile list with each defect present and
reports the difference. The slivers select **68** cells and the placeholder
selects **3**: `N00E000`, `N00W005`, and `N05E000`. `S05E000` and `S05W005`
were never reachable, because the buffered placeholder spans 0.2291 degrees
around the origin and those cells begin five degrees south of it. `N05W005`
lies under the disc and stays in the list, because it holds the coast of Côte
d'Ivoire.

**Forty-nine of the 966 cells had no Landsat coverage, which is what open ocean
looks like from the catalogue.** True as stated and misleading as used. The
corrections remove 71 cells, not 49. The other 22 hold scenes, because Landsat
images open water. Coverage rules a cell out; it does not rule one in.

**The bulk file truncates the acquisition start and stop to whole seconds, so
the computed centre falls within 1.117 s of the published one.** Both halves
fail. Precision is mixed: Landsat 8 2021 carries microseconds, 2022 is 53%
whole-second, and everything later is whole-second. And truncating both
timestamps moves their midpoint by strictly under a second, so 1.117 s cannot
come from truncation and nothing reproduces it. Measured apart,
`measure_scene_centre.py` finds a systematic 4.24 ms definitional offset on the
untruncated rows and a worst case of 0.89 s on the truncated ones.

**Steady state across 520 tiles, at $469 to $539 of on-demand EC2 and $1,202
of S3.** Withdrawn. The 520 has no derivation in this repository or in
`nlebovits/landsat-lst`, and nothing reproduces it. The tile list is now
generated: 895 cells inside +/-60 degrees intersect Natural Earth 10m land
buffered by 25 km, which is the geometry the pixel mask uses. The nearest
figure with a derivation behind it was the 700-tile frozen set in
`landsat_lst.tiling`, built from Natural Earth 110m without a buffer. The
multiplier is now the 769 tiles that hold a thermal band, and `Cost` prices them.
The $2,067 this entry once carried is withdrawn in the entry above. The per-tile
rates are unchanged and still measured; only the multiplier moved.

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
| `README.md` | the known issues a consumer of the output has to know |
| `tests/make_land_slice.py` | cuts the committed geometry fixture from the full artifact |
| `shard_lst_p95.py` | the sharded pipeline, the slicer, and the merge |
| `profile_lst_p95.py` | the array-graph profiling harness |
| `lst_qa.py` | the QA, fill, range, and nodata rules both P95 paths call |
| `cog_catalog.py` | writes the COGs and the Portolan catalog the merge emits |
| `destripe.py` | the two seam rules both P95 paths call: scene offsets, and the per-path cross-fade |
| `tile_prep.py` | one coarse pass per tile for the offsets and the swath geometry |
| `measure_seam.py` | four composites from one load, on a shard that straddles a swath |
| `stac_window.py` | the composite window, and the cache identity it fixes |
| `land_tiles.py` | the buffered land geometry and the generated tile list |
| `masks.py` | the pixel rules: water, and the ASTER emissivity gap |
| `aster_ged.py` | the ASTER GED observation counts, built once and read per tile |
| `measure_ged_registration.py` | the mask against a composite built before it |
| `usgs_inventory.py` | the precompute stage: USGS bulk metadata to one artifact |
| `tile_inventory.py` | the runtime read of one tile, from one row group |
| `fleet_plan.py` | the driver, and the checks that run before the fleet does |
| `stac_reference.py` | Earth Search, kept only as a parity oracle |
| `artifacts/` | `land_tiles.parquet`, the inventory, the buffered geometry, the ASTER GED counts, their manifests, and the committed slices of the three that are gitignored |
| `compare_qa_masks.py` | one shard, run under both masks, in one process |
| `qa-parity/` | that comparison, with both rasters and the difference image |
| `sweep_throughput.py` | configuration sweep driver |
| `cost_report.py` | the labelled, deterministic cost report |
| `staging.py` | fetches each scene object once, beside compute, and the L2SR filter |
| `item_table.py` | the scene table every worker reads instead of receiving |
| `memory_sampler.py` | client and worker RSS, sampled from its own process |
| `measure_shard_memory.py` | what a shard costs in memory, and shard size in compute |
| `measure_s3_requests.py` | counts the S3 GET requests one shard issues |
| `measure_submit_cost.py` | what one `client.submit` costs, against its payload |
| `s3-requests/` | the request measurement: both shard sizes, and the priced tile |
| `dryrun/` | local graph-build runs, no cluster and no reads |
| `ec2-results/` | eight department-scale runs: stages, memory series, frisky reports |
| `fulltile/` | the full-tile run (part files and merged raster gitignored) |
| `smoke/`, `smoke2/`, `split/`, `sweep/` | local runs with full frisky spans and traces |
| `full/`, `full.log` | the 673-scene local run, ended early by a network change |

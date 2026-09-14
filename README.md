# landsat-lst-smoke

## The composite

One tile is one lazy dask-xarray graph. `odc.stac.load` opens every scene of
the tile once, chunked in space and never in time, because a percentile over
time needs every scene of a pixel in one block. Each block is masked,
corrected, reduced and encoded inside one task, and the blocks stream from
the workers into the two COGs the catalog publishes. The driver receives
nothing larger than one block. Every phase the driver runs around that graph
is a frisky client phase, so the dashboard shows the current phase and its
elapsed time while it runs.
`--rehearse N` runs the whole pipeline over N synthetic scenes on local disk,
with no S3 reads, and tags every line and every artifact `REHEARSAL:`.

## Removing the WRS seam

A composite pixel is reduced from the scenes overlapping it, so the value jumps
where that set of scenes changes. The jumps trace WRS-2 footprints
across the raster at about 10 degrees from vertical. A scene offset and a
per-path cross-fade remove them, and neither runs unless the run is given a
prep file.

Build the prep file once per tile, then composite against it:

```bash
uv run tile_prep.py --tile S30W065 --out-dir ./tile-prep
uv run shard_lst_p95.py --tile S30W065 --tile-prep ./tile-prep \
    --stage-dir /mnt/nvme/stage --out-dir ./run
frisky observe overview ./run/spans.json
```

`tile_prep.py` reads the tile once at a quarter of the output resolution and
writes two things a block cannot work out for itself.

**One offset per scene.** Landsat Collection 2 surface temperature is
atmospherically corrected one scene at a time, with a published error of 1 to
5 K that applies to the whole scene. Each scene is compared against a per-pixel
median for its own calendar month, pooled across every year in the window, and
shifted by its bulk deviation. The month is what makes the reference safe to
subtract. An annual reference absorbs the seasonal cycle, and
`nlebovits/landsat-lst` measured that failure at 40.6 C down to 29.8 C. A scene
whose offset exceeds 15 C is discarded rather than clamped, which also means
`qa_count` reports the evidence behind the P95 instead of raw availability.

**One swath per WRS path, and cross-fade weights on it.** The offset is fitted
at the median and the product is a P95, so a tail difference between paths
survives it. Building one percentile per path and blending them on distance to
each swath edge removes that step. A pixel one path reaches is untouched.

The swath comes from the pixels rather than from the item footprints. This
repository reads the USGS bulk metadata, whose corner columns describe the
product bounding rectangle and exceed the imaged area by about 46%. The seam
lies along the imaged edge. So a `(path, row)` quad's swath is the ground where
at least half its scenes produced a valid observation.

Half is a threshold, so ground a path sees less often falls outside every swath
and still carries temperatures. Those pixels take the pooled percentile. The
cross-fade weights one path's estimate against another, and outside every swath
there is one estimate to weigh, so the blend is the pooled value already. The
run summary counts them as `n_pooled_fallback`. A nodata pixel keeps the meaning
it had: nothing was observed there.

Both corrections run inside the block the graph has already loaded, so neither
adds a read and neither adds a pass. The prep file is the extra traversal,
taken once per tile and shared by every block of it.

Turn either off with `--no-destripe` or `--no-feather`. Add `--emit-pooled` to
write `lst_p95_pooled.tif` beside the product. It comes out of the same blocks
after the offsets are applied, so it isolates the cross-fade and says nothing
about the offsets. It also costs 2 bytes per output pixel, which
`composite.block_bytes` counts against the machine before the cluster starts.

**Not yet measured here.** Every number above comes from
`nlebovits/landsat-lst` on its own grid. No tile in this repository has been
prepped and no block has been composited with a correction on. See "What is
not settled" in `FINDINGS.md`.

## Known issues

### A corrected composite measures something else

A de-striped P95 measures how hot a surface gets against its own monthly
normal. The pooled P95 this repository built before measures the hottest
observed value. Both rasters look alike, and no pixel in either one states
which rule produced it. So differencing a corrected tile against an uncorrected
one measures the rule rather than the ground.
`nlebovits/landsat-lst` measured the correction cooling the P95 by about 4 C at
a 21.8% scene rejection.

Every published item states its rule in `processing:lineage`, beside the mask
rules and naming the scene set the offsets were fitted over. Read it before
comparing two tiles. One graph composites one tile under one rule, so a tile
cannot be built under two. A collection can still hold tiles built under
several, so the item is where the claim belongs.

### Evidence, plausibility, and water

A published pixel has a temperature only where every rule below agrees.
`lst_qa.py` defines the two that describe the estimate, and
`composite.reduce_block` applies them. `masks.py` defines the one that
describes the place.

| Rule | Threshold | What a nodata pixel says |
|---|---|---|
| Evidence | 5 clear observations over the window | too few scenes to estimate a percentile |
| Plausibility | -20 C to 80 C, inclusive | the retrieval failed |
| Water | outside the buffered land geometry | never this product's subject |

The water rule zeroes `qa_count` with the temperature, so a count of 0 beside a
nodata pixel is the signature of sea. The other two leave `qa_count` standing,
so a count of 1 to 4 beside a nodata pixel is the evidence rule, and a count
above 5 beside one is a value outside the bounds. Sum the 12 bands to read it.

The evidence rule cuts a thin tail. MEASURED across the five audited tiles: it
removes 0.0003% of N30E075, 0.0005% of S30W065, 0.0069% of N40W080, 0.5965% of
N00E110, and 0.8035% of S25E030 of valid land. Median total observations run
from 56 on N00E110 to 160 on N40W080. A higher floor stops being a tail: 50
observations would remove 39% of N00E110.

Sparse evidence, rather than cold ground, is what reaches the cold bound.
MEASURED, the floor alone lifts the published N00E110 minimum from -39.82 C to
19.96 C and the S25E030 minimum from -0.25 C to 9.84 C, and no pixel of the
five tiles then falls below -20 C. The cold bound is a second line for a sparse
pixel the floor lets through.

The hot bound is the same number as the per-observation ceiling, applied again
because the first application does not hold. `destripe.subtract_offsets` shifts
a decoded value after `lst_qa.in_trusted_range` has passed it. MEASURED on
N30E075, 205 published pixels read above 80 C and the bound removes them, which
takes the tile maximum from 82.99 C to 80.00 C.

The ceiling stays at 80 C on purpose, and the pixels between 60 C and 80 C
that pass it were measured rather than assumed. Some are broad hot ground with
a spatial gradient. MEASURED: rural Rajasthan holds 56 C across 28 km, and the
Baltimore urban core and the South Kalimantan coal belt each decay by 5 C over
8 km. The rest are isolated pixels 25 C above everything within a kilometre, on
a flat radial profile, in the Haryana brick-kiln belt and industrial Cordoba.

A P95 over N observations is near the 0.05N-th hottest value. These pixels have
a median of 190 observations, so a P95 of 70 C needs about ten separate
observations at or above 70 C across five years, which no one-off artifact can
produce. Persistent sub-pixel thermal sources explain them. A 60 C ceiling would
cost under 0.02% of every audited tile and would also delete rural Rajasthan.
UNKNOWN: the ground truth of the isolated spikes. A VIIRS active-fire or gas
flare inventory read against their coordinates would settle it.

### The ASTER GED gap region is reported, not masked

Where ASTER GED caught no clear sky between 2000 and 2008, USGS interpolates
emissivity from the neighbouring cells and retrieves a temperature anyway, and
some of those retrievals fail upward. `masks.py` measures how far that region
reaches and `output_mask` reports it per tile, so a reader can see how much of
a tile rests on interpolated emissivity. A pixel inside it keeps its
temperature.

Masking the region alone removes far more than it should. MEASURED on S30W065:
524 of the 605 gap cells have no pixel at or above 70 C, and masking the
geometry alone removes 701,839 valid pixels to remove 4,588 bad ones.
`nlebovits/landsat-lst` applied the geometry alone, measured 2,799,286 pixels
removed for 2,582 artifacts, and replaced it with a pair: the region and a 70 C
threshold together. This repository shipped that pair and has now withdrawn it.
MEASURED across five tiles, the two halves do not coincide. On N30E075 all 207
pixels at or above 80 C fall outside the region and its one-cell buffer, so the
pair reached none of them. The unconditional 80 C bound above replaced it.

### A cold wedge along WRS path 013 in N40W080

Philadelphia carried a cold swath-edge artifact from path-specific feather
weighting inside `destripe.feathered_percentile`. The pixels are well observed
and wrongly weighted, so `MIN_TOTAL_OBSERVATIONS` does not reach them and
should not. They carry a median of 94 observations. What decides the value is a
different count, the observations one path contributes, and it comes from the
scenes rather than from the product.

MEASURED on one 360 by 360 block of that wedge, rebuilt from its own 765 scenes
and reproduced against the published raster to 0 DN on all 129,600 pixels. Path
013 took a weight of 0.983 over 98.4% of the block on a median of one valid
observation per pixel. Path 014 contributed a median of 159 observations and
took a weight of 0.001. The block published 11.45 C where its neighbours read
33.8 to 39.1 C.

`DESTRIPE_MIN_PATH_OBSERVATIONS` is the fix. A path needs five valid
observations at a pixel before it takes that path's geometric weight there. A
path below the floor drops out and the rest renormalise, which the blend already
did for a path that observed nothing.

MEASURED on five blocks, each reproducing the published product at a floor of
one. The defect block moves to 37.74 C. The seam falls from 60.9 times the
background gradient to 1.16, the ratio ordinary terrain shows. A well-observed
control block, a mostly uncovered block, and one on N00E110 each move 0.00 C.

### Cloud decides how much of a tile exists

Over persistent cloud a five-year window returns nothing to composite. This is
not a rule this repository applies. It is the absence of a clear observation,
and no compositing rule recovers a pixel nothing ever saw.

MEASURED across the five audited tiles, land pixels with no clear observation
in 2021 to 2025:

| tile | place | land pixels | no clear observation |
|---|---|---|---|
| `N30E075` | Delhi, north India | 324,000,000 | 27,915, 0.009% |
| `S30W065` | interior Argentina | 324,000,000 | 85,489, 0.03% |
| `N40W080` | Philadelphia, Pennsylvania | 307,324,200 | 289,580, 0.09% |
| `S25E030` | Durban, South Africa | 178,974,823 | 35,754,489, 20.0% |
| `N00E110` | Borneo, Kalimantan | 233,439,938 | 105,608,892, **45.2%** |

Read `qa_count` before reading a temperature. Its twelve bands sum to the
evidence behind each pixel, and a tile can be 45% empty without any band of
`lst_p95` saying so.

The vertical stripes across `N00E110` are this, and not a seam. MEASURED on
that tile, the column means of `lst_p95` step by 0.058 C between adjacent
columns, and the horizontal gradient across a change of dominant WRS path is
1.14 times the gradient elsewhere. The stripes are the nodata pattern showing
through.

ASTER GED predicts where this happens, which is useful before a run rather than
after one. A GED gap cell is ground ASTER caught no clear sky over between 2000
and 2008, and Landsat fails on the same ground for the same reason. MEASURED,
land pixels with no clear observation that fall outside a GED gap: 0 of 289,580
on `N40W080`, 0 of 35,754,489 on `S25E030`, 5,895 of 105,608,892 on `N00E110`,
and 77 of 85,489 on `S30W065`. `N30E075` has no gap cells and also has almost
no empty land, 27,915 pixels of 324 million. The gap is the wider mask: 78.7%
of `N00E110`'s land sits in one against the 45.2% that came back empty.

So `artifacts/aster_numobs.tif`, a static 43 MB global raster this repository
already builds, bounds how much of a tile can exist before any scene is
read.

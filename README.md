# landsat-lst-smoke

## The composite

One tile is one lazy dask-xarray graph. `odc.stac.load` opens every scene of
the tile once, chunked in space and never in time, because a percentile over
time needs every scene of a pixel in one block. One task masks,
corrects, reduces, and encodes each block, and the blocks stream from the
workers into the two COGs the catalog publishes. The driver receives
nothing larger than one block. Every phase the driver runs around that graph
is a frisky client phase. The dashboard shows the current phase and its
elapsed time while it runs.
`--rehearse N` runs the whole pipeline over N synthetic scenes on local disk,
with no S3 reads, and tags every line and every artifact `REHEARSAL:`.

## Removing the WRS seam

The graph reduces a composite pixel from the scenes overlapping it, so the
value jumps where that set of scenes changes. The jumps trace WRS-2 footprints
across the raster at about 10 degrees from vertical. A scene offset and a
per-path cross-fade remove them, and neither runs unless you give the run a
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
5 K that applies to the whole scene. `tile_prep.py` compares each scene against a
per-pixel median for its own calendar month, pooled across every year in the
window, then shifts the scene by its bulk deviation. The month is what makes the reference safe to
subtract. An annual reference hides the seasonal cycle, and
`nlebovits/landsat-lst` measured that failure at 40.6 C down to 29.8 C. `tile_prep.py`
discards a scene whose offset exceeds 15 C rather than clamping it, which also
means
`qa_count` reports the evidence behind the P95 instead of raw availability.

**One swath per WRS path, and cross-fade weights on it.** `tile_prep.py` fits the offset
at the median, and the product is a P95, so a tail difference between paths
survives it. Building one percentile per path and blending them on distance to
each swath edge removes that step. A pixel that only one path covers takes that path's estimate unchanged.

The swath comes from the pixels rather than from the item footprints. This
repository reads the USGS bulk metadata, whose corner columns describe the
product bounding rectangle and exceed the imaged area by about 46%. The seam
lies along the imaged edge. So a `(path, row)` quad's swath is the ground where
at least half its scenes produced a valid observation.

Half is a threshold, so ground a path sees less often falls outside every swath
and still carries temperatures. Those pixels take the pooled percentile. The
cross-fade weights one path's estimate against another, and outside every swath
there is one estimate to weigh, so the blend is the pooled value already. A
nodata pixel keeps the meaning it had. No scene imaged that ground.

Every run states how much of its raster went that way. `pooled_share` in
`summary.json` is `n_pooled_fallback_retained / retained_pixels`, and the item's
`processing:lineage` carries the same figure in words. The numerator counts
pooled pixels present in the published raster, after the output mask, so it
divides by the count the coverage figures already use. `n_pooled_fallback` is
the kernel's decision before that mask and stays in the summary because the
measurements in `FINDINGS.md` quote it. Both appear on every tile, zero
included.

A whole WRS path can stay under the threshold everywhere on one tile. Its
scenes still load and still reach `qa_count`. They feed the pooled fallback and
take no part in the blend. The prep run used to refuse such a tile and name
`--no-feather` as the way past it. A tile then waited on an operator reading
that message.

The run records the path instead. Its name and its scene count reach three
places: the prep output, `summary.json` under
`correction.paths_without_swath`, and the item lineage. The exclusion is
narrower than the refusal implied.
`DESTRIPE_MIN_PATH_OBSERVATIONS` already drops a thin path from one pixel's
blend and renormalises the rest. A swath-less path is the limiting case of a
rule the composite applies everywhere.

Both corrections run inside the block the graph has already loaded, so neither
adds a read and neither adds a pass. The prep file is the extra traversal,
taken once per tile and shared by every block of it.

Turn either off with `--no-destripe` or `--no-feather`. Add `--emit-pooled` to
write `lst_p95_pooled.tif` beside the product. It comes out of the same blocks
after the graph applies the offsets, so it isolates the cross-fade and says
nothing about the offsets. It also costs 2 bytes per output pixel, which
`composite.block_bytes` counts against the machine before the cluster starts.

**Not yet measured here.** Every number above comes from
`nlebovits/landsat-lst` on its own grid. This repository has prepped no tile,
and it has composited no block with a correction on. See "What is
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
rules, and names the scene set the fit covered. Read it before
comparing two tiles. One graph composites one tile under one rule, so no tile
carries two. A collection can still hold tiles built under
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
| Land geometry | outside the buffered land geometry | never this product's subject |
| Observed water | 90% of clear observations set QA_PIXEL bit 7, and the percentile is at or below 34 C | the surface is water |

Read `qa_count` beside a nodata pixel as evidence, not as an answer. The two
water rules zero the count with the temperature. The other two leave the count
standing, so a count of 1 to 4 is the evidence rule and a count of 5 or more is
a temperature bound. Sum the 12 bands to read it.

A count of 0 identifies nothing on its own. No scene covered the pixel. Or
scenes covered it and every observation failed the QA or range rule. Or the
land geometry removed it. Or the observed-water rule removed it. An earlier
version of this section called a zero count the signature of sea. It never was,
even before the water rule existed.

The land geometry reads land grown by 25 km. That growth stops the mask cutting
a coastal scene at the waterline, and it reaches open sea. So the geometry
decides pixels and does not define land.

### The water a scene photographed

The buffered geometry leaves sea inside the buffer and cannot see a river at
all. The observations can. QA_PIXEL sets bit 7 over water, and
`lst_qa.observed_water` asks what share of a pixel's usable clear observations
did so over the whole five years. At or above 90%, the pixel leaves the product.

One bit is not enough on its own. A reflectance test sets that flag, and it
matches a dark roof or a rail yard on almost every scene. MEASURED in Center
City Philadelphia: 143 pixels reading 36 C to 56 C carried the flag on 86% to
100% of their 80 clear observations. No threshold on the share excludes them,
because their share is the share of open sea.

Water has a ceiling that asphalt does not. So a pixel whose percentile exceeds
`WATER_MAX_C`, 34 C, keeps its temperature whatever the flag counted. Two
blocks with known truth set that bound. The Center City box excludes both
rivers, and the Delaware Bay block lies offshore of either bank.

| bound | Center City masked | Delaware Bay kept | Chesapeake kept |
|---|---|---|---|
| 40 C | 1.33% | 100.00% | 93.07% |
| 36 C | 0.70% | 100.00% | 93.07% |
| 35 C | 0.26% | 100.00% | 93.07% |
| 34 C | 0.00% | 100.00% | 93.06% |
| 32 C | 0.00% | 100.00% | 92.92% |

34 C is where the false positives end and before the cost to open water starts.
It removes every false positive in that box and takes 0.01% of the Chesapeake.

Temperate water set the bound, and a later probe tested it against Kalimantan.
The warm-water fear turned out smaller than expected. The Barito estuary
classifies at a median of 33.18 C and the Kahayan river at 33.92 C, both under
the bound. The Kahayan upper quartile of 34.74 C does cross it, so that slice
of river keeps its temperature and the product publishes it as land. The rule
errs that way deliberately. A warm pond published as land costs less than a
warm roof deleted as water.

One thermal alternative lost on the numbers. Water holds steady across
observations where asphalt swings: the per-pixel spread runs 8.30 C over
Delaware Bay against 12.42 C over Center City. A spread bound at 9.5 C also
reaches 0.00%, costs the Chesapeake 0.24% against 0.01%, and needs a third
accumulator in the kernel.

`composite.reduce_block` counts the share from two unsaturated counters, and
`composite.finalize_block` applies it beside the geometry. The rule drops no
observation before the percentile runs, so a retained pixel is bit-identical to
what the same run produced without it. `tile_prep.py` fits the destripe offsets
and the feather weights and freezes them before the composite starts, so a mask
applied after the percentile cannot move them.

MEASURED 2026-09-15 on four blocks rebuilt from their scenes, each 360 px square
with its whole five-year stack:

| block | scenes | clear observations, median | classified water | published p95 |
|---|---|---|---|---|
| Chesapeake open water | 756 | 322 | 93.06% | 28.69 C |
| Delaware Bay | 190 | 110 | 100.00% | 27.42 C |
| Delaware shoreline | 569 | 99 | 99.72% | 28.41 C |
| Rajasthan desert | 809 | 280 | 0.00% | 55.73 C |

The desert block is the false-positive test, and it classifies no pixel as
water. Bright sand at 55.73 C is what a cloud-and-water classifier confuses, and
this one does not.

The threshold comes from the distribution rather than from convention. MEASURED
over 17 cached blocks of `N40W080`, 2,199,809 observed pixels: 93.8% sit below a
share of 0.05 and 3.0% sit at or above 0.90. The published temperature agrees.
On the Philadelphia river block, pixels below 0.05 have a median of 44.82 C and
pixels above 0.90 have a median of 28.92 C inside a 1.6 C interquartile band.
The bands between run down monotonically: 43.49 C, 39.24 C, 33.87 C. A
threshold of 0.25 would reach a band whose median is 45.00 C, which is land.

### The tropics set the threshold, not the temperate blocks

Left to `N40W080` alone the answer would be 0.75, its emptiest bin. Kalimantan
disagrees. MEASURED on a forest block of `N00E110`, the share ramps smoothly
from 0 to 1 with no empty bin anywhere, and the band at [0.75, 0.90) reads
31.60 C against forest at 33.65 C. Two degrees, where the temperate river sits
eleven below its bank. Those pixels are as likely canopy as stream.

Raising the threshold buys that doubt back for almost nothing, at a bound of
34 C:

| block | truth | at 0.75 | at 0.90 |
|---|---|---|---|
| Delaware Bay | all water | 100.0000% | 100.0000% |
| Chesapeake | mostly water | 93.0602% | 93.0046% |
| Delaware shoreline | mostly water | 99.7207% | 99.6998% |
| Barito estuary | tropical water | 12.0725% | 12.0725% |
| Kahayan river | tropical water | 6.3156% | 6.3156% |
| Kalimantan forest | tropical land | 0.4599% | 0.1111% |
| Sebangau peat | tropical land | 0.0000% | 0.0000% |
| Rajasthan desert | arid land | 0.0000% | 0.0000% |
| Center City | no water | 0.0000% | 0.0000% |

Delaware Bay loses nothing, the Chesapeake loses 0.056%, and the ambiguous
forest classifications fall four times over. Both rules now err the same way.
A pixel of water published as land costs this product less than a pixel of land
deleted as water.

Much of the tropical water is already absent before any rule runs. 69% of the
Barito block and 80% of the Kahayan block carry no usable clear observation.
Landsat retrieves no surface temperature over water, and cloud accounts for
most of the rest.

The rule keeps flooded peat forest, which is the failure this probe went
looking for. Sebangau classifies 3 pixels of 22,515.

Until 2026-09-15 every published share of land divided by it. MEASURED at 3600
pixels per degree, `S40W065` holds 73,254,945 pixels of processing mask and
45,407,126 pixels of land. An item reported the first under the name
`lst:land_pixels`. The properties below state which footprint each one counts.

| Property | Footprint |
|---|---|
| `lst:land_pixels` | Natural Earth 10m land, unbuffered |
| `lst:processing_mask_pixels` | that land grown by 25 km, which the run masked on |
| `lst:coastal_buffer_pixels` | the difference, which is sea |
| `lst:coastal_buffer_valid_pixels` | values the product publishes over that sea |

`lst:valid_fraction`, `lst:empty_land_pixels`, and `lst:ged_gap_fraction` divide
by `lst:land_pixels`. `masks.land_split` builds both masks from one method and
refuses a pair where land reaches outside the mask, which is how two Natural
Earth releases would show up. `land_tiles.py --write-strict-geometry` writes the
unbuffered artifact.

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

The ceiling stays at 80 C on purpose. This repository measured the pixels
between 60 C and 80 C that pass it rather than assuming them. Some are broad hot ground with
a spatial gradient. MEASURED: rural Rajasthan holds 56 C across 28 km, and the
Baltimore urban core and the South Kalimantan coal belt each decay by 5 C over
8 km. The rest are single pixels, 25 C above everything within a kilometre, on a flat
radial profile, in the Haryana brick-kiln belt and industrial Cordoba.

A P95 over N observations is near the 0.05N-th hottest value. These pixels have
a median of 190 observations. A P95 of 70 C then needs about ten separate
observations at or above 70 C across five years. No one-off artifact produces
that. Persistent sub-pixel thermal sources explain them. A 60 C ceiling would
cost under 0.02% of every audited tile and would also delete rural Rajasthan.
UNKNOWN: the ground truth of the isolated spikes. A VIIRS active-fire or gas
flare inventory read against their coordinates would settle it.

### `output_mask` reports the ASTER GED gap region rather than masking it

Where ASTER GED caught no clear sky between 2000 and 2008, USGS interpolates
emissivity from the neighbouring cells and retrieves a temperature anyway, and
some of those retrievals fail upward. `masks.py` measures how far that region
reaches and `output_mask` reports it per tile, so a reader can see how much of
a tile rests on interpolated emissivity. A pixel inside it keeps its
temperature.

Masking the region alone removes far more than it should. MEASURED on S30W065:
524 of the 605 gap cells have no pixel at or above 70 C. Masking the geometry
alone removes 701,839 valid pixels to remove 4,588 bad ones.
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

| tile | place | land pixels | processing mask | no usable observation |
|---|---|---|---|---|
| `N30E075` | Delhi, north India | 324,000,000 | 324,000,000 | 27,915, 0.009% |
| `S30W065` | interior Argentina | 324,000,000 | 324,000,000 | 85,489, 0.03% |
| `N40W080` | Philadelphia, Pennsylvania | 267,003,973 | 307,324,200 | 289,580, 0.09% |
| `S25E030` | Durban, South Africa | 158,478,897 | 178,974,823 | 35,754,489, 20.0% |
| `N00E110` | Borneo, Kalimantan | 209,193,037 | 233,439,938 | 105,608,892, **45.2%** |

Two denominators, because the run that produced the last column divided by the
wrong one. The processing mask is Natural Earth land grown by 25 km, so that a
coastal scene is not cut at the waterline, and it reaches open sea. The land
column is the same geometry unbuffered, MEASURED 2026-09-15 at 3600 pixels per
degree. The percentages are the 2026-09-14 run's own, against the mask.
`publish_catalog.py recount` restates every share against land and reads the
count off the published raster, touching no pixel of it.

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
land pixels with no clear observation that fall outside a GED gap:

- 0 of 289,580 on `N40W080`
- 0 of 35,754,489 on `S25E030`
- 5,895 of 105,608,892 on `N00E110`
- 77 of 85,489 on `S30W065`
 `N30E075` has no gap cells and also has almost
no empty land, 27,915 pixels of 324 million. The gap is the wider mask: 78.7%
of `N00E110`'s land sits in one against the 45.2% that came back empty.

So `artifacts/aster_numobs.tif`, a static 43 MB global raster this repository
already builds, bounds how much of a tile can exist before the graph reads any
scene.

### A scene's footprint is its file, not the ground it photographed

Sharp-edged rectangles of nodata appear inside fully observed farmland, a few
hundred pixels across. Their `qa_count` reads zero in all twelve months while
the ground on every side has 125 to 200 observations. They are neither cloud
nor a pipeline defect.

A Landsat scene is a rotated parallelogram written into an axis-aligned
GeoTIFF, and the corners of that file are fill. The USGS bulk metadata describes
that file: its `corner_*` columns are the product bounding rectangle, which
exceeds the imaged area by about 46%. So the stated footprint of hundreds of
scenes can cover a pixel that none of them photographed.

MEASURED at 26.49 S 61.64 W on `S25W065` and 30.08 S 58.73 W on `S30W060`. The
inventory reports 295 and 212 scenes under 20% cloud whose footprint contains
the point. Reading 30 source pixels at each, across three WRS rows: **60 of 60
are source fill.** No scene imaged them and none came back clear, so the QA
mask rejected nothing. Nothing was there to reject.

The rectangles run 320 by 673, 752 by 641, and 631 by 817 pixels, and their
edges fall nowhere near the 360 pixel block grid the composite writes on. They
cost 0.78% of `S25W065`'s land and 0.53% of `S30W060`'s.

The published rasters are right. `qa_count` reads zero because zero
observations exist, and no compositing rule invents a value from none. The
account the product gives of itself is what misleads: `lst:empty_land_pixels`
counts ground Landsat never photographed alongside ground where every
observation failed the QA or range rule. A wider window helps the second case.
It can do nothing for the first.

`qa_count` cannot separate them. It counts pixels where
`not_fill AND qa_clear AND in_trusted_range` all held. Neither a source fill nor
a rejected observation raises that count. Both cases read the same on disk.
`lst_p95` holds 0 and `qa_count` holds 0 in all twelve bands.

Read a straight-edged region of zeros as ground outside Landsat's imaged
footprint, whatever the scene rectangles say. Those pixels are not cloudy.

Splitting the count needs a per-pixel record of source presence that no
published artifact carries. `lst_qa.not_fill` produces it once per scene per
pixel and `lst_qa.masked_celsius` ANDs it with the QA and range tests in the
same statement, so nothing downstream can recover it. A bare `!= 0` test on the
loaded plane is not a substitute either. Reprojection interpolates fill against
data, so small non-zero DNs appear along a scene edge, decode near -124 C, and
pass every test except the range check. The split belongs in
`composite.reduce_block`, as one boolean reduction over time, and it can describe
only the tiles built after that change.

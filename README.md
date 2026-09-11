# landsat-lst-smoke

## Removing the WRS seam

A composite pixel is reduced from the scenes overlapping it, so the value jumps
where that set of scenes changes. The jumps trace WRS-2 footprints
across the raster at about 10 degrees from vertical. A scene offset and a
per-path cross-fade remove them, and neither runs unless the run is given a
prep file.

Build the prep file once per tile, then composite against it:

```bash
uv run tile_prep.py \
  --tile S30W065 \
  --out-dir ./tile-prep

uv run shard_lst_p95.py \
  --tile S30W065 \
  --tile-prep ./tile-prep \
  --out-dir ./shard-run
```

`tile_prep.py` reads the tile once at a quarter of the output resolution and
writes two things a shard cannot work out for itself.

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

Both corrections run on the stack a shard has already loaded, so neither adds a
read and neither adds a pass. The prep file is the extra traversal, taken once
per tile and shared by every slice.

Turn either off with `--no-destripe` or `--no-feather`. Add `--emit-pooled` to
write the uncorrected percentile beside the product, built from identical
scenes and identical reads. `measure_seam.py` runs all four arms on one shard
that straddles a swath and reports what each one removed.

**Not yet measured here.** Every number above comes from
`nlebovits/landsat-lst` on its own grid. No tile in this repository has been
prepped and no shard has been composited with a correction on. See "What is not
settled" in `FINDINGS.md`.

## Known issues

### Hot pixels left by ASTER GED coverage gaps

Where ASTER GED caught no clear sky between 2000 and 2008, USGS interpolates
emissivity from the neighbouring cells and retrieves a temperature anyway, and
some of those retrievals fail upward. `masks.py` drops a pixel only where two
things are true together: its GED cell reports zero observations or lies one
cell from such a cell, and the pixel reads 70 C or hotter.

A gap cell records where ASTER missed the ground. It says nothing about the
retrieval that consumed the interpolated emissivity, so the geometry alone
removes far more than it should.
MEASURED on S30W065: 524 of the 605 gap cells have no pixel at or above 70 C.
In the 81 that do, the hot pixels are 4.77% of the cell. Masking the geometry
alone removes 701,839 valid pixels to remove 4,588 bad ones. The pair removes
5,432 and reaches more of the tail.

503 hot pixels survive on that tile, in cells with observations. The 70 C
threshold is a screen calibrated on one tile with no published source, so it
makes no claim about the hottest land surface, and a pixel above it outside a
gap is kept. `nlebovits/landsat-lst` applied the geometry alone, measured
2,799,286 pixels removed for 2,582 artifacts, and replaced it with this pair.

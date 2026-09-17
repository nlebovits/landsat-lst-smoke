# Agent guidance

Read [`README.md`](./README.md) first. It is the canonical description of this dataset, and the source repository serves the same text. This document adds what that prose leaves out.

## Reading the data

The `lst-p95-2021-2025` collection has one item per tile of the degree grid, 769 of them. Each item carries two Cloud Optimized GeoTIFFs. `N15E025` is one tile. A tile takes its name from its north and west edges. `S30W065` starts at 30 S, 65 W.

Read a window rather than the whole file. The internal tiles measure 512 by 512 pixels. The overviews store a reduced copy for a whole-tile draw.

## Decoding temperature

```python
import rioxarray

# The decoding rule travels with the file. Scale, offset and nodata
# all come out of the COG, and none of them is repeated here.
celsius = rioxarray.open_rasterio(
    "https://data.source.coop/nlebovits/landsat-lst/lst-p95-2021-2025/N15E025/lst_p95.tif",
    mask_and_scale=True,
)
```

DN 0 is nodata. Treat it as absent rather than cold.

## Nodata semantics

The raster separates none of these meanings:

| Meaning | Rule | Would a wider window fix it |
|---|---|---|
| No usable observation | every observation failed the QA or range rule | yes |
| Never imaged | the pixel is off every imaged footprint, whatever the scene rectangles say | no |
| Outside the land geometry | the pixel is beyond Natural Earth land grown by 25 km | no |
| Observed water | at least 90% of its clear observations set QA_PIXEL bit 7, and it is no hotter than 34 C | no |
| Too little evidence | fewer than 5 clear observations over the whole window | yes |
| Physically impossible | below -20 C or above 80 C | no |

Read `qa_count` beside the pixel, as evidence rather than as an answer. A count above 0 narrows the pixel to the last two rows, and the count itself stands. 1 to 4 is the evidence rule. 5 or more is a temperature bound. A count of 0 covers the other four rows together, and nothing in the raster tells them apart.

Both water rules zero the count with the temperature. Neither a source fill nor a rejected observation increments it. A zero count is no signature of sea, and a nodata pixel is no cloudy one. Each item's `processing:lineage` states every rule, and references by checksum the artifacts each rule reads.

Shape is the one clue the raster does offer. A straight-edged region of zeros is ground outside Landsat's imaged footprint. The scene rectangles that cover it describe files rather than ground.

## Reading the observation counts

`qa_count` has 12 bands, January through December. Band `m` counts the clear observations that entered the percentile for that calendar month, pooled across every year in the window. The count saturates at 255. A zero is a real count rather than a gap, and the band declares no nodata value.

## Comparing two tiles

Check each item's `processing:lineage` first. A tile built with the WRS seam correction carries a percentile taken against each scene's own calendar-month median. An uncorrected tile carries the hottest observed value. Both rasters look alike, and the numbers answer different questions. A difference between tiles built under different rules measures the rule rather than the ground. The lineage records which rule ran and which scenes fitted the offsets.

## Cross-referencing

Read [`items.parquet`](https://data.source.coop/nlebovits/landsat-lst/lst-p95-2021-2025/items.parquet). It gives every item's metadata in one range request. The alternative is one fetch per item, 769 of them.

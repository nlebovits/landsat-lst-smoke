# Landsat LST P95 composites (2021–2025)

This repository contains processing code and metadata for scene-normalized P95
composites of land surface temperature. DERIVED from collection metadata, the
composites use [Landsat 8/9](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature)
observations from 2021 through 2025. [Source Cooperative](https://source.coop/)
publishes the data.

The composites show persistent spatial patterns in extreme surface heat while
reducing artifacts from atmospheric differences between scenes and Landsat's
orbital sampling geometry. The COGs express values in degrees Celsius, but the
values differ from an unmodified P95 of the observations.

## Philosophy

Our goal is to make high-resolution data about extreme surface heat widely
accessible as a decision-support tool. Land surface temperature (LST) does not
fully measure human heat exposure. We publish these data for communities that
might otherwise lack any heat data.

Many communities exposed to extreme heat lack local data and the resources to
run models such as [SOLWEIG](https://umep-dev.github.io/solweig/). Work on
[open-source heat-risk datasets](https://www.wri.org/research/mapping-scenarios-and-estimating-potential-heat-resilient-infrastructure-cities)
often focuses on large, high-impact locations such as the United States, Delhi,
Mexico City, and Nairobi. Smaller cities may wait indefinitely for better open
datasets in their region.

LST is freely available, globally consistent, and scientifically well
understood. It reveals spatial patterns in how landscapes absorb and retain
heat. These patterns can inform tree-planting priorities and analysis of the
relationship between land cover and extreme heat.

This dataset is a practical decision-support resource. Users should
interpret it alongside tree canopy height, building footprints and heights,
and land cover. Together, these data help explain persistent heat patterns and
identify places for intervention.

We consider this dataset a `v0.1`. We prioritized methods that are simple and
transparent, even when they leave coverage gaps. Ideal data products may remain
unavailable globally for years. We see value in publishing an open-source,
fit-for-purpose dataset now, with guidance on its use, limitations, and
interpretation. We hope funders and researchers eventually render this dataset
obsolete by producing better global, high-resolution data on extreme heat.

If data availability is not a concern and you need the most scientifically
precise representation of human heat exposure, consider
[other heat datasets](https://www.wri.org/insights/beyond-thermometer-measuring-heat?trk=feed_main-feed-card_feed-article-content).
If exact Landsat LST values matter and you can compute them for your area,
follow [this USGS tutorial](https://code.usgs.gov/eros-user-services/processing_landsat_data/scaling-landsat-collection-2-level-2-data/-/blob/main/Scaling_Landsat_C2_L2_Data_v2.ipynb?ref_type=heads).
This dataset serves people in data-scarce regions who otherwise lack the
capacity to assess extremes in relative surface temperature.

## Data processing, storage, and access

### Processing

DERIVED from collection metadata, this dataset combines Landsat 8/9
observations from 2021 through 2025 into a P95 composite of land surface
temperature. WRI's work at the [Cool Cities Lab](https://coolcities.wri.org/)
influenced our choice of metric.

DERIVED from production bounds, the product covers land from 60° S to 60° N.
Processing higher latitudes would increase costs while adding little populated
land. We use standard QA masks to exclude clouds, cloud shadow, snow, ice, and
cirrus. We also mask water.

DERIVED from current output limits, the pipeline excludes physically
implausible results, including temperatures above 80°C. The LST COG records all
excluded pixels as nodata.

Some regions contain gaps from missing ASTER coverage or persistent clouds,
primarily in the tropics. We have not filled these gaps. See
[known issues](#known-issues) for details.

DERIVED from the current output rule, a pixel needs five valid observations
across 2021 through 2025 to receive an LST value. The LST COG records pixels
below that threshold as nodata. This permissive minimum makes P95 values near
the threshold sensitive to individual observations. Consult `qa_count`, which
has one band per month, when assessing reliability in persistently cloudy
regions.

Beyond masking, we modify the data only to normalize scenes and reduce
striping. We want less technical users to see plausible spatial patterns
without large temperature jumps along Landsat swaths. We traced the striping to
scene-level atmospheric correction and unequal observation counts where swaths
overlap. We apply scene normalization and feathering to address these sources.

#### Scene normalization

LST can shift between scenes because Landsat's atmospheric correction varies.
These shifts can make a surface appear warmer or colder when its underlying
temperature pattern has not changed. They can also produce striping.

We estimate the scene-level bias from valid pixels. For each pixel, we compare
the scene with its median temperature for the same calendar month across the
full record. We take the median of these differences to get one scene offset,
then subtract it from every valid pixel. The correction changes individual LST
values but preserves relative spatial differences within the scene.

#### Feathering

Pixels where Landsat orbital paths overlap have more observations than pixels
covered by one path. Those extra observations can differ from the surrounding
areas and appear as visible stripes. We address this by calculating P95
separately for each WRS path.

Where paths overlap, we blend the results according to distance from each path
edge. Pixels near an edge receive less weight. The blend produces a gradual
transition between the path estimates. It changes values in overlap areas but
does not smooth variation within either path.

This method supports the product's aim of showing relative surface heat while
minimizing artifacts from Landsat's sampling geometry. The resulting patterns
support planning and decision-making.

### Storage and access

[Source Cooperative](https://data.source.coop/nlebovits/landsat-lst/catalog.json)
hosts the data as Cloud-Optimized GeoTIFFs in a Portolan catalog. DERIVED from
published metadata, each STAC item spans 5 degrees in EPSG:4326 and contains
both COG assets:

| Asset | Contents | Encoding | Nodata |
|---|---|---|---|
| `lst_p95` | Corrected and blended P95 LST | `uint16`, one band, with `celsius = DN * 0.01 - 50.0` | DN `0` |
| `qa_count` | Valid observations by calendar month | `uint8`, 12 bands from January through December | none |

Rasterio, PySTAC, and QGIS can open the data. The example resolves a COG URL
from its STAC item, then reads only the requested area with Rasterio.

```python
import pystac
import rasterio
from rasterio.windows import from_bounds

item = pystac.Item.from_file(
    "https://data.source.coop/nlebovits/landsat-lst/"
    "lst-p95-2021-2025/S30W065/S30W065.json"
)

with rasterio.open(item.assets["lst_p95"].get_absolute_href()) as src:
    # Select a region.
    window = from_bounds(-64, -34, -63.9, -33.9, src.transform)
    lst = src.read(1, window=window, masked=True)

    # Convert the stored integers to degrees Celsius.
    lst_celsius = lst * src.scales[0] + src.offsets[0]
```

## Known issues

### ASTER coverage gaps

The ASTER Global Emissivity Dataset (GED) contains
[documented coverage gaps](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature-data-gaps-due-missing-aster-ged)
caused by persistent cloud. These gaps affect this product.

![Global ASTER GED emissivity coverage. Blue shows available data; white shows gaps.](docs/images/aster-ged-coverage-usgs.jpg)

*Blue shows available data; white shows gaps. Map from
[USGS](https://www.usgs.gov/media/images/aster-ged-emissivity-coverage).*

MEASURED against
[GHS-SMOD R2023A urban land](https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php),
ASTER gaps affect 2.66% of urban land. MEASURED in the same analysis, another
10.23% rests on only one or two ASTER observations. Regional results:

| Region | MEASURED share of urban land with no ASTER observations |
|---|---:|
| Southeast Asia | 12.07% |
| Amazonia | 11.62% |
| Southern Africa | 8.36% |
| Europe | 2.80% |
| North America | 1.18% |
| Australia | 0.30% |
| Sahara and Sahel | 0.00% |

The product preserves nodata where the source lacks a temperature.

### Limited tropical coverage

We have also observed missing patches in the Brazilian Amazon that appear to
result from persistent cloud cover. UNKNOWN: we have not quantified their total
area. These gaps appear mainly in sparsely populated tropical regions.

## Intended use

Use these data to map relative differences in LST and identify surfaces that
raise or reduce heat. The product supports decisions about relative heat
impacts. Use other measurements when you need a precise estimate of human heat
exposure.

Read the [USGS documentation on Landsat surface temperature](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature)
and WRI's [guidance on measuring heat](https://www.wri.org/insights/beyond-thermometer-measuring-heat?trk=feed_main-feed-card_feed-article-content).
Seek technical support from experts at local universities when interpreting
and applying the data.

### Ancillary data

Interpret LST alongside data on buildings, tree canopy, nearby water, and land
cover. When good local data are unavailable, we recommend these sources:

- **Built-up volume** from the [Global Human Settlement Layer](https://human-settlement.emergency.copernicus.eu/ghs_buV2023.php).
- **Global building heights** from the [GlobalBuildingAtlas](https://source.coop/tge-labs/globalbuildingatlas-lod1).
- **Overture Maps** building footprints.
- **Tree canopy height** from [World Resources Institute and Meta](https://source.coop/tge-labs/meta-chm-v2).
- **ASTER Global Water Bodies Database** from [NASA](https://data.nasa.gov/dataset/aster-global-water-bodies-database-v001-7ff0b).

### Example notebook

The product documentation will include a Jupyter notebook that shows how we
recommend using these data. It will follow
[this example](https://nlebovits.github.io/datos-escala-humana/en/en/cookbooks/pergamino.html).
The notebook will show how to ingest the published LST and add ancillary data,
including tree canopy and building heights. It will also show how to assess the
combined information for concrete urban-planning interventions.

## Development

See [`AGENTS.md`](./AGENTS.md).

## License

The code uses the [Apache License 2.0](./LICENSE). The data use `CC0-1.0`, based
on USGS terms for Landsat Collection 2 products.

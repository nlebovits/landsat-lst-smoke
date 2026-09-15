# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "numpy", "rasterio", "geopandas", "shapely", "pyogrio",
# ]
# ///
"""Which pixels a finished tile is allowed to carry a temperature on.

`lst_qa.py` decides whether one observation of one pixel is usable. This module
decides whether the pixel is one the product should describe at all. The two
are different questions and they fail differently. A pixel that was cloudy on
every pass is a thin time axis, and a wider window fixes it. A pixel over the
sea, or one where ASTER GED holds no emissivity, is not thin. It is a pixel the
product has nothing to say about, in this window or any other.

One rule, and it is permanent for the pixel.

Water. `land_tiles.py` selects the tiles the fleet runs by intersecting the
grid with Natural Earth 10m land buffered by 25 km. That module's docstring
states the contract this module completes: one geometry answers both "which
tiles does the fleet run" and "which pixels carry a temperature". A tile chosen
from one geometry and masked with another produces tiles that are entirely
nodata, and pixels no tile ever visits.

The geometry arrives as an artifact rather than as a download.
`load_land_polygons` fetches Natural Earth when its cache is cold, and a fetch
inside a run is what `tests/test_no_stac_at_runtime.py` exists to forbid.
`land_tiles.py --write-geometry` ships it; this module reads it.

The buffered geometry decides pixels. It does not define land, and a published
property that divides by it may not say land either. MEASURED 2026-09-15 at
3600 pixels per degree, S40W065 holds 73,254,945 pixels of processing mask and
45,407,126 pixels of land, so the buffer is a third of what an item used to
call land. `land_split` reads a second, unbuffered artifact and separates the
two, and `coverage` divides by the right one. That is reporting, not masking:
no pixel's value depends on it.

Emissivity, as a region rather than as a rule. Landsat Collection 2 Level-2
Surface Temperature needs a land surface emissivity value per pixel, taken from
ASTER GED, which was built from clear-sky ASTER scenes acquired between 2000 and
2008. Where ASTER never caught clear sky, GED holds no emissivity. USGS does not
leave that pixel alone. It interpolates emissivity from the neighbouring cells
and retrieves a temperature anyway, and some of those retrievals fail upward.

`output_mask` still reports where that region reaches, because a tile with much
of it rests on interpolated emissivity and a reader should know. It no longer
removes anything. An earlier build paired the region with a 70 C threshold and
dropped the pixels where both held. MEASURED across five tiles, the two halves
do not coincide: on N30E075 all 207 pixels at or above 80 C fall outside the
region and its one-cell buffer, so the pair reached none of them. The bound that
replaced it is `lst_qa.LST_OUTPUT_MAX_C`, applied to every pixel wherever it
sits, in `composite.reduce_block`.

Masking is not a second opinion about the temperature, and the water rule does
remove values the composite held: over water Landsat retrieves a real
temperature and the rule drops it. `qa_count` follows that rule alone. Zero
observations is data, and the count layer stays the evidence behind every
surviving p95, including the pixels `lst_qa.supported_output` screens out.
"""

from __future__ import annotations

from pathlib import Path

from aster_ged import (
    COVERAGE_BAND,
    NUMOBS_BAND,
    GedError,
    cell_window_for_bbox,
    cells_to_pixels,
    dilate_cells,
)
from lst_qa import LST_NODATA_DN

DEFAULT_LAND_GEOMETRY_URI = Path("artifacts/land_buffered.gpkg")

#: The same Natural Earth 10m land with no buffer, which is what the word
#: "land" means in a published property. The buffered geometry decides which
#: pixels a run may describe, and it reaches 25 km out to sea so a coastal
#: scene is not cut at the waterline. Dividing by it and calling the quotient a
#: share of land overstates the denominator: MEASURED 2026-09-15 at 3600
#: pixels per degree, S40W065 is 73,254,945 pixels of processing mask and
#: 45,407,126 pixels of land.
#:
#: Written by `land_tiles.py --write-strict-geometry`. Absent without error: a
#: run that has only the buffered artifact reports the counts it can.
STRICT_LAND_GEOMETRY_URI = Path("artifacts/land_strict.gpkg")

#: How many raster rows `count_valid_within` reads at once. Matches
#: `composite.STAGING_BLOCK`, which is the strip height every other scan of a
#: finished tile uses.
COUNT_BLOCK_ROWS = 512

#: A GED cell with this count holds no clear-sky ASTER observation, so USGS
#: interpolated its emissivity. Zero alone: the tiers at one and two
#: observations carry a real observation, and dropping them would remove 6.9%
#: of a measured tile rather than 0.22%.
GAP_NUMOBS = 0

#: How far the reported gap region grows, in GED cells of about 1 km,
#: 8-connected.
#:
#: The failed retrievals sit on the fringe of a gap rather than in its middle.
#: The middle comes back as `ST_B10` fill and never reaches the composite. One
#: cell of growth took the tail the withdrawn pair rule removed from 77.30% to
#: 91.52% on S30W065, and the buffer is kept at that width so the region the
#: counts report is the region the measurements describe.
GAP_BUFFER_CELLS = 1


class MaskError(RuntimeError):
    """The mask cannot be built. Raised before any compute."""


def geometry_checksum(path: Path | str) -> str:
    """SHA-256 of the shipped land geometry.

    `land_tiles.land_geometry_checksum` digests the same bytes out of the cache
    directory. This one digests the artifact, because that is what a run
    actually rasterises and what the ASTER GED manifest has to agree with.
    """
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The tile grid.
# --------------------------------------------------------------------------


def raster_shape(bbox, pixels_per_degree: int) -> tuple[int, int]:
    """`(height, width)` of the tile raster, as `composite.raster_shape` does."""
    west, south, east, north = bbox
    return (
        int(round((north - south) * pixels_per_degree)),
        int(round((east - west) * pixels_per_degree)),
    )


def transform_for(bbox, pixels_per_degree: int):
    """The north-up affine of a tile at `pixels_per_degree`."""
    from rasterio.transform import from_origin

    west, _, _, north = bbox
    res = 1.0 / pixels_per_degree
    return from_origin(west, north, res, res)


# --------------------------------------------------------------------------
# The two rules.
# --------------------------------------------------------------------------


def land_mask(bbox, pixels_per_degree: int, land_geometry_uri=None):
    """True where the land geometry covers the pixel.

    Defaults to the buffered geometry, which is the processing mask. Pass
    `STRICT_LAND_GEOMETRY_URI` for land itself. One function serves both so the
    two masks cannot differ by anything except the polygons they read.

    A pixel counts as land when its centre falls inside the geometry, which is
    `rasterize`'s default. `all_touched=True` would widen every coastline by
    one pixel on all sides, and the 25 km buffer is already the decision about
    how much coast to keep. MEASURED 2026-09-15 on S40W065, `all_touched` moves
    the strict count by 28,378 pixels of 45,407,126.

    Only the polygons that meet the bbox are read. pyogrio pushes the filter
    into the GeoPackage, so a 5-degree tile does not pay for a global geometry.

    Raises:
        MaskError: if the artifact is absent, naming the command that writes it.
    """
    import geopandas as gpd
    import numpy as np
    from rasterio.features import rasterize

    path = Path(land_geometry_uri or DEFAULT_LAND_GEOMETRY_URI)
    if not path.exists():
        strict = path == Path(STRICT_LAND_GEOMETRY_URI)
        which = "strict" if strict else "buffered"
        flag = "--write-strict-geometry" if strict else "--write-geometry"
        msg = (
            f"no {which} land geometry at {path}. Write it with:\n"
            f"  uv run land_tiles.py --out artifacts/land_tiles.parquet "
            f"{flag} {path}\n"
            f"The pixel mask reads an artifact rather than fetching Natural "
            f"Earth, so that a run needs no network."
        )
        raise MaskError(msg)

    height, width = raster_shape(bbox, pixels_per_degree)
    land = gpd.read_file(path, bbox=tuple(bbox))
    if land.empty:
        return np.zeros((height, width), dtype=bool)

    burned = rasterize(
        ((geom, 1) for geom in land.geometry if geom is not None and not geom.is_empty),
        out_shape=(height, width),
        transform=transform_for(bbox, pixels_per_degree),
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)


def land_split(
    bbox,
    pixels_per_degree: int,
    *,
    land_geometry_uri=None,
    strict_land_geometry_uri=None,
    processing=None,
    gap=None,
):
    """Land, and the coast the processing mask adds to it.

    The processing mask is the rule a run masks on and it is not land. It
    reaches 25 km out to sea, so a published property that divides by it and
    calls the quotient a share of land answers a different question from the
    one it states. This separates the two. Neither count is a share of the
    other: they sum to the processing mask.

    Args:
        bbox: the tile's bounds.
        pixels_per_degree: the tile's grid.
        land_geometry_uri: the buffered geometry. Ignored when `processing` is
            given.
        strict_land_geometry_uri: the unbuffered geometry. Defaults to
            `STRICT_LAND_GEOMETRY_URI`.
        processing: the processing mask, when the caller already holds it from
            `output_mask`. Saves rasterising the buffered geometry twice.
        gap: the grown ASTER GED gap region. When given, the counts gain the
            gap's own strict-land numerator, because the published fraction
            divides by land and `pixels_emissivity_gap_on_land` does not.

    Returns:
        `(strict, processing, counts)`. `strict` is True on land. `processing`
        is True where a run may write a temperature. `counts` names both and
        their difference, so a caller reports one set of numbers rather than
        recounting the arrays.

    Raises:
        MaskError: if either artifact is absent, or if `strict` reaches outside
            `processing`. That containment is a property of a buffer, so losing
            it means the two files hold different Natural Earth releases, and
            every count built from them would compare two worlds. MEASURED
            2026-09-15 over eight tiles, `strict AND NOT processing` is empty
            on all of them.
    """
    if processing is None:
        processing = land_mask(bbox, pixels_per_degree, land_geometry_uri)
    strict = land_mask(
        bbox, pixels_per_degree, strict_land_geometry_uri or STRICT_LAND_GEOMETRY_URI
    )
    outside = int((strict & ~processing).sum())
    if outside:
        msg = (
            f"{outside} land pixel(s) fall outside the processing mask. A "
            f"buffer contains what it buffers, so the two geometries hold "
            f"different Natural Earth releases. Rebuild both from one "
            f"land_tiles.py run."
        )
        raise MaskError(msg)
    land, mask_total = int(strict.sum()), int(processing.sum())
    counts = {
        "pixels_strict_land": land,
        "pixels_coastal_buffer": mask_total - land,
        "pixels_processing_mask": mask_total,
    }
    if gap is not None:
        counts["pixels_emissivity_gap_on_strict_land"] = int((strict & gap).sum())
    return strict, processing, counts


def count_valid_within(path, nodata, **regions) -> dict[str, int]:
    """Non-nodata pixels of band 1, counted inside each named region.

    The one implementation of "how many values does this raster carry inside
    this mask". `composite.file_statistics` answers the whole-raster question
    and keeps it, because a masked variant of it would be a second reading of
    the same file under the same name.

    Every region is counted in one pass, so a caller that wants land and coast
    reads the file once. That matters off `/vsis3`: a recount of the five
    published tiles moves 33 MB to 471 MB per tile, and twice that for asking
    twice.

    Read in strips, so the largest thing held is `COUNT_BLOCK_ROWS` rows rather
    than a 324 million pixel band.

    Args:
        path: a local file or a `/vsis3` path. A publish recounts a raster it
            never downloads.
        nodata: the value that means no data. `None` counts every pixel.
        **regions: name to boolean mask, each the shape of the raster. `total`
            is reserved.

    Returns:
        One count per region, under the name it was passed, plus `total`: every
        non-nodata pixel in the band, whatever region it sits in. The regions
        of a tile that carries a water mask sum to `total`, and a caller checks
        that rather than assuming it.

    Raises:
        MaskError: if a region is not the shape of the raster. A recount that
            rasterised its mask on a grid the COG does not use would return a
            plausible number for the wrong ground.
        ValueError: if a region is called `total`.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    if "total" in regions:
        msg = "`total` is the whole-band count and cannot name a region"
        raise ValueError(msg)
    named = {name: np.asarray(mask) for name, mask in regions.items()}
    totals = dict.fromkeys(named, 0)
    totals["total"] = 0
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        shape = (src.height, src.width)
        wrong = {name: m.shape for name, m in named.items() if m.shape != shape}
        if wrong:
            msg = (
                f"{path} is {shape} and {wrong} was rasterised on a different "
                f"grid from the raster it is counting."
            )
            raise MaskError(msg)
        for row in range(0, src.height, COUNT_BLOCK_ROWS):
            height = min(COUNT_BLOCK_ROWS, src.height - row)
            window = Window.from_slices((row, row + height), (0, src.width))
            strip = src.read(1, window=window)
            present = None if nodata is None else strip != nodata
            totals["total"] += int(strip.size if present is None else present.sum())
            for name, mask in named.items():
                keep = mask[row : row + height]
                totals[name] += int(
                    keep.sum() if present is None else (present & keep).sum()
                )
    return totals


def _gap_and_seen(bbox, pixels_per_degree: int, numobs_uri, buffer_cells: int):
    """The grown gap region and the read region, both on the tile's pixels.

    Read at the GED grid's own resolution so the dilation happens in cells,
    then stretched to pixels. Stretching first and dilating in pixel space
    would grow the region by one 30 m pixel instead of one 1 km cell.
    """
    shape = raster_shape(bbox, pixels_per_degree)
    pad = max(int(buffer_cells), 0)
    counts, covered = cell_window_for_bbox(
        numobs_uri,
        bbox,
        pad_cells=pad,
        bands=(NUMOBS_BAND, COVERAGE_BAND),
    )
    seen = covered > 0
    gap = dilate_cells(seen & (counts == GAP_NUMOBS), pad)
    if pad:
        gap = gap[pad:-pad, pad:-pad]
        seen = seen[pad:-pad, pad:-pad]
    return cells_to_pixels(gap, shape), cells_to_pixels(seen, shape)


def emissivity_gap(
    bbox,
    pixels_per_degree: int,
    numobs_uri,
    *,
    buffer_cells: int = GAP_BUFFER_CELLS,
):
    """True where the pixel sits in the ASTER GED gap region.

    A count of zero is not enough on its own. The mosaic writes zero for a cell
    it read and found empty, and zero for a cell it never read, and only the
    coverage band separates the two. So a cell is a gap when the build read it
    AND its count is zero.

    An unread cell keeps its pixels. A gap the artifact cannot demonstrate is
    not a gap, and treating one as such removed 34 whole tiles from the first
    real fleet plan, every one of them for want of a downloaded granule.

    This is a region, not a mask, and nothing removes a pixel for being inside
    it. On its own it would remove 701,839 valid pixels of S30W065, and 87% of
    the cells it covers carry nothing wrong. `output_mask` reports its size so a
    reader can see how much of a tile rests on interpolated emissivity.

    Raises:
        GedError: if the NumObs artifact is absent or does not cover the bbox.
    """
    return _gap_and_seen(bbox, pixels_per_degree, numobs_uri, buffer_cells)[0]


def output_mask(
    bbox,
    pixels_per_degree: int,
    *,
    numobs_uri,
    land_geometry_uri=None,
    buffer_cells: int = GAP_BUFFER_CELLS,
):
    """What the tile may describe, and where the emissivity gap reaches.

    The two rules resolve at different times, so they come back apart. The
    water rule is settled by the tile's bbox alone, which lets a run build it
    before it stages a single object. The emissivity rule needs the assembled
    temperatures, so only its region is known here.

    The strict land mask is not here, because it is not a rule. It decides no
    pixel's value and only supplies a denominator a published property divides
    by. `land_split` owns it, and takes `processing` so it does not rasterise
    the buffered geometry a second time.

    Returns:
        `(keep, gap, counts)`. `keep` is True where the water rule allows a
        temperature. `gap` is True inside the grown ASTER GED gap region.
        `counts` describes what each rule reaches. The counts come back with
        the arrays rather than from a second pass, so a run's summary and a
        test read one set of numbers.

    Raises:
        MaskError: if the land geometry is absent.
        GedError: if the NumObs artifact is absent or does not cover the bbox.
    """
    land = land_mask(bbox, pixels_per_degree, land_geometry_uri)
    gap, seen = _gap_and_seen(bbox, pixels_per_degree, numobs_uri, buffer_cells)
    counts = {
        "pixels_total": int(land.size),
        "pixels_water": int((~land).sum()),
        # The region, which the mask no longer removes. It is reported because
        # a tile with much of it rests on interpolated emissivity.
        "pixels_emissivity_gap": int(gap.sum()),
        # Gap over land, which is the share this region adds on its own. The
        # two rules overlap over sea, where GED has no observation either, so
        # `pixels_water + pixels_emissivity_gap` double-counts.
        "pixels_emissivity_gap_on_land": int((land & gap).sum()),
        # Land the artifact says nothing about, because this build read no
        # granule for its cell. Those pixels are kept, and a tile with many of
        # them is a tile whose mask rests on an incomplete download.
        "pixels_land_unread": int((land & ~seen).sum()),
        "pixels_kept": int(land.sum()),
        "gap_buffer_cells": int(buffer_cells),
    }
    return land, gap, counts


def apply_output_mask(lst, qa, keep, *, scope="tile") -> dict:
    """Write the water rule into an assembled tile, in place.

    `lst` becomes `LST_NODATA_DN` and `qa` becomes 0 outside `keep`. Both, not
    one: a `qa_count` above zero beside a nodata temperature says the pixel had
    observations and lost them to the reduction, which is not what happened
    here.

    This is the only rule left that depends on where the pixel is. The rules
    that depend on what the composite says are `lst_qa.supported_output`, and
    `composite.reduce_block` has already applied them by the time a tile
    reaches this function.

    The graph applies this rule lazily, block by block, in
    `composite.finalize_block`. This is the eager statement of it, for an array
    already in memory.

    Args:
        lst: `(height, width)` uint16 of encoded temperature.
        qa: `(12, height, width)` uint8 of monthly observation counts.
        keep: `(height, width)` boolean from `output_mask`.
        scope: what the returned counts describe. `"tile"` for a whole tile,
            which is what a run composites; anything narrower has to say so.

    Returns:
        What the mask cost. Every count here is scoped to the pixels passed
        in, unlike the tile-wide counts from `output_mask`, and `scope` is
        what says so. `valid_removed_by_mask` counts pixels that held a
        temperature before and nodata after. It cannot be recovered from the
        masks alone.
    """
    import numpy as np

    keep = np.asarray(keep)
    water = ~keep

    # The mask is read before it is written, so the count describes the tile
    # that arrived rather than the one that leaves.
    valid = lst != LST_NODATA_DN
    removed_water = int((valid & water).sum())
    qa_removed = int(np.any(qa, axis=0)[water].sum())

    lst[water] = LST_NODATA_DN
    qa[:, water] = 0
    return {
        "scope": scope,
        "valid_removed_by_water": removed_water,
        "valid_removed_by_mask": removed_water,
        "qa_count_pixels_zeroed": qa_removed,
    }


def coverage(mask_counts, lst_statistics, split=None, valid=None) -> dict | None:
    """How much of the tile's land carries a temperature. None with no mask.

    Every number here is already counted. `output_mask` returns what each rule
    reaches, `land_split` separates land from the coast the processing mask
    adds, and `count_valid_within` says how many values sit in each. This costs
    the divisions and puts the result on the published item. Without it a reader
    fetches a 350 to 640 MB `qa_count` to learn that a tile is half empty.

    `land_pixels` is land, not the processing mask. The two differ by the 25 km
    coastal buffer, which reaches open sea: MEASURED 2026-09-15 at 3600 pixels
    per degree, S40W065 is 73,254,945 pixels of processing mask and 45,407,126
    pixels of land, so a fraction of the first understates a fraction of the
    second by a third. `coastal_buffer_pixels` names the difference and
    `processing_mask_pixels` preserves the total under a name that cannot be
    read as land.

    `empty_land_pixels` is land the five-year window returned no usable
    observation for. It counts two conditions that no published raster
    separates: land Landsat never photographed, and land it photographed where
    every observation failed the QA and range rules. `qa_count` reads zero for
    both, so a split of this number needs a source-presence record the
    composite does not yet keep.

    Args:
        mask_counts: the counts from `output_mask`.
        lst_statistics: the per-band statistics from `composite.file_statistics`.
        split: the counts from `land_split`, with its `gap` argument supplied.
            Absent on a machine that has only the buffered geometry.
        valid: the counts from `count_valid_within`, under the names `land` and
            `coast`. Required with `split` and refused without it.

    Returns:
        The coverage block, or None when the run applied no output mask and so
        has no land to divide by. Without `split` the block holds the two counts
        that need no land geometry and no fraction at all, because naming a
        share of land without land is the defect this function exists to end.
        A tile with no land at all holds the counts and neither fraction, for the
        same reason: the buffer alone can put a cell of open sea in the mask.

    Raises:
        ValueError: if `split` and `valid` do not arrive together, if more
            pixels carry a value than the mask allows, or if the two regions'
            valid counts do not sum to the raster's own.
    """
    if not mask_counts:
        return None
    mask_total = int(mask_counts.get("pixels_kept") or 0)
    if not mask_total:
        return None
    raster_valid = int(lst_statistics[0]["kept"])
    if (split is None) != (valid is None):
        msg = "coverage needs `split` and `valid` together, or neither"
        raise ValueError(msg)
    if split is None:
        return {
            "processing_mask_pixels": mask_total,
            "valid_pixels": raster_valid,
        }

    land = int(split["pixels_strict_land"])
    on_land, on_coast = int(valid["land"]), int(valid["coast"])
    if on_land > land:
        msg = f"{on_land} valid pixels on {land} land pixels"
        raise ValueError(msg)
    if on_land + on_coast != raster_valid:
        msg = (
            f"{on_land} valid on land and {on_coast} in the coastal buffer sum "
            f"to {on_land + on_coast}, and the raster carries {raster_valid}. "
            f"The masks and the raster describe different ground."
        )
        raise ValueError(msg)
    block = {
        "land_pixels": land,
        "valid_pixels": on_land,
        "empty_land_pixels": land - on_land,
        "coastal_buffer_pixels": int(split["pixels_coastal_buffer"]),
        "coastal_buffer_valid_pixels": on_coast,
        "processing_mask_pixels": int(split["pixels_processing_mask"]),
    }
    # A tile can hold no land and still hold pixels. The buffer reaches 25 km
    # out, so a cell of open sea near a coast enters the processing mask on its
    # own. MEASURED 2026-09-15, the published `S35W055` is 588,696 pixels of
    # processing mask over the Atlantic and 0 pixels of land, and it reported
    # all 588,696 as land. There is no share of land to report there, and 0/0 is
    # not zero, so both fractions are absent rather than invented.
    if land:
        block["valid_fraction"] = on_land / land
        block["ged_gap_fraction"] = (
            int(split["pixels_emissivity_gap_on_strict_land"]) / land
        )
    return block


__all__ = [
    "COUNT_BLOCK_ROWS",
    "DEFAULT_LAND_GEOMETRY_URI",
    "GAP_BUFFER_CELLS",
    "GAP_NUMOBS",
    "STRICT_LAND_GEOMETRY_URI",
    "GedError",
    "MaskError",
    "apply_output_mask",
    "count_valid_within",
    "coverage",
    "emissivity_gap",
    "geometry_checksum",
    "land_mask",
    "land_split",
    "output_mask",
    "raster_shape",
    "transform_for",
]

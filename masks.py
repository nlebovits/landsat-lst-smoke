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

Two rules, and each one is permanent for the pixel.

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

Emissivity. Landsat Collection 2 Level-2 Surface Temperature needs a land
surface emissivity value per pixel, taken from ASTER GED, which was built from
clear-sky ASTER scenes acquired between 2000 and 2008. Where ASTER never caught
clear sky, GED holds no emissivity. USGS does not leave that pixel alone. It
interpolates emissivity from the neighbouring cells and retrieves a temperature
anyway, and some of those retrievals fail upward.

So the gap region and the damage are different sets, and the rule is the pair
rather than the geometry. MEASURED on S30W065: the tile holds 605 gap cells,
524 of them carry no pixel at or above 70 C, and in the 81 that do the hot
pixels are 4.77% of the cell. Removing the geometry alone costs 701,839 valid
pixels to remove 4,588 bad ones. Removing a pixel only where the gap region and
70 C coincide costs 5,432 and reaches more of the tail.

`nlebovits/landsat-lst` shipped the geometry alone, measured 2,799,286 pixels
removed for 2,582 artifacts, and replaced it with the pair. This module
reproduces that rule rather than the version it replaced.

Masking is not a second opinion about the temperature, and both rules do remove
values the composite held. Over water Landsat retrieves a real temperature and
the water rule drops it. Over a gap the pair drops a retrieval that reads 70 C
or hotter. `qa_count` follows the water rule alone: zero observations is data,
and the count layer stays the evidence behind every surviving p95.
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
from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE

DEFAULT_LAND_GEOMETRY_URI = Path("artifacts/land_buffered.gpkg")

#: A GED cell with this count holds no clear-sky ASTER observation, so USGS
#: interpolated its emissivity. Zero alone: the tiers at one and two
#: observations carry a real observation, and dropping them would remove 6.9%
#: of a measured tile rather than 0.22%.
GAP_NUMOBS = 0

#: How far the gap region grows, in GED cells of about 1 km, 8-connected.
#:
#: The failures sit on the fringe of a gap rather than in its middle. The
#: middle comes back as `ST_B10` fill and never reaches the composite. One cell
#: of growth takes the tail this rule removes from 77.30% to 91.52% on
#: S30W065. Under the temperature test it costs 844 further pixels, because a
#: grown cell can only remove a pixel that already reads 70 C.
GAP_BUFFER_CELLS = 1

#: Celsius at which a pixel inside the gap region reads as a failed retrieval.
#:
#: Empirical, from one tile, with no published source. It is half of a pair and
#: never acts alone, so it makes no claim about the hottest land surface. A
#: pixel above it outside the gap region survives, and 503 such pixels do
#: survive on S30W065. `nlebovits/landsat-lst` calibrated the same number the
#: same way, in `config.py:193-210`.
GAP_HOT_THRESHOLD_C = 70.0

#: The lowest threshold this module accepts. Below it the pair stops being a
#: screen for failed retrievals and starts deleting ordinary hot ground, which
#: is what the conjunction exists to prevent.
MIN_GAP_HOT_THRESHOLD_C = 50.0


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
    """`(height, width)` of the tile raster, as `plan_shards` computes it."""
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
    """True where the buffered land geometry covers the pixel.

    A pixel counts as land when its centre falls inside the geometry, which is
    `rasterize`'s default. `all_touched=True` would widen every coastline by
    one pixel on all sides, and the 25 km buffer is already the decision about
    how much coast to keep.

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
        msg = (
            f"no buffered land geometry at {path}. Write it with:\n"
            f"  uv run land_tiles.py --out artifacts/land_tiles.parquet "
            f"--write-geometry {path}\n"
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


def gap_hot_dn(celsius: float = GAP_HOT_THRESHOLD_C) -> int:
    """The encoded value a gap pixel has to reach before the mask drops it.

    The test runs against the stored uint16 rather than Celsius. Converting an
    18,000 px tile to float64 to compare it would build a 2.6 GiB array beside
    two that are already live, and the comparison is monotone either way.

    Raises:
        MaskError: if the threshold is below `MIN_GAP_HOT_THRESHOLD_C`.
    """
    if celsius < MIN_GAP_HOT_THRESHOLD_C:
        msg = (
            f"gap hot threshold {celsius} C is below "
            f"{MIN_GAP_HOT_THRESHOLD_C} C. The threshold is half of a pair, "
            f"and this low it stops screening failed retrievals and starts "
            f"deleting ordinary hot ground inside every gap cell."
        )
        raise MaskError(msg)
    return int(round((celsius - LST_OFFSET) / LST_SCALE))


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

    This is the region, not the mask. On its own it removes 701,839 valid
    pixels of S30W065 and 87% of the cells it removes carry nothing wrong.
    `apply_output_mask` intersects it with `GAP_HOT_THRESHOLD_C`, and that
    pair is what a published tile carries.

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
        # The region, which is not what the mask removes. The pair removes the
        # part of it that also reads `GAP_HOT_THRESHOLD_C` or hotter, and
        # `apply_output_mask` reports that number.
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
        "gap_hot_threshold_c": GAP_HOT_THRESHOLD_C,
    }
    return land, gap, counts


def apply_output_mask(lst, qa, keep, gap=None, *, hot_dn=None, scope="tile") -> dict:
    """Write both rules into an assembled tile, in place.

    Water. `lst` becomes `LST_NODATA_DN` and `qa` becomes 0 outside `keep`.
    Both, not one: a `qa_count` above zero beside a nodata temperature says the
    pixel had observations and lost them to the reduction, which is not what
    happened here.

    Emissivity. A pixel inside `gap` that reads `hot_dn` or hotter becomes
    `LST_NODATA_DN`, and its `qa_count` is left alone. The count records how
    many clear observations the pixel had, which stays true whatever the
    retrieval did with them, and it is the evidence a consumer needs to see
    that this pixel was screened rather than never seen.

    Passing no `gap` applies the water rule alone, which is what
    `--no-output-mask` and the merge path want.

    Args:
        lst: `(height, width)` uint16 of encoded temperature.
        qa: `(12, height, width)` uint8 of monthly observation counts.
        keep: `(height, width)` boolean from `output_mask`.
        gap: `(height, width)` boolean from `output_mask`, or None.
        hot_dn: encoded threshold, default `gap_hot_dn()`.
        scope: what the returned counts describe. `"tile"` for a whole-tile
            run, or the `--shard-slice` string for one machine's slice.

    Returns:
        What the mask cost. Every count here is scoped to the pixels this
        process assembled, unlike the tile-wide counts from `output_mask`, and
        `scope` is what says so. `valid_removed_by_mask` counts pixels that
        held a temperature before and nodata after. It cannot be recovered
        from the masks alone.
    """
    import numpy as np

    keep = np.asarray(keep)
    water = ~keep
    if gap is None:
        hot = np.zeros_like(water)
    else:
        if hot_dn is None:
            hot_dn = gap_hot_dn()
        hot = np.asarray(gap) & (lst >= hot_dn)

    # Both masks are read before either is written, so a pixel that is water
    # and hot is counted once, against the water rule.
    valid = lst != LST_NODATA_DN
    removed_water = int((valid & water).sum())
    removed_hot = int((valid & hot & keep).sum())
    qa_removed = int(np.any(qa, axis=0)[water].sum())

    lst[water] = LST_NODATA_DN
    lst[hot] = LST_NODATA_DN
    qa[:, water] = 0
    return {
        "scope": scope,
        "valid_removed_by_water": removed_water,
        "valid_removed_by_emissivity": removed_hot,
        "valid_removed_by_mask": removed_water + removed_hot,
        "qa_count_pixels_zeroed": qa_removed,
    }


__all__ = [
    "DEFAULT_LAND_GEOMETRY_URI",
    "GAP_BUFFER_CELLS",
    "GAP_HOT_THRESHOLD_C",
    "GAP_NUMOBS",
    "MIN_GAP_HOT_THRESHOLD_C",
    "GedError",
    "MaskError",
    "apply_output_mask",
    "emissivity_gap",
    "gap_hot_dn",
    "geometry_checksum",
    "land_mask",
    "output_mask",
    "raster_shape",
    "transform_for",
]

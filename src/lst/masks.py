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

from lst.aster_ged import (
    COVERAGE_BAND,
    NUMOBS_BAND,
    GedError,
    cell_window_for_bbox,
    cells_to_pixels,
    dilate_cells,
)
from lst.lst_qa import LST_NODATA_DN

DEFAULT_LAND_GEOMETRY_URI = Path("artifacts/land_buffered.gpkg")

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
            f"  uv run lst-land-tiles --out artifacts/land_tiles.parquet "
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


__all__ = [
    "DEFAULT_LAND_GEOMETRY_URI",
    "GAP_BUFFER_CELLS",
    "GAP_NUMOBS",
    "GedError",
    "MaskError",
    "apply_output_mask",
    "emissivity_gap",
    "geometry_checksum",
    "land_mask",
    "output_mask",
    "raster_shape",
    "transform_for",
]

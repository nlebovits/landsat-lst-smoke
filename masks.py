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
clear sky, GED holds no emissivity and USGS produces no surface temperature.
`aster_ged.py` ships the observation count that names those cells exactly, and
a count of zero is the gap.

Masking is not a second opinion about the temperature. Every pixel removed here
was already nodata or already outside the product's subject. Over an ASTER gap
USGS writes `ST_B10` fill, `lst_qa.not_fill` rejects it, and `qa_count` is
already 0 there before this module runs. Over water the composite does hold
values, and those are the ones this removes.
"""

from __future__ import annotations

from pathlib import Path

from aster_ged import COVERAGE_BAND, NUMOBS_BAND, GedError, window_for_bbox
from lst_qa import LST_NODATA_DN

DEFAULT_LAND_GEOMETRY_URI = Path("artifacts/land_buffered.gpkg")

#: A GED cell with this count holds no emissivity, so no pixel inside it can
#: hold a surface temperature. Zero alone: the tiers at one and two
#: observations do carry emissivity, and dropping them would remove 6.9% of a
#: measured tile rather than 0.22%.
GAP_NUMOBS = 0


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


def emissivity_gap(bbox, pixels_per_degree: int, numobs_uri):
    """True where ASTER GED holds no clear-sky observation for the pixel.

    A count of zero is not enough on its own. The mosaic writes zero for a cell
    it read and found empty, and zero for a cell it never read, and only the
    coverage band separates the two. So a pixel is a gap when the build read
    its cell AND that cell's count is zero.

    An unread cell keeps its pixels. A gap the artifact cannot demonstrate is
    not a gap, and treating one as such removed 34 whole tiles from the first
    real fleet plan, every one of them for want of a downloaded granule.

    Raises:
        GedError: if the NumObs artifact is absent or does not cover the bbox.
    """
    height, width = raster_shape(bbox, pixels_per_degree)
    counts, covered = window_for_bbox(
        numobs_uri, bbox, (height, width), (NUMOBS_BAND, COVERAGE_BAND)
    )
    return (covered > 0) & (counts == GAP_NUMOBS)


def output_mask(bbox, pixels_per_degree: int, *, numobs_uri, land_geometry_uri=None):
    """The pixels a finished tile may carry a temperature on, and the counts.

    Returns:
        A boolean array, True where the pixel survives both rules, and a dict
        of counts describing what each rule removed. The counts come back with
        the mask rather than from a second pass, so a run's summary and a test
        read one set of numbers.

    Raises:
        MaskError: if the land geometry is absent.
        GedError: if the NumObs artifact is absent or does not cover the bbox.
    """
    height, width = raster_shape(bbox, pixels_per_degree)
    land = land_mask(bbox, pixels_per_degree, land_geometry_uri)
    counts_band, covered = window_for_bbox(
        numobs_uri, bbox, (height, width), (NUMOBS_BAND, COVERAGE_BAND)
    )
    seen = covered > 0
    gap = seen & (counts_band == GAP_NUMOBS)
    keep = land & ~gap
    counts = {
        "pixels_total": int(land.size),
        "pixels_water": int((~land).sum()),
        "pixels_emissivity_gap": int(gap.sum()),
        # Gap over land, which is the share this rule adds on its own. The two
        # rules overlap over sea, where GED has no observation either, so
        # `pixels_water + pixels_emissivity_gap` double-counts and is not the
        # number of pixels removed.
        "pixels_emissivity_gap_on_land": int((land & gap).sum()),
        # Land the artifact says nothing about, because this build read no
        # granule for its cell. Those pixels are kept, and a tile with many of
        # them is a tile whose mask rests on an incomplete download.
        "pixels_land_unread": int((land & ~seen).sum()),
        "pixels_kept": int(keep.sum()),
    }
    return keep, counts


def apply_output_mask(lst, qa, keep) -> dict:
    """Write the mask into an assembled tile, in place.

    `lst` becomes `LST_NODATA_DN` and `qa` becomes 0 outside `keep`. Both, not
    one: a `qa_count` above zero beside a nodata temperature says the pixel had
    observations and lost them to the reduction, which is not what happened
    here.

    Args:
        lst: `(height, width)` uint16 of encoded temperature.
        qa: `(12, height, width)` uint8 of monthly observation counts.
        keep: `(height, width)` boolean from `output_mask`.

    Returns:
        What the mask cost, for the run summary. `valid_removed_by_mask` counts
        pixels that held a temperature before and nodata after. It is the number
        a reviewer checks and it cannot be recovered from the mask alone.
    """
    import numpy as np

    drop = ~np.asarray(keep)
    removed = int(((lst != LST_NODATA_DN) & drop).sum())
    qa_removed = int((qa.sum(axis=0, dtype="int64") > 0)[drop].sum())
    lst[drop] = LST_NODATA_DN
    qa[:, drop] = 0
    return {
        "valid_removed_by_mask": removed,
        "qa_count_pixels_zeroed": qa_removed,
    }


__all__ = [
    "DEFAULT_LAND_GEOMETRY_URI",
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

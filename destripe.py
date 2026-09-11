# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy", "rasterio", "shapely",
# ]
# ///
"""The WRS seam, and the two corrections that remove it.

A composite pixel draws on whichever scenes overlap it. Two things make that
set change abruptly, and both leave a diagonal seam tracing satellite geometry
rather than ground.

**Per-scene bias.** Landsat Collection 2 Level 2 surface temperature is
atmospherically corrected one scene at a time, and the correction rests on
column water vapour estimates carrying a published error of 1 to 5 K. The error
applies to the whole scene at once. `nlebovits/landsat-lst` ADR-005 established
that pooling more years does not cancel it: one year, three and five all
stripe, because the bias is common to a whole scene. The fix is to compare each
scene against what that pixel normally does *in that calendar month*, pooled
across every year in the window, and subtract the scene's bulk deviation. An
annual reference was tried first and absorbed the seasonal cycle itself,
cooling the composite from 40.6 C to 29.8 C at a spatial correlation of 0.44.
The month is what makes the reference safe to subtract.

**Residual tail difference between paths.** The offset is fitted at the median,
and the product is a P95. On S30W065 one WRS path still runs 2.2 to 4.8 C
warmer in the upper tail on identical ground at matched observation counts, so
the pooled percentile steps where that path's coverage stops. Fifteen sampling
and weighting arms recovered at most 44% of that step. Building one P95 per
path and cross-fading them on geometry removed 95.8% of it and retained 97.7%
of spatial variance.

The two corrections are independent. De-striping shifts a scene by one scalar,
so it cannot change spatial structure at all. Feathering never touches a pixel
one path reaches.

This module holds the rules. It reads nothing and writes nothing. `tile_prep`
calls the estimation half once per tile; `shard_lst_p95.process_shard` and
`profile_lst_p95.build_graph` call the application half, so the two P95 paths
cannot drift. Same arrangement as `lst_qa`.
"""

from __future__ import annotations

from dataclasses import dataclass

from lst_qa import LST_SCALE, LST_VALID_MAX_C, LST_VALID_MIN_C

# --------------------------------------------------------------------------
# De-striping constants. Calibrated in `nlebovits/landsat-lst` ADR-007 against
# Pergamino 2021-2025, 390 solar-day scenes. Every one is a screen set on one
# mid-latitude agricultural AOI, so a humid tropical tile is owed its own
# calibration before a global build.
# --------------------------------------------------------------------------

#: Discard a scene whose absolute offset exceeds this. Measured: the offset
#: distribution is not a bell curve. It is a tight core holding 82.7% of scenes
#: at a standard deviation of 5.71 C, plus a one-sided cold tail. 63 scenes fall
#: below -15 C and exactly one rises above it, at +15.55, and that asymmetry is
#: the signature of undetected cloud rather than of correction bias. The cap
#: discards 21.8% of scenes and sits at about 2.6 core sigma.
DESTRIPE_MAX_OFFSET_C = 15.0

#: Sparse floor when the offset is estimated on the output grid, in pixels.
DESTRIPE_MIN_SCENE_PIXELS = 500

#: Sparse floor when the offset is estimated on the prep grid, in prep pixels.
#: It replaces the native floor rather than scaling into it. A coarse valid
#: count cannot be converted back: a single valid native pixel read through a
#: nodata-ignoring average reports as a whole coarse pixel.
#:
#: **This number is carried over, not calibrated here, and the two grids are
#: not the same one.** `nlebovits/landsat-lst` set it on a factor-2 grid over a
#: 5 degree tile. `tile_prep` estimates on a factor-4 grid over the tile plus a
#: 1 degree margin, which holds roughly a fifth as many pixels per scene, so 200
#: screens a different thing here than it did there. It is a placeholder with a
#: citation that does not apply to it. `scripts` owes a sweep of the rejected
#: share against this floor on a real tile, the way the cap was swept.
DESTRIPE_MIN_PREP_SAMPLES = 200

#: Width of one anomaly histogram bin, in Celsius. This is the output encoding
#: step (`lst_qa.LST_SCALE`), so a median read off the histogram is exact to the
#: quantisation the product already carries.
ANOMALY_BIN_C = LST_SCALE

#: The anomaly range, which is exact rather than chosen. Every value reaching
#: the histogram has passed `lst_qa.in_trusted_range`, so it lies in
#: [-50, 80] C and a difference of two such values lies in [-130, 130]. No
#: overflow bin is needed and no value can fall outside.
ANOMALY_MIN_C = LST_VALID_MIN_C - LST_VALID_MAX_C  # -130.0
ANOMALY_MAX_C = LST_VALID_MAX_C - LST_VALID_MIN_C  # +130.0
N_ANOMALY_BINS = int(round((ANOMALY_MAX_C - ANOMALY_MIN_C) / ANOMALY_BIN_C))

# --------------------------------------------------------------------------
# Feathering constants.
# --------------------------------------------------------------------------

#: The swath grid, as a divisor of the output grid. At 3600 px/degree this puts
#: the swath and the cross-fade on a 1/450 degree grid, about 240 m. The ramp is
#: smooth over tens of kilometres, so nothing in it needs 30 m.
SWATH_FACTOR = 8

#: A quad's swath is the ground at least this share of its scenes reached. A
#: union overreaches: one outlying scene extends the swath past where the path
#: contributes and leaves no edge to ramp toward. Grouping by (path, row) quad
#: rather than by path is what makes the share mean anything, because a path's
#: scenes span five rows.
SWATH_QUAD_SHARE = 0.5

#: Resolution divisor for the distance ramp inside an overlap, on top of the
#: swath grid. Containment stays exact on the swath grid whatever this is, so
#: membership and single-path pixels are untouched and only the ratio between
#: overlapping paths is interpolated. Exact point-to-boundary distance was
#: measured at 99.8% of the geometry cost in `nlebovits/landsat-lst`, because a
#: swath boundary carries thousands of vertices and shapely walks them per
#: point. Set to 1 for the exact ramp.
WEIGHT_FACTOR = 4

#: Below this many cells on either side, coarsening carries the ramp on too few
#: cells to be worth the error, so the exact ramp runs instead.
MIN_COARSE_EDGE = 16

#: Rows per block when measuring point-to-boundary distance. Bounds the shapely
#: point array, which is the only large allocation in the ramp.
_ROW_BLOCK = 256

#: The grid every raster here rests on. Written as a string rather than a
#: `rasterio.crs.CRS`, which rasterio resolves itself and which `ty` cannot: the
#: module is a compiled extension with no stub to read. `masks.transform_for`
#: assumes the same geographic grid one level down.
GRID_CRS = "EPSG:4326"


# --------------------------------------------------------------------------
# The climatology and the offset. `tile_prep` calls these, once per tile.
# --------------------------------------------------------------------------


def month_climatology(celsius, months):
    """Per-pixel median for each calendar month present, pooled across years.

    Args:
        celsius: `(n_scenes, ny, nx)` float32, NaN where unusable.
        months: `(n_scenes,)` int, the calendar month of each scene.

    Returns:
        `(planes, ref)`. `planes` holds the months observed, ascending. `ref` is
        `(n_planes, ny, nx)` float32, NaN where a month has no observation.

    A month with no scene gets no plane. Reindexing to twelve and filling would
    invent a reference, and a scene compared against an invented reference is a
    scene with an invented offset.
    """
    import numpy as np

    months = np.asarray(months)
    planes = np.unique(months)
    ref = np.empty((planes.size, *celsius.shape[1:]), dtype="float32")
    for i, month in enumerate(planes):
        with np.errstate(all="ignore"):
            ref[i] = np.nanmedian(celsius[months == month], axis=0)
    return planes, ref


def accumulate_anomaly(hist, n_valid, celsius, months, planes, ref):
    """Bin each scene's anomaly against its own month's reference, in place.

    This is what makes the whole estimate one pass over the source. A spatial
    median does not decompose across blocks, which is why
    `nlebovits/landsat-lst` splits the estimate into a climatology phase and a
    second phase that re-reads every scene. A histogram does decompose, and at a
    bin one output DN wide the median read back off it is exact to the
    quantisation the product already carries.

    Args:
        hist: `(n_scenes, N_ANOMALY_BINS)` uint32, accumulated in place.
        n_valid: `(n_scenes,)` int64, accumulated in place.
        celsius: this block's stack, `(n_scenes, ny, nx)` float32.
        months: `(n_scenes,)` int.
        planes: the months `ref` carries, ascending.
        ref: `(n_planes, ny, nx)` float32 for this block.
    """
    import numpy as np

    plane_of = {int(m): i for i, m in enumerate(planes)}
    for s in range(celsius.shape[0]):
        scene = celsius[s]
        n_valid[s] += int(np.isfinite(scene).sum())
        anomaly = scene - ref[plane_of[int(months[s])]]
        finite = np.isfinite(anomaly)
        if not finite.any():
            continue
        idx = np.floor((anomaly[finite] - ANOMALY_MIN_C) / ANOMALY_BIN_C)
        idx = np.clip(idx, 0, N_ANOMALY_BINS - 1).astype("int64")
        hist[s] += np.bincount(idx, minlength=N_ANOMALY_BINS).astype("uint32")


def offsets_from_histograms(hist):
    """The median anomaly of each scene, read off its accumulated histogram.

    Returns `(n_scenes,)` float64, NaN for a scene that produced no anomaly.

    The two middle order statistics are averaged, as `numpy.nanmedian` does for
    an even count, so the result tracks a direct median to within one bin.
    """
    import numpy as np

    counts = np.asarray(hist, dtype="int64")
    total = counts.sum(axis=1)
    centres = ANOMALY_MIN_C + (np.arange(N_ANOMALY_BINS) + 0.5) * ANOMALY_BIN_C
    out = np.full(counts.shape[0], np.nan, dtype="float64")
    for s in range(counts.shape[0]):
        n = int(total[s])
        if n == 0:
            continue
        cumulative = np.cumsum(counts[s])
        lower = int(np.searchsorted(cumulative, (n - 1) // 2 + 1))
        upper = int(np.searchsorted(cumulative, n // 2 + 1))
        out[s] = 0.5 * (centres[lower] + centres[upper])
    return out


def keep_mask(offset, n_valid, *, floor, max_offset_c=DESTRIPE_MAX_OFFSET_C):
    """Which scenes survive. One rule, so nothing can apply a second one.

    Three conditions: the offset exists, it rests on enough pixels to believe,
    and it is small enough to be a correction rather than a verdict about the
    scene.

    Rejected scenes are dropped from the stack, never clamped and never passed
    through at zero. Bounding a -73 C offset to -15 C leaves about 58 C of
    uncorrected bias in a scene that is almost certainly cloud-contaminated, and
    then presents it as corrected. An uncorrected scene inside a corrected stack
    is the artifact the correction exists to remove.
    """
    import numpy as np

    offset = np.asarray(offset, dtype="float64")
    return (
        np.isfinite(offset)
        & (np.asarray(n_valid) >= floor)
        & (np.abs(offset) <= max_offset_c)
    )


def offset_diagnostics(offset, keep) -> dict:
    """What the offsets looked like and what rejection removed.

    The rejected share is the number to watch per tile. The cap was calibrated
    on mid-latitude cropland, and a tile departing sharply from the 21.8% seen
    there is saying something about itself.
    """
    import numpy as np

    values = np.asarray(offset, dtype="float64")
    finite = values[np.isfinite(values)]
    kept = np.asarray(keep, dtype=bool)
    out = {
        "n_scenes": int(values.size),
        "n_kept": int(kept.sum()),
        "rejected_frac": round(float(1.0 - kept.mean()), 4) if kept.size else 1.0,
    }
    if finite.size:
        p1, p50, p99 = (float(v) for v in np.percentile(finite, [1, 50, 99]))
        out |= {
            "std": round(float(finite.std()), 2),
            "min": round(float(finite.min()), 2),
            "max": round(float(finite.max()), 2),
            "p1": round(p1, 2),
            "p50": round(p50, 2),
            "p99": round(p99, 2),
        }
    return out


# --------------------------------------------------------------------------
# The swath. `tile_prep` builds it from where each path actually contributed a
# valid pixel, rather than from a footprint polygon.
#
# The sibling reads Earth Search, whose item geometry is the imaged
# parallelogram, and rasterises that. This repository reads the USGS bulk
# metadata, whose corner columns describe the product bounding *rectangle*.
# FINDINGS.md measures the gap at about 46% of the area, and no column in the
# bulk file carries the imaged footprint. The seam sits at the imaged edge, so
# rasterising the ring would put the ramp tens of kilometres off the seam it is
# supposed to remove. Counting valid observations answers the question the
# median footprint answers -- what ground did at least half this quad's scenes
# reach -- and answers it from the data rather than from a polygon.
# --------------------------------------------------------------------------


def swath_masks(quad_count, quad_scenes):
    """One boolean swath per path, from per-quad valid-observation counts.

    Args:
        quad_count: `{(path, row): (h, w) uint16}`, how many of that quad's
            scenes reached each swath cell.
        quad_scenes: `{(path, row): int}`, how many scenes the quad has.

    Returns:
        `{path: (h, w) bool}`, with a path omitted when no cell retains it.
    """
    import numpy as np

    by_path: dict[str, np.ndarray] = {}
    for quad, count in sorted(quad_count.items()):
        path, _row = quad
        keep = count >= max(quad_scenes[quad] * SWATH_QUAD_SHARE, 1.0)
        if not keep.any():
            continue
        if path in by_path:
            by_path[path] |= keep
        else:
            by_path[path] = keep
    return {path: mask for path, mask in by_path.items() if mask.any()}


def _boundaries(masks, paths, transform, shape_hw):
    """Each path's swath boundary, with the grid's own edge taken out of it.

    A swath derived from data is clipped wherever the grid ends, so the grid
    edge becomes part of the polygon boundary. That edge is not an acquisition
    edge, and leaving it in makes the ramp fall toward the corner of the raster
    instead of toward the place the path stops contributing. Two neighbouring
    tiles would then disagree along their shared border, which trades the WRS
    seam for a seam on the tile grid.

    `nlebovits/landsat-lst` reaches the same result by rasterising each swath
    over the footprints' own bounds rather than the tile's. This repository
    reads coverage off the data, so there is no wider extent to rasterise over
    and the edge comes out of the linework instead.

    Returns one geometry per path, or None for a path with no edge left. A path
    filling the grid has nothing to ramp toward, and saying so is better than
    ramping toward the raster.
    """
    from rasterio.features import shapes
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    height, width = shape_hw
    step_x, step_y = abs(transform.a), abs(transform.e)
    west = transform.c
    north = transform.f
    interior = box(
        west + step_x,
        north - step_y * height + step_y,
        west + step_x * width - step_x,
        north - step_y,
    )

    out = []
    for path in paths:
        mask = masks[path]
        polygons = [
            shape(geom)
            for geom, value in shapes(
                mask.astype("uint8"), mask=mask, transform=transform
            )
            if value == 1
        ]
        edge = unary_union(polygons).boundary.intersection(interior)
        out.append(None if edge.is_empty else edge)
    return out


def _exact_distances(masks, paths, transform, shape_hw, inside, multi):
    """Point-to-boundary distance on this grid, only where it can matter.

    A pixel one path reaches needs no distance at all, and only the covered
    pixels of a multi-path region are handed to shapely. That is the difference
    between seconds and minutes: on a production band `nlebovits/landsat-lst`
    measured containment at 0.04 s and exact distance at 152 to 180 s.

    A path with no edge inside the grid is held at the grid diagonal, so it
    dominates the blend uniformly and the paths that do have an edge ramp
    against it. No edge means no ramp, and a flat share is what that says.
    """
    import numpy as np
    import shapely

    height, width = shape_hw
    n = len(paths)
    dist = np.zeros((n, height, width), dtype="float32")
    boundaries = _boundaries(masks, paths, transform, shape_hw)
    span = float(np.hypot(abs(transform.a) * width, abs(transform.e) * height))
    lon = transform.c + transform.a * (np.arange(width, dtype="float64") + 0.5)
    for y0 in range(0, height, _ROW_BLOCK):
        y1 = min(y0 + _ROW_BLOCK, height)
        block = multi[y0:y1]
        if not block.any():
            continue
        lat = transform.f + transform.e * (np.arange(y0, y1, dtype="float64") + 0.5)
        yy, xx = np.nonzero(block)
        points = shapely.points(lon[xx], lat[yy])
        for j in range(n):
            sel = inside[j, y0:y1][yy, xx]
            if not sel.any():
                continue
            d = np.zeros(points.size, dtype="float64")
            if boundaries[j] is None:
                d[sel] = span
            else:
                d[sel] = shapely.distance(points[sel], boundaries[j])
            target = dist[j, y0:y1]
            target[yy, xx] = d.astype("float32")
    return dist


def _block_any(mask, factor):
    """Coarsen a mask by "any cell inside", which grows it by up to one cell.

    Subsampling would shrink it instead, and a shrunken coarse mask leaves a
    band one to `factor` cells wide where the exact containment says two paths
    overlap and the coarse ramp is undefined for both. Every pixel in that band
    then falls to the equal-share tie-break, which puts a 0.5 line along the
    swath edge. That line is the artifact this whole module exists to remove.

    Growing instead means the distance is measured to a boundary up to one
    coarse cell outside the true one, and `shapely.distance` is unsigned, so
    the value just outside a boundary is small and positive exactly as it is
    just inside. Interpolation carries it to zero at the edge either way.
    """
    import numpy as np

    height, width = mask.shape
    ch = -(-height // factor)
    cw = -(-width // factor)
    padded = np.zeros((ch * factor, cw * factor), dtype=bool)
    padded[:height, :width] = mask
    return padded.reshape(ch, factor, cw, factor).any(axis=(1, 3))


def _coarse_distances(masks, paths, transform, shape_hw, factor):
    """The ramp measured on a grid `factor` cells coarser, resampled back.

    The ramp is smooth over tens of kilometres, so a coarser grid carries it to
    well under 1% of a weight while cutting the point count by `factor**2`.
    """
    import numpy as np
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    height, width = shape_hw
    coarse = Affine(
        transform.a * factor,
        transform.b,
        transform.c,
        transform.d,
        transform.e * factor,
        transform.f,
    )

    c_masks = {p: _block_any(masks[p], factor) for p in paths}
    c_inside = np.stack([c_masks[p] for p in paths])
    ch, cw = c_inside.shape[1], c_inside.shape[2]
    c_multi = c_inside.sum(axis=0) >= 2
    c_dist = _exact_distances(c_masks, paths, coarse, (ch, cw), c_inside, c_multi)

    out = np.zeros((len(paths), height, width), dtype="float32")
    for j in range(len(paths)):
        reproject(
            source=c_dist[j],
            destination=out[j],
            src_transform=coarse,
            src_crs=GRID_CRS,
            dst_transform=transform,
            dst_crs=GRID_CRS,
            resampling=Resampling.bilinear,
        )
    return out


def _blend(weight, dist, inside, multi, k):
    """Turn distances into shares in place: `w_j = d_j / sum_i d_i`.

    Renormalised on the containment masks, so an interpolated ramp still sums to
    one and still gives a single-path pixel exactly its own weight. On a
    boundary every distance is zero; equal shares are the answer there rather
    than a division by zero, and the pixel is a measure-zero line either way.
    """
    import numpy as np

    n = weight.shape[0]
    total = dist.sum(axis=0)
    safe = np.where(total > 0, total, 1.0)
    for j in range(n):
        share = np.where(total > 0, dist[j] / safe, 0.0)
        weight[j] = np.where(multi & inside[j], share.astype("float32"), weight[j])
    wsum = weight.sum(axis=0)
    fix = multi & (wsum > 0)
    safe_sum = np.where(wsum > 0, wsum, 1.0)
    for j in range(n):
        weight[j] = np.where(fix, weight[j] / safe_sum, weight[j])
    degenerate = multi & (wsum <= 0)
    if degenerate.any():
        for j in range(n):
            sel = degenerate & inside[j]
            weight[j][sel] = 1.0 / k[sel]


def path_weights(masks, transform, *, factor=WEIGHT_FACTOR):
    """Cross-fade weights for `masks` on their own grid.

    Inside a pixel's covering set the weight is its distance to its own swath
    boundary over the sum of those distances, `w_j = d_j / sum_i d_i`. One
    covering path gives exactly 1, so a single-path pixel is untouched. Two give
    the linear cross-fade that reaches 0 at one edge and 1 at the other. Three
    or more stay continuous, which matters because 5.4% of S30W065 is reached by
    three.

    Returns:
        `(paths, weight, inside)`. `paths` is ascending, and that order is
        canonical: the weighted sum downstream runs in a fixed sequence, so
        permuting the input cannot move a floating-point result. `weight` is
        `(n_paths, h, w)` float32 summing to 1 where any path covers and 0
        elsewhere. `inside` is `(n_paths, h, w)` bool.
    """
    import numpy as np

    paths = tuple(sorted(masks))
    n = len(paths)
    if n == 0:
        return (), np.zeros((0, 0, 0), "float32"), np.zeros((0, 0, 0), bool)

    inside = np.stack([masks[p] for p in paths]).astype(bool)
    height, width = inside.shape[1], inside.shape[2]
    k = inside.sum(axis=0).astype("uint8")
    weight = np.zeros((n, height, width), dtype="float32")

    single = k == 1
    if single.any():
        for j in range(n):
            weight[j][single & inside[j]] = 1.0

    multi = k >= 2
    if multi.any():
        coarse_enough = (
            factor > 1 and min(height // factor, width // factor) >= MIN_COARSE_EDGE
        )
        dist = (
            _coarse_distances(masks, paths, transform, (height, width), factor)
            if coarse_enough
            else _exact_distances(
                masks, paths, transform, (height, width), inside, multi
            )
        )
        _blend(weight, dist, inside, multi, k)

    return paths, weight, inside


# --------------------------------------------------------------------------
# Joining a loaded stack back to its scenes.
#
# `odc.stac` sorts the time axis by acquisition datetime, which is not the order
# the item list arrives in, and it drops a step whose scenes miss the window
# entirely. `tests/test_load_parity.py` pins the sort. So every per-scene
# quantity joins on the time coordinate and never on position. The sibling paid
# for this once: labels carried positionally against a stack that de-striping
# had thinned from 1,031 steps to 912 killed all 35 composite shards with an
# `IndexError`.
# --------------------------------------------------------------------------


def path_of(item) -> str:
    """The WRS path of one item, kept as a string so zero padding survives."""
    properties = item["properties"] if isinstance(item, dict) else item.properties
    return str(properties["landsat:wrs_path"])


def quad_of(item) -> tuple[str, str]:
    """The `(path, row)` quad of one item."""
    properties = item["properties"] if isinstance(item, dict) else item.properties
    return (str(properties["landsat:wrs_path"]), str(properties["landsat:wrs_row"]))


def scene_id_of(item) -> str:
    properties = item["properties"] if isinstance(item, dict) else item.properties
    return str(properties["landsat:scene_id"])


def timestamp_of(item):
    """One item's acquisition time, as the nanosecond stamp odc.stac loads."""
    import datetime as dt

    import numpy as np

    properties = item["properties"] if isinstance(item, dict) else item.properties
    text = str(properties["datetime"]).replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(text).astimezone(dt.UTC).replace(tzinfo=None)
    return np.datetime64(parsed, "ns")


def _by_timestamp(items, value_of, what: str) -> dict:
    """Map each acquisition stamp to one value, refusing to guess on a clash.

    Two scenes sharing a stamp is physically possible only across platforms, and
    two of those carrying different values is a case nothing here can resolve.
    Raising names it instead of averaging it away.
    """
    out: dict = {}
    for item in items:
        stamp = timestamp_of(item)
        value = value_of(item)
        if out.setdefault(stamp, value) != value:
            msg = (
                f"two scenes share the acquisition stamp {stamp} and disagree "
                f"about {what} ({out[stamp]!r} against {value!r}). The time axis "
                f"cannot carry a per-scene {what}. Run with --no-destripe "
                f"--no-feather to composite pooled."
            )
            raise ValueError(msg)
    return out


def align_to_time(items, times, *, value_of, what: str, dtype):
    """One per-scene value for each step of a loaded stack, joined on time.

    Raises:
        ValueError: if the stack carries a step no item accounts for. That is a
            defect in the item list this shard was handed, not something to fill
            in with a default.
    """
    import numpy as np

    table = _by_timestamp(items, value_of, what)
    loaded = np.asarray(times).astype("datetime64[ns]")
    missing = [str(t) for t in loaded if t not in table]
    if missing:
        msg = (
            f"the loaded stack carries {len(missing)} time steps no item "
            f"accounts for, first {missing[0]}. The item list and the stack "
            f"come from different searches."
        )
        raise ValueError(msg)

    values = [table[t] for t in loaded]
    if dtype is not object:
        return np.array(values, dtype=dtype)
    # A `(path, row)` quad is a tuple, and `np.array` of uniform tuples builds a
    # two-dimensional array whose rows are arrays. Filling one cell at a time is
    # what keeps a quad a hashable pair.
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


# --------------------------------------------------------------------------
# Applying it. `shard_lst_p95` and `profile_lst_p95` call these.
# --------------------------------------------------------------------------


def weights_for_window(weight, inside, src_transform, dst_transform, shape_hw):
    """Resample the tile's weight field onto one shard's grid.

    Containment is resampled nearest and the ramp bilinearly, then the shares
    are renormalised on the resampled containment. Bilinear alone would leak a
    small weight to a path that does not cover the pixel, and the leak would
    land on the swath edge, which is the one place this has to be right.

    The field is a property of the tile, never of a shard. Deriving it per shard
    would let two shards disagree and invent a seam on the shard grid, which is
    the defect this would be trading the WRS seam for.
    """
    import numpy as np
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    height, width = shape_hw
    n = weight.shape[0]
    out = np.zeros((n, height, width), dtype="float32")
    cover = np.zeros((n, height, width), dtype="uint8")
    for j in range(n):
        for source, destination, method in (
            (weight[j], out[j], Resampling.bilinear),
            (inside[j].astype("uint8"), cover[j], Resampling.nearest),
        ):
            reproject(
                source=source,
                destination=destination,
                src_transform=src_transform,
                src_crs=GRID_CRS,
                dst_transform=dst_transform,
                dst_crs=GRID_CRS,
                resampling=method,
            )
    covered = cover.astype(bool)
    out[~covered] = 0.0
    total = out.sum(axis=0)
    out /= np.where(total > 0, total, np.float32(1.0))
    k = covered.sum(axis=0)
    degenerate = (total <= 0) & (k > 0)
    if degenerate.any():
        for j in range(n):
            sel = degenerate & covered[j]
            out[j][sel] = 1.0 / k[sel]
    return out


def subtract_offsets(celsius, offset):
    """Remove each scene's bias in place, and do nothing else to it.

    The same constant applies to every pixel of a scene, so this shifts that
    scene's baseline and nothing more. It does not alter within-scene contrast,
    create or erase a hot spot, sharpen or blur a feature, or move any value
    relative to its neighbour.
    """
    import numpy as np

    celsius -= np.asarray(offset, dtype="float32")[:, None, None]
    return celsius


def apply_to_stack(celsius, valid, items, times, correction):
    """Reject, de-bias, and label one loaded stack in place.

    Args:
        celsius: `(n_steps, ny, nx)` float32 from `lst_qa.masked_celsius`.
        valid: the boolean mask that produced it. The monthly counts come from
            it, so a rejected scene has to leave it too. `qa_count` then says
            what evidence is behind the P95 rather than what was available.
        items: this shard's items, in the order `correction` runs parallel to.
        times: the loaded time axis.
        correction: the payload `shard_correction` built.

    Returns:
        `(labels, n_rejected)`. `labels` carries each step's WRS path, or None
        for a step whose scene was rejected, which matches no path and so enters
        no reduction.
    """
    import numpy as np

    index_of = {scene_id_of(item): i for i, item in enumerate(items)}
    position = align_to_time(
        items,
        times,
        value_of=lambda item: index_of[scene_id_of(item)],
        what="scene index",
        dtype="int64",
    )
    labels = align_to_time(
        items, times, value_of=path_of, what="WRS path", dtype=object
    )
    keep = np.asarray(correction["keep"], dtype=bool)[position]
    offset = np.asarray(correction["offset"], dtype="float64")[position]

    rejected = ~keep
    if rejected.any():
        celsius[rejected] = np.nan
        valid[rejected] = False
        labels[rejected] = None
    subtract_offsets(celsius, np.where(keep, offset, 0.0))
    return labels, int(rejected.sum())


def feathered_percentile(celsius, path_of_scene, paths, weight, q=95.0):
    """One percentile per WRS path, cross-faded on `weight`.

    The blend happens in value space, between per-path estimates. Pooling the
    samples and taking one weighted percentile is a different operation, and it
    is the operation that produces the step: a pooled percentile already mixes
    the two paths' distributions, which is why it jumps where one path's
    coverage stops.

    A path covering a pixel but observing nothing there drops out of that
    pixel's blend and the remaining paths renormalise, so a thin path cannot
    pull a value toward nodata.

    A pixel no swath covers falls back to the pooled percentile of whatever
    observed it. A swath is the ground at least `SWATH_QUAD_SHARE` of a quad's
    scenes reached, so a pixel one path reaches on a third of its passes lies
    outside every swath and still carries real observations. Returning NaN
    there would discard them while `qa_count` went on counting them, and the
    tiles that lose the most would be the cloudy ones that have the least.

    The fallback is the blend, not an exception to it. Where no swath covers, no
    path has an opinion about the ratio, so `w_j = d_j / sum_i d_i` degenerates
    to the unweighted percentile. It meets the feathered value continuously at
    the swath edge, because the scenes observing a pixel just outside path A's
    swath are almost all A's.

    Args:
        celsius: `(n_scenes, ny, nx)` float32, already de-biased and masked.
        path_of_scene: `(n_scenes,)` path labels, aligned to `celsius`.
        paths: the path labels `weight` carries, in its own order.
        weight: `(n_paths, ny, nx)` float32 on this shard's grid.

    Returns:
        `(field, n_pooled)`. `field` is `(ny, nx)` float32, NaN only where
        nothing was observed. `n_pooled` counts the pixels that took the
        fallback, which is how much of this shard the cross-fade could not
        describe.
    """
    import numpy as np

    labels = np.asarray(path_of_scene)
    shape_hw = celsius.shape[1:]
    numerator = np.zeros(shape_hw, dtype="float32")
    denominator = np.zeros(shape_hw, dtype="float32")

    for j, path in enumerate(paths):
        sel = labels == path
        if not sel.any():
            continue
        subset = celsius[sel]
        present = np.isfinite(subset).any(axis=0)
        if not present.any():
            continue
        with np.errstate(all="ignore"):
            # overwrite_input is safe because fancy indexing already made
            # `subset` a copy, and it keeps the partition off a second one.
            estimate = np.nanpercentile(subset, q, axis=0, overwrite_input=True).astype(
                "float32"
            )
        effective = np.where(present, weight[j], np.float32(0.0))
        numerator += np.where(present, estimate, np.float32(0.0)) * effective
        denominator += effective

    covered = denominator > 0
    safe = np.where(covered, denominator, np.float32(1.0))
    out = np.where(covered, numerator / safe, np.float32(np.nan)).astype("float32")

    observed = np.isfinite(celsius).any(axis=0)
    pooled = observed & ~covered
    if pooled.any():
        with np.errstate(all="ignore"):
            # Reduces only the pixels it rescues, so the cost tracks the loss
            # it prevents. overwrite_input is safe for the reason it is safe
            # above: the boolean index already made a copy.
            out[pooled] = np.nanpercentile(
                celsius[:, pooled], q, axis=0, overwrite_input=True
            ).astype("float32")
    return out, int(pooled.sum())


def scene_digest(scene_ids, window: dict) -> str:
    """A fingerprint of the scene set and the window an offset was fitted over.

    Two prep files built from different scene lists under identical settings are
    otherwise indistinguishable, and a composite built against the wrong one
    finishes and looks ordinary. `nlebovits/landsat-lst` reaches the same place
    by hashing the scene ids into its cache key. Its cache is gone from this
    port and this is what replaces the protection the key was giving.

    Scene ids are sorted because a catalogue returns them in no fixed order.
    """
    import hashlib

    material = "\n".join(
        [
            *(f"{key}={window[key]}" for key in sorted(window)),
            f"lst_valid_min={LST_VALID_MIN_C}",
            f"lst_valid_max={LST_VALID_MAX_C}",
            *sorted(str(s) for s in scene_ids),
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Prep:
    """What `tile_prep` measured for one tile, as a slice reads it back.

    `weight` and `inside` live on the swath grid of the padded prep bbox, which
    is wider than the tile. A shard resamples its own window out of them.
    """

    tile: str
    bbox: tuple[float, float, float, float]
    pixels_per_degree: int
    swath_factor: int
    paths: tuple[str, ...]
    weight: object
    inside: object
    offset: dict[str, float]
    n_valid: dict[str, int]
    meta: dict

    @property
    def digest(self) -> str:
        return str(self.meta.get("scene_digest", ""))

    @property
    def window(self) -> dict:
        return dict(self.meta.get("window", {}))

    @property
    def inventory(self) -> dict:
        return dict(self.meta.get("inventory", {}))


def load_prep(path) -> Prep:
    """Read a `tile_prep` artifact. Both files, or neither.

    The offsets are read; the rejection is not. `tile_prep` reports a rejected
    share under the cap it was given, and the cap that decides a run is applied
    here, so sweeping it costs one read of this file rather than another pass
    over the tile.
    """
    import json
    from pathlib import Path

    import numpy as np

    path = Path(path)
    directory = path if path.is_dir() else path.parent
    meta = json.loads((directory / "tile-prep.json").read_text())
    payload = np.load(directory / "tile-prep.npz", allow_pickle=False)
    ids = [str(s) for s in payload["scene_ids"]]
    return Prep(
        tile=meta["tile"],
        bbox=tuple(meta["bbox"]),
        pixels_per_degree=int(meta["pixels_per_degree"]),
        swath_factor=int(meta["swath_factor"]),
        paths=tuple(str(p) for p in payload["paths"]),
        weight=payload["weight"],
        inside=payload["inside"],
        offset=dict(zip(ids, payload["offset"].tolist(), strict=True)),
        n_valid=dict(zip(ids, payload["n_valid"].tolist(), strict=True)),
        meta=meta,
    )


def prep_transform(prep: Prep):
    """The affine of the swath grid `prep.weight` rests on."""
    from masks import transform_for

    return transform_for(prep.bbox, prep.pixels_per_degree // prep.swath_factor)


def shard_correction(
    prep: Prep,
    item_dicts,
    shard_bbox,
    shape_hw,
    *,
    pixels_per_degree: int,
    max_offset_c=DESTRIPE_MAX_OFFSET_C,
    debias: bool = True,
    feather: bool = True,
    emit_pooled: bool = False,
):
    """Everything one shard needs, cut down to that shard, in the driver.

    The tile's weight field is hundreds of megabytes and every shard wants a few
    hundred kilobytes of it. Resampling the window here keeps the field in one
    process and the task payload small, and it is the same reason
    `items_for_shard` sends a subset of the item list rather than all of it.

    Returned lists run parallel to `item_dicts`, which is the order the shard
    already holds. The shard joins them to its loaded time axis itself.
    """
    import numpy as np
    from masks import transform_for

    ids = [scene_id_of(d) for d in item_dicts]
    if debias:
        offset = np.array([prep.offset.get(s, np.nan) for s in ids], dtype="float64")
        n_valid = np.array([prep.n_valid.get(s, 0) for s in ids], dtype="int64")
        keep = keep_mask(
            offset, n_valid, floor=DESTRIPE_MIN_PREP_SAMPLES, max_offset_c=max_offset_c
        )
    else:
        # Every scene at its own baseline, and none of them rejected. Rejection
        # is part of the correction, not a separate screen: a scene is discarded
        # because its offset cannot be trusted, and without the offset there is
        # nothing to distrust.
        offset = np.zeros(len(ids), dtype="float64")
        keep = np.ones(len(ids), dtype=bool)

    weight = None
    if feather and prep.paths:
        weight = weights_for_window(
            prep.weight,
            prep.inside,
            prep_transform(prep),
            transform_for(shard_bbox, pixels_per_degree),
            shape_hw,
        )
    return {
        "offset": offset,
        "keep": keep,
        "paths": prep.paths if weight is not None else (),
        "weight": weight,
        "emit_pooled": emit_pooled,
    }


def pooled_percentile(celsius, q=95.0):
    """The composite this repository built before feathering, for comparison.

    Not free here. The sharded path is eager, so this is a second reduction over
    the whole stack rather than one more expression on a graph already held.
    """
    import numpy as np

    with np.errstate(all="ignore"):
        return np.nanpercentile(celsius, q, axis=0).astype("float32")


# --------------------------------------------------------------------------
# The same two corrections, kept lazy for `profile_lst_p95.build_graph`.
#
# That path is the array-graph architecture this repository measured and left
# behind, retained as a profiling harness. It still has to agree with the
# sharded path pixel for pixel, which `tests/test_pipeline_paths.py` asserts, so
# the corrections have to reach it too. Rejection drops time steps here rather
# than filling them with NaN: a lazy subset costs nothing, where an eager one
# would copy the stack.
# --------------------------------------------------------------------------


def apply_to_stack_xr(lst, items, correction):
    """Drop rejected steps and subtract the offsets, lazily.

    Returns `(debiased, labels, n_rejected)`, where `labels` runs parallel to
    the surviving time axis.
    """
    import numpy as np
    import xarray as xr

    index_of = {scene_id_of(item): i for i, item in enumerate(items)}
    times = lst["time"].values
    position = align_to_time(
        items,
        times,
        value_of=lambda item: index_of[scene_id_of(item)],
        what="scene index",
        dtype="int64",
    )
    labels = align_to_time(
        items, times, value_of=path_of, what="WRS path", dtype=object
    )
    keep = np.asarray(correction["keep"], dtype=bool)[position]
    offset = np.asarray(correction["offset"], dtype="float64")[position]

    kept = np.flatnonzero(keep)
    survivors = lst.isel(time=kept)
    shift = xr.DataArray(
        offset[kept].astype("float32"),
        dims=["time"],
        coords={"time": survivors["time"]},
    )
    return survivors - shift, labels[kept], int(keep.size - kept.size)


def feathered_quantile_xr(lst, path_of_scene, paths, weight, dims, q=0.95):
    """`feathered_percentile` as one expression on a graph already held.

    Every per-path subset is taken from `lst` after whatever rechunk the caller
    has already applied, so all of them descend from the same source blocks and
    one `dask.compute` reads each block once. Rechunking a subset here would
    give the scheduler two incompatible consumers of the same stack, and it
    would hold the whole thing rather than stream it.

    A pixel no swath covers falls back to the pooled quantile, for the reason
    `feathered_percentile` gives. The fallback is one more reduction over the
    same source blocks, so it joins the same compute rather than adding a pass.

    This returns the field alone. `feathered_percentile` also returns how many
    pixels took the fallback, which is a count a lazy graph cannot produce
    without forcing a compute the caller did not ask for.

    Returns None when no path matched a single step, which leaves the caller to
    composite pooled.
    """
    import numpy as np
    import xarray as xr

    labels = np.asarray(path_of_scene)
    coords = {d: lst[d] for d in dims}
    numerator = None
    denominator = None
    for j, path in enumerate(paths):
        steps = np.flatnonzero(labels == path)
        if steps.size == 0:
            continue
        subset = lst.isel(time=steps)
        present = subset.notnull().sum(dim="time") > 0
        estimate = subset.quantile(q, dim="time").drop_vars("quantile", errors="ignore")
        effective = xr.DataArray(weight[j], dims=list(dims), coords=coords).where(
            present, 0.0
        )
        contribution = xr.where(present, estimate, 0.0) * effective
        numerator = contribution if numerator is None else numerator + contribution
        denominator = effective if denominator is None else denominator + effective

    if numerator is None or denominator is None:
        return None
    covered = denominator > 0
    feathered = xr.where(covered, numerator / denominator.where(covered, 1.0), np.nan)
    pooled = lst.quantile(q, dim="time").drop_vars("quantile", errors="ignore")
    return xr.where(covered, feathered, pooled)

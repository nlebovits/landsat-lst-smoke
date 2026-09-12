# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2", "dask", "odc-stac", "odc-geo", "pystac",
#   "xarray", "rioxarray", "numpy", "geopandas",
#   "psutil", "rich", "boto3", "pyarrow>=16",
#   "rasterio", "shapely", "pyogrio",
#   "stac-geoparquet", "matplotlib",
# ]
# ///
"""The p95 LST composite of one tile, as one lazy dask-xarray graph on frisky.

`composite.build_graph` opens every scene of the tile once, chunked in space
and never in time, and reduces each block to its p95 and its monthly counts
inside one task. The blocks stream from the workers into two staging
GeoTIFFs, which become the two COGs the catalog publishes. Nothing larger
than a block returns to this process, and nothing else is written to disk.

This driver does everything around that graph: it resolves the tile, checks
the mask artifacts and the prep artifact, stages the scene objects to local
disk, starts the frisky cluster, computes, collects the trace, and writes the
catalog. Every phase is a frisky client phase, so the dashboard shows what the
driver is doing while it does it.

    uv run shard_lst_p95.py --tile S30W065 --tile-prep ./tile-prep \\
        --stage-dir /mnt/nvme/stage --out-dir ./run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import aster_ged
import composite
import destripe
import masks
import observe
import staging
from aster_ged import DEFAULT_NUMOBS_URI
from cog_catalog import (
    DEFAULT_HOST_NAME,
    DEFAULT_HOST_URL,
    DEFAULT_LICENSE,
    MONTH_NAMES,
    check_catalog_inputs,
    write_catalog,
    write_cog,
)
from land_tiles import tile_bounds
from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE
from memory_sampler import MemorySampler
from stac_window import (
    DEFAULT_CLOUD_COVER_LT,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
)
from tile_inventory import (
    INVENTORY_SCHEMA_VERSION,
    check_manifest,
    items_for_tile,
    provenance,
    read_manifest,
)

#: Where the staged artifacts live on a VM unless the driver says otherwise.
DEFAULT_INVENTORY_URI = Path("artifacts/tile_scene_inventory.parquet")

#: Where scene objects are fetched to before the cluster starts. Overridden by
#: `LST_STAGE_DIR` and then by `--stage-dir`. The system temp directory is the
#: default because it is the one path that exists on every machine; a fleet
#: instance should point this at its NVMe mount instead.
DEFAULT_STAGE_DIR = Path(tempfile.gettempdir()) / "landsat-lst-stage"

#: The read environment still has a source, because requester-pays and the
#: region belong to the bucket the hrefs point at. It no longer selects a
#: catalogue: the items come from the inventory, and every href it writes is
#: `s3://usgs-landsat`.
#:
#: `planetary-computer` is listed and then refused. Keeping it in `choices`
#: means the run stops with a sentence that says why, rather than with
#: argparse's "invalid choice", which would read as a typo.
READ_SOURCES = ("earth-search", "planetary-computer")

#: The only source this path can read. See `configure_read_env`.
SUPPORTED_READ_SOURCE = "earth-search"

#: Band order of the qa_count asset, and the report's column order. Defined
#: once in cog_catalog, because the COG band descriptions have to match.
MONTHS = MONTH_NAMES
GIB = 1024.0**3

#: What a rehearsal prefixes to every line it prints and every file it names,
#: so a synthetic run can never be read as a measurement.
REHEARSAL_TAG = "REHEARSAL: "


# --------------------------------------------------------------------------
# Read environment
# --------------------------------------------------------------------------


def configure_read_env(source: str = SUPPORTED_READ_SOURCE) -> None:
    """Set the GDAL and AWS variables that every S3 read path depends on.

    The number of HTTP requests GDAL issues is a function of these settings.
    `GDAL_DISABLE_READDIR_ON_OPEN` suppresses a directory listing on each open,
    and `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` collapses adjacent block reads into
    one request. A request count measured without them describes a different
    pipeline, so anything that reads scenes must call this first.

    The workers are spawned processes and inherit only the environment, so
    these have to be set before the cluster starts. frisky has no nanny to set
    the BLAS thread caps, so they are set here too.

    Raises:
        SystemExit: for any source but `earth-search`. The inventory writes
            `s3://usgs-landsat` hrefs and that bucket is requester-pays, so a
            different source would skip `AWS_REQUEST_PAYER` and every read
            would fail on a bucket the run is entitled to read.
    """
    if source != SUPPORTED_READ_SOURCE:
        msg = (
            f"--source {source} cannot read this inventory. Every href it "
            f"holds is s3://usgs-landsat, which is requester-pays, and only "
            f"--source {SUPPORTED_READ_SOURCE} sets AWS_REQUEST_PAYER for it. "
            f"The flag selected a catalogue before the inventory replaced the "
            f"per-tile search. It now selects a read environment only. The "
            f"reference catalogue module is where a Planetary Computer query "
            f"belongs."
        )
        raise SystemExit(msg)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
    os.environ.setdefault(
        "AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-west-2")
    )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def resolve_area(args):
    """The bbox this run covers, and the tile it belongs to.

    `--tile` is the production form: it fixes the bbox on the shared grid, so
    two machines given the same tile cut the same pixels. `--bbox` stays for
    dry runs and rehearsals, which plan blocks without reading the inventory.

    Returns:
        The bbox as `(west, south, east, north)`, and the tile id or None.
    """
    if args.tile and args.bbox:
        raise SystemExit("pass --tile or --bbox, not both")
    if args.tile:
        return tile_bounds(args.tile), args.tile
    if not args.bbox:
        raise SystemExit("pass --tile (production) or --bbox (dry run)")
    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox needs west,south,east,north")
    return bbox, None


def load_tile_items(args, tile_id: str):
    """Every scene for one tile, from the precomputed inventory.

    Replaces the per-tile Earth Search query. The artifact is built once by
    `usgs_inventory`, staged for the fleet, and read here with a row-group
    lookup. Nothing in this function opens a catalogue.

    The manifest is checked before any read. A window or a filter the artifact
    does not cover stops the run here, which is the point: the alternative is
    a finished composite built from the wrong scenes.

    Returns:
        The item dicts, their tile-local bboxes, and the run provenance.
    """
    manifest = read_manifest(args.inventory_uri)
    check_manifest(
        manifest,
        start=args.start,
        end=args.end,
        platforms=args.platforms,
        cloud_cover_lt=args.cloud_cover_lt,
        schema_version=INVENTORY_SCHEMA_VERSION,
    )
    items, boxes = items_for_tile(
        args.inventory_uri, tile_id, bounds=tile_bounds(tile_id)
    )
    return items, boxes, provenance(manifest)


def stage_scenes_for(args, item_dicts, say=print):
    """Fetch the tile's scene objects to local disk, or say why it did not.

    Runs to completion before the cluster starts. Overlapping the fetch with
    the compute it feeds was MEASURED as a loss on an `m6id.16xlarge`: the
    same 1,998 objects and 78.9 GiB stage in 91.9 s with the machine to
    themselves and in 358.7 s beside 64 busy workers, because staging spends
    its time on TLS, HTTP and the copy loop, all of which want a core.

    Returns:
        The staging report, or None under `--rehearse`, which reads files it
        wrote itself.
    """
    if args.rehearse:
        say("stage         skipped: the rehearsal reads its own synthetic files")
        return None
    report = staging.stage_scenes(
        item_dicts,
        range(len(item_dicts)),
        args.stage_dir,
        threads=args.stage_threads,
    )
    report_staging(report, say)
    return report


def report_staging(report, say=print) -> None:
    """The two console lines a staged run prints about what it fetched."""
    say(
        f"stage         {report['objects']:,} objects, "
        f"{report['bytes'] / GIB:.1f} GiB in {report['seconds']:.1f}s "
        f"-> {report['stage_dir']}"
    )
    reused = report.get("reused", 0)
    already = f", {reused:,} already staged" if reused else ""
    say(
        f"              {report['get_requests']:,} billable GETs, "
        f"{report['retries']} retries{already}"
    )


# --------------------------------------------------------------------------
# Early exits. Each writes the summary a driver keys on and says why.
# --------------------------------------------------------------------------


def no_thermal_coverage(args, tile_id, bbox, n_scenes, dropped, run_provenance) -> int:
    """Record a tile that holds no thermal scene, and succeed.

    126 of the 895 land tiles are like this, and every one is an ocean tile
    holding a small island. `thermal_href IS NULL` is exactly `OLI_TIRS_L2SR`
    across all 3,083,129 inventory rows, and USGS emits that product where the
    surface temperature algorithm has no usable emissivity. Compositing there
    is not a failure. There is nothing to composite.
    """
    summary = {
        "status": "no-thermal-coverage",
        "tile": tile_id,
        "bbox": bbox,
        "n_scenes_inventory": n_scenes,
        "n_scenes": 0,
        "scenes_dropped_no_thermal": dropped,
        "inventory": run_provenance,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        f"no thermal    all {n_scenes:,} scenes of {tile_id} are OLI_TIRS_L2SR "
        f"and carry no thermal band"
    )
    print("              nothing to composite; summary written, no rasters")
    print(f"artifacts     {args.out_dir.resolve()}")
    return 0


def no_scene_survives_destriping(args, tile_id, prep, run_provenance) -> int:
    """Every scene of the tile failed the offset rule. Stop rather than ship.

    An empty composite is not a tile with no data. It is a tile whose whole
    scene list was found untrustworthy, and writing it as nodata would present
    that as an observation gap.
    """
    summary = {
        "status": "no-scene-survives-destriping",
        "tile": tile_id,
        "n_scenes": len(prep.offset),
        "max_offset_c": args.max_offset_c,
        "offsets": prep.meta.get("offsets"),
        "inventory": run_provenance,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        f"destripe      all {len(prep.offset):,} scenes of {tile_id} fail the "
        f"{args.max_offset_c:g} C cap or the sparse floor"
    )
    print(f"              {prep.meta.get('offsets')}")
    print("              nothing to composite; summary written, no rasters")
    return 1


def check_mask_inputs(args, say=print) -> dict | None:
    """Refuse a run whose output mask cannot be built, before it costs anything.

    A tile that composites for an hour and then cannot be masked has already
    bought the machine, and a tile written without the mask looks finished
    while carrying sea and ASTER emissivity gaps as temperatures.

    The rehearsal is masked like any other run. Its pixels are synthetic but
    its bbox is not, so the mask covers real ground and a rehearsal proves the
    path the fleet uses rather than a shorter one.

    Returns:
        The ASTER GED provenance for the run summary, or None under
        `--no-output-mask`.

    Raises:
        MaskError: if the land geometry is absent.
        GedError: if the NumObs artifact is absent, or was built from a
            different land geometry.
    """
    if args.no_output_mask:
        say("mask          skipped: --no-output-mask. Sea and ASTER gaps stay")
        return None
    if not args.land_geometry_uri.exists():
        msg = (
            f"no buffered land geometry at {args.land_geometry_uri}. Write it "
            f"with:\n  uv run land_tiles.py --out artifacts/land_tiles.parquet "
            f"--write-geometry {args.land_geometry_uri}"
        )
        raise masks.MaskError(msg)
    manifest = aster_ged.read_manifest(args.numobs_uri)
    aster_ged.check_manifest(
        manifest,
        land_geometry_sha256=masks.geometry_checksum(args.land_geometry_uri),
        path=args.numobs_uri,
    )
    say(
        f"mask          {manifest['granule_count']:,} ASTER GED granules, "
        f"{manifest['collection']['short_name']} v"
        f"{manifest['collection']['version']}"
    )
    return aster_ged.provenance(manifest)


def no_unmasked_pixels(args, tile_id, bbox, counts, ged_provenance) -> int:
    """Record a tile the water rule empties, and succeed."""
    summary = {
        "status": "no-unmasked-pixels",
        "tile": tile_id,
        "bbox": bbox,
        "n_scenes": None,
        "mask": counts
        | {
            "numobs_uri": str(args.numobs_uri),
            "land_geometry_uri": str(args.land_geometry_uri),
            "aster_ged": ged_provenance,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"no coverage   {tile_id} holds no land inside the buffered geometry")
    print("              nothing to publish; summary written, no rasters")
    print(f"artifacts     {args.out_dir.resolve()}")
    return 0


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="p95 LST composite: one lazy dask-xarray graph per tile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tile",
        default=None,
        help="tile id such as S30W065. Sets the bbox from the production grid "
        "and selects this tile's rows in the inventory",
    )
    p.add_argument(
        "--bbox",
        default=None,
        help="west,south,east,north EPSG:4326 (use --bbox=...). Only for "
        "--rehearse and --dry-run; a real run needs --tile, because the "
        "inventory is addressed by tile",
    )
    p.add_argument(
        "--inventory-uri",
        type=Path,
        default=DEFAULT_INVENTORY_URI,
        help="the precomputed tile-scene inventory this run reads",
    )
    p.add_argument(
        "--land-tiles-uri",
        type=Path,
        default=Path("artifacts/land_tiles.parquet"),
        help="the authoritative land-tile list, for the driver",
    )
    p.add_argument(
        "--numobs-uri",
        type=Path,
        default=DEFAULT_NUMOBS_URI,
        help="ASTER GED clear-sky observation counts. A pixel whose count is "
        "zero has no emissivity, so Collection 2 never produced a surface "
        "temperature for it and no window ever will",
    )
    p.add_argument(
        "--land-geometry-uri",
        type=Path,
        default=masks.DEFAULT_LAND_GEOMETRY_URI,
        help="the buffered land geometry the pixel mask rasterises",
    )
    p.add_argument(
        "--no-output-mask",
        action="store_true",
        help="write every pixel the composite produced, including sea and "
        "ASTER emissivity gaps. Kept for measuring the mask against its "
        "absence; a published tile always carries it",
    )
    p.add_argument("--pixels-per-degree", type=int, default=3600)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument(
        "--chunk",
        type=int,
        default=composite.DEFAULT_CHUNK_PX,
        help="block edge in pixels: the dask chunk in both spatial dimensions. "
        "The time axis is always one chunk",
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--cloud-cover-lt", type=int, default=DEFAULT_CLOUD_COVER_LT)
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS)
    p.add_argument("--source", choices=sorted(READ_SOURCES), default="earth-search")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=4)
    p.add_argument("--memory-limit-gib", type=float, default=13.0)
    p.add_argument(
        "--rehearse",
        type=int,
        default=0,
        metavar="N",
        help="run the whole pipeline over N synthetic scenes written to local "
        "disk and no S3 reads. Every line and artifact is tagged REHEARSAL",
    )
    p.add_argument("--out-dir", type=Path, default=Path("./composite-run"))
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=Path(os.environ.get("LST_STAGE_DIR", DEFAULT_STAGE_DIR)),
        help="local directory the scene objects are fetched into, before the "
        "cluster starts. Point it at the NVMe mount on a fleet instance",
    )
    p.add_argument(
        "--stage-threads",
        type=int,
        default=None,
        help="threads in the fetch pool. Defaults to min(64, 4 x cores)",
    )
    p.add_argument(
        "--keep-staged",
        action="store_true",
        help="leave the staged files behind. Steady state runs one tile per "
        "instance back to back, so the default removes them",
    )
    p.add_argument(
        "--keep-scenes-without-thermal",
        action="store_true",
        help="keep the L2SR products that carry no lwir11 band. They load as "
        "fill and reach neither the percentile nor the monthly counts, so "
        "the default drops them",
    )
    p.add_argument(
        "--no-catalog",
        action="store_true",
        help="write the two COGs into --out-dir and skip the STAC catalog",
    )
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="where the catalog lives; defaults to <out-dir>/catalog. Point "
        "every tile at one path to collect them in one catalog",
    )
    p.add_argument(
        "--collection-id",
        default=None,
        help="the published collection id, which is also its directory name. "
        "Defaults to the window the tile was composited over",
    )
    p.add_argument("--host-name", default=DEFAULT_HOST_NAME)
    p.add_argument("--host-url", default=DEFAULT_HOST_URL)
    p.add_argument("--license", default=DEFAULT_LICENSE)
    p.add_argument(
        "--force",
        action="store_true",
        help="run even if the block memory model says the machine is too small",
    )
    p.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="seconds between memory samples written to memory.csv",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the blocks and print the budget; no cluster, no reads",
    )
    p.add_argument(
        "--target-memory-gib",
        type=float,
        default=None,
        help="RAM of the machine this run is planned for, in GiB, for the dry "
        "run's verdict",
    )
    p.add_argument(
        "--tile-prep",
        type=Path,
        default=None,
        help="directory holding tile-prep.npz and tile-prep.json, written by "
        "tile_prep.py. Without it the run composites the pooled percentile "
        "and the WRS seam stays in the output",
    )
    p.add_argument(
        "--no-destripe",
        action="store_true",
        help="keep every scene at its own baseline. Reads the prep file for "
        "the swath geometry and ignores the offsets",
    )
    p.add_argument(
        "--no-feather",
        action="store_true",
        help="one pooled percentile instead of one per WRS path",
    )
    p.add_argument(
        "--max-offset-c",
        type=float,
        default=destripe.DESTRIPE_MAX_OFFSET_C,
        help="discard a scene whose absolute offset exceeds this",
    )
    p.add_argument(
        "--emit-pooled",
        action="store_true",
        help="also write lst_p95_pooled.tif beside the product, the pooled "
        "percentile from the same graph after the offsets are applied",
    )
    p.add_argument(
        "--tracing-capacity",
        type=int,
        default=observe.DEFAULT_TRACING_CAPACITY,
        help="frisky spans kept per process",
    )
    args = p.parse_args(argv)
    if args.tile_prep is None and (args.no_destripe or args.no_feather):
        raise SystemExit(
            "--no-destripe and --no-feather turn off corrections that need "
            "--tile-prep to be on at all. Without --tile-prep the run is "
            "already pooled and un-de-striped."
        )
    if args.source != SUPPORTED_READ_SOURCE:
        configure_read_env(args.source)
    return args


# --------------------------------------------------------------------------
# The rules a run was built under, for the catalog to record
# --------------------------------------------------------------------------


def mask_rule(args, counts, ged_provenance=None) -> dict | None:
    """The rule the tile was masked under. None under `--no-output-mask`.

    The inputs are named by identity, not by path. An absolute path on the
    machine that masked the tile tells a reader of the published catalog
    nothing, and it carries the operator's home directory into a public file.
    """
    if counts is None:
        return None
    rule: dict = {
        "gap_buffer_cells": counts.get("gap_buffer_cells"),
        "gap_hot_threshold_c": counts.get("gap_hot_threshold_c"),
        "land_geometry_sha256": masks.geometry_checksum(args.land_geometry_uri),
    }
    if ged_provenance:
        rule["aster_ged"] = ged_provenance
    return rule


def load_tile_prep(args, tile_id: str, item_dicts, run_provenance):
    """The prep artifact, checked against the run that is about to use it.

    Returns None when the run was not given one, which composites pooled.

    Six checks, and each one guards a failure that produces a finished raster
    rather than an error. `item_dicts` must be the list the graph will load,
    after `staging.drop_scenes_without_thermal`, because that is the list
    `tile_prep` hashed.

    Raises:
        SystemExit: on any mismatch, naming the command that rebuilds the file.
    """
    if args.tile_prep is None:
        return None
    prep = destripe.load_prep(args.tile_prep)
    rebuild = f"Rebuild it with tile_prep.py --tile {tile_id}."

    blocks = prep.meta.get("blocks") or {}
    if blocks.get("partial"):
        raise SystemExit(
            f"{args.tile_prep} was built from {blocks.get('run')} of "
            f"{blocks.get('with_scenes')} blocks holding scenes, under "
            f"--max-blocks {blocks.get('max_blocks')}. Each quad's swath was "
            f"counted over part of the tile and divided by all of its scenes, "
            f"so the swaths are too small and the offsets rest on too few "
            f"pixels. Rerun tile_prep.py without --max-blocks."
        )

    if prep.meta.get("schema_version") != tile_prep_schema_version():
        raise SystemExit(
            f"{args.tile_prep} carries schema version "
            f"{prep.meta.get('schema_version')} and this version reads "
            f"{tile_prep_schema_version()}. {rebuild}"
        )
    if prep.tile != tile_id:
        raise SystemExit(
            f"{args.tile_prep} was written for tile {prep.tile}, and this run "
            f"is building {tile_id}. {rebuild}"
        )
    if prep.pixels_per_degree != args.pixels_per_degree:
        raise SystemExit(
            f"{args.tile_prep} was built on a 1/{prep.pixels_per_degree} degree "
            f"grid and this run is on 1/{args.pixels_per_degree}. The weights "
            f"would land on the wrong ground. {rebuild}"
        )

    window = {
        "start": args.start,
        "end": args.end,
        "platforms": args.platforms,
        "cloud_cover_lt": args.cloud_cover_lt,
    }
    mine = destripe.scene_digest((destripe.scene_id_of(d) for d in item_dicts), window)
    if prep.digest != mine:
        raise SystemExit(
            f"{args.tile_prep} was fitted over a different scene set or a "
            f"different window: it carries digest {prep.digest or 'none'} and "
            f"this run computes {mine}. Its window was {prep.window} against "
            f"{window} here, over {len(prep.offset)} scenes against "
            f"{len(item_dicts)}. {rebuild}"
        )
    if prep.inventory and run_provenance and prep.inventory != run_provenance:
        raise SystemExit(
            f"{args.tile_prep} was built from a different inventory artifact:\n"
            f"  prep: {prep.inventory}\n"
            f"  run:  {run_provenance}\n"
            f"The digests match, so the same scenes are named, and a rebuilt "
            f"artifact can still move a footprint or a href. {rebuild}"
        )
    return prep


def tile_prep_schema_version() -> int:
    """Read lazily, because `tile_prep` imports this module."""
    import tile_prep

    return tile_prep.PREP_SCHEMA_VERSION


def correction_rule(args, prep) -> dict | None:
    """The seam correction the tile was built under, for the catalog.

    None means the pooled percentile with every scene at its own baseline,
    which is itself a rule the catalog has to state.
    """
    if prep is None:
        return None
    return {
        "prep_schema_version": prep.meta.get("schema_version"),
        "prep_scene_digest": prep.digest,
        "prep_window": prep.window,
        "prep_tile": prep.tile,
        "prep_bbox": list(prep.bbox),
        "prep_factor": prep.meta.get("prep_factor"),
        "swath_factor": prep.swath_factor,
        "weight_factor": prep.meta.get("weight_factor"),
        "swath_quad_share": prep.meta.get("swath_quad_share"),
        "anomaly_bin_c": prep.meta.get("anomaly_bin_c"),
        "min_offset_samples": prep.meta.get("min_offset_samples"),
        "max_offset_c": None if args.no_destripe else args.max_offset_c,
        "destripe": not args.no_destripe,
        "feather": not args.no_feather,
        "paths": list(prep.paths),
    }


def run_meta(args, bbox, height, width, mask_rule_value, correction_rule_value) -> dict:
    """What the catalog needs to describe this tile. The former part-meta."""
    return {
        "raster": [height, width],
        "bbox": list(bbox),
        "crs": args.crs,
        "pixels_per_degree": args.pixels_per_degree,
        "chunk_px": args.chunk,
        "mask_rule": mask_rule_value,
        "correction_rule": correction_rule_value,
        "start": args.start,
        "end": args.end,
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _dry_run(args, bbox, height, width, say) -> int:
    """Plan the blocks and print the budget; no cluster, no reads."""
    chunk = args.chunk
    rows = -(-height // chunk)
    cols = -(-width // chunk)
    say(f"blocks        {rows * cols}  of {chunk}x{chunk} px ({rows} x {cols})")
    slots = args.workers * args.threads_per_worker
    say(f"\nnaive budget across {slots} slots, DERIVED from the block model:")
    for n, present in ((711, 200), (1765, 400), (4776, 820)):
        per = composite.block_bytes(chunk, n, present)
        total = per * slots
        verdict = ""
        if args.target_memory_gib:
            over = total - args.target_memory_gib
            verdict = f"   OVER by {over:.1f} GiB" if over > 0 else "   fits"
        say(
            f"  at {n:>5} scenes, {present:>3} present per block: "
            f"{per:5.2f} GiB per block, {total:6.1f} GiB across {slots} slots"
            f"{verdict}"
        )
    (args.out_dir / "blocks.json").write_text(
        json.dumps(
            {"chunk_px": chunk, "rows": rows, "cols": cols, "raster": [height, width]},
            indent=2,
        )
    )
    say(f"\nplan written  {args.out_dir / 'blocks.json'}")
    return 0


def main(argv=None) -> int:  # noqa: C901, PLR0912, PLR0915
    args = parse_args(argv)
    bbox, tile_id = resolve_area(args)
    res = 1.0 / args.pixels_per_degree
    height, width = composite.raster_shape(bbox, args.pixels_per_degree)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def say(line: str = "") -> None:
        prefix = REHEARSAL_TAG if args.rehearse else ""
        for part in line.split("\n"):
            print(prefix + part)

    say(f"bbox          {bbox}")
    say(f"grid          {args.crs} @ 1/{args.pixels_per_degree} deg")
    say(f"raster        {width} x {height} px  ({width * height / 1e6:.0f} Mpx)")

    if args.dry_run:
        return _dry_run(args, bbox, height, width, say)

    import numpy as np
    import psutil

    import frisky

    configure_read_env(args.source)
    # Before any span. Without this the driver records nothing, whatever the
    # workers do.
    observe.enable(args.tracing_capacity)
    slots = args.workers * args.threads_per_worker
    marks: dict[str, float] = {}
    t0 = time.perf_counter()
    t0_wall = time.time()

    # Before the inventory read and before the first GET. The mask depends on
    # the tile's bbox and on two artifacts, and on nothing this run computes,
    # so a tile it empties can be recorded without staging a single object.
    ged_provenance = check_mask_inputs(args, say)
    keep = gap = mask_counts = None
    if ged_provenance is not None:
        with observe.phase("masks", marks=marks):
            keep, gap, mask_counts = masks.output_mask(
                bbox,
                args.pixels_per_degree,
                numobs_uri=args.numobs_uri,
                land_geometry_uri=args.land_geometry_uri,
            )
        say(
            f"              {mask_counts['pixels_kept'] / mask_counts['pixels_total']:.1%} "
            f"of the tile is land: {mask_counts['pixels_water']:,} px sea, "
            f"{mask_counts['pixels_emissivity_gap_on_land']:,} px inside the "
            f"ASTER gap region over land"
        )
        if not mask_counts["pixels_kept"]:
            return no_unmasked_pixels(args, tile_id, bbox, mask_counts, ged_provenance)

    t_search = time.perf_counter()
    if args.rehearse:
        items, item_bboxes = composite.rehearsal_items(
            bbox, args.rehearse, args.out_dir / "rehearsal-scenes"
        )
        run_provenance = {"source": "rehearsal, synthetic scenes", "synthetic": True}
    else:
        if tile_id is None:
            raise SystemExit(
                "a real run needs --tile: the inventory is addressed by tile. "
                "Use --bbox only with --dry-run or --rehearse."
            )
        items, item_bboxes, run_provenance = load_tile_items(args, tile_id)
    t_search = time.perf_counter() - t_search
    say(f"scenes        {len(items)} from the inventory in {t_search:.2f}s")
    if not items:
        say("no scenes matched")
        return 1
    item_dicts = items

    # L2SR products carry no thermal band, load as fill, and reach neither the
    # percentile nor the monthly counts. Dropping them is output-neutral and
    # buys back a layer on the time axis of every block.
    dropped_no_thermal = 0
    if not args.rehearse and not args.keep_scenes_without_thermal:
        item_dicts, item_bboxes, dropped_no_thermal = (
            staging.drop_scenes_without_thermal(item_dicts, item_bboxes)
        )
        if dropped_no_thermal:
            say(
                f"              {dropped_no_thermal} of {len(items)} carry no "
                f"thermal band; dropped"
            )
        if not item_dicts:
            return no_thermal_coverage(
                args, tile_id, bbox, len(items), dropped_no_thermal, run_provenance
            )

    prep = (
        None
        if args.rehearse
        else load_tile_prep(args, tile_id, item_dicts, run_provenance)
    )
    if prep is not None:
        kept = destripe.keep_mask(
            np.array([prep.offset.get(s, np.nan) for s in prep.offset]),
            np.array([prep.n_valid.get(s, 0) for s in prep.offset]),
            floor=destripe.DESTRIPE_MIN_PREP_SAMPLES,
            max_offset_c=args.max_offset_c,
        )
        say(
            f"prep          {len(prep.paths)} WRS paths, "
            f"{1.0 - kept.mean():.1%} of scenes rejected at "
            f"{args.max_offset_c:g} C"
        )
        say(
            f"              destripe {'off' if args.no_destripe else 'on'}, "
            f"feather {'off' if args.no_feather else 'on'}"
        )
        if not args.no_destripe and not kept.any():
            return no_scene_survives_destriping(args, tile_id, prep, run_provenance)
    elif not args.rehearse:
        say(
            "prep          none: pooled percentile, no scene offsets. The WRS "
            "seam stays in the output"
        )

    depths = composite.block_depths(
        bbox, args.pixels_per_degree, args.chunk, item_bboxes
    )
    flat = sorted(int(d) for d in depths.ravel())
    say(
        f"blocks        {depths.size} of {args.chunk}x{args.chunk} px, "
        f"scenes/block min {flat[0]} p50 {flat[len(flat) // 2]} "
        f"p95 {flat[int(len(flat) * 0.95)]} max {flat[-1]}"
    )
    memory_demand_gib = None
    if not args.rehearse and not args.force:
        memory_demand_gib = composite.memory_guard(
            args.chunk, len(item_dicts), depths, slots
        )
        say(
            f"memory        {memory_demand_gib:.1f} GiB demanded across "
            f"{slots} slots (DERIVED), fits"
        )

    sampler = MemorySampler(args.out_dir / "memory.csv", args.sample_interval)
    sampler.start()

    with observe.phase("stage", count=len(item_dicts), marks=marks):
        stage_report = stage_scenes_for(args, item_dicts, say)

    with observe.phase("cluster", marks=marks):
        cluster = frisky.LocalCluster(
            n_workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            processes=True,
            memory_limit=int(args.memory_limit_gib * GIB),
            dashboard_address="127.0.0.1:0",
            silence_summary=True,
        )
        client = cluster.get_client()
        # Import the read stack on every worker before the first real task.
        # See `composite.warm_worker` for the import race this closes.
        n_warm = composite.warm_workers(client, cluster)
    dash = observe.dashboard_url(cluster)
    say(
        f"cluster       {args.workers} workers x {args.threads_per_worker} threads, "
        f"{n_warm} processes warmed in {marks['cluster_s']:.1f}s"
    )
    say(f"dashboard     {dash}")

    scalars: dict = {}
    staged_paths: dict = {}
    trace_report: dict = {}
    proc = psutil.Process()
    try:
        with observe.phase("graph_build", count=depths.size, marks=marks):
            out = composite.build_graph(
                item_dicts,
                bbox,
                crs=args.crs,
                resolution=res,
                chunk=args.chunk,
                prep=prep,
                max_offset_c=args.max_offset_c,
                debias=not args.no_destripe,
                feather=not args.no_feather,
                emit_pooled=args.emit_pooled,
            )
            ydim, xdim = composite.spatial_dims(args.crs)
            counts, staged_paths = composite.staging_writes(
                out,
                args.out_dir,
                crs=args.crs,
                dims=(ydim, xdim),
                keep_mask=keep,
                gap_mask=gap,
            )
            n_tasks = len(dict(counts.__dask_graph__()))
        say(
            f"graph         {n_tasks:,} tasks for {depths.size} blocks over "
            f"{out.attrs['n_scenes']} scenes, built in {marks['graph_build_s']:.1f}s"
        )
        if out.attrs["n_rejected"]:
            say(
                f"              {out.attrs['n_rejected']} scenes rejected by the offset rule"
            )

        with observe.phase("compute", count=depths.size, marks=marks):
            scalars = composite.compute_all(counts)
        say(f"compute       {marks['compute_s']:.1f}s for {depths.size} blocks")

        with observe.phase("collect_trace", marks=marks):
            trace_report = observe.collect(cluster, args.out_dir)
    finally:
        # Close the client before the cluster, so no other thread's bare
        # compute resolves to a half-dead client. Without a close here a
        # failure above hangs the process on a live cluster.
        client.close()
        cluster.close()
        sampler.stop()

    if stage_report is not None and not args.keep_staged:
        staging.cleanup(args.stage_dir, owned=stage_report.get("owns_stage_dir", True))

    memory_peak = sampler.peak_between(0.0, time.monotonic())
    workers_gib = memory_peak.get("workers_rss_peak_mb", 0.0) / 1024
    tree_gib = memory_peak.get("tree_rss_peak_mb", 0.0) / 1024

    if mask_counts is not None:
        mask_counts |= {
            "scope": "tile",
            "valid_removed_by_water": int(scalars.get("removed_water", 0)),
            "valid_removed_by_emissivity": int(scalars.get("removed_hot", 0)),
            "valid_removed_by_mask": int(scalars.get("removed_water", 0))
            + int(scalars.get("removed_hot", 0)),
            "qa_count_pixels_zeroed": int(scalars.get("qa_zeroed", 0)),
        }
        say(
            f"masked        {mask_counts['valid_removed_by_water']:,} px sea, "
            f"{mask_counts['valid_removed_by_emissivity']:,} px hot inside the "
            f"ASTER gap region"
        )

    # The header fields the workers could not write: scale, offset, band
    # names, and the statistics, scanned off the staging file in strips.
    with observe.phase("cog", marks=marks):
        lst_statistics = composite.finish_staging(
            staged_paths["lst_p95"],
            scale=LST_SCALE,
            offset=LST_OFFSET,
            descriptions=("95th percentile LST",),
            nodata=LST_NODATA_DN,
        )
        qa_statistics = composite.finish_staging(
            staged_paths["qa_count"],
            scale=None,
            offset=None,
            descriptions=tuple(MONTH_NAMES),
            nodata=None,
        )
        meta = run_meta(
            args,
            bbox,
            height,
            width,
            mask_rule(args, mask_counts, ged_provenance),
            correction_rule(args, prep),
        )
        catalog_root = None
        if args.no_catalog:
            for name, nodata, scale, offset in (
                ("lst_p95", LST_NODATA_DN, LST_SCALE, LST_OFFSET),
                ("qa_count", None, None, None),
            ):
                write_cog(
                    args.out_dir / f"{name}.tif",
                    bbox=bbox,
                    pixels_per_degree=args.pixels_per_degree,
                    crs=args.crs,
                    nodata=nodata,
                    scale=scale,
                    offset=offset,
                    source_path=staged_paths[name],
                )
        else:
            check_catalog_inputs(meta)
            catalog_root = write_catalog(
                args.catalog_dir or (args.out_dir / "catalog"),
                staged_paths["lst_p95"],
                staged_paths["qa_count"],
                meta,
                collection_id=args.collection_id,
                host_name=args.host_name,
                host_url=args.host_url,
                license_id=args.license,
            )
        if "lst_p95_pooled" in staged_paths:
            composite.finish_staging(
                staged_paths["lst_p95_pooled"],
                scale=LST_SCALE,
                offset=LST_OFFSET,
                descriptions=("pooled 95th percentile LST",),
                nodata=LST_NODATA_DN,
            )
            write_cog(
                args.out_dir / "lst_p95_pooled.tif",
                bbox=bbox,
                pixels_per_degree=args.pixels_per_degree,
                crs=args.crs,
                nodata=LST_NODATA_DN,
                scale=LST_SCALE,
                offset=LST_OFFSET,
                source_path=staged_paths["lst_p95_pooled"],
            )
        composite.cleanup_staging(staged_paths)

    lst_stats = lst_statistics[0]
    valid_fraction = (
        float(lst_stats["kept"]) / float(lst_stats["total"])
        if lst_stats["total"]
        else 0.0
    )
    marks |= {
        "t0_wall": t0_wall,
        "search_s": t_search,
        "wall_s": time.perf_counter() - t0,
    }
    summary = {
        "synthetic": bool(args.rehearse),
        "tile": tile_id,
        "bbox": bbox,
        "crs": args.crs,
        "pixels_per_degree": args.pixels_per_degree,
        "raster": [height, width],
        "chunk_px": args.chunk,
        "n_blocks": int(depths.size),
        "scenes_per_block": {
            "min": flat[0],
            "p50": flat[len(flat) // 2],
            "p95": flat[int(len(flat) * 0.95)],
            "max": flat[-1],
        },
        "n_scenes": len(item_dicts),
        "n_scenes_inventory": len(items),
        "n_scenes_rejected": out.attrs["n_rejected"],
        "scenes_dropped_no_thermal": dropped_no_thermal,
        "n_tasks": n_tasks,
        # Every phase in seconds from one origin, MEASURED. The same figures
        # went to frisky as client phases.
        "phases": marks,
        "memory_demand_gib_derived": memory_demand_gib,
        "client_rss_peak_gib": proc.memory_info().rss / GIB,
        "workers_rss_peak_gib": workers_gib,
        "tree_rss_peak_gib": tree_gib,
        "valid_fraction": valid_fraction,
        "n_pooled_fallback": int(scalars.get("fallback", 0)),
        "inventory": run_provenance,
        "staging": stage_report,
        "mask": (
            None
            if mask_counts is None
            else mask_counts
            | {
                "numobs_uri": str(args.numobs_uri),
                "land_geometry_uri": str(args.land_geometry_uri),
                "aster_ged": ged_provenance,
            }
        ),
        "correction": correction_rule(args, prep),
        "catalog": str(catalog_root) if catalog_root else None,
        "frisky": trace_report,
        "dashboard": dash,
    }
    if lst_stats["kept"]:
        summary |= {
            "min_c": float(lst_stats["min"]) * LST_SCALE + LST_OFFSET,
            "mean_c": float(lst_stats["mean"]) * LST_SCALE + LST_OFFSET,
            "max_c": float(lst_stats["max"]) * LST_SCALE + LST_OFFSET,
        }
        say(
            f"\nLST p95       min {summary['min_c']:.1f} C  "
            f"mean {summary['mean_c']:.1f} C  max {summary['max_c']:.1f} C  "
            f"({100 * valid_fraction:.1f}% valid)"
        )
    qa_mean = {MONTHS[i]: float(qa_statistics[i]["mean"]) for i in range(12)}
    summary["qa_count_per_month"] = qa_mean
    say("qa_count      " + "  ".join(f"{m} {v:.1f}" for m, v in qa_mean.items()))
    say(
        "phases        "
        + ", ".join(
            f"{k[:-2]} {v:.1f}s"
            for k, v in marks.items()
            if k.endswith("_s") and k not in ("wall_s", "search_s")
        )
    )
    say(f"client RSS    {summary['client_rss_peak_gib']:.2f} GiB peak")
    if workers_gib:
        say(f"worker RSS    {workers_gib:.2f} GiB peak (MEASURED)")
    say(
        f"frisky        {trace_report.get('n_spans', 0):,} spans, "
        f"{trace_report.get('n_events', 0):,} events -> spans.json, events.json, "
        f"overview.json"
    )

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    if stage_report is not None:
        (args.out_dir / "staging.json").write_text(
            json.dumps(
                stage_report
                | {
                    "scenes_dropped_no_thermal": dropped_no_thermal,
                    "seconds_in_run": marks.get("stage_s"),
                },
                indent=2,
            )
        )
    say(f"artifacts     {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

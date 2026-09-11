# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "frisky>=0.7.2",
#   "dask",
#   "odc-stac",
#   "pystac-client",
#   "planetary-computer",
#   "xarray",
#   "numpy",
#   "geopandas",
#   "psutil",
#   "rich",
# ]
# ///
"""Profile the Pergamino Landsat p95 LST composite, stage by stage.

Reads no cache and writes no raster. The composite is computed, measured, and
discarded. Every stage is timed and its memory is sampled, in the client process
and in each frisky worker process.

Run it:

    uv run profile_lst_p95.py --max-scenes 40 --out-dir ./smoke
    uv run profile_lst_p95.py --out-dir ./full

Output encoding, per the asset spec:

    lst_p95    uint16, scale 0.01, offset -50.0, nodata 0, celsius
    qa_count   uint8, 12 bands Jan..Dec, no nodata

    celsius = dn * 0.01 + (-50.0)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import destripe
from lst_qa import (
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    encode_celsius_xr,
    masked_celsius_xr,
)
from memory_sampler import MemorySampler
from stac_window import DEFAULT_END, DEFAULT_START, datetime_range

# --------------------------------------------------------------------------
# Constants from the source data and the encoding contract.
# --------------------------------------------------------------------------

BOUNDARY_URL = "https://raw.githubusercontent.com/nlebovits/landsat-lst/main/data/pergamino_dept.gpkg"
STAC_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
SOURCES = {
    # Azure blob, West Europe. Assets need a SAS token, valid about an hour.
    "planetary-computer": STAC_PLANETARY_COMPUTER,
    # s3://usgs-landsat, us-west-2, requester pays. Needs AWS credentials.
    "earth-search": STAC_EARTH_SEARCH,
}
COLLECTION = "landsat-c2-l2"

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MB = 1024.0 * 1024.0
GIB = 1024.0**3


# --------------------------------------------------------------------------
# Memory sampler. Runs in its own process, and lives in `memory_sampler.py`
# because `shard_lst_p95.py` needs the same worker RSS this collects.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Stage instrumentation.
#
# frisky has no public span context manager. record_span and now_ns are the
# semi-public primitives, so this builds on those and adds the host-side
# numbers frisky does not collect.
# --------------------------------------------------------------------------

STAGES: list[dict] = []
_TRACE_ID = 0
_USE_TRACEMALLOC = True


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / MB
    except Exception:
        return float("nan")


def _ru_maxrss_mb() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _record_frisky_span(name: str, t0_ns: int, t1_ns: int, keys=None) -> None:
    """Best effort. Instrumentation must never fail the run."""
    try:
        import frisky

        frisky.record_span(name, t0_ns, t1_ns, trace_id=_TRACE_ID, keys=keys)
    except Exception:
        pass


def _now_ns() -> int:
    try:
        import frisky

        return int(frisky.now_ns())
    except Exception:
        return time.time_ns()


@contextmanager
def stage(name: str, **meta):
    """Time one stage, measure what it cost, and emit a frisky span for it."""
    record = {"stage": name, "meta": meta}
    STAGES.append(record)

    if _USE_TRACEMALLOC:
        try:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            tracemalloc.reset_peak()
        except Exception:
            pass

    record["t_mono_start"] = time.monotonic()
    record["rss_start_mb"] = _rss_mb()
    record["ru_maxrss_start_mb"] = _ru_maxrss_mb()
    t0_ns = _now_ns()
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()
    ok = True
    try:
        yield record
    except BaseException:
        ok = False
        raise
    finally:
        wall1 = time.perf_counter_ns()
        cpu1 = time.process_time_ns()
        t1_ns = _now_ns()
        record["t_mono_end"] = time.monotonic()
        record["wall_s"] = (wall1 - wall0) / 1e9
        record["cpu_s"] = (cpu1 - cpu0) / 1e9
        record["rss_end_mb"] = _rss_mb()
        record["rss_delta_mb"] = record["rss_end_mb"] - record["rss_start_mb"]
        record["ru_maxrss_end_mb"] = _ru_maxrss_mb()
        record["ru_maxrss_delta_mb"] = (
            record["ru_maxrss_end_mb"] - record["ru_maxrss_start_mb"]
        )
        record["ok"] = ok
        if _USE_TRACEMALLOC:
            try:
                cur, peak = tracemalloc.get_traced_memory()
                record["pyheap_current_mb"] = cur / MB
                record["pyheap_peak_mb"] = peak / MB
            except Exception:
                pass
        _record_frisky_span(
            f"profile.{name}",
            t0_ns,
            t1_ns,
            keys=[f"{k}={v}" for k, v in meta.items()] or None,
        )


# --------------------------------------------------------------------------
# Pipeline pieces.
# --------------------------------------------------------------------------


def set_gdal_env(
    gdal_threads: str = "1",
    ingested_bytes: int = 32768,
    http_version: str = "2",
    extra: str = "",
    requester_pays: bool = False,
) -> dict[str, str]:
    """GDAL settings for cloud reads.

    odc.loader.configure_rio accepts a `client` argument and discards it, and
    frisky workers start with spawn. So the only route into a worker is the
    environment, and it has to be set before the cluster is built.

    Every value here is a throughput knob, which is why they are arguments.
    """
    cfg = {
        # Never list the container to open one blob.
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # Coalesce the tile requests for one block into a single HTTP call.
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        # Pull the header and IFD in the first request instead of walking back
        # to the server two or three times before any pixel arrives.
        "GDAL_INGESTED_BYTES_AT_OPEN": str(ingested_bytes),
        # HTTP/2 multiplexes range requests over one connection.
        "GDAL_HTTP_VERSION": http_version,
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
        "CPL_VSIL_CURL_CACHE_SIZE": str(256 * 1024 * 1024),
        "GDAL_NUM_THREADS": str(gdal_threads),
    }
    if requester_pays:
        # usgs-landsat bills the reader. GDAL needs this on every request, and
        # boto/rasterio need a region to resolve the bucket endpoint.
        cfg["AWS_REQUEST_PAYER"] = "requester"
        cfg.setdefault("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-west-2"))
    for pair in extra.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            cfg[k.strip()] = v.strip()
    os.environ.update(cfg)
    return cfg


def fetch_boundary(
    out_dir: Path, url: str
) -> tuple[tuple[float, float, float, float], Path]:
    """Download the department polygon and return its WGS84 bounds."""
    import geopandas as gpd

    dest = out_dir / "pergamino_dept.gpkg"
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())

    gdf = gpd.read_file(dest).to_crs("EPSG:4326")
    bbox = tuple(float(v) for v in gdf.total_bounds)
    # tuple(...) over a 4-element sequence widens to tuple[float, ...].
    return (bbox[0], bbox[1], bbox[2], bbox[3]), dest


# Measured on this host: a frisky worker with GDAL, numpy, and xarray loaded
# sits near 0.7 GiB before it holds any pixel data. Ignoring that floor makes
# the prediction optimistic by several GiB once there are eight of them.
WORKER_BASELINE_GIB = 0.7


def predict_working_set(
    chunk: int, n_scenes: int, concurrency: int, n_workers: int
) -> dict[str, float]:
    """Memory the p95 rechunk needs, per task and in aggregate.

    quantile collapses the time axis into a single chunk. That rechunk, not the
    read, is what sets the peak: every task holds chunk * chunk * n_scenes
    float32 values at once.
    """
    per_task = chunk * chunk * n_scenes * 4 / GIB
    # The sort inside nanquantile needs a second copy of the block.
    per_task_sort = per_task * 2
    data = per_task_sort * concurrency
    baseline = WORKER_BASELINE_GIB * n_workers
    return {
        "chunk": chunk,
        "n_scenes": n_scenes,
        "concurrency": concurrency,
        "per_task_gib": per_task,
        "per_task_with_sort_gib": per_task_sort,
        "data_gib": data,
        "worker_baseline_gib": baseline,
        "aggregate_gib": data + baseline,
    }


def build_graph(
    items,
    bbox,
    chunk: int,
    crs: str,
    # A geographic grid sets this to 1.0 / pixels_per_degree, so it is not
    # an integer. The annotation said int until ty caught the callers.
    resolution: float,
    time_chunk: int,
    load_chunk: int | None = None,
    correction=None,
):
    """Lazy graph: load, mask, convert, reduce to p95, count valid per month.

    Reads and the percentile want opposite block sizes.

    Reads want big blocks. The source is UTM, the destination is not, so every
    destination block needs a skewed source window plus an edge halo, and
    neighbouring blocks re-fetch the same source tiles. Measured on 24 scenes:
    256 px blocks moved ~591 MB, 1024 px blocks moved ~236 MB for identical
    output. That is 2.5x of pure waste.

    The percentile wants small blocks. quantile collapses the time axis into
    one chunk, so each task holds chunk * chunk * n_scenes float32 values. At
    673 scenes a 1024 block is 5.1 GiB per task; a 256 block is 0.17 GiB.

    So load big and reduce small.
    """
    from odc.geo import CRS
    from odc.stac import stac_load

    load_chunk = load_chunk or chunk

    # odc names the spatial dims after the output CRS: x/y when projected,
    # longitude/latitude when geographic. Getting this wrong is silent on the
    # chunks argument (the keys are simply ignored, so the array loads
    # unchunked) and loud on .chunk() (ValueError). Derive it, never assume.
    ydim, xdim = ("y", "x") if CRS(crs).projected else ("latitude", "longitude")

    data = stac_load(
        items,
        bands=("lwir11", "qa_pixel"),
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        groupby="landsat:scene_id",
        # Spatial chunking is mandatory. Left unchunked, the rechunk that
        # quantile forces would put the whole time stack in one block.
        chunks={"time": time_chunk, xdim: load_chunk, ydim: load_chunk},
    )
    missing = {ydim, xdim} - set(data.dims)
    if missing:
        msg = f"expected spatial dims {ydim}/{xdim}, got {tuple(data.dims)}"
        raise RuntimeError(msg)

    # One definition of a usable observation, shared with the sharded path in
    # shard_lst_p95. Three reasons a pixel is unusable: the band is source
    # fill, QA_PIXEL bits 1 to 5 flag dilated cloud, cirrus, cloud, shadow or
    # snow, or the decoded temperature falls outside [-50, 80] C. The last one
    # is what removes reprojected scene edges, where interpolation against the
    # DN 0 fill leaves small nonzero values decoding near -124 C that an exact
    # fill comparison cannot see. Every rejection lands here, before quantile,
    # because a value that reaches the percentile has already moved the answer.
    lst_c = masked_celsius_xr(data["lwir11"], data["qa_pixel"])

    # Split the read blocks down before the time rechunk. This is a pure
    # slice, no shuffle, so it costs nothing on the wire.
    if load_chunk != chunk:
        lst_c = lst_c.chunk({xdim: chunk, ydim: chunk})

    # The seam corrections, if this run was given a tile prep artifact. Both
    # happen on the stack already described by the graph, so neither adds a
    # source pass: de-biasing is a per-scene scalar, and the per-path quantiles
    # reduce disjoint subsets of the same rechunked stack. Rechunking a subset
    # here would give the scheduler two incompatible consumers and it would hold
    # the whole stack rather than stream it.
    feathered = None
    if correction is not None:
        lst_c, labels, _rejected = destripe.apply_to_stack_xr(lst_c, items, correction)
        if correction["paths"]:
            feathered = destripe.feathered_quantile_xr(
                lst_c, labels, correction["paths"], correction["weight"], (ydim, xdim)
            )

    if feathered is None:
        lst_p95 = lst_c.quantile(0.95, dim="time")
    else:
        lst_p95 = feathered
    lst_p95 = lst_p95.astype("float32")
    # quantile leaves a scalar `quantile` coord behind; it would become a
    # stray dimension on the way out.
    lst_p95 = lst_p95.drop_vars("quantile", errors="ignore")

    # Calendar-month climatology, so every year in the window pools into its
    # month bucket. reindex guarantees 12 bands even for a month with no scenes.
    qa_count = (
        lst_c.notnull()
        .groupby("time.month")
        .sum()
        .reindex(month=list(range(1, 13)), fill_value=0)
        .astype("uint8")
    )

    lst_u16 = encode_uint16(lst_p95)
    return lst_u16, qa_count, lst_p95


def encode_uint16(celsius):
    """Celsius to DN: dn = (c - offset) / scale, with 0 reserved for nodata.

    Thin wrapper over the shared encoder, so this module and shard_lst_p95
    write the same DN for the same temperature. Out-of-range and non-finite
    values become nodata rather than clipping: a clipped -124 C would arrive as
    a believable -49.99 C, which is worse than a gap. The floor is
    `LST_MIN_TRUSTED_DN`, not DN 1, because DN 1 is reachable only from the
    encoding floor itself and marks a failed retrieval.
    """
    return encode_celsius_xr(celsius)


def graph_stats(*objs) -> dict:
    """Task counts by key prefix, before and after fusion.

    This is expensive and it does not scale. Counting materialises the whole
    graph, and dask.optimize walks it again. At roughly 44k tasks that costs a
    few seconds. At a quarter of a 5 degree tile, around 500k tasks, it held one
    core for more than six minutes without finishing. Turn it off with
    --no-graph-stats for anything larger than a small area.
    """
    from collections import Counter

    def count(obj) -> Counter:
        try:
            graph = obj.__dask_graph__()
            if graph is None:
                return Counter()
            counter: Counter = Counter()
            for key in graph:
                name = key[0] if isinstance(key, tuple) else key
                counter[str(name).split("-")[0]] += 1
            return counter
        except Exception:
            return Counter()

    raw: Counter = Counter()
    for obj in objs:
        raw += count(obj)

    opt: Counter = Counter()
    try:
        import dask

        for obj in dask.optimize(*objs):
            opt += count(obj)
    except Exception:
        pass

    raw_total = sum(raw.values())
    opt_total = sum(opt.values())
    return {
        "raw_tasks": raw_total,
        "optimized_tasks": opt_total,
        "fusion_ratio": (raw_total / opt_total) if opt_total else None,
        "raw_by_prefix": dict(raw.most_common(30)),
        "optimized_by_prefix": dict(opt.most_common(30)),
    }


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------


def collect_frisky(
    out_dir: Path, dashboard_url: str, limit: int, dump_limit: int
) -> dict:
    """Pull spans off the scheduler dashboard and render every view of them.

    Span volume scales with task count, and task count scales with scenes. A
    12-scene run already emits ~76k spans. The summaries run over everything;
    the verbatim dumps are capped at dump_limit, keeping the longest spans,
    because a multi-gigabyte trace.json helps nobody.
    """
    import frisky

    summary: dict = {"dashboard": dashboard_url}

    spans = []
    try:
        # query_spans, not get_spans: with processes=True the worker spans live
        # in the worker processes, and get_spans only drains this one.
        spans = frisky.query_spans(
            limit=limit, dashboard_url=dashboard_url, request_timeout=60
        )
    except Exception as exc:
        summary["query_spans_error"] = repr(exc)

    if not spans:
        try:
            spans = list(frisky.get_spans())
            summary["span_source"] = "get_spans (fallback)"
        except Exception as exc:
            summary["get_spans_error"] = repr(exc)
    else:
        summary["span_source"] = "query_spans"

    summary["n_spans"] = len(spans)

    dumped = spans
    if len(spans) > dump_limit:
        dumped = sorted(spans, key=lambda s: -(s.get("duration_ns") or 0))[:dump_limit]
        dumped.sort(key=lambda s: s.get("start_ns") or 0)
        summary["dump_truncated_to"] = dump_limit
        summary["dump_rule"] = "longest spans by duration_ns"
    summary["n_spans_dumped"] = len(dumped)

    # Stream it. json.dumps on a million spans builds the whole string first.
    with open(out_dir / "spans.json", "w") as fh:
        json.dump(dumped, fh, default=str)

    if spans:
        try:
            summary["by_name"] = frisky.summarize_spans(spans)
        except Exception as exc:
            summary["summarize_error"] = repr(exc)
        try:
            from frisky.tracing import worker_outliers

            summary["worker_outliers"] = worker_outliers(spans, top=10)
        except Exception as exc:
            summary["outlier_error"] = repr(exc)
        try:
            n = frisky.export_chrome_trace(dumped, str(out_dir / "trace.json"))
            summary["chrome_events"] = n
        except Exception as exc:
            summary["chrome_error"] = repr(exc)
        for view in ("component", "worker", "detailed"):
            try:
                text = frisky.render_timeline(dumped, view=view, width=100)
                (out_dir / f"timeline-{view}.txt").write_text(text)
            except Exception as exc:
                summary[f"timeline_{view}_error"] = repr(exc)

    try:
        (out_dir / "metrics.json").write_text(
            json.dumps(frisky.get_metrics(), indent=2, default=str)
        )
    except Exception as exc:
        summary["metrics_error"] = repr(exc)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def print_stage_table(sampler: MemorySampler) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Stage profile", header_style="bold")
    table.add_column("stage")
    table.add_column("wall s", justify="right")
    table.add_column("cpu s", justify="right")
    table.add_column("cpu %", justify="right")
    table.add_column("RSS start", justify="right")
    table.add_column("RSS delta", justify="right")
    table.add_column("tree peak", justify="right")
    table.add_column("net MB", justify="right")
    table.add_column("MB/s", justify="right")

    total = 0.0
    for rec in STAGES:
        if "wall_s" not in rec:
            continue
        total += rec["wall_s"]
        peak = sampler.peak_between(rec["t_mono_start"], rec["t_mono_end"])
        rec["memory_window"] = peak
        wall = rec["wall_s"]
        table.add_row(
            rec["stage"] + ("" if rec.get("ok", True) else " [FAILED]"),
            f"{wall:,.2f}",
            f"{rec['cpu_s']:,.2f}",
            f"{100 * rec['cpu_s'] / wall:,.0f}" if wall > 0 else "-",
            f"{rec['rss_start_mb']:,.0f}M",
            f"{rec['rss_delta_mb']:+,.0f}M",
            f"{peak.get('tree_rss_peak_mb', float('nan')):,.0f}M" if peak else "-",
            f"{peak.get('net_recv_mb', 0.0):,.0f}" if peak else "-",
            f"{peak.get('net_recv_mb_s', 0.0):,.1f}" if peak else "-",
        )
    table.add_row("TOTAL", f"{total:,.2f}", "", "", "", "", "", "", "", style="bold")
    Console().print(table)


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Profile the Pergamino Landsat p95 LST composite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=Path("./profile-run"))
    p.add_argument(
        "--chunk",
        type=int,
        default=256,
        help="spatial block for the p95 reduction; drives peak memory",
    )
    p.add_argument(
        "--load-chunk",
        type=int,
        default=1024,
        help="spatial block for the COG reads; bigger cuts warp-halo refetching",
    )
    p.add_argument("--time-chunk", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument(
        "--memory-limit-gib",
        type=float,
        default=4.0,
        help="per worker; workers * this is the cluster total",
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--cloud-cover-lt", type=int, default=100)
    p.add_argument(
        "--platforms",
        default="landsat-8,landsat-9",
        help=(
            "comma separated, or 'all'. Landsat 7 carries no lwir11 band "
            "(its thermal band is lwir / ST_B6) and has been SLC-off since 2003, "
            "so it contributes nothing to an lwir11 query"
        ),
    )
    p.add_argument("--crs", default="epsg:3857")
    p.add_argument(
        "--resolution",
        type=float,
        default=30.0,
        help="pixel size in CRS units: metres for a projected CRS, degrees for "
        "a geographic one. Prefer --pixels-per-degree on a geographic grid",
    )
    p.add_argument(
        "--pixels-per-degree",
        type=int,
        default=None,
        help="geographic grids only. Sets resolution to 1/N, which keeps every "
        "tile a whole number of pixels. The 5 degree grid uses 3600",
    )
    p.add_argument("--max-scenes", type=int, default=None, help="cap for a smoke run")
    p.add_argument("--sample-interval", type=float, default=0.1)
    p.add_argument("--gdal-threads", default="1", help="GDAL_NUM_THREADS, or ALL_CPUS")
    p.add_argument(
        "--ingested-bytes",
        type=int,
        default=32768,
        help="GDAL_INGESTED_BYTES_AT_OPEN; header bytes grabbed on open",
    )
    p.add_argument("--http-version", default="2", help="GDAL_HTTP_VERSION")
    p.add_argument("--gdal-extra", default="", help="extra GDAL config, K=V,K=V")
    p.add_argument("--tracing-capacity", type=int, default=10_000_000)
    p.add_argument(
        "--span-limit",
        type=int,
        default=2_000_000,
        help="max spans to pull from frisky",
    )
    p.add_argument(
        "--span-dump-limit",
        type=int,
        default=300_000,
        help="max spans written verbatim to spans.json and trace.json",
    )
    p.add_argument("--boundary-url", default=BOUNDARY_URL)
    p.add_argument(
        "--bbox",
        default=None,
        help="west,south,east,north in EPSG:4326. Overrides the boundary file. "
        "Use it to run a grid tile instead of the department, e.g. the 5 degree "
        "tile S30W065 is -65,-35,-60,-30.",
    )
    p.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="planetary-computer",
        help="earth-search reads s3://usgs-landsat, which is requester pays",
    )
    p.add_argument(
        "--no-tracemalloc", action="store_true", help="python-heap tracking off"
    )
    p.add_argument(
        "--compare-passes",
        action="store_true",
        help="also time the two-call variant, to price the extra pass",
    )
    p.add_argument(
        "--no-graph-stats",
        action="store_true",
        help="skip task counting and dask.optimize. Both materialise the whole "
        "graph and do not scale; required above roughly 200k tasks",
    )
    p.add_argument(
        "--force", action="store_true", help="run even if predicted to overflow"
    )
    return p.parse_args(argv)


# The CLI entry point. Nine timed stages in sequence, each with its own
# fallback. The sequence is the measurement, so it stays in one function.
def main(argv=None) -> int:  # noqa: C901
    global _USE_TRACEMALLOC, _TRACE_ID

    args = parse_args(argv)
    _USE_TRACEMALLOC = not args.no_tracemalloc

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pixels_per_degree:
        args.resolution = 1.0 / args.pixels_per_degree

    # A degree is not a metre. Passing --resolution 30 with a geographic CRS
    # asks for 30-degree pixels; passing 1/3600 with a projected one asks for
    # sub-millimetre pixels. Both build a graph and fail later, expensively.
    from odc.geo import CRS as _CRS

    _projected = _CRS(args.crs).projected
    if _projected and args.resolution < 0.01:
        msg = (
            f"--crs {args.crs} is projected, so --resolution {args.resolution} "
            f"means {args.resolution} metres. Did you mean --pixels-per-degree?"
        )
        raise SystemExit(msg)
    if not _projected and args.resolution > 1.0:
        msg = (
            f"--crs {args.crs} is geographic, so --resolution {args.resolution} "
            f"means {args.resolution} DEGREES. Use --pixels-per-degree 3600 "
            f"for the 5 degree tile grid."
        )
        raise SystemExit(msg)

    concurrency = args.workers * args.threads_per_worker
    budget_gib = args.workers * args.memory_limit_gib

    # Both of these must be in the environment before the cluster is built.
    # Workers spawn, and they read their configuration from os.environ.
    gdal_cfg = set_gdal_env(
        gdal_threads=args.gdal_threads,
        ingested_bytes=args.ingested_bytes,
        http_version=args.http_version,
        extra=args.gdal_extra,
        requester_pays=(args.source == "earth-search"),
    )
    os.environ["FRISKY_TRACING_CAPACITY"] = str(args.tracing_capacity)

    print(f"host          {platform.node()}  {os.cpu_count()} cores")
    print(f"source        {args.source}  {SOURCES[args.source]}")
    print(
        f"cluster       {args.workers} workers x {args.threads_per_worker} threads "
        f"x {args.memory_limit_gib} GiB = {budget_gib:.0f} GiB total"
    )
    print(
        f"chunking      read {args.load_chunk or args.chunk}, "
        f"reduce {args.chunk}, time {args.time_chunk}"
    )
    print(
        f"grid          {args.crs} @ {args.resolution:g}"
        f"{' m' if _projected else ' deg'}"
    )
    print(f"out-dir       {out_dir.resolve()}")
    print("raster output disabled: the composite is computed and discarded\n")

    sampler = MemorySampler(out_dir / "memory.csv", args.sample_interval)
    sampler.start()

    run: dict = {
        "config": vars(args) | {"out_dir": str(out_dir)},
        "gdal": gdal_cfg,
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "encoding": {
            "dtype": "uint16",
            "scale": LST_SCALE,
            "offset": LST_OFFSET,
            "nodata": LST_NODATA_DN,
            "units": "celsius",
            "decode": "celsius = dn * 0.01 + (-50.0)",
        },
    }
    cluster = None

    try:
        with stage("imports"):
            import dask  # noqa: F401
            import frisky
            import numpy as np
            import planetary_computer
            import pystac_client
            import xarray as xr  # noqa: F401
            from odc.stac import configure_rio

            frisky.enable_tracing(args.tracing_capacity)
            _TRACE_ID = int.from_bytes(os.urandom(8), "big") >> 1
            configure_rio(cloud_defaults=True)
            run["versions"] = {
                "frisky": frisky.__version__,
                "dask": dask.__version__,
                "xarray": xr.__version__,
                "numpy": np.__version__,
            }

        with stage("boundary", url=args.boundary_url) as rec:
            if args.bbox:
                bbox = tuple(float(v) for v in args.bbox.split(","))
                if len(bbox) != 4:
                    msg = "--bbox needs exactly four values: west,south,east,north"
                    raise ValueError(msg)
                rec["meta"]["source"] = "--bbox"
            else:
                bbox, path = fetch_boundary(out_dir, args.boundary_url)
                rec["meta"]["source"] = str(path)
                rec["meta"]["bytes"] = path.stat().st_size
            rec["meta"]["bbox"] = bbox
        print(f"bbox          {bbox}")
        run["bbox"] = bbox

        with stage("cluster_start", workers=args.workers) as rec:
            cluster = frisky.LocalCluster(
                n_workers=args.workers,
                threads_per_worker=args.threads_per_worker,
                processes=True,  # not the frisky default; we need per-worker RSS
                memory_limit=int(args.memory_limit_gib * GIB),  # BYTES, not "4GB"
                dashboard_address="127.0.0.1:0",
                silence_summary=True,
            )
            # Bound, not used: the client must outlive this block, and
            # frisky.Client(cluster) would raise.
            _client = cluster.get_client()
            dashboard = cluster.dashboard_address
            if not str(dashboard).startswith("http"):
                dashboard = f"http://{dashboard}"
            rec["meta"]["dashboard"] = dashboard
        print(f"dashboard     {dashboard}")
        run["dashboard"] = dashboard

        with stage("stac_search", cloud_lt=args.cloud_cover_lt) as rec:
            # Deliberately unsigned. Planetary Computer SAS tokens last about
            # an hour and the graph embeds whatever URL it was built with, so
            # signing happens later, as close to the compute as possible.
            catalog = pystac_client.Client.open(SOURCES[args.source])
            stac_query: dict = {"eo:cloud_cover": {"lt": args.cloud_cover_lt}}
            platforms = (
                []
                if args.platforms.strip().lower() == "all"
                else [p.strip() for p in args.platforms.split(",") if p.strip()]
            )
            if platforms:
                stac_query["platform"] = {"in": platforms}
            query = catalog.search(
                collections=[COLLECTION],
                bbox=bbox,
                datetime=datetime_range(args.start, args.end),
                query=stac_query,
            )
            items = list(query.items())
            if args.max_scenes:
                items = items[: args.max_scenes]
            rec["meta"]["n_items"] = len(items)
            rec["meta"]["platforms"] = platforms or "all"
            by_platform: dict[str, int] = {}
            for it in items:
                key = str(it.properties.get("platform"))
                by_platform[key] = by_platform.get(key, 0) + 1
            rec["meta"]["by_platform"] = by_platform
            missing = sum(1 for it in items if "lwir11" not in it.assets)
            rec["meta"]["items_without_lwir11"] = missing
        print(f"scenes        {len(items)}  {by_platform}")
        run["n_scenes"] = len(items)
        run["by_platform"] = by_platform
        if missing:
            print(
                f"WARNING       {missing} of {len(items)} scenes carry no lwir11 asset "
                f"and will load as nodata"
            )

        if not items:
            print("no scenes matched; nothing to profile")
            return 1

        pred = predict_working_set(args.chunk, len(items), concurrency, args.workers)
        run["prediction"] = pred | {"budget_gib": budget_gib}
        print(
            f"predicted     {pred['per_task_with_sort_gib']:.2f} GiB per task x "
            f"{concurrency} concurrent = {pred['data_gib']:.1f} GiB data "
            f"+ {pred['worker_baseline_gib']:.1f} GiB worker baseline "
            f"= {pred['aggregate_gib']:.1f} GiB (budget {budget_gib:.0f} GiB)"
        )
        if pred["aggregate_gib"] > budget_gib and not args.force:
            print(
                f"\nREFUSING TO START. The p95 rechunk is predicted to need "
                f"{pred['aggregate_gib']:.1f} GiB but the cluster budget is "
                f"{budget_gib:.0f} GiB.\n"
                f"Lower --chunk (try {max(128, args.chunk // 2)}), raise "
                f"--memory-limit-gib, or pass --force to run anyway."
            )
            return 2
        print()

        with stage("sign_assets", scenes=len(items), source=args.source) as rec:
            if args.source == "planetary-computer":
                for item in items:
                    planetary_computer.sign_inplace(item)
                rec["meta"]["signed"] = True
            else:
                # s3:// hrefs authenticate per request from the ambient AWS
                # credentials, so there is nothing to sign and no clock to beat.
                rec["meta"]["signed"] = False
            run["signed_at"] = time.time()
            rec["meta"]["signed_at"] = run["signed_at"]

        with stage("graph_build", chunk=args.chunk, scenes=len(items)) as rec:
            lst_u16, qa_count, lst_c = build_graph(
                items,
                bbox,
                args.chunk,
                args.crs,
                args.resolution,
                args.time_chunk,
                load_chunk=args.load_chunk,
            )
            rec["meta"]["load_chunk"] = args.load_chunk or args.chunk
            rec["meta"]["shape"] = list(lst_u16.shape)
            rec["meta"]["chunks"] = str(lst_u16.chunks)
        print(f"raster        {lst_u16.shape[1]} x {lst_u16.shape[0]} px")
        run["shape"] = list(lst_u16.shape)

        with stage("graph_optimize", skipped=args.no_graph_stats) as rec:
            if args.no_graph_stats:
                stats = {"skipped": True}
            else:
                stats = graph_stats(lst_u16, qa_count)
                rec["meta"].update(
                    {
                        k: stats[k]
                        for k in ("raw_tasks", "optimized_tasks", "fusion_ratio")
                    }
                )
        (out_dir / "graph.json").write_text(json.dumps(stats, indent=2, default=str))
        if args.no_graph_stats:
            print("tasks         not counted (--no-graph-stats)\n")
        else:
            print(
                f"tasks         {stats['raw_tasks']:,} raw, "
                f"{stats['optimized_tasks']:,} fused\n"
            )

        print("computing (this is the stage you care about)...")
        with stage("compute", scenes=len(items), chunk=args.chunk) as rec:
            rec["meta"]["compute_started_at"] = time.time()
            rec["meta"]["s_since_signing"] = time.time() - run["signed_at"]
            lst_result, qa_result = dask.compute(lst_u16, qa_count)

        with stage("stats"):
            dn = lst_result.values
            valid = dn != LST_NODATA_DN
            n_valid = int(valid.sum())
            frac = n_valid / dn.size if dn.size else 0.0
            summary_stats = {
                "n_pixels": int(dn.size),
                "n_valid": n_valid,
                "valid_fraction": frac,
                "dtype": str(dn.dtype),
            }
            if n_valid:
                cel = dn[valid].astype("float64") * LST_SCALE + LST_OFFSET
                summary_stats |= {
                    "min_c": float(cel.min()),
                    "max_c": float(cel.max()),
                    "mean_c": float(cel.mean()),
                    "p50_c": float(np.percentile(cel, 50)),
                }
            qa_vals = qa_result.values
            summary_stats["qa_count"] = {
                "dtype": str(qa_vals.dtype),
                "bands": int(qa_vals.shape[0]),
                "per_month_mean": {
                    MONTH_NAMES[i]: float(qa_vals[i].mean())
                    for i in range(qa_vals.shape[0])
                },
            }
        run["result"] = summary_stats

        if args.compare_passes:
            print("\ncomparing one-call against two-call compute...")
            with stage("compute_two_calls_lst"):
                dask.compute(lst_u16)
            with stage("compute_two_calls_qa"):
                dask.compute(qa_count)

        with stage("report"):
            frisky_summary = collect_frisky(
                out_dir, dashboard, args.span_limit, args.span_dump_limit
            )
        run["frisky"] = frisky_summary

    finally:
        if cluster is not None:
            try:
                cluster.close()
            except Exception:
                pass
        sampler.stop()

    # ---------------- output ----------------
    print()
    print_stage_table(sampler)

    run["stages"] = STAGES
    series = sampler.series()
    if series:
        run["memory"] = {
            "samples": len(series),
            "interval_s": args.sample_interval,
            "tree_rss_peak_mb": max(float(r["tree_rss_mb"]) for r in series),
            "client_rss_peak_mb": max(float(r["client_rss_mb"]) for r in series),
            "workers_rss_peak_mb": max(float(r["workers_rss_mb"]) for r in series),
            "sys_available_min_mb": min(float(r["sys_available_mb"]) for r in series),
        }
    print()
    res = run.get("result", {})
    if "mean_c" in res:
        print(
            f"LST p95       min {res['min_c']:.1f} C  "
            f"mean {res['mean_c']:.1f} C  max {res['max_c']:.1f} C  "
            f"({res['valid_fraction'] * 100:.1f}% valid)"
        )
    mem = run.get("memory", {})
    if mem:
        print(
            f"peak memory   {mem['tree_rss_peak_mb'] / 1024:.2f} GiB tree  "
            f"({mem['client_rss_peak_mb'] / 1024:.2f} client + "
            f"{mem['workers_rss_peak_mb'] / 1024:.2f} workers)"
        )

    # Throughput is the number that matters for an I/O bound pipeline.
    comp = next((r for r in STAGES if r["stage"] == "compute"), None)
    win = (comp or {}).get("memory_window") or {}
    n_scenes = run.get("n_scenes") or 0
    if win.get("net_recv_mb") and comp:
        mb = win["net_recv_mb"]
        wall = comp["wall_s"]
        thr = {
            "net_recv_mb": mb,
            "wall_s": wall,
            "mb_per_s": mb / wall if wall else 0.0,
            "mb_per_s_peak": win.get("net_recv_mb_s_peak", 0.0),
            "mb_per_scene": mb / n_scenes if n_scenes else 0.0,
            "s_per_scene": wall / n_scenes if n_scenes else 0.0,
            "concurrency": run["config"]["workers"]
            * run["config"]["threads_per_worker"],
        }
        # Time actually spent inside task bodies, over wall. Tells us how much
        # of the available concurrency the reads managed to use.
        by_name = run.get("frisky", {}).get("by_name", {})
        execmsg = by_name.get("worker.exec.call", {}).get("total_ms")
        if execmsg:
            thr["exec_s"] = execmsg / 1000.0
            thr["effective_parallelism"] = (execmsg / 1000.0) / wall if wall else 0.0
            thr["utilisation_pct"] = (
                100.0 * thr["effective_parallelism"] / thr["concurrency"]
            )
        run["throughput"] = thr
        print(
            f"throughput    {thr['mb_per_s']:.1f} MB/s mean, "
            f"{thr['mb_per_s_peak']:.1f} MB/s peak, "
            f"{mb:,.0f} MB total, {thr['mb_per_scene']:.1f} MB/scene"
        )
        if "effective_parallelism" in thr:
            print(
                f"concurrency   {thr['effective_parallelism']:.1f}x effective of "
                f"{thr['concurrency']} slots ({thr['utilisation_pct']:.0f}% used), "
                f"{thr['s_per_scene']:.2f} s/scene"
            )
    print(f"spans         {run.get('frisky', {}).get('n_spans', 0):,}")
    (out_dir / "stages.json").write_text(json.dumps(run, indent=2, default=str))
    print(f"artifacts     {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

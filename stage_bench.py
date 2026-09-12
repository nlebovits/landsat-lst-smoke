# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "boto3", "psutil", "pyarrow>=16", "shapely",
# ]
# ///
"""Which staging setting binds? Fetch one sample repeatedly and find out.

MEASURED on an `m6id.16xlarge` in us-west-2 against `s3://usgs-landsat`:
9,552 objects and 381.6 GiB in 1,189.6 s, about 344 MB/s, with the machine 16%
busy, 280 to 340 MB/s on the wire and iowait under 1%. Nothing there was CPU
bound or disk bound, and 344 MB/s is a small fraction of a 25 Gbit link, so the
ceiling is in the request path. DERIVED from the same figures, each of the 64
connections carried 5.4 MB/s, against the 50 to 90 MB/s a single S3 connection
is normally good for.

Three explanations fit that shape and the run could not separate them, because
one flag set all of them: the pool was smaller than the thread count, the
connections were not reused, or the per-object overhead dominated. `staging.py`
now has a knob per explanation. This runs one fixed sample once per setting and
prints the throughput each produced, so the next fleet run is configured off a
measurement rather than off the argument above.

    uv run stage_bench.py --i-am-in-region \\
      --tile S30W065 --root /mnt/nvme --objects 200

The sample is the first N objects of the tile's manifest in the manifest's own
sorted order, so every configuration fetches the same bytes in the same
sequence, and the directory is removed between configurations so none of them
reads another's files off disk.

Costs, DERIVED: N objects per configuration at one GET each, plus a GET per
part where the part concurrency is above one, at $0.0004 per 1,000. 200 objects
across six configurations is about 2,000 requests and under a cent. The bytes
are free in region and are the reason for `--i-am-in-region`: the same run
across a region boundary bills about 8 GB per configuration at $0.02 per GB,
and it measures the wrong network.

Run it on the instance that will run the fleet. It measures that instance's
link, and a figure from anywhere else describes a different machine.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import staging
from land_tiles import tile_bounds
from tile_inventory import items_for_tile

#: The bucket's region. A cross-region run bills egress and measures a link
#: the fleet will never use.
EXPECTED_REGION = "us-west-2"

#: Objects per configuration. 200 is about 8 GB, which is long enough for the
#: connection pool to reach steady state and short enough to run six of them.
DEFAULT_OBJECTS = 200

#: How often the sampler reads the CPU and the RSS, in seconds.
SAMPLE_S = 1.0

#: What `--config` understands, and what each key sets.
CONFIG_KEYS = {
    "threads": "threads",
    "connections": "connections",
    "parts": "part_concurrency",
    "mb": "part_mb",
    "fsync": "fsync",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--i-am-in-region",
        action="store_true",
        help=f"assert this instance is in {EXPECTED_REGION}, with the bucket. "
        f"Required. Without it this refuses to run, because a cross-region "
        f"sweep bills egress per gigabyte and measures the wrong network",
    )
    p.add_argument("--tile", required=True, help="the tile whose manifest to sample")
    p.add_argument(
        "--inventory-uri",
        type=Path,
        default=Path("artifacts/tile_scene_inventory.parquet"),
        help="the scene inventory the fleet reads",
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="volume to stage into. The sample directory is created under it "
        "and removed between configurations. Point it at the NVMe mount",
    )
    p.add_argument("--objects", type=int, default=DEFAULT_OBJECTS)
    p.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="threads=64,connections=256,parts=4,mb=16",
        help="one configuration, repeatable. Keys: "
        + ", ".join(sorted(CONFIG_KEYS))
        + ". Anything omitted takes staging's default, and an omitted "
        "connection count becomes threads x parts. Without any --config the "
        "default matrix below runs",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per configuration. The first fetch of a cold sample warms "
        "nothing this measures, so 1 is usually right",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="run even though the session's region is not the bucket's",
    )
    return p.parse_args(argv)


def default_matrix():
    """Threads against part concurrency, with the part size where it matters.

    Six configurations. The part size is swept only where objects are split,
    because it does nothing at a part concurrency of one, and running it twice
    there would pay for a repeat of the baseline and call it a comparison.
    """
    out = [{"threads": t} for t in (64, 128)]
    out += [{"threads": t, "parts": 4, "mb": mb} for t in (64, 128) for mb in (8, 16)]
    return out


def parse_config(text: str) -> dict:
    """`threads=64,parts=4,mb=16` to a dict of the keys `CONFIG_KEYS` names.

    Raises:
        SystemExit: on an unknown key. A typo that fell through would run the
            default configuration under another one's name and put a wrong
            number in a report.
    """
    out: dict = {}
    for piece in text.split(","):
        if not piece.strip():
            continue
        key, _, value = piece.partition("=")
        key = key.strip()
        if key not in CONFIG_KEYS:
            sys.exit(f"unknown --config key {key!r}. Known: {sorted(CONFIG_KEYS)}")
        out[key] = value.strip() if key == "fsync" else float(value)
    return out


def settings_for(config: dict) -> staging.FetchSettings:
    """One `FetchSettings` from one `--config` entry."""
    part_mb = config.get("mb")
    return staging.FetchSettings.build(
        threads=_as_int(config.get("threads")),
        connections=_as_int(config.get("connections")),
        part_bytes=None if part_mb is None else int(part_mb * 1024**2),
        part_concurrency=_as_int(config.get("parts")),
        fsync=config.get("fsync"),
    )


def _as_int(value):
    return None if value is None else int(value)


class Sampler:
    """Machine CPU and this process's RSS, once a second, in a thread.

    System-wide CPU rather than this process's, because the figure it has to
    be compared against is `sar`'s 16%, which is the machine. Peak RSS is the
    process, because that is what a fleet instance has to fit beside its
    workers.
    """

    def __init__(self, interval: float = SAMPLE_S):
        import psutil

        self.interval = interval
        self.cpu: list[float] = []
        self.rss_peak = 0
        self._psutil = psutil
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self._psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.interval):
            self.cpu.append(self._psutil.cpu_percent(interval=None))
            self.rss_peak = max(self.rss_peak, self._process.memory_info().rss)

    def __enter__(self):
        self.rss_peak = self._process.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False

    @property
    def mean_cpu(self) -> float:
        return statistics.fmean(self.cpu) if self.cpu else float("nan")


class CountingClient:
    """A client factory that counts GETs where botocore sends them.

    `staging.json` counts its own requests, and this counts them again through
    a botocore event so the two can be compared. They agree when the module's
    accounting is right, and a disagreement is the finding.

    `before-send` rather than `before-call`: it fires once per HTTP request
    rather than once per API call, so a botocore-level retry would show here
    and nowhere else. `staging._default_client` disables those, and this is
    what would say if a change re-enabled them.
    """

    def __init__(self, build=None):
        self.gets = 0
        self._build = build or staging._default_client
        self._lock = threading.Lock()

    def __call__(self, connections: int):
        client = self._build(connections)
        client.meta.events.register("before-send.s3.GetObject", self._count)
        return client

    def _count(self, **_kw):
        with self._lock:
            self.gets += 1
        return None


def sample_manifest(inventory_uri: Path, tile: str, n: int):
    """The first `n` objects of a tile's manifest, in the manifest's order.

    `staging.staging_manifest` sorts, so this is the same sample on every run
    and on every machine. It is not a random sample of the tile: it is a fixed
    one, which is what a comparison between configurations needs.
    """
    items, _ = items_for_tile(inventory_uri, tile, bounds=tile_bounds(tile))
    items, _, _ = staging.drop_scenes_without_thermal(
        items, [tuple(i["bbox"]) for i in items]
    )
    manifest = staging.staging_manifest(items, range(len(items)))
    if not manifest:
        sys.exit(f"tile {tile} has no objects to stage")
    return manifest[:n]


def run_once(manifest, stage_dir: Path, settings, build=None) -> dict:
    """Stage the sample once and return everything the line prints.

    The directory goes before the fetch as well as after it. A leftover from a
    crashed run would be skipped rather than fetched, and the configuration
    would post a throughput figure for objects it never pulled.

    `build` is the client behind the counter. It is `staging._default_client`
    on the instance, and a loopback endpoint in the tests, which is what lets
    the timing and counting here be exercised without a bucket.
    """
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    counter = CountingClient(build)
    try:
        with Sampler() as sampler:
            run = staging.StagingRun(
                list(manifest),
                stage_dir,
                settings=settings,
                client_factory=counter,
            )
            with run:
                run.drain()
        report = run.report()
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    seconds = report["seconds"] or float("nan")
    return report | {
        "mb_s": report["bytes"] / 1e6 / seconds,
        "objects_s": report["objects"] / seconds,
        "mean_cpu": sampler.mean_cpu,
        "rss_peak": sampler.rss_peak,
        "wire_gets": counter.gets,
    }


def header() -> str:
    return (
        f"{'threads':>7} {'conn':>5} {'parts':>5} {'part MB':>7} {'fsync':>5} "
        f"{'MB/s':>7} {'obj/s':>6} {'sec':>7} {'CPU %':>6} {'RSS MB':>7} "
        f"{'GETs':>6} {'wire':>6} {'retry':>5}"
    )


def line(settings, result: dict) -> str:
    return (
        f"{settings.threads:>7} {settings.connections:>5} "
        f"{settings.part_concurrency:>5} "
        f"{settings.part_bytes / 1024**2:>7.0f} {settings.fsync:>5} "
        f"{result['mb_s']:>7.1f} {result['objects_s']:>6.2f} "
        f"{result['seconds']:>7.1f} {result['mean_cpu']:>6.1f} "
        f"{result['rss_peak'] / 1024**2:>7.0f} {result['get_requests']:>6} "
        f"{result['wire_gets']:>6} {result['retries']:>5}"
    )


def check_region(args) -> None:
    """Refuse a run that would bill egress and measure the wrong link.

    Two checks. The flag is the operator's assertion and is required, because
    nothing this script can read proves intent. The session region is the
    machine's answer to the same question, and it disagrees for free.
    """
    if not args.i_am_in_region:
        sys.exit(
            f"stage_bench refuses to run without --i-am-in-region. It fetches "
            f"whole scene objects from s3://usgs-landsat, which lives in "
            f"{EXPECTED_REGION}. In region the bytes are free and the figures "
            f"describe the fleet's network. Anywhere else it bills about "
            f"$0.02 per GB and measures a link no fleet instance has."
        )
    import boto3

    region = boto3.Session().region_name
    if region and region != EXPECTED_REGION and not args.force:
        sys.exit(
            f"this session's region is {region} and the bucket is in "
            f"{EXPECTED_REGION}. Fix the region, or pass --force if the "
            f"session region is wrong and the instance is not."
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    check_region(args)

    configs = [parse_config(c) for c in args.config] or default_matrix()
    manifest = sample_manifest(args.inventory_uri, args.tile, args.objects)
    stage_dir = args.root / f"stage-bench-{args.tile}"

    print(
        f"sample        {len(manifest):,} objects of {args.tile}, "
        f"first in manifest order, into {stage_dir}"
    )
    print(f"MEASURED      one run per configuration, {args.repeat} repeat(s)")
    print(header())
    for config in configs:
        settings = settings_for(config)
        for _ in range(args.repeat):
            result = run_once(manifest, stage_dir, settings)
            print(line(settings, result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

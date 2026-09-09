"""Fetch each scene object once, instead of once per shard that touches it.

Requester-pays bills requests, not bytes. In-region S3 to EC2 transfer is
$0.00, so S3 charges for the shape of an access pattern rather than its volume,
and the sharded read path has the worst shape available.

The multiplier is geometry. A 512 px shard at 3600 px per degree covers about
209 km². A Landsat scene covers 185 x 180 km, or 33,300 km². So about 155
shards touch each scene, and each one opens the file again, in a different
worker process, with no shared cache. `measure_s3_requests.py` counts 4.77
ranged GETs per open. The product is 739 requests per object, and across 895
tiles it comes to about $1,822 against $807 to $928 of on-demand compute.

The bytes were never the problem. A full tile holds about 215 GB of distinct
source data and the overlapping shard windows already move about 233 GB, so
staging moves less data than the reads it replaces, not more.

This module fetches every object the slice needs with one GET, writes it to
local disk, and rewrites the item hrefs to point there. The 155 reads still
happen. They stop being billable, and the run's S3 line becomes a counted total
rather than a figure derived from a sampled requests-per-read.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

#: The two assets `process_shard` loads. Staging fetches these and nothing else.
BANDS = ("lwir11", "qa_pixel")

#: Bytes to reserve per object before the first GET, so the disk guard can run
#: without a HEAD per object. HEAD is billable, and the guard needs an answer
#: before anything is fetched.
#:
#: `FINDINGS.md` measures about 233 GB of reads over 3,910 scenes, or 60 MB per
#: scene for both bands together. The thermal band carries almost all of it:
#: `ST_B10` is a full-scene uint16 raster, and `QA_PIXEL` is the same shape but
#: compresses far harder because it holds bit flags.
ESTIMATED_BYTES = {"lwir11": 52_000_000, "qa_pixel": 8_000_000}

#: Applied to the estimate above. A scene larger than the mean must not fill
#: the disk halfway through a slice.
DISK_SAFETY_FACTOR = 1.25

#: Staging aborts if free space falls below this while fetching. The estimate
#: is a mean, so the guard that runs up front can pass on a slice whose scenes
#: are all larger than average.
FREE_SPACE_FLOOR_BYTES = 5 * 1024**3

#: Attempts per object, first try included. A failed GET is billed, so the
#: count belongs in the report rather than inside a client's retry handler.
MAX_ATTEMPTS = 3

#: Seconds before a retry. Short, because a failure here is usually a reset
#: connection rather than throttling.
RETRY_BACKOFF_S = 1.0


class StagingError(RuntimeError):
    """A staged object is missing, short, or unfetchable."""


def drop_scenes_without_thermal(item_dicts, item_bboxes):
    """Remove the Level 2 products that carry no thermal band.

    `tile_inventory.build_item` omits the `lwir11` asset entirely when the
    inventory row has no `thermal_href`, which is what Earth Search returns for
    an `OLI_TIRS_L2SR` product. Those scenes reach `process_shard`, load as
    fill, and contribute nothing.

    The removal is output-neutral by construction. `lst_qa.valid_observation`
    starts with `not_fill`, so a scene of pure fill is false everywhere in both
    the percentile and the monthly counts. It still costs a `qa_pixel` fetch and
    a layer on the time axis of every shard it touches, and that time axis is
    what caps shard size at 94% of the worker memory limit.

    `item_bboxes` is index-parallel to `item_dicts`. Filtering one without the
    other would leave every shard reading the wrong scenes, so both are
    filtered together and returned together.

    Returns:
        The kept items, their bboxes, and how many scenes were dropped.
    """
    kept = [
        (item, box)
        for item, box in zip(item_dicts, item_bboxes, strict=True)
        if "lwir11" in item.get("assets", {})
    ]
    n_dropped = len(item_dicts) - len(kept)
    if not kept:
        return [], [], n_dropped
    items, boxes = zip(*kept, strict=True)
    return list(items), list(boxes), n_dropped


def split_s3_uri(href: str) -> tuple[str, str]:
    """`s3://bucket/key` to `(bucket, key)`.

    Raises:
        StagingError: for anything that is not an `s3://` URI. The inventory
            writes only those, so a different scheme means the item did not
            come from where this module assumes.
    """
    parsed = urlparse(href)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        msg = f"staging expects an s3:// href, got {href!r}"
        raise StagingError(msg)
    return parsed.netloc, parsed.path.lstrip("/")


def staging_manifest(item_dicts, indices):
    """The distinct objects a slice needs, as `(item_id, band, href)`.

    One scene appearing in 155 shards produces two objects, not 310. That
    collapse is the entire saving, so it happens here, before anything is
    fetched.
    """
    seen: dict[tuple[str, str], tuple[str, str, str]] = {}
    for i in indices:
        item = item_dicts[i]
        item_id = item["id"]
        if "/" in item_id or item_id in {".", ".."}:
            msg = f"item id {item_id!r} is not usable as a directory name"
            raise StagingError(msg)
        for band in BANDS:
            asset = item.get("assets", {}).get(band)
            if asset and asset.get("href"):
                seen.setdefault((item_id, band), (item_id, band, asset["href"]))
    return sorted(seen.values())


def estimated_bytes(manifest) -> int:
    """Bytes the manifest is expected to need on disk, with the safety factor."""
    raw = sum(ESTIMATED_BYTES.get(band, 0) for _, band, _ in manifest)
    return int(raw * DISK_SAFETY_FACTOR)


def disk_guard(manifest, stage_dir: Path) -> int:
    """Refuse to start a fetch that cannot finish.

    Runs before the first GET, off the estimate rather than a HEAD per object,
    because HEAD is billable too. The message names both figures and the escape,
    since the operator's next decision is whether to resize the volume or fall
    back to reading from S3.

    Returns:
        The estimated bytes, so the caller can report what it reserved.
    """
    need = estimated_bytes(manifest)
    free = shutil.disk_usage(stage_dir).free
    if free < need:
        msg = (
            f"staging {len(manifest):,} objects needs about "
            f"{need / 1024**3:.1f} GiB and {stage_dir} has "
            f"{free / 1024**3:.1f} GiB free. Point --stage-dir at a larger "
            f"volume, or pass --no-stage to read from S3 instead, which costs "
            f"about 739 requests per object."
        )
        raise StagingError(msg)
    return need


def _default_client():
    """One boto3 S3 client.

    Called once per worker thread. botocore clients are safe to share, but a
    client per thread avoids contention on the connection pool and costs
    nothing to build.
    """
    import boto3

    return boto3.client("s3")


def _fetch_one(client, bucket: str, key: str, dest: Path) -> tuple[int, int]:
    """One object, one GET, streamed to `dest`.

    `get_object` rather than `download_file`: the transfer manager splits
    anything over its 8 MB threshold into several ranged GETs, which would turn
    2 requests per scene into about 8 and give back a quarter of the saving.
    Concurrency belongs across objects, not inside one.

    The response `ContentLength` is checked against the bytes written. A short
    read leaves no file behind, because a truncated GeoTIFF would produce a
    wrong percentile in silence rather than an error.

    Returns:
        The bytes written and the number of billable requests it took.
    """
    attempts = 0
    last: Exception | None = None
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            resp = client.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
            expected = int(resp["ContentLength"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                shutil.copyfileobj(resp["Body"], fh)
            written = dest.stat().st_size
            if written != expected:
                dest.unlink(missing_ok=True)
                msg = (
                    f"s3://{bucket}/{key} returned {expected:,} bytes and "
                    f"{written:,} reached disk"
                )
                raise StagingError(msg)
        except Exception as exc:  # noqa: BLE001 - every failure is retryable here
            dest.unlink(missing_ok=True)
            last = exc
            if attempts < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S)
            continue
        else:
            return written, attempts
    msg = f"s3://{bucket}/{key} failed after {attempts} attempts: {last}"
    raise StagingError(msg)


def stage_scenes(
    item_dicts,
    indices,
    stage_dir: Path,
    *,
    threads: int | None = None,
    client_factory=_default_client,
) -> dict:
    """Fetch every object the slice needs and repoint the items at local disk.

    Mutates the asset hrefs of the staged items in place. The href becomes an
    absolute local path rather than a `file://` URL, because rasterio opens the
    former directly.

    Returns:
        A report: objects, bytes, seconds, billable GETs, retries, and the
        estimate the disk guard reserved. It is the run's S3 line, counted
        rather than derived.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = staging_manifest(item_dicts, indices)
    if not manifest:
        return _empty_report(stage_dir)
    reserved = disk_guard(manifest, stage_dir)

    local = threading.local()
    counters = {"bytes": 0, "requests": 0}
    lock = threading.Lock()

    def fetch(entry):
        item_id, band, href = entry
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = client_factory()
        bucket, key = split_s3_uri(href)
        dest = stage_dir / item_id / f"{band}{_suffix(key)}"
        written, attempts = _fetch_one(client, bucket, key, dest)
        with lock:
            counters["bytes"] += written
            counters["requests"] += attempts
            free = shutil.disk_usage(stage_dir).free
        if free < FREE_SPACE_FLOOR_BYTES:
            msg = (
                f"{stage_dir} fell to {free / 1024**3:.1f} GiB free while "
                f"staging, below the {FREE_SPACE_FLOOR_BYTES / 1024**3:.0f} "
                f"GiB floor. The per-object estimate was too low for this slice."
            )
            raise StagingError(msg)
        return item_id, band, str(dest.resolve())

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads or _default_threads()) as pool:
        placed = list(pool.map(fetch, manifest))
    elapsed = time.perf_counter() - t0

    by_object = {(item_id, band): path for item_id, band, path in placed}
    for i in indices:
        item = item_dicts[i]
        for band in BANDS:
            path = by_object.get((item["id"], band))
            if path is not None:
                item["assets"][band]["href"] = path

    return {
        "objects": len(manifest),
        "bytes": counters["bytes"],
        "seconds": elapsed,
        "get_requests": counters["requests"],
        "retries": counters["requests"] - len(manifest),
        "reserved_bytes": reserved,
        "stage_dir": str(stage_dir),
    }


def _suffix(key: str) -> str:
    """The source file's extension, so a staged file still looks like a TIFF."""
    return Path(key).suffix or ".TIF"


def _default_threads() -> int:
    """Threads for the fetch pool.

    Staging is network-bound and each object goes over one connection, so the
    pool oversubscribes the cores on purpose. Throughput comes from many
    objects in flight, not from splitting one.
    """
    return min(64, 4 * (os.cpu_count() or 4))


def _empty_report(stage_dir: Path) -> dict:
    return {
        "objects": 0,
        "bytes": 0,
        "seconds": 0.0,
        "get_requests": 0,
        "retries": 0,
        "reserved_bytes": 0,
        "stage_dir": str(stage_dir),
    }


def cleanup(stage_dir: Path) -> None:
    """Remove the staged files.

    Steady state runs one tile per instance, back to back. A stage directory
    left behind fills the disk on the second tile.
    """
    shutil.rmtree(stage_dir, ignore_errors=True)

"""Fetch each scene object once, instead of once per block that touches it.

Requester-pays bills requests, not bytes. In-region S3 to EC2 transfer is
$0.00, so S3 charges for the shape of an access pattern rather than its volume,
and reading straight from the bucket, block by block, has the worst shape
available.

The multiplier is geometry. A 512 px block at 3600 px per degree covers about
209 km². A Landsat scene covers 185 x 180 km, or 33,300 km². So about 155
blocks touch each scene, and each one opens the file again, in a different
worker process, with no shared cache. `measure_s3_requests.py` counts 4.77
ranged GETs per open. The product is 739 requests per object, and across 895
tiles it comes to about $1,822 against $807 to $928 of on-demand compute. The
default 360 px block is smaller still, so it can only make the count worse.

The bytes were never the problem either. MEASURED on one 512 px block of three
real scenes: the windowed reads pull 0.342 MB per object, so the 155 blocks
pull 53.0 MB against 33.6 MB for the whole object. Staging moves 0.63x the
bytes, because the overlapping windows fetch the same blocks again for every
one that touches them.

This module fetches every object the tile needs with one GET, writes it to
local disk, and rewrites the item hrefs to point there. The 155 reads still
happen. They stop being billable, and the run's S3 line becomes a counted total
rather than a figure derived from a sampled requests-per-read.

The hrefs are rewritten before the first GET rather than after the last one,
because the destination is a pure function of the item id, the band, and the
source key. Nothing depends on that ordering now, and it costs nothing: the
manifest is built once and read twice.

The fetch finishes before the cluster starts. Running it beside the blocks it
feeds was tried and MEASURED as a loss. The same 1,998 objects and 78.9 GiB
stage in 91.9 s on an idle `m6id.16xlarge` and in 358.7 s beside 64 busy
workers, because this phase spends its time on TLS, HTTP and the copy loop and
all three want a core. See `FINDINGS.md`.

## What bounds the throughput

MEASURED on an `m6id.16xlarge` in us-west-2 against `s3://usgs-landsat`: 9,552
objects and 381.6 GiB in 1,189.6 s, or about 344 MB/s, with the machine 16%
busy, 280 to 340 MB/s on the wire and iowait under 1%. Neither the cores nor
the disk were the ceiling, and a 25 Gbit link is not either.

One knob set the shape of that run. `threads` sized the fetch pool, and
`_default_client` sized the connection pool from the same number, so 64 threads
held 64 connections and each one carried a whole object end to end. 64 objects
in flight is the whole of the concurrency, and DERIVED from the figures above
each connection averaged 5.4 MB/s, which is well under the 50 to 90 MB/s one
S3 connection carries.

The flush is not the answer either. DERIVED from the 3,139 MB/s that fsync and
`POSIX_FADV_DONTNEED` MEASURED together: 409.7 GB of writes is 130 s of thread
time, which across 64 threads is 2 s of the 1,190 s window. `--stage-fsync`
exists to rule it out on the wire rather than on paper.

`FetchSettings` takes that one number apart into four, so the next run can say
which of them binds. Concurrency across objects is `threads`; sockets available
to them is `connections`; concurrency inside one object is `part_concurrency`
over `part_bytes` ranges; and `fsync` is whether the copy loop stops to flush.
The defaults reproduce the run above exactly, so nothing moves without a flag.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlparse

#: The two bands `composite.open_stack` loads. Staging fetches these and
#: nothing else.
BANDS = ("lwir11", "qa_pixel")

#: Bytes to reserve per object before the first GET, so the disk guard can run
#: without a HEAD per object. HEAD is billable, and the guard needs an answer
#: before anything is fetched.
#:
#: MEASURED by `HEAD` over 30 scenes of each platform, drawn at random from the
#: inventory. `ST_B10` runs 11.2 to 93.6 MB with a mean of 76.0 and a p95 of
#: 93.1. `QA_PIXEL` runs 0.3 to 9.7 MB with a mean of 2.5, because it holds bit
#: flags and compresses far harder. Together a scene averages 78.5 MB and tops
#: out near 100.
#:
#: These sit at the top of the measured range, not at the mean. The two errors
#: are not symmetric: over-reserving costs a refusal the operator can override,
#: and under-reserving costs a run that dies with a full disk after it has
#: already paid for its instance and its requests.
ESTIMATED_BYTES = {"lwir11": 95_000_000, "qa_pixel": 10_000_000}

#: Applied to the estimate above, which is already near the measured maximum.
DISK_SAFETY_FACTOR = 1.15

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

#: Copy buffer for one object, against `shutil.copyfileobj`'s 64 KiB default.
#: See `_fetch_one`.
COPY_BUFFER_BYTES = 1024 * 1024

#: Bytes per ranged GET once `part_concurrency` is above one, and the threshold
#: with it: an object no larger than this still takes a single GET, because the
#: first range covers the whole of it and the `Content-Range` says so.
#:
#: 8 MiB is `boto3.s3.transfer.TransferConfig`'s own default. It is carried
#: here rather than taken from that class because this module issues its own
#: `Range` GETs; see `_fetch_ranged`.
DEFAULT_PART_BYTES = 8 * 1024 * 1024

#: `none` skips the flush entirely, `file` is what staging has always done, and
#: `dir` adds an fsync of the containing directory. See `_release_page_cache`.
FSYNC_MODES = ("none", "file", "dir")


class StagingError(RuntimeError):
    """A staged object is missing, short, or unfetchable."""


@dataclasses.dataclass(frozen=True)
class FetchSettings:
    """The four numbers and one mode that decide how fast the fetch can go.

    They used to be one number. `threads` sized the fetch pool, and the client
    took its `max_pool_connections` from the same value, so asking for more
    concurrency across objects also asked for more sockets and there was no way
    to ask for either alone, or to put two sockets on one 84 MB object.

    - `threads`: objects in flight. The fetch pool's `max_workers`.
    - `connections`: botocore's `max_pool_connections`. Defaults to
      `threads * part_concurrency`, which is every socket the fetch can want
      at once, and reduces to `threads` at the default part concurrency.
    - `part_bytes`: the size of one ranged GET, and the size below which an
      object is fetched whole.
    - `part_concurrency`: ranged GETs in flight for one object. `1`, the
      default, is the off switch: it issues one GET per object, which is what
      the request count in `staging.json` has always meant.
    - `fsync`: one of `FSYNC_MODES`.

    Raising `part_concurrency` multiplies the billable GETs for the large
    objects. An 84 MB `ST_B10` at 16 MiB parts is 6 requests rather than 1, and
    across the fleet's 5.8 million objects that is DERIVED at about $14 against
    $2.32. It buys sockets per object, and it is priced in requests.

    Every field reaches `staging.json` through `StagingRun.report`, so a run's
    throughput can be read back against the settings that produced it.
    """

    threads: int
    connections: int
    part_bytes: int
    part_concurrency: int
    fsync: str

    @classmethod
    def build(
        cls,
        *,
        threads: int | None = None,
        connections: int | None = None,
        part_bytes: int | None = None,
        part_concurrency: int | None = None,
        fsync: str | None = None,
    ) -> FetchSettings:
        """Fill in the defaults and refuse a setting that cannot work.

        `None` everywhere reproduces the behaviour staging had before these
        knobs existed, which is what keeps a run without flags comparable to
        every run already measured.

        `None` and not falsiness. `--stage-connections 0` is a mistake and has
        to read as one; treating it as "unset" would answer a typo with a
        working run at a different setting than the operator asked for.
        """
        threads = _default_threads() if threads is None else threads
        part_concurrency = 1 if part_concurrency is None else part_concurrency
        settings = cls(
            threads=threads,
            connections=(
                threads * part_concurrency if connections is None else connections
            ),
            part_bytes=DEFAULT_PART_BYTES if part_bytes is None else part_bytes,
            part_concurrency=part_concurrency,
            fsync="file" if fsync is None else fsync,
        )
        settings.check()
        return settings

    def check(self) -> None:
        """Raises:
        StagingError: for a setting no fetch can honour. A zero thread count
            hangs, a zero part size loops forever, and an unknown fsync mode
            would silently fall through to no flush at all.
        """
        # The given settings before the derived one, so a bad part concurrency
        # is named rather than the connection count it made impossible.
        for name in ("threads", "part_bytes", "part_concurrency", "connections"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                msg = f"staging {name} must be a positive integer, got {value!r}"
                raise StagingError(msg)
        if self.fsync not in FSYNC_MODES:
            msg = f"staging fsync must be one of {FSYNC_MODES}, got {self.fsync!r}"
            raise StagingError(msg)

    @property
    def multipart(self) -> bool:
        """Whether an object above `part_bytes` is split across connections."""
        return self.part_concurrency > 1

    def as_dict(self) -> dict:
        """The settings as `staging.json` records them."""
        return dataclasses.asdict(self)


def drop_scenes_without_thermal(item_dicts, item_bboxes):
    """Remove the Level 2 products that carry no thermal band.

    `tile_inventory.build_item` omits the `lwir11` asset entirely when the
    inventory row has no `thermal_href`, which is what Earth Search returns for
    an `OLI_TIRS_L2SR` product. Those scenes reach the graph, load as fill, and
    contribute nothing.

    The removal is output-neutral by construction. `lst_qa.valid_observation`
    starts with `not_fill`, so a scene of pure fill is false everywhere in both
    the percentile and the monthly counts. It still costs a `qa_pixel` fetch
    and a step on the time axis of every block it touches, and that time axis
    is what `composite.block_bytes` prices the block against the worker memory
    limit.

    `item_bboxes` is index-parallel to `item_dicts`. Filtering one without the
    other would leave every block reading the wrong scenes, so both are
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

    One scene appearing in 155 blocks produces two objects, not 310. That
    collapse is the entire saving, so it happens here, before anything is
    fetched.

    Sorted, so two runs over the same slice fetch in the same order. Nothing
    downstream depends on the order now that staging finishes before the
    cluster starts.
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


def staged_path(stage_dir, item_id: str, band: str, key: str) -> Path:
    """Where one staged object lands: `<stage_dir>/<item_id>/<band><suffix>`.

    One definition, read by the fetch and by the href rewrite that now runs in
    front of it. A `--keep-staged` directory is read back off disk by name
    alone, with no manifest, so the shape is a contract rather than an
    implementation detail.
    """
    return Path(stage_dir) / item_id / f"{band}{_suffix(key)}"


def repoint_items(item_dicts, indices, stage_dir):
    """Point every asset href at its staged file, before the first GET.

    The destination is a pure function of the item id, the band, and the source
    key, so it can be named before anything is fetched. That is what lets a
    unit of work be submitted while other scenes are still arriving: the item
    list is final from the start, and the only thing that changes is whether a
    given file has landed yet. The caller releases a unit when its own objects
    have.

    The rewrite used to run after the whole fetch pool drained, which is what
    forced staging to finish before the cluster could start.

    The href becomes an absolute local path rather than a `file://` URL,
    because rasterio opens the former directly.

    Returns:
        The staging manifest, so the caller does not build it twice.
    """
    stage_dir = Path(stage_dir)
    manifest = staging_manifest(item_dicts, indices)
    placed = {
        (item_id, band): str(
            staged_path(stage_dir, item_id, band, split_s3_uri(href)[1]).resolve()
        )
        for item_id, band, href in manifest
    }
    for i in indices:
        item = item_dicts[i]
        for band in BANDS:
            path = placed.get((item["id"], band))
            if path is not None:
                item["assets"][band]["href"] = path
    return manifest


def estimated_bytes(manifest, stage_dir=None) -> int:
    """Bytes the manifest is expected to need on disk, with the safety factor.

    Given `stage_dir`, objects already staged there are left out. They occupy
    the volume rather than asking it for more, so counting them would refuse a
    rerun on the one volume already holding the data.
    """
    entries = manifest
    if stage_dir is not None:
        entries = [
            e
            for e in manifest
            if not staged_path(stage_dir, e[0], e[1], split_s3_uri(e[2])[1]).exists()
        ]
    raw = sum(ESTIMATED_BYTES.get(band, 0) for _, band, _ in entries)
    return int(raw * DISK_SAFETY_FACTOR)


def disk_guard(manifest, stage_dir: Path) -> int:
    """Refuse to start a fetch that cannot finish.

    Runs before the first GET, off the estimate rather than a HEAD per object,
    because HEAD is billable too. The message names both figures and the one
    escape, which is a larger volume: a run composites one whole tile on one
    machine, so there is no smaller slice of it to fall back to. Reading from
    S3 instead is not an escape either, because the unstaged path costs about
    739 requests per object against one, and runs about twice as slow.

    Returns:
        The estimated bytes, so the caller can report what it reserved.
    """
    need = estimated_bytes(manifest, stage_dir)
    free = shutil.disk_usage(stage_dir).free
    if free < need:
        msg = (
            f"staging {len(manifest):,} objects needs about "
            f"{need / 1024**3:.1f} GiB and {stage_dir} has "
            f"{free / 1024**3:.1f} GiB free. Point --stage-dir at a volume "
            f"that holds the tile: one machine composites one whole tile, so "
            f"there is no smaller slice to cut it into, and there is no "
            f"unstaged path to fall back to."
        )
        raise StagingError(msg)
    return need


def _default_client(pool_size: int | None = None):
    """One boto3 S3 client, configured so this module can count and saturate.

    Two defaults have to go, and neither shows up in a small test.

    `retries` defaults to `legacy`, which retries a 500 or a 503 up to five
    times inside `get_object`. Every one of those is a billable request that
    `_fetch_one` never sees, so a throttled run would report fewer GETs than it
    paid for. Counting is the reason `staging.json` exists, so retrying belongs
    here where `MAX_ATTEMPTS` bounds it and the report carries it.

    `total_max_attempts`, not `max_attempts`: botocore reads the latter as
    retries after the first try, so `max_attempts: 1` still sends two.

    `max_pool_connections` defaults to 10. The fetch pool runs up to 64
    threads, so 54 of them would queue behind a connection instead of pulling
    an object, and a fleet-sized slice would take six times longer than the
    network needs. `pool_size` comes from the pool that will use it, because a
    caller passing more threads than the default would meet the same queue
    this argument exists to remove.

    `pool_size` is `FetchSettings.connections`, which is no longer the same
    number as the thread count: a run splitting one object across four ranged
    GETs wants four sockets per fetch thread, and one that wants to know
    whether the sockets bind can set them apart with `--stage-connections`.

    One client, shared. `botocore.Session.create_client` is not thread-safe,
    and building one per thread also multiplies the connection pool by the
    thread count. The client itself is thread-safe for calls, which is the
    part that matters here.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            retries={"total_max_attempts": 1, "mode": "standard"},
            max_pool_connections=pool_size or _default_threads(),
        ),
    )


def _release_page_cache(fd: int, dest: Path, mode: str) -> None:
    """Tell the kernel it can drop what we just wrote.

    A mean tile stages about 278 GB onto an instance holding 256 GiB of RAM.
    The written volume exceeds memory, so the kernel has to reclaim that cache
    whatever happens. Releasing each object as it lands bounds what there is to
    reclaim to one object, instead of letting it grow until the kernel decides.

    The workers re-read these files from NVMe seconds later, so the cache is
    not worth holding: the read repopulates what it needs.

    `fsync` first, because `POSIX_FADV_DONTNEED` cannot drop a dirty page and
    would silently do nothing without it. MEASURED at 3,139 MB/s against
    7,711 MB/s for the same writes without either call, on 24 objects of 85 MB
    across 16 threads. The cost is real and it does not bind: staging is
    network-bound at the 922 MB/s an `m6id.16xlarge` measured in region, which
    is 3.4x below the slower figure.

    This is not what fixed the run that died at cluster start. That was the
    per-unit memory model under-reporting by 3.9x, and a corrected budget alone
    would have fit; `composite.block_bytes` is the corrected model. The justification here is the written volume against RAM, which
    holds independently.

    Best effort. `posix_fadvise` is Linux-only and advisory everywhere.

    `mode` is `--stage-fsync`. `file` is the behaviour described above and the
    default. `none` skips the call, which is worth measuring because the fsync
    runs on the fetch thread and holds it: the 3,139 MB/s figure is the cost
    with the machine to itself, and a fetch thread waiting on a flush is a
    thread not pulling the next object. DERIVED against a fleet-sized prep,
    that comes to 2 s of a 1,190 s window, so this is a knob to rule the flush
    out with rather than one expected to move anything. It gives up the
    page-cache release with it, so a run that uses it is asking the kernel to
    reclaim 278 GB on its own schedule.

    `dir` adds an fsync of the containing directory, which makes the file's
    name durable rather than only its contents. Nothing here needs that, and it
    exists so an operator can rule it out as a cost.

    The `posix_fadvise` guard applies to `file` alone. Without the advice there
    is nothing for the fsync to serve, so on a platform that lacks it `file`
    does nothing. `dir` is a durability request rather than a cache one and
    runs everywhere.
    """
    if mode == "none":
        return
    if mode == "file" and not hasattr(os, "posix_fadvise"):
        return
    try:
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    if mode == "dir":
        _fsync_dir(dest.parent)


def _fsync_dir(directory: Path) -> None:
    """Flush a directory entry. Best effort, and not available on every OS."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class _RequestCounter:
    """Billable GETs, counted where they are issued.

    One GET per object made the attempt number a request count by accident. A
    ranged fetch issues several per attempt and can fail halfway through one,
    and a failed GET is billed like any other, so the count has to be taken at
    the call rather than inferred from the outcome.

    Locked, because the parts of one object are counted from several threads.
    """

    def __init__(self) -> None:
        self.n = 0
        self._lock = threading.Lock()

    def bump(self) -> None:
        with self._lock:
            self.n += 1


def _fetch_one(
    client, bucket: str, key: str, dest: Path, settings=None, parts=None
) -> tuple[int, int, int]:
    """One object to `dest`, retried as a whole.

    A retry restarts the object rather than the part that failed. The unit that
    has to be correct is the file, and a partially reassembled one is exactly
    the truncated GeoTIFF this module refuses to leave behind.

    Returns:
        The bytes written, the billable requests it took, and the retries.
        Requests and retries are two numbers now: a default fetch issues one
        GET per attempt and the two still agree, and a ranged fetch issues
        several and they do not.
    """
    settings = settings or FetchSettings.build()
    if settings.multipart and parts is None:
        # A programming error rather than a fetch failure, so it says which
        # caller is missing rather than retrying three times and timing out.
        msg = "a ranged fetch needs a part pool, which `StagingRun.start` builds"
        raise StagingError(msg)
    counter = _RequestCounter()
    attempts = 0
    last: Exception | None = None
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            if settings.multipart:
                written = _fetch_ranged(
                    client, bucket, key, dest, settings, counter, parts
                )
            else:
                written = _fetch_whole(client, bucket, key, dest, settings, counter)
        except Exception as exc:  # noqa: BLE001 - every failure is retryable here
            dest.unlink(missing_ok=True)
            last = exc
            if attempts < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S)
            continue
        else:
            return written, counter.n, attempts - 1
    msg = f"s3://{bucket}/{key} failed after {attempts} attempts: {last}"
    raise StagingError(msg)


def _fetch_whole(client, bucket: str, key: str, dest: Path, settings, counter) -> int:
    """One object, one GET, streamed to `dest`. The default path.

    `get_object` rather than `download_file`: the transfer manager splits
    anything over its 8 MB threshold into several ranged GETs, which would turn
    2 requests per scene into about 8 and give back a quarter of the saving.
    Concurrency belongs across objects unless an operator asks otherwise, which
    is what `--stage-part-concurrency` is and why it is off by default.

    The response `ContentLength` is checked against the bytes written. A short
    read leaves no file behind, because a truncated GeoTIFF would produce a
    wrong percentile in silence rather than an error.

    `copyfileobj` gets a 1 MiB buffer rather than its 64 KiB default. At the
    measured 922 MB/s the default acquires and releases the GIL about 14,000
    times a second, in the same interpreter that now runs the frisky client and
    the assembly loop. One keyword cuts that by 16x and changes nothing else.
    """
    counter.bump()
    resp = client.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
    expected = int(resp["ContentLength"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        shutil.copyfileobj(resp["Body"], fh, COPY_BUFFER_BYTES)
        fh.flush()
        _release_page_cache(fh.fileno(), dest, settings.fsync)
    written = dest.stat().st_size
    if written != expected:
        dest.unlink(missing_ok=True)
        msg = (
            f"s3://{bucket}/{key} returned {expected:,} bytes and "
            f"{written:,} reached disk"
        )
        raise StagingError(msg)
    return written


def _fetch_ranged(
    client, bucket: str, key: str, dest: Path, settings, counter, parts
) -> int:
    """One object as several ranged GETs, reassembled at their offsets.

    A single S3 connection carries 50 to 90 MB/s, so an 84 MB `ST_B10` occupies
    one socket for about a second whatever else the machine is doing. This lets
    several sockets land it, and charges one billable GET per part for the
    privilege.

    No HEAD. The size arrives with the first part: a `Range` request answers
    `Content-Range: bytes 0-8388607/88080384`, and the figure after the slash
    is the whole object. HEAD is billable, and buying one would spend a request
    to learn what a request is about to say anyway. An object no larger than
    one part therefore still costs exactly one GET, which is why the 2 MB
    `QA_PIXEL` objects are untouched by this path.

    `os.pwrite` at each part's own offset, so the parts need no ordering
    between them and no buffer to be assembled in. The file is opened once and
    sized by the writes.

    The spans start at the bytes the first part actually delivered rather than
    at the requested part size. A server that answers a range with the whole
    object leaves nothing to fetch, and this is what says so.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = settings.part_bytes
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        total, written = _fetch_part(client, bucket, key, fd, 0, size, counter)
        spans = [(off, min(size, total - off)) for off in range(written, total, size)]
        for i in range(0, len(spans), settings.part_concurrency):
            batch = spans[i : i + settings.part_concurrency]
            written += _gather_parts(
                client, bucket, key, fd, batch, counter, parts, total
            )
        if written != total:
            msg = (
                f"s3://{bucket}/{key} announced {total:,} bytes and "
                f"{written:,} reached disk"
            )
            raise StagingError(msg)
        _release_page_cache(fd, dest, settings.fsync)
    finally:
        os.close(fd)
    return written


def _gather_parts(client, bucket, key, fd, spans, counter, parts, total) -> int:
    """Fetch one batch of ranges at once and return the bytes they wrote.

    Every future is waited on before this returns, failure included. The caller
    closes the descriptor these threads write through, and a part still running
    when it does would write into whatever the process opens next.

    Raises:
        StagingError: if two parts disagree about the object's size, which
            means it was replaced mid-fetch and the reassembly would splice two
            versions of the scene together.
    """
    futures = [
        parts.submit(_fetch_part, client, bucket, key, fd, off, length, counter)
        for off, length in spans
    ]
    wait(futures)
    written = 0
    failure: BaseException | None = None
    for future in futures:
        exc = future.exception()
        if exc is not None:
            failure = failure or exc
            continue
        part_total, n = future.result()
        written += n
        if part_total != total:
            msg = (
                f"s3://{bucket}/{key} changed size mid-fetch: {total:,} bytes "
                f"and then {part_total:,}"
            )
            failure = failure or StagingError(msg)
    if failure is not None:
        raise failure
    return written


def _fetch_part(
    client, bucket: str, key: str, fd: int, offset: int, length: int, counter
) -> tuple[int, int]:
    """One ranged GET, written where it belongs in the file.

    Returns:
        The object's total size as the response reports it, and the bytes this
        part wrote.
    """
    counter.bump()
    resp = client.get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes={offset}-{offset + length - 1}",
        RequestPayer="requester",
    )
    body = resp["Body"]
    written = 0
    while True:
        chunk = body.read(COPY_BUFFER_BYTES)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            n = os.pwrite(fd, view, offset + written)
            view = view[n:]
            written += n
    return _total_bytes(resp, written), written


def _total_bytes(resp, written: int) -> int:
    """The whole object's size, from what a ranged GET answered with.

    `Content-Range: bytes 0-8388607/88080384`. The figure after the slash is
    the object, which is what makes the first part enough to plan the rest and
    a HEAD unnecessary.

    A server that answers a range request with `200` and no `Content-Range`
    sent the whole object, so `ContentLength` is the size. A `*` after the
    slash is a server declining to say, and then the only figure available is
    what arrived.
    """
    header = resp.get("ContentRange") or ""
    tail = header.rsplit("/", 1)[-1].strip()
    if tail.isdigit():
        return int(tail)
    length = resp.get("ContentLength")
    return int(length) if length is not None else written


class StagingRun:
    """Fetch a manifest, and say which objects have landed while it runs.

    Two callers, one fetch. `stage_scenes` drains this to the end and returns a
    report, which is what `--no-overlap` does. An overlapped run drains it a few
    objects at a time between gathering results, and releases each unit of work
    as the last object it waits on arrives.

    The fetch itself is the same either way, so the two orderings cannot
    diverge on request counts, retries, or the bytes they verify.
    """

    def __init__(
        self,
        manifest,
        stage_dir: Path,
        *,
        threads: int | None = None,
        settings: FetchSettings | None = None,
        client_factory=None,
    ):
        self.manifest = list(manifest)
        self.stage_dir = Path(stage_dir)
        self.settings = _resolve_settings(threads, settings)
        self.threads = self.settings.threads
        #: Objects still to land. `landed` decrements it, so an overlapped
        #: caller can tell a quiet moment from a finished fetch.
        self.outstanding = len(self.manifest)
        self.seconds = 0.0
        self._counters = {"bytes": 0, "requests": 0, "retries": 0, "reused": 0}
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        # Resolved in `start`, not bound here. A default argument would
        # capture `_default_client` at import, which is not what a caller
        # swapping it for a fake bucket expects.
        self._client_factory = client_factory
        self._pool: ThreadPoolExecutor | None = None
        #: The second pool, for the ranges of one object. `None` at the
        #: default part concurrency, where there are none. It has to be a
        #: separate pool: a fetch thread submitting its own parts to the pool
        #: it is running on would wait for a slot it is holding.
        self._parts: ThreadPoolExecutor | None = None
        self._futures: list = []
        self._t0 = 0.0
        # Set by the first failed fetch. Every queued entry checks it on entry
        # and gives up, because a failed GET is billed and the run is already
        # lost: without the scene the composite would be wrong, not late.
        self._stop = threading.Event()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, _exc, _tb):
        # Cancel on the way out of a failure. The alternative is fetching the
        # remaining thousands of objects for a run that has already lost a
        # scene and cannot produce a correct percentile.
        self.close(cancel=exc_type is not None)
        return False

    def start(self):
        """Put every object in flight and return immediately."""
        # One client, shared, sized to the connections the settings ask for.
        # See `_default_client`.
        factory = self._client_factory or _default_client
        client = factory(self.settings.connections)
        self._t0 = time.perf_counter()
        self._pool = ThreadPoolExecutor(max_workers=self.threads)
        if self.settings.multipart:
            # A fetch thread waits while its parts run, so the part pool holds
            # every socket in flight and the fetch pool holds none of them.
            self._parts = ThreadPoolExecutor(
                max_workers=self.threads * self.settings.part_concurrency,
                thread_name_prefix="stage-part",
            )
        for entry in self.manifest:
            future = self._pool.submit(self._fetch, client, entry)
            future.add_done_callback(self._record)
            self._futures.append(future)
        return self

    def _fetch(self, client, entry):
        item_id, band, href = entry
        if self._stop.is_set():
            msg = f"staging abandoned before {item_id} {band}: an earlier object failed"
            raise StagingError(msg)
        bucket, key = split_s3_uri(href)
        dest = staged_path(self.stage_dir, item_id, band, key)
        if dest.exists():
            # Already here, so it is complete. `_fetch_one` unlinks on every
            # failure path including a short read, which is what makes
            # existence enough to trust and a HEAD unnecessary. A HEAD is
            # billable, so checking would cost most of what skipping saves.
            #
            # This is the second traversal a tile now takes. `tile_prep` stages
            # the tile and `shard_lst_p95` stages the same objects behind it,
            # and refetching them cost 322 s and 297 GB on a mean tile, or 21%
            # of the run.
            written, requests, retries, reused = dest.stat().st_size, 0, 0, 1
        else:
            written, requests, retries = _fetch_one(
                client, bucket, key, dest, self.settings, self._parts
            )
            reused = 0
        with self._lock:
            self._counters["bytes"] += written
            self._counters["requests"] += requests
            self._counters["retries"] += retries
            self._counters["reused"] += reused
            free = shutil.disk_usage(self.stage_dir).free
        if free < FREE_SPACE_FLOOR_BYTES:
            msg = (
                f"{self.stage_dir} fell to {free / 1024**3:.1f} GiB free while "
                f"staging, below the {FREE_SPACE_FLOOR_BYTES / 1024**3:.0f} "
                f"GiB floor. The per-object estimate was too low for this slice."
            )
            raise StagingError(msg)
        return item_id, band

    def _record(self, future) -> None:
        """Runs on the fetch thread as each object finishes, good or bad."""
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            self._stop.set()
            self._queue.put(("error", exc))
        else:
            self._queue.put(("object", future.result()))

    def landed(self) -> list[tuple[str, str]]:
        """The `(item_id, band)` pairs that reached disk since the last call.

        Never blocks, so the caller can interleave it with gathering results.

        Raises:
            Whatever the fetch raised, for the first object that failed. A
            missing scene is not recoverable here: a run that kept computing
            without it would write a wrong percentile rather than fail.
        """
        out = []
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                return out
            self.outstanding -= 1
            if kind == "error":
                raise payload
            out.append(payload)

    @property
    def done(self) -> bool:
        return self.outstanding <= 0

    def drain(self) -> None:
        """Block until every object has landed. The serial path."""
        while not self.done:
            kind, payload = self._queue.get()
            self.outstanding -= 1
            if kind == "error":
                raise payload

    def close(self, *, cancel: bool = False) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=cancel)
            self._pool = None
            self.seconds = time.perf_counter() - self._t0
        if self._parts is not None:
            # After the fetch pool, never with it. A fetch thread on its way
            # out is still waiting on the parts it submitted.
            self._parts.shutdown(wait=True, cancel_futures=cancel)
            self._parts = None

    def report(self, *, reserved: int = 0, owns_stage_dir: bool = True) -> dict:
        """This run's S3 line, counted rather than derived.

        Objects, bytes, seconds, billable GETs, retries, how many objects were
        already on disk, the estimate the disk guard reserved, and the settings
        that produced all of it. `cost_report.py --s3-get-requests` prices
        `get_requests` directly, so these key names are an interface.

        `get_requests` and `retries` are counted separately rather than one
        derived from the other. They were the same number while every object
        took one GET; a ranged fetch issues one per part, so a clean run of
        6 parts would otherwise report 5 retries it never made.

        `settings` is what makes a throughput figure readable a month later.
        Two runs of the same tile at 344 and 900 MB/s say nothing without the
        thread, connection and part counts that separate them.
        """
        return {
            "objects": len(self.manifest),
            "bytes": self._counters["bytes"],
            "seconds": self.seconds,
            "get_requests": self._counters["requests"],
            "retries": self._counters["retries"],
            "reused": self._counters["reused"],
            "reserved_bytes": reserved,
            "stage_dir": str(self.stage_dir),
            "owns_stage_dir": owns_stage_dir,
            "settings": self.settings.as_dict(),
        }


def stage_scenes(
    item_dicts,
    indices,
    stage_dir: Path,
    *,
    threads: int | None = None,
    settings: FetchSettings | None = None,
    client_factory=None,
) -> dict:
    """Fetch every object the slice needs and repoint the items at local disk.

    The serial form: nothing else runs until the last object lands. An
    overlapped run drives `StagingRun` directly instead, so that it can submit
    work while the fetch continues.

    Mutates the asset hrefs of the staged items in place, through
    `repoint_items`, before the first GET rather than after the last one.

    `threads` and `settings` are the same argument at two sizes, and passing
    both raises. A caller with one number keeps passing it; a caller with a
    parsed command line builds a `FetchSettings` and passes that.

    Returns:
        The staging report. See `StagingRun.report`.
    """
    resolved = _resolve_settings(threads, settings)
    stage_dir = Path(stage_dir)
    # Whether this call owns the directory. `cleanup` removes what it is
    # given, and an operator pointing --stage-dir at an NVMe mount rather than
    # a directory on it would otherwise lose everything else there.
    pre_existing = stage_dir.exists() and any(stage_dir.iterdir())
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = repoint_items(item_dicts, indices, stage_dir)
    if not manifest:
        return _empty_report(stage_dir, resolved)
    reserved = disk_guard(manifest, stage_dir)

    run = StagingRun(
        manifest, stage_dir, settings=resolved, client_factory=client_factory
    )
    with run:
        run.drain()
    return run.report(reserved=reserved, owns_stage_dir=not pre_existing)


def _resolve_settings(threads, settings) -> FetchSettings:
    """One `FetchSettings` from the two ways a caller can ask for one.

    Raises:
        StagingError: if both are given. Two answers to the same question, and
            whichever one lost would lose in silence.
    """
    if settings is None:
        return FetchSettings.build(threads=threads)
    if threads is not None:
        msg = "staging takes threads or settings, not both"
        raise StagingError(msg)
    return settings


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


def _empty_report(stage_dir: Path, settings: FetchSettings) -> dict:
    return {
        "objects": 0,
        "bytes": 0,
        "seconds": 0.0,
        "get_requests": 0,
        "retries": 0,
        "reserved_bytes": 0,
        "stage_dir": str(stage_dir),
        "owns_stage_dir": True,
        "settings": settings.as_dict(),
    }


def cleanup(stage_dir: Path, *, owned: bool = True) -> None:
    """Remove the staged files.

    Steady state runs one tile per instance, back to back. A stage directory
    left behind fills the disk on the second tile.

    `owned` is what `stage_scenes` reports. This removes a whole directory
    tree, and `FINDINGS` tells an operator to point `--stage-dir` at an NVMe
    mount. A path one level up from the documented `/mnt/nvme/stage` would take
    everything else on the volume with it, so a directory that already held
    files is left alone and named.
    """
    stage_dir = Path(stage_dir)
    if not owned:
        print(
            f"stage         {stage_dir} held files before this run and is left "
            f"in place. Remove the staged scenes by hand, or point --stage-dir "
            f"at a directory this run creates."
        )
        return
    shutil.rmtree(stage_dir, ignore_errors=True)

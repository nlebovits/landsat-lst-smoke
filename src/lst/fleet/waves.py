"""Run many tiles as a sequence of waves, each one launched, driven, and torn down.

`launch.py` places instances. `drive.sh` starts one. `watch.py` reports. Each
does its own job and none of them runs a hundred tiles, so until now a
continent meant a hundred hand-typed commands. The largest fleet this
repository has ever run is four instances.

A wave is the unit. Launch `width` instances, drive them all at once, watch
until every one is finished or dead, tear the wave down, then start the next.
Nothing carries between waves except the queue and the width.

Three things here exist because of what the account and the pipeline are.

**The width is discovered, not configured.** `servicequotas:GetServiceQuota` is
denied for this role, so nobody can read the concurrent vCPU ceiling before
launching. The driver asks for `--width`, and if AWS refuses part way through
with `VcpuLimitExceeded` it keeps the instances it placed, runs that narrower
wave, and uses the width it achieved for every later wave. A refusal costs
nothing: `launch_one` deletes the key of the instance that was refused.

**A tile is judged by `summary.json`, not by an exit status.** A tile with no
thermal coverage writes a summary and exits 0. A tile whose instance died
writes nothing. `watch.classify` already tells those apart, so this module
calls it rather than reimplementing it.

**Teardown runs even when the wave fails.** An instance that finished and was
not terminated bills until its 75 minute deadline, which costs more than the
tile did. Every exit path through a wave passes through teardown.

    uv run lst-fleet-waves --tiles-file artifacts/south_america.txt \
        --width 20 --commit "$SHA"

## Refilling, for a run that cannot afford the wave barrier

A wave waits for its slowest tile. MEASURED on 2026-09-15: tiles inside one
wave finished between 8 and 44 minutes, so a 20-wide wave paid for about 20
machine-minutes of nothing per tile, and no new tile could start until the last
one landed. Over 101 tiles that was tolerable. Over 665 it is not.

`--refill` holds `--width` instances alive instead. A slot that frees is filled
on the next poll, so the wall clock is the total work divided by the width, and
the run costs its tiles rather than its slowest tiles.

    uv run lst-fleet-waves --refill --width 80 \
        --tiles-file artifacts/rest_of_world.txt --commit "$SHA"

Add `--dry-run` for the projected instance-hours, dollars, and wall clock at
several widths, from the cost model fitted on the South America run. See
`FIXED_SECONDS`.

Two things follow from a run that never tears down until the end. Instances are
placed in parallel, because 665 serial launches is 4.1 hours on its own: see
`LAUNCH_WORKERS`. And the run prices itself as it goes, because AWS forgets a
terminated instance after about an hour: see `COST_EVERY_MINUTES`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from lst.fleet import launch, watch
from lst.fleet.launch import (
    DEFAULT_FLEET_DIR,
    CapacityExhausted,
    QuotaExhausted,
    RunManifest,
    launch_one,
    load_config,
    parse_tiles,
    render_user_data,
    resolve_commit,
)

#: How long a wave may run before the driver stops waiting on it.
#:
#: MEASURED tile wall clock is 26 to 43 minutes, and `instance.deadline_minutes`
#: halts a box at 75. Waiting past the instance deadline waits on something
#: that cannot still be alive, so this matches it with a margin for the upload
#: that follows `all_done`.
WAVE_TIMEOUT_MINUTES = 85

#: How often the driver asks object storage where a wave has got to.
POLL_SECONDS = 60.0

#: Terminal failures from `watch.classify`. `hung` is not one: a tile with no
#: heartbeat for five minutes may still recover, and only the wave timeout
#: decides it never will.
TERMINAL_BAD = ("failed", "gone")

#: The phase `watch.classify` gives a tile whose `_MANIFEST.json` says the
#: upload finished. It is the only phase this driver accepts as done.
#:
#: `classify` also calls a tile `finished` on the `all_done` marker alone, with
#: the detail `waiting on upload`. That is right for a watcher, which only
#: reports. It is wrong for a driver, which terminates the instance next.
#: MEASURED on 2026-09-15: `N00W045` was torn down in that state and lost its
#: `_MANIFEST.json`. Its rasters had landed seconds earlier, so the loss was
#: one marker file. A tile whose 500 MB `qa_count.tif` was still going up would
#: have lost the raster.
#:
#: `fleet/README.md` states the rule this restores: `_MANIFEST.json` with
#: `complete: true` is the only proof an upload finished.
UPLOADED_PHASE = "uploaded"

#: How long a tile may sit at `all_done` without its manifest appearing.
#:
#: The uploader polls every 30 s, and one pass after `all_done` it writes the
#: manifest. A tile still waiting well past that is not uploading. Its instance
#: hit the deadline, or the uploader died, and either way the wave should stop
#: holding 19 other machines for it.
UPLOAD_GRACE_MINUTES = 12.0

#: States that send a tile back to the queue.
RETRYABLE = ("failed", "gone", "hung", "timeout", "not-driven", "upload-lost")

#: How many instances the driver places at the same time.
#:
#: MEASURED on 2026-09-15: a serial launcher placed one instance every 22
#: seconds. Almost all of that is `aws ec2 wait instance-running`, which polls
#: on a 15 second cycle, so the time is spent waiting rather than computing.
#: 665 tiles at 22 seconds each is 4.1 hours of launching alone.
#:
#: Eight is chosen against the waiter, not against the CPU. Eight concurrent
#: waiters bring the effective rate to about 3 seconds per instance, which puts
#: 665 launches under 35 minutes. Going wider trades against the EC2 API
#: request rate, which answers a burst and then throttles.
LAUNCH_WORKERS = 8

#: How often a run prices itself while it is still running.
#:
#: AWS answers `describe-instances` for a terminated instance for about an
#: hour, then drops it. A wave-based run tore down every 40 minutes and priced
#: itself inside that window. A refill run never tears down until the end, so a
#: five-hour run would reach its cost report with the first four hours already
#: unpriceable.
#:
#: Each snapshot lists every tagged instance AWS still remembers. The windows
#: overlap, so `total_from_snapshots` reads them all and counts each instance
#: id once.
COST_EVERY_MINUTES = 30.0

#: How long the driver waits before asking for capacity again after a region
#: refused every zone. Short enough to keep slots full, long enough that a
#: genuinely full region is not asked 60 times a minute.
CAPACITY_BACKOFF_SECONDS = 120.0


def is_done(state) -> bool:
    """Whether a wave may stop waiting on one tile.

    Success needs the manifest, not the marker. See `UPLOADED_PHASE`.
    """
    if state.status in TERMINAL_BAD:
        return True
    return state.status == "finished" and state.phase == UPLOADED_PHASE


def settle_uploads(
    statuses: dict[str, str],
    phases: dict[str, str],
    waiting_since: dict[str, float],
    now: float,
    *,
    grace_minutes: float = UPLOAD_GRACE_MINUTES,
) -> dict[str, str]:
    """Re-label a tile stuck at `all_done` whose manifest never arrived.

    A tile that reaches `all_done` and never publishes a manifest produced no
    proof that its rasters are whole, so it reads `upload-lost` and goes back
    to the queue rather than counting as finished.
    """
    settled = dict(statuses)
    for tile, status in statuses.items():
        if status != "finished" or phases.get(tile) == UPLOADED_PHASE:
            waiting_since.pop(tile, None)
            continue
        started = waiting_since.setdefault(tile, now)
        if (now - started) > grace_minutes * 60:
            settled[tile] = "upload-lost"
    return settled


def say_now(*args) -> None:
    """Print and flush.

    A driver redirected to a file writes nothing for an hour otherwise. Python
    buffers stdout when it is not a terminal, and this module's own progress
    lines are the only view anyone has of a running wave. MEASURED on
    2026-09-15: an adopted 20-tile wave ran for minutes with an empty log,
    because the launch path happens to call `print(flush=True)` and the adopt
    path calls nothing that flushes.
    """
    print(*args, flush=True)


def load_tiles(path: Path | str) -> list[str]:
    """Tile ids from a file, one per line, blanks and `#` comments ignored.

    Validated through `launch.parse_tiles`, which checks every id against the
    5 degree grid. One wrong id otherwise costs a whole instance: the box
    launches, clones, downloads 229 MB of artifacts, and fails minutes into
    billing.
    """
    lines = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return parse_tiles(lines)


def drive(
    fleet_dir: Path, manifest: Path, tile: str, log_dir: Path
) -> subprocess.Popen:
    """Start `drive.sh` for one tile, with its output on disk.

    `drive.sh` polls for sshd and for `/mnt/nvme/SETUP_DONE` on its own, so the
    driver starts all of a wave's instances at once and lets each one wait for
    its own box. Output goes to a file per tile rather than to a shared pipe,
    because twenty interleaved poll loops on one terminal are unreadable and a
    full pipe buffer would block a driver nobody is reading.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"{tile}.drive.log").open("w")
    return subprocess.Popen(
        [str(fleet_dir / "drive.sh"), str(manifest), tile],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


class Placement:
    """What one parallel launch attempt produced.

    Attributes:
        placed: the manifest entries of the instances that exist now.
        refused: the tiles AWS would not place, in the order they were asked.
        reasons: the AWS error code per refused tile.
        quota: whether a refusal was account-wide rather than a full zone.
    """

    def __init__(self):
        self.placed: list[dict] = []
        self.refused: list[str] = []
        self.reasons: dict[str, str] = {}
        self.quota = False


def launch_many(
    tiles: list[str],
    *,
    cfg: dict,
    run_id: str,
    user_data: Path,
    manifest: RunManifest,
    workers: int = LAUNCH_WORKERS,
    say=say_now,
) -> Placement:
    """Place several instances at once, and report what AWS allowed.

    `launch_one` spends most of its time in `aws ec2 wait instance-running`,
    so the launches overlap almost perfectly. `RunManifest` takes a lock for
    this, because every thread writes the same file.

    A quota refusal stops the rest. It is an account-and-region fact, so the
    tiles not yet started would meet it too, and asking costs a key pair each.
    A capacity refusal does not stop anything: it names one instance type in
    four zones at one moment, and the next tile may still be placed.

    Any other AWS error still raises `SystemExit` out of its thread and through
    this function. Instances placed by the other threads are already in the
    manifest, so teardown can still reach them.
    """
    stop = threading.Event()
    result = Placement()
    lock = threading.Lock()

    def one(tile: str) -> None:
        if stop.is_set():
            with lock:
                result.refused.append(tile)
            return
        try:
            entry = launch_one(cfg, tile, run_id, False, user_data, manifest)
        except QuotaExhausted as err:
            stop.set()
            with lock:
                result.quota = True
                result.refused.append(tile)
                result.reasons[tile] = err.code
            return
        except CapacityExhausted as err:
            with lock:
                result.refused.append(tile)
                result.reasons[tile] = err.code
            say(f"{tile}  no capacity in any zone, back to the queue")
            return
        with lock:
            result.placed.append(entry)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # `list` so an exception set on a future is raised here rather than
        # discarded. A malformed request must still stop the run.
        list(pool.map(one, tiles))
    return result


def poll_states(run: dict, s3, ec2) -> list[watch.State]:
    """Where every tile in one manifest has got to, from object storage.

    Delegates to `watch.classify` rather than reading markers again here. That
    function is pure and tested, and it already knows that staging writes no
    marker for 19 minutes.
    """
    cfg = run["config"]
    bucket = cfg["storage"]["bucket"]
    runs_prefix = cfg["storage"]["runs_prefix"]

    def fetch(key: str) -> str | None:
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        except Exception:
            return None

    alive = watch.running_instances(
        ec2,
        [e["instance_id"] for e in run["instances"] if e.get("instance_id")],
        profile=cfg["aws"]["profile"],
        region=cfg["aws"]["region"],
    )
    now = datetime.now(timezone.utc)
    states = []
    for e in run["instances"]:
        prefix = f"{runs_prefix}/{e['name']}"
        states.append(
            watch.classify(
                e["tile"],
                fetch(f"{prefix}/markers.txt"),
                fetch(f"{prefix}/_MANIFEST.json") is not None,
                None if alive is None else e.get("instance_id") in alive,
                now,
                heartbeat=fetch(f"{prefix}/heartbeat.txt"),
            )
        )
    return states


def teardown(manifest: Path, *, skip_cost: bool = False) -> int:
    """Terminate one wave and price it, as a subprocess.

    A subprocess so that a teardown that raises cannot take the driver down
    with it and leave the rest of the wave billing. The exit status is
    reported, never acted on: there is nothing useful the driver can do about a
    refused termination that `teardown.py`'s own message does not already say.
    """
    argv = [sys.executable, "-m", "lst.fleet.teardown", "--manifest", str(manifest)]
    if skip_cost:
        argv.append("--skip-cost")
    out = subprocess.run(argv, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    return out.returncode


def run_wave(
    tiles: list[str],
    *,
    cfg: dict,
    commit: str,
    fleet_dir: Path,
    manifest_dir: Path,
    user_data: Path,
    poll_seconds: float,
    timeout_minutes: float,
    workers: int = LAUNCH_WORKERS,
    say=say_now,
) -> tuple[dict[str, str], int, Path]:
    """One wave: launch, drive, watch, tear down.

    Returns:
        `(status_by_tile, placed, manifest_path)`. A status is a
        `watch.classify` status, or `timeout` when the wave ran out of time,
        or `not-driven` when `drive.sh` never started. `placed` is how many
        instances AWS actually allowed, which is below `len(tiles)` only when
        quota refused the rest.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_path = manifest_dir / f"run-{run_id}.json"
    manifest = RunManifest(
        manifest_path,
        {"run_id": run_id, "commit": commit, "tiles": tiles, "config": cfg},
    )
    say(f"manifest {manifest_path}")

    launched: list[dict] = []
    refused: list[str] = []
    try:
        result = launch_many(
            tiles,
            cfg=cfg,
            run_id=run_id,
            user_data=user_data,
            manifest=manifest,
            workers=workers,
            say=say,
        )
        launched, refused = result.placed, result.refused
        if refused:
            reason = "quota" if result.quota else "capacity"
            say(
                f"\n{reason} stopped the wave at {len(launched)} instance(s). "
                f"Returning {len(refused)} tile(s) to the queue."
            )

        if not launched:
            return ({t: "not-driven" for t in tiles}, 0, manifest_path)

        manifest.header["tiles"] = [e["tile"] for e in launched]
        manifest.write()

        log_dir = manifest_dir / f"logs-{run_id}"
        drivers = {
            e["tile"]: drive(fleet_dir, manifest_path, e["tile"], log_dir)
            for e in launched
        }
        say(f"drove {len(drivers)} tile(s), logs in {log_dir}")

        statuses = _await_wave(
            manifest_path,
            cfg,
            poll_seconds=poll_seconds,
            timeout_minutes=timeout_minutes,
            say=say,
        )
        for tile, proc in drivers.items():
            if proc.poll() is None:
                proc.terminate()
            elif proc.returncode != 0 and statuses.get(tile) == "starting":
                statuses[tile] = "not-driven"
    finally:
        # Every path out of a wave terminates it. An instance that finished and
        # was not torn down bills to its 75 minute deadline.
        say("\ntearing down the wave")
        teardown(manifest_path)

    for tile in refused:
        statuses[tile] = "not-driven"
    return statuses, len(launched), manifest_path


def _await_wave(
    manifest_path: Path,
    cfg: dict,
    *,
    poll_seconds: float,
    timeout_minutes: float,
    say=say_now,
) -> dict[str, str]:
    """Poll until every tile is terminal or the wave runs out of time."""
    import boto3

    s3 = boto3.Session(profile_name=cfg["storage"].get("upload_profile")).client(
        "s3", region_name=cfg["aws"]["region"]
    )
    ec2 = boto3.Session(profile_name=cfg["aws"]["profile"]).client(
        "ec2", region_name=cfg["aws"]["region"]
    )

    deadline = time.monotonic() + timeout_minutes * 60
    waiting_since: dict[str, float] = {}
    reaped: set[str] = set()
    statuses: dict[str, str] = {}
    while True:
        run = json.loads(manifest_path.read_text())
        states = poll_states(run, s3, ec2)
        now = time.monotonic()
        statuses = settle_uploads(
            {s.tile: s.status for s in states},
            {s.tile: s.phase for s in states},
            waiting_since,
            now,
        )
        reaped |= reap(run, cfg, states, reaped, say=say)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        tally = {
            k: sum(1 for v in statuses.values() if v == k)
            for k in sorted(set(statuses.values()))
        }
        say(f"{stamp}  " + "  ".join(f"{k}={v}" for k, v in tally.items()))
        for s in states:
            if statuses[s.tile] != "running":
                say(
                    f"           {s.tile:9} {statuses[s.tile]:11} "
                    f"{s.phase:16} {s.detail}"
                )
        # A tile at `all_done` without its manifest is still uploading. Waiting
        # costs the wave a minute. Terminating costs the tile its rasters.
        if all(is_done(s) or statuses[s.tile] == "upload-lost" for s in states):
            return statuses
        if now > deadline:
            say(f"\nwave ran past {timeout_minutes:.0f} minutes, giving up on it")
            done = {s.tile for s in states if is_done(s)}
            return {
                tile: (status if tile in done or status == "upload-lost" else "timeout")
                for tile, status in statuses.items()
            }
        time.sleep(poll_seconds)


def uploaded_tiles(cfg: dict) -> set[str]:
    """Every tile with a `_MANIFEST.json` under the runs prefix.

    The question a relaunch has to answer is which tiles are actually done, and
    the only honest source is the bucket. A driver's own tally dies with the
    driver, a manifest on disk records what was launched rather than what
    landed, and a tile that ran twice appears under two run prefixes.

    A run prefix is named `lst-<TILE>-<run id>`, so the tile is read back out of
    the prefix rather than by opening 100 JSON files.
    """
    bucket = cfg["storage"]["bucket"]
    prefix = cfg["storage"]["runs_prefix"].rstrip("/")
    out = subprocess.run(
        ["aws", "s3", "ls", "--recursive", f"s3://{bucket}/{prefix}/"],
        capture_output=True,
        text=True,
        env=os.environ | {"AWS_PROFILE": cfg["storage"].get("upload_profile", "")},
    )
    if out.returncode != 0:
        raise SystemExit(f"cannot list the runs prefix:\n{out.stderr.strip()}")
    done = set()
    for line in out.stdout.splitlines():
        key = line.split()[-1]
        if not key.endswith("/_MANIFEST.json"):
            continue
        name = key.removeprefix(prefix + "/").split("/", 1)[0]
        parts = name.split("-")
        if len(parts) >= 2 and parts[0] == "lst":
            done.add(parts[1])
    return done


def terminate_one(cfg: dict, instance_id: str) -> tuple[int, str]:
    """Stop one instance billing. Returns `(returncode, message)`."""
    code, _, err = launch.aws_try(
        [
            "aws",
            "ec2",
            "terminate-instances",
            "--profile",
            cfg["aws"]["profile"],
            "--region",
            cfg["aws"]["region"],
            "--instance-ids",
            instance_id,
        ]
    )
    return code, err


def reap(run: dict, cfg: dict, states: list, reaped: set[str], say=say_now) -> set[str]:
    """Terminate each instance whose upload is proven, without waiting for the wave.

    A wave costs the lifetime of its slowest tile times its width, not the sum
    of its tiles. MEASURED on 2026-09-15: wave 1 held 20 machines for an
    average of 38.5 minutes each while its tiles finished between 16:20 and
    16:50. The first tile done paid for 30 minutes of doing nothing, and the
    wave cost $2.44 a tile against the $1.35 a three-tile wave measured.

    Only a tile with its `_MANIFEST.json` is reaped. A failed tile is left for
    the wave teardown, because its uploader may still be pushing the log that
    says why it failed, and that log is the whole value of a failed tile.

    Returns:
        The names newly reaped, to be added to `reaped` by the caller.
    """
    by_tile = {e["tile"]: e for e in run["instances"]}
    fresh = set()
    for state in states:
        entry = by_tile.get(state.tile)
        if entry is None or entry["name"] in reaped:
            continue
        if not (state.status == "finished" and state.phase == UPLOADED_PHASE):
            continue
        instance_id = entry.get("instance_id")
        if not instance_id:
            continue
        code, err = terminate_one(cfg, instance_id)
        if code != 0:
            say(f"           {state.tile:9} could not terminate early: {err}")
            continue
        fresh.add(entry["name"])
        say(f"           {state.tile:9} uploaded, terminated {instance_id}")
    return fresh


#: One priced instance line out of a `lst-cost-report` listing.
INSTANCE_LINE = re.compile(r"^(i-[0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\d+)\s*$")

#: The derived line that names the hourly rate the report used.
RATE_LINE = re.compile(r"^\s*(\S+)\s+\d+s / 3600 x \$([0-9.]+)")

#: Consecutive polls with nothing running and nothing placeable before the
#: driver stops waiting. At the default 60 second poll this is 20 minutes.
STALL_LIMIT = 20

#: How long a failed tile's instance is left alive before it is terminated.
#:
#: `reap` deliberately leaves a failed instance alone, because its uploader may
#: still be pushing the log that says why it failed, and that log is the whole
#: value of a failed tile. Under waves, the wave teardown collected it minutes
#: later. A refill run has no wave teardown, so without this a failed instance
#: bills to its 75 minute deadline while its tile runs again somewhere else.
#:
#: MEASURED on 2026-09-16: 60 tiles failed at once, left the pool, and kept
#: both their billing and their share of the account vCPU, which then refused
#: every replacement.
FAILED_GRACE_MINUTES = 3.0


#: MEASURED cost model, fitted on 2026-09-16 over the 99 South America tiles
#: that ran with early reaping:
#:
#:     instance seconds = 816.6 + 0.29843 x scenes      (R^2 = 0.666)
#:
#: The constant is boot, clone, artifact download, and shutdown. It is work no
#: machine size shortens and no tile avoids, and at 665 tiles it is 151 of the
#: 366 projected instance-hours. The slope is staging and compositing.
#:
#: A projection is a projection. The fit explains two thirds of the variance,
#: so read the output as plus or minus about 15%.
FIXED_SECONDS = 816.6
SECONDS_PER_SCENE = 0.29843

#: MEASURED on-demand rate for `m6id.16xlarge` in `us-west-2`, from the AWS
#: public price list, the same source `cost_report` verifies against.
HOURLY_USD = 3.7968

#: Where `plan_text` reads scene counts from. Each entry carries `tile_id` and
#: `scenes`, which is what the cost model needs.
DEFAULT_PLAN_FILE = Path("artifacts/fleet_plan.json")


def scene_counts(path: Path) -> dict[str, int]:
    """Scenes per tile, or an empty mapping when the plan file is absent.

    A missing plan file must not stop a launch. It only costs the projection.
    """
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return {t["tile_id"]: t["scenes"] for t in plan.get("tiles", [])}


def project(
    tiles: list[str], scenes: dict[str, int], width: int
) -> tuple[float, float, float]:
    """Instance-hours, dollars, and wall-clock hours for a refill run.

    Wall clock is the total work divided by the width. That is what refilling
    buys: with no wave barrier, a slot that frees is filled at once, so the
    machines stay busy until the queue is empty. A ramp of one launch round is
    added, because the pool has to fill before it can be held full.

    A tile with no scene count is priced at the mean of the tiles that have
    one, so an unknown tile is not silently free.
    """
    known = [scenes[t] for t in tiles if t in scenes]
    mean = sum(known) / len(known) if known else 0.0
    seconds = sum(
        FIXED_SECONDS + SECONDS_PER_SCENE * scenes.get(t, mean) for t in tiles
    )
    hours = seconds / 3600
    ramp = FIXED_SECONDS / 3600
    return hours, hours * HOURLY_USD, hours / max(width, 1) + ramp


def plan_text(tiles: list[str], a) -> str:
    """What a dry run prints: the shape of the run, and what it should cost."""
    lines = []
    if a.refill:
        scenes = scene_counts(a.plan_file)
        hours, dollars, wall = project(tiles, scenes, a.width)
        missing = [t for t in tiles if t not in scenes]
        lines.append(
            f"  pool {a.width}, placed {a.launch_workers} at a time, "
            f"cost snapshot every {a.cost_every:.0f} min"
        )
        lines.append(
            f"  PROJECTED {hours:.0f} instance-hours, ${dollars:.0f}, "
            f"{wall:.1f} h wall clock at width {a.width}"
        )
        lines.append("  projection is +/- 15%. Model R^2 is 0.666 on n=99.")
        if missing:
            lines.append(
                f"  {len(missing)} tile(s) have no scene count and were "
                f"priced at the mean"
            )
        for other in (40, 60, 80, 100):
            if other != a.width:
                _, _, w = project(tiles, scenes, other)
                lines.append(f"    width {other:3d}: {w:.1f} h")
        return "\n".join(lines)
    waves = -(-len(tiles) // a.width)
    for i in range(waves):
        batch = tiles[i * a.width : (i + 1) * a.width]
        lines.append(f"  wave {i + 1}: {len(batch)}  {' '.join(batch)}")
    return "\n".join(lines)


def cost_snapshot(cfg: dict, path: Path, say=say_now) -> bool:
    """Append a priced listing of every tagged instance AWS still remembers.

    See `COST_EVERY_MINUTES` for why a long run has to do this while it runs.
    """
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "lst.fleet.cost_report",
            "--tag",
            f"purpose={cfg['tags']['purpose']}",
            "--region",
            cfg["aws"]["region"],
            "--profile",
            cfg["aws"]["profile"],
        ],
        capture_output=True,
        text=True,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a") as fh:
        fh.write(f"\n=== snapshot {stamp} ===\n")
        fh.write(out.stdout)
        fh.write(out.stderr)
    return out.returncode == 0


def total_from_snapshots(path: Path) -> tuple[int, int, float]:
    """Price a whole run out of its overlapping snapshots.

    Snapshots overlap, because each one lists every instance AWS still
    remembers rather than only the new ones. Summing the printed totals
    therefore counts most instances several times. MEASURED on 2026-09-15:
    summing six wave reports gave $346 against a true $195.

    Each instance id is counted once, at the largest seconds any snapshot
    reported for it. Instance seconds only grow, so the largest is the final
    one.

    Returns:
        `(instances, seconds, dollars)`.
    """
    seconds: dict[str, int] = {}
    kind: dict[str, str] = {}
    rate: dict[str, float] = {}
    for line in path.read_text().splitlines():
        found = INSTANCE_LINE.match(line)
        if found:
            iid, typ, sec = found.groups()
            seconds[iid] = max(seconds.get(iid, 0), int(sec))
            kind[iid] = typ
            continue
        found = RATE_LINE.match(line)
        if found:
            rate[found.group(1)] = float(found.group(2))
    dollars = sum(sec / 3600 * rate.get(kind[iid], 0.0) for iid, sec in seconds.items())
    return len(seconds), sum(seconds.values()), dollars


class Gate:
    """How many more instances the driver may ask for right now.

    Two refusals mean different things, so the gate answers them differently.

    A quota refusal is an account-and-region fact. The gate records the number
    that was allowed and holds to it, which stops the driver asking every
    minute for something the account has already refused.

    A capacity refusal is one instance type in four zones at one moment. The
    gate waits `CAPACITY_BACKOFF_SECONDS` and asks again.

    Both expire. An account that refused 60 instances may have been busy with
    somebody else's work. An earlier wave driver kept the narrowed width for
    the rest of the run, which would have run a continent at a number
    discovered during one overlap.
    """

    def __init__(self, width: int):
        self.width = width
        self.ceiling = 0
        self.ceiling_until = 0.0
        self.hold_until = 0.0

    def room(self, live: int, now: float) -> int:
        """How many instances may be asked for, given what is already running."""
        if now < self.hold_until:
            return 0
        cap = self.width
        if self.ceiling and now < self.ceiling_until:
            cap = min(self.width, self.ceiling)
        return max(cap - live, 0)

    def observe(self, placement: Placement, live: int, now: float, say=say_now) -> None:
        """Learn from what AWS just allowed.

        A quota refusal with nothing running is not a ceiling of zero. It means
        something outside this pool holds the account's vCPU: instances this
        driver has condemned but not yet terminated, or another run. MEASURED
        on 2026-09-16: 60 instances failed at once, left the pool, and kept
        their quota for a minute, and the gate wrote `account ceiling met at 0
        concurrent instance(s)`. A ceiling of zero says never launch again.
        """
        if placement.quota and live > 0:
            self.ceiling = live
            self.ceiling_until = now + CAPACITY_BACKOFF_SECONDS
            say(f"account ceiling met at {live} concurrent instance(s)")
        elif placement.quota:
            self.hold_until = now + CAPACITY_BACKOFF_SECONDS
            say(
                f"the account vCPU is held by something outside this pool. "
                f"Asking again in {CAPACITY_BACKOFF_SECONDS / 60:.0f} minute(s)."
            )
        elif placement.refused and not placement.placed:
            self.hold_until = now + CAPACITY_BACKOFF_SECONDS
            say(
                f"every zone is full. Asking again in "
                f"{CAPACITY_BACKOFF_SECONDS / 60:.0f} minute(s)."
            )


class Pool:
    """The instances running right now, and the queue waiting for a slot.

    Split out of `refill` so that filling a slot and retiring a tile are each
    one readable function that a test can drive without AWS.
    """

    def __init__(
        self, tiles: list[str], *, fleet_dir: Path, manifest_path: Path, log_dir: Path
    ):
        self.queue: list[str] = list(tiles)
        self.live: dict[str, dict] = {}
        self.drivers: dict[str, subprocess.Popen] = {}
        self.started: dict[str, float] = {}
        self.attempts: dict[str, int] = dict.fromkeys(tiles, 0)
        self.settled: dict[str, str] = {}
        self.waiting_since: dict[str, float] = {}
        self.reaped: set[str] = set()
        #: instance id -> (tile, monotonic deadline). See `FAILED_GRACE_MINUTES`.
        self.condemned: dict[str, tuple[str, float]] = {}
        self.peak = 0
        self.fleet_dir = fleet_dir
        self.manifest_path = manifest_path
        self.log_dir = log_dir

    def busy(self) -> bool:
        return bool(self.queue or self.live)

    def view(self, cfg: dict) -> dict:
        """The live instances shaped as a manifest, for `poll_states` and `reap`.

        Polling reads three objects per instance. Passing the whole run would
        make every poll proportional to the 665 tiles already finished rather
        than to the width.
        """
        return {"config": cfg, "instances": list(self.live.values())}

    def fill(
        self,
        count: int,
        *,
        cfg,
        run_id,
        user_data,
        manifest,
        workers,
        drive_fn=None,
        say=say_now,
    ) -> Placement:
        """Place up to `count` instances and start a driver for each.

        Refused tiles keep their place at the head of the queue.
        """
        drive_fn = drive_fn or drive
        batch = self.queue[:count]
        del self.queue[:count]
        say(
            f"\nfilling {len(batch)} slot(s): {len(self.live)} live, "
            f"{len(self.queue)} queued"
        )
        placement = launch_many(
            batch,
            cfg=cfg,
            run_id=run_id,
            user_data=user_data,
            manifest=manifest,
            workers=workers,
            say=say,
        )
        for entry in placement.placed:
            tile = entry["tile"]
            self.live[tile] = entry
            self.started[tile] = time.monotonic()
            self.drivers[tile] = drive_fn(
                self.fleet_dir, self.manifest_path, tile, self.log_dir
            )
        self.queue[:0] = placement.refused
        self.peak = max(self.peak, len(self.live))
        return placement

    def retire(
        self,
        states: list,
        statuses: dict[str, str],
        now: float,
        *,
        timeout_minutes: float,
        retries: int,
        say=say_now,
    ) -> None:
        """Move every terminal tile out of the pool, and requeue what may retry.

        A tile leaves only when its upload is proven, when its manifest never
        came, or when it ran past the deadline. See `UPLOADED_PHASE`.
        """
        for state in states:
            tile = state.tile
            status = statuses[tile]
            finished = is_done(state) or status == "upload-lost"
            over = (now - self.started[tile]) > timeout_minutes * 60
            if not finished and not over:
                continue
            if not finished:
                status = "timeout"
                say(f"           {tile:9} ran past {timeout_minutes:.0f} min")
            proc = self.drivers.pop(tile, None)
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                elif proc.returncode != 0 and status == "starting":
                    status = "not-driven"
            entry = self.live.pop(tile, None)
            self.waiting_since.pop(tile, None)
            if status != "finished" and entry and entry.get("instance_id"):
                self.condemned[entry["instance_id"]] = (
                    tile,
                    now + FAILED_GRACE_MINUTES * 60,
                )
            one, again = sort_wave(
                [tile],
                {tile: status},
                attempts=self.attempts,
                retries=retries,
                say=say,
            )
            self.settled |= one
            self.queue.extend(again)

    def sweep(self, cfg: dict, now: float, say=say_now) -> int:
        """Terminate the failed instances whose grace has run out.

        Returns:
            How many were stopped.
        """
        due = [i for i, (_, when) in self.condemned.items() if now >= when]
        stopped = 0
        for instance_id in due:
            tile, _ = self.condemned[instance_id]
            code, err = terminate_one(cfg, instance_id)
            if code != 0:
                say(f"           {tile:9} could not terminate {instance_id}: {err}")
                continue
            del self.condemned[instance_id]
            stopped += 1
            say(f"           {tile:9} failed, terminated {instance_id}")
        return stopped

    def stop_drivers(self) -> None:
        for proc in self.drivers.values():
            if proc.poll() is None:
                proc.terminate()


def refill(
    tiles: list[str],
    *,
    cfg: dict,
    commit: str,
    fleet_dir: Path,
    manifest_dir: Path,
    user_data: Path,
    width: int,
    poll_seconds: float,
    timeout_minutes: float,
    retries: int,
    workers: int = LAUNCH_WORKERS,
    cost_every_minutes: float = COST_EVERY_MINUTES,
    say=say_now,
) -> tuple[dict[str, str], list[str], Path, Path]:
    """Hold `width` instances alive until the queue empties. No wave barrier.

    A wave costs the lifetime of its slowest tile, times its width. MEASURED on
    2026-09-15: tile wall clock ranged from 8 to 44 minutes inside one wave, so
    a 20-wide wave paid for about 20 machine-minutes of nothing per tile. Early
    reaping cut that to $1.87 a tile from $2.23, but the wave still could not
    start a new tile until its last one landed.

    Refilling removes the barrier. A slot that frees is filled from the queue
    on the next poll, so the run costs its tiles and nothing else, and its wall
    clock is the total work divided by the width.

    Everything a wave does correctly is kept. `reap` still proves an upload
    before terminating. `settle_uploads` still refuses to call a tile finished
    on the marker alone. `sort_wave` still decides retries, one tile at a time
    here instead of a batch at a time. Teardown still runs on every exit path.

    Returns:
        `(settled, leftover, manifest_path, cost_path)`. `leftover` is
        non-empty only when the run stalled for `STALL_LIMIT` polls.
    """
    import boto3

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_path = manifest_dir / f"run-{run_id}.json"
    cost_path = manifest_dir / f"run-{run_id}.snapshots.txt"
    manifest = RunManifest(
        manifest_path,
        {"run_id": run_id, "commit": commit, "tiles": tiles, "config": cfg},
    )
    pool = Pool(
        tiles,
        fleet_dir=fleet_dir,
        manifest_path=manifest_path,
        log_dir=manifest_dir / f"logs-{run_id}",
    )
    say(f"manifest {manifest_path}")
    say(f"cost snapshots {cost_path}")
    say(f"logs {pool.log_dir}")

    s3 = boto3.Session(profile_name=cfg["storage"].get("upload_profile")).client(
        "s3", region_name=cfg["aws"]["region"]
    )
    ec2 = boto3.Session(profile_name=cfg["aws"]["profile"]).client(
        "ec2", region_name=cfg["aws"]["region"]
    )

    gate = Gate(width)
    stalled = 0
    next_cost = time.monotonic() + cost_every_minutes * 60

    try:
        while pool.busy():
            # A condemned instance still holds its vCPU until it is swept, so
            # it counts against the ceiling. Asking for a slot it holds buys a
            # refusal and a wasted key pair.
            room = gate.room(len(pool.live) + len(pool.condemned), time.monotonic())
            if room > 0 and pool.queue:
                placement = pool.fill(
                    room,
                    cfg=cfg,
                    run_id=run_id,
                    user_data=user_data,
                    manifest=manifest,
                    workers=workers,
                    say=say,
                )
                gate.observe(placement, len(pool.live), time.monotonic(), say=say)
                stalled = 0 if placement.placed else stalled + 1
            elif not pool.live and pool.queue:
                stalled += 1

            if pool.live:
                stalled = 0
                view = pool.view(cfg)
                states = poll_states(view, s3, ec2)
                now = time.monotonic()
                statuses = settle_uploads(
                    {s.tile: s.status for s in states},
                    {s.tile: s.phase for s in states},
                    pool.waiting_since,
                    now,
                )
                pool.reaped |= reap(view, cfg, states, pool.reaped, say=say)
                pool.retire(
                    states,
                    statuses,
                    now,
                    timeout_minutes=timeout_minutes,
                    retries=retries,
                    say=say,
                )

            pool.sweep(cfg, time.monotonic(), say=say)

            stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            say(
                f"{stamp}  live={len(pool.live)} queued={len(pool.queue)} "
                f"done={len(pool.settled)}/{len(tiles)} peak={pool.peak} "
                f"condemned={len(pool.condemned)}"
            )

            if time.monotonic() >= next_cost:
                cost_snapshot(cfg, cost_path, say=say)
                next_cost = time.monotonic() + cost_every_minutes * 60

            if stalled >= STALL_LIMIT:
                say(
                    f"\nnothing placed and nothing running for {stalled} polls. "
                    f"Stopping with {len(pool.queue)} tile(s) unrun."
                )
                break
            if pool.busy():
                time.sleep(poll_seconds)
    finally:
        pool.stop_drivers()
        say("\nterminating anything still alive")
        teardown(manifest_path, skip_cost=True)
        cost_snapshot(cfg, cost_path, say=say)

    return pool.settled, pool.queue, manifest_path, cost_path


def report_remaining(tiles: list[str], cfg: dict) -> int:
    """Print the tiles with no upload manifest, one per line, and nothing else.

    The count goes to stderr so the tile list on stdout pipes straight into a
    file or into `--tiles-file`.
    """
    done = uploaded_tiles(cfg)
    left = [t for t in tiles if t not in done]
    print(
        f"# {len(tiles)} asked, {len(tiles) - len(left)} uploaded, {len(left)} left",
        file=sys.stderr,
    )
    for tile in left:
        print(tile)
    return 0


def adopt(
    manifest_path: Path,
    *,
    poll_seconds: float = POLL_SECONDS,
    timeout_minutes: float = WAVE_TIMEOUT_MINUTES,
    say=say_now,
) -> dict[str, str]:
    """Finish a wave whose driver is gone, then tear it down.

    A driver that dies leaves its instances running and billing to the 75
    minute deadline, with nothing watching them and no cost report. This
    happened on 2026-09-15: the driver had to be killed mid-wave to stop it
    terminating instances on the marker instead of the manifest, and 20
    machines were left with no owner.

    The manifest on disk holds everything needed to take the wave over. It
    names the instances, their tiles, and the bucket the uploader writes to.

    Returns:
        The final status of each tile.
    """
    run = json.loads(manifest_path.read_text())
    cfg = run["config"]
    tiles = [e["tile"] for e in run["instances"]]
    say(f"adopting {len(tiles)} tile(s) from {manifest_path}")
    try:
        return _await_wave(
            manifest_path,
            cfg,
            poll_seconds=poll_seconds,
            timeout_minutes=timeout_minutes,
            say=say,
        )
    finally:
        say("\ntearing down the adopted wave")
        teardown(manifest_path)


def sort_wave(
    batch: list[str],
    statuses: dict[str, str],
    *,
    attempts: dict[str, int],
    retries: int,
    say=say_now,
) -> tuple[dict[str, str], list[str]]:
    """Split one wave's results into settled tiles and tiles to run again.

    A tile is settled when it finished, when its status is not one a rerun
    could fix, or when it has used its retries. Returns
    `(settled, requeued)`.
    """
    settled: dict[str, str] = {}
    requeued: list[str] = []
    for tile in batch:
        status = statuses.get(tile, "not-driven")
        if status == "finished":
            settled[tile] = status
            continue
        attempts[tile] += 1
        if status in RETRYABLE and attempts[tile] <= retries:
            say(f"  {tile} {status}, back to the queue (attempt {attempts[tile]})")
            requeued.append(tile)
        else:
            settled[tile] = status
            say(f"  {tile} {status}, giving up after {attempts[tile]} attempt(s)")
    return settled, requeued


def drain(
    tiles: list[str],
    a,
    *,
    cfg: dict,
    commit: str,
    user_data: Path,
    say=say_now,
) -> tuple[dict[str, str], list[str], int, list[Path]]:
    """Run waves until the queue empties or `--max-waves` stops it.

    Returns:
        `(settled, leftover, waves_run, manifests)`. `leftover` is non-empty
        only when `--max-waves` cut the run short.
    """
    queue = list(tiles)
    attempts: dict[str, int] = dict.fromkeys(tiles, 0)
    done: dict[str, str] = {}
    # Every wave asks for the full width again. An earlier version kept the
    # narrower number a refusal produced, which is wrong whenever the account
    # was busy for a reason that later goes away: another wave of this
    # operator's own still draining, or somebody else's instances. It would
    # have run a continent at a width discovered during one overlap. A refusal
    # costs three API calls and the key it made is deleted, so asking again is
    # cheaper than guessing low for four hours.
    wave_no = 0
    manifests: list[Path] = []

    while queue:
        wave_no += 1
        if a.max_waves and wave_no > a.max_waves:
            say(f"\nstopping after {a.max_waves} wave(s), {len(queue)} tile(s) left")
            wave_no -= 1
            break
        batch, queue = queue[: a.width], queue[a.width :]
        say(f"\n=== wave {wave_no}: {len(batch)} tile(s) === {' '.join(batch)}")

        statuses, placed, manifest_path = run_wave(
            batch,
            cfg=cfg,
            commit=commit,
            fleet_dir=a.fleet_dir,
            manifest_dir=a.manifest_dir,
            user_data=user_data,
            poll_seconds=a.poll_seconds,
            timeout_minutes=a.timeout_minutes,
            workers=a.launch_workers,
            say=say,
        )
        manifests.append(manifest_path)

        if 0 < placed < len(batch):
            say(
                f"quota allowed {placed} of {len(batch)} this wave. "
                f"The next wave asks for {a.width} again."
            )

        settled, requeued = sort_wave(
            batch, statuses, attempts=attempts, retries=a.retries, say=say
        )
        done |= settled
        queue.extend(requeued)

    return done, queue, wave_no, manifests


def report(
    headline: str,
    done: dict[str, str],
    leftover: list[str],
    manifests: list[Path],
) -> int:
    """Print what landed and what did not. Non-zero when anything did not."""
    finished = sorted(t for t, s in done.items() if s == "finished")
    broken = sorted((t, s) for t, s in done.items() if s != "finished")
    print(f"{headline}  {len(finished)} finished, {len(broken)} not")
    for tile, status in broken:
        print(f"  {tile:9} {status}")
    for tile in leftover:
        print(f"  {tile:9} never ran")
    for path in manifests:
        print(f"  manifest {path}")
    return 0 if not broken and not leftover else 1


def run_adopt(a) -> int:
    """Take over one wave whose driver is gone, then report it."""
    statuses = adopt(
        a.adopt,
        poll_seconds=a.poll_seconds,
        timeout_minutes=a.timeout_minutes,
    )
    bad = sorted(t for t, s in statuses.items() if s != "finished")
    print(f"\n{len(statuses) - len(bad)} finished, {len(bad)} not")
    for tile in bad:
        print(f"  {tile:9} {statuses[tile]}")
    return 0 if not bad else 1


def run_refill(
    tiles: list[str], a, *, cfg: dict, commit: str, user_data: Path
) -> tuple[dict[str, str], list[str], list[Path], str]:
    """Drive a refill run and price it from its own snapshots.

    The headline is priced here rather than by `teardown`, which sees only the
    instances AWS still remembers. See `COST_EVERY_MINUTES`.
    """
    done, leftover, manifest_path, cost_path = refill(
        tiles,
        cfg=cfg,
        commit=commit,
        fleet_dir=a.fleet_dir,
        manifest_dir=a.manifest_dir,
        user_data=user_data,
        width=a.width,
        poll_seconds=a.poll_seconds,
        timeout_minutes=a.timeout_minutes,
        retries=a.retries,
        workers=a.launch_workers,
        cost_every_minutes=a.cost_every,
    )
    count, seconds, dollars = total_from_snapshots(cost_path)
    headline = (
        f"\n=== refill === {count} instance(s), {seconds / 3600:.1f} "
        f"instance-hours, ${dollars:.2f}"
    )
    return done, leftover, [manifest_path], headline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--remaining",
        action="store_true",
        help="with --tiles-file, print the tiles that have no upload manifest "
        "in the bucket yet, and launch nothing",
    )
    p.add_argument(
        "--adopt",
        type=Path,
        metavar="MANIFEST",
        help="take over an existing wave whose driver is gone: watch it to "
        "the upload manifest, then tear it down and price it",
    )
    p.add_argument("--tiles-file", type=Path, help="tile ids, one per line")
    p.add_argument("--tiles", nargs="+", help="tile ids, in place of --tiles-file")
    p.add_argument("--commit", help="full 40-character SHA. Required to launch")
    p.add_argument(
        "--width",
        type=int,
        default=20,
        help="how many instances run at once. Under --refill this is the pool "
        "size. Otherwise it is the wave size",
    )
    p.add_argument(
        "--refill",
        action="store_true",
        help="hold --width instances alive and start a new tile the moment "
        "one finishes, instead of running fixed waves. Removes the wave "
        "barrier, which costs about 20 machine-minutes per tile at width 20",
    )
    p.add_argument(
        "--launch-workers",
        type=int,
        default=LAUNCH_WORKERS,
        help="how many instances are placed at the same time",
    )
    p.add_argument(
        "--cost-every",
        type=float,
        default=COST_EVERY_MINUTES,
        help="minutes between cost snapshots under --refill. AWS forgets a "
        "terminated instance after about an hour",
    )
    p.add_argument(
        "--max-waves",
        type=int,
        default=0,
        help="stop after this many waves. 0 runs the whole queue",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="how many times a failed tile goes back to the queue",
    )
    p.add_argument("--fleet-dir", type=Path, default=DEFAULT_FLEET_DIR)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--manifest-dir", type=Path, default=Path.home() / ".landsat-lst-run"
    )
    p.add_argument(
        "--plan-file",
        type=Path,
        default=DEFAULT_PLAN_FILE,
        help="scene counts, for the dry-run cost projection only",
    )
    p.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    p.add_argument("--timeout-minutes", type=float, default=WAVE_TIMEOUT_MINUTES)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the wave plan and create nothing",
    )
    a = p.parse_args(argv)

    if a.adopt:
        if a.tiles_file or a.tiles:
            p.error("--adopt takes over one wave and launches nothing")
        return run_adopt(a)

    if bool(a.tiles_file) == bool(a.tiles):
        p.error("pass exactly one of --tiles-file or --tiles")
    tiles = load_tiles(a.tiles_file) if a.tiles_file else parse_tiles(a.tiles)

    if a.remaining:
        return report_remaining(tiles, load_config(a.config, a.fleet_dir))
    if a.width < 1:
        p.error("--width must be at least 1")
    if a.launch_workers < 1:
        p.error("--launch-workers must be at least 1")

    if not a.commit:
        p.error("--commit is required to launch")
    cfg = load_config(a.config, a.fleet_dir)
    commit = resolve_commit(a.commit, repo=Path.cwd())
    instance = cfg["instance"]["type"]

    waves = -(-len(tiles) // a.width)
    mode = "refill" if a.refill else f"{waves} wave(s)"
    print(f"{len(tiles)} tile(s)  width {a.width}  {mode}  on {instance}")
    print(f"commit {commit}")
    if a.dry_run:
        print(plan_text(tiles, a))
        print("\nDry run. Nothing was created.")
        return 0

    a.manifest_dir.mkdir(parents=True, exist_ok=True)
    user_data = render_user_data(cfg, a.manifest_dir, a.fleet_dir)

    if a.refill:
        done, leftover, manifests, headline = run_refill(
            tiles, a, cfg=cfg, commit=commit, user_data=user_data
        )
    else:
        done, leftover, wave_no, manifests = drain(
            tiles, a, cfg=cfg, commit=commit, user_data=user_data
        )
        headline = f"\n=== {wave_no} wave(s) ==="

    return report(headline, done, leftover, manifests)


if __name__ == "__main__":
    raise SystemExit(main())

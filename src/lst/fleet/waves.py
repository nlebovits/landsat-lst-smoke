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
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lst.fleet import watch
from lst.fleet.launch import (
    DEFAULT_FLEET_DIR,
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
        for tile in tiles:
            try:
                launched.append(
                    launch_one(cfg, tile, run_id, False, user_data, manifest)
                )
            except QuotaExhausted as err:
                # Everything from here on would be refused too, so stop asking.
                refused = tiles[len(launched) :]
                say(
                    f"\nquota reached after {len(launched)} instance(s): "
                    f"{err.code}. Running this wave at {len(launched)} and "
                    f"returning {len(refused)} tile(s) to the queue."
                )
                break

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
    width = a.width
    wave_no = 0
    manifests: list[Path] = []

    while queue:
        wave_no += 1
        if a.max_waves and wave_no > a.max_waves:
            say(f"\nstopping after {a.max_waves} wave(s), {len(queue)} tile(s) left")
            wave_no -= 1
            break
        batch, queue = queue[:width], queue[width:]
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
            say=say,
        )
        manifests.append(manifest_path)

        if 0 < placed < len(batch):
            # The account said no. That number is the real ceiling, so every
            # later wave uses it rather than asking again and paying another
            # round of refusals.
            width = placed
            say(f"width is now {width}, discovered from the quota refusal")

        settled, requeued = sort_wave(
            batch, statuses, attempts=attempts, retries=a.retries, say=say
        )
        done |= settled
        queue.extend(requeued)

    return done, queue, wave_no, manifests


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
    p.add_argument("--width", type=int, default=20, help="instances per wave")
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

    if bool(a.tiles_file) == bool(a.tiles):
        p.error("pass exactly one of --tiles-file or --tiles")
    tiles = load_tiles(a.tiles_file) if a.tiles_file else parse_tiles(a.tiles)

    if a.remaining:
        return report_remaining(tiles, load_config(a.config, a.fleet_dir))
    if a.width < 1:
        p.error("--width must be at least 1")

    if not a.commit:
        p.error("--commit is required to launch")
    cfg = load_config(a.config, a.fleet_dir)
    commit = resolve_commit(a.commit, repo=Path.cwd())
    instance = cfg["instance"]["type"]

    waves = -(-len(tiles) // a.width)
    print(f"{len(tiles)} tile(s)  width {a.width}  {waves} wave(s)  on {instance}")
    print(f"commit {commit}")
    if a.dry_run:
        for i in range(waves):
            batch = tiles[i * a.width : (i + 1) * a.width]
            print(f"  wave {i + 1}: {len(batch)}  {' '.join(batch)}")
        print("\nDry run. Nothing was created.")
        return 0

    a.manifest_dir.mkdir(parents=True, exist_ok=True)
    user_data = render_user_data(cfg, a.manifest_dir, a.fleet_dir)

    done, leftover, wave_no, manifests = drain(
        tiles, a, cfg=cfg, commit=commit, user_data=user_data
    )

    finished = sorted(t for t, s in done.items() if s == "finished")
    broken = sorted((t, s) for t, s in done.items() if s != "finished")
    print(f"\n=== {wave_no} wave(s) === {len(finished)} finished, {len(broken)} not")
    for tile, status in broken:
        print(f"  {tile:9} {status}")
    for path in manifests:
        print(f"  manifest {path}")
    return 0 if not broken and not leftover else 1


if __name__ == "__main__":
    raise SystemExit(main())

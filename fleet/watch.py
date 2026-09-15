# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["boto3"]
# ///
"""Report where each tile of a run has got to, from object storage alone.

It polls the bucket, not SSH. The uploader is already pushing `markers.txt`
every interval, so progress is visible without a session open and without the
laptop staying awake.

`get-console-output` was the previous channel. It was polled eight times and
returned nothing while three of four instances had already failed and
self-terminated, which is eight minutes of four-instance billing spent watching
an empty channel.

Four states matter, and each one has cost money here:

    running   markers advancing
    hung      no marker for longer than the stall window
    failed    a marker carrying rc other than 0
    finished  `all_done rc=0`, or `_MANIFEST.json` with complete true
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: How long a tile may go without any sign of life before it reads `hung`.
#:
#: The signal is `heartbeat.txt`, which `run.sh` appends to every 60 seconds,
#: not the phase markers. MEASURED, staging runs about 19 minutes between
#: `prep_start` and `prep_done` and writes no marker at all, so a window sized
#: against markers either calls a healthy tile hung or has to be widened until
#: it reports nothing useful. The first run of this watcher did the former.
STALL = timedelta(minutes=5)


@dataclass(frozen=True)
class State:
    tile: str
    phase: str
    status: str
    detail: str = ""


def parse_markers(text: str) -> list[tuple[str, int, datetime]]:
    """`MARKER <phase> rc=<n> <iso8601>` lines, oldest first."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] != "MARKER":
            continue
        phase, rc, stamp = parts[1], parts[2], parts[3]
        if not rc.startswith("rc="):
            continue
        try:
            out.append(
                (
                    phase,
                    int(rc[3:]),
                    datetime.fromisoformat(stamp.replace("Z", "+00:00")),
                )
            )
        except ValueError:
            continue
    return out


def last_beat(text: str | None) -> datetime | None:
    """The most recent `BEAT <iso8601>` line, or None if there is no heartbeat."""
    for line in reversed((text or "").splitlines()):
        parts = line.split()
        if len(parts) == 2 and parts[0] == "BEAT":
            try:
                return datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def classify(
    tile: str,
    markers: str | None,
    complete: bool,
    instance_alive: bool | None,
    now: datetime,
    heartbeat: str | None = None,
    stall: timedelta = STALL,
) -> State:
    """One tile's state. Pure, so the interesting cases have tests.

    `instance_alive` is None when nobody could ask. An expired SSO token took
    the whole poll down on the first run, including the object-storage half
    that was working, so not knowing is a third answer rather than an error.
    """
    if complete:
        return State(tile, "uploaded", "finished", "manifest complete")
    events = parse_markers(markers or "")
    if not events:
        if instance_alive is False:
            return State(tile, "-", "gone", "no instance and no markers")
        return State(tile, "-", "starting", "no markers yet")
    phase, rc, when = events[-1]
    if rc != 0:
        return State(tile, phase, "failed", f"rc={rc}")
    if phase == "all_done":
        return State(tile, phase, "finished", "waiting on upload")
    if instance_alive is False:
        return State(tile, phase, "gone", "instance ended before all_done")
    beat = last_beat(heartbeat)
    if beat and beat > when:
        since, source = now - beat, "beat"
    else:
        since, source = now - when, "marker"
    unknown = " (instance state unknown)" if instance_alive is None else ""
    minutes = int(since.total_seconds() // 60)
    if since > stall:
        return State(tile, phase, "hung", f"no {source} for {minutes} min{unknown}")
    return State(
        tile, phase, "running", f"{int(since.total_seconds())}s since {source}{unknown}"
    )


def running_instances(ec2, ids: list[str]) -> set[str] | None:
    """Which of `ids` are alive, or None when nobody could ask.

    An expired SSO token must not take down the markers. Those come from object
    storage on a separate, static key that does not expire, and on the first
    run of this watcher the EC2 half took the whole poll down with it.
    """
    if not ids:
        return set()
    try:
        desc = ec2.describe_instances(InstanceIds=ids)
    except Exception as err:
        print(
            f"# cannot reach EC2, reporting from markers only: {type(err).__name__}",
            flush=True,
        )
        return None
    return {
        i["InstanceId"]
        for r in desc["Reservations"]
        for i in r["Instances"]
        if i["State"]["Name"] in ("pending", "running")
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="the run manifest fleet/launch.py wrote",
    )
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    import boto3

    run = json.loads(a.manifest.read_text())
    cfg = run["config"]
    bucket = cfg["storage"]["bucket"]
    runs_prefix = cfg["storage"]["runs_prefix"]
    s3 = boto3.Session(profile_name=cfg["storage"].get("upload_profile")).client(
        "s3", region_name=cfg["aws"]["region"]
    )
    ec2 = boto3.Session(profile_name=cfg["aws"]["profile"]).client(
        "ec2", region_name=cfg["aws"]["region"]
    )

    def fetch(key: str) -> str | None:
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        except Exception:
            return None

    while True:
        alive = running_instances(
            ec2, [e["instance_id"] for e in run["instances"] if e.get("instance_id")]
        )
        now = datetime.now(timezone.utc)
        states = []
        for e in run["instances"]:
            prefix = f"{runs_prefix}/{e['name']}"
            states.append(
                classify(
                    e["tile"],
                    fetch(f"{prefix}/markers.txt"),
                    fetch(f"{prefix}/_MANIFEST.json") is not None,
                    None if alive is None else e.get("instance_id") in alive,
                    now,
                    heartbeat=fetch(f"{prefix}/heartbeat.txt"),
                )
            )
        stamp = now.strftime("%H:%M:%SZ")
        for s in states:
            print(
                f"{stamp}  {s.tile:9} {s.status:9} {s.phase:16} {s.detail}", flush=True
            )
        if a.once or all(s.status in ("finished", "failed", "gone") for s in states):
            bad = [s for s in states if s.status in ("failed", "gone", "hung")]
            if bad and not a.once:
                print(
                    f"\n{len(bad)} tile(s) need attention. Instances are still "
                    f"billing until teardown:",
                    flush=True,
                )
                print(f"  uv run fleet/teardown.py --manifest {a.manifest}", flush=True)
            return 1 if bad else 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

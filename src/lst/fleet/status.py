"""A live view of a running fleet, for somebody who is not reading the log.

The driver writes one status line a minute and thousands of other lines. This
reads the status lines back, works out the rate the run is actually going at,
and says when it will finish. The projection in `waves.plan_text` is a model
fitted before the run. This is the run itself.

    uv run lst-fleet-status            # newest run, refreshing
    uv run lst-fleet-status --once     # one block, for a pipe

**A stale log is the headline, not a footnote.** A driver that dies leaves
instances billing with nobody watching, and the log simply stops. Twice on
2026-09-16 a run died and the only sign was a log that had not moved. The age
of the last line is therefore reported on every block, and once it passes
`STALE_MINUTES` it replaces the summary rather than sitting under it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: The driver's once-a-minute status line.
STATUS = re.compile(
    r"^(?P<stamp>\d{2}:\d{2}:\d{2})Z\s+live=(?P<live>\d+)\s+queued=(?P<queued>\d+)\s+"
    r"done=(?P<done>\d+)/(?P<total>\d+)\s+peak=(?P<peak>\d+)"
    r"(?:\s+condemned=(?P<condemned>\d+))?"
)

#: How long the log may go quiet before the dashboard calls the driver dead.
#: The driver polls every 60 seconds, so three minutes is three missed polls.
STALE_MINUTES = 3.0

#: How far back the measured rate looks. Short enough to notice a stall, long
#: enough that one slow minute does not move the estimate.
RATE_WINDOW_MINUTES = 30.0

#: Where the driver writes, and the shape of the names it uses.
RUN_DIR = Path.home() / ".landsat-lst-run"
LOG_GLOB = "*.log"

BAR_WIDTH = 28


@dataclass(frozen=True)
class Snapshot:
    """One status line, parsed."""

    at: datetime
    live: int
    queued: int
    done: int
    total: int
    peak: int
    condemned: int

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


def parse_status(line: str, day: datetime) -> Snapshot | None:
    """One status line, or None when the line is something else.

    The driver prints a time of day and no date, so the date comes from the
    file's own day. A run that crosses midnight rolls forward rather than
    reporting a rate of minus one thousand tiles an hour.
    """
    found = STATUS.match(line.strip())
    if not found:
        return None
    hour, minute, second = (int(p) for p in found["stamp"].split(":"))
    at = day.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if at < day:
        at += timedelta(days=1)
    return Snapshot(
        at=at,
        live=int(found["live"]),
        queued=int(found["queued"]),
        done=int(found["done"]),
        total=int(found["total"]),
        peak=int(found["peak"]),
        condemned=int(found["condemned"] or 0),
    )


def read_log(path: Path) -> tuple[list[Snapshot], int, int]:
    """Every status line in one log, plus the failure and throttle counts.

    Returns:
        `(snapshots, failures, throttles)`.
    """
    text = path.read_text(errors="replace")
    started = started_at(text) or datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    day = started.replace(hour=0, minute=0, second=0, microsecond=0)
    shots = []
    failures = throttles = 0
    for line in text.splitlines():
        shot = parse_status(line, day)
        if shot is not None:
            shots.append(shot)
        elif "back to the queue" in line:
            failures += 1
        elif "throttling" in line:
            throttles += 1
    return shots, failures, throttles


def started_at(text: str) -> datetime | None:
    """The run id out of the manifest line, which carries the date."""
    found = re.search(r"run-(\d{8})-(\d{6})\.json", text)
    if not found:
        return None
    return datetime.strptime(found[1] + found[2], "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def rate_per_hour(shots: list[Snapshot], window_minutes: float) -> float | None:
    """Tiles an hour, measured over the last `window_minutes` of the log.

    None until two samples exist far enough apart to divide by.
    """
    if len(shots) < 2:
        return None
    last = shots[-1]
    cutoff = last.at - timedelta(minutes=window_minutes)
    window = [s for s in shots if s.at >= cutoff]
    if len(window) < 2:
        # A window narrower than the poll interval holds one sample, and one
        # sample divided by zero elapsed hours is not a rate. Two samples are
        # the fewest that measure anything.
        window = shots[-2:]
    first = window[0]
    hours = (last.at - first.at).total_seconds() / 3600
    if hours <= 0:
        return None
    return (last.done - first.done) / hours


def eta(shot: Snapshot, per_hour: float | None) -> datetime | None:
    """When the queue empties at the measured rate."""
    if not per_hour or per_hour <= 0:
        return None
    left = shot.total - shot.done
    return shot.at + timedelta(hours=left / per_hour)


def newest_log(run_dir: Path, pattern: str = LOG_GLOB) -> Path | None:
    logs = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def running_instances(profile: str, region: str) -> int | None:
    """How many instances AWS says are alive. None when it could not answer.

    Asked of AWS rather than read from the log, because the disagreement is the
    interesting case: a driver that has died still leaves its last status line
    saying 58 are live.
    """
    out = subprocess.run(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--profile",
            profile,
            "--region",
            region,
            "--filters",
            "Name=instance-state-name,Values=pending,running",
            "--query",
            "length(Reservations[].Instances[])",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(fraction * width))
    return "#" * filled + "-" * (width - filled)


def clock(when: datetime, offset_hours: float) -> str:
    return (when + timedelta(hours=offset_hours)).strftime("%H:%M")


def elapsed(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60:02d}m"


def render(
    log: Path,
    shots: list[Snapshot],
    failures: int,
    throttles: int,
    *,
    now: datetime,
    alive: int | None,
    offset_hours: float,
    cost: tuple[int, int, float] | None = None,
    stale_minutes: float = STALE_MINUTES,
) -> str:
    """The block the dashboard prints."""
    head = f"  {log.name}   {clock(now, offset_hours)} local"
    if not shots:
        return f"{head}\n  waiting for the first status line"

    last = shots[-1]
    age = (now - last.at).total_seconds()
    began = shots[0].at
    lines = [head]

    if age > stale_minutes * 60:
        lines.append("")
        lines.append(f"  !! THE LOG HAS NOT MOVED FOR {elapsed(age)}")
        lines.append("  !! The driver polls every minute. It has probably died.")
        if alive:
            lines.append(f"  !! {alive} instance(s) are still running and billing.")
            lines.append("  !! Stop them, or take the run over with --adopt.")
        elif alive == 0:
            lines.append("  !! No instances are running. Nothing is billing.")
        lines.append("")

    per_hour = rate_per_hour(shots, RATE_WINDOW_MINUTES)
    finish = eta(last, per_hour)
    pct = 100 * last.fraction
    lines.append(
        f"  tiles   {last.done:4d}/{last.total:<4d} {pct:5.1f}%  [{bar(last.fraction)}]"
    )
    if per_hour:
        done_at = clock(finish, offset_hours) if finish else "?"
        lines.append(
            f"  rate    {per_hour:5.1f} tiles/hour over "
            f"{RATE_WINDOW_MINUTES:.0f} min   finishes about {done_at} local"
        )
    else:
        lines.append("  rate    not enough samples yet")
    lines.append(
        f"  pool    live {last.live:3d}   queued {last.queued:4d}   "
        f"condemned {last.condemned:2d}   peak {last.peak:3d}"
    )
    counted = "?" if alive is None else str(alive)
    lines.append(
        f"  ec2     {counted} running per AWS   "
        f"requeued {failures}   throttles {throttles}"
    )
    if cost:
        count, seconds, dollars = cost
        lines.append(
            f"  cost    ${dollars:8.2f} measured over {count} instance(s), "
            f"{seconds / 3600:.1f} instance-hours"
        )
    lines.append(
        f"  up      {elapsed((now - began).total_seconds())}   "
        f"last line {elapsed(age)} ago"
    )
    return "\n".join(lines)


def snapshots_beside(log: Path) -> Path | None:
    """The cost snapshot file the driver writes next to its manifest."""
    found = re.search(r"run-\d{8}-\d{6}", log.read_text(errors="replace")[:4096])
    if not found:
        return None
    path = log.parent / f"{found[0]}.snapshots.txt"
    return path if path.exists() else None


def block(log: Path, *, profile: str, region: str, offset_hours: float) -> str:
    from lst.fleet.waves import total_from_snapshots

    shots, failures, throttles = read_log(log)
    snaps = snapshots_beside(log)
    cost = None
    if snaps:
        try:
            cost = total_from_snapshots(snaps)
        except OSError:
            cost = None
    return render(
        log,
        shots,
        failures,
        throttles,
        now=datetime.now(timezone.utc),
        alive=running_instances(profile, region),
        offset_hours=offset_hours,
        cost=cost,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", type=Path, help="default: the newest log in the run dir")
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--once", action="store_true", help="print one block and stop")
    p.add_argument("--profile", default="radiant-earth")
    p.add_argument("--region", default="us-west-2")
    p.add_argument(
        "--utc-offset",
        type=float,
        default=2.0,
        help="hours to add to UTC when printing a local time",
    )
    a = p.parse_args(argv)

    log = a.log or newest_log(a.run_dir)
    if log is None:
        print(f"no log in {a.run_dir}", file=sys.stderr)
        return 1

    while True:
        text = block(log, profile=a.profile, region=a.region, offset_hours=a.utc_offset)
        if a.once:
            print(text)
            return 0
        # Clear and home, so the block stays in one place instead of scrolling.
        print("\033[2J\033[H" + text, flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())

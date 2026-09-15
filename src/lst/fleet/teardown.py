"""Terminate a run, then price it. In that order, because the order matters.

`lst.fleet.cost_report` reads `StateTransitionReason` for the end of an instance's
life and excludes anything still running from its totals. Pricing first
therefore reports only the instances that had already stopped: the first run of
this script priced one of five and put $2.98 against a run that cost about
$12.60.

Terminating first fixes that, and the window is wide enough to allow it. AWS
answers queries about a terminated instance for roughly an hour.

Terminating is often refused for an agent by the permission layer. When that
happens this prints the exact command for a human rather than leaving an
instance billing, which is how one box came to idle for over an hour after
finishing, wasting more than the tile's own compute cost.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lst.fleet.launch import delete_key_pair


def sweep_keys(cfg: dict, run: dict, *, remove_files: bool = True) -> tuple[int, int]:
    """Delete the key pair of every instance in the manifest.

    `launch.py` creates one key pair per instance, named for the tile and the
    run. Nothing deleted them until now, so a 100-tile run left 100 key pairs
    in EC2 and 100 private keys in `~/.ssh`. The account limit is 5,000, which
    a few continent-scale runs reach.

    This runs only after termination succeeds. The key is the single way into a
    running instance, because `ec2-instance-connect`, `ssm`, and the serial
    console are all denied for this role. Deleting it while the instance lives
    would strand a box that is still billing.

    Returns:
        `(deleted, attempted)`.
    """
    deleted = 0
    entries = [e for e in run["instances"] if e.get("name")]
    for entry in entries:
        if delete_key_pair(cfg, entry["name"]):
            deleted += 1
        if remove_files and entry.get("pem"):
            Path(entry["pem"]).expanduser().unlink(missing_ok=True)
    return deleted, len(entries)


def terminate_argv(cfg: dict, ids: list[str]) -> list[str]:
    a = cfg["aws"]
    return [
        "aws",
        "ec2",
        "terminate-instances",
        "--profile",
        a["profile"],
        "--region",
        a["region"],
        "--instance-ids",
        *ids,
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--skip-cost", action="store_true")
    p.add_argument(
        "--keep-keys",
        action="store_true",
        help="leave the key pairs in place. Without it, terminating also "
        "deletes the key pair and the private key of every instance in the "
        "manifest, which is safe only because they are already terminated",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    run = json.loads(a.manifest.read_text())
    cfg = run["config"]
    ids = [e["instance_id"] for e in run["instances"] if e.get("instance_id")]
    if not ids:
        print("no instances in the manifest")
        return 0

    argv = terminate_argv(cfg, ids)
    if a.dry_run:
        print(" ".join(argv))
        if not a.skip_cost:
            print("\nthen the cost report, once every instance has stopped")
        return 0

    terminated = False
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
        print(f"terminated {len(ids)}: {' '.join(ids)}")
        terminated = True
    except (subprocess.CalledProcessError, PermissionError) as err:
        print(
            f"\nCould not terminate: {err}\n"
            f"These instances are still billing. Run this yourself:\n\n"
            f"  {' '.join(argv)}\n",
            file=sys.stderr,
        )

    if terminated:
        # One wait serves both of the steps below. `StateTransitionReason`
        # carries the timestamp the report prices against and is not set the
        # instant the call returns, and a key may only be deleted once its
        # instance is gone.
        print("waiting for the instances to stop")
        subprocess.run(
            [
                "aws",
                "ec2",
                "wait",
                "instance-terminated",
                "--profile",
                cfg["aws"]["profile"],
                "--region",
                cfg["aws"]["region"],
                "--instance-ids",
                *ids,
            ],
            capture_output=True,
            text=True,
        )
        if not a.keep_keys:
            # After termination, never before. The key is the only way into a
            # live instance for this role.
            deleted, attempted = sweep_keys(cfg, run)
            print(f"deleted {deleted} of {attempted} key pair(s)")

    if a.skip_cost:
        return 0 if terminated else 1

    report = a.manifest.with_suffix(".cost.txt")
    # Still a subprocess, and still `-m`: the cost report has to survive this
    # process failing, and its output is captured verbatim into a file beside
    # the manifest. `-m` rather than a file path, because the module no longer
    # sits at a path this one can name relatively.
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
    report.write_text(out.stdout + out.stderr)
    print(f"cost report saved to {report}")
    for line in out.stdout.splitlines():
        if "TOTAL" in line or "subtotal" in line or "known lines" in line:
            print(f"  {line.strip()}")
    return 0 if terminated else 1


if __name__ == "__main__":
    sys.exit(main())

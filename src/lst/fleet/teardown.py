"""Terminate a run, then price it. In that order, because the order matters.

`cost_report.py` reads `StateTransitionReason` for the end of an instance's
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

    if a.skip_cost:
        return 0 if terminated else 1

    if terminated:
        # `StateTransitionReason` carries the timestamp the report prices
        # against, and it is not set the instant the call returns.
        print("waiting for the instances to stop, so the report can price them")
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

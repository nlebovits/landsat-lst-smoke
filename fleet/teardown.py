# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = []
# ///
"""Price a run, then terminate it. In that order, because the order matters.

A terminated instance stays queryable for about an hour, after which
`LaunchTime` and `StateTransitionReason` are gone and its lifetime is
unrecoverable. So the cost report runs first and its output is saved.

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

HERE = Path(__file__).resolve().parent


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

    if not a.skip_cost:
        report = a.manifest.with_suffix(".cost.txt")
        argv = [
            sys.executable,
            str(HERE.parent / "cost_report.py"),
            "--tag",
            f"purpose={cfg['tags']['purpose']}",
            "--region",
            cfg["aws"]["region"],
            "--profile",
            cfg["aws"]["profile"],
        ]
        print(
            f"cost report first, the query window is about an hour after "
            f"termination\n  {' '.join(argv)}"
        )
        if not a.dry_run:
            out = subprocess.run(argv, capture_output=True, text=True)
            report.write_text(out.stdout + out.stderr)
            print(f"  saved {report}")

    argv = terminate_argv(cfg, ids)
    if a.dry_run:
        print("\n" + " ".join(argv))
        return 0
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
        print(f"terminated {len(ids)}: {' '.join(ids)}")
        return 0
    except (subprocess.CalledProcessError, PermissionError) as err:
        print(
            f"\nCould not terminate: {err}\n"
            f"These instances are still billing. Run this yourself:\n\n"
            f"  {' '.join(argv)}\n",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

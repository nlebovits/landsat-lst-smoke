#!/usr/bin/env python3
"""Sweep the read-concurrency knobs and report throughput for each.

The composite is I/O bound: 97% of worker execution time is reading COGs and
the workers hold the GIL 0.02% of the time. So the knobs that matter are the
ones that change how many byte ranges are in flight at once, not the ones that
change how fast the arithmetic runs.

Each configuration runs in a fresh subprocess, because peak RSS is a per-process
high-water mark and a second config in the first one's interpreter would
inherit its peak.

    uv run sweep_throughput.py --scenes 24
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = HERE / "profile_lst_p95.py"

# (label, workers, threads_per_worker, chunk, gdal_threads, extra)
CONFIGS = [
    ("base   4w x 2t  c256", 4, 2, 256, "1", ""),
    ("thread 4w x 8t  c256", 4, 8, 256, "1", ""),
    ("thread 4w x 16t c256", 4, 16, 256, "1", ""),
    ("procs  8w x 8t  c256", 8, 8, 256, "1", ""),
    ("chunk  4w x 8t  c512", 4, 8, 512, "1", ""),
    ("chunk  4w x 8t  c1024", 4, 8, 1024, "1", ""),
    ("gdalth 4w x 8t  c512", 4, 8, 512, "ALL_CPUS", ""),
    ("http11 4w x 8t  c512", 4, 8, 512, "1", "GDAL_HTTP_VERSION=1.1"),
]


def run_one(
    label, workers, threads, chunk, gdal_threads, extra, scenes, out_root, source
):
    out_dir = (
        out_root / label.split()[0] / f"{workers}w{threads}t_c{chunk}_{gdal_threads}"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        "uv",
        "run",
        "--python",
        "3.12",
        str(SCRIPT),
        # Named rather than inherited. `profile_lst_p95.py` defaults to
        # planetary-computer, so a sweep that leaves this out measures a
        # different bucket, a different signing path, and a different request
        # count than the sharded pipeline it exists to inform.
        "--source",
        source,
        "--max-scenes",
        str(scenes),
        "--workers",
        str(workers),
        "--threads-per-worker",
        str(threads),
        "--chunk",
        str(chunk),
        "--memory-limit-gib",
        "3",
        "--gdal-threads",
        gdal_threads,
        "--no-tracemalloc",
        "--span-limit",
        "400000",
        "--span-dump-limit",
        "1000",
        "--out-dir",
        str(out_dir),
        "--force",
    ]
    if extra:
        cmd += ["--gdal-extra", extra]

    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    stages = out_dir / "stages.json"
    if not stages.exists():
        return {"label": label, "error": proc.stderr.strip()[-400:] or "no stages.json"}

    run = json.loads(stages.read_text())
    thr = run.get("throughput", {})
    comp = next((s for s in run["stages"] if s["stage"] == "compute"), {})
    mem = run.get("memory", {})
    return {
        "label": label,
        "workers": workers,
        "threads": threads,
        "slots": workers * threads,
        "chunk": chunk,
        "gdal_threads": gdal_threads,
        "extra": extra,
        "wall_s": comp.get("wall_s"),
        "mb_s": thr.get("mb_per_s"),
        "mb_s_peak": thr.get("mb_per_s_peak"),
        "net_mb": thr.get("net_recv_mb"),
        "s_per_scene": thr.get("s_per_scene"),
        "parallelism": thr.get("effective_parallelism"),
        "utilisation": thr.get("utilisation_pct"),
        "peak_gib": (mem.get("tree_rss_peak_mb") or 0) / 1024,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--out-root", type=Path, default=HERE / "sweep")
    ap.add_argument("--only", default="", help="substring filter on the label")
    ap.add_argument(
        "--source",
        default="earth-search",
        help="passed through to profile_lst_p95.py, which defaults to "
        "planetary-computer. earth-search is the bucket the fleet reads",
    )
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    configs = [c for c in CONFIGS if args.only in c[0]]
    results = []
    for i, (label, w, t, c, gt, extra) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {label} ...", flush=True)
        res = run_one(
            label, w, t, c, gt, extra, args.scenes, args.out_root, args.source
        )
        results.append(res)
        if "error" in res:
            print(f"    FAILED: {res['error'][:200]}", flush=True)
        else:
            print(
                f"    {res['wall_s']:.1f}s  {res['mb_s']:.1f} MB/s  "
                f"par {res['parallelism']:.1f}/{res['slots']}  "
                f"peak {res['peak_gib']:.1f} GiB",
                flush=True,
            )

    (args.out_root / "sweep.json").write_text(json.dumps(results, indent=2))

    ok = [r for r in results if "error" not in r]
    if not ok:
        print("\nall configurations failed")
        return 1
    best = max(ok, key=lambda r: r["mb_s"] or 0)
    print(
        f"\n{'config':24s}{'slots':>6}{'wall s':>9}{'MB/s':>8}{'peak MB/s':>11}"
        f"{'par':>7}{'use%':>7}{'peak GiB':>10}{'s/scene':>9}"
    )
    for r in ok:
        print(
            f"{r['label']:24s}{r['slots']:6d}{r['wall_s']:9.1f}{r['mb_s']:8.1f}"
            f"{r['mb_s_peak']:11.1f}{r['parallelism']:7.1f}{r['utilisation']:7.0f}"
            f"{r['peak_gib']:10.2f}{r['s_per_scene']:9.2f}"
        )
    print(f"\nfastest: {best['label']} at {best['mb_s']:.1f} MB/s")
    print(f"artifacts: {args.out_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

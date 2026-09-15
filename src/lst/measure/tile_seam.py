"""How far two adjacent tiles disagree about the same scene.

Each tile fits its own offsets over its own region: `tile_prep.prep_bbox` runs
the tile out by `margin_deg` and the monthly reference is built inside that.
So a scene crossing the line between two tiles is corrected by one amount on
one side and a different amount on the other, and `composite.reduce_block`
subtracts whichever its own tile fitted.

That difference is the seam. It needs no composite to measure, because both
numbers are already in `tile-prep.npz`, keyed by scene id. A step at a tile
boundary is the worst kind of artifact to publish: straight, axis-aligned, and
sitting on a round-number coordinate.

    uv run lst-measure-tile-seam --prep a/tile-prep.npz b/tile-prep.npz

Only scenes both tiles kept are compared. A scene one tile rejected at the 15 C
cap never reaches that tile's composite, so it cannot contribute a step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FLOOR = 200
CAP = 15.0


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    offset, n_valid = z["offset"], z["n_valid"]
    keep = np.isfinite(offset) & (n_valid >= FLOOR) & (np.abs(offset) <= CAP)
    return {
        "ids": np.asarray(z["scene_ids"]),
        "offset": offset,
        "keep": keep,
        "paths": [str(p) for p in z["paths"]],
    }


def compare(a: dict, b: dict) -> dict:
    """The offsets each tile fitted for the scenes both of them kept."""
    ia = {s: i for i, s in enumerate(a["ids"]) if a["keep"][i]}
    ib = {s: i for i, s in enumerate(b["ids"]) if b["keep"][i]}
    shared = sorted(set(ia) & set(ib))
    if not shared:
        return {"shared": 0}
    oa = np.array([a["offset"][ia[s]] for s in shared])
    ob = np.array([b["offset"][ib[s]] for s in shared])
    d = np.abs(oa - ob)
    return {
        "in_a": int(a["keep"].sum()),
        "in_b": int(b["keep"].sum()),
        "shared": len(shared),
        "mean_abs": float(d.mean()),
        "median_abs": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "bias": float((oa - ob).mean()),
        "worst": [
            {"scene": shared[i], "a": float(oa[i]), "b": float(ob[i])}
            for i in np.argsort(-d)[:5]
        ],
    }


def verdict(r: dict) -> str:
    """What the number means for a mosaic, and what it would take to fix.

    The thresholds are judgements, not measurements, and they are here so the
    reading does not drift between sessions.
    """
    if not r.get("shared"):
        return "The tiles share no kept scene. They cannot seam against each other."
    m = r["median_abs"]
    if m < 0.1:
        return (
            f"Median disagreement {m:.3f} C. The margin already makes the "
            f"offset near enough tile-independent. Document it, change nothing."
        )
    if m < 1.0:
        return (
            f"Median disagreement {m:.3f} C. Visible on a graticule line, "
            f"small in absolute terms. Widen `margin_deg` and measure again "
            f"before anything larger."
        )
    return (
        f"Median disagreement {m:.3f} C. The offset is a per-tile quantity "
        f"and a mosaic will show it. Fit each scene once against its own "
        f"footprint into a shared artifact, and let every tile read it."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prep", nargs=2, type=Path, required=True, metavar=("A", "B"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()

    left, right = (load(path) for path in a.prep)
    r = compare(left, right)
    print(f"{a.prep[0]}  paths {','.join(left['paths'])}")
    print(f"{a.prep[1]}  paths {','.join(right['paths'])}\n")
    if not r["shared"]:
        print(verdict(r))
        return 0
    print(f"  scenes kept        {r['in_a']:,} and {r['in_b']:,}")
    print(f"  shared and kept    {r['shared']:,}")
    print("  |offset_a - offset_b|")
    for k in ("median_abs", "mean_abs", "p90", "p99", "max"):
        print(f"    {k:11} {r[k]:8.3f} C")
    print(f"  signed mean        {r['bias']:+8.3f} C  (a systematic step, not scatter)")
    print("\n  worst five scenes")
    for w in r["worst"]:
        print(f"    {w['scene']}  a {w['a']:+7.2f}  b {w['b']:+7.2f}")
    print(f"\n{verdict(r)}")
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

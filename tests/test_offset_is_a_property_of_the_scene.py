"""One scene, one offset, whichever tile is asking.

`destripe.subtract_offsets` shifts every observation of a scene by the offset
its own tile fitted. `tile_prep.prep_bbox` runs the tile out by `margin_deg`
and fits inside that, so two tiles sharing a scene fit it over different
ground and subtract different amounts. The same observation then means two
things depending on which side of a tile line it lands on.

MEASURED on the S30W065 and S30W070 prep artifacts, 1,289 scenes both tiles
kept: median disagreement 0.310 C, p99 4.21 C, worst 10.14 C. The signed mean
is -0.034 C, so it is scatter rather than a step.

The cause is not thin evidence. Raising `DESTRIPE_MIN_PREP_SAMPLES` from 200 to
12,800 left the median at 0.310 and the worst at 10.14. It is that the two
tiles evaluate the scene over different ground:

    how unevenly the two tiles see it   median disagreement
    both see nearly the same ground            0.150 C
    one sees a sliver                          3.020 C

correlation +0.527. The offset is a property of the ground it was fitted over,
not of the scene.

This test states the invariant a mosaic needs. **It fails today.** It is here so
that the day someone fits each scene once against its own footprint, the test
says so, and so that nobody has to rediscover the defect from a picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lst import destripe

ROOT = Path(__file__).resolve().parents[1]


PAIR = Path.home() / "Documents/dev/radiant-earth/lst-tiles/seam"
TOLERANCE_C = 0.01  # the output encoding step, `lst_qa.LST_SCALE`


def offsets(path: Path) -> dict[str, tuple[float, int]]:
    z = np.load(path, allow_pickle=True)
    return {
        str(s): (float(z["offset"][i]), int(z["n_valid"][i]))
        for i, s in enumerate(z["scene_ids"])
    }


def kept(value: float, count: int) -> bool:
    return bool(
        destripe.keep_mask([value], [count], floor=destripe.DESTRIPE_MIN_PREP_SAMPLES)[
            0
        ]
    )


@pytest.fixture(scope="module")
def shared():
    a, b = PAIR / "S30W065.npz", PAIR / "S30W070.npz"
    if not (a.is_file() and b.is_file()):
        pytest.skip(
            "needs the prep artifacts of two adjacent tiles. Rebuild with "
            "`uv run lst-fleet-launch --tiles S30W070 ...` and PREP_ONLY=1, "
            "then fetch both `prep/tile-prep.npz`."
        )
    left, right = offsets(a), offsets(b)
    return {
        s: (left[s][0], right[s][0])
        for s in set(left) & set(right)
        if kept(*left[s]) and kept(*right[s])
    }


@pytest.mark.xfail(
    reason="the offset is fitted per tile, so two tiles disagree about a "
    "shared scene by a median of 0.310 C. Fitting each scene once against its "
    "own footprint would satisfy this.",
    strict=True,
)
def test_adjacent_tiles_agree_about_a_shared_scene(shared):
    """The invariant. One scene, one correction, whoever is asking."""
    worst = max(shared, key=lambda s: abs(shared[s][0] - shared[s][1]))
    a, b = shared[worst]
    assert abs(a - b) <= TOLERANCE_C, (
        f"{worst} is corrected by {a:+.2f} C in one tile and {b:+.2f} C in the "
        f"other, a difference of {abs(a - b):.2f} C"
    )


def test_the_disagreement_is_scatter_and_not_a_step(shared):
    """What keeps this off the blocker list.

    A systematic step would draw a line on the graticule, which is the worst
    artifact to publish. A signed mean near zero means the tiles do not agree
    that one side is warmer, so the boundary blurs rather than steps.
    """
    signed = np.array([a - b for a, b in shared.values()])
    assert abs(signed.mean()) < 0.1, (
        f"signed mean {signed.mean():+.3f} C is a systematic step, not scatter"
    )


def test_the_sample_floor_does_not_fix_it(shared):
    """Recorded because it was the first thing tried, and it did nothing.

    The worst-disagreeing scene carries 68,392 valid pixels on its smaller
    side. It is not short of evidence, so no floor reaches it.
    """
    a, b = PAIR / "S30W065.npz", PAIR / "S30W070.npz"
    left, right = offsets(a), offsets(b)
    high = {
        s: (left[s][0], right[s][0])
        for s in set(left) & set(right)
        if left[s][1] >= 12800
        and right[s][1] >= 12800
        and abs(left[s][0]) <= 15
        and abs(right[s][0]) <= 15
    }
    d = np.array([abs(x - y) for x, y in high.values()])
    assert d.max() > 5.0, (
        "a 64x sample floor was expected to leave the worst disagreement "
        "intact. If this now passes, the cause has changed and the analysis "
        "above needs redoing."
    )

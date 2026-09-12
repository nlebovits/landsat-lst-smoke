"""Staging must not change the composite, only where the bytes come from.

The graph reads whatever the item href points at. Staging rewrites that href
to a local path, so the reader, the mask, the reduction and the encoder are
all untouched and the output should be identical rather than close.

That is the claim, and it needs the bucket to test. `tests/test_staging.py`
covers the fetch offline against a fake client, and
`tests/test_composite_graph.py` covers the reduction against a synthetic
stack. Neither reads a real COG through both paths, which is the one thing
that would catch a staged file truncated, mis-decoded, or attached to the
wrong scene.

Marked `s3` because it is requester-pays. Twelve scenes over a small window
costs about $0.09 in egress from outside the region, and a few cents inside.

    AWS_PROFILE=... uv run pytest -m s3 tests/test_staged_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import dask
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import composite  # noqa: E402
import shard_lst_p95 as S  # noqa: E402
import staging  # noqa: E402

pytestmark = [pytest.mark.s3, pytest.mark.timeout(1800)]

TILE = "S30W065"
#: Enough scenes that a mix-up between them would show, and few enough to stay
#: inside a few cents of egress.
N_SCENES = 12
#: A 0.4 degree window at four 360 px blocks a side, so the block edges and
#: the concurrent fetch both have something to confuse.
WINDOW = (-62.5, -32.9, -62.1, -32.5)
CHUNK = 360


def intersecting(boxes, window):
    w, s, e, n = window
    return [
        i
        for i, (iw, isouth, ie, inorth) in enumerate(boxes)
        if iw < e and ie > w and isouth < n and inorth > s
    ]


def composited(items):
    out = composite.build_graph(
        items, WINDOW, crs="EPSG:4326", resolution=1 / 3600, chunk=CHUNK
    )
    with dask.config.set(scheduler="sync"):
        return out.compute()


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    """The same window computed twice: from S3, then from staged files."""
    inventory = ROOT / "artifacts" / "tile_scene_inventory.parquet"
    if not inventory.exists():
        pytest.skip("run usgs_inventory.py to build the full artifact")

    S.configure_read_env("earth-search")
    bbox = S.tile_bounds(TILE)
    items, boxes = S.items_for_tile(inventory, TILE, bounds=bbox)
    items, boxes, _ = staging.drop_scenes_without_thermal(items, boxes)
    needed = intersecting(boxes, WINDOW)[:N_SCENES]
    assert len(needed) >= N_SCENES, f"only {len(needed)} scenes reach {WINDOW}"
    chosen = [items[i] for i in needed]

    unstaged = composited(chosen)

    stage_dir = tmp_path_factory.mktemp("stage")
    report = staging.stage_scenes(items, needed, stage_dir)
    staged = composited(chosen)
    yield unstaged, staged, report, needed
    staging.cleanup(stage_dir)


class TestTheCompositeIsUnchanged:
    def test_the_raster_encodes_identically(self, paired):
        unstaged, staged, _, _ = paired
        assert np.array_equal(unstaged["lst_p95"].values, staged["lst_p95"].values), (
            "the window differs between the two read paths"
        )

    def test_every_monthly_count_is_identical(self, paired):
        unstaged, staged, _, _ = paired
        assert np.array_equal(unstaged["qa_count"].values, staged["qa_count"].values)

    def test_the_comparison_covers_real_temperatures(self, paired):
        """A pair of all-nodata rasters would match and prove nothing."""
        unstaged, _, _, _ = paired
        valid = int((unstaged["lst_p95"].values != S.LST_NODATA_DN).sum())
        assert valid > 100_000, f"only {valid} valid pixels compared"


class TestTheFetchIsExact:
    def test_one_get_per_object(self, paired):
        """On the wire, against the bucket, not against a fake client."""
        _, _, report, needed = paired
        assert report["objects"] == len(needed) * 2
        assert report["get_requests"] == report["objects"]
        assert report["retries"] == 0

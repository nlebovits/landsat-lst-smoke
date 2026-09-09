"""Staging must not change the composite, only where the bytes come from.

`process_shard` reads whatever the item href points at. Staging rewrites that
href to a local path, so the reader, the mask, the reduction and the encoder
are all untouched and the output should be identical rather than close.

That is the claim, and it needs the bucket to test. `tests/test_staging.py`
covers the fetch offline against a fake client, and `tests/test_pipeline_paths.py`
covers the reduction against a synthetic stack. Neither reads a real COG
through both paths, which is the one thing that would catch a staged file
truncated, mis-decoded, or attached to the wrong scene.

Marked `s3` because it is requester-pays. Twelve scenes over four shards costs
about $0.09 in egress from outside the region, and a few cents inside it.

    AWS_PROFILE=... uv run pytest -m s3 tests/test_staged_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import shard_lst_p95 as S  # noqa: E402
import staging  # noqa: E402

pytestmark = [pytest.mark.s3, pytest.mark.timeout(1800)]

TILE = "S30W065"
#: Enough scenes that a mix-up between them would show, and few enough to stay
#: inside a few cents of egress.
N_SCENES = 12
#: More than one shard, so the concurrent fetch has something to confuse.
N_SHARDS = 4
SHARD_PX = 360


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    """The same shards computed twice: from S3, then from staged files."""
    inventory = ROOT / "artifacts" / "tile_scene_inventory.parquet"
    if not inventory.exists():
        pytest.skip("run usgs_inventory.py to build the full artifact")

    S.configure_read_env("earth-search")
    bbox = S.tile_bounds(TILE)
    items, boxes = S.items_for_tile(inventory, TILE, bounds=bbox)
    items, boxes, _ = staging.drop_scenes_without_thermal(items, boxes)
    shards, _, _ = S.plan_shards(bbox, 3600, SHARD_PX)

    work = [
        (sh, idx[:N_SCENES])
        for sh, idx in ((sh, S.items_for_shard(sh, boxes)) for sh in shards)
        if len(idx) >= N_SCENES
    ][:N_SHARDS]
    assert work, f"no shard of {TILE} holds {N_SCENES} scenes"

    unstaged = {
        (sh.row, sh.col): S.process_shard(
            sh, [items[i] for i in idx], "EPSG:4326", 1 / 3600, 4
        )
        for sh, idx in work
    }

    stage_dir = tmp_path_factory.mktemp("stage")
    needed = sorted({i for _, idx in work for i in idx})
    report = staging.stage_scenes(items, needed, stage_dir)
    staged = {
        (sh.row, sh.col): S.process_shard(
            sh, [items[i] for i in idx], "EPSG:4326", 1 / 3600, 4
        )
        for sh, idx in work
    }
    yield unstaged, staged, report, needed
    staging.cleanup(stage_dir)


class TestTheCompositeIsUnchanged:
    def test_every_shard_encodes_identically(self, paired):
        unstaged, staged, _, _ = paired
        for key in unstaged:
            assert np.array_equal(unstaged[key]["lst_p95"], staged[key]["lst_p95"]), (
                f"shard {key} differs between the two read paths"
            )

    def test_every_monthly_count_is_identical(self, paired):
        unstaged, staged, _, _ = paired
        for key in unstaged:
            assert np.array_equal(unstaged[key]["qa_count"], staged[key]["qa_count"])

    def test_the_comparison_covers_real_temperatures(self, paired):
        """A pair of all-nodata rasters would match and prove nothing."""
        unstaged, _, _, _ = paired
        valid = sum(
            int((out["lst_p95"] != S.LST_NODATA_DN).sum()) for out in unstaged.values()
        )
        assert valid > 100_000, f"only {valid} valid pixels compared"


class TestTheFetchIsExact:
    def test_one_get_per_object(self, paired):
        """On the wire, against the bucket, not against a fake client."""
        _, _, report, needed = paired
        assert report["objects"] == len(needed) * 2
        assert report["get_requests"] == report["objects"]
        assert report["retries"] == 0

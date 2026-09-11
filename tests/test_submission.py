"""What a shard carries to the worker that runs it, and how results come back.

A shard used to be submitted with its own list of STAC item dicts. At a p50 of
509 scenes that is 423,128 B per task, so a full tile pushed 0.55 GB at the
scheduler to describe 3,910 items that pickle to 3.2 MB between them. The four
committed full-tile slices spent 373.7 s submitting.

It now carries a path and a list of positions, and every worker parses the table
once. MEASURED by `measure_submit_cost.py`: 5.3 to 6.1 ms per submit against
0.024 to 0.031 ms, about 200x. MEASURED on an `m6id.16xlarge` over 250 real
shards: 0.14 ms each.

Staging still finishes before the cluster starts. Overlapping it was tried and
measured as a loss; `staging.stage_scenes_for` carries the numbers.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import shard_lst_p95  # noqa: E402
import staging  # noqa: E402
from conftest import make_row  # noqa: E402
from test_staging import FakeS3  # noqa: E402
from tile_inventory import build_item  # noqa: E402

TILE = "S30W065"


def items(n, *, thermal=True):
    return [build_item(make_row(TILE, i, thermal=thermal)) for i in range(n)]


def work_for(item_dicts, groups):
    """`work_idx`, where `groups[n]` is the item indices shard n reads."""
    return [
        (shard_lst_p95.Shard(n, 0, n * 8, 0, 8, 8, (0.0, 0.0, 1.0, 1.0)), list(idx))
        for n, idx in enumerate(groups)
    ]


def echo(shard, table_path, indices, crs, resolution, read_threads=4):
    """A shard that returns its own position instead of reading anything."""
    return {
        "row": shard.row,
        "col": shard.col,
        "y0": shard.y0,
        "x0": shard.x0,
        "n_scenes": len(indices),
        "load_s": 0.0,
        "reduce_s": 0.0,
    }


def echo_correction(shard, table_path, indices, crs, resolution, read_threads, corr):
    """`echo` with the seam correction, returned so a test can see which one."""
    return echo(shard, table_path, indices, crs, resolution, read_threads) | {
        "correction": corr,
    }


@pytest.fixture(scope="module")
def client():
    """One real cluster. Two workers is enough to schedule."""
    import frisky

    cluster = frisky.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    yield cluster.get_client()
    cluster.close()


@pytest.mark.timeout(180)
class TestDriveShards:
    def test_every_shard_is_submitted_and_gathered(self, client):
        item_dicts = items(4)
        work_idx = work_for(item_dicts, [[0], [1], [2], [3]])
        marks: dict = {}

        stats, submit_s = shard_lst_p95.drive_shards(
            client,
            echo,
            work_idx,
            None,
            "EPSG:4326",
            1 / 3600,
            1,
            assemble=lambda f: f.result(),
            marks=marks,
            t0=time.perf_counter(),
        )

        assert sorted(s["row"] for s in stats) == [0, 1, 2, 3]
        assert submit_s >= 0.0
        assert marks["first_submit_s"] <= marks["last_submit_s"]
        assert marks["first_result_s"] <= marks["last_result_s"]

    def test_a_run_without_a_prep_file_submits_no_correction_argument(self, client):
        """The argument list a pooled run sends is the one it always sent.

        `correction_of` is None without `--tile-prep`, and a task that takes
        six arguments still runs. `echo` is that task, and it would raise a
        TypeError on a seventh.
        """
        work_idx = work_for(items(2), [[0], [1]])

        stats, _ = shard_lst_p95.drive_shards(
            client,
            echo,
            work_idx,
            None,
            "EPSG:4326",
            1 / 3600,
            1,
            assemble=lambda f: f.result(),
            marks={},
            t0=time.perf_counter(),
            correction_of=None,
        )

        assert sorted(s["row"] for s in stats) == [0, 1]

    def test_each_shard_gets_its_own_correction(self, client):
        """The weights are cut to one shard's window, so the wrong one is wrong.

        A shard carries a path and positions, and the scene table resolves the
        items inside the worker. The correction cannot travel that way: it is
        resampled onto the shard's own grid in the driver. This pins that each
        shard receives the one built for it.
        """
        work_idx = work_for(items(6), [[0, 1], [2, 3], [4, 5]])

        stats, _ = shard_lst_p95.drive_shards(
            client,
            echo_correction,
            work_idx,
            None,
            "EPSG:4326",
            1 / 3600,
            1,
            assemble=lambda f: f.result(),
            marks={},
            t0=time.perf_counter(),
            correction_of=lambda shard, idx: {"row": shard.row, "idx": list(idx)},
        )

        assert {s["row"]: s["correction"] for s in stats} == {
            0: {"row": 0, "idx": [0, 1]},
            1: {"row": 1, "idx": [2, 3]},
            2: {"row": 2, "idx": [4, 5]},
        }


@pytest.mark.timeout(300)
class TestARehearsedRun:
    """`main` end to end, with synthetic shards and no object store."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory, slice_artifact):
        out = tmp_path_factory.mktemp("rehearsal")
        code = shard_lst_p95.main(
            [
                "--tile", TILE,
                "--inventory-uri", str(slice_artifact),
                "--rehearse", "60",
                "--max-shards", "40",
                "--workers", "2",
                "--threads-per-worker", "1",
                "--read-threads", "1",
                "--no-output-mask",
                "--out-dir", str(out),
            ]
        )  # fmt: skip
        return code, json.loads((out / "summary.json").read_text()), out

    def test_it_exits_zero_with_every_shard(self, run):
        code, summary, _ = run
        assert code == 0
        assert summary["n_shards"] == 40
        assert summary["n_shards_errored"] == 0
        assert len(summary["shard_stats"]) == 40

    def test_the_summary_carries_the_phase_marks(self, run):
        # The submit time used to be printed and then discarded, so a
        # before-and-after table had to be scraped out of stdout.
        _, summary, _ = run
        phases = summary["phases"]

        for key in (
            "cluster_start_s",
            "stage_s",
            "first_submit_s",
            "last_submit_s",
            "first_result_s",
            "last_result_s",
            "submit_total_s",
            "compute_s",
            "compute_from_last_submit_s",
            "wall_s",
        ):
            assert key in phases, key
        assert phases["first_submit_s"] <= phases["last_submit_s"]
        assert phases["first_result_s"] <= phases["last_result_s"]
        assert phases["last_result_s"] <= phases["wall_s"]

    def test_the_rehearsal_stages_nothing(self, run):
        _, summary, _ = run
        assert summary["staging"] is None
        assert summary["item_table"] is None


@pytest.mark.timeout(300)
class TestAStagedRun:
    """`main`'s staged path against a bucket on disk.

    The rehearsal stages nothing, so it never reaches the branch a production
    run takes. This does, with `staging._default_client` swapped for a fake.

    The scenes are georeferenced away from the tile, so the composite is
    nodata. What this checks is the machinery around the read. The pixels a
    real read produces are checked against the unstaged path in
    `tests/test_no_stac_at_runtime.py`.
    """

    SCENES = 8

    @pytest.fixture(scope="class")
    def bucket(self, tmp_path_factory):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        from conftest import write_inventory

        root = tmp_path_factory.mktemp("bucket")
        rows = []
        for n in range(self.SCENES):
            row = make_row(TILE, n)
            row["proj_shape_y"] = row["proj_shape_x"] = 64
            rows.append(row)
        inventory = write_inventory(root / "inventory.parquet", rows)

        blobs = {}
        for band, value in (("ST_B10", 40000), ("QA_PIXEL", 0b1000000)):
            path = root / f"{band}.TIF"
            with rasterio.open(
                path, "w", driver="GTiff", height=64, width=64, count=1,
                dtype="uint16", crs="EPSG:32620",
                transform=from_origin(608385.0, -3300285.0, 30, 30),
            ) as dst:  # fmt: skip
                dst.write(np.full((64, 64), value, dtype="uint16"), 1)
            key = staging.split_s3_uri(
                rows[0]["thermal_href" if band == "ST_B10" else "qa_href"]
            )[1]
            blobs[key] = path.read_bytes()
        return inventory, blobs

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory, bucket):
        inventory, blobs = bucket
        root = tmp_path_factory.mktemp("staged")

        class Bucket(FakeS3):
            def __init__(self):
                super().__init__()
                self.blobs = blobs

            def get_object(self, *, Bucket, Key, RequestPayer=None):  # noqa: N803
                import io

                self.calls.append((Bucket, Key))
                self.payers.append(RequestPayer)
                payload = self.blobs[Key]
                return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}

        def fake_client(pool_size: int | None = None):
            return Bucket()

        # `main` exposes no client seam, so the module default is swapped.
        # ty types a module-level def as that one function, so no replacement
        # is assignable to it; the shapes match, which is what matters here.
        original = staging._default_client
        staging._default_client = fake_client  # ty: ignore[invalid-assignment]
        try:
            code = shard_lst_p95.main(
                [
                    "--tile", TILE,
                    "--inventory-uri", str(inventory),
                    "--shard", "512",
                    "--max-shards", "4",
                    "--pixels-per-degree", "360",
                    "--workers", "2",
                    "--threads-per-worker", "1",
                    "--read-threads", "1",
                    "--no-output-mask",
                    "--stage-dir", str(root / "stage"),
                    "--out-dir", str(root / "out"),
                ]
            )  # fmt: skip
        finally:
            staging._default_client = original
        return (
            code,
            json.loads((root / "out" / "summary.json").read_text()),
            json.loads((root / "out" / "staging.json").read_text()),
            root,
        )

    def test_it_exits_zero(self, run):
        assert run[0] == 0

    def test_each_object_is_fetched_once(self, run):
        # The saving staging exists for. One GET per object, not one per shard
        # that reads it.
        report = run[2]
        assert report["objects"] == 2 * self.SCENES
        assert report["get_requests"] == report["objects"]
        assert report["retries"] == 0

    def test_staging_finishes_before_any_shard_runs(self, run):
        # The ordering the measurement settled: the fetch is processor-bound,
        # so sharing cores with the workers costs it 3.9x.
        phases = run[1]["phases"]
        assert phases["stage_end_s"] <= phases["first_submit_s"]
        assert phases["stage_s"] > 0.0

    def test_the_item_table_is_written_and_recorded(self, run):
        summary = run[1]
        table = summary["item_table"]
        assert table["n_items"] == self.SCENES
        assert table["bytes"] > 0
        assert Path(table["path"]).name == shard_lst_p95.ITEM_TABLE_NAME

    def test_a_part_file_is_written(self, run):
        import numpy as np

        root = run[3]
        part = np.load(root / "out" / "part-000.npz")
        assert part.files

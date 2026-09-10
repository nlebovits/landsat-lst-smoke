"""Each script has to run on a machine that holds only its own dependencies.

The suite runs inside the `dev` group, which is the union of every script's
inline block, so a module missing from one script's PEP 723 metadata is still
importable here and every other test passes. A fleet instance has no such
union. It runs `uv run <script>.py`, gets the environment the inline block
asks for, and stops at the first import the block forgot.

That gap cost a live `c6id.16xlarge`. `shard_lst_p95.py` gained
`from tile_inventory import ...` when the inventory replaced the per-tile
catalogue search, `tile_inventory.read_manifest` imports `pyarrow.parquet`, and
the inline block never gained `pyarrow`. Local runs missed it because
`--dry-run` and `--rehearse` are the two paths that never call
`load_tile_items`, and both were what got exercised.

Each test copies the scripts into a fresh directory before running them. That
is not tidiness. `uv` caches a script environment and reuses one whose contents
already satisfy the block, so a run that once declared `pyarrow` keeps resolving
after the line is deleted, and the check passes against a machine state no fleet
instance shares. A new path is a cold key, which is the fleet's condition.

Marked `packaging` because the run resolves against PyPI. It is opt-in for the
same reason the network tests are.

Scoped to the scripts a fleet instance runs: `shard_lst_p95.py` for the tile,
`fleet_plan.py` before the fleet starts, and `cost_report.py` after it stops.
`staging.py` and `tile_inventory.py` are covered through the first of those,
because they are imported rather than run. The bench and measurement tools stay
out: a missing dependency there costs an operator one retry, and the same
mistake in a fleet script costs an instance.

The module carries its own timeout. `addopts` sets `--timeout=60` for the
default suite, and a cold `uv` resolve of `frisky`, `odc-stac` and `geopandas`
can exceed that on its own, which would fail the test for the wrong reason.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SLICE = ROOT / "artifacts" / "inventory_slice.parquet"
TILE = "S30W065"

#: The subprocess allows 900 s and pytest-timeout has to allow at least as
#: much, or the CLI timeout is unreachable and a slow resolve reads as a
#: dependency failure.
pytestmark = [pytest.mark.packaging, pytest.mark.timeout(900)]


def fresh_checkout(tmp_path, *, with_mask=False):
    """The repository's scripts at a path `uv` has never resolved before.

    Mirrors what a fleet instance holds after `git clone`: the modules, the
    committed artifacts, and no `.venv`.

    Args:
        tmp_path: Where to build the checkout.
        with_mask: Also write the two artifacts the output mask reads. The
            buffered geometry is committed and is copied; the ASTER GED
            observation counts are gitignored and several gigabytes of granules
            behind, so a synthetic mosaic stands in. What is under test is
            whether the inline block declares what the mask imports, and a
            synthetic mosaic exercises the same imports as a real one.
    """
    work = tmp_path / "checkout"
    (work / "artifacts").mkdir(parents=True)
    for script in ROOT.glob("*.py"):
        shutil.copy2(script, work / script.name)
    shutil.copy2(SLICE, work / "artifacts" / SLICE.name)
    if with_mask:
        from conftest import LAND_GEOMETRY, write_numobs

        shutil.copy2(LAND_GEOMETRY, work / "artifacts" / LAND_GEOMETRY.name)
        write_numobs(work / "artifacts" / "aster_numobs.tif")
    return work


def run_script(work, *args, timeout=900):
    """One script, in the environment its own inline block declares."""
    return subprocess.run(
        ["uv", "run", *args],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class TestShardRuntimeResolves:
    def test_the_load_path_runs_on_the_inline_block_alone(self, tmp_path):
        """`--search-in-dry-run` is the cheapest path through `load_tile_items`.

        It reads the manifest, checks it, reads the row group, and builds the
        items, which is every import the fleet needs before it touches S3. It
        starts no cluster and reads no scene, so it costs nothing to run.
        """
        work = fresh_checkout(tmp_path)
        proc = run_script(
            work,
            "shard_lst_p95.py",
            "--tile",
            TILE,
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--search-in-dry-run",
        )
        assert proc.returncode == 0, (
            f"shard_lst_p95.py cannot run on its own dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr
        assert "total shard-scene reads" in proc.stdout

    def test_the_rehearsal_runs_on_the_inline_block_alone(self, tmp_path):
        """The rehearsal starts a real cluster, so it covers the submit path.

        It builds the output mask too, which is the newest reason this test
        exists. The mask reads a GeoPackage and a GeoTIFF, so the fleet's
        inline block has to declare rasterio, geopandas, shapely and pyogrio.
        A block that forgot one would pass every other test in the suite,
        because the suite runs inside the union of every block.
        """
        work = fresh_checkout(tmp_path, with_mask=True)
        proc = run_script(
            work,
            "shard_lst_p95.py",
            "--tile",
            TILE,
            "--rehearse",
            "40",
            "--max-shards",
            "8",
            "--workers",
            "2",
            "--threads-per-worker",
            "1",
            "--memory-limit-gib",
            "2",
            "--out-dir",
            str(tmp_path / "out2"),
        )
        assert proc.returncode == 0, (
            f"the rehearsal cannot run on its own dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr


class TestTheFleetPlannerResolves:
    """`fleet_plan.py` runs before the fleet, so its failure is the cheap one.

    It is still a failure that stops everything, and its inline block is the
    shortest in the repository, which is exactly the block a new import gets
    added around without being declared.
    """

    def test_it_builds_a_plan_on_the_inline_block_alone(self, tmp_path):
        # The planner screens every tile for ASTER emissivity, so it imports
        # the mask as well. Its inline block was the shortest in the
        # repository before this and is the one a new import gets added around.
        work = fresh_checkout(tmp_path, with_mask=True)
        shutil.copy2(
            ROOT / "artifacts" / "land_tiles.parquet",
            work / "artifacts" / "land_tiles.parquet",
        )
        proc = run_script(
            work,
            "fleet_plan.py",
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--out",
            str(tmp_path / "plan.json"),
        )
        assert proc.returncode == 0, (
            f"fleet_plan.py cannot run on its own dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr
        assert "tiles         3 to launch" in proc.stdout


class TestTheCostReportResolves:
    """`cost_report.py` prices the run after it ends.

    `pyproject` calls it standard library only. That claim is worth a check,
    because it is the reason nothing installs anything for it.
    """

    def test_it_prices_a_recorded_run_on_the_inline_block_alone(self, tmp_path):
        work = fresh_checkout(tmp_path)
        proc = run_script(
            work,
            "cost_report.py",
            # --tag is required and names an EC2 filter. With --recorded there
            # is nothing to look up, so the tag never reaches the API.
            "--tag",
            "lst-smoke-test",
            "--recorded",
            "c6i.16xlarge:1:3600",
            "--s3-get-requests",
            "1998",
        )
        assert proc.returncode == 0, (
            f"cost_report.py cannot run on its own dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr
        assert "GETs counted on the wire" in proc.stdout


def test_python_is_new_enough_for_the_pinned_floor():
    """The blocks pin >=3.12. A test that runs under less proves nothing."""
    assert sys.version_info >= (3, 12)

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

pytestmark = pytest.mark.packaging


def fresh_checkout(tmp_path):
    """The repository's scripts at a path `uv` has never resolved before.

    Mirrors what a fleet instance holds after `git clone`: the modules, the
    committed artifacts, and no `.venv`.
    """
    work = tmp_path / "checkout"
    (work / "artifacts").mkdir(parents=True)
    for script in ROOT.glob("*.py"):
        shutil.copy2(script, work / script.name)
    shutil.copy2(SLICE, work / "artifacts" / SLICE.name)
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
        """The rehearsal starts a real cluster, so it covers the submit path."""
        work = fresh_checkout(tmp_path)
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


def test_python_is_new_enough_for_the_pinned_floor():
    """The blocks pin >=3.12. A test that runs under less proves nothing."""
    assert sys.version_info >= (3, 12)

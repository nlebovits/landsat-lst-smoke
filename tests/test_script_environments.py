"""What a fleet instance holds after a clone has to be enough to run a tile.

The suite runs inside the `dev` group, so anything the tests import is
importable whether or not the project declares it. An instance has no such
slack. It clones, runs `uv sync --frozen --no-dev`, and stops at the first
import the dependency list forgot.

That gap cost a live `c6id.16xlarge`. `shard_lst_p95` gained an import of
`tile_inventory` when the inventory replaced the per-tile catalogue search,
`tile_inventory.read_manifest` imports `pyarrow.parquet`, and the declaration
never gained `pyarrow`. Local runs missed it because `--dry-run` and
`--rehearse` are the two paths that never call `load_tile_items`, and both were
what got exercised.

These tests build a checkout at a path `uv` has never resolved and sync it
there. That is not tidiness: `uv` reuses an environment whose contents already
satisfy a request, so a dependency deleted from `pyproject.toml` keeps
resolving on a machine that once had it, and the check would pass against a
state no instance shares. A new path is a cold key, which is the instance's
condition.

`--no-dev` is the flag `fleet/run.sh:25` uses, and running anything else here
would test a machine the fleet never builds. Not `--only-group`: that installs
a group instead of the project, so the console scripts would be absent, which
is a failure mode worth stating because it looks like the right flag.

Marked `packaging` because the sync resolves against PyPI. Opt-in for the same
reason the network tests are, which also means the default suite cannot see a
failure here. `.github/workflows/ci.yml` runs it as its own step.

Scoped to the commands an instance runs: `lst-prep` and `lst-shard` for the
tile, `lst-fleet-plan` before the fleet starts, and `lst-cost-report` after it
stops. The library modules are covered through those, because they are imported
rather than invoked. The measurement tools stay out: a missing dependency there
costs an operator one retry, and the same mistake in a fleet command costs an
instance.

The module carries its own timeout. `addopts` sets `--timeout=60` for the
default suite, and a cold resolve of `frisky`, `odc-stac` and `geopandas` can
exceed that on its own, which would fail the test for the wrong reason.
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

#: What a clone carries that the project needs to build and resolve.
CHECKOUT_FILES = ("pyproject.toml", "uv.lock", "README.md")

#: The subprocess allows 900 s and pytest-timeout has to allow at least as
#: much, or the CLI timeout is unreachable and a slow resolve reads as a
#: dependency failure.
pytestmark = [pytest.mark.packaging, pytest.mark.timeout(900)]


def fresh_checkout(tmp_path, *, with_mask=False):
    """The repository at a path `uv` has never resolved, synced the fleet's way.

    Mirrors what an instance holds after `git clone`: the package source, the
    project files, the committed artifacts, and no `.venv` until the sync
    builds one.

    The package is copied as a tree rather than as a glob of files. The
    previous version globbed `*.py` at the repository root, which was right
    while every module was a root file and silently copied nothing the moment
    they moved. A tree copy tracks the layout instead of restating it.

    Args:
        tmp_path: Where to build the checkout.
        with_mask: Also write the artifacts the output mask reads, and restamp
            the tile list and the inventory so the three agree about the land
            geometry. The committed geometry slice stands in for the 16 MB
            artifact and the synthetic mosaic for the 45 MB one. Both are
            gitignored and neither is what is under test: this asks whether the
            project declares what the mask imports, and a stand-in exercises
            the same imports as the real thing.
    """
    work = tmp_path / "checkout"
    (work / "artifacts").mkdir(parents=True)
    shutil.copytree(ROOT / "src", work / "src")
    for name in CHECKOUT_FILES:
        shutil.copy2(ROOT / name, work / name)
    shutil.copy2(SLICE, work / "artifacts" / SLICE.name)

    # The flag the instance uses, at `fleet/run.sh:25`. A failure here is the
    # one this module exists to catch, so it is reported rather than raised as
    # a CalledProcessError with the output swallowed.
    sync = subprocess.run(
        ["uv", "sync", "--frozen", "--no-dev"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert sync.returncode == 0, (
        f"a clean checkout cannot sync the way fleet/run.sh does:\n"
        f"{sync.stdout[-2000:]}\n{sync.stderr[-2000:]}"
    )
    if not with_mask:
        return work

    from conftest import LAND_GEOMETRY, restamped_plan_inputs, write_numobs

    # Under the names the commands default to, not the fixtures' own. An
    # instance holds the real geometry at `artifacts/land_buffered.gpkg`, and
    # the point of this checkout is to be that instance.
    shutil.copy2(LAND_GEOMETRY, work / "artifacts" / "land_buffered.gpkg")
    write_numobs(work / "artifacts" / "aster_numobs.tif")

    # And internally consistent. `lst-fleet-plan` refuses a plan whose tile
    # list, inventory and mosaic disagree about the land geometry, so all three
    # name the digest of the geometry this checkout actually holds. The
    # restamping lives in conftest because the `masked_plan_inputs` fixture
    # needs the same agreement, and two copies of it can drift apart.
    restamped_plan_inputs(work / "artifacts")
    return work


def run_script(work, *args, timeout=900):
    """One console script, in the environment the fleet's sync produced.

    `--no-sync`, so this runs against what `fresh_checkout` installed and
    cannot quietly repair a missing dependency by resolving again.
    """
    return subprocess.run(
        ["uv", "run", "--no-sync", *args],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class TestShardRuntimeResolves:
    def test_the_load_path_runs_on_the_declared_dependencies(self, tmp_path):
        """The inventory read, which is where the missing `pyarrow` bit.

        There is no longer a flag that reads the inventory and stops, because
        `--dry-run` plans the blocks from the bbox alone. So the run is given a
        window the artifact does not cover: `load_tile_items` reads the
        manifest through `pyarrow.parquet` and then refuses it, which is every
        import the fleet needs before it touches S3, and no GET.

        The failure is therefore expected. What must not appear is the other
        kind: a module the project never declared.
        """
        work = fresh_checkout(tmp_path)
        proc = run_script(
            work,
            "lst-shard",
            "--tile",
            TILE,
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--start",
            "2019-01-01",
            "--no-output-mask",
            "--out-dir",
            str(tmp_path / "out"),
        )
        assert "ModuleNotFoundError" not in proc.stderr, (
            f"shard_lst_p95.py cannot run on the declared dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "InventoryError" in proc.stderr, (
            f"the run stopped somewhere other than the manifest gate:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )

    def test_the_rehearsal_runs_on_the_declared_dependencies(self, tmp_path):
        """The rehearsal starts a real cluster, so it covers the whole graph.

        It builds the output mask too, which is the newest reason this test
        exists. The mask reads a GeoPackage and a GeoTIFF, so the fleet's
        dependency list has to declare rasterio, geopandas, shapely and pyogrio.
        A block that forgot one would pass every other test in the suite,
        because the suite runs inside the union of every block.

        It writes the catalog as well, and that is deliberate: `rio.to_raster`
        needs rioxarray and the item mirror needs stac-geoparquet, and neither
        is reached by any shorter path.
        """
        work = fresh_checkout(tmp_path, with_mask=True)
        proc = run_script(
            work,
            "lst-shard",
            "--tile",
            TILE,
            "--rehearse",
            "8",
            "--pixels-per-degree",
            "120",
            "--chunk",
            "100",
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
            f"the rehearsal cannot run on the declared dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr


class TestThePrepRuntimeResolves:
    """`lst-prep` runs first on an instance and was covered by nothing.

    `fleet/run.sh` runs it at line 71, before `lst-shard` at line 95. It was the
    only entry point without a packaging check, and it was the first one an
    instance executes. A prep that cannot start wastes the whole box.

    This covers the declared dependencies and nothing else. It does not pin an
    interpreter, and which one a subprocess gets here depends on how pytest was
    launched. Anything that varies by interpreter belongs in a check that runs
    no subprocess. See `tests/test_entry_point_parsers.py`.
    """

    def test_the_prep_path_runs_on_the_declared_dependencies(self, tmp_path):
        """A window the artifact does not cover, so the run stops at the gate.

        Same shape as the shard load-path test above: the manifest read pulls in
        every import the prep needs before it touches S3, and then refuses the
        artifact. The refusal is the expected end. What must not appear is an
        undeclared module, or a parser that cannot build itself.
        """
        work = fresh_checkout(tmp_path)
        proc = run_script(
            work,
            "lst-prep",
            "--tile",
            TILE,
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--start",
            "2019-01-01",
            "--out-dir",
            str(tmp_path / "prep"),
        )
        assert "ModuleNotFoundError" not in proc.stderr, (
            f"tile_prep.py cannot run on the declared dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "InventoryError" in proc.stderr, (
            f"the prep stopped somewhere other than the manifest gate:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )


class TestTheFleetPlannerResolves:
    """`lst.fleet.planner` runs before the fleet, so its failure is the cheap one.

    It is still a failure that stops everything, and its dependency surface is the
    shortest in the repository, which is exactly the block a new import gets
    added around without being declared.
    """

    def test_it_builds_a_plan_on_the_declared_dependencies(self, tmp_path):
        # The planner screens every tile for ASTER emissivity, so it imports
        # the mask as well. Its dependency surface was the smallest in the
        # repository before this and is the one a new import gets added around.
        work = fresh_checkout(tmp_path, with_mask=True)
        proc = run_script(
            work,
            "lst-fleet-plan",
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--out",
            str(tmp_path / "plan.json"),
        )
        assert proc.returncode == 0, (
            f"fleet_plan.py cannot run on the declared dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr
        assert "tiles         3 to launch" in proc.stdout

    def test_the_default_plan_needs_no_unbuffered_geometry(self, tmp_path):
        """`fleet/config.toml` ships six artifacts to an instance and
        `land_strict.gpkg` is not one of them, so an instance and this checkout
        both plan without it. The coverage screen once ran by default and made
        the planner exit 1 here, on a run whose plan had already been written.
        """
        work = fresh_checkout(tmp_path, with_mask=True)
        assert not (work / "artifacts" / "land_strict.gpkg").exists()
        proc = run_script(
            work,
            "lst-fleet-plan",
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--out",
            str(tmp_path / "plan.json"),
        )
        assert proc.returncode == 0, proc.stdout[-2000:]
        assert not (work / "artifacts" / "fleet_plan.jsonl").exists()
        assert "coverage" not in proc.stdout

    def test_the_screen_runs_on_the_inline_block_alone(self, tmp_path):
        """And on the same dependencies. The screen rasterises two geometries
        rather than one, so it is the import a new dependency arrives through.
        """
        import json

        from conftest import STRICT_LAND_GEOMETRY

        work = fresh_checkout(tmp_path, with_mask=True)
        shutil.copy2(STRICT_LAND_GEOMETRY, work / "artifacts" / "land_strict.gpkg")
        proc = run_script(
            work,
            "lst-fleet-plan",
            "--coverage",
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--out",
            str(tmp_path / "plan.json"),
            "--out-coverage",
            str(tmp_path / "screen.jsonl"),
        )
        assert proc.returncode == 0, f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        assert "ModuleNotFoundError" not in proc.stderr
        lines = (tmp_path / "screen.jsonl").read_text().splitlines()
        assert len(lines) == 895
        assert json.loads(lines[0])["tile_id"]

    def test_asking_for_the_screen_without_the_geometry_says_which_command(
        self, tmp_path
    ):
        """Opt-in, and loud when the person opted in. The silent skip is the
        other failure this pair guards."""
        work = fresh_checkout(tmp_path, with_mask=True)
        proc = run_script(
            work,
            "lst-fleet-plan",
            "--coverage",
            "--inventory-uri",
            "artifacts/inventory_slice.parquet",
            "--out",
            str(tmp_path / "plan.json"),
        )
        assert proc.returncode == 1
        assert "--write-strict-geometry" in proc.stdout


class TestTheCostReportResolves:
    """`lst.fleet.cost_report` prices the run after it ends.

    `pyproject` calls it standard library only. That claim is worth a check,
    because it is the reason nothing installs anything for it.
    """

    def test_it_prices_a_recorded_run_on_the_declared_dependencies(self, tmp_path):
        work = fresh_checkout(tmp_path)
        proc = run_script(
            work,
            "lst-cost-report",
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
            f"cost_report.py cannot run on the declared dependencies:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert "ModuleNotFoundError" not in proc.stderr
        assert "GETs counted on the wire" in proc.stdout


def test_python_is_new_enough_for_the_pinned_floor():
    """`requires-python` pins >=3.12. A test running under less proves nothing."""
    assert sys.version_info >= (3, 12)

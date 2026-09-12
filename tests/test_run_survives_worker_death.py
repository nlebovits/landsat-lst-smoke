"""What a fleet driver may key on when workers die under it.

`frisky` aborts a worker process on a Rust panic it cannot unwind, and this
pipeline has seen that happen. `FINDINGS.md` recorded it as a teardown panic
that leaves the exit code meaningless. Both halves were wrong. The console
lines put it before the dashboard print, so it fires at cluster start, and the
exit code is 0.

More to the point, the panic loses no work. frisky reschedules a dead worker's
task, so what actually costs a tile is a task that raises. The shard graph
caught those per shard, recorded them in the summary, wrote a part file and
returned 0, and a driver reading the exit code would call that tile done. The
dask-xarray graph catches nothing: a block that raises propagates out of
`composite.compute_all`, through `main`, and off the exit code.

So two properties, and they pull in opposite directions:

* a worker dying is survivable, and must stay exit 0, with a COG at the end
* a block erring is not, and must not

Both are exercised through `main`, because the exit code is the contract and
only `main` produces one. The rehearsal path is used throughout: it starts a
real cluster with real worker processes and reads no scene, so these cost
nothing and touch no bucket.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TILE = "S30W065"

#: 1,800 px of tile in 200 px blocks: 81 of them. Enough that the cluster is
#: still working when the killer fires, and few enough to stay under a minute.
PPD = 360
CHUNK = 200
SCENES = 200
WORKERS = 8

pytestmark = pytest.mark.timeout(600)


def argv(out_dir, *extra):
    """The production entry point, in the mode that starts a cluster and reads
    nothing. `--no-catalog` because the assertion is about the pixels landing,
    not about the catalog: `tests/test_merge_catalog.py` covers that.
    """
    return [
        sys.executable,
        str(ROOT / "shard_lst_p95.py"),
        "--tile",
        TILE,
        "--rehearse",
        str(SCENES),
        "--pixels-per-degree",
        str(PPD),
        "--chunk",
        str(CHUNK),
        "--workers",
        str(WORKERS),
        "--threads-per-worker",
        "1",
        "--memory-limit-gib",
        "2",
        # This module is about worker death, not about the mask, and it runs
        # from a checkout that need not hold the ASTER GED artifact.
        # `test_output_mask_run.py` covers the masked path.
        "--no-output-mask",
        "--no-catalog",
        "--out-dir",
        str(out_dir),
        *extra,
    ]


def rehearsal(out_dir, extra=()):
    """Run it to completion as a subprocess, because the assertion is a code."""
    return subprocess.run(
        argv(out_dir, *extra),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=540,
        check=False,
    )


def opens_as_a_raster(path: Path) -> tuple[int, int]:
    import rasterio

    with rasterio.open(path) as src:
        return src.height, src.width


class TestAWorkerDyingDoesNotLoseTheTile:
    """SIGABRT is what a non-unwinding panic does to a worker process."""

    @pytest.fixture(scope="class")
    def aborted(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("aborted")
        proc = subprocess.Popen(
            argv(out),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        killed = []

        def reaper():
            # Wait for the workers to exist, then abort half of them. Reading
            # children from /proc keeps this to the standard library.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and len(killed) < WORKERS // 2:
                try:
                    kids = (
                        Path(f"/proc/{proc.pid}/task/{proc.pid}/children")
                        .read_text()
                        .split()
                    )
                except OSError:
                    return
                for pid in kids:
                    if len(killed) >= WORKERS // 2:
                        break
                    try:
                        os.kill(int(pid), signal.SIGABRT)
                    except OSError:
                        continue
                    killed.append(int(pid))
                time.sleep(0.2)

        killer = threading.Thread(target=reaper, daemon=True)
        killer.start()
        stdout, _ = proc.communicate(timeout=540)
        killer.join(timeout=5)
        return proc.returncode, stdout, killed, out

    def test_workers_really_died(self, aborted):
        """Otherwise the rest of this class passes for the wrong reason."""
        _, _, killed, _ = aborted
        assert len(killed) == WORKERS // 2, f"aborted only {killed}"

    def test_the_run_still_succeeds(self, aborted):
        code, stdout, _, _ = aborted
        assert code == 0, stdout[-3000:]

    def test_no_block_is_lost(self, aborted):
        """frisky reschedules a dead worker's task, so the tile is complete.

        There is no per-block error record to walk any more, and there does not
        need to be: a block that never finished would have taken the exit code
        with it. What is left to check is that the raster the run promised is
        on disk and whole.
        """
        _, stdout, _, out = aborted
        cog = out / "lst_p95.tif"
        assert cog.is_file(), stdout[-3000:]
        assert opens_as_a_raster(cog) == (5 * PPD, 5 * PPD)

    def test_both_bands_land(self, aborted):
        _, _, _, out = aborted
        assert (out / "qa_count.tif").is_file()
        assert (out / "summary.json").is_file()


class TestABlockErringFailsTheRun:
    """The failure that used to report success.

    A shard's exceptions were caught per shard, so one bad shard could not kill
    the tile. That was right. Returning 0 afterwards was not: the run wrote a
    part file and a summary for a tile it knew was incomplete.

    The graph has no such catch, and this is the check that none grows back.
    """

    def test_a_clean_rehearsal_is_the_control(self, tmp_path):
        """The same command with nothing broken exits 0."""
        proc = rehearsal(tmp_path / "ok")
        assert proc.returncode == 0, proc.stdout[-2000:]
        assert (tmp_path / "ok" / "lst_p95.tif").is_file()

    def test_a_kernel_that_raises_is_not_exit_zero(self, tmp_path):
        """The block kernel is broken in a copy of the checkout.

        Not monkeypatched in this process: the workers are spawned processes
        that re-import `composite` themselves, so a patch here would never
        reach them. Same reason `test_script_environments.py` copies before it
        runs.
        """
        work = tmp_path / "checkout"
        work.mkdir(parents=True)
        for script in ROOT.glob("*.py"):
            shutil.copy2(script, work / script.name)

        target = work / "composite.py"
        source = target.read_text()
        marker = "def reduce_block("
        # Past the signature's closing paren and past the docstring, so the
        # raise is the first statement the kernel runs.
        body = source.index('    """\n', source.index(marker)) + len('    """\n')
        target.write_text(
            source[:body]
            + '    raise RuntimeError("the block failed on purpose")\n'
            + source[body:]
        )

        out = tmp_path / "broken"
        proc = subprocess.run(
            [
                sys.executable,
                "shard_lst_p95.py",
                *argv(out)[2:],
                # Small: the run is expected to die on the first block, so
                # there is nothing to be gained by writing 200 scenes first.
                "--rehearse",
                "20",
                "--workers",
                "2",
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode != 0, proc.stdout[-2000:]
        assert "the block failed on purpose" in proc.stdout + proc.stderr
        assert not (out / "lst_p95.tif").exists()
        assert not (out / "summary.json").exists()

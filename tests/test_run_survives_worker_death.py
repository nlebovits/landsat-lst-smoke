"""What a fleet driver may key on when workers die under it.

`frisky` aborts a worker process on a Rust panic it cannot unwind, and this
pipeline has seen that happen. `FINDINGS.md` recorded it as a teardown panic
that leaves the exit code meaningless. Both halves were wrong. The console
lines put it between the worst-shard print and the dashboard print, so it fires
at cluster start, and the exit code is 0.

More to the point, the panic loses no work. frisky reschedules a dead worker's
task, so what actually costs a tile is a shard that raises: `main` gathered
whatever arrived, recorded the rest under `error` in `shard_stats`, wrote a
part file and a summary, and returned 0. A driver reading the exit code, or
reading the summary without walking `shard_stats`, would call that tile done.

So two properties, and they pull in opposite directions:

* a worker dying is survivable, and must stay exit 0
* a shard erring is not, and must not

Both are exercised through `main`, because the exit code is the contract and
only `main` produces one. The rehearsal path is used throughout: it starts a
real cluster with real worker processes and reads no scene, so these cost
nothing and touch no bucket.
"""

from __future__ import annotations

import json
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

#: Enough shards that the cluster is still working when the killer fires, and
#: few enough that the test stays under a few seconds.
SHARDS = 200
WORKERS = 8

pytestmark = pytest.mark.timeout(600)


def rehearsal(out_dir, slice_artifact, extra=()):
    """The production entry point, in the mode that starts a cluster and reads
    nothing. Run as a subprocess because the assertion is about an exit code.
    """
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "shard_lst_p95.py"),
            "--tile",
            TILE,
            "--inventory-uri",
            str(slice_artifact),
            "--rehearse",
            "200",
            "--max-shards",
            str(SHARDS),
            "--workers",
            str(WORKERS),
            "--threads-per-worker",
            "1",
            "--read-threads",
            "1",
            "--out-dir",
            str(out_dir),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=540,
    )


class TestAWorkerDyingDoesNotLoseTheTile:
    """SIGABRT is what a non-unwinding panic does to a worker process."""

    @pytest.fixture(scope="class")
    def aborted(self, tmp_path_factory, slice_artifact):
        out = tmp_path_factory.mktemp("aborted")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "shard_lst_p95.py"),
                "--tile",
                TILE,
                "--inventory-uri",
                str(slice_artifact),
                "--rehearse",
                "200",
                "--max-shards",
                str(SHARDS),
                "--workers",
                str(WORKERS),
                "--threads-per-worker",
                "1",
                "--read-threads",
                "1",
                "--out-dir",
                str(out),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        killed = []

        def reaper():
            # Wait for the workers to exist, then abort half of them. Reading
            # children from /proc keeps this to the standard library.
            deadline = time.monotonic() + 60
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
        summary_path = out / "summary.json"
        summary = (
            json.loads(summary_path.read_text()) if summary_path.exists() else None
        )
        return proc.returncode, summary, stdout, killed

    def test_workers_really_died(self, aborted):
        """Otherwise the rest of this class passes for the wrong reason."""
        _, _, _, killed = aborted
        assert len(killed) == WORKERS // 2, f"aborted only {killed}"

    def test_the_run_still_succeeds(self, aborted):
        code, _, stdout, _ = aborted
        assert code == 0, stdout[-3000:]

    def test_no_shard_is_lost(self, aborted):
        """frisky reschedules a dead worker's task, so the tile is complete."""
        _, summary, stdout, _ = aborted
        assert summary is not None
        assert summary["n_shards_errored"] == 0, stdout[-3000:]
        assert len(summary["shard_stats"]) == summary["n_shards"] == SHARDS

    def test_the_summary_says_the_tile_is_whole(self, aborted):
        # The field a driver reads. Walking shard_stats is the thing it should
        # not have to do.
        _, summary, _, _ = aborted
        assert summary["n_shards_errored"] == 0


class TestAShardErringFailsTheRun:
    """The failure that used to report success.

    `process_shard` runs inside a worker and its exceptions are caught per
    shard, so one bad shard cannot kill the tile. That is right. Returning 0
    afterwards was not: the run wrote a part file and a summary for a tile it
    knew was incomplete.
    """

    def test_a_clean_rehearsal_is_the_control(self, tmp_path, slice_artifact):
        """The same command with nothing broken exits 0."""
        proc = rehearsal(tmp_path / "ok", slice_artifact)
        assert proc.returncode == 0, proc.stdout[-2000:]

    def test_the_exit_code_reports_lost_shards(self, tmp_path, slice_artifact):
        """Exit 3, the console line, and the count in the summary.

        The shard function is broken in a copy of the checkout rather than
        patched in this process, because the worker re-imports the script and
        the script runs as `__main__`. Same reason
        `test_script_environments.py` copies before it runs.
        """
        work = tmp_path / "checkout"
        (work / "artifacts").mkdir(parents=True)
        for script in ROOT.glob("*.py"):
            shutil.copy2(script, work / script.name)
        shutil.copy2(slice_artifact, work / "artifacts" / slice_artifact.name)

        target = work / "shard_lst_p95.py"
        source = target.read_text()
        marker = "def rehearse_shard("
        body = source.index(":\n", source.index(marker)) + 2
        target.write_text(
            source[:body]
            + '    raise RuntimeError("shard failed on purpose")\n'
            + source[body:]
        )

        out = tmp_path / "out"
        proc = subprocess.run(
            [
                sys.executable,
                "shard_lst_p95.py",
                "--tile",
                TILE,
                "--inventory-uri",
                f"artifacts/{slice_artifact.name}",
                "--rehearse",
                "40",
                "--max-shards",
                "4",
                "--workers",
                "2",
                "--threads-per-worker",
                "1",
                "--read-threads",
                "1",
                "--out-dir",
                str(out),
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=300,
        )
        summary = json.loads((out / "summary.json").read_text())
        # Every shard failed, and the run says so three ways.
        assert summary["n_shards_errored"] == summary["n_shards"] == 4
        assert "FAILED" in proc.stdout, proc.stdout[-2000:]
        assert proc.returncode == 3, proc.stdout[-2000:]

    def test_the_summary_still_lands_for_the_post_mortem(
        self, tmp_path, slice_artifact
    ):
        """Exit 3 must not mean no artifact. The errors are the evidence."""
        proc = rehearsal(tmp_path / "ok2", slice_artifact)
        summary = json.loads((tmp_path / "ok2" / "summary.json").read_text())
        assert proc.returncode == 0
        assert "n_shards_errored" in summary

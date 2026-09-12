"""Ranged staging, over a real socket, against real `Content-Range` headers.

`test_staging.py` owns the request count and the saving it protects. This owns
the transport: whether the bytes a split fetch reassembles are the bytes a
single GET returns, and whether the settings that produced a run reach the
report that describes it.

The split cannot be checked against a fake client. `staging._fetch_ranged`
takes the object's size out of a header, derives every offset from it, and lets
several threads `pwrite` into one descriptor. A range that is one byte short,
a total read from the wrong side of the slash in `bytes 0-8388607/88080384`,
or two parts landing at the same offset each produce a file of the right length
holding the wrong bytes, and a fake `get_object` returning a whole `BytesIO`
would pass all three. `tests/local_s3.py` serves real ranges instead.

Everything binds `127.0.0.1` on a kernel-chosen port. No test here reaches a
network, and none of them needs credentials that exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import local_s3  # noqa: E402
import stage_bench  # noqa: E402
import staging  # noqa: E402

#: Small enough that a whole test module of objects costs a few megabytes, and
#: still large enough to need several `COPY_BUFFER_BYTES` reads per part.
PART = 64 * 1024

BUCKET = "usgs-landsat"


@pytest.fixture
def server():
    with local_s3.LocalS3() as running:
        yield running


def manifest_for(server, sizes):
    """One synthetic object per size, as a staging manifest."""
    entries = []
    for i, size in enumerate(sizes):
        key = f"collection02/level-2/scene-{i}/ST_B10.TIF"
        href = server.add_sized(BUCKET, key, size)
        entries.append((f"scene-{i}", "lwir11", href))
    return entries


def stage(server, manifest, stage_dir, **kw):
    """Run a manifest through `StagingRun` against the loopback server."""
    settings = staging.FetchSettings.build(**kw)
    run = staging.StagingRun(
        manifest,
        stage_dir,
        settings=settings,
        client_factory=lambda n: local_s3.client_for(server, n),
    )
    with run:
        run.drain()
    return run.report()


def staged_bytes(stage_dir, manifest):
    """What landed, in manifest order."""
    return [
        staging.staged_path(
            stage_dir, item, band, staging.split_s3_uri(href)[1]
        ).read_bytes()
        for item, band, href in manifest
    ]


class TestReassembly:
    """A split fetch has to produce the file a single GET produces."""

    #: Sizes on both sides of the part threshold and on it. The three
    #: boundaries are where range arithmetic goes wrong: an object of exactly
    #: one part must not ask for a second, an object of exactly two must not
    #: ask for a third, and a partial last part must be the remainder rather
    #: than a whole one clamped by the server.
    SIZES = (1, PART - 1, PART, PART + 1, 2 * PART, 2 * PART + 1, 5 * PART + 977)

    def test_the_split_fetch_matches_the_single_get_byte_for_byte(
        self, server, tmp_path
    ):
        manifest = manifest_for(server, self.SIZES)

        stage(server, manifest, tmp_path / "whole", part_bytes=PART)
        stage(
            server,
            manifest,
            tmp_path / "split",
            part_bytes=PART,
            part_concurrency=4,
            threads=3,
        )

        whole = staged_bytes(tmp_path / "whole", manifest)
        split = staged_bytes(tmp_path / "split", manifest)
        assert [len(b) for b in split] == list(self.SIZES)
        assert split == whole

    def test_the_bytes_are_the_ones_the_server_holds(self, server, tmp_path):
        # Length and self-consistency are not enough: two parts written at the
        # same offset would give a file of the right length holding one part
        # twice, and the comparison above would only catch it if the whole
        # fetch were wrong in the same way.
        size = 5 * PART + 977
        manifest = manifest_for(server, [size])

        stage(server, manifest, tmp_path, part_bytes=PART, part_concurrency=4)

        served = server.store[f"{BUCKET}/collection02/level-2/scene-0/ST_B10.TIF"]
        assert staged_bytes(tmp_path, manifest)[0] == served.at(0, size)

    def test_every_byte_of_the_object_is_requested_exactly_once(self, server, tmp_path):
        size = 5 * PART + 977
        manifest = manifest_for(server, [size])

        stage(server, manifest, tmp_path, part_bytes=PART, part_concurrency=2)

        spans = sorted((start, length) for _, start, length in server.ranges)
        assert spans[0][0] == 0
        covered = 0
        for start, length in spans:
            assert start == covered, f"a hole or an overlap at {start}"
            covered += length
        assert covered == size

    @pytest.mark.parametrize("mode", staging.FSYNC_MODES)
    def test_the_fsync_mode_does_not_change_the_bytes(self, server, tmp_path, mode):
        manifest = manifest_for(server, [3 * PART + 11])

        stage(server, manifest, tmp_path / mode, part_bytes=PART, fsync=mode)
        stage(
            server,
            manifest,
            tmp_path / f"{mode}-split",
            part_bytes=PART,
            part_concurrency=3,
            fsync=mode,
        )

        assert staged_bytes(tmp_path / mode, manifest) == staged_bytes(
            tmp_path / f"{mode}-split", manifest
        )


class TestTheRequestCount:
    """`staging.json` prices GETs. A count taken inside the module is checked
    here against one taken by the server that answered them."""

    def test_the_default_settings_issue_one_unranged_get_per_object(
        self, server, tmp_path
    ):
        # The saving `test_staging.py` guards, restated where a real client
        # could quietly disagree with a fake one. `download_file` and a
        # transfer manager would both show up here as several ranged GETs.
        manifest = manifest_for(server, [PART * 4] * 5)

        report = stage(server, manifest, tmp_path, threads=2)

        assert server.requests == 5
        assert server.ranged == 0
        assert report["get_requests"] == 5
        assert report["retries"] == 0

    def test_a_split_object_costs_one_get_per_part(self, server, tmp_path):
        manifest = manifest_for(server, [5 * PART + 1])

        report = stage(server, manifest, tmp_path, part_bytes=PART, part_concurrency=4)

        assert server.requests == 6
        assert server.ranged == 6
        assert report["get_requests"] == 6

    def test_an_object_inside_one_part_still_costs_one_get(self, server, tmp_path):
        # The 2 MB QA_PIXEL objects are most of the manifest. Splitting is for
        # the 84 MB thermal band, and the first range answers for the whole of
        # a small object, so raising the part concurrency must not touch them.
        manifest = manifest_for(server, [PART - 1, PART])

        report = stage(server, manifest, tmp_path, part_bytes=PART, part_concurrency=4)

        assert server.requests == 2
        assert report["get_requests"] == 2

    def test_a_failed_part_is_billed_and_retried(self, server, tmp_path):
        manifest = manifest_for(server, [3 * PART])
        key = f"{BUCKET}/collection02/level-2/scene-0/ST_B10.TIF"
        server.fail[key] = 1

        report = stage(server, manifest, tmp_path, part_bytes=PART, part_concurrency=3)

        # One failure, then the whole object again: 1 + 3 requests, one retry.
        assert report["get_requests"] == 4
        assert report["retries"] == 1
        assert server.requests == 4
        assert staged_bytes(tmp_path, manifest)[0] == server.store[key].at(0, 3 * PART)

    def test_retries_are_counted_apart_from_requests(self, server, tmp_path):
        # They were one number while every object took one GET. Deriving
        # retries from requests would report five on a clean six-part fetch.
        manifest = manifest_for(server, [6 * PART])

        report = stage(server, manifest, tmp_path, part_bytes=PART, part_concurrency=6)

        assert report["get_requests"] == 6
        assert report["retries"] == 0


class TestTheReportCarriesTheSettings:
    """A throughput figure is unreadable without them."""

    def test_every_setting_reaches_the_report(self, server, tmp_path):
        manifest = manifest_for(server, [PART])

        report = stage(
            server,
            manifest,
            tmp_path,
            threads=3,
            connections=7,
            part_bytes=PART,
            part_concurrency=2,
            fsync="dir",
        )

        assert report["settings"] == {
            "threads": 3,
            "connections": 7,
            "part_bytes": PART,
            "part_concurrency": 2,
            "fsync": "dir",
        }

    def test_an_empty_manifest_still_reports_the_settings(self, tmp_path):
        # `stage_scenes` returns early on a tile with no objects, and a report
        # missing the key would break a sweep reading them back.
        report = staging.stage_scenes([], [], tmp_path, threads=4)
        assert report["settings"]["threads"] == 4

    def test_the_defaults_are_what_staging_always_did(self):
        settings = staging.FetchSettings.build()
        assert settings.threads == staging._default_threads()
        assert settings.connections == settings.threads
        assert settings.part_concurrency == 1
        assert settings.multipart is False
        assert settings.fsync == "file"


class TestTheConnectionPool:
    """The pool is a setting of its own now, not a copy of the thread count."""

    def test_the_client_is_built_with_the_connections_asked_for(self, server, tmp_path):
        built = []

        def factory(n):
            built.append(n)
            return local_s3.client_for(server, n)

        manifest = manifest_for(server, [PART])
        run = staging.StagingRun(
            manifest,
            tmp_path,
            settings=staging.FetchSettings.build(threads=4, connections=11),
            client_factory=factory,
        )
        with run:
            run.drain()

        assert built == [11]

    def test_the_default_pool_covers_every_socket_a_split_fetch_wants(self):
        # 64 threads each holding 4 ranges open is 256 sockets. A pool sized to
        # the thread count would put three quarters of them in a queue, which
        # is the bug this default exists to avoid.
        settings = staging.FetchSettings.build(threads=64, part_concurrency=4)
        assert settings.connections == 256

    def test_the_real_client_gets_the_pool_and_the_retry_mode(self):
        client = staging._default_client(256)
        assert client.meta.config.max_pool_connections == 256
        assert client.meta.config.retries["total_max_attempts"] == 1


class TestRefusedSettings:
    """A setting no fetch can honour stops at the parse, not in a thread."""

    @pytest.mark.parametrize(
        "kw",
        [
            {"threads": -1},
            {"connections": 0},
            {"part_bytes": -8},
            {"part_concurrency": -2},
            {"fsync": "sometimes"},
        ],
    )
    def test_a_setting_out_of_range_is_refused(self, kw):
        with pytest.raises(staging.StagingError):
            staging.FetchSettings.build(**kw)

    def test_threads_and_settings_together_are_refused(self, tmp_path):
        # Two answers to one question, and whichever lost would lose silently.
        with pytest.raises(staging.StagingError, match="not both"):
            staging.stage_scenes(
                [],
                [],
                tmp_path,
                threads=4,
                settings=staging.FetchSettings.build(threads=8),
            )


class TestStageBench:
    """The sweep tool, driven against the loopback server instead of a bucket.

    It runs on an instance and this laptop has no bucket, so what is checked
    here is everything but the network: the refusal, the matrix, the counting
    and the arithmetic on the line it prints. The bytes it reports are the
    bytes the loopback server sent.
    """

    def test_it_refuses_to_run_out_of_region(self):
        # A cross-region sweep bills egress per gigabyte and measures a link
        # no fleet instance has. The flag is the only way past it.
        with pytest.raises(SystemExit, match="i-am-in-region"):
            stage_bench.check_region(
                stage_bench.parse_args(["--tile", "X", "--root", "/tmp"])
            )

    def test_the_default_matrix_sweeps_threads_against_part_concurrency(self):
        built = [stage_bench.settings_for(c) for c in stage_bench.default_matrix()]

        assert [
            (s.threads, s.part_concurrency, s.part_bytes // 1024**2) for s in built
        ] == [
            (64, 1, 8),
            (128, 1, 8),
            (64, 4, 8),
            (64, 4, 16),
            (128, 4, 8),
            (128, 4, 16),
        ]
        # Every socket a split fetch can want, rather than the thread count.
        assert [s.connections for s in built] == [64, 128, 256, 256, 512, 512]

    def test_an_unknown_config_key_stops_the_sweep(self):
        # A typo that fell through would run the default configuration under
        # another one's name and publish the number as that configuration's.
        with pytest.raises(SystemExit, match="unknown --config key"):
            stage_bench.parse_config("thread=64")

    def test_a_run_reports_the_bytes_the_server_sent(self, server, tmp_path):
        manifest = manifest_for(server, [4 * PART, 4 * PART, PART])
        settings = staging.FetchSettings.build(
            threads=2, part_bytes=PART, part_concurrency=2
        )

        result = stage_bench.run_once(
            manifest,
            tmp_path / "sample",
            settings,
            build=lambda n: local_s3.client_for(server, n),
        )

        assert result["bytes"] == 9 * PART
        assert result["mb_s"] == pytest.approx(
            result["bytes"] / 1e6 / result["seconds"]
        )
        assert result["objects_s"] == pytest.approx(3 / result["seconds"])
        # Four parts each for the two large objects, one for the small one.
        assert result["get_requests"] == 9
        assert result["wire_gets"] == 9
        assert server.requests == 9

    def test_the_sample_is_removed_between_configurations(self, server, tmp_path):
        # A leftover is skipped rather than fetched, and the configuration
        # after it would post a throughput figure for objects it never pulled.
        manifest = manifest_for(server, [PART, PART])
        settings = staging.FetchSettings.build(threads=2)
        sample = tmp_path / "sample"

        for _ in range(2):
            result = stage_bench.run_once(
                manifest,
                sample,
                settings,
                build=lambda n: local_s3.client_for(server, n),
            )
            assert result["reused"] == 0

        assert not sample.exists()
        assert server.requests == 4

"""Guards on the wave driver, the AOI selector, and the prune tool.

Each rule here protects an instance-hour or an object. A wave that loses its
width to a quota refusal pays a refusal every wave afterwards. A prune that
matches a keeper deletes a tile that took 43 minutes to build. A driver that
forgets to tear a wave down bills to the 75 minute deadline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lst.fleet import aoi, launch, prune, waves

SHA = "9e2b703abec9756945686d5e8037788a666d28e6"


@pytest.fixture(scope="module")
def cfg():
    return launch.load_config()


class TestAQuotaRefusalNarrowsTheWave:
    """`servicequotas` is denied, so the ceiling is discovered by meeting it.

    Before this, any AWS error that was not a capacity refusal stopped the run
    with `SystemExit`. A wave of 20 on an account that allows 8 would have
    killed the driver on the ninth instance and left eight billing.
    """

    def test_a_quota_code_raises_quota_exhausted(self, cfg, monkeypatch):
        detail = (
            "An error occurred (VcpuLimitExceeded) when calling the RunInstances "
            "operation: You have requested more vCPU capacity than your current "
            "vCPU limit of 640 allows."
        )
        monkeypatch.setattr(launch, "aws_try", lambda argv: (255, "", detail))
        with pytest.raises(launch.QuotaExhausted) as err:
            launch.run_instances(
                cfg, "lst-S30W065-x", "S30W065", Path("ud.sh"), say=lambda *a: None
            )
        assert err.value.code == "VcpuLimitExceeded"
        assert err.value.tile == "S30W065"

    def test_a_quota_refusal_does_not_try_the_other_zones(self, cfg, monkeypatch):
        """Quota is an account fact. Retrying four zones buys four refusals."""
        calls = []

        def fake(argv):
            calls.append(argv)
            return (255, "", "An error occurred (VcpuLimitExceeded) when calling it")

        monkeypatch.setattr(launch, "aws_try", fake)
        with pytest.raises(launch.QuotaExhausted):
            launch.run_instances(
                cfg, "n", "S30W065", Path("ud.sh"), say=lambda *a: None
            )
        assert len(calls) == 1

    def test_capacity_still_walks_the_zones(self, cfg, monkeypatch):
        """The existing fallback must survive the new branch beside it."""
        calls = []

        def fake(argv):
            calls.append(argv)
            return (
                255,
                "",
                f"An error occurred ({launch.CAPACITY_ERROR}) when calling it",
            )

        monkeypatch.setattr(launch, "aws_try", fake)
        with pytest.raises(SystemExit):
            launch.run_instances(
                cfg, "n", "S30W065", Path("ud.sh"), say=lambda *a: None
            )
        assert len(calls) == len(launch.subnet_candidates(cfg))


class TestARefusedInstanceLeavesNothingBehind:
    """A key pair reaches the manifest before `run-instances` is called.

    Without cleanup, every refusal across every wave leaves one dead key pair
    in EC2 and one private key in `~/.ssh`.
    """

    def test_the_key_is_deleted_and_the_entry_removed(self, cfg, tmp_path, monkeypatch):
        deleted = []
        monkeypatch.setattr(launch, "aws", lambda argv: "PRIVATE KEY MATERIAL")
        monkeypatch.setattr(
            launch, "delete_key_pair", lambda c, name: deleted.append(name) or True
        )

        def refuse(*args, **kwargs):
            raise launch.QuotaExhausted("S30W065", "VcpuLimitExceeded", "no room")

        monkeypatch.setattr(launch, "run_instances", refuse)
        # `key_path` refuses a key directory under /tmp, which is the rule that
        # keeps a reboot from stranding a running instance. The test wants the
        # cleanup, not that guard, so it hands the path over directly.
        keys = tmp_path / "keys"
        monkeypatch.setattr(launch, "key_path", lambda d, name: keys / f"{name}.pem")

        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, {"run_id": "x", "config": cfg})
        with pytest.raises(launch.QuotaExhausted):
            launch.launch_one(cfg, "S30W065", "x", False, tmp_path / "ud.sh", manifest)

        assert deleted == ["lst-S30W065-x"]
        assert json.loads(path.read_text())["instances"] == []
        assert not (keys / "lst-S30W065-x.pem").exists()


class TestTheDriverSortsAWave:
    """A tile is judged by its state, not by an exit status."""

    def test_a_finished_tile_settles(self):
        attempts = {"S30W065": 0}
        settled, requeued = waves.sort_wave(
            ["S30W065"],
            {"S30W065": "finished"},
            attempts=attempts,
            retries=1,
            say=lambda *a: None,
        )
        assert settled == {"S30W065": "finished"}
        assert requeued == []
        assert attempts["S30W065"] == 0

    def test_a_failed_tile_is_requeued_once_and_not_twice(self):
        attempts = dict.fromkeys(["S30W065"], 0)
        for _ in range(2):
            settled, requeued = waves.sort_wave(
                ["S30W065"],
                {"S30W065": "failed"},
                attempts=attempts,
                retries=1,
                say=lambda *a: None,
            )
        assert requeued == []
        assert settled == {"S30W065": "failed"}
        assert attempts["S30W065"] == 2

    def test_a_tile_quota_refused_is_requeued(self):
        """A tile that never launched costs nothing and deserves another wave."""
        attempts = {"S50W070": 0}
        _, requeued = waves.sort_wave(
            ["S50W070"],
            {"S50W070": "not-driven"},
            attempts=attempts,
            retries=1,
            say=lambda *a: None,
        )
        assert requeued == ["S50W070"]

    def test_an_unknown_tile_reads_as_not_driven(self):
        """A tile missing from the status map never reached an instance."""
        attempts = {"S50W070": 0}
        _, requeued = waves.sort_wave(
            ["S50W070"], {}, attempts=attempts, retries=1, say=lambda *a: None
        )
        assert requeued == ["S50W070"]


class TestTheWidthSurvivesTheWave:
    """Discovering the ceiling is worth nothing if the next wave forgets it."""

    def _args(self, tmp_path, **over):
        import argparse

        base = dict(
            width=4,
            max_waves=0,
            retries=0,
            fleet_dir=Path("fleet"),
            manifest_dir=tmp_path,
            poll_seconds=0.0,
            timeout_minutes=1.0,
        )
        return argparse.Namespace(**(base | over))

    def test_a_refusal_does_not_narrow_the_next_wave(self, tmp_path, monkeypatch):
        """The account may be busy for a reason that goes away.

        An earlier version kept the narrower number, which would have run a
        continent at a width discovered during one overlap with another wave.
        """
        widths = []

        def fake_wave(batch, **kwargs):
            widths.append(len(batch))
            placed = 2 if len(widths) == 1 else len(batch)
            statuses = {t: "finished" for t in batch[:placed]}
            statuses |= {t: "not-driven" for t in batch[placed:]}
            return statuses, placed, tmp_path / f"run-{len(widths)}.json"

        monkeypatch.setattr(waves, "run_wave", fake_wave)
        # retries=1 so the two refused tiles come back and the second wave has
        # a full queue to draw from. Otherwise a short second wave would look
        # like a narrowed one.
        done, leftover, wave_no, _ = waves.drain(
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            self._args(tmp_path, retries=1),
            cfg={},
            commit=SHA,
            user_data=tmp_path / "ud.sh",
            say=lambda *a: None,
        )
        # Wave 1 asks 4 and places 2. Wave 2 asks 4 again, not 2.
        assert widths[0] == 4
        assert widths[1] == 4
        assert leftover == []
        assert done == dict.fromkeys("ABCDEFGH", "finished")

    def test_max_waves_stops_and_reports_the_leftover(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            waves,
            "run_wave",
            lambda batch, **k: (
                {t: "finished" for t in batch},
                len(batch),
                tmp_path / "m.json",
            ),
        )
        done, leftover, wave_no, _ = waves.drain(
            ["A", "B", "C", "D", "E", "F"],
            self._args(tmp_path, width=2, max_waves=2),
            cfg={},
            commit=SHA,
            user_data=tmp_path / "ud.sh",
            say=lambda *a: None,
        )
        assert wave_no == 2
        assert len(done) == 4
        assert leftover == ["E", "F"]


class TestEveryWaveIsTornDown:
    """An instance that finished and was not terminated bills to its deadline."""

    def test_teardown_runs_even_when_the_wave_raises(self, tmp_path, monkeypatch, cfg):
        torn = []
        monkeypatch.setattr(waves, "teardown", lambda path, **k: torn.append(path) or 0)
        monkeypatch.setattr(
            waves, "launch_one", lambda *a, **k: {"tile": a[1], "name": "n", "pem": "p"}
        )
        monkeypatch.setattr(waves, "drive", lambda *a, **k: _Proc())

        def boom(*a, **k):
            raise RuntimeError("object storage is down")

        monkeypatch.setattr(waves, "_await_wave", boom)
        with pytest.raises(RuntimeError):
            waves.run_wave(
                ["S30W065"],
                cfg=cfg,
                commit=SHA,
                fleet_dir=Path("fleet"),
                manifest_dir=tmp_path,
                user_data=tmp_path / "ud.sh",
                poll_seconds=0.0,
                timeout_minutes=1.0,
                say=lambda *a: None,
            )
        assert len(torn) == 1


class _Proc:
    returncode = 0

    def poll(self):
        return 0

    def terminate(self):
        pass


class TestTeardownSweepsTheKeys:
    """100 tiles left 100 key pairs. The account limit is 5,000."""

    def test_every_named_instance_loses_its_key(self, cfg, tmp_path, monkeypatch):
        from lst.fleet import teardown

        deleted = []
        monkeypatch.setattr(
            teardown, "delete_key_pair", lambda c, name: deleted.append(name) or True
        )
        pem = tmp_path / "lst-S30W065-x.pem"
        pem.write_text("KEY")
        run = {
            "instances": [
                {"tile": "S30W065", "name": "lst-S30W065-x", "pem": str(pem)},
                {"tile": "S50W070", "name": "lst-S50W070-x"},
            ]
        }
        assert teardown.sweep_keys(cfg, run) == (2, 2)
        assert deleted == ["lst-S30W065-x", "lst-S50W070-x"]
        assert not pem.exists()


class TestThePruneRefusesAKeeper:
    """An exclude pattern that matches nothing deletes everything it guarded."""

    def test_a_misspelled_keeper_stops_the_command(self, monkeypatch):
        monkeypatch.setattr(prune, "run_prefixes", lambda uri: ["run-a", "run-b"])
        with pytest.raises(ValueError, match="does not hold"):
            prune.plan_runs("s3://b/runs", frozenset({"run-c"}))

    def test_a_named_keeper_is_never_in_the_delete_list(self, monkeypatch):
        monkeypatch.setattr(
            prune, "run_prefixes", lambda uri: ["run-a", "run-b", "run-c"]
        )
        doomed, kept = prune.plan_runs("s3://b/runs", frozenset({"run-b"}))
        assert doomed == ["run-a", "run-c"]
        assert kept == ["run-b"]

    def test_the_artifacts_prefix_cannot_be_deleted(self):
        """Every instance downloads it. Losing it stops the fleet."""
        with pytest.raises(ValueError, match="protected"):
            prune.delete_prefix(
                "s3://b/nlebovits/landsat-lst-test/artifacts", dry_run=True
            )

    def test_a_listing_line_without_a_size_is_ignored(self, monkeypatch):
        """`aws s3 ls` prints a `PRE name/` line that carries no object."""
        monkeypatch.setattr(
            prune,
            "s3",
            lambda *a: (
                "                           PRE runs/\n"
                "2026-09-15 17:02:36       1544 nlebovits/runs/lst-A/_MANIFEST.json\n"
            ),
        )
        assert prune.listing("s3://b/runs") == [
            ("nlebovits/runs/lst-A/_MANIFEST.json", 1544)
        ]


class TestTheAoiPartitionsThePlan:
    """A tile that falls through the partition is a tile nobody runs."""

    def test_exclude_accepts_both_spellings(self):
        assert aoi.parse_exclude(["S30W065,S50W075", "s35w055"]) == frozenset(
            {"S30W065", "S50W075", "S35W055"}
        )

    def test_the_four_lists_partition_the_plan(self, tmp_path, monkeypatch):
        plan = {"tiles": [{"tile_id": t} for t in ("S30W065", "S30W070", "N40E010")]}
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))

        class FakeTree:
            def __init__(self, geoms):
                pass

            def query(self, cell, predicate=None):
                # Only the two South American tiles intersect.
                return [0] if cell.bounds[0] < -60 else []

        monkeypatch.setattr(aoi, "load_continent", lambda c, d: _FakeLand())
        monkeypatch.setattr("shapely.STRtree", FakeTree)
        monkeypatch.setattr(aoi, "land_area_km2", lambda g: 1000.0)
        kept, off, dropped, slivers = aoi.select(
            "South America", plan_path=path, exclude=frozenset({"S30W065"})
        )
        assert [t for t, _ in kept] == ["S30W070"]
        assert off == ["N40E010"]
        assert dropped == ["S30W065"]
        assert len(kept) + len(off) + len(dropped) + len(slivers) == 3

    def test_a_sliver_below_the_threshold_leaves_the_kept_list(
        self, tmp_path, monkeypatch
    ):
        """N20W065 is South America only through a 0.6 km2 sandbar."""
        plan = {"tiles": [{"tile_id": "N20W065"}]}
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        monkeypatch.setattr(aoi, "load_continent", lambda c, d: _FakeLand())
        monkeypatch.setattr("shapely.STRtree", lambda geoms: _AlwaysHit())
        monkeypatch.setattr(aoi, "land_area_km2", lambda g: 0.6)
        kept, _, _, slivers = aoi.select(
            "South America", plan_path=path, min_land_km2=2.0
        )
        assert kept == []
        assert slivers == [("N20W065", 0.6)]


class _FakeLand:
    class _Geom:
        values = [
            __import__("shapely.geometry", fromlist=["box"]).box(-80, -40, -30, 10)
        ]

    geometry = _Geom()


class _AlwaysHit:
    def query(self, cell, predicate=None):
        return [0]


class TestTheWatcherFallsBackToTheCli:
    """botocore and the AWS CLI read different credential caches.

    MEASURED on 2026-09-15: the SSO access token expired at 15:27Z while the
    CLI's role credential stayed valid until 23:17Z. For those eight hours the
    watcher lost the `gone` state, which is the one that sends a dead instance
    back to the queue.
    """

    def test_the_cli_answers_when_botocore_refuses(self, monkeypatch):
        from lst.fleet import watch

        class Refuses:
            def describe_instances(self, InstanceIds):
                raise RuntimeError("UnauthorizedSSOTokenError")

        monkeypatch.setattr(watch, "running_via_cli", lambda ids, p, r: {"i-aaa"})
        assert watch.running_instances(
            Refuses(), ["i-aaa", "i-bbb"], profile="radiant-earth", region="us-west-2"
        ) == {"i-aaa"}

    def test_no_profile_means_no_fallback(self, monkeypatch):
        """The old two-argument call must keep its old behaviour."""
        from lst.fleet import watch

        class Refuses:
            def describe_instances(self, InstanceIds):
                raise RuntimeError("UnauthorizedSSOTokenError")

        called = []
        monkeypatch.setattr(
            watch, "running_via_cli", lambda ids, p, r: called.append(1) or set()
        )
        assert watch.running_instances(Refuses(), ["i-aaa"]) is None
        assert called == []

    def test_a_cli_that_also_fails_reports_not_knowing(self, monkeypatch):
        """Not knowing is a third answer. It must not read as `every instance is gone`."""
        from lst.fleet import watch

        class Refuses:
            def describe_instances(self, InstanceIds):
                raise RuntimeError("UnauthorizedSSOTokenError")

        monkeypatch.setattr(watch, "running_via_cli", lambda ids, p, r: None)
        assert (
            watch.running_instances(
                Refuses(), ["i-aaa"], profile="p", region="us-west-2"
            )
            is None
        )

    def test_the_cli_reports_nothing_alive_as_an_empty_set(self, monkeypatch):
        """An empty answer is knowledge. None is the absence of it."""
        from lst.fleet import watch

        monkeypatch.setattr(
            watch.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "\n"})(),
        )
        assert watch.running_via_cli(["i-aaa"], "p", "us-west-2") == set()


class TestADriverWaitsForTheUploadManifest:
    """`all_done` is the pipeline finishing. The manifest is the upload finishing.

    MEASURED on 2026-09-15: `N00W045` was torn down at `all_done` and lost its
    `_MANIFEST.json`. Its rasters had landed seconds earlier. A tile still
    pushing a 500 MB `qa_count.tif` would have lost the raster.
    """

    def _state(self, phase, status, detail=""):
        from lst.fleet.watch import State

        return State("S30W065", phase, status, detail)

    def test_the_marker_alone_is_not_done(self):
        assert not waves.is_done(
            self._state("all_done", "finished", "waiting on upload")
        )

    def test_the_manifest_is_done(self):
        assert waves.is_done(self._state("uploaded", "finished", "manifest complete"))

    def test_a_failure_is_done(self):
        assert waves.is_done(self._state("prep", "failed", "rc=1"))
        assert waves.is_done(self._state("prep", "gone", "instance ended"))

    def test_a_running_tile_is_not_done(self):
        assert not waves.is_done(self._state("prep", "running", "3s since beat"))


class TestAnUploadThatNeverArrives:
    """Waiting forever on a dead uploader holds 19 other machines."""

    def test_inside_the_grace_the_tile_still_counts_as_finished(self):
        waiting: dict[str, float] = {}
        out = waves.settle_uploads(
            {"A": "finished"}, {"A": "all_done"}, waiting, 0.0, grace_minutes=12.0
        )
        assert out["A"] == "finished"

    def test_past_the_grace_it_becomes_upload_lost(self):
        waiting: dict[str, float] = {}
        waves.settle_uploads(
            {"A": "finished"}, {"A": "all_done"}, waiting, 0.0, grace_minutes=12.0
        )
        out = waves.settle_uploads(
            {"A": "finished"}, {"A": "all_done"}, waiting, 800.0, grace_minutes=12.0
        )
        assert out["A"] == "upload-lost"

    def test_upload_lost_goes_back_to_the_queue(self):
        assert "upload-lost" in waves.RETRYABLE

    def test_a_manifest_that_arrives_clears_the_clock(self):
        waiting: dict[str, float] = {}
        waves.settle_uploads(
            {"A": "finished"}, {"A": "all_done"}, waiting, 0.0, grace_minutes=12.0
        )
        assert "A" in waiting
        out = waves.settle_uploads(
            {"A": "finished"}, {"A": "uploaded"}, waiting, 800.0, grace_minutes=12.0
        )
        assert out["A"] == "finished"
        assert "A" not in waiting


class TestAnOrphanedWaveCanBeAdopted:
    """A driver that dies leaves instances billing to their 75 minute deadline."""

    def test_adopt_tears_the_wave_down_even_when_watching_raises(
        self, tmp_path, monkeypatch, cfg
    ):
        torn = []
        monkeypatch.setattr(waves, "teardown", lambda path, **k: torn.append(path) or 0)

        def boom(*a, **k):
            raise RuntimeError("object storage is down")

        monkeypatch.setattr(waves, "_await_wave", boom)
        path = tmp_path / "run.json"
        path.write_text(json.dumps({"config": cfg, "instances": [{"tile": "S30W065"}]}))
        with pytest.raises(RuntimeError):
            waves.adopt(path, say=lambda *a: None)
        assert torn == [path]


class TestWhatIsLeftComesFromTheBucket:
    """A driver's tally dies with the driver. The bucket does not."""

    def _cfg(self):
        return {
            "storage": {
                "bucket": "b",
                "runs_prefix": "nl/lst-test/runs",
                "upload_profile": "source-coop",
            }
        }

    def _listing(self, text):
        return type("R", (), {"returncode": 0, "stdout": text, "stderr": ""})()

    def test_only_a_manifest_counts_as_uploaded(self, monkeypatch):
        listing = (
            "2026-09-15 17:02:36 1544 nl/lst-test/runs/lst-S30W065-20260915-1/"
            "_MANIFEST.json\n"
            "2026-09-15 17:01:31 49 nl/lst-test/runs/lst-N00W045-20260915-1/"
            "tile/catalog/lst-p95-2021-2025/N00W045/lst_p95.tif\n"
        )
        monkeypatch.setattr(
            waves.subprocess, "run", lambda *a, **k: self._listing(listing)
        )
        # N00W045 has rasters and no manifest, so it is not done.
        assert waves.uploaded_tiles(self._cfg()) == {"S30W065"}

    def test_a_tile_that_ran_twice_is_counted_once(self, monkeypatch):
        listing = (
            "2026-09-14 1 1 nl/lst-test/runs/lst-S30W065-20260914-1/_MANIFEST.json\n"
            "2026-09-15 1 1 nl/lst-test/runs/lst-S30W065-20260915-2/_MANIFEST.json\n"
        )
        monkeypatch.setattr(
            waves.subprocess, "run", lambda *a, **k: self._listing(listing)
        )
        assert waves.uploaded_tiles(self._cfg()) == {"S30W065"}

    def test_an_old_style_prefix_is_ignored(self, monkeypatch):
        """`N40W080-lst-N40W080-...` predates the naming rule and is pruned."""
        listing = (
            "2026-09-13 1 1 nl/lst-test/runs/N40W080-lst-N40W080-1/_MANIFEST.json\n"
        )
        monkeypatch.setattr(
            waves.subprocess, "run", lambda *a, **k: self._listing(listing)
        )
        assert waves.uploaded_tiles(self._cfg()) == set()

    def test_a_failed_listing_stops_rather_than_reporting_nothing_done(
        self, monkeypatch
    ):
        """Reporting an empty set would relaunch all 101 tiles."""
        monkeypatch.setattr(
            waves.subprocess,
            "run",
            lambda *a, **k: type(
                "R", (), {"returncode": 1, "stdout": "", "stderr": "AccessDenied"}
            )(),
        )
        with pytest.raises(SystemExit, match="cannot list"):
            waves.uploaded_tiles(self._cfg())


class TestAFinishedInstanceStopsBilling:
    """A wave costs its slowest tile times its width, not the sum of its tiles.

    MEASURED on 2026-09-15: wave 1 held 20 machines for 38.5 minutes each while
    its tiles finished between 16:20 and 16:50, at $2.44 a tile against the
    $1.35 a three-tile wave measured.
    """

    def _run(self):
        return {
            "instances": [
                {"tile": "A", "name": "lst-A-1", "instance_id": "i-a"},
                {"tile": "B", "name": "lst-B-1", "instance_id": "i-b"},
            ]
        }

    def _state(self, tile, phase, status):
        from lst.fleet.watch import State

        return State(tile, phase, status)

    def test_an_uploaded_tile_is_terminated_at_once(self, monkeypatch, cfg):
        killed = []
        monkeypatch.setattr(
            waves.launch,
            "aws_try",
            lambda argv: killed.append(argv[-1]) or (0, "", ""),
        )
        fresh = waves.reap(
            self._run(),
            cfg,
            [
                self._state("A", "uploaded", "finished"),
                self._state("B", "composite", "running"),
            ],
            set(),
            say=lambda *a: None,
        )
        assert killed == ["i-a"]
        assert fresh == {"lst-A-1"}

    def test_a_tile_at_all_done_is_left_alone(self, monkeypatch, cfg):
        """Its upload is still going. This is the bug that lost N00W045."""
        killed = []
        monkeypatch.setattr(
            waves.launch, "aws_try", lambda argv: killed.append(argv[-1]) or (0, "", "")
        )
        waves.reap(
            self._run(),
            cfg,
            [self._state("A", "all_done", "finished")],
            set(),
            say=lambda *a: None,
        )
        assert killed == []

    def test_a_failed_tile_keeps_its_instance_for_the_log(self, monkeypatch, cfg):
        """Its uploader may still be pushing the log that says why it failed."""
        killed = []
        monkeypatch.setattr(
            waves.launch, "aws_try", lambda argv: killed.append(argv[-1]) or (0, "", "")
        )
        waves.reap(
            self._run(),
            cfg,
            [self._state("A", "prep", "failed")],
            set(),
            say=lambda *a: None,
        )
        assert killed == []

    def test_an_instance_is_not_terminated_twice(self, monkeypatch, cfg):
        killed = []
        monkeypatch.setattr(
            waves.launch, "aws_try", lambda argv: killed.append(argv[-1]) or (0, "", "")
        )
        states = [self._state("A", "uploaded", "finished")]
        reaped = waves.reap(self._run(), cfg, states, set(), say=lambda *a: None)
        waves.reap(self._run(), cfg, states, reaped, say=lambda *a: None)
        assert killed == ["i-a"]

    def test_a_refused_termination_is_retried_next_poll(self, monkeypatch, cfg):
        """The wave teardown is the backstop, but the next poll tries again."""
        monkeypatch.setattr(
            waves.launch, "aws_try", lambda argv: (255, "", "UnauthorizedOperation")
        )
        fresh = waves.reap(
            self._run(),
            cfg,
            [self._state("A", "uploaded", "finished")],
            set(),
            say=lambda *a: None,
        )
        assert fresh == set()

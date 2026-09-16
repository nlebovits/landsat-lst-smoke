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
        with pytest.raises(launch.CapacityExhausted):
            launch.run_instances(
                cfg, "n", "S30W065", Path("ud.sh"), say=lambda *a: None
            )
        assert len(calls) == len(launch.subnet_candidates(cfg))

    def test_every_zone_full_does_not_kill_the_run(self, cfg, monkeypatch):
        """This used to be a `SystemExit`, which ended a whole continent.

        At width 20 it never fired. At width 60 or more, one tile meeting a
        full region would have taken the other 600 with it.
        """

        def fake(argv):
            return (
                255,
                "",
                f"An error occurred ({launch.CAPACITY_ERROR}) when calling it",
            )

        monkeypatch.setattr(launch, "aws_try", fake)
        with pytest.raises(launch.LaunchRefused) as caught:
            launch.run_instances(
                cfg, "n", "S30W065", Path("ud.sh"), say=lambda *a: None
            )
        assert not isinstance(caught.value, SystemExit)
        assert caught.value.tile == "S30W065"
        assert caught.value.code == launch.CAPACITY_ERROR

    def test_a_malformed_request_still_stops_the_run(self, cfg, monkeypatch):
        """Only quota and capacity are recoverable. Everything else is a bug."""

        def fake(argv):
            return (255, "", "An error occurred (InvalidGroup.NotFound) when calling")

        monkeypatch.setattr(launch, "aws_try", fake)
        with pytest.raises(SystemExit):
            launch.run_instances(
                cfg, "n", "S30W065", Path("ud.sh"), say=lambda *a: None
            )


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
            launch_workers=1,
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


class FakeProc:
    """A `drive.sh` that is already finished, or still going."""

    def __init__(self, returncode=0, alive=False):
        self.returncode = returncode
        self._alive = alive
        self.terminated = False

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False


class FakeState:
    """What `watch.classify` returns, without a bucket behind it."""

    def __init__(self, tile, status="running", phase="compositing", detail=""):
        self.tile = tile
        self.status = status
        self.phase = phase
        self.detail = detail


def a_manifest(tmp_path, cfg):
    return launch.RunManifest(
        tmp_path / "run.json", {"run_id": "x", "commit": SHA, "config": cfg}
    )


class TestInstancesArePlacedInParallel:
    """MEASURED: a serial launcher placed one instance every 22 seconds.

    Almost all of it is `aws ec2 wait instance-running`, which polls on a 15
    second cycle. 665 tiles serially is 4.1 hours of launching alone, which on
    its own misses a one-day run.
    """

    def test_every_tile_is_placed(self, cfg, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(
            waves,
            "launch_one",
            lambda cfg, tile, *a, **k: seen.append(tile) or {"tile": tile},
        )
        tiles = ["S30W065", "S30W060", "S35W055", "S35W060"]
        out = waves.launch_many(
            tiles,
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=4,
            say=lambda *a: None,
        )
        assert sorted(e["tile"] for e in out.placed) == sorted(tiles)
        assert out.refused == []
        assert sorted(seen) == sorted(tiles)

    def test_the_launches_actually_overlap(self, cfg, tmp_path, monkeypatch):
        """Four 0.2 s launches on four workers finish in well under 0.8 s."""
        import time

        def slow(cfg, tile, *a, **k):
            time.sleep(0.2)
            return {"tile": tile}

        monkeypatch.setattr(waves, "launch_one", slow)
        start = time.monotonic()
        waves.launch_many(
            ["S30W065", "S30W060", "S35W055", "S35W060"],
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=4,
            say=lambda *a: None,
        )
        assert (time.monotonic() - start) < 0.6

    def test_a_quota_refusal_stops_the_rest_from_asking(
        self, cfg, tmp_path, monkeypatch
    ):
        """Quota is account-wide. Asking again costs a key pair per tile."""
        asked = []

        def fake(cfg, tile, *a, **k):
            asked.append(tile)
            raise launch.QuotaExhausted(tile, "VcpuLimitExceeded", "full")

        monkeypatch.setattr(waves, "launch_one", fake)
        out = waves.launch_many(
            [f"S{n:02d}W060" for n in range(10, 60, 5)],
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=1,
            say=lambda *a: None,
        )
        assert out.quota is True
        assert len(asked) == 1
        assert len(out.refused) == 10

    def test_a_capacity_refusal_stops_only_its_own_tile(
        self, cfg, tmp_path, monkeypatch
    ):
        """One full region moment is not a reason to stop placing other tiles."""

        def fake(cfg, tile, *a, **k):
            if tile == "S35W055":
                raise launch.CapacityExhausted(tile, launch.CAPACITY_ERROR, "full")
            return {"tile": tile}

        monkeypatch.setattr(waves, "launch_one", fake)
        out = waves.launch_many(
            ["S30W065", "S35W055", "S30W060"],
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=1,
            say=lambda *a: None,
        )
        assert out.quota is False
        assert out.refused == ["S35W055"]
        assert sorted(e["tile"] for e in out.placed) == ["S30W060", "S30W065"]

    def test_an_unrecoverable_error_still_stops_the_run(
        self, cfg, tmp_path, monkeypatch
    ):
        """A wrong security group is a bug, not something a retry fixes."""

        def fake(cfg, tile, *a, **k):
            raise SystemExit("aws ec2 run-instances failed")

        monkeypatch.setattr(waves, "launch_one", fake)
        with pytest.raises(SystemExit):
            waves.launch_many(
                ["S30W065"],
                cfg=cfg,
                run_id="x",
                user_data=Path("ud.sh"),
                manifest=a_manifest(tmp_path, cfg),
                workers=2,
                say=lambda *a: None,
            )


class TestTheManifestSurvivesConcurrentWriters:
    """Every thread writes the same file through one fixed temporary path.

    An entry lost to that race is a running instance nobody can terminate.
    """

    def test_no_entry_is_lost(self, tmp_path, cfg):
        from concurrent.futures import ThreadPoolExecutor

        manifest = a_manifest(tmp_path, cfg)
        tiles = [f"N{n:02d}E010" for n in range(0, 50)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda t: manifest.add({"tile": t}), tiles))
        assert len(manifest.instances) == len(tiles)
        on_disk = json.loads((tmp_path / "run.json").read_text())
        assert len(on_disk["instances"]) == len(tiles)
        assert sorted(e["tile"] for e in on_disk["instances"]) == sorted(tiles)


class TestTheGateAnswersRefusalsDifferently:
    """Quota is an account fact. Capacity is one type in four zones, for now."""

    def test_an_empty_pool_may_ask_for_the_full_width(self):
        assert waves.Gate(80).room(0, 0.0) == 80

    def test_a_full_pool_asks_for_nothing(self):
        assert waves.Gate(80).room(80, 0.0) == 0

    def test_a_quota_refusal_caps_the_next_ask(self):
        gate = waves.Gate(80)
        out = waves.Placement()
        out.quota = True
        out.refused = ["S30W065"]
        gate.observe(out, 34, 0.0, say=lambda *a: None)
        assert gate.room(34, 1.0) == 0
        assert gate.room(33, 1.0) == 1

    def test_the_cap_expires_so_a_busy_hour_does_not_set_the_day(self):
        """The account may have been busy with somebody else's work."""
        gate = waves.Gate(80)
        out = waves.Placement()
        out.quota = True
        out.refused = ["S30W065"]
        gate.observe(out, 34, 0.0, say=lambda *a: None)
        later = waves.CAPACITY_BACKOFF_SECONDS + 1
        assert gate.room(34, later) == 80 - 34

    def test_a_full_region_is_not_asked_again_at_once(self):
        gate = waves.Gate(80)
        out = waves.Placement()
        out.refused = ["S30W065"]
        gate.observe(out, 0, 0.0, say=lambda *a: None)
        assert gate.room(0, 1.0) == 0
        assert gate.room(0, waves.CAPACITY_BACKOFF_SECONDS + 1) == 80

    def test_a_partly_filled_ask_is_not_a_refusal(self):
        """Some placed means the region has room. Do not back off."""
        gate = waves.Gate(80)
        out = waves.Placement()
        out.placed = [{"tile": "S30W065"}]
        out.refused = ["S35W055"]
        gate.observe(out, 1, 0.0, say=lambda *a: None)
        assert gate.room(1, 1.0) == 79


class TestThePoolFillsAndRetires:
    """The wave barrier cost about 20 machine-minutes a tile at width 20."""

    def a_pool(self, tmp_path, tiles):
        return waves.Pool(
            tiles,
            fleet_dir=Path("fleet"),
            manifest_path=tmp_path / "run.json",
            log_dir=tmp_path / "logs",
        )

    def fill(self, pool, cfg, tmp_path, monkeypatch, count):
        monkeypatch.setattr(
            waves, "launch_one", lambda cfg, tile, *a, **k: {"tile": tile}
        )
        return pool.fill(
            count,
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=2,
            drive_fn=lambda *a: FakeProc(alive=True),
            say=lambda *a: None,
        )

    def test_filling_takes_from_the_queue_and_drives_each_tile(
        self, cfg, tmp_path, monkeypatch
    ):
        pool = self.a_pool(tmp_path, ["S30W065", "S30W060", "S35W055"])
        self.fill(pool, cfg, tmp_path, monkeypatch, 2)
        assert len(pool.live) == 2
        assert len(pool.drivers) == 2
        assert pool.queue == ["S35W055"]
        assert pool.peak == 2

    def test_a_refused_tile_keeps_its_place_at_the_head(
        self, cfg, tmp_path, monkeypatch
    ):
        def fake(cfg, tile, *a, **k):
            if tile == "S30W065":
                raise launch.CapacityExhausted(tile, launch.CAPACITY_ERROR, "full")
            return {"tile": tile}

        monkeypatch.setattr(waves, "launch_one", fake)
        pool = self.a_pool(tmp_path, ["S30W065", "S30W060", "S35W055"])
        pool.fill(
            2,
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=1,
            drive_fn=lambda *a: FakeProc(alive=True),
            say=lambda *a: None,
        )
        assert pool.queue == ["S30W065", "S35W055"]

    def test_polling_asks_about_the_live_instances_only(
        self, cfg, tmp_path, monkeypatch
    ):
        """Polling the whole run would grow with the tiles already finished."""
        pool = self.a_pool(tmp_path, [f"N{n:02d}E010" for n in range(0, 40, 5)])
        self.fill(pool, cfg, tmp_path, monkeypatch, 3)
        assert len(pool.view(cfg)["instances"]) == 3

    def test_an_uploaded_tile_leaves_the_pool_and_frees_its_slot(
        self, cfg, tmp_path, monkeypatch
    ):
        pool = self.a_pool(tmp_path, ["S30W065", "S30W060"])
        self.fill(pool, cfg, tmp_path, monkeypatch, 2)
        states = [
            FakeState("S30W065", "finished", waves.UPLOADED_PHASE),
            FakeState("S30W060", "running", "compositing"),
        ]
        pool.retire(
            states,
            {"S30W065": "finished", "S30W060": "running"},
            pool.started["S30W065"] + 1,
            timeout_minutes=85.0,
            retries=1,
            say=lambda *a: None,
        )
        assert pool.settled == {"S30W065": "finished"}
        assert list(pool.live) == ["S30W060"]
        assert pool.queue == []

    def test_a_tile_at_the_marker_without_its_manifest_stays(
        self, cfg, tmp_path, monkeypatch
    ):
        """MEASURED: N00W045 was torn down at `all_done` and lost its manifest."""
        pool = self.a_pool(tmp_path, ["S30W065"])
        self.fill(pool, cfg, tmp_path, monkeypatch, 1)
        states = [FakeState("S30W065", "finished", "all_done", "waiting on upload")]
        pool.retire(
            states,
            {"S30W065": "finished"},
            pool.started["S30W065"] + 1,
            timeout_minutes=85.0,
            retries=1,
            say=lambda *a: None,
        )
        assert list(pool.live) == ["S30W065"]
        assert pool.settled == {}

    def test_a_failed_tile_goes_back_to_the_queue_once(
        self, cfg, tmp_path, monkeypatch
    ):
        pool = self.a_pool(tmp_path, ["S30W065"])
        self.fill(pool, cfg, tmp_path, monkeypatch, 1)
        for _ in range(2):
            states = [FakeState("S30W065", "failed", "compositing")]
            pool.retire(
                states,
                {"S30W065": "failed"},
                pool.started["S30W065"] + 1,
                timeout_minutes=85.0,
                retries=1,
                say=lambda *a: None,
            )
            if pool.queue:
                self.fill(pool, cfg, tmp_path, monkeypatch, 1)
        assert pool.queue == []
        assert pool.settled == {"S30W065": "failed"}
        assert pool.attempts["S30W065"] == 2

    def test_a_tile_past_its_own_deadline_is_a_timeout(
        self, cfg, tmp_path, monkeypatch
    ):
        """Each tile carries its own deadline. A pool has no wave to time out."""
        pool = self.a_pool(tmp_path, ["S30W065"])
        self.fill(pool, cfg, tmp_path, monkeypatch, 1)
        states = [FakeState("S30W065", "running", "compositing")]
        pool.retire(
            states,
            {"S30W065": "running"},
            pool.started["S30W065"] + 86 * 60,
            timeout_minutes=85.0,
            retries=0,
            say=lambda *a: None,
        )
        assert pool.settled == {"S30W065": "timeout"}
        assert pool.live == {}

    def test_retiring_terminates_a_driver_that_is_still_polling(
        self, cfg, tmp_path, monkeypatch
    ):
        pool = self.a_pool(tmp_path, ["S30W065"])
        self.fill(pool, cfg, tmp_path, monkeypatch, 1)
        proc = pool.drivers["S30W065"]
        pool.retire(
            [FakeState("S30W065", "finished", waves.UPLOADED_PHASE)],
            {"S30W065": "finished"},
            pool.started["S30W065"] + 1,
            timeout_minutes=85.0,
            retries=1,
            say=lambda *a: None,
        )
        assert proc.terminated is True


class TestARunPricesItselfWhileItRuns:
    """AWS forgets a terminated instance after about an hour.

    A wave run tore down every 40 minutes and priced itself inside that
    window. A five-hour refill run would reach its report with the first four
    hours already unpriceable.
    """

    def snapshots(self, tmp_path, *blocks):
        path = tmp_path / "snap.txt"
        path.write_text("\n".join(blocks))
        return path

    BLOCK = (
        "=== snapshot {stamp} ===\n"
        "instance             type           AMI          launch    term      sec\n"
        "{rows}\n"
        "=== DERIVED: EC2 ===\n"
        "  m6id.16xlarge      {total}s / 3600 x $3.7968  = $ 1.00\n"
    )

    def row(self, iid, sec):
        return f"{iid}  m6id.16xlarge  ami-04678417  18:14:20  18:53:28  {sec:8d}"

    def test_an_instance_in_two_snapshots_is_counted_once(self, tmp_path):
        """MEASURED: summing six wave reports gave $346 against a true $195."""
        first = self.BLOCK.format(
            stamp="a", rows=self.row("i-0000000000000000a", 3600), total=3600
        )
        second = self.BLOCK.format(
            stamp="b",
            rows=self.row("i-0000000000000000a", 3600)
            + "\n"
            + self.row("i-0000000000000000b", 1800),
            total=5400,
        )
        count, seconds, dollars = waves.total_from_snapshots(
            self.snapshots(tmp_path, first, second)
        )
        assert count == 2
        assert seconds == 5400
        assert dollars == pytest.approx(1.5 * 3.7968, rel=1e-6)

    def test_a_growing_instance_is_priced_at_its_final_seconds(self, tmp_path):
        """Instance seconds only grow, so the largest reading is the last."""
        first = self.BLOCK.format(
            stamp="a", rows=self.row("i-0000000000000000a", 600), total=600
        )
        second = self.BLOCK.format(
            stamp="b", rows=self.row("i-0000000000000000a", 3600), total=3600
        )
        count, seconds, _ = waves.total_from_snapshots(
            self.snapshots(tmp_path, first, second)
        )
        assert count == 1
        assert seconds == 3600

    def test_an_empty_file_prices_at_zero(self, tmp_path):
        path = tmp_path / "snap.txt"
        path.write_text("")
        assert waves.total_from_snapshots(path) == (0, 0, 0.0)


class TestTheProjectionIsHonest:
    """A dry run that understates the bill is worse than no dry run."""

    def test_the_fixed_cost_is_carried_per_tile(self):
        scenes = {"N00E005": 0, "N00E010": 0}
        hours, dollars, _ = waves.project(list(scenes), scenes, 1)
        assert hours == pytest.approx(2 * waves.FIXED_SECONDS / 3600)
        assert dollars == pytest.approx(hours * waves.HOURLY_USD)

    def test_scenes_add_to_the_cost(self):
        few = waves.project(["N00E005"], {"N00E005": 1000}, 1)[0]
        many = waves.project(["N00E005"], {"N00E005": 5000}, 1)[0]
        assert many > few
        assert (many - few) * 3600 == pytest.approx(4000 * waves.SECONDS_PER_SCENE)

    def test_width_shortens_the_wall_clock_but_not_the_bill(self):
        tiles = [f"N{n:02d}E010" for n in range(0, 60, 5)]
        scenes = dict.fromkeys(tiles, 4000)
        narrow = waves.project(tiles, scenes, 10)
        wide = waves.project(tiles, scenes, 40)
        assert narrow[1] == pytest.approx(wide[1])
        assert wide[2] < narrow[2]

    def test_a_tile_with_no_scene_count_is_not_free(self):
        """An unknown tile priced at zero would understate the run."""
        scenes = {"N00E005": 4000}
        one = waves.project(["N00E005"], scenes, 1)[1]
        two = waves.project(["N00E005", "N00E010"], scenes, 1)[1]
        assert two == pytest.approx(2 * one)

    def test_the_model_matches_the_measured_south_america_run(self):
        """MEASURED: 99 tiles, 348,834 scenes in the continent, about $195."""
        tiles = [f"T{n:03d}" for n in range(99)]
        scenes = dict.fromkeys(tiles, 3354)
        _, dollars, _ = waves.project(tiles, scenes, 20)
        assert 175 < dollars < 215


class TestAnUnpushedCommitNeverReachesAnInstance:
    """MEASURED on 2026-09-16: 60 instances launched at an unpushed commit.

    Every one wrote `MARKER checkout rc=128` about four minutes into billing,
    having done nothing. The mistake cost about $42 before the run was stopped.
    At 665 tiles it would have run the whole queue twice on `git checkout`.
    """

    def a_repo(self, tmp_path, monkeypatch, on_remote):
        calls = []

        def fake(argv, **kwargs):
            calls.append(argv)
            import subprocess as sp

            if argv[3:5] == ["branch", "-r"]:
                out = "  origin/main\n" if on_remote else ""
                return sp.CompletedProcess(argv, 0, out, "")
            return sp.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(launch.subprocess, "run", fake)
        return calls

    def test_a_commit_on_a_remote_branch_is_accepted(self, tmp_path, monkeypatch):
        self.a_repo(tmp_path, monkeypatch, on_remote=True)
        assert launch.resolve_commit(SHA, repo=tmp_path) == SHA

    def test_a_commit_on_no_remote_branch_is_refused(self, tmp_path, monkeypatch):
        self.a_repo(tmp_path, monkeypatch, on_remote=False)
        with pytest.raises(SystemExit, match="not on any remote branch"):
            launch.resolve_commit(SHA, repo=tmp_path)

    def test_the_refusal_says_to_push(self, tmp_path, monkeypatch):
        self.a_repo(tmp_path, monkeypatch, on_remote=False)
        with pytest.raises(SystemExit) as err:
            launch.resolve_commit(SHA, repo=tmp_path)
        assert "Push it first" in str(err.value)

    def test_it_fetches_once_before_answering_no(self, tmp_path, monkeypatch):
        """The local copy of the remote refs may be older than the push."""
        calls = self.a_repo(tmp_path, monkeypatch, on_remote=False)
        with pytest.raises(SystemExit):
            launch.resolve_commit(SHA, repo=tmp_path)
        assert ["fetch", "--quiet"] == calls[1][3:5]
        assert sum(1 for c in calls if c[3] == "fetch") == 1

    def test_no_repo_means_no_check(self, tmp_path, monkeypatch):
        """`--commit` alone still resolves, for callers with no working copy."""
        assert launch.resolve_commit(SHA) == SHA


class TestAFailedInstanceStopsBilling:
    """`reap` leaves a failed instance alone so its log can finish uploading.

    Under waves the wave teardown collected it minutes later. A refill run has
    no wave teardown. MEASURED on 2026-09-16: 60 failed tiles left the pool and
    kept both their billing and their share of the account vCPU.
    """

    def a_pool(self, tmp_path):
        return waves.Pool(
            ["S30W065"],
            fleet_dir=Path("fleet"),
            manifest_path=tmp_path / "run.json",
            log_dir=tmp_path / "logs",
        )

    def failed_pool(self, cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(
            waves,
            "launch_one",
            lambda cfg, tile, *a, **k: {"tile": tile, "instance_id": "i-0abc"},
        )
        pool = self.a_pool(tmp_path)
        pool.fill(
            1,
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=1,
            drive_fn=lambda *a: FakeProc(alive=True),
            say=lambda *a: None,
        )
        pool.retire(
            [FakeState("S30W065", "failed", "compositing")],
            {"S30W065": "failed"},
            pool.started["S30W065"] + 1,
            timeout_minutes=85.0,
            retries=0,
            say=lambda *a: None,
        )
        return pool

    def test_a_failed_instance_is_condemned(self, cfg, tmp_path, monkeypatch):
        pool = self.failed_pool(cfg, tmp_path, monkeypatch)
        assert "i-0abc" in pool.condemned
        assert pool.condemned["i-0abc"][0] == "S30W065"

    def test_it_is_not_terminated_inside_the_grace(self, cfg, tmp_path, monkeypatch):
        """Its uploader may still be pushing the log that says why it failed."""
        stopped = []
        pool = self.failed_pool(cfg, tmp_path, monkeypatch)
        monkeypatch.setattr(
            waves, "terminate_one", lambda cfg, i: (stopped.append(i), (0, ""))[1]
        )
        assert pool.sweep(cfg, 0.0, say=lambda *a: None) == 0
        assert stopped == []

    def test_it_is_terminated_once_the_grace_expires(self, cfg, tmp_path, monkeypatch):
        stopped = []
        pool = self.failed_pool(cfg, tmp_path, monkeypatch)
        monkeypatch.setattr(
            waves, "terminate_one", lambda cfg, i: (stopped.append(i), (0, ""))[1]
        )
        due = pool.condemned["i-0abc"][1] + 1
        assert pool.sweep(cfg, due, say=lambda *a: None) == 1
        assert stopped == ["i-0abc"]
        assert pool.condemned == {}

    def test_a_refused_termination_is_tried_again(self, cfg, tmp_path, monkeypatch):
        """An instance left condemned is an instance still billing."""
        pool = self.failed_pool(cfg, tmp_path, monkeypatch)
        monkeypatch.setattr(waves, "terminate_one", lambda cfg, i: (255, "denied"))
        due = pool.condemned["i-0abc"][1] + 1
        assert pool.sweep(cfg, due, say=lambda *a: None) == 0
        assert "i-0abc" in pool.condemned

    def test_an_uploaded_tile_is_not_condemned(self, cfg, tmp_path, monkeypatch):
        """`reap` already terminated it. Condemning it would ask twice."""
        monkeypatch.setattr(
            waves,
            "launch_one",
            lambda cfg, tile, *a, **k: {"tile": tile, "instance_id": "i-0abc"},
        )
        pool = self.a_pool(tmp_path)
        pool.fill(
            1,
            cfg=cfg,
            run_id="x",
            user_data=Path("ud.sh"),
            manifest=a_manifest(tmp_path, cfg),
            workers=1,
            drive_fn=lambda *a: FakeProc(alive=True),
            say=lambda *a: None,
        )
        pool.retire(
            [FakeState("S30W065", "finished", waves.UPLOADED_PHASE)],
            {"S30W065": "finished"},
            pool.started["S30W065"] + 1,
            timeout_minutes=85.0,
            retries=0,
            say=lambda *a: None,
        )
        assert pool.condemned == {}


class TestACeilingOfZeroIsNotACeiling:
    """A ceiling of zero says never launch again.

    MEASURED on 2026-09-16: 60 instances failed at once, left the pool, and
    kept their quota. The gate read the next refusal against a live count of
    zero and wrote `account ceiling met at 0 concurrent instance(s)`.
    """

    def test_a_refusal_with_nothing_live_sets_no_ceiling(self):
        gate = waves.Gate(80)
        out = waves.Placement()
        out.quota = True
        out.refused = ["S30W065"]
        gate.observe(out, 0, 0.0, say=lambda *a: None)
        assert gate.ceiling == 0
        assert gate.room(0, waves.CAPACITY_BACKOFF_SECONDS + 1) == 80

    def test_it_backs_off_rather_than_asking_every_poll(self):
        gate = waves.Gate(80)
        out = waves.Placement()
        out.quota = True
        out.refused = ["S30W065"]
        gate.observe(out, 0, 0.0, say=lambda *a: None)
        assert gate.room(0, 1.0) == 0

    def test_a_refusal_with_instances_live_still_sets_the_ceiling(self):
        gate = waves.Gate(80)
        out = waves.Placement()
        out.quota = True
        out.refused = ["S30W065"]
        gate.observe(out, 60, 0.0, say=lambda *a: None)
        assert gate.ceiling == 60
        assert gate.room(60, 1.0) == 0
        assert gate.room(59, 1.0) == 1

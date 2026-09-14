"""Guards on the launch mechanism, one per failure that has already cost money.

The mechanism itself worked before this file existed. What it lacked was any
way to notice when one of its rules had been quietly broken, and each rule here
was written after the rule was broken in production.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))

import launch  # noqa: E402
import upload  # noqa: E402
import watch  # noqa: E402

SHA = "9e2b703abec9756945686d5e8037788a666d28e6"


@pytest.fixture(scope="module")
def cfg():
    return launch.load_config()


class TestTheCommitIsPinned:
    """A published tile has to name the code that built it.

    The previous mechanism took a branch name and ran `git pull --ff-only` on
    the instance, so the commit depended on when the box woke up.
    """

    def test_a_full_sha_is_accepted(self):
        assert launch.resolve_commit(SHA) == SHA

    def test_a_branch_name_is_refused(self):
        with pytest.raises(SystemExit) as err:
            launch.resolve_commit("fleet-sizing")
        assert "40-character SHA" in str(err.value)

    def test_a_short_sha_is_refused(self):
        """An abbreviation is ambiguous across a growing history."""
        with pytest.raises(SystemExit):
            launch.resolve_commit(SHA[:12])

    def test_the_refusal_offers_the_sha_it_resolves_to(self):
        with pytest.raises(SystemExit) as err:
            launch.resolve_commit("HEAD", repo=ROOT)
        assert "which you can pass instead" in str(err.value)


class TestTheKeyOutlivesAReboot:
    """A key under a session scratchpad was cleared by a workstation restart
    while its instance kept running. This role cannot call
    `ec2-instance-connect`, `ssm`, or the serial console, so the run was lost.
    """

    def test_it_lands_under_the_configured_home(self, cfg):
        path = launch.key_path(cfg["paths"]["key_dir"], "lst-N40W080-x")
        assert path == Path.home() / ".ssh" / "lst-N40W080-x.pem"

    @pytest.mark.parametrize("bad", ["/tmp", "/tmp/keys", "/var/tmp", "/dev/shm"])
    def test_a_volatile_directory_is_refused(self, bad):
        with pytest.raises(SystemExit) as err:
            launch.key_path(bad, "k")
        assert "cleared on reboot" in str(err.value)

    def test_a_relative_directory_is_refused(self, bad="keys"):
        with pytest.raises(SystemExit):
            launch.key_path(bad, "k")


class TestTheCallCarriesWhatTeardownNeeds:
    def argv(self, cfg):
        return launch.run_instances_argv(cfg, "lst-T-1", "T", Path("/u.sh"))

    def test_every_identifier_comes_from_the_config(self, cfg):
        argv = self.argv(cfg)
        for value in (
            cfg["instance"]["ami"],
            cfg["instance"]["type"],
            cfg["instance"]["subnet"],
            cfg["instance"]["security_group"],
            cfg["aws"]["region"],
            cfg["aws"]["profile"],
        ):
            assert value in argv, value

    def test_the_purpose_tag_is_the_one_cost_report_filters_on(self, cfg):
        """`cost_report.py --tag Key=Value` is the only way a finished run gets
        priced, and an untagged instance cannot be found by teardown either."""
        spec = launch.tag_spec(cfg, "lst-T-1", "T")
        assert f"{{Key=purpose,Value={cfg['tags']['purpose']}}}" in spec
        assert "{Key=tile,Value=T}" in spec

    def test_the_instance_terminates_when_it_halts(self, cfg):
        argv = self.argv(cfg)
        i = argv.index("--instance-initiated-shutdown-behavior")
        assert argv[i + 1] == "terminate"

    def test_the_deadline_reaches_the_user_data(self, cfg, tmp_path):
        """The only bound on what a hung run can cost. It lives in the config
        because a number buried in a shell script is one nobody revises."""
        rendered = launch.render_user_data(cfg, tmp_path).read_text()
        assert f"shutdown -h +{cfg['instance']['deadline_minutes']}" in rendered
        assert "__DEADLINE__" not in rendered


class TestTheUploaderShipsResultsNotScratch:
    """Every run used to upload `qa_count.staging.tif` at 4.08 GB and
    `lst_p95.staging.tif` at 0.68 GB. `composite.cleanup_staging` deletes both
    on the success path, so the bucket was storing an intermediate as a
    product: about 85% of each run's uploaded bytes.
    """

    @pytest.mark.parametrize(
        "rel",
        [
            "tile/catalog/lst-p95-2021-2025/N40W080/lst_p95.tif",
            "tile/summary.json",
            "markers.txt",
            "commit.txt",
            "prep/tile-prep.npz",
        ],
    )
    def test_a_result_is_uploaded(self, rel):
        assert upload.wanted(rel)

    @pytest.mark.parametrize(
        "rel",
        [
            "tile/qa_count.staging.tif",
            "tile/lst_p95.staging.tif",
            "tile/lst_p95.staging.tif.lock",
            "stage/LC08_L2SP_x_ST_B10.TIF",
        ],
    )
    def test_scratch_is_not(self, rel):
        assert not upload.wanted(rel)

    def test_it_walks_a_real_tree(self, tmp_path):
        (tmp_path / "tile").mkdir()
        (tmp_path / "tile" / "summary.json").write_text("{}")
        (tmp_path / "tile" / "qa_count.staging.tif").write_bytes(b"x" * 10)
        assert set(upload.scan(tmp_path)) == {"tile/summary.json"}


class TestTheWatcherNamesTheState:
    """`get-console-output` was polled eight times and returned nothing while
    three of four instances had already failed, which is eight minutes of
    four-instance billing spent watching an empty channel.
    """

    NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)

    def marker(self, phase, rc, minutes_ago):
        when = self.NOW - timedelta(minutes=minutes_ago)
        return f"MARKER {phase} rc={rc} {when.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    def test_advancing_markers_are_running(self):
        s = watch.classify("T", self.marker("prep_start", 0, 2), False, True, self.NOW)
        assert s.status == "running"

    def test_a_stale_marker_is_hung(self):
        s = watch.classify("T", self.marker("prep_start", 0, 30), False, True, self.NOW)
        assert s.status == "hung"

    def test_a_nonzero_rc_is_failed(self):
        s = watch.classify(
            "T", self.marker("composite_done", 1, 1), False, True, self.NOW
        )
        assert (s.status, s.detail) == ("failed", "rc=1")

    def test_an_ended_instance_before_all_done_is_gone(self):
        """One instance of a four-instance fleet sat stranded while its
        siblings self-terminated. Nothing reported it."""
        s = watch.classify("T", self.marker("prep_start", 0, 1), False, False, self.NOW)
        assert s.status == "gone"

    def test_the_manifest_flag_is_the_only_proof_of_a_finished_upload(self):
        s = watch.classify("T", None, True, False, self.NOW)
        assert s.status == "finished"

    def test_a_marker_line_the_pipeline_never_wrote_is_ignored(self):
        """`MARKER sweep_done rc=$?` after a pipe captured `tee`'s status and
        always read 0. Malformed lines must not become phantom phases."""
        assert watch.parse_markers("MARKER broken\nnoise\n") == []


class TestTeardownPricesBeforeItTerminates:
    def test_the_terminate_call_names_every_instance(self, cfg):
        import teardown

        argv = teardown.terminate_argv(cfg, ["i-1", "i-2"])
        assert argv[:3] == ["aws", "ec2", "terminate-instances"]
        assert argv[-2:] == ["i-1", "i-2"]


class TestTheConfigCarriesTheKnowledge:
    """Until 2026-09-14 these identifiers lived in one laptop's shell history.
    A later session searched the repository, found nothing, and nearly rebuilt
    a launcher that already existed.
    """

    def test_every_section_a_launch_reads_is_present(self, cfg):
        for section in (
            "aws",
            "instance",
            "tags",
            "paths",
            "storage",
            "artifacts",
            "upload",
        ):
            assert section in cfg

    def test_the_six_artifacts_travel_together(self, cfg):
        """`check_mask_inputs` and `check_manifest` refuse a run whose
        artifacts disagree, so a partial set is worse than none."""
        assert set(cfg["artifacts"]["files"]) == {
            "tile_scene_inventory.parquet",
            "land_tiles.parquet",
            "aster_numobs.tif",
            "aster_numobs_manifest.json",
            "land_buffered.gpkg",
            "land_buffered_sha256.txt",
        }

    def test_the_deadline_clears_the_measured_wall_clock(self, cfg):
        """MEASURED tile wall clock is 26 to 43 minutes. Too tight kills a slow
        tile; the previous 120 let a finished box idle for over an hour."""
        assert 50 <= cfg["instance"]["deadline_minutes"] <= 90

    def test_the_staging_directory_is_not_the_root_volume(self, cfg):
        """A tile stages 222 to 382 GiB against a 150 GB root."""
        assert cfg["paths"]["stage_dir"].startswith("/mnt/")


class TestTheRunScriptKeepsTheLoadBearingFlags:
    """Four flags whose defaults are wrong for an instance. Losing any one
    produces a finished run that is slow, refused, or quietly incorrect.
    """

    @pytest.fixture(scope="class")
    def script(self):
        return (ROOT / "fleet" / "run.sh").read_text()

    @pytest.mark.parametrize(
        "flag,why",
        [
            ("--engine fused", "not the default; changes speed and the memory model"),
            ("--stage-dir /mnt/nvme/stage", "the default is the 150 GB root volume"),
            ("--keep-staged", "without it the two passes pay for every object twice"),
            ("--tile-prep", "omitted, it composites pooled and leaves the WRS seam"),
        ],
    )
    def test_the_flag_survives(self, script, flag, why):
        assert flag in script, why

    def test_it_pins_a_commit_rather_than_following_a_branch(self, script):
        assert "git checkout --detach" in script
        assert "git pull --ff-only" not in script

    def test_it_verifies_the_checkout_landed_on_that_commit(self, script):
        assert 'test "$(git rev-parse HEAD)" = "$COMMIT"' in script

    def test_a_marker_never_takes_its_status_through_a_pipe(self, script):
        """`MARKER sweep_done rc=$?` followed a `| tee` and always read 0. A
        live markers.txt in the bucket says rc=0 and proves nothing."""
        for line in script.splitlines():
            if "mark " in line and "rc" in line:
                assert "|" not in line.split("mark")[0]


class TestTheOneOffAssertionIsGone:
    def test_no_hardcoded_grep_for_one_days_fix(self):
        """`git grep -n "cut: bool = False"` was an assertion for a single
        2026-09-13 change, hardcoded into the general launcher."""
        assert "cut: bool = False" not in (ROOT / "fleet" / "run.sh").read_text()

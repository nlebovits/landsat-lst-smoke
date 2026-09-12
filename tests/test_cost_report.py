"""The cost arithmetic was wrong twice, and both errors reached the write-up.

First the full tile read $10.70, from assuming each instance ran an hour when
it ran 642 s. Then an invented requests-per-read multiplier produced $0.09 per
tile and a $650 global figure, and that guess was used to reverse an
optimisation recommendation.

These tests run the real CLI and assert the numbers FINDINGS.md publishes, so
a change to the arithmetic has to change the document too.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COST_REPORT = ROOT / "cost_report.py"

# The full-tile fleet, exactly as FINDINGS.md records it.
FLEET = "c6i.16xlarge:4:642"
SHARD_SCENE_READS = 605_617
REQUESTS_PER_READ = 4.77


def run_report(*extra, tmp_path):
    """Drive the CLI and return its JSON, plus stdout."""
    out = tmp_path / "cost.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(COST_REPORT),
            "--tag",
            "purpose=test",
            "--region",
            "us-west-2",
            "--ebs-gb",
            "150",
            "--recorded",
            FLEET,
            "--json",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text()), proc.stdout


@pytest.fixture
def priced(tmp_path):
    return run_report(
        "--shard-scene-reads",
        str(SHARD_SCENE_READS),
        "--requests-per-read",
        str(REQUESTS_PER_READ),
        tmp_path=tmp_path,
    )


class TestFullTileCost:
    def test_instance_seconds_come_from_the_recorded_lifetime(self, priced):
        report, _ = priced
        assert report["measured"]["total_instance_seconds"] == 2568.0

    def test_ec2_line(self, priced):
        """2,568 s / 3600 x $2.72 = $1.94, and never a whole-hour assumption."""
        report, _ = priced
        assert report["derived"]["ec2_usd"] == pytest.approx(1.94, abs=0.005)

    def test_s3_line(self, priced):
        """605,617 reads x 2 bands x 4.77 = 5,777,586 GETs, priced at $2.31."""
        report, _ = priced
        assert report["derived"]["s3_usd"] == pytest.approx(2.31, abs=0.005)

    def test_ebs_and_ipv4_stay_under_a_cent_and_a_half(self, priced):
        report, _ = priced
        assert report["derived"]["ebs_usd"] == pytest.approx(0.012, abs=0.001)
        assert report["derived"]["ipv4_usd"] == pytest.approx(0.0036, abs=0.0005)

    def test_the_address_line_counts_each_instance_once(self, priced):
        """One address per instance, and the count is already in the seconds.

        `total_instance_seconds` sums every instance's lifetime, so
        multiplying by the instance count again charges the fleet once per
        machine squared. At four machines that read $0.014 against $0.0036 and
        nobody noticed. Priced across 3,076 machines it reads $8,437 against
        $2.74, which is most of a fleet estimate.
        """
        report, _ = priced
        hours = report["measured"]["total_instance_seconds"] / 3600
        rate = report["rates"]["ipv4_hr"]
        assert report["derived"]["ipv4_usd"] == pytest.approx(hours * rate)

    def test_the_total_is_the_published_figure(self, priced):
        report, _ = priced
        d = report["derived"]
        total = d["ec2_usd"] + d["s3_usd"] + d["ebs_usd"] + d["ipv4_usd"]
        assert total == pytest.approx(4.27, abs=0.005)

    def test_s3_exceeds_ec2(self, priced):
        """FINDINGS.md rests a recommendation on this ordering."""
        report, _ = priced
        assert report["derived"]["s3_usd"] > report["derived"]["ec2_usd"]

    def test_a_whole_hour_assumption_would_be_caught(self, priced):
        """The $10.70 error, restated as a test.

        Four c6i.16xlarge for one hour is 4 x $2.72 = $10.88. The fleet lived
        642 s, so nothing here may come near that.
        """
        report, _ = priced
        assert report["derived"]["ec2_usd"] < 2.0


class TestS3RemainsUnknownWithoutAMeasurement:
    """The script refuses to price S3 from a guess. That refusal is the fix
    for the second error, so it needs a test of its own.
    """

    def test_omitting_requests_per_read_reports_s3_as_null(self, tmp_path):
        """The key stays, carrying null. A consumer that reads the report
        cannot mistake a missing S3 line for a zero one.
        """
        report, stdout = run_report(
            "--shard-scene-reads", str(SHARD_SCENE_READS), tmp_path=tmp_path
        )
        assert report["derived"]["s3_usd"] is None
        assert "UNKNOWN" in stdout

    def test_the_report_says_lower_bound_rather_than_total(self, tmp_path):
        _, stdout = run_report(
            "--shard-scene-reads", str(SHARD_SCENE_READS), tmp_path=tmp_path
        )
        assert "lower bound" in stdout.lower()


class TestACountedRunPricesItselfDirectly:
    """A staged run fetches each object once and counts every attempt.

    So its S3 line is the wire total, not `reads x bands x a sampled rate`.
    The derivation exists because the unstaged path cannot count itself; a run
    that can must not be pushed back through it.
    """

    #: Three scenes, two bands, one GET each, as `staging.json` records it.
    STAGED_GETS = 7_820

    def test_a_counted_total_prices_s3_without_a_sample(self, tmp_path):
        report, stdout = run_report(
            "--s3-get-requests", str(self.STAGED_GETS), tmp_path=tmp_path
        )
        assert report["measured"]["s3_get_requests"] == self.STAGED_GETS
        assert report["derived"]["s3_usd"] == pytest.approx(
            self.STAGED_GETS / 1000 * 0.0004
        )
        assert "counted on the wire" in stdout
        assert report["unknown"] == []

    def test_it_needs_neither_reads_nor_a_rate(self, tmp_path):
        # The point of staging is that neither figure exists any more. If the
        # refusal still fired here, a staged run could not be priced at all.
        _, stdout = run_report(
            "--s3-get-requests", str(self.STAGED_GETS), tmp_path=tmp_path
        )
        assert "UNKNOWN" not in stdout
        assert "lower bound" not in stdout.lower()

    def test_a_counted_total_wins_over_the_derivation(self, tmp_path):
        report, _ = run_report(
            "--s3-get-requests",
            str(self.STAGED_GETS),
            "--shard-scene-reads",
            str(SHARD_SCENE_READS),
            "--requests-per-read",
            str(REQUESTS_PER_READ),
            tmp_path=tmp_path,
        )
        assert report["derived"]["s3_usd"] == pytest.approx(
            self.STAGED_GETS / 1000 * 0.0004
        )

    def test_staging_moves_the_s3_line_below_the_compute(self, tmp_path):
        """The headline of the change, as arithmetic the report agrees with."""
        staged, _ = run_report(
            "--s3-get-requests", str(self.STAGED_GETS), tmp_path=tmp_path
        )
        unstaged, _ = run_report(
            "--shard-scene-reads",
            str(SHARD_SCENE_READS),
            "--requests-per-read",
            str(REQUESTS_PER_READ),
            tmp_path=tmp_path,
        )
        assert unstaged["derived"]["s3_usd"] > unstaged["derived"]["ec2_usd"]
        assert staged["derived"]["s3_usd"] < unstaged["derived"]["ec2_usd"] / 100


class TestRecordedParsing:
    def test_a_malformed_fleet_spec_is_rejected(self, tmp_path):
        proc = subprocess.run(
            [
                sys.executable,
                str(COST_REPORT),
                "--tag",
                "purpose=test",
                "--recorded",
                "c6i.16xlarge:4",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode != 0
        assert "TYPE:COUNT:SECONDS" in proc.stderr + proc.stdout

    def test_two_fleets_add_up(self, tmp_path):
        report, _ = run_report("--recorded", "m6i.4xlarge:1:100", tmp_path=tmp_path)
        assert report["measured"]["total_instance_seconds"] == 2568.0 + 100.0

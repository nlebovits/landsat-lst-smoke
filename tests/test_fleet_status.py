"""The dashboard, and the one thing it exists to shout about.

A driver that dies leaves its instances billing with nobody watching, and the
only sign is a log that stops moving. Twice on 2026-09-16 that is exactly what
happened. A dashboard that reports a cheerful 58 live from a log written an
hour ago is worse than no dashboard.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lst.fleet import status

DAY = datetime(2026, 9, 16, tzinfo=timezone.utc)
LINE = "08:09:55Z  live=58 queued=603 done=12/661 peak=58 condemned=2"


class TestTheStatusLineIsRead:
    def test_every_field_lands(self):
        shot = status.parse_status(LINE, DAY)
        assert shot is not None
        assert (shot.live, shot.queued, shot.done, shot.total) == (58, 603, 12, 661)
        assert (shot.peak, shot.condemned) == (58, 2)
        assert shot.at == DAY.replace(hour=8, minute=9, second=55)

    def test_an_older_line_without_condemned_still_reads(self):
        """The field was added mid-run. A log holds both shapes."""
        shot = status.parse_status(
            "07:06:25Z  live=60 queued=605 done=0/665 peak=60", DAY
        )
        assert shot is not None and shot.condemned == 0

    def test_another_line_is_not_a_status(self):
        assert status.parse_status("filling 80 slot(s): 0 live", DAY) is None
        assert status.parse_status("N00E005  i-0abc  1.2.3.4", DAY) is None

    def test_a_run_across_midnight_rolls_forward(self):
        """The driver prints a time and no date. A six-hour run can cross it."""
        day = datetime(2026, 9, 16, 22, 0, tzinfo=timezone.utc)
        shot = status.parse_status("01:30:00Z  live=1 queued=0 done=5/6 peak=1", day)
        assert shot is not None
        assert shot.at.day == 17

    def test_the_fraction_is_safe_at_zero_total(self):
        shot = status.parse_status("08:00:00Z  live=0 queued=0 done=0/0 peak=0", DAY)
        assert shot is not None and shot.fraction == 0.0


def shots(*pairs):
    return [
        status.Snapshot(
            at=DAY + timedelta(minutes=m),
            live=1,
            queued=0,
            done=d,
            total=100,
            peak=1,
            condemned=0,
        )
        for m, d in pairs
    ]


class TestTheRateIsMeasuredNotModelled:
    """`waves.plan_text` projects from a fit made before the run. This is the
    run itself, which is the number somebody watching actually wants."""

    def test_sixty_tiles_in_an_hour_reads_as_sixty_an_hour(self):
        assert status.rate_per_hour(shots((0, 0), (60, 60)), 120) == 60.0

    def test_one_sample_has_no_rate(self):
        assert status.rate_per_hour(shots((0, 0)), 30) is None

    def test_the_window_ignores_the_early_slow_part(self):
        """The first half hour places instances and finishes nothing."""
        history = shots((0, 0), (30, 0), (60, 30), (90, 60))
        assert status.rate_per_hour(history, 60) == 60.0

    def test_a_window_holding_one_sample_falls_back_to_the_last_two(self):
        assert status.rate_per_hour(shots((0, 0), (600, 100)), 1) == 10.0

    def test_a_stalled_run_reads_as_zero_not_as_finished(self):
        assert status.rate_per_hour(shots((0, 50), (60, 50)), 120) == 0.0

    def test_the_eta_follows_the_rate(self):
        history = shots((0, 0), (60, 50))
        last = history[-1]
        when = status.eta(last, status.rate_per_hour(history, 120))
        assert when == last.at + timedelta(hours=1)

    def test_a_zero_rate_gives_no_eta(self):
        assert status.eta(shots((0, 1))[0], 0.0) is None
        assert status.eta(shots((0, 1))[0], None) is None


class TestADeadDriverIsTheHeadline:
    """The failure this module exists for."""

    def block(self, age_minutes, alive, path=Path("world.log")):
        history = shots((0, 10), (30, 40))
        now = history[-1].at + timedelta(minutes=age_minutes)
        return status.render(
            path,
            history,
            failures=0,
            throttles=0,
            now=now,
            alive=alive,
            offset_hours=2.0,
        )

    def test_a_fresh_log_says_nothing_about_staleness(self):
        assert "HAS NOT MOVED" not in self.block(1, 58)

    def test_a_stale_log_shouts(self):
        text = self.block(10, 58)
        assert "HAS NOT MOVED" in text
        assert "probably died" in text

    def test_a_stale_log_names_the_instances_still_billing(self):
        text = self.block(10, 58)
        assert "58 instance(s) are still running and billing" in text
        assert "--adopt" in text

    def test_a_stale_log_with_nothing_running_says_so(self):
        text = self.block(10, 0)
        assert "Nothing is billing" in text
        assert "--adopt" not in text

    def test_the_progress_still_prints_under_the_warning(self):
        """A dead driver does not make the tiles it finished disappear."""
        assert "40/100" in self.block(10, 58)


class TestTheBlockReadsAtAGlance:
    def test_it_reports_what_aws_says_not_only_what_the_log_says(self):
        history = shots((0, 0), (30, 40))
        text = status.render(
            Path("w.log"),
            history,
            failures=3,
            throttles=1,
            now=history[-1].at,
            alive=12,
            offset_hours=2.0,
        )
        assert "live   1" in text
        assert "12 running per AWS" in text
        assert "requeued 3" in text
        assert "throttles 1" in text

    def test_an_empty_log_says_it_is_waiting(self):
        text = status.render(
            Path("w.log"),
            [],
            failures=0,
            throttles=0,
            now=DAY,
            alive=None,
            offset_hours=2.0,
        )
        assert "waiting for the first status line" in text

    def test_the_local_clock_carries_the_offset(self):
        assert status.clock(DAY.replace(hour=8, minute=9), 2.0) == "10:09"

    def test_the_bar_fills_with_the_fraction(self):
        assert status.bar(0.0, 10) == "-" * 10
        assert status.bar(1.0, 10) == "#" * 10
        assert status.bar(0.5, 10).count("#") == 5

    def test_the_cost_line_appears_only_with_measurements(self):
        history = shots((0, 0), (30, 40))
        without = status.render(
            Path("w"),
            history,
            failures=0,
            throttles=0,
            now=history[-1].at,
            alive=1,
            offset_hours=2.0,
        )
        assert "cost" not in without
        with_cost = status.render(
            Path("w"),
            history,
            failures=0,
            throttles=0,
            now=history[-1].at,
            alive=1,
            offset_hours=2.0,
            cost=(60, 7200, 7.59),
        )
        assert "$    7.59" in with_cost
        assert "60 instance(s)" in with_cost


class TestTheLogIsReadWhole:
    def test_it_counts_requeues_and_throttles(self, tmp_path):
        log = tmp_path / "world.log"
        log.write_text(
            "manifest /x/run-20260916-080353.json\n"
            f"{LINE}\n"
            "  N00E005 failed, back to the queue (attempt 1)\n"
            "N00E010  AWS is throttling, waiting 4s\n"
            "  N00E015 gone, back to the queue (attempt 1)\n"
        )
        found, failures, throttles = status.read_log(log)
        assert len(found) == 1
        assert failures == 2
        assert throttles == 1

    def test_the_date_comes_from_the_manifest_name(self, tmp_path):
        log = tmp_path / "world.log"
        log.write_text(f"manifest /x/run-20260916-080353.json\n{LINE}\n")
        found, _, _ = status.read_log(log)
        assert found[0].at.date() == DAY.date()

    def test_the_newest_log_is_the_default(self, tmp_path):
        import os

        old, new = tmp_path / "a.log", tmp_path / "b.log"
        old.write_text("")
        new.write_text("")
        os.utime(old, (1, 1))
        assert status.newest_log(tmp_path) == new

    def test_an_empty_directory_has_no_log(self, tmp_path):
        assert status.newest_log(tmp_path) is None

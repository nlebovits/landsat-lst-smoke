"""The composite window is 2021-2025, and a cache cannot cross windows.

Two defects shipped together in the 2020-01-01/2025-01-01 default: it included
all of 2020, which is outside the five-year window, and it excluded all of
2025, which is inside it. A cached STAC item list made the second one sticky,
because the cache file name said nothing about the window it was searched over.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import measure_s3_requests  # noqa: E402
import profile_lst_p95  # noqa: E402
import shard_lst_p95  # noqa: E402
from stac_window import (  # noqa: E402
    DEFAULT_END,
    DEFAULT_START,
    datetime_range,
    items_cache_path,
    query_digest,
)

QUARTER_TILE = (-62.5, -35.0, -60.0, -32.5)
OLD_START, OLD_END = "2020-01-01", "2025-01-01"


class TestDefaultWindow:
    def test_window_opens_on_the_first_day_of_2021(self):
        assert DEFAULT_START == "2021-01-01"

    def test_window_closes_on_the_last_instant_of_2025(self):
        assert DEFAULT_END == "2025-12-31T23:59:59Z"

    def test_window_is_five_calendar_years(self):
        first = int(DEFAULT_START[:4])
        last = int(DEFAULT_END[:4])
        assert last - first + 1 == 5

    def test_the_search_string_carries_both_endpoints(self):
        assert datetime_range() == "2021-01-01/2025-12-31T23:59:59Z"

    def test_the_end_keeps_a_time_of_day(self):
        # A STAC datetime range is closed at both ends. A bare date would drop
        # every scene acquired on 31 December 2025.
        assert "T" in DEFAULT_END


class TestEntryPointDefaults:
    """Every path that searches STAC has to ask for the same five years."""

    def test_shard_pipeline(self):
        args = shard_lst_p95.parse_args(["--bbox=-62.5,-35.0,-60.0,-32.5"])
        assert (args.start, args.end) == (DEFAULT_START, DEFAULT_END)

    def test_profile_harness(self):
        args = profile_lst_p95.parse_args([])
        assert (args.start, args.end) == (DEFAULT_START, DEFAULT_END)

    def test_s3_request_measurement(self):
        # main() builds its parser inline, so assert on the constants it hands
        # argparse plus the fact that it hands them and not a literal.
        assert (measure_s3_requests.DEFAULT_START, measure_s3_requests.DEFAULT_END) == (
            DEFAULT_START,
            DEFAULT_END,
        )
        src = Path(measure_s3_requests.__file__).read_text()
        assert 'ap.add_argument("--start", default=DEFAULT_START)' in src
        assert 'ap.add_argument("--end", default=DEFAULT_END)' in src

    def test_dryrun_script_source_uses_the_shared_window(self):
        src = (
            Path(__file__).resolve().parent.parent / "dryrun" / "dryrun.py"
        ).read_text()
        assert "datetime_range(DEFAULT_START, DEFAULT_END)" in src
        assert OLD_START not in src


class TestStacQuery:
    def test_search_items_sends_the_intended_timestamps(self, monkeypatch):
        """The string that reaches pystac_client, not the argparse default."""
        seen = {}

        class FakeSearch:
            def items(self):
                return []

        class FakeClient:
            @staticmethod
            def open(url):
                return FakeClient()

            def search(self, **kwargs):
                seen.update(kwargs)
                return FakeSearch()

        import pystac_client

        monkeypatch.setattr(pystac_client, "Client", FakeClient)
        args = shard_lst_p95.parse_args(["--bbox=-62.5,-35.0,-60.0,-32.5"])
        items, bboxes = shard_lst_p95.search_items(args, QUARTER_TILE)

        assert items == [] and bboxes == []
        assert seen["datetime"] == "2021-01-01/2025-12-31T23:59:59Z"
        assert seen["collections"] == ["landsat-c2-l2"]


class TestCacheIdentity:
    def test_the_default_window_names_one_file(self):
        assert items_cache_path(QUARTER_TILE) == items_cache_path(
            QUARTER_TILE, DEFAULT_START, DEFAULT_END
        )

    def test_a_2020_cache_cannot_satisfy_a_2021_request(self):
        old = items_cache_path(QUARTER_TILE, OLD_START, OLD_END)
        new = items_cache_path(QUARTER_TILE, DEFAULT_START, DEFAULT_END)
        assert old != new
        assert old.name != new.name

    def test_moving_the_start_changes_the_cache(self):
        a = items_cache_path(QUARTER_TILE, DEFAULT_START, DEFAULT_END)
        b = items_cache_path(QUARTER_TILE, "2020-01-01", DEFAULT_END)
        assert a != b

    def test_moving_the_end_changes_the_cache(self):
        a = items_cache_path(QUARTER_TILE, DEFAULT_START, DEFAULT_END)
        b = items_cache_path(QUARTER_TILE, DEFAULT_START, "2025-01-01")
        assert a != b

    def test_the_window_is_readable_in_the_file_name(self):
        name = items_cache_path(QUARTER_TILE).name
        assert "20210101" in name
        assert "20251231T235959Z" in name

    def test_a_different_bbox_changes_the_cache(self):
        a = items_cache_path(QUARTER_TILE)
        b = items_cache_path((-65.0, -35.0, -60.0, -30.0))
        assert a != b

    def test_the_query_filters_are_part_of_the_identity(self):
        base = query_digest(QUARTER_TILE)
        assert base != query_digest(QUARTER_TILE, cloud_cover_lt=50)
        assert base != query_digest(QUARTER_TILE, platforms="landsat-9")
        assert base != query_digest(QUARTER_TILE, collection="landsat-c2-l1")

    def test_the_digest_is_stable_across_calls(self):
        assert query_digest(QUARTER_TILE) == query_digest(QUARTER_TILE)


def test_no_entry_point_still_carries_the_old_window():
    root = Path(__file__).resolve().parent.parent
    scripts = [
        root / "shard_lst_p95.py",
        root / "profile_lst_p95.py",
        root / "measure_s3_requests.py",
        *sorted((root / "dryrun").glob("*.py")),
    ]
    offenders = [p.name for p in scripts if '"2020-01-01"' in p.read_text()]
    assert offenders == []

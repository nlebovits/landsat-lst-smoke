"""The scene table: written once, read once per worker, never stale.

Every shard used to carry its own list of STAC item dicts to the worker that
ran it. A shard sees a p50 of 509 scenes, so a full tile pushed 0.55 GB of task
payload to describe 3,910 items that pickle to 3.2 MB between them. MEASURED by
`measure_submit_cost.py`, that costs 5.3 to 6.1 ms per submit against 0.024 to
0.031 ms for a path and a list of positions.

What can go wrong is narrow and it is all here: a table that does not survive
the round trip would change the composite, and a cache that outlives its file
would compute one tile from another tile's scenes.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import item_table  # noqa: E402
from tile_inventory import items_for_tile  # noqa: E402

TILE = "S30W065"


@pytest.fixture
def real_items(slice_artifact):
    """Item dicts for one tile of the committed inventory slice."""
    items, _ = items_for_tile(slice_artifact, TILE)
    return items


@pytest.fixture(autouse=True)
def _empty_cache():
    """Each test starts with nothing parsed. The cache is a module global."""
    item_table._CACHE.clear()
    yield
    item_table._CACHE.clear()


def as_json_shape(value):
    """`value` with every tuple turned into a list, and nothing else changed.

    JSON has no tuple, so the one thing the round trip alters is that
    `build_item` writes each geometry corner as a tuple and reads it back as a
    two-element list. This normalises that and only that, so the tests below
    still compare every number rather than excusing the whole geometry.
    """
    if isinstance(value, dict):
        return {k: as_json_shape(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json_shape(v) for v in value]
    return value


class TestTheRoundTrip:
    def test_the_corner_tuples_are_the_only_thing_that_moves(self, real_items):
        # Named exactly, because "close enough" about the scene list is how a
        # composite comes out of the wrong scenes.
        item = real_items[0]
        back = json.loads(json.dumps(item))

        assert back != item
        assert back == as_json_shape(item)
        ring = item["geometry"]["coordinates"][0]
        assert all(isinstance(corner, tuple) for corner in ring)
        assert back["geometry"]["coordinates"][0] == [list(c) for c in ring]

    def test_pystac_carries_the_difference_no_further(self, real_items):
        # What actually consumes these. `process_shard` calls
        # `pystac.Item.from_dict` on whatever the table hands it, and pystac
        # passes the geometry through rather than normalising it, so the two
        # items still differ in the corner shape and in nothing else. That the
        # difference never reaches a pixel is checked against the arrays
        # themselves, in tests/test_no_stac_at_runtime.py.
        import pystac

        item = real_items[0]
        before = pystac.Item.from_dict(item).to_dict()
        after = pystac.Item.from_dict(json.loads(json.dumps(item))).to_dict()

        assert as_json_shape(before) == as_json_shape(after)
        assert {k: v for k, v in before.items() if k != "geometry"} == {
            k: v for k, v in after.items() if k != "geometry"
        }

    def test_write_then_load_returns_what_went_in(self, real_items, tmp_path):
        report = item_table.write(tmp_path / "t.json", real_items)

        assert item_table.load(report["path"]) == as_json_shape(real_items)
        assert report["n_items"] == len(real_items)
        assert report["bytes"] == (tmp_path / "t.json").stat().st_size

    def test_select_takes_items_by_position(self, real_items, tmp_path):
        report = item_table.write(tmp_path / "t.json", real_items)
        picked = item_table.select(report["path"], [2, 0, 2])
        assert picked == as_json_shape([real_items[2], real_items[0], real_items[2]])

    def test_a_missing_table_fails_loudly(self, tmp_path):
        # Every shard would fail anyway. Failing here names the file.
        with pytest.raises(FileNotFoundError):
            item_table.load(tmp_path / "absent.json")


class TestTheCache:
    def test_the_table_is_parsed_once_per_process(self, real_items, tmp_path):
        report = item_table.write(tmp_path / "t.json", real_items)

        first = item_table.load(report["path"])
        second = item_table.load(report["path"])

        # The same object, not an equal one. A worker runs many shards and
        # parsing 7.8 MB for each of them would give the saving straight back.
        assert first is second
        assert len(item_table._CACHE) == 1

    def test_a_rewritten_table_is_not_served_from_the_cache(self, tmp_path):
        # The tests run several pipelines in one process, and a fleet instance
        # runs tiles back to back. A cache keyed on the path alone would hand
        # the second tile the first one's scenes.
        path = tmp_path / "t.json"
        item_table.write(path, [{"id": "first"}])
        assert item_table.load(path) == [{"id": "first"}]

        item_table.write(path, [{"id": "second"}, {"id": "third"}])

        assert item_table.load(path) == [{"id": "second"}, {"id": "third"}]
        assert len(item_table._CACHE) == 1

    def test_two_threads_parse_it_once(self, real_items, tmp_path, monkeypatch):
        # Both of a worker's threads reach the first shard together.
        report = item_table.write(tmp_path / "t.json", real_items)
        parses = []
        real_loads = json.loads

        def counting_loads(payload):
            parses.append(1)
            return real_loads(payload)

        monkeypatch.setattr(item_table.json, "loads", counting_loads)

        start = threading.Barrier(4)
        results = []

        def worker():
            start.wait()
            results.append(item_table.load(report["path"]))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(parses) == 1
        assert all(r is results[0] for r in results)

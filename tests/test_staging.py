"""Staging: one GET per object, verified, and countable.

The saving these tests protect is a factor of 739. It comes from two places and
both are easy to lose without an error appearing anywhere.

The first is the deduplication. About 155 shards touch each scene, so a manifest
that keeps one entry per shard-scene pair rather than per object gives all of it
back while still producing a correct composite.

The second is `get_object`. `boto3`'s `download_file` splits anything over 8 MB
into several ranged GETs, so swapping one call for the other quadruples the
request count and changes nothing else a test would notice. `test_one_get_per_object`
is the check that says so.

Everything here runs offline against a fake S3 client. `slice_artifact` supplies
real item dicts, because a hand-built asset dictionary cannot prove the module
reads what `tile_inventory.build_item` writes.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import staging  # noqa: E402
from conftest import make_row  # noqa: E402
from tile_inventory import build_item, items_for_tile  # noqa: E402

TILE = "S30W065"


class FakeBody(io.BytesIO):
    """An S3 streaming body. `copyfileobj` only needs `read`."""


class FakeS3:
    """Counts every call and serves a body of the size it announces.

    `short_by` makes the announced `ContentLength` exceed what the body holds,
    which is the truncation case. `fail_first` raises on the opening attempts
    for one key, to exercise the retry counter.
    """

    def __init__(self, *, payload=b"tiff-bytes", short_by=0, fail_first=0):
        self.payload = payload
        self.short_by = short_by
        self.fail_first = fail_first
        self.calls: list[tuple[str, str]] = []
        self.payers: list[str | None] = []

    def get_object(self, *, Bucket, Key, RequestPayer=None):  # noqa: N803
        self.calls.append((Bucket, Key))
        self.payers.append(RequestPayer)
        if len(self.calls) <= self.fail_first:
            msg = "connection reset"
            raise OSError(msg)
        return {
            "ContentLength": len(self.payload) + self.short_by,
            "Body": FakeBody(self.payload),
        }


@pytest.fixture
def real_items(slice_artifact):
    """Item dicts and bboxes for one tile of the committed inventory slice."""
    items, boxes = items_for_tile(slice_artifact, TILE)
    return items, boxes


def thermal_items(n, *, thermal=True):
    """`n` item dicts built through the production row-to-item path."""
    return [make_row(TILE, i, thermal=thermal) for i in range(n)]


class TestDropScenesWithoutThermal:
    def test_a_product_with_no_thermal_band_goes(self):
        rows = thermal_items(3) + thermal_items(2, thermal=False)
        items = [build_item(r) for r in rows]
        boxes = [tuple(i["bbox"]) for i in items]

        kept, kept_boxes, dropped = staging.drop_scenes_without_thermal(items, boxes)

        assert dropped == 2
        assert len(kept) == 3
        assert all("lwir11" in i["assets"] for i in kept)
        assert kept_boxes == boxes[:3]

    def test_the_bboxes_stay_aligned_with_the_items(self):
        # An item list and a bbox list that disagree by one index sends every
        # shard a different scene than it selected, and the composite still
        # finishes. Nothing downstream can catch it, so it is caught here.
        rows = [
            make_row(TILE, 0, thermal=False),
            make_row(TILE, 1),
            make_row(TILE, 2, thermal=False),
            make_row(TILE, 3),
        ]
        items = [build_item(r) for r in rows]
        boxes = [(float(n), 0.0, 0.0, 0.0) for n in range(4)]

        kept, kept_boxes, _ = staging.drop_scenes_without_thermal(items, boxes)

        assert [i["id"] for i in kept] == [items[1]["id"], items[3]["id"]]
        assert kept_boxes == [boxes[1], boxes[3]]

    def test_every_scene_lacking_thermal_leaves_nothing(self):
        items = [build_item(r) for r in thermal_items(3, thermal=False)]
        boxes = [tuple(i["bbox"]) for i in items]

        kept, kept_boxes, dropped = staging.drop_scenes_without_thermal(items, boxes)

        assert (kept, kept_boxes, dropped) == ([], [], 3)

    def test_real_inventory_items_survive(self, real_items):
        items, boxes = real_items
        kept, _, dropped = staging.drop_scenes_without_thermal(items, boxes)
        assert len(kept) + dropped == len(items)
        assert kept, "the slice holds no scene with a thermal band"


class TestManifest:
    def test_one_scene_in_many_shards_yields_two_objects(self, real_items):
        items, _ = real_items
        # The same three scenes, as 155 shards would present them.
        indices = [0, 1, 2] * 155

        manifest = staging.staging_manifest(items, indices)

        assert len(manifest) == 6
        assert {band for _, band, _ in manifest} == {"lwir11", "qa_pixel"}

    def test_an_item_id_that_is_a_path_is_refused(self):
        item = build_item(make_row(TILE, 0))
        item["id"] = "../../etc"
        with pytest.raises(staging.StagingError, match="directory name"):
            staging.staging_manifest([item], [0])

    def test_a_product_with_no_thermal_contributes_one_object(self):
        item = build_item(make_row(TILE, 0, thermal=False))
        manifest = staging.staging_manifest([item], [0])
        assert [band for _, band, _ in manifest] == ["qa_pixel"]


class TestSplitS3Uri:
    def test_a_bucket_and_key_come_apart(self):
        assert staging.split_s3_uri("s3://usgs-landsat/a/b/c.TIF") == (
            "usgs-landsat",
            "a/b/c.TIF",
        )

    @pytest.mark.parametrize(
        "href",
        ["https://example.invalid/a.TIF", "s3://bucket", "s3:///key", "/local/a.TIF"],
    )
    def test_anything_else_is_refused(self, href):
        with pytest.raises(staging.StagingError):
            staging.split_s3_uri(href)


class TestStageScenes:
    def test_one_get_per_object(self, real_items, tmp_path):
        # The whole saving in one assertion. `download_file` would issue about
        # four calls per object here and every other test would still pass.
        items, _ = real_items
        fake = FakeS3()

        report = staging.stage_scenes(
            items, [0, 1, 2] * 155, tmp_path, threads=4, client_factory=lambda: fake
        )

        assert len(fake.calls) == 6
        assert report["objects"] == 6
        assert report["get_requests"] == 6
        assert report["retries"] == 0

    def test_every_request_declares_the_requester_pays(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3()
        staging.stage_scenes(
            items, [0], tmp_path, threads=2, client_factory=lambda: fake
        )
        assert set(fake.payers) == {"requester"}

    def test_the_hrefs_point_at_the_staged_files(self, real_items, tmp_path):
        items, _ = real_items
        before = dict(items[0]["assets"]["lwir11"])
        fake = FakeS3()

        staging.stage_scenes(
            items, [0], tmp_path, threads=2, client_factory=lambda: fake
        )

        after = items[0]["assets"]["lwir11"]
        staged = Path(after["href"])
        assert staged.is_file()
        assert staged.is_absolute()
        assert not after["href"].startswith("file://")
        # Nothing but the href moved. `raster:bands` carries the scale and
        # offset the loader decodes with.
        assert {k: v for k, v in after.items() if k != "href"} == {
            k: v for k, v in before.items() if k != "href"
        }

    def test_the_staged_file_holds_what_s3_sent(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3(payload=b"x" * 4096)
        staging.stage_scenes(
            items, [0], tmp_path, threads=2, client_factory=lambda: fake
        )
        assert Path(items[0]["assets"]["qa_pixel"]["href"]).read_bytes() == b"x" * 4096

    def test_a_short_read_fails_and_leaves_no_file(self, real_items, tmp_path):
        # A truncated GeoTIFF produces a wrong percentile in silence. It has to
        # stop the run, and it must not survive as a plausible cache entry.
        items, _ = real_items
        fake = FakeS3(short_by=10)

        with pytest.raises(staging.StagingError, match="reached disk"):
            staging.stage_scenes(
                items, [0], tmp_path, threads=1, client_factory=lambda: fake
            )

        assert not list(tmp_path.rglob("*.TIF"))

    def test_a_retry_is_counted_as_a_billable_request(self, real_items, tmp_path):
        # A failed GET is billed. Counting only successes would understate the
        # S3 line for exactly the runs that went badly.
        items, _ = real_items
        fake = FakeS3(fail_first=1)

        report = staging.stage_scenes(
            items, [0], tmp_path, threads=1, client_factory=lambda: fake
        )

        assert report["objects"] == 2
        assert report["get_requests"] == 3
        assert report["retries"] == 1

    def test_an_object_that_never_arrives_stops_the_run(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3(fail_first=staging.MAX_ATTEMPTS)
        with pytest.raises(staging.StagingError, match="attempts"):
            staging.stage_scenes(
                items, [0], tmp_path, threads=1, client_factory=lambda: fake
            )

    def test_no_indices_fetches_nothing(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3()
        report = staging.stage_scenes(items, [], tmp_path, client_factory=lambda: fake)
        assert fake.calls == []
        assert report["objects"] == 0


class TestTheDefaultClient:
    """Two botocore defaults would break the two claims this module makes.

    Neither is visible in a small run. A clean fetch of six objects reports the
    right count and saturates a pool of ten, so only a fleet-sized slice under
    throttling would show either one.
    """

    def test_botocore_does_not_retry_behind_the_counter(self):
        # `legacy` retries a 503 up to five times inside get_object. Those are
        # billable GETs that `_fetch_one` cannot see, so a throttled run would
        # under-report the S3 line that `cost_report.py --s3-get-requests`
        # prices. Retrying belongs to this module, where MAX_ATTEMPTS bounds it.
        client = staging._default_client()
        # total_max_attempts, not max_attempts: botocore reads the latter as
        # retries after the first try, so a 1 there still sends two requests.
        assert client.meta.config.retries["total_max_attempts"] == 1

    def test_the_connection_pool_matches_the_fetch_pool(self):
        # Default 10, against up to 64 fetch threads. The surplus threads queue
        # on a connection rather than on the network.
        client = staging._default_client()
        assert client.meta.config.max_pool_connections >= staging._default_threads()


class TestDiskGuard:
    def test_it_refuses_a_slice_that_does_not_fit(
        self, real_items, tmp_path, monkeypatch
    ):
        items, _ = real_items
        manifest = staging.staging_manifest(items, range(len(items)))
        need = staging.estimated_bytes(manifest)
        monkeypatch.setattr(
            staging.shutil, "disk_usage", lambda _p: type("U", (), {"free": need - 1})()
        )

        with pytest.raises(staging.StagingError) as exc:
            staging.disk_guard(manifest, tmp_path)

        # Both figures and the escape, because the next decision is whether to
        # resize the volume or to read from S3.
        message = str(exc.value)
        assert f"{need / 1024**3:.1f} GiB" in message
        assert "--no-stage" in message

    def test_it_passes_when_the_volume_is_large_enough(self, tmp_path, monkeypatch):
        manifest = [("id", "lwir11", "s3://b/k.TIF")]
        monkeypatch.setattr(
            staging.shutil,
            "disk_usage",
            lambda _p: type("U", (), {"free": 1 << 60})(),
        )
        assert staging.disk_guard(manifest, tmp_path) == staging.estimated_bytes(
            manifest
        )

    def test_the_estimate_carries_the_safety_factor(self):
        manifest = [("a", "lwir11", "s3://b/k"), ("a", "qa_pixel", "s3://b/q")]
        raw = staging.ESTIMATED_BYTES["lwir11"] + staging.ESTIMATED_BYTES["qa_pixel"]
        assert staging.estimated_bytes(manifest) == int(
            raw * staging.DISK_SAFETY_FACTOR
        )


class TestCleanup:
    def test_it_removes_the_directory(self, tmp_path):
        target = tmp_path / "stage"
        (target / "scene").mkdir(parents=True)
        (target / "scene" / "lwir11.TIF").write_bytes(b"x")

        staging.cleanup(target)

        assert not target.exists()

    def test_a_directory_that_is_already_gone_is_not_an_error(self, tmp_path):
        staging.cleanup(tmp_path / "never-existed")

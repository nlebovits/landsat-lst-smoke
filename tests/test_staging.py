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

import copy
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


class TestStagedPaths:
    """The href is named before the GET, so the graph can be built while it runs."""

    def test_the_rewrite_agrees_with_what_the_fetch_writes(self, real_items, tmp_path):
        # The two used to be one statement. They are now a prediction and a
        # write, and if they disagree every block opens a file that is not
        # there.
        items, _ = real_items
        fake = FakeS3()

        staging.stage_scenes(
            items, [0, 1], tmp_path, threads=2, client_factory=lambda _n: fake
        )

        for item in items[:2]:
            for band in staging.BANDS:
                href = item["assets"][band]["href"]
                assert Path(href).is_file()
                assert Path(href).parent.name == item["id"]
                assert Path(href).stem == band

    def test_the_hrefs_are_rewritten_before_the_first_get(self, real_items, tmp_path):
        # What makes the overlap possible. The table is final from the start,
        # so the only thing a shard waits on is its own files arriving.
        items, _ = real_items
        seen_at_get = []

        class Watching(FakeS3):
            def get_object(self, **kw):  # noqa: N803
                seen_at_get.append(items[0]["assets"]["lwir11"]["href"])
                return super().get_object(**kw)

        staging.stage_scenes(
            items, [0], tmp_path, threads=1, client_factory=lambda _n: Watching()
        )

        assert seen_at_get, "nothing was fetched"
        assert not any(h.startswith("s3://") for h in seen_at_get)

    def test_the_manifest_comes_back_so_it_is_not_built_twice(
        self, real_items, tmp_path
    ):
        items, _ = real_items
        manifest = staging.repoint_items(items, [0, 1], tmp_path)

        assert len(manifest) == 4
        assert all(href.startswith("s3://") for _, _, href in manifest)
        assert all(
            item["assets"]["lwir11"]["href"].startswith(str(tmp_path))
            for item in items[:2]
        )

    def test_repointing_an_already_staged_item_is_refused(self, real_items, tmp_path):
        # The rewrite reads the s3 href to name the destination, so it can only
        # run once. Running it twice means the caller lost track of which
        # ordering it is in, and a silent second pass would name files after
        # local paths.
        items, _ = real_items
        staging.repoint_items(items, [0], tmp_path)

        with pytest.raises(staging.StagingError, match="expects an s3:// href"):
            staging.repoint_items(items, [0], tmp_path)


class TestStagingRun:
    """The streaming form. `stage_scenes` is its serial drain."""

    def test_every_object_lands_exactly_once(self, real_items, tmp_path):
        items, _ = real_items
        manifest = staging.repoint_items(items, [0, 1, 2], tmp_path)
        fake = FakeS3()

        landed = []
        with staging.StagingRun(
            manifest, tmp_path, threads=2, client_factory=lambda _n: fake
        ) as run:
            while not run.done:
                landed.extend(run.landed())

        assert sorted(landed) == sorted((i, b) for i, b, _ in manifest)
        assert run.report()["objects"] == len(manifest)

    def test_the_report_matches_what_stage_scenes_returns(self, real_items, tmp_path):
        # One fetch, two orderings. They must not diverge on the counts that
        # `cost_report.py` prices.
        items, _ = real_items
        fresh = copy.deepcopy(items)
        serial = staging.stage_scenes(
            items, [0, 1], tmp_path / "a", threads=2, client_factory=lambda _n: FakeS3()
        )
        manifest = staging.repoint_items(fresh, [0, 1], tmp_path / "b")
        with staging.StagingRun(
            manifest, tmp_path / "b", threads=2, client_factory=lambda _n: FakeS3()
        ) as run:
            run.drain()
        streamed = run.report()

        for key in ("objects", "bytes", "get_requests", "retries"):
            assert serial[key] == streamed[key], key

    def test_a_failed_fetch_reaches_the_caller(self, real_items, tmp_path):
        items, _ = real_items
        manifest = staging.repoint_items(items, [0], tmp_path)
        fake = FakeS3(fail_first=staging.MAX_ATTEMPTS)

        with pytest.raises(staging.StagingError):  # noqa: PT012
            with staging.StagingRun(
                manifest, tmp_path, threads=1, client_factory=lambda _n: fake
            ) as run:
                while not run.done:
                    run.landed()

    def test_one_failure_abandons_the_rest_of_the_manifest(self, real_items, tmp_path):
        # A failed GET is billed. Draining the pool after the run is already
        # lost buys nothing and pays for thousands of objects.
        items, _ = real_items
        manifest = staging.repoint_items(items, list(range(20)), tmp_path)
        fake = FakeS3(fail_first=staging.MAX_ATTEMPTS)

        with pytest.raises(staging.StagingError):  # noqa: PT012
            with staging.StagingRun(
                manifest, tmp_path, threads=1, client_factory=lambda _n: fake
            ) as run:
                run.drain()

        assert len(fake.calls) < len(manifest), (
            f"{len(fake.calls)} GETs billed for a run abandoned at the first"
        )


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
            items, [0, 1, 2] * 155, tmp_path, threads=4, client_factory=lambda _n: fake
        )

        assert len(fake.calls) == 6
        assert report["objects"] == 6
        assert report["get_requests"] == 6
        assert report["retries"] == 0

    def test_a_second_run_into_a_warm_directory_pays_nothing(
        self, real_items, tmp_path
    ):
        """The tile now takes two traversals, and it used to fetch twice.

        `tile_prep` stages the tile and `shard_lst_p95` stages the same objects
        behind it. Refetching them cost 322 s and 297 GB on a mean tile, which
        is 21% of the run, and doubled the GET count.

        Existence is enough to trust because `_fetch_one` unlinks on every
        failure path including a short read, so a file that is there is whole.
        A HEAD would confirm it and would be billable, which is most of what
        skipping saves.
        """
        items, _ = real_items
        # `repoint_items` rewrites hrefs in place, and a second process builds
        # its own items from the inventory rather than inheriting these.
        fresh = copy.deepcopy(items)
        first = FakeS3()
        a = staging.stage_scenes(
            items, [0, 1], tmp_path, threads=2, client_factory=lambda _n: first
        )
        second = FakeS3()
        b = staging.stage_scenes(
            fresh, [0, 1], tmp_path, threads=2, client_factory=lambda _n: second
        )

        assert a["get_requests"] == 4
        assert a["reused"] == 0
        assert len(second.calls) == 0
        assert b["get_requests"] == 0
        assert b["reused"] == 4
        assert b["objects"] == 4

    def test_a_reused_object_is_not_counted_as_a_negative_retry(
        self, real_items, tmp_path
    ):
        # `retries` is requests minus objects fetched. Counting a skipped
        # object as fetched makes a warm rerun report -4 retries.
        items, _ = real_items
        fresh = copy.deepcopy(items)
        staging.stage_scenes(
            items, [0, 1], tmp_path, threads=2, client_factory=lambda _n: FakeS3()
        )
        report = staging.stage_scenes(
            fresh, [0, 1], tmp_path, threads=2, client_factory=lambda _n: FakeS3()
        )
        assert report["retries"] == 0

    def test_a_partial_file_is_never_left_to_be_trusted(self, real_items, tmp_path):
        """What makes existence a safe test, restated where the skip reads it.

        A truncated GeoTIFF would produce a wrong percentile in silence. If a
        short read ever left one behind, the skip above would adopt it.
        """
        items, _ = real_items
        with pytest.raises(staging.StagingError):
            staging.stage_scenes(
                items,
                [0],
                tmp_path,
                threads=1,
                client_factory=lambda _n: FakeS3(short_by=3),
            )
        assert not list(tmp_path.rglob("*.TIF"))

    def test_every_request_declares_the_requester_pays(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3()
        staging.stage_scenes(
            items, [0], tmp_path, threads=2, client_factory=lambda _n: fake
        )
        assert set(fake.payers) == {"requester"}

    def test_the_hrefs_point_at_the_staged_files(self, real_items, tmp_path):
        items, _ = real_items
        before = dict(items[0]["assets"]["lwir11"])
        fake = FakeS3()

        staging.stage_scenes(
            items, [0], tmp_path, threads=2, client_factory=lambda _n: fake
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
            items, [0], tmp_path, threads=2, client_factory=lambda _n: fake
        )
        assert Path(items[0]["assets"]["qa_pixel"]["href"]).read_bytes() == b"x" * 4096

    def test_a_short_read_fails_and_leaves_no_file(self, real_items, tmp_path):
        # A truncated GeoTIFF produces a wrong percentile in silence. It has to
        # stop the run, and it must not survive as a plausible cache entry.
        items, _ = real_items
        fake = FakeS3(short_by=10)

        with pytest.raises(staging.StagingError, match="reached disk"):
            staging.stage_scenes(
                items, [0], tmp_path, threads=1, client_factory=lambda _n: fake
            )

        assert not list(tmp_path.rglob("*.TIF"))

    def test_a_retry_is_counted_as_a_billable_request(self, real_items, tmp_path):
        # A failed GET is billed. Counting only successes would understate the
        # S3 line for exactly the runs that went badly.
        items, _ = real_items
        fake = FakeS3(fail_first=1)

        report = staging.stage_scenes(
            items, [0], tmp_path, threads=1, client_factory=lambda _n: fake
        )

        assert report["objects"] == 2
        assert report["get_requests"] == 3
        assert report["retries"] == 1

    def test_an_object_that_never_arrives_stops_the_run(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3(fail_first=staging.MAX_ATTEMPTS)
        with pytest.raises(staging.StagingError, match="attempts"):
            staging.stage_scenes(
                items, [0], tmp_path, threads=1, client_factory=lambda _n: fake
            )

    def test_no_indices_fetches_nothing(self, real_items, tmp_path):
        items, _ = real_items
        fake = FakeS3()
        report = staging.stage_scenes(
            items, [], tmp_path, client_factory=lambda _n: fake
        )
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

    def test_the_pool_follows_the_thread_count_it_was_given(self):
        # Reading the module default instead would reintroduce the queue for
        # any caller that asks for more threads than the default.
        client = staging._default_client(128)
        assert client.meta.config.max_pool_connections == 128


class TestOneClientForEveryThread:
    """`botocore.Session.create_client` is not thread-safe.

    A client per thread also gives each thread its own connection pool, so 64
    threads would hold up to 4,096 sockets against the 64 the config asks for.
    The client is thread-safe for calls, which is the part staging needs.
    """

    def test_the_factory_is_called_once(self, real_items, tmp_path):
        made = []

        def factory(pool_size):
            made.append(pool_size)
            return FakeS3()

        staging.stage_scenes(
            items := real_items[0],
            range(3),
            tmp_path,
            threads=8,
            client_factory=factory,
        )
        assert items  # the fixture supplied something to fetch
        assert made == [8], "one client, built with the pool it will use"


class TestDiskGuard:
    def test_it_reserves_nothing_for_what_is_already_staged(self, real_items, tmp_path):
        """A rerun must not be refused by the volume already holding the data.

        The guard reserves the whole manifest before the first GET. Once the
        fetch skips what is present, counting those objects would refuse the
        second traversal on exactly the volume that makes it free.
        """
        items, _ = real_items
        manifest = staging.staging_manifest(items, range(len(items)))
        cold = staging.estimated_bytes(manifest, tmp_path)
        assert cold == staging.estimated_bytes(manifest)

        for item_id, band, href in manifest:
            key = staging.split_s3_uri(href)[1]
            dest = staging.staged_path(tmp_path, item_id, band, key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"staged")

        assert staging.estimated_bytes(manifest, tmp_path) == 0
        # And the guard passes on a volume with nothing free, because the
        # objects are on it already.
        assert staging.disk_guard(manifest, tmp_path) == 0

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

        # Both figures, the escape, and the one route it must not offer.
        # Reading from S3 is not a fallback: the unstaged path is gone, and
        # the message says so rather than leaving a reader to wonder.
        message = str(exc.value)
        assert f"{need / 1024**3:.1f} GiB" in message
        assert f"{len(manifest):,} objects" in message
        assert "--stage-dir" in message
        assert "no unstaged path" in message
        assert "--no-stage" not in message

    def test_no_entry_point_can_be_told_to_skip_staging(self):
        """There is no unstaged path, and there is no flag that makes one.

        Unstaged reads cost about 739 requests per object against one, because
        roughly 155 shards open each scene and every open is 4.77 requests.
        MEASURED across the fleet inventory: 4.2 billion GETs against 5.8
        million, which is $1,681 against $2.32. It is also about twice as slow.
        MEASURED in PR #7, commit `c771f27`, over four shards run twice in one
        process: 11 s staged against 21 s unstaged.

        A flag worth $1,679 and half the wall clock is not a flag. This asserts
        the parsers reject it rather than trusting a reviewer to notice one
        coming back.
        """
        import shard_lst_p95
        import tile_prep

        for module in (shard_lst_p95, tile_prep):
            with pytest.raises(SystemExit):
                module.parse_args(["--tile", "S30W065", "--no-stage"])

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

    def test_a_directory_this_run_did_not_create_survives(self, tmp_path):
        # FINDINGS tells an operator to point --stage-dir at an NVMe mount.
        # One level up from the documented path is the whole volume, and this
        # removes a tree.
        target = tmp_path / "nvme"
        target.mkdir()
        (target / "somebody-elses.db").write_bytes(b"x")

        staging.cleanup(target, owned=False)

        assert (target / "somebody-elses.db").exists()

    def test_staging_reports_whether_it_created_the_directory(
        self, real_items, tmp_path
    ):
        items, _ = real_items
        fresh = tmp_path / "fresh"
        report = staging.stage_scenes(
            items, [0], fresh, threads=2, client_factory=lambda _n: FakeS3()
        )
        assert report["owns_stage_dir"] is True

        occupied = tmp_path / "occupied"
        occupied.mkdir()
        (occupied / "prior.txt").write_bytes(b"x")
        report = staging.stage_scenes(
            items, [1], occupied, threads=2, client_factory=lambda _n: FakeS3()
        )
        assert report["owns_stage_dir"] is False

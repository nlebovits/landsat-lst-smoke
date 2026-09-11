"""The production path must not reach a catalogue, even if one is available.

Removing the per-tile Earth Search query only pays if it cannot come back. A
fallback that quietly re-opens the catalogue when the inventory looks wrong
turns a configuration mistake into hundreds of machines doing the slow thing,
and it would pass every other test in this suite.

So two checks. A static one, that the runtime modules do not name a STAC
client. A live one, that the inventory read and the shard plan finish with
every socket in the process blocked.

What the live check covers, precisely: everything from the manifest gate
through `process_shard`, which is the whole run. Staging closed the gap that
used to end this list at `items_for_shard`. The run now has two phases, a stage
phase that opens S3 and a shard phase that reads local files, so the composite
itself can be computed under the socket guard rather than asserted to be
reachable without one. `tests/test_load_parity.py` still covers the unstaged
read, against the bucket, under the `s3` marker.

These tests run against `artifacts/inventory_slice.parquet`, which is committed.
They used to run against the 167 MB build, which is gitignored, so on a clean
checkout every one of them skipped and the guarantee went unchecked in CI.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import masks  # noqa: E402
import shard_lst_p95  # noqa: E402
import staging  # noqa: E402
import tile_inventory  # noqa: E402
from land_tiles import tile_bounds  # noqa: E402

#: The tile the offline tests read. It is in the committed slice.
TILE = "S30W065"

#: The modules a VM imports to turn a tile into a composite.
RUNTIME_MODULES = (
    "shard_lst_p95",
    "tile_inventory",
    "land_tiles",
    "masks",
    "aster_ged",
    "staging",
    "item_table",
    "memory_sampler",
)

#: Modules that mean the mask is about to fetch its own inputs. `land_tiles`
#: and `aster_ged` reach these on purpose, at build time, and `masks` reads the
#: artifacts they write instead.
FETCH_NAMES = ("earthaccess", "requests", "urllib")

#: Names that mean a catalogue is in reach.
STAC_NAMES = ("pystac_client", "stac_reference", "earth-search.aws", "Client.open")


class NetworkBlocked(AssertionError):
    """Raised in place of any outbound connection."""


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly on any socket connection, from anywhere in the process."""

    def refuse(*args, **kwargs):
        msg = f"the runtime opened a network connection: {args!r}"
        raise NetworkBlocked(msg)

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return refuse


class TestNoCatalogueInTheRuntime:
    @pytest.mark.parametrize("name", RUNTIME_MODULES)
    def test_module_does_not_name_a_stac_client(self, name):
        source = (ROOT / f"{name}.py").read_text()
        for needle in STAC_NAMES:
            assert needle not in source, (
                f"{name}.py names {needle!r}. The production path must not "
                f"reach a catalogue; put it in stac_reference.py instead."
            )

    def test_the_shard_runtime_has_no_search_function(self):
        assert not hasattr(shard_lst_p95, "search_items")

    def test_the_reference_search_is_not_imported_by_the_runtime(self):
        """stac_reference may exist. It may not be reachable from here."""
        for name in RUNTIME_MODULES:
            module = sys.modules[name]
            for attr in vars(module).values():
                mod = getattr(attr, "__module__", "")
                assert mod != "stac_reference", (
                    f"{name} imported {attr!r} from stac_reference"
                )

    def test_the_reference_module_still_exists_for_the_oracle(self):
        """It is kept on purpose, for the parity tests and the dry runs."""
        import stac_reference

        assert hasattr(stac_reference, "search_items")

    def test_the_mask_never_calls_its_own_download(self):
        """`masks` reads two artifacts. It fetches neither of them.

        `load_land_polygons` downloads Natural Earth on a cold cache, and
        `aster_ged.fetch_granules` needs a NASA Earthdata Login. Both belong to
        the build, which runs once on a laptop. A mask that reached either
        would put a download inside every one of 895 runs.

        The check walks the syntax tree rather than the text, because the
        module explains in prose why it does not call these, and a substring
        search cannot tell an explanation from a call.
        """
        import ast

        forbidden = {"load_land_polygons", "fetch_granules", *FETCH_NAMES}
        tree = ast.parse((ROOT / "masks.py").read_text())
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Import):
                used.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                used.add((node.module or "").split(".")[0])
                used.update(alias.name for alias in node.names)
        offenders = sorted(used & forbidden)
        assert not offenders, (
            f"masks.py calls {offenders!r}. The pixel mask reads artifacts, "
            f"so that a run needs no credentials and no network."
        )

    def test_the_runtime_does_not_import_the_earthdata_client(self):
        """`earthaccess` belongs to `aster_ged`'s build half, behind a call."""
        assert "earthaccess" not in sys.modules


class TestTheMaskBuildsOffline:
    """The mask runs between the gather and the summary, inside the run.

    This module's docstring claims the run is offline from the manifest gate
    through `process_shard`. The mask sits inside that span and reads two files
    that both have network-fetching builders behind them.
    """

    def test_the_land_rule_rasterises_with_every_socket_blocked(
        self, no_network, land_geometry
    ):
        land = masks.land_mask(tile_bounds(TILE), 100, land_geometry)
        assert land.any()

    def test_the_emissivity_rule_reads_with_every_socket_blocked(
        self, no_network, numobs_artifact
    ):
        gap = masks.emissivity_gap(tile_bounds(TILE), 100, numobs_artifact)
        assert not gap.any()

    def test_the_whole_mask_builds_with_every_socket_blocked(
        self, no_network, numobs_artifact, land_geometry
    ):
        keep, gap, counts = masks.output_mask(
            tile_bounds(TILE),
            100,
            numobs_uri=numobs_artifact,
            land_geometry_uri=land_geometry,
        )
        assert counts["pixels_kept"] == int(keep.sum()) > 0
        assert counts["pixels_emissivity_gap"] == int(gap.sum())


class TestRuntimeWorksOffline:
    def test_the_guard_actually_blocks(self, no_network):
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("earth-search.aws.element84.com", 443))

    def test_a_tile_read_succeeds_with_the_network_blocked(
        self, no_network, slice_artifact
    ):
        items, boxes = tile_inventory.items_for_tile(
            slice_artifact, TILE, bounds=tile_bounds(TILE)
        )
        assert len(items) == len(boxes) > 0
        assert items[0]["assets"]["qa_pixel"]["href"].startswith("s3://usgs-landsat")

    def test_the_manifest_gate_runs_offline(self, no_network, slice_artifact):
        from usgs_inventory import INVENTORY_SCHEMA_VERSION

        manifest = tile_inventory.read_manifest(slice_artifact)
        tile_inventory.check_manifest(
            manifest,
            start=manifest["start"],
            end=manifest["end"],
            platforms=manifest["platforms"],
            cloud_cover_lt=manifest["cloud_cover_lt"],
            schema_version=INVENTORY_SCHEMA_VERSION,
        )

    def test_the_whole_load_path_runs_offline(self, no_network, slice_artifact):
        """`load_tile_items` is what main() calls. It reads and checks."""
        args = shard_lst_p95.parse_args(
            ["--tile", TILE, "--inventory-uri", str(slice_artifact)]
        )
        items, boxes, prov = shard_lst_p95.load_tile_items(args, TILE)
        assert len(items) == len(boxes) > 0
        assert prov["source_url"].startswith("https://landsat.usgs.gov/")
        assert len(prov["source_sha256"]) == 64

    def test_the_shard_plan_runs_offline(self, no_network, slice_artifact):
        """Everything that decides what to read, up to the read itself.

        `items_for_shard` is the last step before the read. Running it under
        the socket guard covers the whole decision path: the manifest gate, the
        row-group read, `build_item`, the shard geometry, and the per-shard
        overlap filter. `TestTheComputePathIsOffline` carries it through the
        read as well.
        """
        args = shard_lst_p95.parse_args(
            ["--tile", TILE, "--inventory-uri", str(slice_artifact)]
        )
        bbox, tile_id = shard_lst_p95.resolve_area(args)
        assert tile_id == TILE
        items, boxes, _ = shard_lst_p95.load_tile_items(args, tile_id)

        shards, _, _ = shard_lst_p95.plan_shards(bbox, args.pixels_per_degree, 512)
        assert shards
        hit = [len(shard_lst_p95.items_for_shard(sh, boxes)) for sh in shards]
        assert sum(hit) > 0, "no shard overlapped any scene in the slice"
        assert max(hit) <= len(items)


#: A small EPSG:4326 raster, so the synthetic scenes need no reprojection and
#: the test measures the read path rather than a warp.
SCENE_RES = 1 / 3600
SCENE_PX = 64
SCENE_WEST, SCENE_SOUTH = -64.0, -34.0
#: DN 40000 decodes to 12.57 C, which is inside the trusted range.
SCENE_THERMAL_DN = 40000
#: QA_PIXEL bit 6, "clear". None of the excluded bits 1 to 5 are set.
SCENE_QA_CLEAR = 0b1000000


class FileBackedS3:
    """Serves bytes from a dict of key to payload. Counts every call."""

    def __init__(self, blobs):
        self.blobs = blobs
        self.calls = []

    def get_object(self, *, Bucket, Key, RequestPayer=None):  # noqa: N803
        import io

        self.calls.append((Bucket, Key))
        payload = self.blobs[Key]
        return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}


def _write_band(path, value):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SCENE_PX,
        width=SCENE_PX,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(
            SCENE_WEST, SCENE_SOUTH + SCENE_PX * SCENE_RES, SCENE_RES, SCENE_RES
        ),
    ) as ds:
        ds.write(np.full((SCENE_PX, SCENE_PX), value, "uint16"), 1)
    return path.read_bytes()


@pytest.fixture
def scenes_in_a_bucket(tmp_path):
    """Three scenes as real GeoTIFFs, addressed the way the inventory does.

    The items go through `tile_inventory.build_item`, so the asset shape is the
    production one. Only the projection fields are overridden, to describe a
    64 px raster rather than a whole Landsat scene.

    Returns:
        The item dicts, the shard covering them, and the object store contents.
    """
    from conftest import make_row
    from tile_inventory import build_item

    source = tmp_path / "source"
    source.mkdir()
    north = SCENE_SOUTH + SCENE_PX * SCENE_RES
    east = SCENE_WEST + SCENE_PX * SCENE_RES

    items, blobs = [], {}
    for i in range(3):
        keys = {
            "lwir11": f"scene/{i}/ST_B10.TIF",
            "qa_pixel": f"scene/{i}/QA_PIXEL.TIF",
        }
        blobs[keys["lwir11"]] = _write_band(source / f"t{i}.TIF", SCENE_THERMAL_DN)
        blobs[keys["qa_pixel"]] = _write_band(source / f"q{i}.TIF", SCENE_QA_CLEAR)

        row = make_row(TILE, i)
        row["proj_epsg"] = 4326
        row["proj_shape_y"] = row["proj_shape_x"] = SCENE_PX
        row["thermal_href"] = f"s3://usgs-landsat/{keys['lwir11']}"
        row["qa_href"] = f"s3://usgs-landsat/{keys['qa_pixel']}"
        row["bbox_west"], row["bbox_south"] = SCENE_WEST, SCENE_SOUTH
        row["bbox_east"], row["bbox_north"] = east, north
        for k in range(4):
            row[f"corner_lon{k}"], row[f"corner_lat{k}"] = SCENE_WEST, SCENE_SOUTH

        item = build_item(row)
        item["properties"]["proj:transform"] = [
            SCENE_RES,
            0.0,
            SCENE_WEST,
            0.0,
            -SCENE_RES,
            north,
        ]
        items.append(item)

    half = SCENE_PX // 2
    shard = shard_lst_p95.Shard(
        0,
        0,
        0,
        0,
        half,
        half,
        (
            SCENE_WEST,
            SCENE_SOUTH,
            SCENE_WEST + half * SCENE_RES,
            SCENE_SOUTH + half * SCENE_RES,
        ),
    )
    return items, shard, blobs


class TestTheComputePathIsOffline:
    """Stage, then compute a composite with every socket in the process shut.

    This is what staging buys beyond the money. The read used to reach S3
    through GDAL, in C, where a patched `socket.socket` cannot see it, so the
    offline guarantee had to stop at the last Python call before the read.
    Staging splits the run in two: one phase opens the bucket, and the phase
    that does the work reads local files. So the second phase can be run under
    the guard rather than reasoned about.
    """

    def test_staging_then_a_composite_needs_no_socket(
        self, no_network, scenes_in_a_bucket, tmp_path
    ):
        import numpy as np

        items, shard, blobs = scenes_in_a_bucket
        fake = FileBackedS3(blobs)

        report = staging.stage_scenes(
            items,
            range(len(items)),
            tmp_path / "stage",
            threads=2,
            client_factory=lambda _n: fake,
        )

        # Three scenes, two bands, one GET each. Not one per shard that reads
        # them, which is the entire point of the phase.
        assert report["objects"] == 6
        assert report["get_requests"] == 6
        assert all(
            not item["assets"][band]["href"].startswith("s3://")
            for item in items
            for band in ("lwir11", "qa_pixel")
        )

        out = shard_lst_p95.process_shard(shard, items, "EPSG:4326", SCENE_RES)

        assert out["n_scenes"] == 3
        assert out["lst_p95"].shape == (SCENE_PX // 2, SCENE_PX // 2)
        # Every pixel is clear in every scene, so nothing is nodata and the
        # monthly counts hold all three observations.
        assert not np.any(out["lst_p95"] == shard_lst_p95.LST_NODATA_DN)
        assert int(out["qa_count"].sum(axis=0).max()) == 3

    def test_the_table_path_gives_the_same_pixels(
        self, no_network, scenes_in_a_bucket, tmp_path
    ):
        """What `shard_task` changed, checked against the arrays.

        A shard used to receive its item dicts in the task message. It now
        receives a path and a list of positions, and the dicts make a round
        trip through JSON on the way. That round trip turns each geometry
        corner from a tuple into a list, which is the one thing it alters, and
        this is the check that the alteration reaches no pixel.
        """
        import numpy as np

        import item_table

        items, shard, blobs = scenes_in_a_bucket
        staging.stage_scenes(
            items,
            range(len(items)),
            tmp_path / "stage",
            threads=2,
            client_factory=lambda _n: FileBackedS3(blobs),
        )

        direct = shard_lst_p95.process_shard(shard, items, "EPSG:4326", SCENE_RES)
        report = item_table.write(tmp_path / "item-table.json", items)
        through = shard_lst_p95.shard_task(
            shard, report["path"], list(range(len(items))), "EPSG:4326", SCENE_RES
        )

        assert np.array_equal(direct["lst_p95"], through["lst_p95"])
        assert np.array_equal(direct["qa_count"], through["qa_count"])
        assert direct["n_scenes"] == through["n_scenes"]

    def test_the_staged_run_reads_no_object_twice(
        self, no_network, scenes_in_a_bucket, tmp_path
    ):
        """Two shards over the same scenes, and still six GETs.

        The unstaged path pays 4.77 requests per shard-scene read, so two
        shards over three scenes would cost about 28 requests instead of six.
        """
        items, shard, blobs = scenes_in_a_bucket
        fake = FileBackedS3(blobs)

        staging.stage_scenes(
            items,
            [0, 1, 2, 0, 1, 2],
            tmp_path / "stage",
            threads=2,
            client_factory=lambda _n: fake,
        )

        assert len(fake.calls) == 6
        for _ in range(2):
            shard_lst_p95.process_shard(shard, items, "EPSG:4326", SCENE_RES)
        assert len(fake.calls) == 6


class TestFailureIsLoud:
    def test_a_missing_inventory_stops_the_run(self, tmp_path, no_network):
        args = shard_lst_p95.parse_args(
            ["--tile", TILE, "--inventory-uri", str(tmp_path / "absent.parquet")]
        )
        with pytest.raises(tile_inventory.InventoryError, match="no inventory at"):
            shard_lst_p95.load_tile_items(args, TILE)

    def test_a_window_mismatch_stops_the_run(self, no_network, slice_artifact):
        args = shard_lst_p95.parse_args(
            [
                "--tile",
                TILE,
                "--inventory-uri",
                str(slice_artifact),
                "--start",
                "2020-01-01",
            ]
        )
        with pytest.raises(tile_inventory.InventoryError, match="start"):
            shard_lst_p95.load_tile_items(args, TILE)

    def test_a_real_run_without_a_tile_is_refused(self, slice_artifact):
        """A bbox cannot address the inventory, so it cannot start a run."""
        args = shard_lst_p95.parse_args(
            ["--bbox=-65,-35,-60,-30", "--inventory-uri", str(slice_artifact)]
        )
        bbox, tile_id = shard_lst_p95.resolve_area(args)
        assert tile_id is None
        assert bbox == (-65.0, -35.0, -60.0, -30.0)

    def test_planetary_computer_is_refused_rather_than_misconfigured(self):
        """The hrefs are requester-pays, and only earth-search sets the header.

        This used to configure a read environment without `AWS_REQUEST_PAYER`
        and let every read fail against the bucket.
        """
        with pytest.raises(SystemExit, match="requester-pays"):
            shard_lst_p95.configure_read_env("planetary-computer")

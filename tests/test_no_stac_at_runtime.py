"""The production path must not reach a catalogue, even if one is available.

Removing the per-tile Earth Search query only pays if it cannot come back. A
fallback that quietly re-opens the catalogue when the inventory looks wrong
turns a configuration mistake into hundreds of machines doing the slow thing,
and it would pass every other test in this suite.

So two checks. A static one, that the runtime modules do not name a STAC
client. A live one, that the inventory read and the shard plan finish with
every socket in the process blocked.

What the live check covers, precisely: everything from the manifest gate
through `items_for_shard`, which is the whole of the run that decides what to
read. It stops short of the read itself. `stac_load` reaches S3 through GDAL,
in C, and a patched `socket.socket` cannot speak for it. `tests/test_load_parity.py`
covers that path, against the bucket, under the `s3` marker.

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

import shard_lst_p95  # noqa: E402
import tile_inventory  # noqa: E402
from land_tiles import tile_bounds  # noqa: E402

#: The tile the offline tests read. It is in the committed slice.
TILE = "S30W065"

#: The modules a VM imports to turn a tile into a composite.
RUNTIME_MODULES = ("shard_lst_p95", "tile_inventory", "land_tiles")

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

        `items_for_shard` is the last step before `stac_load`. Running it under
        the socket guard covers the whole decision path: the manifest gate, the
        row-group read, `build_item`, the shard geometry, and the per-shard
        overlap filter. What remains is GDAL, which is C and reaches S3 without
        touching `socket.socket`.
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

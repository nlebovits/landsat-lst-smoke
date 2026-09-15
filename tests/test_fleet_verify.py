"""The check that a published raster is whole, proved against a broken one.

`cog_catalog._verify_cog` runs on the instance, against a local path, at the
moment the raster is written. It catches a bad writer. The upload happens
afterwards, and an upload cut part way leaves a file whose header is complete
and whose tail is missing. Every header check passes on that file.

So these tests build a real COG, cut its tail off, and require the checker to
notice. A checker that passes a truncated file would let the catalog publish
one.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from lst.cog_catalog import BLOCK_SIZE
from lst.fleet import verify

SIDE = BLOCK_SIZE * 3


@pytest.fixture
def cog(tmp_path):
    """A tiled raster with overviews, large enough to need them."""
    path = tmp_path / "lst_p95.tif"
    data = np.arange(SIDE * SIDE, dtype="uint16").reshape(SIDE, SIDE) % 60000
    profile = {
        "driver": "GTiff",
        "height": SIDE,
        "width": SIDE,
        "count": 1,
        "dtype": "uint16",
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "compress": "deflate",
    }
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.build_overviews([2, 4])
    return path


class TestAWholeRasterPasses:
    def test_the_written_file_reads_to_its_last_block(self, cog):
        assert verify.check_raster(str(cog)) is None


class TestATruncatedRasterFails:
    """The failure this checker exists for. The header survives; the tail does not."""

    @pytest.mark.parametrize("keep", [0.3, 0.6, 0.9])
    def test_a_file_missing_its_tail_is_caught(self, cog, keep):
        size = cog.stat().st_size
        with cog.open("r+b") as fh:
            fh.truncate(int(size * keep))
        assert verify.check_raster(str(cog)) is not None

    def test_the_fault_names_something_a_reader_can_act_on(self, cog):
        with cog.open("r+b") as fh:
            fh.truncate(cog.stat().st_size // 2)
        fault = verify.check_raster(str(cog))
        assert fault
        assert "\n" not in fault
        assert len(fault) <= 200


class TestStructureFaults:
    """A writer fault, reported differently from an upload fault."""

    def test_an_untiled_raster_is_caught(self, tmp_path):
        path = tmp_path / "stripey.tif"
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=SIDE,
                width=SIDE,
                count=1,
                dtype="uint16",
            ) as dst:
                dst.write(np.zeros((SIDE, SIDE), dtype="uint16"), 1)
        fault = verify.check_raster(str(path))
        assert fault and "internal tiles" in fault

    def test_a_big_raster_without_overviews_is_caught(self, tmp_path):
        path = tmp_path / "flat.tif"
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=SIDE,
                width=SIDE,
                count=1,
                dtype="uint16",
                tiled=True,
                blockxsize=BLOCK_SIZE,
                blockysize=BLOCK_SIZE,
            ) as dst:
                dst.write(np.zeros((SIDE, SIDE), dtype="uint16"), 1)
        fault = verify.check_raster(str(path))
        assert fault and "overviews" in fault

    def test_a_missing_file_is_caught(self, tmp_path):
        assert verify.check_raster(str(tmp_path / "absent.tif")) is not None


class TestAMissingTileIsAFault:
    """Reporting only on what was found would call an absent tile clean."""

    def test_a_tile_the_bucket_lacks_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            verify, "find_tiles", lambda runs, coll: {"S30W065": "s3://b/x/S30W065"}
        )
        monkeypatch.setattr(verify, "check_tile", lambda t, p: (t, []))
        faults = verify.verify(
            "s3://b/runs",
            "lst-p95-2021-2025",
            tiles=["S30W065", "S50W070"],
            say=lambda *a: None,
        )
        assert faults == {"S50W070": ["no item in the bucket"]}

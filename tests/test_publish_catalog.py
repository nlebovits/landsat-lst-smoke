"""The publish step that rewrites what a published item claims.

`lst:land_pixels` reported the processing mask until 2026-09-15. That mask is
Natural Earth land grown by 25 km so a coastal scene is not cut at the
waterline, and it reaches open sea, so every share of land on a coastal item
divided by a denominator that included water. MEASURED at 3600 pixels per
degree, `S40W065` is 73,254,945 pixels of mask and 45,407,126 pixels of land.

Five tiles were already published under the old meaning. `recount` fixes them
from the rasters that are already there, and the one thing it must never do is
touch a raster. So the test that matters here is not an arithmetic test. It is
the one that hashes both COGs before and after.

Two more failures this guards against, both of which return a number rather
than an error:

The grid. A recount rasterises its masks and counts pixels inside them. If it
ever derived the grid from anywhere but the raster it is counting, both masks
would still burn, both counts would still look plausible, and they would
describe different ground.

Running twice. `recount` appends a lineage sentence and rewrites properties.
A second run has to be a no-op, or every rerun grows the item.

`aws s3` is replaced by a local tree. The step's own S3 calls are `ls` and `cp`
with no options this test cares about, and what is under test is the item
rewrite, the arithmetic, and the raster's stillness.

The published tree is module-scoped, because writing it composites a 1,800 px
tile and copying it per test would put minutes on the file. So the tests run
against an item a previous test may already have corrected. That is safe
because a recount is idempotent, which `TestRecountIsSafeToRepeat` asserts
rather than assumes, and because every claim here is stated against the
fixture's own recorded inputs rather than against the item's current state.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cog_catalog  # noqa: E402
import masks  # noqa: E402
import publish_catalog  # noqa: E402
from conftest import needs_strict_land_geometry, write_numobs  # noqa: E402
from land_tiles import tile_bounds  # noqa: E402
from lst_qa import LST_NODATA_DN, encode_celsius  # noqa: E402

#: The New Jersey coast. MEASURED at 1/360 degree, 4.43% of the tile is land,
#: inside a processing mask that reaches well into the Atlantic. The tile has to
#: be coastal or the two denominators coincide and nothing is under test.
TILE = "N40W075"

#: Coarse enough to composite a synthetic tile in a test, and a whole number of
#: degrees, so the GED cells land on pixel boundaries.
PPD = 360

COLLECTION = "lst-p95-2021-2025"

pytestmark = [needs_strict_land_geometry, pytest.mark.timeout(300)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalS3:
    """`aws s3 ls` and `cp` against a directory, for the recount loop.

    Keeps a call log, so a test can assert that no raster key was ever copied.
    """

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[str, ...]] = []

    def path_for(self, uri: str) -> Path:
        return self.root / uri.removeprefix("s3://")

    def __call__(self, *args: str, capture: bool = True) -> str:
        self.calls.append(args)
        if args[0] == "ls":
            # `aws s3 ls --recursive` prints keys without the bucket, which is
            # what the caller then puts back with `s3://{bucket}/{key}`.
            uri = args[-1]
            bucket = uri.removeprefix("s3://").split("/", 1)[0]
            base = self.path_for(uri)
            lines = []
            for found in sorted(base.rglob("*")):
                if found.is_file():
                    key = found.relative_to(self.root / bucket).as_posix()
                    lines.append(f"2026-09-15 00:00:00 {found.stat().st_size} {key}")
            return "\n".join(lines) + "\n"
        if args[0] == "cp":
            source, target = args[1], args[2]
            src = self.path_for(source) if source.startswith("s3://") else Path(source)
            dst = self.path_for(target) if target.startswith("s3://") else Path(target)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            return ""
        raise AssertionError(f"unexpected aws s3 call: {args}")


class Args:
    """The parsed CLI namespace `cmd_recount` reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


@pytest.fixture(scope="module")
def published(tmp_path_factory, land_geometry, strict_land_geometry):
    """One tile published the way the September 2026 run published five.

    Written by `cog_catalog.write_catalog`, then handed a coverage block built
    the old way: the processing mask under the name `land_pixels`, and every
    surviving pixel of the raster under `valid_pixels`. That is the item this
    step exists to correct.
    """
    root = tmp_path_factory.mktemp("published")
    bucket = root / "bucket"
    prefix = f"catalog/{COLLECTION}"
    bbox = tile_bounds(TILE)
    height, width = masks.raster_shape(bbox, PPD)

    numobs = write_numobs(root / "aster_numobs.tif")
    keep, _gap, mask_counts = masks.output_mask(
        bbox, PPD, numobs_uri=numobs, land_geometry_uri=land_geometry
    )

    rng = np.random.default_rng(20260915)
    celsius = rng.uniform(10.0, 40.0, (height, width)).astype("float32")
    # A band of empty land across the tile, so `empty_land_pixels` is not zero.
    celsius[: height // 10] = np.nan
    lst = encode_celsius(celsius)
    qa = np.zeros((12, height, width), dtype="uint8")
    qa[0] = 7
    masks.apply_output_mask(lst, qa, keep)

    written = int((lst != LST_NODATA_DN).sum())
    meta = {
        "raster": [height, width],
        "bbox": list(bbox),
        "crs": "EPSG:4326",
        "pixels_per_degree": PPD,
        "start": "2021-01-01",
        "end": "2025-12-31T23:59:59Z",
        "mask_rule": {
            "gap_buffer_cells": masks.GAP_BUFFER_CELLS,
            "land_geometry_sha256": masks.geometry_checksum(land_geometry),
        },
        # The old meaning: the mask under the name land, and the whole raster's
        # surviving pixels against it.
        "coverage": {
            "land_pixels": mask_counts["pixels_kept"],
            "valid_pixels": written,
            "empty_land_pixels": mask_counts["pixels_kept"] - written,
            "valid_fraction": written / mask_counts["pixels_kept"],
            "ged_gap_fraction": (
                mask_counts["pixels_emissivity_gap_on_land"]
                / mask_counts["pixels_kept"]
            ),
        },
    }
    cog_catalog.write_catalog(
        bucket / "catalog", lst, qa, meta, collection_id=COLLECTION
    )
    return {
        "root": root,
        "bucket": bucket,
        "prefix": prefix,
        "item": bucket / prefix / TILE / f"{TILE}.json",
        "numobs": numobs,
        "mask_counts": mask_counts,
        "written": written,
    }


@pytest.fixture
def run_recount(monkeypatch, published, land_geometry, strict_land_geometry):
    """Drive `cmd_recount` against the local tree.

    `monkeypatch` puts both names back afterwards, so a failure in one test
    cannot leave the module pointed at a temporary directory for the next.

    Returns:
        A callable giving `(exit code, the LocalS3 that served it)`.
    """

    def drive(*, dry_run=False, tile=None):
        s3 = LocalS3(published["root"])
        monkeypatch.setattr(publish_catalog, "s3", s3)
        monkeypatch.setattr(
            publish_catalog,
            "lst_uri_for",
            lambda bucket, prefix, tile: str(
                published["root"] / bucket / prefix / tile / "lst_p95.tif"
            ),
        )
        args = Args(
            dest="s3://bucket/catalog",
            collection=COLLECTION,
            numobs=published["numobs"],
            land_geometry_uri=land_geometry,
            strict_land_geometry_uri=strict_land_geometry,
            gap_buffer_cells=masks.GAP_BUFFER_CELLS,
            tile=tile,
            dry_run=dry_run,
        )
        return publish_catalog.cmd_recount(args), s3

    return drive


class TestRecountLeavesTheRastersAlone:
    """The claim the whole step rests on.

    A metadata fix that rewrote a 471 MB COG would be a republication of the
    data, with a new checksum on every item asset and a new download for every
    consumer. Hashing both files is the only assertion that can tell.
    """

    def test_no_raster_byte_changes(self, published, run_recount):
        rasters = [
            published["bucket"] / published["prefix"] / TILE / name
            for name in ("lst_p95.tif", "qa_count.tif")
        ]
        before = [digest(path) for path in rasters]
        code, _ = run_recount()
        assert code == 0
        assert [digest(path) for path in rasters] == before

    def test_no_raster_key_is_ever_copied(self, run_recount):
        """Read the rasters, write none. GDAL reads them out of band."""
        _code, s3 = run_recount()
        copied = [args for args in s3.calls if args[0] == "cp"]
        assert copied
        for args in copied:
            assert not any(arg.endswith(".tif") for arg in args)


class TestRecountCorrectsTheDenominator:
    def recounted(self, published, run_recount):
        code, _ = run_recount()
        assert code == 0
        return json.loads(published["item"].read_text())["properties"]

    def test_land_is_smaller_than_the_mask_it_replaced(self, published, run_recount):
        """The old denominator is compared against the fixture, not the item.

        The published tree is module-scoped and a recount is idempotent, so by
        the time this runs the item may already be corrected. What the old
        `lst:land_pixels` was is `pixels_kept`, which is what the fixture wrote
        it from.
        """
        was_land = published["mask_counts"]["pixels_kept"]
        props = self.recounted(published, run_recount)
        assert props["lst:land_pixels"] < was_land
        assert props["lst:processing_mask_pixels"] == was_land

    def test_the_three_equations_hold(self, published, run_recount):
        props = self.recounted(published, run_recount)
        assert (
            props["lst:land_pixels"] + props["lst:coastal_buffer_pixels"]
            == props["lst:processing_mask_pixels"]
        )
        assert (
            props["lst:land_pixels"] - props["lst:valid_pixels"]
            == props["lst:empty_land_pixels"]
        )
        assert (
            props["lst:valid_pixels"] + props["lst:coastal_buffer_valid_pixels"]
            == published["written"]
        )

    def test_the_land_count_is_the_geometry_not_the_raster(
        self, published, run_recount, land_geometry, strict_land_geometry
    ):
        """The recount has to agree with `land_split` on the same bbox.

        Reading the grid off the COG is what makes that true. A recount that
        guessed the grid would return a plausible land count for a different
        footprint.
        """
        props = self.recounted(published, run_recount)
        _, _, split = masks.land_split(
            tile_bounds(TILE),
            PPD,
            land_geometry_uri=land_geometry,
            strict_land_geometry_uri=strict_land_geometry,
        )
        assert props["lst:land_pixels"] == split["pixels_strict_land"]
        assert props["lst:coastal_buffer_pixels"] == split["pixels_coastal_buffer"]

    def test_the_lineage_names_the_geometry_that_divided(
        self, published, run_recount, strict_land_geometry
    ):
        props = self.recounted(published, run_recount)
        sentence = cog_catalog.strict_land_sentence(
            masks.geometry_checksum(strict_land_geometry)
        )
        assert sentence in props["processing:lineage"]

    def test_it_stamps_the_item_as_updated(self, published, run_recount):
        before = json.loads(published["item"].read_text())["properties"]["updated"]
        props = self.recounted(published, run_recount)
        assert props["updated"] >= before


class TestRecountIsSafeToRepeat:
    def test_a_second_run_changes_nothing(self, published, run_recount):
        """An append that ran twice would say the same sentence twice."""
        run_recount()
        first = published["item"].read_text()
        code, _ = run_recount()
        assert code == 0
        after = json.loads(published["item"].read_text())
        assert json.loads(first)["properties"] == after["properties"]

    def test_a_dry_run_writes_nothing(self, published, run_recount):
        original = published["item"].read_text()
        code, _ = run_recount(dry_run=True)
        assert code == 0
        assert published["item"].read_text() == original


class TestTheGridComesFromTheRaster:
    def test_it_reads_the_bbox_and_the_grid_off_the_file(self, published):
        lst = published["bucket"] / published["prefix"] / TILE / "lst_p95.tif"
        bbox, ppd, height, width = publish_catalog.grid_of(str(lst))
        assert ppd == PPD
        assert (height, width) == masks.raster_shape(tile_bounds(TILE), PPD)
        assert [round(v, 9) for v in bbox] == [round(v, 9) for v in tile_bounds(TILE)]

    def test_a_raster_off_the_degree_grid_is_refused(self, tmp_path):
        """A grid of 7.5 pixels per degree cannot carry a rasterised mask.

        The masks are built from a bbox and a whole number of pixels per degree.
        A raster on any other grid has to stop the step rather than be rounded
        onto one.
        """
        import rasterio
        from rasterio.transform import from_origin

        path = tmp_path / "odd.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(0.0, 0.0, 1 / 7.5, 1 / 7.5),
        ) as dst:
            dst.write(np.zeros((8, 8), dtype="uint16"), 1)
        with pytest.raises(SystemExit, match="not a whole number"):
            publish_catalog.grid_of(str(path))


class TestTheTileFilter:
    """`--tile` keeps the first run against a real catalog cheap.

    A recount reads every `lst_p95` it touches, which is 33 MB to 471 MB per
    tile. An operator checking the numbers on one tile should not pay for five.
    """

    def test_a_named_tile_is_the_only_one_read(self, published, run_recount):
        code, s3 = run_recount(tile=[TILE])
        assert code == 0
        assert any(TILE in "".join(args) for args in s3.calls if args[0] == "cp")

    def test_a_tile_that_is_not_published_stops_the_step(
        self, published, run_recount, capsys
    ):
        """Silence would be worse: the step would report nothing to do and exit
        0, and an operator would read that as a recount that found no change."""
        code, _ = run_recount(tile=["S40W065"])
        assert code == 1
        assert "not published at the destination" in capsys.readouterr().err

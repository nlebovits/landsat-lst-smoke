"""The mask as a run applies it, not as a function returning an array.

`test_masks.py` checks the rules. This checks the four things only a whole run
can get wrong.

That the mask reaches the artifacts. The summary statistics and the published
COGs have to describe the same product, so the mask goes on inside the graph,
before anything reads a block, and a run that masked only its printout would
pass every test in `test_masks.py`.

That it stops early. The mask depends on the tile's bbox and two artifacts, and
on nothing the run computes. A tile it empties should cost no staged object and
no cluster, the way a tile with no thermal scene already costs none.

That it cannot be skipped by accident. `--no-output-mask` exists for measuring
the mask against its absence, and it is the only way to write a tile that
carries sea as temperature. A missing artifact stops the run instead of
producing an unmasked raster that looks finished.

That `lst_p95` and `qa_count` stay consistent. A `qa_count` above zero beside a
nodata temperature means the pixel had observations and lost them to the
reduction, which is a different fact from a masked pixel.

The runs here are rehearsals. `--rehearse` writes synthetic scenes to local
disk and reads no object, but `--tile` fixes the bbox on the production grid,
so the geography the mask sees is the real one. That is the whole reason the
rehearsal is masked like any other run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aster_ged  # noqa: E402
import masks  # noqa: E402
import shard_lst_p95  # noqa: E402
from conftest import needs_land_geometry, write_numobs  # noqa: E402
from land_tiles import tile_bounds  # noqa: E402
from lst_qa import LST_NODATA_DN  # noqa: E402

#: Interior South America. Every pixel is land, so the land rule removes
#: nothing and the emissivity rule is the only one that can act.
INLAND = "S30W065"
#: The New Jersey coast and a great deal of Atlantic. MEASURED at 1/360 degree:
#: 4.43% land.
COASTAL = "N40W075"

#: Small enough to rehearse in seconds, and a whole number of degrees, so the
#: GED cells still land on pixel boundaries.
PPD = 360

#: The dask block edge. 1,800 px over 100 gives 18 x 18 blocks.
CHUNK = 100

#: The collection the default 2021-2025 window earns. A tile's item id is its
#: tile id, because the grid names both from the same north and west edges.
COLLECTION_ID = "lst-p95-2021-2025"

pytestmark = [needs_land_geometry, pytest.mark.timeout(600)]


def rehearse(out_dir, tile, numobs_uri, land_geometry, *extra):
    """One masked rehearsal, and its summary."""
    argv = [
        "--tile",
        tile,
        "--rehearse",
        "20",
        "--numobs-uri",
        str(numobs_uri),
        "--land-geometry-uri",
        str(land_geometry),
        "--pixels-per-degree",
        str(PPD),
        "--chunk",
        str(CHUNK),
        "--workers",
        "2",
        "--threads-per-worker",
        "1",
        "--out-dir",
        str(out_dir),
        *extra,
    ]
    code = shard_lst_p95.main(argv)
    summary = json.loads((out_dir / "summary.json").read_text())
    return code, summary


def item_dir(out_dir: Path, tile: str) -> Path:
    return Path(out_dir) / "catalog" / COLLECTION_ID / tile


def published_bands(out_dir: Path, tile: str):
    """The two COGs the run published, as arrays.

    The pixels are read back from the files a consumer would download, not
    from anything the run kept in memory. That is the whole point: a mask that
    ran after the statistics and before the write, or the other way round,
    would leave the two disagreeing and only a read can tell.
    """
    import rasterio

    here = item_dir(out_dir, tile)
    with rasterio.open(here / "lst_p95.tif") as src:
        lst = src.read(1)
    with rasterio.open(here / "qa_count.tif") as src:
        qa = src.read()
    return lst, qa


@pytest.fixture(scope="module")
def masked_coastal(tmp_path_factory, numobs_artifact, land_geometry):
    """One masked rehearsal of the coastal tile, shared by every test that
    reads it. Each of these drives a real cluster over 324 blocks, so a
    function-scoped fixture would put minutes on the file for nothing.
    """
    out = tmp_path_factory.mktemp("masked")
    return rehearse(out, COASTAL, numobs_artifact, land_geometry) + (out,)


class TestAMaskedRun:
    @pytest.fixture(scope="module")
    def run(self, masked_coastal):
        return masked_coastal

    def test_it_succeeds(self, run):
        code, _, _ = run
        assert code == 0

    def test_the_summary_carries_the_mask(self, run):
        _, summary, _ = run
        assert summary["mask"] is not None
        for field in (
            "pixels_total",
            "pixels_water",
            "pixels_emissivity_gap",
            "pixels_kept",
            "valid_removed_by_mask",
            "valid_removed_by_water",
            "valid_removed_by_emissivity",
            "qa_count_pixels_zeroed",
        ):
            assert field in summary["mask"]

    def test_the_counts_add_up_to_the_raster(self, run):
        _, summary, _ = run
        mask = summary["mask"]
        height, width = summary["raster"]
        assert mask["pixels_total"] == height * width
        assert mask["pixels_kept"] + mask["pixels_water"] == mask["pixels_total"]

    def test_the_removals_add_up(self, run):
        _, summary, _ = run
        mask = summary["mask"]
        assert (
            mask["valid_removed_by_water"] + mask["valid_removed_by_emissivity"]
            == mask["valid_removed_by_mask"]
        )

    def test_it_names_the_artifacts_that_decided(self, run):
        # A masked tile is only reproducible if the two artifacts are named.
        # `summary.json` is the operator's record and keeps the paths, because
        # an operator rerunning one tile does want them.
        _, summary, _ = run
        assert summary["mask"]["numobs_uri"]
        assert summary["mask"]["land_geometry_uri"]
        assert summary["mask"]["aster_ged"]["short_name"] == "AG1km"

    def test_the_item_names_the_artifacts_by_identity_not_by_path(
        self, run, numobs_artifact, land_geometry
    ):
        # The item reaches the published catalog, and an absolute path on the
        # masking machine tells a reader of it nothing. It would also carry the
        # operator's home directory into a public file.
        _, _, out = run
        item = json.loads((item_dir(out, COASTAL) / f"{COASTAL}.json").read_text())
        text = json.dumps(item)
        assert str(numobs_artifact) not in text
        assert str(land_geometry) not in text
        lineage = item["properties"]["processing:lineage"]
        assert "Land geometry sha256" in lineage
        assert "AG1km" in lineage
        assert item["properties"]["sci:publications"][0]["doi"].startswith("10.5067/")

    def test_the_mask_reaches_the_published_raster(
        self, run, numobs_artifact, land_geometry
    ):
        _, _, out = run
        lst, _ = published_bands(out, COASTAL)
        keep, _, _ = masks.output_mask(
            tile_bounds(COASTAL),
            PPD,
            numobs_uri=numobs_artifact,
            land_geometry_uri=land_geometry,
        )
        assert not (lst[~keep] != LST_NODATA_DN).any()

    def test_the_statistics_describe_the_masked_product(self, run):
        # The mask goes on before anything is measured. A summary computed
        # first would report a valid fraction the raster does not have.
        _, summary, out = run
        lst, _ = published_bands(out, COASTAL)
        written = int((lst != LST_NODATA_DN).sum())
        assert summary["valid_fraction"] == pytest.approx(
            written / summary["mask"]["pixels_total"]
        )

    def test_the_water_rule_zeroes_both_bands_together(
        self, run, numobs_artifact, land_geometry
    ):
        """`lst_p95` and `qa_count` cannot disagree about a masked pixel.

        Water removes the temperature and the counts together, because the
        pixel was never this product's subject. The emissivity rule removes
        only the temperature, so the converse does not hold and is not
        asserted: a nodata pixel with counts is a hot retrieval inside the gap.
        """
        _, _, out = run
        lst, qa = published_bands(out, COASTAL)
        keep, _, _ = masks.output_mask(
            tile_bounds(COASTAL),
            PPD,
            numobs_uri=numobs_artifact,
            land_geometry_uri=land_geometry,
        )
        assert not qa.sum(axis=0)[~keep].any()
        assert (lst[qa.sum(axis=0) == 0] == LST_NODATA_DN).all()

    def test_the_run_says_it_is_a_whole_tile(self, run):
        # `output_mask` runs over the whole bbox, so `pixels_total` and its
        # siblings describe the tile rather than any part of it. The scope
        # says so, because summing them across machines would multiply it.
        _, summary, _ = run
        assert summary["mask"]["scope"] == "tile"


class TestAnInlandTileKeepsEverything:
    """The land rule must not eat a tile that is entirely land."""

    @pytest.fixture(scope="module")
    def run(self, tmp_path_factory, numobs_artifact, land_geometry):
        out = tmp_path_factory.mktemp("inland")
        return rehearse(out, INLAND, numobs_artifact, land_geometry) + (out,)

    def test_no_pixel_is_water(self, run):
        _, summary, _ = run
        assert summary["mask"]["pixels_water"] == 0

    def test_no_pixel_is_gap(self, run):
        _, summary, _ = run
        assert summary["mask"]["pixels_emissivity_gap"] == 0

    def test_nothing_is_removed(self, run):
        _, summary, _ = run
        assert summary["mask"]["valid_removed_by_mask"] == 0
        assert summary["mask"]["pixels_kept"] == summary["mask"]["pixels_total"]

    def test_no_count_is_zeroed(self, run):
        _, summary, _ = run
        assert summary["mask"]["qa_count_pixels_zeroed"] == 0


class TestATileOfNothingButGapStillPublishes:
    """The gap region is not a removal, so it cannot empty a tile.

    An earlier build dropped a tile whose every cell was a gap, on the reading
    that Collection 2 published nothing there. It publishes plenty: USGS
    interpolates emissivity across a gap cell and retrieves a temperature, and
    only the ones that fail upward come out.
    """

    @pytest.fixture(scope="module")
    def run(self, tmp_path_factory, land_geometry):
        out = tmp_path_factory.mktemp("allgap")
        gapped = write_numobs(
            out.parent / "all_gap.tif",
            value=8,
            gaps=[(-65.0, -35.0, -60.0, -30.0)],
        )
        return rehearse(out, INLAND, gapped, land_geometry) + (out,)

    def test_it_succeeds(self, run):
        code, _, _ = run
        assert code == 0

    def test_every_pixel_is_inside_the_gap_region(self, run):
        _, summary, _ = run
        mask = summary["mask"]
        assert mask["pixels_emissivity_gap"] == mask["pixels_total"]

    def test_the_tile_still_keeps_its_pixels(self, run):
        _, summary, _ = run
        mask = summary["mask"]
        assert mask["pixels_kept"] == mask["pixels_total"]
        assert summary.get("status") != "no-unmasked-pixels"

    def test_it_publishes_both_cogs(self, run):
        _, _, out = run
        assert (item_dir(out, INLAND) / "lst_p95.tif").is_file()
        assert (item_dir(out, INLAND) / "qa_count.tif").is_file()

    def test_only_the_hot_pixels_come_out(self, run):
        # The region covers every pixel of the tile. The rule still removes
        # almost none of them, because it removes a pixel for its temperature
        # and not for its cell. Under the old rule this number was the whole
        # tile.
        _, summary, _ = run
        mask = summary["mask"]
        assert mask["valid_removed_by_emissivity"] < mask["pixels_total"] / 1000
        assert mask["valid_removed_by_water"] == 0


class TestATileWithNoLand:
    """The water rule can still empty a tile, and that path still works.

    `land_tiles.py` selects tiles from the same geometry the mask rasterises,
    so no tile on the fleet's list reaches here. An operator naming a bbox by
    hand does.
    """

    @pytest.fixture(scope="module")
    def run(self, tmp_path_factory, numobs_artifact, land_geometry):
        out = tmp_path_factory.mktemp("nolands")
        # Open Pacific, well outside the 25 km buffer of any land.
        argv = [
            # One token, because argparse reads a leading minus as an option.
            "--bbox=-140,-30,-135,-25",
            "--rehearse",
            "20",
            "--numobs-uri",
            str(numobs_artifact),
            "--land-geometry-uri",
            str(land_geometry),
            "--pixels-per-degree",
            str(PPD),
            "--chunk",
            str(CHUNK),
            "--workers",
            "2",
            "--threads-per-worker",
            "1",
            "--out-dir",
            str(out),
        ]
        code = shard_lst_p95.main(argv)
        summary = json.loads((out / "summary.json").read_text())
        return code, summary, out

    def test_it_succeeds(self, run):
        # A correct outcome reading as a dead machine is the distinction these
        # early-exit records exist to preserve.
        code, _, _ = run
        assert code == 0

    def test_the_summary_says_why(self, run):
        _, summary, _ = run
        assert summary["status"] == "no-unmasked-pixels"
        assert summary["mask"]["pixels_kept"] == 0

    def test_the_scene_count_is_unknown_rather_than_zero(self, run):
        # This path runs before the scene list is built, so nothing was asked.
        _, summary, _ = run
        assert summary["n_scenes"] is None

    def test_it_names_the_mosaic_that_decided(self, run):
        _, summary, _ = run
        assert summary["mask"]["aster_ged"]["short_name"] == "AG1km"
        assert summary["mask"]["numobs_uri"]

    def test_it_writes_no_rasters(self, run):
        _, _, out = run
        assert not list(out.rglob("*.tif"))
        assert not (out / "catalog").exists()

    def test_it_reaches_no_cluster(self, run):
        # Both masks are built before staging and before the cluster, so an
        # empty tile costs neither. `memory.csv` is written by the sampler,
        # which starts with the cluster.
        _, _, out = run
        assert not (out / "memory.csv").exists()
        assert not (out / "spans.json").exists()


class TestTheEscapeHatchAndTheGuards:
    def test_no_output_mask_writes_the_unmasked_raster(
        self, tmp_path, masked_coastal, numobs_artifact, land_geometry
    ):
        # Against the masked run of the same tile, which the module already
        # holds. The coast is 4.4% land, so the difference is most of the tile.
        _, masked, _ = masked_coastal
        plain = rehearse(
            tmp_path / "off",
            COASTAL,
            numobs_artifact,
            land_geometry,
            "--no-output-mask",
        )[1]
        assert plain["mask"] is None
        assert plain["valid_fraction"] > masked["valid_fraction"]

    def test_a_missing_mosaic_stops_the_run(self, tmp_path, land_geometry):
        with pytest.raises(aster_ged.GedError, match="uv run aster_ged.py"):
            rehearse(tmp_path / "out", INLAND, tmp_path / "absent.tif", land_geometry)

    def test_a_missing_geometry_stops_the_run(self, tmp_path, numobs_artifact):
        with pytest.raises(masks.MaskError, match="--write-geometry"):
            rehearse(
                tmp_path / "out", INLAND, numobs_artifact, tmp_path / "absent.gpkg"
            )

    def test_a_mosaic_built_from_another_geometry_stops_the_run(
        self, tmp_path, land_geometry
    ):
        rows, cols = aster_ged.mosaic_shape(60)
        manifest = aster_ged.build_manifest(
            {},
            lat_limit=60,
            buffer_meters=25_000,
            land_geometry_sha256="f" * 64,
            cell_count=0,
        )
        other = aster_ged.write_numobs(
            tmp_path / "other.tif", np.full((rows, cols), 8, "uint8"), manifest
        )
        with pytest.raises(aster_ged.GedError, match="land_geometry_sha256"):
            rehearse(tmp_path / "out", INLAND, other, land_geometry)

    def test_a_dry_run_needs_no_mask_artifacts(self, tmp_path, capsys):
        # Planning must never be blocked by an artifact the plan does not read.
        code = shard_lst_p95.main(
            [
                "--tile",
                INLAND,
                "--dry-run",
                "--pixels-per-degree",
                str(PPD),
                "--chunk",
                str(CHUNK),
                "--numobs-uri",
                str(tmp_path / "absent.tif"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0
        assert "blocks.json" in capsys.readouterr().out

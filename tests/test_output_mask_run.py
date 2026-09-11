"""The mask as a run applies it, not as a function returning an array.

`test_masks.py` checks the rules. This checks the four things only a whole run
can get wrong.

That the mask reaches the artifacts. The summary statistics, the part file, and
a merge of parts from several machines all have to describe the same product.
The mask therefore goes on between the gather and everything that reads the
arrays, and a run that masked only its printout would pass every test in
`test_masks.py`.

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

The runs here are rehearsals. `--rehearse` fills the shards with synthetic
pixels and reads no object, but `--tile` fixes the bbox on the production grid,
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
from lst_qa import LST_NODATA_DN  # noqa: E402

#: Interior South America. Every pixel is land, so the land rule removes
#: nothing and the emissivity rule is the only one that can act.
INLAND = "S30W065"
#: The New Jersey coast and a great deal of Atlantic. MEASURED at 1/360 degree:
#: 4.43% land.
COASTAL = "N40W075"

#: Small enough to rehearse in a second, and a whole number of degrees, so the
#: GED cells still land on pixel boundaries.
PPD = 360

pytestmark = [needs_land_geometry, pytest.mark.timeout(180)]


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
        "--shard",
        "512",
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


class TestAMaskedRun:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory, numobs_artifact, land_geometry):
        out = tmp_path_factory.mktemp("masked")
        return rehearse(out, COASTAL, numobs_artifact, land_geometry) + (out,)

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
        ):
            assert field in summary["mask"]

    def test_the_counts_add_up_to_the_raster(self, run):
        _, summary, _ = run
        mask = summary["mask"]
        height, width = summary["raster"]
        assert mask["pixels_total"] == height * width
        assert mask["pixels_kept"] + mask["pixels_water"] == mask["pixels_total"]

    def test_it_names_the_artifacts_that_decided(self, run):
        # A masked tile is only reproducible if the two artifacts are named.
        # `summary.json` is the operator's record and keeps the paths, because
        # an operator rerunning one slice does want them.
        _, summary, _ = run
        assert summary["mask"]["numobs_uri"]
        assert summary["mask"]["land_geometry_uri"]
        assert summary["mask"]["aster_ged"]["short_name"] == "AG1km"

    def test_the_part_names_the_artifacts_by_identity_not_by_path(self, run):
        # `part-meta.json` reaches the published catalog, and an absolute path
        # on the masking machine tells a reader of it nothing. It would also
        # carry the operator's home directory into a public file.
        _, _, out = run
        rule = json.loads((out / "part-meta.json").read_text())["mask_rule"]
        assert "numobs_uri" not in rule
        assert "land_geometry_uri" not in rule
        assert rule["land_geometry_sha256"]
        assert rule["aster_ged"]["doi"].startswith("10.5067/")
        # The digest rides along, empty here: a raster cannot hold its own, and
        # this fixture writes no sidecar to carry it.
        assert "raster_sha256" in rule["aster_ged"]

    def test_the_statistics_describe_the_masked_product(self, run):
        # The mask goes on before anything is measured. A summary computed
        # first would report a valid fraction the raster does not have.
        _, summary, out = run
        with np.load(out / "part-000.npz") as parts:
            written = sum(
                int((parts[key] != LST_NODATA_DN).sum())
                for key in parts.files
                if key.startswith("lst_")
            )
        assert summary["valid_fraction"] == pytest.approx(
            written / summary["mask"]["pixels_total"]
        )

    def test_the_part_file_is_already_masked(self, run):
        """`merge_parts` needs no mask of its own.

        Masking at merge instead would let two machines' parts disagree about
        the rule, and would leave a single-machine run unmasked.
        """
        _, _, out = run
        with np.load(out / "part-000.npz") as parts:
            for key in parts.files:
                if not key.startswith("qa_"):
                    continue
                lst = parts["lst_" + key[3:]]
                qa = parts[key]
                assert np.array_equal(lst == LST_NODATA_DN, qa.sum(axis=0) == 0)


class TestTheMergeInheritsTheMask:
    """`merge_parts` applies no rule of its own, and must not need to.

    Every part it reads was masked by the machine that wrote it. If that were
    not true, a tile assembled from four slices could carry four different
    answers about the same coastline, and a single-machine run would carry
    none.
    """

    @pytest.fixture(scope="class")
    def merged(self, tmp_path_factory, numobs_artifact, land_geometry):
        out = tmp_path_factory.mktemp("tomerge")
        rehearse(out, COASTAL, numobs_artifact, land_geometry)
        tile = out.parent / "merged"
        code = shard_lst_p95.main(["--merge", str(out), "--out-dir", str(tile)])
        return code, tile

    def test_the_merge_succeeds(self, merged):
        code, _ = merged
        assert code == 0

    def test_the_merged_raster_is_masked(self, merged, numobs_artifact, land_geometry):
        _, tile = merged
        lst = np.load(tile / "lst_p95_dn.npy")
        keep, _, _ = masks.output_mask(
            shard_lst_p95.tile_bounds(COASTAL),
            PPD,
            numobs_uri=numobs_artifact,
            land_geometry_uri=land_geometry,
        )
        assert not (lst[~keep] != LST_NODATA_DN).any()

    def test_the_merge_records_the_rule_every_part_agreed_on(self, merged):
        _, tile = merged
        report = json.loads((tile / "merge.json").read_text())
        rule = report["mask_rule"]
        assert rule["gap_hot_threshold_c"] == masks.GAP_HOT_THRESHOLD_C
        assert rule["gap_buffer_cells"] == masks.GAP_BUFFER_CELLS

    def test_parts_masked_under_different_rules_are_refused(
        self, merged, tmp_path, numobs_artifact, land_geometry
    ):
        # Two machines that masked the same tile differently make one raster
        # that no single rule describes. The merge stops rather than blending.
        _, tile = merged
        first = tmp_path / "a"
        second = tmp_path / "b"
        rehearse(first, COASTAL, numobs_artifact, land_geometry)
        rehearse(second, COASTAL, numobs_artifact, land_geometry, "--no-output-mask")
        with pytest.raises(SystemExit, match="different rules"):
            shard_lst_p95.main(
                ["--merge", str(first), str(second), "--out-dir", str(tmp_path / "m")]
            )

    def test_the_merged_bands_agree(self, merged):
        _, tile = merged
        lst = np.load(tile / "lst_p95_dn.npy")
        qa = np.load(tile / "qa_count.npy")
        assert np.array_equal(lst == LST_NODATA_DN, qa.sum(axis=0) == 0)


class TestAnInlandTileKeepsEverything:
    """The land rule must not eat a tile that is entirely land."""

    @pytest.fixture(scope="class")
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


class TestASliceSaysWhatItsCountsDescribe:
    """Some counts are tile-wide and some are this machine's, so say which.

    `output_mask` runs over the whole bbox in every run, slice or not, so
    `pixels_total` and its siblings are the same number on every machine.
    `valid_removed_by_mask` counts only the pixels this process assembled.
    Summing the first group across four slices would multiply the tile by four.
    """

    @pytest.fixture(scope="class")
    def slice_run(self, tmp_path_factory, numobs_artifact, land_geometry):
        out = tmp_path_factory.mktemp("slice")
        return rehearse(
            out, INLAND, numobs_artifact, land_geometry, "--shard-slice", "8:16"
        )

    def test_the_scope_names_the_slice(self, slice_run):
        _, summary = slice_run
        assert summary["mask"]["scope"] == "shards[8:16]"

    def test_the_tile_wide_counts_still_describe_the_tile(
        self, slice_run, run_whole_tile
    ):
        # Identical on every machine, which is what makes them unsummable.
        _, sliced = slice_run
        _, whole, _ = run_whole_tile
        for field in ("pixels_total", "pixels_water", "pixels_kept"):
            assert sliced["mask"][field] == whole["mask"][field]

    def test_a_whole_tile_run_says_so(self, run_whole_tile):
        _, summary, _ = run_whole_tile
        assert summary["mask"]["scope"] == "tile"

    @pytest.fixture(scope="class")
    def run_whole_tile(self, tmp_path_factory, numobs_artifact, land_geometry):
        out = tmp_path_factory.mktemp("whole")
        return rehearse(out, INLAND, numobs_artifact, land_geometry) + (out,)


class TestATileOfNothingButGapStillPublishes:
    """The gap region is not a removal, so it cannot empty a tile.

    An earlier build dropped a tile whose every cell was a gap, on the reading
    that Collection 2 published nothing there. It publishes plenty: USGS
    interpolates emissivity across a gap cell and retrieves a temperature, and
    only the ones that fail upward come out.
    """

    @pytest.fixture(scope="class")
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

    def test_it_writes_a_part(self, run):
        _, _, out = run
        assert list(out.glob("part-*.npz"))

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

    @pytest.fixture(scope="class")
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
            "--shard",
            "512",
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
        # A correct outcome reading as a dead machine is the distinction the
        # barren-shard records exist to preserve.
        code, _, _ = run
        assert code == 0

    def test_the_summary_says_why(self, run):
        _, summary, _ = run
        assert summary["status"] == "no-unmasked-pixels"
        assert summary["mask"]["pixels_kept"] == 0

    def test_the_scene_count_is_unknown_rather_than_zero(self, run):
        # This path runs before the search, so the archive was never asked.
        _, summary, _ = run
        assert summary["n_scenes"] is None

    def test_it_names_the_mosaic_that_decided(self, run):
        _, summary, _ = run
        assert summary["mask"]["aster_ged"]["short_name"] == "AG1km"
        assert summary["mask"]["numobs_uri"]

    def test_it_writes_no_parts(self, run):
        _, _, out = run
        assert not list(out.glob("part-*.npz"))

    def test_it_reaches_no_cluster(self, run):
        # Both masks are built before staging and before the cluster, so an
        # empty tile costs neither. `memory.csv` is written by the sampler,
        # which starts with the cluster.
        _, _, out = run
        assert not (out / "memory.csv").exists()
        assert not (out / "spans.json").exists()


class TestTheEscapeHatchAndTheGuards:
    def test_no_output_mask_writes_the_unmasked_raster(
        self, tmp_path, numobs_artifact, land_geometry
    ):
        masked = rehearse(tmp_path / "on", COASTAL, numobs_artifact, land_geometry)[1]
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
                "--shard",
                "512",
                "--numobs-uri",
                str(tmp_path / "absent.tif"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        assert code == 0
        assert "shards.json" in capsys.readouterr().out

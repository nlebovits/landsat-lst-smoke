"""The glue that decides whether the right prep file reaches the right tile.

`tests/test_destripe.py` covers the numerics and `tests/test_tile_prep.py`
covers the pass that produces them. Neither touches the code that reads the
artifact back and refuses the wrong one, and that code guards a worse class of
failure: a run composited against another tile's offsets, another grid's
weights, or a different scene list finishes and writes an ordinary-looking
raster. The numerics fail loudly. This fails quietly.

Nothing here reads S3 or opens a catalogue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import destripe  # noqa: E402
import shard_lst_p95  # noqa: E402
import tile_prep  # noqa: E402

TILE = "S30W065"
BBOX = (-66.0, -36.0, -59.0, -29.0)
PPD = 3600
WEST, EAST = "228", "229"
WINDOW = {
    "start": "2021-01-01",
    "end": "2025-12-31",
    "platforms": "landsat-8,landsat-9",
    "cloud_cover_lt": 100,
}


def item(scene_id: str, path: str = WEST) -> dict:
    return {
        "properties": {
            "datetime": "2021-01-05T14:02:11.123456Z",
            "landsat:scene_id": scene_id,
            "landsat:wrs_path": path,
            "landsat:wrs_row": "030",
        }
    }


def items(n: int = 4) -> list[dict]:
    return [item(f"SCENE{i:02d}", WEST if i % 2 == 0 else EAST) for i in range(n)]


def write_prep(directory: Path, item_dicts, **overrides) -> Path:
    """A prep artifact for `item_dicts`, with any field overridden."""
    scene_ids = [destripe.scene_id_of(d) for d in item_dicts]
    window = overrides.pop("window", WINDOW)
    meta = {
        "schema_version": tile_prep.PREP_SCHEMA_VERSION,
        "scene_digest": destripe.scene_digest(scene_ids, window),
        "tile": TILE,
        "bbox": list(BBOX),
        "pixels_per_degree": PPD,
        "prep_factor": 4,
        "swath_factor": 8,
        "weight_factor": 4,
        "margin_deg": 1.0,
        "anomaly_bin_c": destripe.ANOMALY_BIN_C,
        "min_offset_samples": destripe.DESTRIPE_MIN_PREP_SAMPLES,
        "swath_quad_share": destripe.SWATH_QUAD_SHARE,
        "offsets": {"n_scenes": len(scene_ids)},
        "inventory": {"schema_version": 1, "built_at": "2026-09-01"},
        "window": window,
    } | overrides
    payload = {
        "scene_ids": np.array(scene_ids),
        "offset": overrides.pop("offset", np.zeros(len(scene_ids))),
        "n_valid": overrides.pop("n_valid", np.full(len(scene_ids), 9_000)),
        "paths": np.array([WEST, EAST]),
        "weight": np.zeros((2, 8, 8), dtype="float32"),
        "inside": np.ones((2, 8, 8), dtype=bool),
    }
    tile_prep.write_artifact(directory, payload, meta)
    return directory


def run_args(prep_dir: Path | None, **overrides):
    base = {
        "tile_prep": prep_dir,
        "pixels_per_degree": PPD,
        "max_offset_c": destripe.DESTRIPE_MAX_OFFSET_C,
        "no_destripe": False,
        "no_feather": False,
        "emit_pooled": False,
        "out_dir": prep_dir,
        **WINDOW,
    }
    return argparse.Namespace(**(base | overrides))


def merge_args(**overrides):
    """What `merge_parts` reads besides the parts themselves.

    The catalog is off. These tests fabricate a part-meta to make two parts
    disagree about one rule, and a fabricated meta carries none of the window
    a catalog item needs. `tests/test_output_mask_run.py` covers the catalog
    against a meta a real run wrote.
    """
    return argparse.Namespace(**({"no_catalog": True} | overrides))


PROVENANCE = {"schema_version": 1, "built_at": "2026-09-01"}


class TestLoadingThePrepFile:
    def test_no_prep_file_composites_pooled(self):
        assert (
            shard_lst_p95.load_tile_prep(run_args(None), TILE, items(), PROVENANCE)
            is None
        )

    def test_a_matching_file_loads(self, tmp_path):
        write_prep(tmp_path, items())
        prep = shard_lst_p95.load_tile_prep(
            run_args(tmp_path), TILE, items(), PROVENANCE
        )
        assert prep is not None
        assert prep.tile == TILE
        assert prep.paths == (WEST, EAST)

    def test_another_tile_is_refused(self, tmp_path):
        write_prep(tmp_path, items(), tile="N40W075")
        with pytest.raises(SystemExit, match="written for tile N40W075"):
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(), PROVENANCE)

    def test_an_unknown_schema_version_is_refused(self, tmp_path):
        write_prep(tmp_path, items(), schema_version=999)
        with pytest.raises(SystemExit, match="schema version"):
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(), PROVENANCE)

    def test_another_grid_is_refused(self, tmp_path):
        write_prep(tmp_path, items(), pixels_per_degree=1800)
        with pytest.raises(SystemExit, match="would land on the wrong ground"):
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(), PROVENANCE)

    def test_a_different_scene_set_is_refused(self, tmp_path):
        """The check that replaces the sibling's offset cache key.

        Same tile, same grid, same window, same every parameter. Only the scene
        list differs, and without the digest these two are indistinguishable.
        """
        write_prep(tmp_path, items(4))
        with pytest.raises(SystemExit, match="different scene set"):
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(5), PROVENANCE)

    def test_a_different_window_is_refused(self, tmp_path):
        write_prep(tmp_path, items(), window=WINDOW | {"end": "2024-12-31"})
        with pytest.raises(SystemExit, match="different scene set or a different"):
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(), PROVENANCE)

    def test_a_rebuilt_inventory_is_refused(self, tmp_path):
        """The scene ids can match while a footprint or a href has moved."""
        write_prep(tmp_path, items())
        with pytest.raises(SystemExit, match="different inventory artifact"):
            shard_lst_p95.load_tile_prep(
                run_args(tmp_path),
                TILE,
                items(),
                {"schema_version": 1, "built_at": "2026-09-10"},
            )

    def test_a_smoke_run_artifact_is_refused(self, tmp_path):
        """`--max-blocks` measures coverage over part of the tile.

        Every quad's swath is counted from the blocks that ran and divided by
        all of that quad's scenes, so the half-the-scenes threshold has the
        wrong denominator and the swaths come out small. The offsets rest on
        too few pixels for the same reason. A run against it finishes and
        writes an ordinary-looking raster.
        """
        write_prep(
            tmp_path,
            items(),
            blocks={
                "planned": 412,
                "with_scenes": 412,
                "run": 8,
                "max_blocks": 8,
                "partial": True,
            },
        )
        with pytest.raises(SystemExit, match="8 of 412 blocks"):
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(), PROVENANCE)

    def test_a_whole_run_carries_no_partial_flag(self, tmp_path):
        write_prep(
            tmp_path,
            items(),
            blocks={
                "planned": 412,
                "with_scenes": 400,
                "run": 400,
                "max_blocks": None,
                "partial": False,
            },
        )
        prep = shard_lst_p95.load_tile_prep(
            run_args(tmp_path), TILE, items(), PROVENANCE
        )
        assert prep is not None

    def test_an_artifact_with_no_block_record_still_loads(self, tmp_path):
        # `blocks` is absent from a file written before the key existed, and
        # absent is not partial. Refusing on a missing key would reject every
        # prep file the earlier version wrote.
        write_prep(tmp_path, items())
        assert (
            shard_lst_p95.load_tile_prep(run_args(tmp_path), TILE, items(), PROVENANCE)
            is not None
        )

    def test_the_scene_order_cannot_change_the_digest(self):
        forward = [destripe.scene_id_of(d) for d in items(5)]
        assert destripe.scene_digest(forward, WINDOW) == destripe.scene_digest(
            list(reversed(forward)), WINDOW
        )

    def test_a_scene_added_or_removed_changes_the_digest(self):
        four = [destripe.scene_id_of(d) for d in items(4)]
        assert destripe.scene_digest(four, WINDOW) != destripe.scene_digest(
            four[:-1], WINDOW
        )


class TestTheCorrectionRule:
    def test_no_prep_file_is_itself_a_rule(self):
        assert shard_lst_p95.correction_rule(run_args(None), None) is None

    def test_it_carries_the_scene_digest(self, tmp_path):
        write_prep(tmp_path, items())
        prep = destripe.load_prep(tmp_path)
        rule = shard_lst_p95.correction_rule(run_args(tmp_path), prep)
        assert rule is not None
        assert rule["prep_scene_digest"] == prep.digest
        assert rule["prep_scene_digest"]

    def test_two_prep_files_over_different_scenes_give_different_rules(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        write_prep(a, items(4))
        write_prep(b, items(5))
        rule_a = shard_lst_p95.correction_rule(run_args(a), destripe.load_prep(a))
        rule_b = shard_lst_p95.correction_rule(run_args(b), destripe.load_prep(b))
        assert rule_a != rule_b

    def test_turning_a_correction_off_changes_the_rule(self, tmp_path):
        write_prep(tmp_path, items())
        prep = destripe.load_prep(tmp_path)
        on = shard_lst_p95.correction_rule(run_args(tmp_path), prep)
        off = shard_lst_p95.correction_rule(run_args(tmp_path, no_feather=True), prep)
        assert on is not None and off is not None
        assert on["feather"] is True
        assert off["feather"] is False
        assert on != off

    def test_the_cap_is_absent_when_there_is_nothing_to_cap(self, tmp_path):
        write_prep(tmp_path, items())
        prep = destripe.load_prep(tmp_path)
        rule = shard_lst_p95.correction_rule(run_args(tmp_path, no_destripe=True), prep)
        assert rule is not None
        assert rule["max_offset_c"] is None


def write_part(directory: Path, rule, *, pooled=None) -> Path:
    """One part file and its meta, covering a 4 x 4 tile in a single shard.

    `pooled` is the baseline array `--emit-pooled` would have written. Nothing
    in the meta records the flag, so the merge finds the key by looking, and
    this is what a part that ran without it looks like.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "lst_0_0": np.zeros((4, 4), dtype="uint16"),
        "qa_0_0": np.zeros((12, 4, 4), dtype="uint8"),
    }
    if pooled is not None:
        arrays["pooled_0_0"] = pooled
    # numpy declares savez_compressed(**kwds: ArrayLike) alongside a bool
    # allow_pickle, so a dict of arrays collides with the named parameter.
    np.savez_compressed(
        directory / "part-000.npz",
        **arrays,  # ty: ignore[invalid-argument-type]
    )
    (directory / "part-meta.json").write_text(
        json.dumps(
            {
                "raster": [4, 4],
                "bbox": list(BBOX),
                "crs": "EPSG:4326",
                "pixels_per_degree": PPD,
                "shard_px": 4,
                "n_shards": 1,
                "mask_rule": None,
                "correction_rule": rule,
            }
        )
    )
    return directory


class TestTheMergedPooledBaseline:
    """`--emit-pooled` wrote a raster the merge used to drop on the floor.

    Every part carried `pooled_<y0>_<x0>` keys and `merge_parts` skipped every
    key that did not start with `lst_`, so the baseline existed only inside
    part files, one per slice, and the README promised a raster beside the
    product.
    """

    def test_a_run_without_the_flag_writes_no_baseline(self, tmp_path):
        a = write_part(tmp_path / "a", None)
        assert shard_lst_p95.merge_parts([a], tmp_path / "out", merge_args()) in (0, 2)
        assert not (tmp_path / "out" / "lst_p95_pooled_dn.npy").exists()
        record = json.loads((tmp_path / "out" / "merge.json").read_text())
        assert record["pooled_coverage"] is None

    def test_the_baseline_is_assembled_beside_the_product(self, tmp_path):
        pooled = np.full((4, 4), 4_242, dtype="uint16")
        a = write_part(tmp_path / "a", None, pooled=pooled)
        assert shard_lst_p95.merge_parts([a], tmp_path / "out", merge_args()) in (0, 2)
        written = np.load(tmp_path / "out" / "lst_p95_pooled_dn.npy")
        assert np.array_equal(written, pooled)
        record = json.loads((tmp_path / "out" / "merge.json").read_text())
        assert record["pooled_coverage"] == 1.0

    def test_one_slice_without_the_flag_reports_partial_coverage(self, tmp_path):
        """A diagnostic raster is no reason to refuse a product.

        The mask and correction rules stop a merge because they decide pixel
        values. This one is a baseline for comparison, so a partial one is
        reported and the tile still merges.
        """
        pooled = np.full((2, 2), 7, dtype="uint16")
        a = tmp_path / "a"
        a.mkdir(parents=True)
        np.savez_compressed(
            a / "part-000.npz",
            lst_0_0=np.zeros((2, 2), dtype="uint16"),
            qa_0_0=np.zeros((12, 2, 2), dtype="uint8"),
            pooled_0_0=pooled,
            lst_2_0=np.zeros((2, 2), dtype="uint16"),
            qa_2_0=np.zeros((12, 2, 2), dtype="uint8"),
        )
        (a / "part-meta.json").write_text(
            json.dumps(
                {
                    "raster": [4, 2],
                    "bbox": list(BBOX),
                    "crs": "EPSG:4326",
                    "pixels_per_degree": PPD,
                    "shard_px": 2,
                    "n_shards": 2,
                    "mask_rule": None,
                    "correction_rule": None,
                }
            )
        )
        assert shard_lst_p95.merge_parts([a], tmp_path / "out", merge_args()) in (0, 2)
        record = json.loads((tmp_path / "out" / "merge.json").read_text())
        assert record["pooled_coverage"] == 0.5
        assert record["coverage"] == 1.0


class TestMergingRefusesMixedParts:
    def part(self, directory: Path, rule) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "part-000.npz",
            lst_0_0=np.zeros((4, 4), dtype="uint16"),
            qa_0_0=np.zeros((12, 4, 4), dtype="uint8"),
        )
        (directory / "part-meta.json").write_text(
            json.dumps(
                {
                    "raster": [4, 4],
                    "bbox": list(BBOX),
                    "crs": "EPSG:4326",
                    "pixels_per_degree": PPD,
                    "shard_px": 4,
                    "n_shards": 1,
                    "mask_rule": None,
                    "correction_rule": rule,
                }
            )
        )
        return directory

    def test_one_corrected_part_beside_one_uncorrected_stops_the_merge(self, tmp_path):
        a = self.part(tmp_path / "a", {"prep_scene_digest": "abc", "feather": True})
        b = self.part(tmp_path / "b", None)
        with pytest.raises(SystemExit, match="corrected under 2 different rules"):
            shard_lst_p95.merge_parts([a, b], tmp_path / "out", merge_args())

    def test_parts_built_against_different_prep_files_stop_the_merge(self, tmp_path):
        rule = {"prep_scene_digest": "abc", "feather": True}
        a = self.part(tmp_path / "a", rule)
        b = self.part(tmp_path / "b", rule | {"prep_scene_digest": "def"})
        with pytest.raises(SystemExit, match="corrected under 2 different rules"):
            shard_lst_p95.merge_parts([a, b], tmp_path / "out", merge_args())

    def test_matching_parts_merge(self, tmp_path):
        rule = {"prep_scene_digest": "abc", "feather": True}
        a = self.part(tmp_path / "a", rule)
        b = self.part(tmp_path / "b", dict(rule))
        assert shard_lst_p95.merge_parts([a, b], tmp_path / "out", merge_args()) in (
            0,
            2,
        )


class TestEveryRejectedScene:
    def test_the_run_stops_rather_than_writing_an_empty_tile(self, tmp_path):
        """An empty composite is not a tile with no data.

        It is a tile whose whole scene list was found untrustworthy, and
        writing it as nodata presents that as an observation gap.
        """
        write_prep(
            tmp_path,
            items(),
            offset=np.full(4, -73.0),
        )
        prep = destripe.load_prep(tmp_path)
        code = shard_lst_p95.no_scene_survives_destriping(
            run_args(tmp_path), TILE, prep, PROVENANCE
        )
        assert code == 1
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["status"] == "no-scene-survives-destriping"
        assert summary["n_scenes"] == 4
        assert not list(tmp_path.glob("part-*.npz"))


class TestTheShardSliceOfThePrepFile:
    def test_it_runs_parallel_to_the_item_list_it_was_given(self, tmp_path):
        write_prep(tmp_path, items(4), offset=np.array([0.0, 1.0, 2.0, 3.0]))
        prep = destripe.load_prep(tmp_path)
        # The shard holds two of the four scenes, in its own order.
        subset = [items(4)[3], items(4)[1]]
        shard = shard_lst_p95.Shard(0, 0, 0, 0, 8, 8, (-61.0, -33.0, -60.9, -32.9))
        correction = shard_lst_p95.shard_correction_for(
            run_args(tmp_path), prep, shard, subset
        )
        assert list(correction["offset"]) == [3.0, 1.0]
        assert list(correction["keep"]) == [True, True]
        assert correction["weight"].shape == (2, 8, 8)

    def test_turning_the_offsets_off_zeroes_them_and_keeps_every_scene(self, tmp_path):
        write_prep(tmp_path, items(4), offset=np.full(4, -73.0))
        prep = destripe.load_prep(tmp_path)
        shard = shard_lst_p95.Shard(0, 0, 0, 0, 8, 8, (-61.0, -33.0, -60.9, -32.9))
        correction = shard_lst_p95.shard_correction_for(
            run_args(tmp_path, no_destripe=True), prep, shard, items(4)
        )
        assert not correction["offset"].any()
        assert correction["keep"].all()

    def test_turning_the_cross_fade_off_sends_no_weights(self, tmp_path):
        write_prep(tmp_path, items(4))
        prep = destripe.load_prep(tmp_path)
        shard = shard_lst_p95.Shard(0, 0, 0, 0, 8, 8, (-61.0, -33.0, -60.9, -32.9))
        correction = shard_lst_p95.shard_correction_for(
            run_args(tmp_path, no_feather=True), prep, shard, items(4)
        )
        assert correction["weight"] is None
        assert correction["paths"] == ()

    def test_a_scene_the_prep_file_never_saw_is_rejected(self, tmp_path):
        """Not defaulted to zero. An unknown offset is not a zero offset."""
        write_prep(tmp_path, items(4))
        prep = destripe.load_prep(tmp_path)
        shard = shard_lst_p95.Shard(0, 0, 0, 0, 8, 8, (-61.0, -33.0, -60.9, -32.9))
        correction = shard_lst_p95.shard_correction_for(
            run_args(tmp_path), prep, shard, [item("SCENE99")]
        )
        assert np.isnan(correction["offset"][0])
        assert not correction["keep"][0]


class TestThePrepMemoryModel:
    def model(
        self,
        block: int = 512,
        scenes_per_block=(2_000,),
        n_scenes: int = 4_776,
        n_quads: int = 25,
        swath_shape=(3150, 3150),
        slots: int = 8,
    ):
        return tile_prep.memory_model(
            block, scenes_per_block, n_scenes, n_quads, swath_shape, slots
        )

    def test_the_histogram_a_block_returns_is_named_and_counted(self):
        """The term that surprises: 104 KB a scene, shipped whole."""
        model = self.model()
        per_scene = destripe.N_ANOMALY_BINS * 4
        assert model["worker_histogram_gib"] == pytest.approx(
            2_000 * per_scene / 1024**3
        )
        assert model["worker_histogram_gib"] > 0.15

    def test_the_total_is_the_sum_of_its_named_terms(self):
        model = self.model()
        assert model["total_gib"] == pytest.approx(
            model["workers_gib"]
            + model["results_in_flight_gib"]
            + model["driver_histogram_gib"]
            + model["driver_coverage_gib"]
        )

    def test_a_run_that_will_not_fit_stops_before_it_reads_anything(self):
        with pytest.raises(SystemExit, match="needs .* GiB and the machine has"):
            tile_prep.memory_guard(self.model(), target_gib=4.0, force=False)

    def test_force_runs_it_anyway(self):
        tile_prep.memory_guard(self.model(), target_gib=4.0, force=True)

    def test_no_target_checks_nothing(self):
        tile_prep.memory_guard(self.model(), target_gib=None, force=False)

    def test_a_bigger_block_costs_more(self):
        assert self.model(block=1024)["worker_block_stack_gib"] == pytest.approx(
            4 * self.model(block=512)["worker_block_stack_gib"]
        )

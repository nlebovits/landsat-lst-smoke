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

import composite  # noqa: E402
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


def item(scene_id: str, path: str = WEST, *, minute: int = 0) -> dict:
    """One item. Each carries its own stamp, because the join is on time.

    `composite.per_scene_vectors` maps every value onto the loaded time axis
    through `destripe.align_to_time`, which refuses two scenes that share a
    stamp and disagree. A fixture that gave every scene the same minute could
    not exercise the join at all.
    """
    return {
        "properties": {
            "datetime": f"2021-01-05T14:{minute:02d}:11.123456Z",
            "landsat:scene_id": scene_id,
            "landsat:wrs_path": path,
            "landsat:wrs_row": "030",
        }
    }


def items(n: int = 4) -> list[dict]:
    return [
        item(f"SCENE{i:02d}", WEST if i % 2 == 0 else EAST, minute=i) for i in range(n)
    ]


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
        assert not list(tmp_path.glob("*.tif"))


def vectors(prep_dir: Path, item_dicts, times, **overrides):
    """`per_scene_vectors` under the flags the driver would have parsed."""
    args = run_args(prep_dir, **overrides)
    return composite.per_scene_vectors(
        item_dicts,
        times,
        destripe.load_prep(prep_dir),
        max_offset_c=args.max_offset_c,
        debias=not args.no_destripe,
    )


class TestThePerSceneVectors:
    """The offsets, the keep flags and the path codes the kernel is handed.

    One value per step of the loaded time axis, joined on the acquisition stamp
    and never on position. The sibling paid for a positional join once: labels
    carried positionally against a stack that de-striping had thinned killed
    all 35 of its shards with an `IndexError`. `composite.build_graph` calls
    this and hands the result to `reduce_block` as an `apply_ufunc` core
    dimension, so a vector out of step with the stack corrects the wrong scene
    and says nothing.
    """

    def test_they_run_parallel_to_the_time_axis_they_were_given(self, tmp_path):
        write_prep(tmp_path, items(4), offset=np.array([0.0, 1.0, 2.0, 3.0]))
        # The stack holds two of the four scenes, in neither the item order nor
        # the prep order.
        scenes = items(4)
        times = [destripe.timestamp_of(scenes[3]), destripe.timestamp_of(scenes[1])]
        offset, keep, path_code, paths = vectors(tmp_path, scenes, times)
        assert list(offset) == [3.0, 1.0]
        assert list(keep) == [True, True]
        # Both of those scenes are on the eastern path, which is the prep
        # file's second, so both codes are 1.
        assert list(path_code) == [1, 1]
        assert paths == (WEST, EAST)

    def test_turning_the_offsets_off_zeroes_them_and_keeps_every_scene(self, tmp_path):
        write_prep(tmp_path, items(4), offset=np.full(4, -73.0))
        scenes = items(4)
        times = [destripe.timestamp_of(d) for d in scenes]
        offset, keep, _, _ = vectors(tmp_path, scenes, times, no_destripe=True)
        assert not offset.any()
        assert keep.all()

    def test_a_scene_the_prep_file_never_saw_is_rejected(self, tmp_path):
        """Not defaulted to zero. An unknown offset is not a zero offset."""
        write_prep(tmp_path, items(4))
        unknown = item("SCENE99", minute=41)
        offset, keep, _, _ = vectors(
            tmp_path, [unknown], [destripe.timestamp_of(unknown)]
        )
        assert np.isnan(offset[0])
        assert not keep[0]

    def test_no_prep_file_keeps_every_scene_at_its_own_baseline(self):
        scenes = items(3)
        times = [destripe.timestamp_of(d) for d in scenes]
        offset, keep, path_code, paths = composite.per_scene_vectors(
            scenes,
            times,
            None,
            max_offset_c=destripe.DESTRIPE_MAX_OFFSET_C,
            debias=True,
        )
        assert not offset.any()
        assert keep.all()
        # -1 matches no path, so the cross-fade has nothing to blend.
        assert list(path_code) == [-1, -1, -1]
        assert paths == ()

    def test_a_stack_step_no_item_accounts_for_is_refused(self, tmp_path):
        """A vector shorter than the stack is the failure this join prevents."""
        write_prep(tmp_path, items(4))
        scenes = items(4)
        stranger = destripe.timestamp_of(item("SCENE99", minute=55))
        with pytest.raises(ValueError, match="no item accounts for"):
            vectors(
                tmp_path,
                scenes,
                [*[destripe.timestamp_of(d) for d in scenes], stranger],
            )


class TestThePooledBaseline:
    """`--emit-pooled` writes a second raster beside the product.

    The pooled percentile after the offsets are applied, from the same graph
    and the same blocks, so the two can be differenced pixel for pixel. It used
    to exist only inside part files, one per slice, while the README promised a
    raster beside the product.
    """

    @pytest.mark.timeout(300)
    def test_the_flag_writes_the_baseline_beside_the_summary(self, tmp_path):
        out = tmp_path / "run"
        code = shard_lst_p95.main(
            [
                "--bbox=-65.0,-32.5,-64.5,-32.0",
                "--rehearse",
                "6",
                "--pixels-per-degree",
                "120",
                "--chunk",
                "30",
                "--workers",
                "2",
                "--threads-per-worker",
                "1",
                "--no-output-mask",
                "--no-catalog",
                "--emit-pooled",
                "--out-dir",
                str(out),
            ]
        )
        assert code == 0
        assert (out / "summary.json").is_file()
        assert (out / "lst_p95_pooled.tif").is_file()
        assert (out / "lst_p95.tif").is_file()
        # And nothing of the staging files it was assembled from survives.
        assert not list(out.glob("*.staging.tif"))


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

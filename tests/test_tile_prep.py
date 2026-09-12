"""The one coarse pass, and the two things it has to get right.

`tile_prep` exists so that no shard has to guess at a quantity belonging to the
whole tile. Two claims carry it.

A block returns per-scene histograms rather than per-scene medians, because a
median does not survive being split across blocks and a histogram does. So
splitting the grid differently must not move an offset. If it did, this design
would be paying for the second source traversal it claims to avoid.

A block's rows are indexed by the loaded time axis, which `odc.stac` sorts by
datetime. The scene they belong to is found by joining on that time, never by
position in the item list. Getting this wrong credits one scene's observations
to another, silently.

Nothing here reads S3 or opens a catalogue. `odc.stac.stac_load` is replaced by
a fixture with known contents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import destripe  # noqa: E402
import tile_prep  # noqa: E402
from lst_qa import LWIR_OFFSET_C, LWIR_SCALE  # noqa: E402

NY = NX = 8
RATIO = 2
N_SCENES = 12
QA_CLEAR = 0b1000000
WEST, EAST = "228", "229"


def dn_of(celsius: float) -> int:
    return int(round((celsius - LWIR_OFFSET_C) / LWIR_SCALE))


def make_items():
    """Twelve scenes over a year, two paths, in an order the loader will change.

    Sub-second stamps throughout. A whole-second axis hides the precision class
    of bug this join has already cost the sibling repository one production run.
    """
    items = []
    for s in range(N_SCENES):
        month = s + 1
        path, row = (WEST, "030") if s % 2 == 0 else (EAST, "031")
        items.append(
            {
                "properties": {
                    "datetime": f"2021-{month:02d}-05T14:02:{s:02d}.123456Z",
                    "landsat:scene_id": f"SCENE{s:02d}",
                    "landsat:wrs_path": path,
                    "landsat:wrs_row": row,
                }
            }
        )
    # The inventory order is not the acquisition order, and neither is the
    # order the loader returns. Shuffle so a positional join cannot pass.
    return [items[i] for i in (7, 0, 11, 3, 5, 9, 1, 8, 2, 10, 4, 6)]


def dataset_for(items, warm=None, fill=()):
    """The stack `stac_load` would return: sorted by time, one step per scene."""
    ordered = sorted(items, key=destripe.timestamp_of)
    times = np.array(
        [destripe.timestamp_of(i) for i in ordered], dtype="datetime64[ns]"
    )
    warm = np.linspace(18.0, 44.0, len(ordered)) if warm is None else np.asarray(warm)
    dn = np.empty((len(ordered), NY, NX), dtype="uint16")
    for t, celsius in enumerate(warm):
        dn[t, :, :] = dn_of(float(celsius))
    qa = np.full((len(ordered), NY, NX), QA_CLEAR, dtype="uint16")
    for t, y, x in fill:
        dn[t, y, x] = 0
    return ordered, xr.Dataset(
        {
            "lwir11": (("time", "latitude", "longitude"), dn),
            "qa_pixel": (("time", "latitude", "longitude"), qa),
        },
        coords={"time": times},
    )


@pytest.fixture
def block():
    return tile_prep.Shard(0, 0, 0, 0, NY, NX, (-62.0, -34.0, -61.9, -33.9))


@pytest.fixture
def loaded(monkeypatch):
    """Wire the fake loader and hand back the items in inventory order."""
    import odc.stac
    import pystac

    items = make_items()
    _ordered, dataset = dataset_for(items)
    monkeypatch.setattr(odc.stac, "stac_load", lambda *a, **k: dataset.copy(deep=True))
    monkeypatch.setattr(pystac.Item, "from_dict", staticmethod(lambda d: d))
    return items


BLOCK = tile_prep.DEFAULT_BLOCK


class TestTheGrid:
    def test_a_factor_that_does_not_divide_the_output_grid_is_refused(self):
        with pytest.raises(SystemExit, match="does not divide"):
            tile_prep.check_grid(3600, 7, 8, BLOCK)

    def test_a_swath_cell_may_not_straddle_two_prep_blocks(self):
        with pytest.raises(SystemExit, match="does not divide by swath grid"):
            tile_prep.check_grid(3600, 16, 8, BLOCK)

    def test_the_ratio_is_the_prep_grid_over_the_swath_grid(self):
        assert tile_prep.check_grid(3600, 4, 8, BLOCK) == 2
        assert tile_prep.check_grid(3600, 1, 8, BLOCK) == 8
        assert tile_prep.check_grid(3600, 8, 8, BLOCK) == 1

    def test_a_block_that_is_not_whole_swath_cells_is_refused(self):
        """The one ragged ratio `check_grid` used to let through.

        `plan_shards` starts block n at `n * block`, and `prep_block` reports
        its coarse origin as `block.y0 // ratio`. At ratio 4 and block 510,
        block 1 lands on coarse cell 127 where its origin is 127.5, so two
        blocks write the same coarse row and `accumulate` counts it twice.
        """
        with pytest.raises(SystemExit, match="not a whole number of swath cells"):
            tile_prep.check_grid(3600, 4, 16, 510)

    def test_the_default_block_divides_the_default_ratio(self):
        assert tile_prep.check_grid(3600, 4, 8, tile_prep.DEFAULT_BLOCK) == 2

    def test_a_message_names_every_problem_at_once(self):
        # An operator fixing one flag should not have to rerun to find the
        # next, which is why `check_grid` collects before it raises.
        with pytest.raises(SystemExit) as caught:
            tile_prep.check_grid(3600, 7, 7, 510)
        assert str(caught.value).count(";") >= 1

    def test_a_swath_grid_that_would_drop_a_row_is_refused(self):
        """`swath_shape` floors, and `accumulate` clamps to it.

        A remainder therefore discards the last row of swath cells without a
        word, and a path whose only cell is in there loses its swath. MEASURED
        at `--margin-deg 0.125` on a 7 degree tile at prep factor 4: 6,525 prep
        rows against a ratio of 2.
        """
        with pytest.raises(SystemExit, match="the swath grid would drop"):
            tile_prep.check_swath_grid(6525, 6300, 2)

    def test_a_grid_the_ratio_divides_passes(self):
        assert tile_prep.check_swath_grid(8100, 8100, 2) is None

    def test_the_message_names_the_flag_that_moves_the_remainder(self):
        with pytest.raises(SystemExit) as caught:
            tile_prep.check_swath_grid(6525, 6300, 2)
        assert "--margin-deg" in str(caught.value)

    def test_the_margin_runs_the_grid_past_the_tile(self):
        assert tile_prep.prep_bbox((-65.0, -30.0, -60.0, -25.0), 1.0) == (
            -66.0,
            -31.0,
            -59.0,
            -24.0,
        )

    def test_the_margin_stops_at_the_pole(self):
        _w, south, _e, north = tile_prep.prep_bbox((-65.0, -90.0, -60.0, 90.0), 1.0)
        assert (south, north) == (-90.0, 90.0)


class TestOneBlock:
    def test_every_scene_is_credited_to_itself(self, block, loaded):
        """The join. A positional label would scramble this completely."""
        result = tile_prep.prep_block(
            block, loaded, list(range(len(loaded))), "EPSG:4326", 1 / 900, RATIO, 1
        )
        ordered = sorted(loaded, key=destripe.timestamp_of)
        expected = [loaded.index(item) for item in ordered]
        assert list(result["global_idx"]) == expected

    def test_the_valid_counts_are_the_pixels_that_survived_the_qa_rule(
        self, block, loaded
    ):
        result = tile_prep.prep_block(
            block, loaded, list(range(len(loaded))), "EPSG:4326", 1 / 900, RATIO, 1
        )
        assert list(result["n_valid"]) == [NY * NX] * N_SCENES

    def test_coverage_counts_the_scenes_of_each_quad_that_reached_a_cell(
        self, block, loaded
    ):
        result = tile_prep.prep_block(
            block, loaded, list(range(len(loaded))), "EPSG:4326", 1 / 900, RATIO, 1
        )
        coverage = result["coverage"]
        assert set(coverage) == {(WEST, "030"), (EAST, "031")}
        for reach in coverage.values():
            assert reach.shape == (NY // RATIO, NX // RATIO)
            assert (reach == N_SCENES // 2).all()

    def test_the_block_reports_its_own_corner_on_the_swath_grid(self, loaded):
        offset_block = tile_prep.Shard(
            1, 1, 8, 16, NY, NX, (-62.0, -34.0, -61.9, -33.9)
        )
        result = tile_prep.prep_block(
            offset_block,
            loaded,
            list(range(len(loaded))),
            "EPSG:4326",
            1 / 900,
            RATIO,
            1,
        )
        assert (result["y0"], result["x0"]) == (4, 8)

    def test_the_histogram_holds_one_entry_per_valid_pixel(self, block, loaded):
        result = tile_prep.prep_block(
            block, loaded, list(range(len(loaded))), "EPSG:4326", 1 / 900, RATIO, 1
        )
        assert list(result["hist"].sum(axis=1)) == [NY * NX] * N_SCENES

    def test_it_returns_no_pixels(self, block, loaded):
        """A block's own array must not cross the worker boundary."""
        result = tile_prep.prep_block(
            block, loaded, list(range(len(loaded))), "EPSG:4326", 1 / 900, RATIO, 1
        )
        big = [
            k for k, v in result.items() if isinstance(v, np.ndarray) and v.ndim == 3
        ]
        assert big == []


class TestSplittingTheGrid:
    """The claim that buys back the second source traversal."""

    def _accumulate(self, results, n_scenes, swath_shape):
        hist = np.zeros((n_scenes, destripe.N_ANOMALY_BINS), dtype="uint32")
        n_valid = np.zeros(n_scenes, dtype="int64")
        quad_count: dict = {}
        for result in results:
            tile_prep.accumulate(result, hist, n_valid, quad_count, swath_shape)
        return hist, n_valid, quad_count

    def test_two_blocks_accumulate_into_one_tile(self):
        items = make_items()
        ordered = sorted(items, key=destripe.timestamp_of)
        gidx = np.array([items.index(i) for i in ordered])
        left = {
            "y0": 0,
            "x0": 0,
            "global_idx": gidx,
            "hist": np.ones((N_SCENES, destripe.N_ANOMALY_BINS), dtype="uint32"),
            "n_valid": np.full(N_SCENES, 5, dtype="int64"),
            "coverage": {(WEST, "030"): np.full((4, 4), 3, dtype="uint16")},
        }
        right = dict(left, x0=4, coverage={(WEST, "030"): np.full((4, 4), 2, "uint16")})

        hist, n_valid, quad_count = self._accumulate([left, right], N_SCENES, (4, 8))
        assert (hist == 2).all()
        assert (n_valid == 10).all()
        assert (quad_count[(WEST, "030")][:, :4] == 3).all()
        assert (quad_count[(WEST, "030")][:, 4:] == 2).all()

    def test_a_block_overhanging_the_swath_grid_is_clipped(self):
        result = {
            "y0": 2,
            "x0": 2,
            "global_idx": np.array([0]),
            "hist": np.zeros((1, destripe.N_ANOMALY_BINS), dtype="uint32"),
            "n_valid": np.zeros(1, dtype="int64"),
            "coverage": {(WEST, "030"): np.full((4, 4), 1, dtype="uint16")},
        }
        _hist, _n, quad_count = self._accumulate([result], 1, (4, 4))
        assert quad_count[(WEST, "030")].shape == (4, 4)
        assert quad_count[(WEST, "030")][2:, 2:].sum() == 4

    def test_the_offsets_do_not_depend_on_where_the_block_edges_fall(self):
        """Split the same field three ways and read the same offsets back."""
        months = np.repeat(np.arange(1, 13), 3)
        rng = np.random.default_rng(19)
        bias = rng.normal(0.0, 4.0, months.size)
        field = np.linspace(20.0, 45.0, 64, dtype="float32").reshape(8, 8)
        stack = np.stack(
            [
                field + 8.0 * np.cos((m - 1) * np.pi / 6.0) + b
                for m, b in zip(months, bias, strict=True)
            ]
        ).astype("float32")

        answers = []
        for edge in (8, 4, 2):
            hist = np.zeros((months.size, destripe.N_ANOMALY_BINS), dtype="uint32")
            n_valid = np.zeros(months.size, dtype="int64")
            for y0 in range(0, 8, edge):
                for x0 in range(0, 8, edge):
                    piece = stack[:, y0 : y0 + edge, x0 : x0 + edge]
                    planes, ref = destripe.month_climatology(piece, months)
                    destripe.accumulate_anomaly(
                        hist, n_valid, piece, months, planes, ref
                    )
            answers.append(destripe.offsets_from_histograms(hist))

        for other in answers[1:]:
            assert np.abs(other - answers[0]).max() <= destripe.ANOMALY_BIN_C


class TestEveryPathNeedsASwath:
    """A path with no swath cell would load its scenes and composite none.

    `feathered_percentile` reduces one subset per path the prep file names. A
    scene whose path is not on that list enters no subset, so its observations
    reach `qa_count` and not the temperature, and where another path covers the
    same ground the value is fitted without them. That is a wrong number rather
    than a missing one, so the prep run stops instead.
    """

    def test_a_path_with_no_swath_cell_stops_the_run(self):
        items = make_items()
        with pytest.raises(SystemExit, match="reached no swath cell"):
            tile_prep.check_every_path_has_a_swath(items, (WEST,))

    def test_the_message_names_every_missing_path(self):
        items = make_items()
        with pytest.raises(SystemExit) as caught:
            tile_prep.check_every_path_has_a_swath(items, ())
        assert WEST in str(caught.value)
        assert EAST in str(caught.value)
        assert "--no-feather" in str(caught.value)

    def test_every_path_present_passes(self):
        items = make_items()
        assert tile_prep.check_every_path_has_a_swath(items, (WEST, EAST)) is None

    def test_a_path_the_scenes_never_carry_is_not_required(self):
        """The check is one-way. A prep file may name more paths than it needs.

        `swath_masks` only ever returns paths it saw, so this cannot happen
        today. Asserting it keeps the check from becoming an equality test,
        which would fail a tile whose margin caught a path the tile did not.
        """
        items = make_items()
        assert (
            tile_prep.check_every_path_has_a_swath(items, (WEST, EAST, "999")) is None
        )


class TestTheArtifact:
    def test_it_round_trips_every_field_a_slice_reads(self, tmp_path):
        payload = {
            "scene_ids": np.array(["SCENE00", "SCENE01"]),
            "offset": np.array([0.5, -3.25]),
            "n_valid": np.array([9000, 12]),
            "paths": np.array([WEST, EAST]),
            "weight": np.zeros((2, 4, 4), dtype="float32"),
            "inside": np.ones((2, 4, 4), dtype=bool),
        }
        tile_prep.write_artifact(tmp_path, payload, {"tile": "S30W065"})
        loaded = np.load(tmp_path / "tile-prep.npz")
        assert list(loaded["scene_ids"]) == ["SCENE00", "SCENE01"]
        assert list(loaded["offset"]) == [0.5, -3.25]
        assert loaded["inside"].dtype == bool
        assert (tmp_path / "tile-prep.json").exists()

    def test_the_scene_table_counts_each_quad(self):
        items = make_items()
        scene_ids, quads, quad_scenes = tile_prep.scene_table(items)
        assert len(scene_ids) == N_SCENES
        assert quad_scenes == {(WEST, "030"): 6, (EAST, "031"): 6}
        assert quads[0] == destripe.quad_of(items[0])

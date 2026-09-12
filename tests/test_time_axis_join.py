"""The one assumption in `destripe` that a synthetic loader cannot check.

Every per-scene quantity, the offset and the WRS path both, reaches a loaded
stack by matching `destripe.timestamp_of(item)` against the `time` coordinate
`odc.stac.stac_load` produced. `destripe.align_to_time` raises when a step has
no item, which is the right failure, and it is a failure the whole tile takes
at once.

`tests/test_destripe.py` and `tests/test_tile_prep.py` replace `stac_load` with
a fixture built to return exactly those stamps, so they assert the assumption
back at themselves. They cannot catch odc-stac rounding a stamp, taking a group
timestamp that is not the item's, or changing its rule in a minor release.
Only a real load can.

`nlebovits/landsat-lst` guards the same join by reimplementing odc-stac's
grouping from its private functions and comparing. This repository groups by
`landsat:scene_id`, where one step is one scene, so there is nothing to
reimplement. This is what replaces that guard.

Reads `s3://usgs-landsat`, which is requester-pays. Opt-in and capped:

    uv run pytest tests/test_time_axis_join.py -m s3
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = [pytest.mark.s3, pytest.mark.timeout(1800)]

import destripe  # noqa: E402

ARTIFACT = ROOT / "artifacts" / "tile_scene_inventory.parquet"

#: The window every other measurement in this repository was taken over.
TILE = "S30W065"
SHARD_BBOX = (-60.6, -33.9, -60.45, -33.75)

#: Scenes to load. Each one is two COG reads over a small window, and the
#: cap is the spend control.
MAX_SCENES = 6

CRS = "EPSG:4326"
RESOLUTION = 1.0 / 3600


@pytest.fixture(scope="module")
def loaded():
    """A real stack, and the inventory items it was built from."""
    if not ARTIFACT.exists():
        pytest.skip("run usgs_inventory.py to build artifacts/")

    import dask

    import composite
    import shard_lst_p95
    from tile_inventory import items_for_tile

    shard_lst_p95.configure_read_env("earth-search")
    inventory, boxes = items_for_tile(ARTIFACT, TILE)
    w, s, e, n = SHARD_BBOX
    idx = [
        i
        for i, (iw, isouth, ie, inorth) in enumerate(boxes)
        if iw < e and ie > w and isouth < n and inorth > s
    ]
    item_dicts = [inventory[i] for i in idx][:MAX_SCENES]
    if len(item_dicts) < 2:
        pytest.skip(f"only {len(item_dicts)} scenes over the fixed window")

    # The loader exactly as the graph opens it: time in one chunk.
    stack = composite.open_stack(
        item_dicts, SHARD_BBOX, crs=CRS, resolution=RESOLUTION, chunk=1024
    )
    with dask.config.set(scheduler="threads"):
        data = stack.compute()
    return item_dicts, data


class TestTheJoinHoldsOnRealData:
    def test_every_loaded_step_matches_an_item_stamp_exactly(self, loaded):
        """The assumption, stated as an equality rather than a tolerance.

        `align_to_time` compares nanosecond stamps for equality. A tolerance
        here would pass while the thing the code does fails.
        """
        item_dicts, data = loaded
        mine = {destripe.timestamp_of(d) for d in item_dicts}
        loaded_stamps = set(data["time"].values.astype("datetime64[ns]"))
        assert loaded_stamps <= mine, (
            f"{len(loaded_stamps - mine)} loaded steps match no item stamp. "
            f"odc-stac no longer takes the item datetime as the group "
            f"timestamp, and every per-scene join in destripe.py rests on it."
        )

    def test_one_loaded_step_is_one_scene(self, loaded):
        """What removes `path_of_steps` from this port.

        Grouping by scene id gives one step per scene, so the WRS path is a
        property of the item and no step can span two paths. The sibling groups
        by solar day, where a step can, and pays fifty lines to say so.
        """
        item_dicts, data = loaded
        assert data.sizes["time"] == len({destripe.scene_id_of(d) for d in item_dicts})

    def test_align_to_time_resolves_the_whole_axis(self, loaded):
        item_dicts, data = loaded
        labels = destripe.align_to_time(
            item_dicts,
            data["time"].values,
            value_of=destripe.path_of,
            what="WRS path",
            dtype=object,
        )
        assert len(labels) == data.sizes["time"]
        assert all(label and label.isdigit() for label in labels)

    def test_a_correction_reaches_the_scenes_it_was_fitted_for(self, loaded):
        """End to end: a per-scene value survives the trip to the loaded axis."""
        item_dicts, data = loaded
        offset = np.arange(len(item_dicts), dtype="float64")
        correction = {
            "offset": offset,
            "keep": np.ones(len(item_dicts), dtype=bool),
            "paths": (),
            "weight": None,
            "emit_pooled": False,
        }
        celsius = np.zeros(
            (data.sizes["time"], *data["lwir11"].shape[1:]), dtype="float32"
        )
        valid = np.ones(celsius.shape, dtype=bool)
        labels, rejected = destripe.apply_to_stack(
            celsius, valid, item_dicts, data["time"].values, correction
        )
        assert rejected == 0
        # Each step now carries minus the offset of its own scene, and the set
        # of shifts is the set of offsets, permuted by the loader's sort.
        applied = sorted(-float(celsius[t, 0, 0]) for t in range(celsius.shape[0]))
        expected = sorted(
            offset[i]
            for i, d in enumerate(item_dicts)
            if destripe.timestamp_of(d) in set(data["time"].values)
        )
        assert applied == expected
        assert len(labels) == celsius.shape[0]

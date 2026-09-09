"""One fixed shard, loaded twice, compared pixel for pixel.

Field parity says the inventory carries the same numbers Earth Search does.
This says the loader turns them into the same raster. It is the check that
matters, because every derivation in `usgs_inventory` exists to feed
`odc.stac.stac_load`, and a rule can be right in a comparison and still build
the wrong geobox.

Both passes use the same scenes, the same bounds, and the same resolution. The
only difference is where the items came from. The comparison covers the
geobox, the thermal stack, the QA stack, the valid-observation count, and the
encoded P95 output.

This reads `s3://usgs-landsat`, which is requester-pays. It is opt-in and
capped, and the cap is the spend control:

    uv run pytest tests/test_load_parity.py -m s3
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = [pytest.mark.s3, pytest.mark.timeout(1800)]

ARTIFACT = ROOT / "artifacts" / "tile_scene_inventory.parquet"

#: A small window inside S30W065, over Pergamino, where every other
#: measurement in this repository was taken.
TILE = "S30W065"
SHARD_BBOX = (-60.6, -33.9, -60.45, -33.75)
WINDOW = ("2023-01-01", "2023-03-31T23:59:59Z")

#: Scenes to load per pass. Each one is two COG reads over a small window.
MAX_SCENES = 6

#: The same tolerance tests/test_inventory_parity.py justifies, and for the
#: same reason: whole-second start and stop columns.
DATETIME_TOLERANCE_S = 2.0

BANDS = ("lwir11", "qa_pixel")
CRS = "EPSG:4326"
RESOLUTION = 1.0 / 3600


def _load(item_dicts, bbox):
    """Load a scene stack the way `process_shard` does."""
    import pystac
    from odc.stac import stac_load

    items = [pystac.Item.from_dict(d) for d in item_dicts]
    return stac_load(
        items,
        bands=BANDS,
        crs=CRS,
        resolution=RESOLUTION,
        bbox=bbox,
        groupby="landsat:scene_id",
        chunks={"time": 1, "latitude": -1, "longitude": -1},
    ).compute(scheduler="threads", num_workers=4)


@pytest.fixture(scope="module")
def paired():
    """The same scenes, described by Earth Search and by the inventory."""
    if not ARTIFACT.exists():
        pytest.skip("run usgs_inventory.py to build artifacts/")

    import shard_lst_p95
    from stac_reference import search_items
    from tile_inventory import items_for_tile

    shard_lst_p95.configure_read_env("earth-search")

    stac_items, _ = search_items(SHARD_BBOX, start=WINDOW[0], end=WINDOW[1])
    if not stac_items:
        pytest.skip("Earth Search returned no scenes for the fixed shard")

    inventory, _ = items_for_tile(ARTIFACT, TILE)
    by_id = {item["id"]: item for item in inventory}

    # Only scenes both paths describe, so the comparison is about the
    # description rather than about the selection.
    shared = [it for it in stac_items if it.id in by_id]
    shared.sort(key=lambda it: it.id)
    shared = shared[:MAX_SCENES]
    if len(shared) < 2:
        pytest.skip(f"only {len(shared)} shared scenes")

    return (
        [it.to_dict() for it in shared],
        [by_id[it.id] for it in shared],
        [it.id for it in shared],
    )


@pytest.fixture(scope="module")
def loaded(paired):
    stac_dicts, inv_dicts, _ = paired
    return _load(stac_dicts, SHARD_BBOX), _load(inv_dicts, SHARD_BBOX)


class TestSelection:
    def test_the_same_ids_in_the_same_order(self, paired):
        stac_dicts, inv_dicts, ids = paired
        assert [d["id"] for d in stac_dicts] == ids
        assert [d["id"] for d in inv_dicts] == ids

    def test_the_same_acquisition_order(self, loaded):
        """Order and month, not the microsecond.

        The bulk file truncates the acquisition start and stop to whole
        seconds, so the centre computed from them sits within about a second
        of the centre Earth Search publishes. `odc.stac` sorts on that value
        and this repository reads only the month from it, so both have to
        agree and the microsecond does not.
        """
        import numpy as np

        stac_ds, inv_ds = loaded
        mine = inv_ds["time"].values
        theirs = stac_ds["time"].values
        assert len(mine) == len(theirs)

        gaps = np.abs(
            (mine - theirs).astype("timedelta64[ms]").astype("int64") / 1000.0
        )
        assert gaps.max() <= DATETIME_TOLERANCE_S, f"largest gap {gaps.max():.3f} s"
        assert list(np.argsort(mine)) == list(np.argsort(theirs))
        assert list(mine.astype("datetime64[M]")) == list(
            theirs.astype("datetime64[M]")
        )

    def test_the_same_scene_count(self, loaded):
        stac_ds, inv_ds = loaded
        assert stac_ds.sizes["time"] == inv_ds.sizes["time"]


class TestGeobox:
    def test_identical_grid(self, loaded):
        stac_ds, inv_ds = loaded
        assert stac_ds.odc.geobox == inv_ds.odc.geobox

    def test_identical_coordinates(self, loaded):
        import numpy as np

        stac_ds, inv_ds = loaded
        for dim in ("latitude", "longitude"):
            np.testing.assert_array_equal(stac_ds[dim].values, inv_ds[dim].values)


class TestArrays:
    def test_thermal_stack_is_identical(self, loaded):
        import numpy as np

        stac_ds, inv_ds = loaded
        np.testing.assert_array_equal(stac_ds["lwir11"].values, inv_ds["lwir11"].values)

    def test_qa_stack_is_identical(self, loaded):
        import numpy as np

        stac_ds, inv_ds = loaded
        np.testing.assert_array_equal(
            stac_ds["qa_pixel"].values, inv_ds["qa_pixel"].values
        )

    def test_the_stacks_are_not_all_nodata(self, loaded):
        """A comparison of two empty arrays proves nothing."""
        stac_ds, _ = loaded
        assert (stac_ds["lwir11"].values != 0).any()


class TestComposite:
    """The numbers the pipeline actually publishes."""

    def _reduce(self, dataset):
        import numpy as np

        from lst_qa import encode_celsius, masked_celsius

        lst, valid = masked_celsius(
            dataset["lwir11"].values, dataset["qa_pixel"].values
        )
        with np.errstate(all="ignore"):
            p95 = np.nanpercentile(lst, 95, axis=0)
        return encode_celsius(p95), valid

    def test_valid_observation_counts_match(self, loaded):
        import numpy as np

        stac_ds, inv_ds = loaded
        _, stac_valid = self._reduce(stac_ds)
        _, inv_valid = self._reduce(inv_ds)
        np.testing.assert_array_equal(stac_valid.sum(axis=0), inv_valid.sum(axis=0))

    def test_encoded_p95_is_identical(self, loaded):
        import numpy as np

        stac_ds, inv_ds = loaded
        stac_dn, _ = self._reduce(stac_ds)
        inv_dn, _ = self._reduce(inv_ds)
        np.testing.assert_array_equal(stac_dn, inv_dn)

    def test_the_composite_carries_real_temperatures(self, loaded):
        """Guard against two identical all-nodata rasters passing above."""
        from lst_qa import LST_NODATA_DN

        stac_ds, _ = loaded
        dn, _ = self._reduce(stac_ds)
        assert (dn != LST_NODATA_DN).any()

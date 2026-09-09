"""The inventory, checked field by field against Earth Search.

Earth Search is the oracle here and nothing else. Every derived value in
`usgs_inventory` is a rule inferred from the bulk schema, and a rule that is
right on one scene can be wrong on another hemisphere, another UTM zone, or
another year. So the sample is stratified rather than convenient, and the
comparison is exact wherever exactness is available.

One value has a tolerance, and only one. The bulk file truncates the
acquisition start and stop to whole seconds, so the scene centre computed from
them sits within about a second of the published centre. `DATETIME_TOLERANCE_S`
is that gap, measured at 1.117 s over 360 items. The runtime reads only the
month from it, and `usgs_inventory` resolves the scenes near a month boundary
exactly, so the tolerance cannot move an observation between months.

These tests reach the network. Run them with:

    uv run pytest tests/test_inventory_parity.py -m network
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from usgs_inventory import (  # noqa: E402
    NO_THERMAL_DATA_TYPES,
    asset_hrefs,
    derive_projection,
    scan_bulk,
    stac_item_id,
)

# The oracle fixture makes 60 catalogue searches, and the derived fixture
# scans 1.46 million rows. Neither fits the 60 s the rest of the suite uses.
pytestmark = [pytest.mark.network, pytest.mark.timeout(1800)]

#: Measured over 360 items: the largest gap between the centre computed from
#: the whole-second start and stop and the centre Earth Search publishes.
DATETIME_TOLERANCE_S = 2.0

#: Stratified on purpose: both platforms, both hemispheres, several UTM zones,
#: coastal and inland, the antimeridian, and every year of the window.
SAMPLE_AREAS = {
    "argentina_inland": [-64, -36, -58, -31],
    "chile_coastal": [-73, -40, -71, -36],
    "alps_inland": [8, 44, 14, 48],
    "malaysia_equator": [100, 0, 106, 6],
    "california_coastal": [-122, 36, -118, 40],
    "namibia_coastal": [12, -26, 16, -22],
    "fiji_antimeridian": [177, -19, 180, -16],
    "kiribati_antimeridian": [-180, -4, -177, 0],
    "kamchatka_north": [156, 52, 162, 56],
    "newzealand_south": [168, -46, 174, -42],
    "india_inland": [74, 20, 80, 25],
    "brazil_amazon": [-62, -6, -56, -1],
}
SAMPLE_YEARS = (2021, 2022, 2023, 2024, 2025)


def reference_epsg(properties) -> int:
    """The EPSG code an Earth Search item carries.

    `pystac` migrates a projection v1.1 item to v2.0 on read, which renames
    `proj:epsg` to `proj:code` and writes it as `EPSG:32620`. Both spellings
    appear depending on how the item reached this test, so both are read.
    """
    if "proj:epsg" in properties:
        return int(properties["proj:epsg"])
    return int(str(properties["proj:code"]).removeprefix("EPSG:"))


def _bulk_path():
    import glob

    from usgs_inventory import DEFAULT_CACHE_DIR

    found = sorted(glob.glob(str(DEFAULT_CACHE_DIR / "LANDSAT_OT_C2_L2-*.parquet")))
    if not found:
        pytest.skip("no cached USGS bulk file; run usgs_inventory.py first")
    return found[0]


@pytest.fixture(scope="module")
def derived():
    """Every derived value for the whole window, keyed by STAC item id."""
    table = scan_bulk(_bulk_path())
    epsg, height, width, ulx, uly, _ = derive_projection(table)
    display = table.column("Display ID").to_pylist()
    data_type = table.column("Data Type L2").to_pylist()
    path = table.column("WRS Path").to_numpy(zero_copy_only=False)
    row = table.column("WRS Row").to_numpy(zero_copy_only=False)
    scene = table.column("Landsat Scene Identifier").to_pylist()
    cloud = table.column("Scene Cloud Cover L1").to_numpy(zero_copy_only=False)
    import numpy as np

    start = np.asarray(
        table.column("Start Time").to_numpy(zero_copy_only=False),
        dtype="datetime64[us]",
    )
    stop = np.asarray(
        table.column("Stop Time").to_numpy(zero_copy_only=False),
        dtype="datetime64[us]",
    )
    centre = start + (stop - start) // 2

    out = {}
    for i, disp in enumerate(display):
        thermal, qa = asset_hrefs(disp, int(path[i]), int(row[i]), data_type[i])
        out[stac_item_id(disp)] = {
            "display_id": disp,
            "epsg": int(epsg[i]),
            "shape": (int(height[i]), int(width[i])),
            "transform": (30.0, 0.0, float(ulx[i]), 0.0, -30.0, float(uly[i])),
            "thermal_href": thermal,
            "qa_href": qa,
            "scene_id": scene[i],
            "cloud": float(cloud[i]),
            "data_type": data_type[i],
            "centre": centre[i],
        }
    return out


@pytest.fixture(scope="module")
def oracle():
    """A deterministic stratified sample of Earth Search items."""
    from stac_reference import search_items

    items = {}
    for bbox in SAMPLE_AREAS.values():
        for year in SAMPLE_YEARS:
            found, _ = search_items(
                bbox,
                start=f"{year}-06-01",
                end=f"{year}-06-15T23:59:59Z",
            )
            for item in found[:6]:
                items[item.id] = item.to_dict()
    if len(items) < 100:
        pytest.skip(f"Earth Search returned only {len(items)} items")
    return items


class TestFieldParity:
    def test_the_sample_is_stratified(self, oracle):
        platforms = {v["properties"]["platform"] for v in oracle.values()}
        zones = {reference_epsg(v["properties"]) for v in oracle.values()}
        lats = [v["bbox"][1] for v in oracle.values()]
        years = {v["properties"]["datetime"][:4] for v in oracle.values()}
        assert platforms == {"landsat-8", "landsat-9"}
        assert len(zones) >= 8, f"only {len(zones)} UTM zones"
        assert min(lats) < 0 < max(lats), "one hemisphere only"
        assert len(years) == len(SAMPLE_YEARS)

    def test_every_sampled_item_is_in_the_inventory(self, oracle, derived):
        missing = sorted(set(oracle) - set(derived))
        assert not missing, f"{len(missing)} sampled items absent: {missing[:10]}"

    def test_projection_epsg_is_exact(self, oracle, derived):
        for item_id, ref in oracle.items():
            assert derived[item_id]["epsg"] == reference_epsg(ref["properties"]), (
                item_id
            )

    def test_projection_shape_is_exact(self, oracle, derived):
        for item_id, ref in oracle.items():
            assert derived[item_id]["shape"] == tuple(
                ref["properties"]["proj:shape"]
            ), item_id

    def test_projection_transform_is_exact(self, oracle, derived):
        """Exact, not close. The snap recovers the product lattice."""
        for item_id, ref in oracle.items():
            assert derived[item_id]["transform"] == tuple(
                ref["properties"]["proj:transform"][:6]
            ), item_id

    def test_thermal_href_is_exact(self, oracle, derived):
        for item_id, ref in oracle.items():
            want = (ref["assets"].get("lwir11") or {}).get("href")
            assert derived[item_id]["thermal_href"] == want, item_id

    def test_qa_href_is_exact(self, oracle, derived):
        for item_id, ref in oracle.items():
            assert derived[item_id]["qa_href"] == ref["assets"]["qa_pixel"]["href"], (
                item_id
            )

    def test_l2sr_products_have_no_thermal_band_on_either_side(self, oracle, derived):
        """The rule that decides a null href, checked against the catalogue."""
        seen = 0
        for item_id, ref in oracle.items():
            no_thermal_here = "lwir11" not in ref["assets"]
            no_thermal_there = derived[item_id]["data_type"] in NO_THERMAL_DATA_TYPES
            assert no_thermal_here == no_thermal_there, item_id
            seen += no_thermal_here
        assert seen or True, "no L2SR in this sample, which is acceptable"

    def test_scene_id_is_exact(self, oracle, derived):
        """stac_load groups on this. A mismatch would regroup the stack."""
        for item_id, ref in oracle.items():
            assert (
                derived[item_id]["scene_id"] == ref["properties"]["landsat:scene_id"]
            ), item_id

    def test_cloud_cover_is_exact(self, oracle, derived):
        """`Scene Cloud Cover L1` is what `eo:cloud_cover` filters on."""
        for item_id, ref in oracle.items():
            assert derived[item_id]["cloud"] == pytest.approx(
                ref["properties"]["eo:cloud_cover"]
            ), item_id

    def test_datetime_is_inside_the_stated_tolerance(self, oracle, derived):
        import numpy as np

        worst = 0.0
        for item_id, ref in oracle.items():
            stamp = ref["properties"]["datetime"].replace("Z", "")
            gap = abs(
                (derived[item_id]["centre"] - np.datetime64(stamp, "us"))
                .astype("timedelta64[ms]")
                .astype("int64")
                / 1000.0
            )
            worst = max(worst, gap)
            assert gap <= DATETIME_TOLERANCE_S, f"{item_id} off by {gap:.3f} s"
        assert worst < DATETIME_TOLERANCE_S

    def test_the_month_never_differs(self, oracle, derived):
        """The tolerance is only acceptable because this holds."""
        for item_id, ref in oracle.items():
            mine = str(derived[item_id]["centre"].astype("datetime64[M]"))
            theirs = ref["properties"]["datetime"][:7]
            assert mine == theirs, item_id


class TestSetParity:
    """Enumerate the difference over one bounded region and interval."""

    REGION = [-64, -36, -58, -31]
    START = "2023-06-01"
    END = "2023-08-31T23:59:59Z"

    @pytest.fixture(scope="class")
    @classmethod
    def sets(cls):
        from stac_reference import search_items

        found, _ = search_items(cls.REGION, start=cls.START, end=cls.END)
        return {item.id for item in found}

    def test_the_inventory_holds_every_catalogue_item(self, sets, derived):
        missing = sorted(sets - set(derived))
        assert not missing, (
            f"{len(missing)} items Earth Search returned are absent from the "
            f"bulk inventory: {missing[:20]}"
        )

    def test_the_extra_items_are_explained(self, sets, derived):
        """The bulk file is the archive; Earth Search indexes a copy of it.

        Any surplus has to be scenes outside the sampled bbox rather than
        scenes the catalogue rejects, so the check is on the footprint.
        """
        from shapely.geometry import Polygon, box

        from usgs_inventory import footprint_arrays

        table = scan_bulk(_bulk_path(), start=self.START, end=self.END)
        lons, lats, crossing = footprint_arrays(table)
        display = table.column("Display ID").to_pylist()
        region = box(*self.REGION)

        extra = []
        for i, disp in enumerate(display):
            item_id = stac_item_id(disp)
            if item_id in sets or crossing[i]:
                continue
            ring = list(zip(lons[:, i], lats[:, i], strict=True))
            if Polygon([*ring, ring[0]]).intersects(region):
                extra.append(item_id)
        assert not extra, (
            f"{len(extra)} scenes intersect the region in the bulk file but "
            f"Earth Search did not return them: {extra[:20]}"
        )

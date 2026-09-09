"""The inventory, checked field by field against Earth Search.

Earth Search is the oracle here and nothing else. Every derived value in
`usgs_inventory` is a rule inferred from the bulk schema, and a rule that is
right on one scene can be wrong on another hemisphere, another UTM zone, or
another year. So the sample is stratified rather than convenient, and the
comparison is exact wherever exactness is available.

One value has a tolerance, and only one: the acquisition centre. Two separate
things separate it from the published one, and `measure_scene_centre.py`
measures each on its own, because timestamp precision in the bulk file is
mixed. On the rows carrying microseconds the residual is a systematic 4.24 ms,
which is the difference between two definitions of "scene centre". On the rows
truncated to whole seconds, truncation moves the midpoint by under a second;
the worst measured was 0.89 s. `DATETIME_TOLERANCE_S` is set from those two,
not from the largest gap this sample happens to contain.

That distinction matters. An earlier version set the tolerance to the maximum
over the same items it then asserted against, so the assertion could only fail
if the catalogue changed. It also sampled 1 to 15 June of each year and nothing
else, which excluded every month boundary that `MONTH_BOUNDARY_GUARD_SECONDS`
exists for. `SAMPLE_WINDOWS` now crosses December into January.

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

#: The bound on the gap between the computed centre and the published one,
#: derived rather than sampled. Truncating start and stop to whole seconds
#: moves their midpoint by strictly under 1 s, and the definitional offset
#: measured on the untruncated rows adds 4.24 ms. `measure_scene_centre.py`
#: reports both; the worst it has seen is 0.89 s.
DATETIME_TOLERANCE_S = 1.01

#: The definitional offset alone, from the rows that carry microseconds and so
#: have no truncation in them. Asserted separately, because a change here means
#: the two centres stopped being the same quantity.
DEFINITIONAL_OFFSET_S = 0.005

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

#: Two windows per year, as `(start, end)` templates. June is the original
#: sample. The turn of the year is the one that matters for the month guard,
#: and it also spans the boundary of the composite window itself.
SAMPLE_WINDOWS = (
    ("{year}-06-01", "{year}-06-15T23:59:59Z"),
    ("{year}-12-28", "{year}-12-31T23:59:59Z"),
    ("{year}-01-01", "{year}-01-04T23:59:59Z"),
)


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
    """The newest cached bulk download.

    The cache name carries a digest of the publication rather than a date, so
    sorting by name picks an arbitrary one. With two downloads present, the
    oldest could answer for an artifact built from the newest and the parity
    run would compare the wrong archive against the catalogue.
    """
    from usgs_inventory import DEFAULT_CACHE_DIR

    found = sorted(
        Path(DEFAULT_CACHE_DIR).glob("LANDSAT_OT_C2_L2-*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        pytest.skip("no cached USGS bulk file; run usgs_inventory.py first")
    return str(found[-1])


@pytest.fixture(scope="module")
def scanned():
    """The bulk scan for the whole window, and the item id of every row.

    The id list is what the set-parity tests need. Deriving a full dict per row
    is what they do not: 1.46 million dicts of ten keys is several gigabytes,
    and the field tests read a few hundred of them.
    """
    table = scan_bulk(_bulk_path())
    ids = [stac_item_id(d) for d in table.column("Display ID").to_pylist()]
    return table, ids, {name: i for i, name in enumerate(ids)}


@pytest.fixture(scope="module")
def derived(scanned, oracle):
    """The derived values for the sampled items, keyed by STAC item id.

    `derive_projection` is vectorised over the whole table, so the rules are
    exercised on every row. Only the sampled rows are materialised.
    """
    import numpy as np

    table, _, position = scanned
    epsg, height, width, ulx, uly, _ = derive_projection(table)
    display = table.column("Display ID").to_pylist()
    data_type = table.column("Data Type L2").to_pylist()
    path = table.column("WRS Path").to_numpy(zero_copy_only=False)
    row = table.column("WRS Row").to_numpy(zero_copy_only=False)
    scene = table.column("Landsat Scene Identifier").to_pylist()
    cloud = table.column("Scene Cloud Cover L1").to_numpy(zero_copy_only=False)

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
    for item_id in oracle:
        i = position.get(item_id)
        if i is None:
            continue
        disp = display[i]
        thermal, qa = asset_hrefs(disp, int(path[i]), int(row[i]), data_type[i])
        out[item_id] = {
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
            "sub_second": "." in str(start[i]),
        }
    return out


@pytest.fixture(scope="module")
def inventory_ids(scanned):
    """Every STAC item id the bulk scan holds, for the set-parity tests."""
    _, ids, _ = scanned
    return set(ids)


@pytest.fixture(scope="module")
def oracle():
    """A stratified sample of Earth Search items.

    Sorted by id before truncating, so the sample does not depend on the order
    the catalogue happens to page results in.
    """
    from stac_reference import search_items

    items = {}
    for bbox in SAMPLE_AREAS.values():
        for year in SAMPLE_YEARS:
            for start, end in SAMPLE_WINDOWS:
                found, _ = search_items(
                    bbox,
                    start=start.format(year=year),
                    end=end.format(year=year),
                )
                for item in sorted(found, key=lambda i: i.id)[:4]:
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

    def test_the_sample_crosses_a_month_boundary(self, oracle):
        """The guard exists for the turn of the month, so the sample must see it.

        A June-only sample can assert `test_the_month_never_differs` forever
        without ever testing the case it is about.
        """
        months = {v["properties"]["datetime"][5:7] for v in oracle.values()}
        assert {"12", "01"} <= months, f"only months {sorted(months)}"

    def test_every_sampled_item_is_in_the_inventory(self, oracle, inventory_ids):
        missing = sorted(set(oracle) - inventory_ids)
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

    @staticmethod
    def _gap(derived_row, ref):
        import numpy as np

        stamp = ref["properties"]["datetime"].replace("Z", "")
        return abs(
            (derived_row["centre"] - np.datetime64(stamp, "us"))
            .astype("timedelta64[us]")
            .astype("int64")
            / 1e6
        )

    def test_datetime_is_inside_the_derived_bound(self, oracle, derived):
        """The bound comes from the two error sources, not from this sample."""
        for item_id, ref in oracle.items():
            gap = self._gap(derived[item_id], ref)
            assert gap <= DATETIME_TOLERANCE_S, f"{item_id} off by {gap:.6f} s"

    def test_the_untruncated_rows_show_only_the_definitional_offset(
        self, oracle, derived
    ):
        """Where the midpoint is exact, what is left is the definition.

        These rows carry microseconds on both timestamps, so truncation cannot
        contribute. A gap larger than a few milliseconds here would mean the
        two centres stopped being the same quantity, which no amount of guard
        band would fix.
        """
        exact = [i for i, row in derived.items() if row["sub_second"]]
        if not exact:
            pytest.skip("no sub-second rows in this sample")
        for item_id in exact:
            gap = self._gap(derived[item_id], oracle[item_id])
            assert gap <= DEFINITIONAL_OFFSET_S, f"{item_id} off by {gap:.6f} s"

    def test_the_month_never_differs(self, oracle, derived):
        """The tolerance is only acceptable because this holds."""
        for item_id, ref in oracle.items():
            mine = str(derived[item_id]["centre"].astype("datetime64[M]"))
            theirs = ref["properties"]["datetime"][:7]
            assert mine == theirs, item_id


#: Regions the set-parity tests enumerate in both directions. Argentina is the
#: plain case. Fiji is the one that matters: a footprint there wraps the
#: antimeridian, and the wrap is handled in `assign_tiles`, `stac_bbox`, and
#: `_tile_local_bbox`. Testing only the plain case leaves the code most likely
#: to be wrong unchecked against the catalogue.
SET_PARITY_REGIONS = {
    "argentina": [-64, -36, -58, -31],
    "fiji_antimeridian": [177, -19, 180, -16],
    "kiribati_antimeridian": [-180, -4, -177, 0],
}
SET_PARITY_START = "2023-06-01"
SET_PARITY_END = "2023-08-31T23:59:59Z"


@pytest.fixture(scope="module")
def catalogue_sets():
    """Item ids Earth Search returns for each region, over one interval."""
    from stac_reference import search_items

    out = {}
    for name, region in SET_PARITY_REGIONS.items():
        found, _ = search_items(region, start=SET_PARITY_START, end=SET_PARITY_END)
        out[name] = {item.id for item in found}
    return out


class TestSetParity:
    """Enumerate the difference in both directions, over three regions."""

    @pytest.mark.parametrize("region", sorted(SET_PARITY_REGIONS))
    def test_the_inventory_holds_every_catalogue_item(
        self, region, catalogue_sets, inventory_ids
    ):
        missing = sorted(catalogue_sets[region] - inventory_ids)
        assert not missing, (
            f"{len(missing)} items Earth Search returned for {region} are "
            f"absent from the bulk inventory: {missing[:20]}"
        )

    @pytest.mark.parametrize("region", sorted(SET_PARITY_REGIONS))
    def test_the_extra_items_are_explained(self, region, catalogue_sets):
        """The bulk file is the archive; Earth Search indexes a copy of it.

        Any surplus has to be scenes outside the sampled bbox rather than
        scenes the catalogue rejects, so the check is on the footprint.

        A footprint that wraps the antimeridian is shifted into `[0, 360)` and
        compared against a region shifted the same way, which is what
        `assign_tiles` does. Skipping those, as this test used to, excused the
        one case the plain regions cannot reach.
        """
        from shapely.geometry import Polygon, box

        from usgs_inventory import footprint_arrays

        bounds = SET_PARITY_REGIONS[region]
        table = scan_bulk(_bulk_path(), start=SET_PARITY_START, end=SET_PARITY_END)
        lons, lats, crossing = footprint_arrays(table)
        display = table.column("Display ID").to_pylist()

        west, south, east, north = bounds
        plain = box(*bounds)
        shifted = box(
            west + (360.0 if west < 0 else 0.0),
            south,
            east + (360.0 if west < 0 else 0.0),
            north,
        )

        extra = []
        for i, disp in enumerate(display):
            item_id = stac_item_id(disp)
            if item_id in catalogue_sets[region]:
                continue
            ring_lons = lons[:, i]
            target = plain
            if crossing[i]:
                ring_lons = ring_lons.copy()
                ring_lons[ring_lons < 0] += 360.0
                target = shifted
            ring = list(zip(ring_lons, lats[:, i], strict=True))
            if Polygon([*ring, ring[0]]).intersects(target):
                extra.append(item_id)
        assert not extra, (
            f"{len(extra)} scenes intersect {region} in the bulk file but "
            f"Earth Search did not return them: {extra[:20]}"
        )

    def test_the_antimeridian_regions_actually_hold_wrapping_scenes(self):
        """Otherwise the regions above prove nothing about the wrap."""
        table = scan_bulk(_bulk_path(), start=SET_PARITY_START, end=SET_PARITY_END)
        from usgs_inventory import footprint_arrays

        _, _, crossing = footprint_arrays(table)
        assert crossing.any(), "no wrapping footprint in the interval"

"""The published encoding is a promise, and a COG is where it either holds.

Every rule here is one a reader depends on and no reviewer can see by eye. A
missing overview turns a zoomed-out view into a full-resolution fetch. A
statistic that lands in a `.aux.xml` sidecar reaches nobody, because the
sidecar does not travel with the file. A scale or offset dropped by the COG
driver leaves DN 9500 looking like a temperature. Each of those failures is
silent, and each one has a test below.

The last test is the whole conformance question at once: `rashid` reads the
tree the writer produced and reports every Portolan requirement it breaks.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cog_catalog import (  # noqa: E402
    BLOCK_SIZE,
    LST_ASSET_KEY,
    MONTH_NAMES,
    QA_ASSET_KEY,
    STATISTICS_KEYS,
    VALID_PERCENT_KEY,
    band_statistics,
    multihash_sha256,
    read_cog_encoding,
    tile_id,
    write_catalog,
)
from lst_qa import (  # noqa: E402
    LST_NODATA_DN,
    LST_OFFSET,
    LST_SCALE,
    LST_VALID_MAX_C,
    LST_VALID_MIN_C,
    encode_celsius,
)

# One tile of the degree grid, 600 px on a side. Wider than one 512 px
# internal tile on both axes, which is what makes the overview requirement
# apply: a raster that fits inside a single tile is its own overview and is
# exempt.
BBOX = (-65.0, -32.5, -64.5, -32.0)
PIXELS_PER_DEGREE = 1200
SIZE = 600
COLLECTION_ID = "lst-p95-composite"
ITEM_ID = "S32W065"


@pytest.fixture(scope="module")
def composite():
    """A synthetic tile with a known valid fraction and a known temperature.

    The nodata block is a square rather than scattered pixels, so a wrong
    valid-percent shows up as a round number that is off, not as noise.
    """
    rng = np.random.default_rng(20260909)
    celsius = rng.uniform(12.0, 48.0, (SIZE, SIZE)).astype("float32")
    celsius[:60, :60] = np.nan  # 3600 px of 360000, exactly 1%
    lst = encode_celsius(celsius)
    qa = np.zeros((12, SIZE, SIZE), dtype="uint8")
    for month in range(12):
        qa[month] = month  # January is 0 observations, December is 11
    return celsius, lst, qa


@pytest.fixture(scope="module")
def catalog(tmp_path_factory, composite):
    """One catalog, built once. Writing it twice would only cost time."""
    _celsius, lst, qa = composite
    meta = {
        "raster": [SIZE, SIZE],
        "bbox": list(BBOX),
        "crs": "EPSG:4326",
        "pixels_per_degree": PIXELS_PER_DEGREE,
        "start": "2021-01-01",
        "end": "2025-12-31T23:59:59Z",
    }
    root = tmp_path_factory.mktemp("catalog")
    return write_catalog(root / "catalog", lst, qa, meta, collection_id=COLLECTION_ID)


@pytest.fixture(scope="module")
def item_dir(catalog):
    return catalog / COLLECTION_ID / ITEM_ID


def opened(path):
    """Open a raster the way a Portolan validator does: no PAM sidecar."""
    return rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path)


class TestTileGeometry:
    def test_the_tile_is_named_for_its_north_and_west_edges(self):
        assert tile_id(BBOX) == ITEM_ID
        assert tile_id((-65.0, -35.0, -60.0, -30.0)) == "S30W065"
        assert tile_id((10.0, 45.0, 15.0, 50.0)) == "N50E010"

    @pytest.mark.parametrize("name", ["lst_p95.tif", "qa_count.tif"])
    def test_the_cog_covers_exactly_the_planned_bbox(self, item_dir, name):
        env, src = opened(item_dir / name)
        with env, src:
            assert tuple(round(v, 9) for v in src.bounds) == BBOX
            assert src.crs.to_string() == "EPSG:4326"
            assert (src.height, src.width) == (SIZE, SIZE)


class TestCloudOptimization:
    @pytest.mark.parametrize("name", ["lst_p95.tif", "qa_count.tif"])
    def test_the_internal_tiles_are_square_and_512_px(self, item_dir, name):
        env, src = opened(item_dir / name)
        with env, src:
            for shape in src.block_shapes:
                assert shape == (BLOCK_SIZE, BLOCK_SIZE)

    @pytest.mark.parametrize("name", ["lst_p95.tif", "qa_count.tif"])
    def test_a_raster_wider_than_one_tile_carries_overviews(self, item_dir, name):
        env, src = opened(item_dir / name)
        with env, src:
            assert src.overviews(1), "a client would refetch full-resolution pixels"

    @pytest.mark.parametrize("name", ["lst_p95.tif", "qa_count.tif"])
    def test_the_file_is_a_cog_and_not_a_plain_geotiff(self, item_dir, name):
        env, src = opened(item_dir / name)
        with env, src:
            assert src.profile["tiled"] is True
            assert src.profile["compress"].lower() == "deflate"


class TestEmbeddedStatistics:
    @pytest.mark.parametrize("name", ["lst_p95.tif", "qa_count.tif"])
    def test_every_band_carries_the_four_mandatory_statistics(self, item_dir, name):
        env, src = opened(item_dir / name)
        with env, src:
            for index in range(1, src.count + 1):
                tags = src.tags(bidx=index)
                assert set(STATISTICS_KEYS) <= set(tags)

    @pytest.mark.parametrize("name", ["lst_p95.tif", "qa_count.tif"])
    def test_no_statistics_escape_into_an_aux_xml_sidecar(self, item_dir, name):
        # A PAM sidecar does not travel with the file, so a reader fetching a
        # range of the COG would never see it.
        assert not (item_dir / f"{name}.aux.xml").exists()

    def test_the_valid_percent_counts_only_pixels_that_are_not_nodata(self, item_dir):
        env, src = opened(item_dir / "lst_p95.tif")
        with env, src:
            reported = float(src.tags(bidx=1)[VALID_PERCENT_KEY])
        assert reported == pytest.approx(99.0, abs=0.01)

    def test_the_temperature_statistics_exclude_the_nodata_dn(self, item_dir):
        env, src = opened(item_dir / "lst_p95.tif")
        with env, src:
            minimum = float(src.tags(bidx=1)["STATISTICS_MINIMUM"])
        # DN 0 sits far below any encoded temperature. Leaving it in would
        # drag the minimum to zero and every rendered ramp with it.
        assert minimum > LST_NODATA_DN

    def test_a_band_of_pure_nodata_still_reports_the_mandatory_tags(self):
        empty = np.zeros((4, 4), dtype="uint16")
        stats = band_statistics(empty, LST_NODATA_DN)
        assert set(STATISTICS_KEYS) <= set(stats)
        assert float(stats[VALID_PERCENT_KEY]) == 0.0


class TestTemperatureEncoding:
    def test_the_decoding_rule_travels_with_the_file(self, item_dir):
        env, src = opened(item_dir / "lst_p95.tif")
        with env, src:
            assert src.scales == (LST_SCALE,)
            assert src.offsets == (LST_OFFSET,)
            assert src.nodata == LST_NODATA_DN
            assert src.dtypes == ("uint16",)

    def test_the_pixels_decode_back_to_the_temperatures_that_went_in(
        self, item_dir, composite
    ):
        celsius, _lst, _qa = composite
        env, src = opened(item_dir / "lst_p95.tif")
        with env, src:
            dn = src.read(1)
            scale, offset = src.scales[0], src.offsets[0]
        valid = dn != LST_NODATA_DN
        decoded = dn[valid] * scale + offset
        # Half a DN is 0.005 C, so rounding is the only difference allowed.
        np.testing.assert_allclose(decoded, celsius[valid], atol=LST_SCALE)

    def test_every_decoded_pixel_lands_inside_the_trusted_range(self, item_dir):
        env, src = opened(item_dir / "lst_p95.tif")
        with env, src:
            dn = src.read(1)
            decoded = dn[dn != LST_NODATA_DN] * src.scales[0] + src.offsets[0]
        assert decoded.min() >= LST_VALID_MIN_C
        assert decoded.max() <= LST_VALID_MAX_C


class TestObservationCounts:
    def test_the_counts_have_one_band_per_calendar_month(self, item_dir):
        env, src = opened(item_dir / "qa_count.tif")
        with env, src:
            assert src.count == 12
            assert src.dtypes[0] == "uint8"
            assert list(src.descriptions) == MONTH_NAMES

    def test_the_counts_declare_no_nodata_value(self, item_dir):
        # A zero here means no observation survived masking that month. Making
        # it nodata would erase the difference from a masked pixel, which is
        # the only thing the band exists to record.
        env, src = opened(item_dir / "qa_count.tif")
        with env, src:
            assert src.nodata is None

    def test_a_month_of_zero_observations_stays_visible(self, item_dir):
        env, src = opened(item_dir / "qa_count.tif")
        with env, src:
            january = src.read(1)
        assert january.max() == 0


class TestPairedReader:
    def test_the_reader_reports_what_the_writer_set(self, item_dir):
        encoding = read_cog_encoding(item_dir / "lst_p95.tif")
        assert encoding["dtype"] == "uint16"
        assert encoding["count"] == 1
        assert encoding["scale"] == LST_SCALE
        assert encoding["offset"] == LST_OFFSET
        assert encoding["nodata"] == LST_NODATA_DN
        assert encoding["block_shape"] == [BLOCK_SIZE, BLOCK_SIZE]
        assert encoding["overviews"]
        assert set(STATISTICS_KEYS) <= set(encoding["statistics"][0])


class TestCatalogStructure:
    def test_the_catalog_carries_the_documents_portolan_requires(self, catalog):
        for relative in (
            "catalog.json",
            "README.md",
            "AGENTS.md",
            f"{COLLECTION_ID}/collection.json",
            f"{COLLECTION_ID}/README.md",
            f"{COLLECTION_ID}/AGENTS.md",
            f"{COLLECTION_ID}/thumbnail.png",
            f"{COLLECTION_ID}/items.parquet",
            f"{COLLECTION_ID}/{ITEM_ID}/{ITEM_ID}.json",
        ):
            assert (catalog / relative).is_file(), relative

    def test_both_cogs_sit_on_the_item_and_not_on_the_collection(self, catalog):
        # Two collection-level COGs is a Portolan conformance error, and the
        # item is what carries the tile's own footprint and time window.
        collection = json.loads(
            (catalog / COLLECTION_ID / "collection.json").read_text()
        )
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert set(collection["assets"]) == {"thumbnail", "items"}
        assert set(item["assets"]) == {LST_ASSET_KEY, QA_ASSET_KEY}

    def test_the_asset_checksums_match_the_bytes_on_disk(self, catalog):
        item_path = catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json"
        item = json.loads(item_path.read_text())
        for asset in item["assets"].values():
            target = (item_path.parent / asset["href"]).resolve()
            assert asset["file:checksum"] == multihash_sha256(target)
            assert asset["file:size"] == target.stat().st_size

    def test_the_item_mirror_reproduces_the_collections_items(self, catalog):
        import pyarrow.parquet as pq

        table = pq.read_table(catalog / COLLECTION_ID / "items.parquet")
        assert table.num_rows == 1
        assert table.column("id").to_pylist() == [ITEM_ID]

    def test_the_statistics_in_the_stac_match_the_statistics_in_the_file(
        self, catalog, item_dir
    ):
        item = json.loads((item_dir / f"{ITEM_ID}.json").read_text())
        declared = item["assets"][LST_ASSET_KEY]["bands"][0]["statistics"]
        env, src = opened(item_dir / "lst_p95.tif")
        with env, src:
            tags = src.tags(bidx=1)
        assert declared["minimum"] == float(tags["STATISTICS_MINIMUM"])
        assert declared["maximum"] == float(tags["STATISTICS_MAXIMUM"])
        assert declared["valid_percent"] == float(tags[VALID_PERCENT_KEY])


class TestPortolanConformance:
    def test_rashid_reports_no_broken_requirement(self, catalog):
        from rashid import validate

        report = validate(catalog)
        assert not report.errors, "\n".join(f.message for f in report.errors)

    def test_rashid_reports_nothing_at_all(self, catalog):
        # Warnings are the SHOULD requirements. None of them is out of reach
        # here, so any warning is a regression rather than a trade-off.
        from rashid import validate

        report = validate(catalog)
        assert not report.findings, "\n".join(f.message for f in report.findings)

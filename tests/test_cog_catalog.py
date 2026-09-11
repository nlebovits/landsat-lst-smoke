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
from typing import Any

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cog_catalog import (  # noqa: E402
    BLOCK_SIZE,
    LST_ASSET_KEY,
    MONTH_NAMES,
    PROCESSING_EXTENSION,
    QA_ASSET_KEY,
    RASTER_EXTENSION,
    RENDER_EXTENSION,
    SCIENTIFIC_EXTENSION,
    STATISTICS_KEYS,
    THUMBNAIL_FILENAME,
    VALID_PERCENT_KEY,
    _renders,
    _rfc3339,
    _verify_cog,
    band_statistics,
    catalog_provenance,
    check_raster_shape,
    mask_lineage,
    multihash_sha256,
    read_cog_encoding,
    read_items,
    tile_id,
    transform_for,
    write_catalog,
    write_cog,
)
from lst_qa import (  # noqa: E402
    LST_MAX_DN,
    LST_MIN_DN,
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

# The tile immediately north, for the tests that need a collection to hold
# more than one item. Its northern edge is a half degree, which is the case
# that a rounded tile name used to collapse onto its neighbour.
NEIGHBOUR_BBOX = (-65.0, -32.0, -64.5, -31.5)
NEIGHBOUR_ID = "S31.5W065"
UNION_BBOX = [-65.0, -32.5, -64.5, -31.5]


def meta_for(bbox):
    """The `part-meta.json` payload for one tile of the test grid."""
    return {
        "raster": [SIZE, SIZE],
        "bbox": list(bbox),
        "crs": "EPSG:4326",
        "pixels_per_degree": PIXELS_PER_DEGREE,
        "start": "2021-01-01",
        "end": "2025-12-31T23:59:59Z",
    }


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
    root = tmp_path_factory.mktemp("catalog")
    return write_catalog(
        root / "catalog", lst, qa, meta_for(BBOX), collection_id=COLLECTION_ID
    )


@pytest.fixture(scope="module")
def item_dir(catalog):
    return catalog / COLLECTION_ID / ITEM_ID


@pytest.fixture(scope="module")
def pair(tmp_path_factory, composite):
    """Two tiles written into one catalog, the second after the first.

    Its own tree, because the single-tile tests read a collection that says it
    holds one item. This is the shape the 520-tile plan needs: the writer is
    called once per tile and the collection describes every tile so far.
    """
    _celsius, lst, qa = composite
    root = tmp_path_factory.mktemp("pair") / "catalog"
    write_catalog(root, lst, qa, meta_for(BBOX), collection_id=COLLECTION_ID)
    return write_catalog(
        root, lst, qa, meta_for(NEIGHBOUR_BBOX), collection_id=COLLECTION_ID
    )


@pytest.fixture(scope="module")
def pair_collection(pair):
    return json.loads((pair / COLLECTION_ID / "collection.json").read_text())


def opened(path):
    """Open a raster the way a Portolan validator does: no PAM sidecar."""
    return rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path)


class TestTileGeometry:
    def test_the_tile_is_named_for_its_north_and_west_edges(self):
        assert tile_id(BBOX) == ITEM_ID
        assert tile_id((-65.0, -35.0, -60.0, -30.0)) == "S30W065"
        assert tile_id((10.0, 45.0, 15.0, 50.0)) == "N50E010"

    def test_neighbouring_half_degree_tiles_get_different_names(self):
        # Rounding the edges gave all three of these the name S32W065, so the
        # second merge overwrote the first tile's COGs and the third
        # overwrote the second.
        names = {
            tile_id((-65.0, -32.5, -64.5, -32.0)),
            tile_id((-65.0, -32.0, -64.5, -31.5)),
            tile_id((-65.0, -33.0, -64.5, -32.5)),
        }
        assert len(names) == 3

    def test_a_fractional_edge_stays_in_the_name(self):
        assert tile_id((-62.5, -35.0, -60.0, -32.5)) == "S32.5W062.5"

    def test_a_whole_degree_edge_carries_no_fraction(self):
        assert "." not in tile_id((-65.0, -35.0, -60.0, -30.0))

    def test_the_equator_and_the_prime_meridian_read_as_north_and_east(self):
        assert tile_id((0.0, -1.0, 1.0, 0.0)) == "N00E000"

    def test_a_raster_that_does_not_cover_its_bbox_is_refused(self):
        # `transform_for` reads only the west and north edges, so a raster of
        # the wrong shape would land on the grid and quietly cover an extent
        # that no item bbox mentions.
        check_raster_shape((SIZE, SIZE), BBOX, PIXELS_PER_DEGREE)
        with pytest.raises(ValueError, match="does not cover"):
            check_raster_shape((SIZE, SIZE - 1), BBOX, PIXELS_PER_DEGREE)

    def test_write_cog_refuses_an_array_of_the_wrong_shape(self, tmp_path):
        short = np.zeros((SIZE - 1, SIZE), dtype="uint16")
        with pytest.raises(ValueError, match="does not cover"):
            write_cog(
                tmp_path / "short.tif",
                short,
                bbox=BBOX,
                pixels_per_degree=PIXELS_PER_DEGREE,
            )

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


class TestTheBandsCarryTheDecodingRule:
    """Scale and offset belong to the raster extension, not a custom prefix.

    Portolan reuses an established extension wherever one applies rather than
    re-encoding the same fact. Every field here has a registered home, so none
    of them needs an `lst:` twin in the item properties.
    """

    def test_the_temperature_band_declares_its_scale_and_offset(self, catalog):
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        band = item["assets"][LST_ASSET_KEY]["bands"][0]
        assert band["raster:scale"] == LST_SCALE
        assert band["raster:offset"] == LST_OFFSET
        assert band["unit"] == "celsius"
        assert band["nodata"] == LST_NODATA_DN

    def test_the_statistics_stay_in_the_stored_digital_numbers(self, catalog):
        # The COG header reports raw DN, and a reader meets them before it
        # applies raster:scale. Decoding them here would invite a client that
        # honours the scale to apply it twice.
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        band = item["assets"][LST_ASSET_KEY]["bands"][0]
        assert band["statistics"]["minimum"] > 1000.0

    def test_the_counts_declare_no_scale_to_decode(self, catalog):
        # An identity scale on a count would invite a reader to decode it.
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        for band in item["assets"][QA_ASSET_KEY]["bands"]:
            assert "raster:scale" not in band
            assert "raster:offset" not in band

    def test_every_band_says_a_pixel_covers_an_area(self, catalog):
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        for key in (LST_ASSET_KEY, QA_ASSET_KEY):
            for band in item["assets"][key]["bands"]:
                assert band["raster:sampling"] == "area"

    def test_the_item_declares_the_extension_its_fields_come_from(self, catalog):
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert RASTER_EXTENSION in item["stac_extensions"]
        assert RENDER_EXTENSION in item["stac_extensions"]

    def test_no_custom_prefix_restates_what_the_bands_already_say(self, catalog):
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert not [key for key in item["properties"] if key.startswith("lst:")]


class TestTheItemDescribesItself:
    def test_the_counts_carry_the_role_a_reader_filters_on(self, catalog):
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert item["assets"][QA_ASSET_KEY]["roles"] == ["data", "quality"]
        assert item["assets"][LST_ASSET_KEY]["roles"] == ["data"]

    def test_the_counts_say_what_a_zero_means(self, catalog):
        # The band has no nodata value, so nothing else in the asset does.
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert (
            "masking removed every observation"
            in (item["assets"][QA_ASSET_KEY]["description"])
        )

    def test_the_title_names_the_tile_and_the_window(self, catalog):
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert item["properties"]["title"] == (
            f"{ITEM_ID} land surface temperature, 2021-2025"
        )

    def test_the_item_draws_itself_without_the_collection(self, catalog):
        # A client that opens one tile never reads the collection, so the item
        # carries the ramp for its own pixels.
        item = json.loads(
            (catalog / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        render = item["properties"]["renders"][LST_ASSET_KEY]
        assert render["assets"] == [LST_ASSET_KEY]
        low, high = render["rescale"][0]
        band = item["assets"][LST_ASSET_KEY]["bands"][0]
        assert (low, high) == (
            band["statistics"]["minimum"],
            band["statistics"]["maximum"],
        )


ASTER_GED: dict[str, Any] = {
    "short_name": "AG1km",
    "version": "003",
    "doi": "10.5067/COMMUNITY/ASTER_GED/AG1KM.003",
    "granule_count": 14128,
    "raster_sha256": "6c3b0f845242bdf5",
}
MASK_RULE: dict[str, Any] = {
    "gap_buffer_cells": 1,
    "gap_hot_threshold_c": 70.0,
    "land_geometry_sha256": "35170d2371beacac",
    "aster_ged": ASTER_GED,
}


class TestTheMaskExplainsItself:
    """A nodata pixel means one of three things, and the raster says which.

    Water, a failed emissivity retrieval inside an ASTER GED gap, and no usable
    observation all read as DN 0. The item states the rules instead, and names
    the artifacts they read by checksum rather than by a path on the machine
    that ran the mask.
    """

    def test_the_lineage_names_both_output_rules(self):
        lineage = mask_lineage(MASK_RULE)["processing:lineage"]
        assert "Water:" in lineage
        assert "Emissivity:" in lineage
        assert "70.0 C or hotter" in lineage
        assert "lies 1 cell from such a cell" in lineage

    def test_the_lineage_names_the_artifacts_by_checksum(self):
        lineage = mask_lineage(MASK_RULE)["processing:lineage"]
        assert "6c3b0f845242bdf5" in lineage
        assert "35170d2371beacac" in lineage
        assert "AG1km v003" in lineage

    def test_the_aster_doi_is_cited(self):
        publications = mask_lineage(MASK_RULE)["sci:publications"]
        assert publications[0]["doi"] == "10.5067/COMMUNITY/ASTER_GED/AG1KM.003"

    def test_an_empty_digest_is_left_out_rather_than_published(self):
        # A raster cannot hold its own digest, so it is empty whenever the
        # sidecar carrying it is absent. An empty one reads as a checksum a
        # consumer can compare against, which is worse than none.
        rule = MASK_RULE | {"aster_ged": ASTER_GED | {"raster_sha256": ""}}
        lineage = mask_lineage(rule)["processing:lineage"]
        assert "raster sha256" not in lineage
        assert "14,128 granules." in lineage

    def test_an_unmasked_tile_says_the_gaps_are_still_there(self):
        lineage = mask_lineage(None)
        assert "No output mask ran" in lineage["processing:lineage"]
        # Nothing read ASTER GED, so nothing cites it.
        assert "sci:publications" not in lineage

    def test_no_path_from_the_masking_machine_reaches_the_item(self, tmp_path):
        # An absolute path tells a reader of a published catalog nothing, and
        # it carries the operator's home directory into a public file.
        _lst = encode_celsius(np.full((SIZE, SIZE), 30.0, dtype="float32"))
        qa = np.zeros((12, SIZE, SIZE), dtype="uint8")
        meta = meta_for(BBOX) | {"mask_rule": MASK_RULE}
        root = write_catalog(
            tmp_path / "catalog", _lst, qa, meta, collection_id=COLLECTION_ID
        )
        for path in root.rglob("*.json"):
            assert "/home/" not in path.read_text()
        for path in root.rglob("*.md"):
            assert "/home/" not in path.read_text()

    def test_the_item_declares_the_extensions_the_lineage_needs(self, tmp_path):
        _lst = encode_celsius(np.full((SIZE, SIZE), 30.0, dtype="float32"))
        qa = np.zeros((12, SIZE, SIZE), dtype="uint8")
        meta = meta_for(BBOX) | {"mask_rule": MASK_RULE}
        root = write_catalog(
            tmp_path / "catalog", _lst, qa, meta, collection_id=COLLECTION_ID
        )
        item = json.loads(
            (root / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert PROCESSING_EXTENSION in item["stac_extensions"]
        assert SCIENTIFIC_EXTENSION in item["stac_extensions"]


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


@pytest.fixture(scope="module")
def report(catalog):
    from rashid import validate

    return validate(catalog)


@pytest.fixture(scope="module")
def pair_report(pair):
    from rashid import validate

    return validate(pair)


class TestPortolanConformance:
    def test_rashid_reports_no_broken_requirement(self, report):
        assert not report.errors, "\n".join(f.message for f in report.errors)

    def test_rashid_reports_nothing_at_all(self, report):
        # Warnings are the SHOULD requirements. None of them is out of reach
        # here, so any warning is a regression rather than a trade-off.
        assert not report.findings, "\n".join(f.message for f in report.findings)

    def test_a_two_tile_catalog_is_conformant_too(self, pair_report):
        # Several requirements only start applying at the second item: every
        # item needs its own link, and the mirror needs a row per item.
        assert not pair_report.findings, "\n".join(
            f.message for f in pair_report.findings
        )


class TestASecondTile:
    """The 520-tile claim, at the smallest size that can test it."""

    def test_the_first_tile_survives_the_second(self, pair):
        for item_id in (ITEM_ID, NEIGHBOUR_ID):
            assert (pair / COLLECTION_ID / item_id / "lst_p95.tif").is_file()
            assert (pair / COLLECTION_ID / item_id / f"{item_id}.json").is_file()

    def test_the_collection_links_every_item(self, pair_collection):
        hrefs = [
            link["href"] for link in pair_collection["links"] if link["rel"] == "item"
        ]
        assert hrefs == [
            f"./{NEIGHBOUR_ID}/{NEIGHBOUR_ID}.json",
            f"./{ITEM_ID}/{ITEM_ID}.json",
        ]

    def test_the_spatial_extent_leads_with_the_union(self, pair_collection):
        # A validator reads entry 0 as the overall extent and compares it
        # against the footprints in items.parquet.
        boxes = pair_collection["extent"]["spatial"]["bbox"]
        assert boxes[0] == UNION_BBOX
        assert sorted(boxes[1:]) == sorted([list(BBOX), list(NEIGHBOUR_BBOX)])

    def test_the_temporal_extent_leads_with_the_union(self, pair_collection):
        interval = pair_collection["extent"]["temporal"]["interval"]
        assert interval[0] == ["2021-01-01T00:00:00Z", "2025-12-31T23:59:59Z"]
        assert len(interval) == 3

    def test_the_mirror_carries_one_row_per_item(self, pair):
        import pyarrow.parquet as pq

        table = pq.read_table(pair / COLLECTION_ID / "items.parquet")
        assert sorted(table.column("id").to_pylist()) == sorted([ITEM_ID, NEIGHBOUR_ID])

    def test_the_thumbnail_covers_both_tiles(self, pair, catalog):
        import matplotlib.pyplot as plt

        one = plt.imread(catalog / COLLECTION_ID / THUMBNAIL_FILENAME)
        both = plt.imread(pair / COLLECTION_ID / THUMBNAIL_FILENAME)
        # One tile is half a degree square. Two stacked are twice as tall as
        # they are wide, so a thumbnail of the last tile written would not be.
        assert one.shape[0] == one.shape[1]
        assert both.shape[0] == pytest.approx(2 * both.shape[1], abs=1)

    def test_the_readme_lists_both_tiles(self, pair):
        readme = (pair / COLLECTION_ID / "README.md").read_text()
        assert f"./{ITEM_ID}/{ITEM_ID}.json" in readme
        assert f"./{NEIGHBOUR_ID}/{NEIGHBOUR_ID}.json" in readme

    def test_rewriting_the_same_tile_is_allowed(self, tmp_path, composite):
        _celsius, lst, qa = composite
        root = tmp_path / "catalog"
        write_catalog(root, lst, qa, meta_for(BBOX), collection_id=COLLECTION_ID)
        write_catalog(root, lst, qa, meta_for(BBOX), collection_id=COLLECTION_ID)
        assert list(read_items(root / COLLECTION_ID)) == [ITEM_ID]

    def test_a_different_tile_may_not_take_an_existing_name(self, tmp_path, composite):
        # Two grids sharing a corner is the one way a name can still collide:
        # a 1-degree tile and the 5-degree tile that starts where it does.
        _celsius, lst, qa = composite
        root = tmp_path / "catalog"
        write_catalog(root, lst, qa, meta_for(BBOX), collection_id=COLLECTION_ID)
        wider = meta_for(BBOX) | {"bbox": [-65.0, -33.0, -64.5, -32.0]}
        with pytest.raises(ValueError, match="shares its corner"):
            write_catalog(root, lst, qa, wider, collection_id=COLLECTION_ID)


def tiled_geotiff(path, *, scale=None, tags=None):
    """A tiled GeoTIFF, with a scale and statistics tags only if asked.

    Stands in for a COG that lost something on the way through the driver. It
    fits inside one internal tile, so the overview requirement does not apply
    and whatever the caller withheld is the only thing left to object to.
    """
    band = np.full((BLOCK_SIZE, BLOCK_SIZE), 5000, dtype="uint16")
    profile = {
        "driver": "GTiff",
        "height": BLOCK_SIZE,
        "width": BLOCK_SIZE,
        "count": 1,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": transform_for((-65.0, -34.0, -64.0, -33.0), BLOCK_SIZE),
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "nodata": LST_NODATA_DN,
    }
    with (
        rasterio.Env(GDAL_PAM_ENABLED="NO"),
        rasterio.open(path, "w", **profile) as dst,
    ):
        dst.write(band, 1)
        if scale is not None:
            dst.scales = (scale,)
            dst.offsets = (LST_OFFSET,)
        dst.update_tags(
            1, **(band_statistics(band, LST_NODATA_DN) if tags is None else tags)
        )
    return path


class TestTheWriterChecksItsOwnOutput:
    """`_verify_cog` is the guard against a silent regression in GDAL.

    A dropped scale leaves DN 9500 reading as a temperature, which is the
    failure this writer picked GDAL's COG driver to avoid. The guard has to
    check the thing it was built to protect.
    """

    def verify(self, path):
        _verify_cog(path, 1, scale=LST_SCALE, offset=LST_OFFSET, nodata=LST_NODATA_DN)

    def test_a_file_that_kept_everything_passes(self, tmp_path):
        self.verify(tiled_geotiff(tmp_path / "kept.tif", scale=LST_SCALE))

    def test_a_dropped_scale_is_caught(self, tmp_path):
        with pytest.raises(RuntimeError, match="scale"):
            self.verify(tiled_geotiff(tmp_path / "no_scale.tif"))

    def test_a_wrong_scale_is_caught(self, tmp_path):
        with pytest.raises(RuntimeError, match="scale"):
            self.verify(tiled_geotiff(tmp_path / "bad_scale.tif", scale=1.0))

    def test_a_missing_valid_percent_is_caught(self, tmp_path):
        # Portolan makes the valid percent a MUST once a band has nodata, so
        # the four headline statistics are not the whole requirement.
        path = tiled_geotiff(
            tmp_path / "no_percent.tif",
            scale=LST_SCALE,
            tags=dict.fromkeys(STATISTICS_KEYS, "1.0"),
        )
        with pytest.raises(RuntimeError, match=VALID_PERCENT_KEY):
            self.verify(path)

    def test_a_missing_statistic_is_caught(self, tmp_path):
        path = tiled_geotiff(
            tmp_path / "no_stats.tif",
            scale=LST_SCALE,
            tags={VALID_PERCENT_KEY: "100.0"},
        )
        with pytest.raises(RuntimeError, match="STATISTICS_MINIMUM"):
            self.verify(path)


class TestAssetChecksums:
    def test_the_checksum_matches_a_single_pass_hash(self, tmp_path):
        # The file is read in chunks, because a full-tile COG runs to
        # hundreds of MB and hashing it must not hold it in memory.
        import hashlib

        payload = np.random.default_rng(7).bytes(3 * (1 << 20) + 17)
        path = tmp_path / "big.bin"
        path.write_bytes(payload)
        assert multihash_sha256(path) == f"1220{hashlib.sha256(payload).hexdigest()}"

    def test_the_multihash_prefix_names_sha256_at_32_bytes(self, tmp_path):
        path = tmp_path / "small.bin"
        path.write_bytes(b"")
        digest = multihash_sha256(path)
        assert digest.startswith("1220")
        assert len(digest) == 4 + 64


def lst_item(minimum, maximum, valid_percent):
    """An item stripped to the one field the colour ramp reads."""
    return {
        "assets": {
            LST_ASSET_KEY: {
                "bands": [
                    {
                        "statistics": {
                            "minimum": minimum,
                            "maximum": maximum,
                            "valid_percent": valid_percent,
                        }
                    }
                ]
            }
        }
    }


class TestTheColourRamp:
    def test_the_ramp_spans_every_tile(self):
        rescale = _renders(
            [lst_item(4000.0, 5000.0, 99.0), lst_item(6000.0, 7000.0, 99.0)]
        )
        assert rescale[LST_ASSET_KEY]["rescale"] == [[4000.0, 7000.0]]

    def test_a_tile_with_no_valid_pixel_does_not_pull_the_ramp_to_nodata(self):
        # An empty band reports zeros because the tags are mandatory and it
        # has nothing to report. Reading them would put the low end of the
        # ramp on the nodata DN.
        rescale = _renders([lst_item(0.0, 0.0, 0.0), lst_item(4000.0, 5000.0, 99.0)])
        assert rescale[LST_ASSET_KEY]["rescale"] == [[4000.0, 5000.0]]

    def test_a_collection_of_empty_tiles_still_gets_a_drawable_ramp(self):
        # A tiler dividing by a zero-width range fails. An ocean-only slice of
        # the plan is a real case, so the ramp falls back to the encodable
        # range rather than collapsing.
        rescale = _renders([lst_item(0.0, 0.0, 0.0)])
        assert rescale[LST_ASSET_KEY]["rescale"] == [
            [float(LST_MIN_DN), float(LST_MAX_DN)]
        ]


class TestTimestamps:
    def test_a_bare_date_becomes_midnight_utc(self):
        assert _rfc3339("2021-01-01") == "2021-01-01T00:00:00Z"

    def test_an_instant_already_in_utc_is_left_alone(self):
        assert _rfc3339("2025-12-31T23:59:59Z") == "2025-12-31T23:59:59Z"

    def test_a_negative_utc_offset_is_left_alone(self):
        # The sign lives after the T. Searching the whole string finds the one
        # in the date and appends a second zone to an instant that has one.
        assert _rfc3339("2021-01-01T00:00:00-05:00") == "2021-01-01T00:00:00-05:00"

    def test_a_positive_utc_offset_is_left_alone(self):
        assert _rfc3339("2021-01-01T00:00:00+05:00") == "2021-01-01T00:00:00+05:00"


class TestTheWindowIsNeverAssumed:
    """A window the parts did not state is an error, not a default.

    `part-meta.json` gained `start` and `end` with this writer. Stamping the
    current window over parts from an earlier run would publish a claim about
    the pixels that no reader of the catalog could question.
    """

    @pytest.mark.parametrize("key", ["start", "end", "bbox", "crs"])
    def test_a_missing_key_is_refused(self, key):
        meta = meta_for(BBOX)
        del meta[key]
        with pytest.raises(ValueError, match=key):
            catalog_provenance(meta, collection_id=COLLECTION_ID)

    def test_a_complete_payload_is_reported_back(self):
        provenance = catalog_provenance(meta_for(BBOX), collection_id=COLLECTION_ID)
        assert provenance["start"] == "2021-01-01"
        assert provenance["end"] == "2025-12-31T23:59:59Z"
        assert provenance["lst_scale"] == LST_SCALE

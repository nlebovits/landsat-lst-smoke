"""A run writes a catalog. The path from `main` to a published tile.

There is no merge any more. One process builds the whole tile as one lazy
graph, streams the blocks into two staging GeoTIFFs, and hands those files to
`cog_catalog.write_catalog`. So the wiring these tests check moved: that the
run calls the writer, that the flags reach it, that `--no-catalog` writes the
two COGs into `--out-dir` instead, and that the rules the run was built under
reach the item.

`tests/test_cog_catalog.py` tests the writer itself. Nothing here reaches the
network: every run is a `--rehearse`, which writes its own synthetic scenes
under `<out-dir>/rehearsal-scenes` and reads no object.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import composite  # noqa: E402
import shard_lst_p95  # noqa: E402
from cog_catalog import collection_id_for_window, write_catalog  # noqa: E402

BBOX = "-65.0,-32.5,-64.5,-32.0"
NEIGHBOUR_BBOX = "-65.0,-32.0,-64.5,-31.5"
#: What the 2021-2025 window the driver defaults to must derive to.
COLLECTION_ID = "lst-p95-2021-2025"
PIXELS_PER_DEGREE = 120
SIZE = 60
ITEM_ID = "S32W065"
NEIGHBOUR_ID = "S31.5W065"

#: A tile-less rehearsal at 1/120 degree over half a degree: a 60 x 60 raster
#: in four blocks, over six synthetic scenes. About five seconds end to end.
BASE_ARGV = (
    "--rehearse",
    "6",
    "--pixels-per-degree",
    str(PIXELS_PER_DEGREE),
    "--chunk",
    "30",
    "--workers",
    "2",
    "--threads-per-worker",
    "1",
    "--no-output-mask",
)

pytestmark = pytest.mark.timeout(300)


def run(out_dir: Path, *extra, bbox: str = BBOX) -> int:
    """One rehearsal through the production entry point, in this process."""
    return shard_lst_p95.main(
        [f"--bbox={bbox}", *BASE_ARGV, "--out-dir", str(out_dir), *extra]
    )


def meta_for(*, correction_rule=None, mask_rule=None, bbox: str = BBOX, **overrides):
    """The `run_meta` payload the driver hands the catalog writer.

    Built through `shard_lst_p95.run_meta` rather than by hand, so a field the
    driver stops passing is a failure here rather than a silently thinner item.
    """
    args = shard_lst_p95.parse_args([f"--bbox={bbox}", *BASE_ARGV])
    area, _ = shard_lst_p95.resolve_area(args)
    height, width = composite.raster_shape(area, args.pixels_per_degree)
    return shard_lst_p95.run_meta(
        args, area, height, width, mask_rule, correction_rule
    ) | dict(overrides)


class TestTheRunWritesACatalog:
    @pytest.fixture(scope="class")
    def published(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("tile")
        assert run(out) == 0
        return out

    def test_both_cogs_land_under_the_item(self, published):
        item = published / "catalog" / COLLECTION_ID / ITEM_ID
        assert (item / "lst_p95.tif").is_file()
        assert (item / "qa_count.tif").is_file()

    def test_the_summary_names_the_catalog(self, published):
        summary = json.loads((published / "summary.json").read_text())
        assert Path(summary["catalog"]) == (published / "catalog")
        assert summary["synthetic"] is True

    def test_a_pooled_run_records_the_correction_as_none(self, published):
        # Absent and null have to read the same. A rehearsal takes no prep
        # file, so its pixels are the pooled percentile, and that is itself a
        # rule the summary states rather than omits.
        summary = json.loads((published / "summary.json").read_text())
        assert summary["correction"] is None

    def test_the_pooled_item_says_so(self, published):
        item = json.loads(
            (
                published / "catalog" / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json"
            ).read_text()
        )
        assert "pooled" in item["properties"]["processing:lineage"]

    def test_the_item_carries_the_window_the_run_composited(self, published):
        item = json.loads(
            (
                published / "catalog" / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json"
            ).read_text()
        )
        assert item["properties"]["start_datetime"] == "2021-01-01T00:00:00Z"
        assert item["properties"]["end_datetime"] == "2025-12-31T23:59:59Z"

    def test_the_item_says_which_correction_produced_its_pixels(
        self, published, tmp_path
    ):
        """The rule the driver assembled has to reach the published item.

        The run's own COGs are republished under a `run_meta` carrying a
        correction, because a rehearsal takes no prep file and so can only ever
        produce the pooled rule. What is under test is the hand-off, not the
        numerics: `tests/test_prep_wiring.py` builds the rule.
        """
        rule = {
            "destripe": True,
            "feather": False,
            "max_offset_c": 15.0,
            "prep_scene_digest": "abc123",
            "prep_window": {"start": "2021-01-01", "end": "2025-12-31"},
        }
        source = published / "catalog" / COLLECTION_ID / ITEM_ID
        root = write_catalog(
            tmp_path / "corrected",
            source / "lst_p95.tif",
            source / "qa_count.tif",
            meta_for(correction_rule=rule),
        )
        item = json.loads(
            (root / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        lineage = item["properties"]["processing:lineage"]
        assert "Scene offsets:" in lineage
        assert "abc123" in lineage


class TestNoCatalog:
    @pytest.fixture(scope="class")
    def bare(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("bare")
        assert run(out, "--no-catalog") == 0
        return out

    def test_the_flag_leaves_the_two_cogs_in_the_out_dir(self, bare):
        assert (bare / "lst_p95.tif").is_file()
        assert (bare / "qa_count.tif").is_file()

    def test_the_flag_writes_no_catalog(self, bare):
        assert not (bare / "catalog").exists()

    def test_the_summary_names_no_catalog(self, bare):
        summary = json.loads((bare / "summary.json").read_text())
        assert summary["catalog"] is None

    def test_the_cogs_carry_the_raster_the_run_planned(self, bare):
        import rasterio

        summary = json.loads((bare / "summary.json").read_text())
        with rasterio.open(bare / "lst_p95.tif") as src:
            assert [src.height, src.width] == summary["raster"] == [SIZE, SIZE]


class TestTheCatalogCheckRunsBeforeTheWriter:
    """A tile the writer cannot describe is refused before it is written.

    `main` calls `check_catalog_inputs` on the run's own meta before
    `write_catalog`, and every rule it enforces is answerable from that meta
    alone. Checking it there turns a traceback halfway through publishing into
    a message, and it is why `--no-catalog` is the escape the message offers.
    """

    def check(self, **overrides):
        from cog_catalog import check_catalog_inputs

        with pytest.raises(ValueError) as failure:
            check_catalog_inputs(meta_for(**overrides))
        return str(failure.value)

    def test_a_projected_crs_is_refused(self):
        assert "EPSG:4326" in self.check(crs="EPSG:3857")

    @pytest.mark.parametrize("key", ["start", "end"])
    def test_a_meta_that_states_no_window_is_refused(self, key):
        # The window must come from the run, never from a default: stamping
        # one over the pixels publishes a claim nobody could question.
        meta = meta_for()
        del meta[key]
        from cog_catalog import check_catalog_inputs

        with pytest.raises(ValueError, match=key):
            check_catalog_inputs(meta)

    def test_a_raster_that_does_not_cover_its_bbox_is_refused(self):
        assert "does not cover" in self.check(pixels_per_degree=3600)

    def test_the_escape_hatch_the_message_offers_works(self, tmp_path):
        # A run that cannot be catalogued still has to leave its pixels behind.
        assert run(tmp_path / "escape", "--no-catalog") == 0
        assert (tmp_path / "escape" / "lst_p95.tif").is_file()


class TestTheCollectionIdCarriesTheWindow:
    def test_a_multi_year_window_spans_its_first_and_last_year(self):
        assert (
            collection_id_for_window({"start": "2021-01-01", "end": "2025-12-31"})
            == "lst-p95-2021-2025"
        )

    def test_a_single_year_window_names_that_year_once(self):
        assert (
            collection_id_for_window({"start": "2024-01-01", "end": "2024-12-31"})
            == "lst-p95-2024"
        )

    def test_the_run_derives_it_with_no_flag(self, tmp_path):
        # The window the run composited over decides the directory. Two windows
        # therefore cannot collect into one collection by omission.
        out = tmp_path / "derived"
        assert run(out) == 0
        assert (out / "catalog" / COLLECTION_ID / "collection.json").is_file()

    @pytest.mark.parametrize("missing", ["start", "end"])
    def test_a_meta_with_no_window_earns_no_id(self, missing):
        window = {"start": "2021-01-01", "end": "2025-12-31"}
        del window[missing]
        with pytest.raises(ValueError, match=missing):
            collection_id_for_window(window)


class TestCatalogFlags:
    def test_the_identity_flags_reach_the_collection(self, tmp_path):
        out = tmp_path / "flagged"
        assert (
            run(
                out,
                "--collection-id",
                "lst-p95-test",
                "--host-name",
                "Example Lab",
                "--host-url",
                "https://example.org/lab",
                "--license",
                "CC-BY-4.0",
            )
            == 0
        )
        collection = json.loads(
            (out / "catalog" / "lst-p95-test" / "collection.json").read_text()
        )
        assert collection["id"] == "lst-p95-test"
        assert collection["license"] == "CC-BY-4.0"
        assert collection["providers"][-1] == {
            "name": "Example Lab",
            "url": "https://example.org/lab",
            "roles": ["processor", "host"],
        }


class TestOneCatalogForEveryTile:
    """Two tiles, two runs, one catalog. The 520-tile plan in miniature."""

    @pytest.fixture(scope="class")
    def two_tiles(self, tmp_path_factory):
        work = tmp_path_factory.mktemp("fleet")
        catalog = work / "catalog"
        for name, bbox in (("a", BBOX), ("b", NEIGHBOUR_BBOX)):
            assert (
                run(work / f"tile-{name}", "--catalog-dir", str(catalog), bbox=bbox)
                == 0
            )
        return catalog

    def test_both_tiles_are_items_of_one_collection(self, two_tiles):
        collection_dir = two_tiles / COLLECTION_ID
        assert (collection_dir / ITEM_ID / "lst_p95.tif").is_file()
        assert (collection_dir / NEIGHBOUR_ID / "lst_p95.tif").is_file()
        collection = json.loads((collection_dir / "collection.json").read_text())
        items = [link for link in collection["links"] if link["rel"] == "item"]
        assert len(items) == 2

    def test_the_collection_extent_spans_both(self, two_tiles):
        collection = json.loads(
            (two_tiles / COLLECTION_ID / "collection.json").read_text()
        )
        assert collection["extent"]["spatial"]["bbox"][0] == [
            -65.0,
            -32.5,
            -64.5,
            -31.5,
        ]

    def test_the_validator_accepts_the_result(self, two_tiles):
        from rashid import validate

        report = validate(two_tiles)
        assert not report.findings, "\n".join(f.message for f in report.findings)

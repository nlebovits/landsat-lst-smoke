"""`--merge`, the path that turns part files into a published catalog.

`tests/test_cog_catalog.py` tests the writer. These tests check the wiring
around it: that the merge calls the writer, that the flags reach it, that
`--no-catalog` stops it, and that a tile the writer cannot describe is refused
before the merge spends an hour assembling arrays it will then abandon.

Nothing here reaches the network. The part files are written in `tmp_path` by
`write_parts`, in the shape `--shard-slice` leaves behind.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shard_lst_p95  # noqa: E402
from cog_catalog import collection_id_for_window  # noqa: E402
from lst_qa import encode_celsius  # noqa: E402

BBOX = (-65.0, -32.5, -64.5, -32.0)
NEIGHBOUR_BBOX = (-65.0, -32.0, -64.5, -31.5)
#: What the 2021-2025 window the fixture records must derive to.
COLLECTION_ID = "lst-p95-2021-2025"
PIXELS_PER_DEGREE = 1200
SIZE = 600
ITEM_ID = "S32W065"
NEIGHBOUR_ID = "S31.5W065"


def write_parts(part_dir: Path, bbox=BBOX, **overrides) -> Path:
    """One part file and its meta, covering a whole tile in a single shard."""
    rng = np.random.default_rng(20260909)
    celsius = rng.uniform(12.0, 48.0, (SIZE, SIZE)).astype("float32")
    lst = encode_celsius(celsius)
    qa = np.full((12, SIZE, SIZE), 3, dtype="uint8")

    part_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(part_dir / "part-000.npz", lst_0_0=lst, qa_0_0=qa)
    meta = {
        "raster": [SIZE, SIZE],
        "bbox": list(bbox),
        "crs": "EPSG:4326",
        "pixels_per_degree": PIXELS_PER_DEGREE,
        "shard_px": SIZE,
        "n_shards": 1,
        "start": "2021-01-01",
        "end": "2025-12-31T23:59:59Z",
    }
    meta.update(overrides)
    for key, value in list(meta.items()):
        if value is None:
            del meta[key]
    (part_dir / "part-meta.json").write_text(json.dumps(meta, indent=2))
    return part_dir


def merge(part_dir: Path, out_dir: Path, *extra) -> int:
    """Drive the CLI in process, the way `test_time_window.py` does.

    `--bbox` is required on the parser even though `--merge` reads the bbox
    from `part-meta.json` and ignores this one.
    """
    return shard_lst_p95.main(
        [
            "--bbox=-65.0,-32.5,-64.5,-32.0",
            "--merge",
            str(part_dir),
            "--out-dir",
            str(out_dir),
            *extra,
        ]
    )


class TestTheMergeWritesACatalog:
    def test_the_arrays_and_the_catalog_land_together(self, tmp_path):
        out = tmp_path / "tile"
        assert merge(write_parts(tmp_path / "part0"), out) == 0
        assert (out / "lst_p95_dn.npy").is_file()
        assert (out / "qa_count.npy").is_file()
        item = out / "catalog" / COLLECTION_ID / ITEM_ID
        assert (item / "lst_p95.tif").is_file()
        assert (item / "qa_count.tif").is_file()

    def test_the_merge_record_names_the_catalog(self, tmp_path):
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0"), out)
        record = json.loads((out / "merge.json").read_text())
        assert Path(record["catalog"]) == (out / "catalog").resolve()
        assert record["coverage"] == 1.0

    def test_the_merge_record_hoists_both_rules(self, tmp_path):
        """A reader of the merged tile should not have to open a part.

        `merge_parts` already refuses parts that disagree about either rule, so
        the one it merged under is a fact about the whole tile. `mask_rule` was
        hoisted for that reason and `correction_rule` was not.
        """
        rule = {"destripe": True, "feather": True, "prep_scene_digest": "abc"}
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0", correction_rule=rule), out)
        record = json.loads((out / "merge.json").read_text())
        assert record["correction_rule"] == rule

    def test_a_pooled_tile_records_the_rule_as_none(self, tmp_path):
        # Absent and null have to read the same, because a part written before
        # the correction existed carries no key at all.
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0"), out)
        record = json.loads((out / "merge.json").read_text())
        assert record["correction_rule"] is None

    def test_the_item_says_which_correction_produced_its_pixels(self, tmp_path):
        rule = {
            "destripe": True,
            "feather": False,
            "max_offset_c": 15.0,
            "prep_scene_digest": "abc123",
            "prep_window": {"start": "2021-01-01", "end": "2025-12-31"},
        }
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0", correction_rule=rule), out)
        item = json.loads(
            (out / "catalog" / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        lineage = item["properties"]["processing:lineage"]
        assert "Scene offsets:" in lineage
        assert "abc123" in lineage

    def test_the_item_carries_the_window_the_parts_recorded(self, tmp_path):
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0"), out)
        item = json.loads(
            (out / "catalog" / COLLECTION_ID / ITEM_ID / f"{ITEM_ID}.json").read_text()
        )
        assert item["properties"]["start_datetime"] == "2021-01-01T00:00:00Z"
        assert item["properties"]["end_datetime"] == "2025-12-31T23:59:59Z"


class TestNoCatalog:
    def test_the_flag_leaves_only_the_arrays(self, tmp_path):
        out = tmp_path / "tile"
        assert merge(write_parts(tmp_path / "part0"), out, "--no-catalog") == 0
        assert (out / "lst_p95_dn.npy").is_file()
        assert not (out / "catalog").exists()

    def test_the_merge_record_is_still_written(self, tmp_path):
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0"), out, "--no-catalog")
        record = json.loads((out / "merge.json").read_text())
        assert "catalog" not in record
        assert record["shards"] == 1

    def test_a_tile_the_writer_cannot_describe_still_merges(self, tmp_path):
        # The escape hatch the error message offers has to work.
        out = tmp_path / "tile"
        parts = write_parts(tmp_path / "part0", crs="EPSG:3857")
        assert merge(parts, out, "--no-catalog") == 0
        assert (out / "lst_p95_dn.npy").is_file()


class TestTheMergeFailsBeforeItWorks:
    """A tile the writer cannot describe is refused up front.

    Every rule is answerable from `part-meta.json`, so checking it first turns
    an hour of merging followed by a traceback into a message in a second.
    """

    def run_and_expect_exit(self, tmp_path, **overrides):
        out = tmp_path / "tile"
        parts = write_parts(tmp_path / "part0", **overrides)
        with pytest.raises(SystemExit) as failure:
            merge(parts, out)
        assert not out.exists(), "the merge ran before the check"
        return str(failure.value)

    def test_a_projected_crs_is_refused(self, tmp_path):
        message = self.run_and_expect_exit(tmp_path, crs="EPSG:3857")
        assert "EPSG:4326" in message

    @pytest.mark.parametrize("key", ["start", "end"])
    def test_parts_that_state_no_window_are_refused(self, tmp_path, key):
        # Parts written before the catalog writer existed carry neither. The
        # window must come from the run, never from a default.
        assert key in self.run_and_expect_exit(tmp_path, **{key: None})

    def test_a_raster_that_does_not_cover_its_bbox_is_refused(self, tmp_path):
        message = self.run_and_expect_exit(tmp_path, pixels_per_degree=3600)
        assert "does not cover" in message


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

    def test_the_merge_derives_it_with_no_flag(self, tmp_path):
        # The window the parts recorded decides the directory. Two windows
        # therefore cannot collect into one collection by omission.
        out = tmp_path / "tile"
        merge(write_parts(tmp_path / "part0"), out)
        assert (out / "catalog" / COLLECTION_ID / "collection.json").is_file()

    @pytest.mark.parametrize("missing", ["start", "end"])
    def test_a_part_with_no_window_earns_no_id(self, missing):
        window = {"start": "2021-01-01", "end": "2025-12-31"}
        del window[missing]
        with pytest.raises(ValueError, match=missing):
            collection_id_for_window(window)


class TestCatalogFlags:
    def test_the_identity_flags_reach_the_collection(self, tmp_path):
        out = tmp_path / "tile"
        merge(
            write_parts(tmp_path / "part0"),
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
    """Two tiles, two merges, one catalog. The 520-tile plan in miniature."""

    @pytest.fixture
    def two_tiles(self, tmp_path):
        catalog = tmp_path / "catalog"
        for name, bbox in (("a", BBOX), ("b", NEIGHBOUR_BBOX)):
            merge(
                write_parts(tmp_path / f"part-{name}", bbox=bbox),
                tmp_path / f"tile-{name}",
                "--catalog-dir",
                str(catalog),
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

"""The tracked catalog is what the generator produces from the item JSON.

Item JSON is primary. A run writes it, `lst-publish-catalog recount` rewrites
it, and neither is reproducible without the COGs. Everything above the items
is derived: `collection.json`, `catalog.json`, `items.parquet`, and the four
Markdown files all come out of `rebuild_collection`.

Deriving them does not stop anyone editing them. `catalog/` is 777 tracked
files and a hand edit to one of them looks exactly like a generated one in a
diff. It survives review, and then the next `finish` step overwrites it with
no warning at all.

So this module regenerates the derived tree from the committed items and
compares. Edit the generator, not the output.

MEASURED 2026-09-17, on the committed tree: all seven derived files match byte
for byte, `items.parquet` included. The plan for this work expected Parquet to
need a table comparison instead of a byte comparison. It does not, at
pyarrow 21. `test_the_item_mirror_holds_the_same_table` compares the table
anyway, because that is the test that separates the two ways this can fail.
A serializer change breaks the bytes and keeps the table. A metadata change
breaks both.

What this does not cover is `thumbnail.png`. Drawing it reads all 769 COGs,
which live in the bucket and never enter git, so the fixture keeps the
committed PNG and stubs the renderer. `test_the_thumbnail_matches_its_own
_checksum` is what catches a stale one.
"""

from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path

import pytest
from lst import cog_catalog

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog"
COLLECTION_ID = "lst-p95-2021-2025"

# Everything `rebuild_collection` writes, except the thumbnail.
DERIVED = (
    "catalog.json",
    "README.md",
    "AGENTS.md",
    f"{COLLECTION_ID}/collection.json",
    f"{COLLECTION_ID}/README.md",
    f"{COLLECTION_ID}/AGENTS.md",
    f"{COLLECTION_ID}/items.parquet",
)


def keep_the_committed_thumbnail(path: Path, tiles, *, long_edge: int = 480) -> Path:
    """Stand in for `render_thumbnail`, and draw nothing.

    The real one opens all 769 published COGs, which git does not hold. The
    committed PNG is already at `path`, so returning it unchanged lets
    `_collection_assets` read its real size and checksum.

    It carries the full signature rather than `**kwargs` so that a change to
    the real renderer's parameters fails `ty` here instead of silently
    accepting a call this stub would have mishandled.
    """
    return Path(path)


@pytest.fixture(scope="module")
def committed():
    return json.loads((CATALOG / COLLECTION_ID / "collection.json").read_text())


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory, committed):
    """The derived tree, rebuilt from the committed item JSON alone.

    Module-scoped: copying 769 item directories is the expensive part, and
    every test in this file reads the same result.

    `updated` comes from the committed collection rather than the clock.
    `rebuild_collection` stamps it into both the collection and the root
    catalog, so a fresh timestamp would differ on every run and prove nothing.

    The renderer is stubbed to return the path it was given. The committed
    PNG is already at that path, so `_collection_assets` reads the real size
    and the real checksum off it and the comparison stays honest.
    """
    dest = tmp_path_factory.mktemp("regen") / "catalog"
    (dest / COLLECTION_ID).mkdir(parents=True)
    for item_dir in sorted((CATALOG / COLLECTION_ID).iterdir()):
        if item_dir.is_dir():
            shutil.copytree(item_dir, dest / COLLECTION_ID / item_dir.name)
    shutil.copy2(
        CATALOG / COLLECTION_ID / cog_catalog.THUMBNAIL_FILENAME,
        dest / COLLECTION_ID / cog_catalog.THUMBNAIL_FILENAME,
    )

    # `MonkeyPatch.context()` rather than the `monkeypatch` fixture, which is
    # function-scoped and cannot be requested here. Rebinding the attribute by
    # hand works too, and `ty` rejects it: a module-level function's type is
    # nominal, so no other function is assignable to it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cog_catalog, "render_thumbnail", keep_the_committed_thumbnail)
        cog_catalog.rebuild_collection(
            dest,
            COLLECTION_ID,
            license_id=committed["license"],
            updated=committed["updated"],
        )
    return dest


class TestTheDerivedTreeRegenerates:
    @pytest.mark.parametrize("relative", DERIVED)
    def test_the_file_is_what_the_generator_writes(self, relative, regenerated):
        """A hand edit to a derived file fails here.

        `filecmp.cmp(shallow=False)` reads both files. The default compares
        size and mtime, which a fresh copy defeats.
        """
        committed_path = CATALOG / relative
        rebuilt = regenerated / relative
        assert rebuilt.exists(), f"the generator no longer writes {relative}"
        assert filecmp.cmp(committed_path, rebuilt, shallow=False), (
            f"{relative} is not what rebuild_collection produces. Edit the "
            f"generator in src/lst/cog_catalog.py, then rerun "
            f"`lst-publish-catalog finish`."
        )

    def test_the_generator_writes_nothing_else(self, regenerated):
        """A new derived file has to be added to `DERIVED` to be compared.

        Without this, a generator that started writing a fifth Markdown
        document would leave it untested and unpublished.
        """
        # Depth, not a name test. An item is `<collection>/<tile>/<tile>.json`
        # and nothing else in the tree is three deep. Matching the stem
        # against the parent directory would also drop `catalog/catalog.json`.
        written = {
            str(p.relative_to(regenerated))
            for p in regenerated.rglob("*")
            if p.is_file() and len(p.relative_to(regenerated).parts) < 3
        }
        assert written == {*DERIVED, f"{COLLECTION_ID}/thumbnail.png"}

    def test_every_item_survives_the_rebuild(self, regenerated, committed):
        """`rebuild_collection` reads the items and never writes one."""
        for item_dir in sorted((CATALOG / COLLECTION_ID).iterdir()):
            if not item_dir.is_dir():
                continue
            name = f"{item_dir.name}/{item_dir.name}.json"
            source = CATALOG / COLLECTION_ID / name
            assert filecmp.cmp(
                source, regenerated / COLLECTION_ID / name, shallow=False
            )


class TestTheItemMirror:
    def test_the_item_mirror_holds_the_same_table(self, regenerated):
        """The content, separately from the bytes.

        A pyarrow or a stac-geoparquet release can change the encoding without
        changing a single value. That breaks the byte comparison above and
        leaves this one green, which is the difference between "regenerate and
        commit" and "something in the metadata moved".
        """
        import pyarrow.parquet as pq

        committed_table = pq.read_table(CATALOG / COLLECTION_ID / "items.parquet")
        rebuilt = pq.read_table(regenerated / COLLECTION_ID / "items.parquet")
        assert committed_table.column_names == rebuilt.column_names
        assert committed_table.num_rows == rebuilt.num_rows
        assert committed_table.sort_by("id").equals(rebuilt.sort_by("id"))

    def test_the_mirror_describes_every_tracked_item(self):
        """`PTL-MIR-002`, checked against the directory rather than the writer.

        A reader that fetches `items.parquet` instead of 769 item documents
        gets the same set of items, or the mirror is a trap.
        """
        import pyarrow.parquet as pq

        on_disk = {d.name for d in (CATALOG / COLLECTION_ID).iterdir() if d.is_dir()}
        mirror = pq.read_table(
            CATALOG / COLLECTION_ID / "items.parquet", columns=["id"]
        )
        assert set(mirror.column("id").to_pylist()) == on_disk


class TestTheThumbnail:
    def test_the_thumbnail_matches_its_own_checksum(self, committed):
        """The one derived file the offline rebuild cannot redraw.

        Drawing it reads all 769 COGs from the bucket. So the fixture keeps
        the committed PNG, and this is what fails if someone replaces the
        image without rerunning `finish`.
        """
        path = CATALOG / COLLECTION_ID / cog_catalog.THUMBNAIL_FILENAME
        declared = committed["assets"]["thumbnail"]
        assert path.stat().st_size == declared["file:size"]
        assert cog_catalog.multihash_sha256(path) == declared["file:checksum"]

    def test_the_mirror_matches_its_own_checksum(self, committed):
        path = CATALOG / COLLECTION_ID / cog_catalog.MIRROR_FILENAME
        declared = committed["assets"]["items"]
        assert path.stat().st_size == declared["file:size"]
        assert cog_catalog.multihash_sha256(path) == declared["file:checksum"]

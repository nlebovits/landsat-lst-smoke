"""What `lst-publish-catalog sync` is allowed to upload.

The boundary is one directory. `sync` walks `catalog/` and nothing else, and
the reason is that a publish copies a tree wholesale. MEASURED on the
2026-09-14 run: two `qa_count.tif.ovr.tmp` files reached a bucket prefix that
way, one of them 199 MB, written by GDAL while it built overviews and caught
mid-write by an uploader that took everything it found.

The template this pattern comes from states the same rule twice: scope by
path, and scope again by extension. An exclusion list goes stale the first
time a new kind of scratch file appears. An allow-list does not.

So these tests build a tree holding all three kinds of file, and assert set
equality on what `sync` would send. A widened boundary fails here rather than
in the bucket.

The second half is the address. One publish target used to live in this
module's `--dest` default, in `fleet/config.toml`, and in the URLs the
published documents print. Three copies of an address can disagree, and the
symptom is a catalog that answers at one URL and describes itself at another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lst import cog_catalog
from lst.fleet import publish_catalog

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tree(tmp_path):
    """A catalog directory holding one of everything `sync` has to decide on."""
    catalog = tmp_path / "catalog"
    collection = catalog / "lst-p95-2021-2025" / "N00E005"
    collection.mkdir(parents=True)

    published = {
        catalog / "catalog.json": b'{"type": "Catalog"}',
        catalog / "README.md": b"# Landsat LST\n",
        catalog / "AGENTS.md": b"# Agent guidance\n",
        catalog / "lst-p95-2021-2025" / "collection.json": b'{"type": "Collection"}',
        catalog / "lst-p95-2021-2025" / "items.parquet": b"PAR1",
        catalog / "lst-p95-2021-2025" / "thumbnail.png": b"\x89PNG",
        collection / "N00E005.json": b'{"type": "Feature"}',
    }
    for path, blob in published.items():
        path.write_bytes(blob)
    return catalog, set(published)


class TestThePublishBoundary:
    def test_every_metadata_file_is_published(self, tree):
        catalog, published = tree
        assert set(publish_catalog.publishable_files(catalog)) == published

    def test_a_sibling_directory_is_outside_the_boundary(self, tree):
        """`sync` takes a directory, not a repository.

        The tracked tree sits beside the source, the artifacts, and whatever
        a local run left behind. None of it is the publisher's to send.
        """
        catalog, published = tree
        outside = catalog.parent / "artifacts"
        outside.mkdir()
        (outside / "land_buffered.gpkg").write_bytes(b"GPKG")
        (catalog.parent / "notes.md").write_text("scratch\n")
        assert set(publish_catalog.publishable_files(catalog)) == published

    def test_a_raster_inside_the_boundary_stops_the_step(self, tree):
        """The one failure worth stopping for.

        `.gitignore` re-ignores `catalog/**/*.tif`, so a COG here is untracked
        and a reviewer never sees it. Uploading it would replace a published
        raster with whatever a local run happened to write, at hundreds of
        megabytes a tile. `copy` is what puts rasters in the bucket.
        """
        catalog, _published = tree
        (catalog / "lst-p95-2021-2025" / "N00E005" / "lst_p95.tif").write_bytes(b"II*")
        with pytest.raises(SystemExit, match="does not publish"):
            publish_catalog.publishable_files(catalog)

    def test_an_unknown_extension_stops_the_step(self, tree):
        """No guessed content type.

        A file `sync` cannot name a media type for is a file nobody decided to
        publish. Sending it as `binary/octet-stream` would serve it, which is
        worse than refusing it.
        """
        catalog, _published = tree
        (catalog / "scratch.ovr.tmp").write_bytes(b"x")
        with pytest.raises(SystemExit, match="does not publish"):
            publish_catalog.publishable_files(catalog)

    def test_every_published_extension_carries_a_content_type(self):
        for suffix, media_type in publish_catalog.PUBLISHABLE.items():
            assert suffix.startswith(".")
            assert "/" in media_type

    def test_the_tracked_catalog_is_entirely_publishable(self):
        """The real tree, not a fixture.

        This is the test that fires when a run writes its output into the
        checkout. It reads no bucket and uploads nothing.
        """
        files = publish_catalog.publishable_files(ROOT / "catalog")
        assert len(files) > 700
        assert not [p for p in files if p.suffix == ".tif"]


class TestChangeDetection:
    """Size, then ETag. A 777-file catalog with four edits uploads four files."""

    def test_a_file_the_bucket_already_holds_is_skipped(self, tmp_path):
        path = tmp_path / "collection.json"
        path.write_bytes(b'{"type": "Collection"}')
        etag = publish_catalog.local_md5(path)
        assert publish_catalog.is_unchanged(path, (path.stat().st_size, etag))

    def test_a_different_size_is_a_change(self, tmp_path):
        path = tmp_path / "collection.json"
        path.write_bytes(b'{"type": "Collection"}')
        etag = publish_catalog.local_md5(path)
        assert not publish_catalog.is_unchanged(path, (99, etag))

    def test_a_same_size_edit_is_a_change(self, tmp_path):
        """What the ETag is for. Two documents of one length differ."""
        path = tmp_path / "collection.json"
        path.write_bytes(b'{"id": "aaaa"}')
        stale = publish_catalog.local_md5(path)
        path.write_bytes(b'{"id": "bbbb"}')
        assert not publish_catalog.is_unchanged(path, (path.stat().st_size, stale))

    def test_a_multipart_etag_compares_on_size_alone(self, tmp_path):
        """A multipart ETag is not the MD5 of the file.

        It is the MD5 of the concatenated part digests, with a part count
        after a dash. Comparing a local MD5 against it would report every
        large object as changed on every run.
        """
        path = tmp_path / "items.parquet"
        path.write_bytes(b"PAR1" * 100)
        size = path.stat().st_size
        assert publish_catalog.is_unchanged(path, (size, "d41d8cd98f00b204e980-3"))
        assert not publish_catalog.is_unchanged(path, (size + 1, "d41d8-3"))

    def test_an_object_the_bucket_has_never_seen_is_a_change(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text("# Landsat LST\n")
        assert not publish_catalog.is_unchanged(path, None)


class TestOneAddress:
    """The publisher, the deployment assets, and the documents agree."""

    def test_the_default_destination_comes_from_the_fleet_config(self):
        dest = publish_catalog.configured_dest(ROOT / "fleet")
        assert dest.startswith("s3://")
        assert dest == dest.rstrip("/")

    def test_the_documents_name_the_address_the_publisher_writes_to(self):
        """`cog_catalog` cannot read `fleet/config.toml`.

        The import-linter contract `no-tooling-in-the-run-path` forbids
        `lst.cog_catalog` from importing `lst.fleet`, and the run path is
        where the documents are generated. So the address is a constant there
        and a config value here, and this is what keeps the two equal.
        """
        dest = publish_catalog.configured_dest(ROOT / "fleet")
        assert publish_catalog.public_base_for(dest) == (
            cog_catalog.DEFAULT_PUBLIC_BASE
        )

    def test_the_published_documents_use_that_address(self):
        """Not the constant. The bytes in the tracked tree.

        A generator that produces the right URL and a tree published before
        the generator changed are different things, and only the second is
        what a reader fetches.
        """
        base = cog_catalog.DEFAULT_PUBLIC_BASE
        for name in ("AGENTS.md", "lst-p95-2021-2025/AGENTS.md"):
            assert base in (ROOT / "catalog" / name).read_text(encoding="utf-8")

    def test_the_root_catalog_points_back_at_the_repository(self):
        """`vcs` and `issues` are how a reader sends a metadata fix.

        Neither is a Core requirement and rashid checks neither, so nothing
        else fails when they go missing. `docs/conformance.md` records why
        they are here.
        """
        catalog = json.loads((ROOT / "catalog" / "catalog.json").read_text())
        links = {link["rel"]: link["href"] for link in catalog["links"]}
        assert links["vcs"] == cog_catalog.DEFAULT_HOST_URL
        assert links["issues"] == f"{cog_catalog.DEFAULT_HOST_URL}/issues"


class TestOneIdentity:
    """Which AWS profile writes the bucket, decided once.

    Two profiles reach this project and only one can write the published
    prefix. `radiant-earth` is the SSO role the instances assume, and it
    expires in about an hour. `source-coop` is the static IAM user that owns
    the bucket. `fleet/config.toml` has recorded that since the fleet work.

    Nothing read it, so the choice fell to whoever typed the command, and the
    wrong choice fails with "The SSO session associated with this profile has
    expired". That sentence sends the reader to `aws sso login`, which cannot
    help, because the profile that needs no login is the other one.
    """

    def test_the_profile_comes_from_the_fleet_config(self, monkeypatch):
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        assert publish_catalog.configured_profile(ROOT / "fleet") == "source-coop"

    def test_an_explicit_profile_in_the_environment_wins(self, monkeypatch):
        """`AWS_PROFILE` overrides the config without editing it."""
        monkeypatch.setenv("AWS_PROFILE", "someone-else")
        assert publish_catalog.configured_profile(ROOT / "fleet") is None

    def test_explicit_keys_in_the_environment_win(self, monkeypatch):
        """CI holds keys that belong to no profile.

        Passing `--profile` there names a profile the runner does not have,
        and the AWS CLI stops rather than falling back to the keys.
        """
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        assert publish_catalog.configured_profile(ROOT / "fleet") is None

    def test_the_profile_reaches_the_command_line(self, monkeypatch):
        """The argv, not the lookup. `_aws_argv` is what both helpers call."""
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.chdir(ROOT)
        argv = publish_catalog._aws_argv("s3api", ("list-objects-v2",))
        assert argv[:4] == ["aws", "s3api", "--profile", "source-coop"]
        assert argv[-1] == "list-objects-v2"

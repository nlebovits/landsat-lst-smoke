"""Publish finished tiles from their run prefixes into one public catalog.

A run writes a single-item catalog under its own prefix, because one graph
composites one tile and the instance that ran it has no view of the others.
The published collection has to describe all of them. `cog_catalog` already
derives the collection from the items on disk, so this brings the items
together and calls that.

The metadata is git-backed. `catalog/` at the repository root is the source of
truth for every metadata object in the bucket, and this module is the only
writer of that prefix. The COGs never enter git: they are copied inside the
bucket by `copy` and read in place through `/vsis3`.

Five steps, separately runnable, because the first is reversible and the rest
change a tracked tree or a public address:

    uv run lst-publish-catalog plan    --runs <uri>
    uv run lst-publish-catalog copy    --runs <uri>
    uv run lst-publish-catalog recount
    uv run lst-publish-catalog finish
    uv run lst-publish-catalog sync --confirm

`plan` lists what `copy` would write and what it would replace, and writes
nothing. `copy` moves each tile's COGs and item document into the bucket.
`recount` rebuilds each tracked item's coverage block against the land
geometry, touching no raster in the bucket. `finish` rebuilds the collection,
the root catalog, the thumbnail, the item mirror, and both Markdown files from
every tracked item. `sync` uploads `catalog/` and nothing else.

`recount` and `finish` write into `catalog/`, so their output goes through
review and `git diff` before anyone runs `sync`. `sync` without `--confirm`
lists what it would upload and uploads nothing.

`--dest` defaults to the address `fleet/config.toml` states, so the publisher
and the deployment assets cannot name different buckets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from lst import cog_catalog, masks
from lst.fleet.launch import load_config
from lst.lst_qa import LST_NODATA_DN

ITEM_FILES = ("lst_p95.tif", "qa_count.tif")

#: The tracked metadata tree, relative to the repository root.
#:
#: `.gitignore` negates the blanket `**/catalog/` rule for this one directory
#: and re-ignores `catalog/**/*.tif` inside it, so the item documents are
#: tracked and the rasters they describe are not.
DEFAULT_CATALOG_DIR = Path("catalog")

#: Every file extension `sync` will upload, and what it declares each as.
#:
#: An allow-list rather than an exclusion list. A publish copies a tree
#: wholesale, and an exclusion list goes stale the first time a new kind of
#: scratch file appears beside the metadata. MEASURED on the 2026-09-14 run:
#: two `qa_count.tif.ovr.tmp` files reached the runs prefix that way, one of
#: them 199 MB.
#:
#: `.tif` is absent on purpose. The COGs are hundreds of megabytes each and
#: `copy` already puts them in the bucket. A `.tif` under `catalog/` is a
#: mistake, and `sync` stops rather than uploading it.
PUBLISHABLE = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".parquet": "application/vnd.apache.parquet",
    ".png": "image/png",
}

#: How large a part `aws s3 cp` uses before an ETag stops being an MD5.
#:
#: A multipart ETag is the MD5 of the concatenated part digests with a part
#: count after a dash, so it cannot be compared against a local file's MD5.
#: Those objects compare on size alone. Only `items.parquet` is near this
#: size today, at 674 KB, so in practice every ETag here is an MD5.
MULTIPART_MARKER = "-"

#: A publish copies a prefix wholesale, so anything a run left behind becomes
#: part of the published catalog. MEASURED on the 2026-09-14 run: two
#: `qa_count.tif.ovr.tmp` files, one of them 199 MB, which GDAL wrote while
#: building overviews and the uploader caught mid-write.
COPY_EXCLUDE = ("*.tmp", "*.staging.tif", "*.lock")
COLLECTION_FILES = (
    "collection.json",
    "items.parquet",
    "thumbnail.png",
    "README.md",
    "AGENTS.md",
)


def configured_profile(fleet_dir: Path | None = None) -> str | None:
    """The AWS profile that can write to the published prefix.

    `fleet/config.toml:106` has named `source-coop` since the fleet work, and
    nothing read it. So the caller had to know that `radiant-earth` is the SSO
    profile the instances assume, and `source-coop` is the static IAM user that
    owns the bucket. Choosing the first reports an expired SSO session, which
    reads like a login to renew rather than the wrong profile entirely.

    An explicit credential in the environment still wins. `AWS_PROFILE` is how
    a caller overrides the config without editing it, and CI supplies keys that
    belong to no profile at all.
    """
    if os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID"):
        return None
    return load_config(fleet_dir=fleet_dir)["storage"].get("upload_profile")


def _aws_argv(service: str, args: tuple[str, ...]) -> list[str]:
    """`aws <service> ...`, with `--profile` when the config picks one."""
    profile = configured_profile()
    head = ["aws", service, *(("--profile", profile) if profile else ())]
    return [*head, *args]


def s3(*args: str, capture: bool = True) -> str:
    """One `aws s3` call, against the profile `configured_profile` selects."""
    out = subprocess.run(
        _aws_argv("s3", args), check=True, capture_output=capture, text=True
    )
    return out.stdout if capture else ""


def s3api(*args: str) -> str:
    """One `aws s3api` call. `aws s3 ls` does not report an ETag.

    `capture_output` hides stderr, so a bare `check=True` turns every AWS
    failure into a `CalledProcessError` traceback ending in the argv list. The
    most common one says only that the SSO session expired, and that sentence
    never reaches the terminal. Raise `SystemExit` carrying the message
    instead, so the reader sees what to do.

    The message prints the argv with `--profile` in it. That is what tells a
    reader which identity failed, and it is the first thing to check when a
    listing that worked yesterday reports no access today.
    """
    argv = _aws_argv("s3api", args)
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        detail = out.stderr.strip() or f"aws s3api exited {out.returncode}"
        raise SystemExit(f"{detail}\n  while running: {' '.join(argv)}")
    return out.stdout


def split(uri: str) -> tuple[str, str]:
    rest = uri.removeprefix("s3://").rstrip("/")
    bucket, _, key = rest.partition("/")
    return bucket, key


def configured_dest(fleet_dir: Path | None = None) -> str:
    """The published catalog address, as `fleet/config.toml` states it.

    One address, read once. It used to live in this module's `--dest` default,
    in `fleet/config.toml`, and in the URLs the published documents print, and
    three copies of one address can disagree. `tests/test_publish_catalog.py`
    asserts that this and `cog_catalog.DEFAULT_PUBLIC_BASE` describe the same
    prefix.
    """
    storage = load_config(fleet_dir=fleet_dir)["storage"]
    return f"s3://{storage['bucket']}/{storage['published_prefix']}"


def public_base_for(dest_uri: str) -> str:
    """The HTTPS address a reader uses for an `s3://` publish prefix.

    Source Cooperative serves `s3://us-west-2.opendata.source.coop/<path>` at
    `https://data.source.coop/<path>`. The bucket name carries the region and
    the public host does not.
    """
    _bucket, key = split(dest_uri)
    return f"https://data.source.coop/{key}"


def local_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_index(bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
    """Every published key under `prefix`, mapped to its size and ETag.

    One listing rather than one `head-object` per file. A 777-file catalog is
    777 round trips the other way, and the listing already carries both fields
    change detection needs.
    """
    index: dict[str, tuple[int, str]] = {}
    token: str | None = None
    while True:
        argv = [
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix.rstrip("/") + "/",
            "--output",
            "json",
        ]
        if token:
            argv += ["--starting-token", token]
        page = json.loads(s3api(*argv) or "{}")
        for obj in page.get("Contents", ()):
            index[obj["Key"]] = (int(obj["Size"]), obj["ETag"].strip('"'))
        token = page.get("NextToken") or page.get("NextContinuationToken")
        if not token:
            return index


def is_unchanged(path: Path, published: tuple[int, str] | None) -> bool:
    """Whether the bucket already holds this exact file.

    Size first, because it is free and it settles almost every comparison. The
    ETag settles the rest, except on a multipart object, whose ETag is not an
    MD5 of the whole file. Those compare on size alone, which can miss an edit
    that preserves the byte count. `--force` exists for that case.
    """
    if published is None:
        return False
    size, etag = published
    if path.stat().st_size != size:
        return False
    if MULTIPART_MARKER in etag:
        return True
    return local_md5(path) == etag


def publishable_files(catalog_dir: Path) -> list[Path]:
    """Every file `sync` may upload, sorted, with the boundary enforced.

    The boundary is the whole point of this function. `sync` walks
    `catalog_dir` and nothing else, and it refuses an extension outside
    `PUBLISHABLE` rather than guessing a content type for it.

    Raises:
        SystemExit: if the tree holds a file this module will not publish.
    """
    found = sorted(p for p in catalog_dir.rglob("*") if p.is_file())
    refused = [p for p in found if p.suffix.lower() not in PUBLISHABLE]
    if refused:
        listing = "\n".join(f"  {p}" for p in refused[:20])
        more = "" if len(refused) <= 20 else f"\n  ... and {len(refused) - 20} more"
        raise SystemExit(
            f"{len(refused)} file(s) under {catalog_dir} have an extension "
            f"this module does not publish:\n{listing}{more}\n"
            f"Publishable extensions: {', '.join(sorted(PUBLISHABLE))}. A "
            f"`.tif` here means a run wrote its rasters into the tracked "
            f"tree; `copy` is what puts those in the bucket."
        )
    return found


def find_tiles(runs_uri: str, collection_id: str) -> dict[str, str]:
    """Each tile id under `runs_uri`, mapped to the item prefix that holds it.

    A tile that ran more than once appears under more than one run prefix, and
    the most recently written item wins.

    The tie-break reads the object's own timestamp rather than sorting the run
    ids. MEASURED on 2026-09-14: sorting by key chose `lst-tile-20260913-125234`
    over `lst-S30W065-20260914-170047`, because an uppercase `S` precedes a
    lowercase `t`. The run ids come from several eras and share no convention,
    so their names carry no order. That would have published the tile this
    day's work exists to replace.
    """
    listing = s3("ls", "--recursive", runs_uri.rstrip("/") + "/")
    found: dict[str, tuple[str, str]] = {}
    bucket, _ = split(runs_uri)
    marker = f"/catalog/{collection_id}/"
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        stamp, key = f"{fields[0]} {fields[1]}", fields[-1]
        if marker not in key or not key.endswith(".json"):
            continue
        tail = key.split(marker, 1)[1]
        parts = tail.split("/")
        if len(parts) != 2 or parts[1] != f"{parts[0]}.json":
            continue
        tile = parts[0]
        prefix = f"s3://{bucket}/{key.rsplit('/', 1)[0]}"
        if tile not in found or stamp > found[tile][0]:
            found[tile] = (stamp, prefix)
    return {tile: prefix for tile, (_, prefix) in sorted(found.items())}


def existing(dest_uri: str) -> list[str]:
    try:
        listing = s3("ls", "--recursive", dest_uri.rstrip("/") + "/")
    except subprocess.CalledProcessError:
        return []
    return [line.split()[-1] for line in listing.splitlines() if line.strip()]


def cmd_plan(a) -> int:
    tiles = find_tiles(a.runs, a.collection)
    here = existing(a.dest)
    print(f"source runs   {a.runs}")
    print(f"destination   {a.dest}")
    print(f"collection    {a.collection}\n")
    print(f"{len(tiles)} tile(s) to publish:")
    for tile, prefix in tiles.items():
        print(f"  {tile:10} <- {prefix}")
    keep = {f"{t}/{f}" for t in tiles for f in ITEM_FILES}
    keep |= {f"{t}/{t}.json" for t in tiles}
    keep |= set(COLLECTION_FILES)
    _, dest_key = split(a.dest)
    # Only the collection is this script's to describe. Sibling prefixes such
    # as `inputs/` hold the artifacts a run reads, belong to whoever put them
    # there, and are none of its business.
    inside = f"{dest_key}/{a.collection}/".lstrip("/")
    orphans = [
        k for k in here if k.startswith(inside) and k.removeprefix(inside) not in keep
    ]
    print(
        f"\n{len(here)} object(s) already at the destination, "
        f"{sum(1 for k in here if k.startswith(inside))} inside the "
        f"collection."
    )
    if orphans:
        print(
            f"{len(orphans)} inside the collection would NOT be replaced, "
            f"and no item references them:"
        )
        for k in orphans:
            print(f"  {k}")
        print(
            "\nThey are not deleted by `copy` or `finish`. Remove them "
            "yourself once you have read the list."
        )
    return 0


def cmd_copy(a) -> int:
    tiles = find_tiles(a.runs, a.collection)
    if not tiles:
        print("no tiles found under the run prefix", file=sys.stderr)
        return 1
    for tile, prefix in tiles.items():
        target = f"{a.dest.rstrip('/')}/{a.collection}/{tile}"
        print(f"{tile}: {prefix} -> {target}", flush=True)
        argv = ["cp", prefix + "/", target + "/", "--recursive"]
        for pattern in COPY_EXCLUDE:
            argv += ["--exclude", pattern]
        s3(*argv, capture=False)
    print(
        f"\n{len(tiles)} tile(s) copied. Run `finish` to rebuild the "
        f"collection from them."
    )
    return 0


def published_items(bucket: str, prefix: str) -> list[str]:
    """Every item document key under a published collection prefix.

    An item lives at `<tile>/<tile>.json`, which is what tells it apart from
    `collection.json` and from anything else a run left in the prefix.
    """
    listing = s3("ls", "--recursive", f"s3://{bucket}/{prefix}/")
    return [
        k
        for k in (line.split()[-1] for line in listing.splitlines())
        if k.endswith(".json") and Path(k).stem == Path(k).parent.name
    ]


def lst_uri_for(bucket: str, prefix: str, tile: str) -> str:
    """Where the published temperature raster is, as GDAL should open it.

    `/vsis3` reads ranges out of the object in place, so a recount and a
    thumbnail both touch the bytes without downloading the file. Named as a
    function because a test points it at a local tree, and because the two
    callers should not spell the path twice.
    """
    return f"/vsis3/{bucket}/{prefix}/{tile}/{ITEM_FILES[0]}"


def grid_of(lst_uri: str) -> tuple[tuple[float, float, float, float], int, int, int]:
    """The tile's bbox and grid, read off the published raster.

    The raster is asked rather than the item, because the item carries no
    `proj:` fields and a bbox alone does not fix a pixel grid. A recount that
    rasterised its masks on a grid the COG does not use would return a
    plausible number for the wrong ground.

    Returns:
        `(bbox, pixels_per_degree, height, width)`.

    Raises:
        SystemExit: if the raster is not on a whole number of pixels per degree,
            or if its own shape disagrees with the bbox that grid implies.
    """
    import rasterio

    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(lst_uri) as src:
        transform, height, width = src.transform, src.height, src.width
    per_degree = 1.0 / transform.a
    pixels_per_degree = int(round(per_degree))
    if abs(per_degree - pixels_per_degree) > 1e-6:
        raise SystemExit(
            f"{lst_uri} has {per_degree} pixels per degree, which is not a "
            f"whole number. The land masks cannot be rasterised to match it."
        )
    west, north = transform.c, transform.f
    bbox = (
        west,
        north - height / pixels_per_degree,
        west + width / pixels_per_degree,
        north,
    )
    if masks.raster_shape(bbox, pixels_per_degree) != (height, width):
        raise SystemExit(
            f"{lst_uri} is {(height, width)} and its bounds imply "
            f"{masks.raster_shape(bbox, pixels_per_degree)}"
        )
    return bbox, pixels_per_degree, height, width


def recount_item(item: dict, lst_uri: str, a) -> dict:
    """The coverage block this item should carry, from its own raster.

    Reads the published `lst_p95` once, through `/vsis3`, and counts the
    surviving values inside land and inside the coastal buffer in the same pass.
    The geometries come from local artifacts. Nothing is written here.

    The rules are not re-derived. `masks.output_mask` and `masks.land_split`
    build the same masks a run built, and `masks.coverage` does the same
    arithmetic the run did, so a recounted item and a freshly composited one
    cannot disagree about what a property means.

    Raises:
        SystemExit: if the item's own bbox disagrees with its raster.
    """
    bbox, pixels_per_degree, height, width = grid_of(lst_uri)
    stated = [round(float(v), 9) for v in item["bbox"]]
    if [round(float(v), 9) for v in bbox] != stated:
        raise SystemExit(
            f"{item['id']} states bbox {stated} and its raster covers "
            f"{[round(float(v), 9) for v in bbox]}"
        )
    keep, gap, mask_counts = masks.output_mask(
        bbox,
        pixels_per_degree,
        numobs_uri=a.numobs,
        land_geometry_uri=a.land_geometry_uri,
        buffer_cells=a.gap_buffer_cells,
    )
    strict, _, split = masks.land_split(
        bbox,
        pixels_per_degree,
        strict_land_geometry_uri=a.strict_land_geometry_uri,
        processing=keep,
        gap=gap,
    )
    counts = masks.count_valid_within(
        lst_uri,
        LST_NODATA_DN,
        land=strict,
        coast=keep & ~strict,
    )
    statistics = [{"kept": counts["total"], "total": height * width}]
    fresh = masks.coverage(mask_counts, statistics, split=split, valid=counts)
    if fresh is None:
        # `coverage` returns None only for an empty processing mask, and
        # `shard_lst_p95.no_unmasked_pixels` refuses to publish a tile with
        # one. Reaching this means a published item describes ground the mask
        # keeps nothing of, which is worth stopping for rather than widening
        # this function's return type to carry a case that cannot happen.
        raise SystemExit(
            f"{item['id']} has an empty processing mask, so there is nothing "
            f"to recount. A tile that masks to nothing should never have been "
            f"published."
        )
    return fresh


def rewrite_coverage(item: dict, fresh: dict, sentence: str, *, dry_run: bool) -> bool:
    """Print one item's coverage diff, and apply it unless this is a dry run.

    Reports whether the item changed, so the caller counts and uploads only what
    moved. An item already recounted under the same geometry changes nothing,
    which is what makes the step safe to repeat: the properties match and the
    lineage already carries `sentence`.

    Mutates `item` in place when `dry_run` is false. The old coverage keys are
    removed before the new ones go in, so a property that stops being produced
    stops being published rather than standing at a stale value.
    """
    properties = item["properties"]
    names = [name for _, name, _ in cog_catalog.COVERAGE_PROPERTIES]
    # Only the keys the item actually carries. Filling every name with None
    # compares a dense dict against a sparse one, and `coverage_properties` is
    # sparse on purpose: a tile with no strict land omits every share that
    # would divide by it.
    #
    # MEASURED on 2026-09-15: `S35W055` holds 0 land pixels, so it publishes
    # no `lst:valid_fraction` and no `lst:ged_gap_fraction`. Under the dense
    # comparison it read as changed on every recount while every printed value
    # matched, so it would be rewritten forever, each time with a new
    # `updated` timestamp and nothing else.
    #
    # A key the item carries and `after` does not is still a change. That is
    # the stale property this function exists to remove.
    before = {name: properties[name] for name in names if name in properties}
    after = cog_catalog.coverage_properties(fresh)
    # Flushed per tile. A 30-tile recount reads a gigabyte and runs for tens of
    # minutes, and MEASURED on 2026-09-15 the first attempt died at tile 18 with
    # eighteen tiles of report still in the buffer. A step this long has to
    # report as it goes or a failure takes its own evidence with it.
    print(f"{item['id']}", flush=True)
    for name in names:
        old, new = before.get(name), after.get(name)
        if old != new:
            print(f"  {name:38} {_show(old)} -> {_show(new)}", flush=True)
    if before == after and sentence in properties.get("processing:lineage", ""):
        print("  unchanged", flush=True)
        return False
    if dry_run:
        return True
    for name in names:
        properties.pop(name, None)
    properties |= after
    lineage = properties.get("processing:lineage", "")
    if lineage and sentence not in lineage:
        properties["processing:lineage"] = f"{lineage} {sentence}"
    properties["updated"] = cog_catalog.now_utc()
    return True


def cmd_recount(a) -> int:
    """Rebuild each published item's coverage block against the land geometry.

    `lst:land_pixels` used to report the processing mask, which is Natural Earth
    land grown by 25 km so that a coastal scene is not cut at the waterline.
    Every share of land divided by it. MEASURED 2026-09-15 at 3600 pixels per
    degree, that denominator is 73,254,945 on S40W065 where the land is
    45,407,126, so a coastal tile understated its own coverage by a third.

    This is the one step that reads pixels. It reads each published `lst_p95`
    once, through `/vsis3`, which is 33 MB to 471 MB per tile on the five
    published in September 2026. It writes item JSON under `catalog/` and
    nothing else: no raster is rewritten, no composite runs, and the bucket is
    not touched. `sync` is what puts the result in the bucket, after a human
    has read `git diff`.

    `--dry-run` prints each tile's old and new block and writes nothing.
    `--tile` narrows it to named tiles, so the first run against a real catalog
    can read 33 MB rather than a gigabyte.
    """
    bucket, key = split(a.dest)
    prefix = f"{key}/{a.collection}".strip("/")
    collection_dir = a.catalog / a.collection
    items = sorted(
        p for p in collection_dir.glob("*/*.json") if p.stem == p.parent.name
    )
    if not items:
        print(f"no items under {collection_dir}", file=sys.stderr)
        return 1
    wanted = set(getattr(a, "tile", None) or ())
    if wanted:
        items = [p for p in items if p.parent.name in wanted]
        missing = wanted - {p.parent.name for p in items}
        if missing:
            print(
                f"not tracked under {collection_dir}: {', '.join(sorted(missing))}",
                file=sys.stderr,
            )
            return 1
    if not Path(a.strict_land_geometry_uri).exists():
        print(
            f"no strict land geometry at {a.strict_land_geometry_uri}. Write "
            f"it with:\n  uv run lst-land-tiles --out "
            f"artifacts/land_tiles.parquet --write-strict-geometry "
            f"{a.strict_land_geometry_uri}",
            file=sys.stderr,
        )
        return 1
    digest = masks.geometry_checksum(a.strict_land_geometry_uri)
    sentence = cog_catalog.strict_land_sentence(digest)
    print(f"land geometry {a.strict_land_geometry_uri}, sha256 {digest[:16]}")
    print(f"gap buffer    {a.gap_buffer_cells} GED cell(s)")
    print(f"{len(items)} item(s) to recount\n")
    changed = 0
    for item_path in items:
        tile = item_path.parent.name
        item = json.loads(item_path.read_text())
        fresh = recount_item(item, lst_uri_for(bucket, prefix, tile), a)
        if not rewrite_coverage(item, fresh, sentence, dry_run=a.dry_run):
            continue
        changed += 1
        if a.dry_run:
            continue
        item_path.write_text(json.dumps(item, indent=2) + "\n")
    if a.dry_run:
        print(f"\n{changed} item(s) would change. Nothing was written.")
        return 0
    print(
        f"\n{changed} item(s) rewritten under {a.catalog}, rasters untouched. "
        f"Read `git diff`, then run `finish` to rebuild the collection."
    )
    return 0


def _show(value) -> str:
    """A coverage value for the diff, or a dash when the item did not carry it."""
    if value is None:
        return "—"
    return f"{value:,}" if isinstance(value, int) else f"{value:.6f}"


def cmd_finish(a) -> int:
    """Rebuild every derived document in `catalog/` from the items it holds.

    The item documents are primary: a run writes one, and `recount` rewrites
    one. Everything above them derives from them, so this reads them off disk
    and calls `cog_catalog.rebuild_collection`. Nothing is uploaded. Read the
    diff, commit it, then run `sync`.

    The thumbnail is the one output that needs pixels. It reads the published
    rasters through `/vsis3` rather than downloading them, so a 769-tile
    rebuild moves a few hundred kilobytes to draw a 480 px preview.
    """
    bucket, key = split(a.dest)
    prefix = f"{key}/{a.collection}".strip("/")
    collection_dir = a.catalog / a.collection
    items = sorted(
        p.parent.name
        for p in collection_dir.glob("*/*.json")
        if p.stem == p.parent.name
    )
    if not items:
        print(f"no items under {collection_dir}", file=sys.stderr)
        return 1
    print(f"read {len(items)} item(s) from {collection_dir}")

    def lst_uri(item_id: str) -> str:
        return lst_uri_for(bucket, prefix, item_id)

    cog_catalog.rebuild_collection(
        a.catalog,
        a.collection,
        license_id=a.license,
        lst_uri=lst_uri,
        public_base=public_base_for(a.dest),
    )
    print(
        f"collection, catalog, thumbnail, mirror and docs rebuilt under "
        f"{a.catalog}. Read `git diff`, commit, then run `sync --confirm`."
    )
    return 0


def cmd_sync(a) -> int:
    """Upload `catalog/` to the published prefix, and nothing else.

    Two guards, and both matter more than the upload does.

    The publish boundary is `publishable_files`: this walks `catalog/` and
    refuses any extension outside `PUBLISHABLE`. `tests/test_publish_boundary.py`
    builds a tree of tracked-and-published, tracked-not-published, and ignored
    files, and asserts set equality on what this would send.

    Change detection compares local size and MD5 against the listing's size and
    ETag. A 777-file catalog where four documents moved uploads four objects.

    This never deletes. A key under the prefix with no local file is printed
    and left alone, because the COGs live under the same prefix and a delete
    pass would take 400 GB of them with it.
    """
    catalog_dir = a.catalog
    if not catalog_dir.is_dir():
        print(f"no catalog at {catalog_dir}", file=sys.stderr)
        return 1
    bucket, key = split(a.dest)
    files = publishable_files(catalog_dir)
    if not files:
        print(f"no publishable file under {catalog_dir}", file=sys.stderr)
        return 1
    published = remote_index(bucket, key)

    planned: list[tuple[Path, str]] = []
    for path in files:
        remote_key = f"{key}/{path.relative_to(catalog_dir).as_posix()}".lstrip("/")
        if not a.force and is_unchanged(path, published.get(remote_key)):
            continue
        planned.append((path, remote_key))

    local_keys = {
        f"{key}/{p.relative_to(catalog_dir).as_posix()}".lstrip("/") for p in files
    }
    # Metadata only. The COGs share this prefix and no local file describes
    # them, so every raster would read as an orphan.
    orphans = sorted(
        k
        for k in published
        if k not in local_keys and Path(k).suffix.lower() in PUBLISHABLE
    )

    by_suffix: dict[str, int] = {}
    for path, _ in planned:
        by_suffix[path.suffix.lower()] = by_suffix.get(path.suffix.lower(), 0) + 1

    print(f"source        {catalog_dir}")
    print(f"destination   {a.dest}")
    print(f"{len(files)} publishable file(s), {len(published)} object(s) published")
    print(f"{len(planned)} to upload:")
    for suffix in sorted(by_suffix):
        print(f"  {suffix:10} {by_suffix[suffix]:>5}")
    if orphans:
        print(
            f"\n{len(orphans)} published metadata object(s) have no local "
            f"file. They are NOT deleted:"
        )
        for k in orphans[:20]:
            print(f"  {k}")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")

    if not a.confirm:
        print("\nNothing was uploaded. Re-run with --confirm.")
        return 0
    for index, (path, remote_key) in enumerate(planned, start=1):
        print(f"[{index}/{len(planned)}] {remote_key}", flush=True)
        s3(
            "cp",
            str(path),
            f"s3://{bucket}/{remote_key}",
            "--content-type",
            PUBLISHABLE[path.suffix.lower()],
            capture=False,
        )
    print(f"\n{len(planned)} object(s) uploaded to {a.dest}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("plan", "copy", "recount", "finish", "sync"))
    p.add_argument("--runs", help="prefix holding the finished run catalogs")
    p.add_argument(
        "--dest",
        default=None,
        help="published catalog root. Defaults to the bucket and prefix "
        "fleet/config.toml states",
    )
    p.add_argument(
        "--fleet-dir",
        type=Path,
        default=None,
        help="the repository's fleet/ directory, holding config.toml",
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help="the tracked metadata tree. recount and finish write here, and "
        "sync uploads it",
    )
    p.add_argument("--collection", default="lst-p95-2021-2025")
    p.add_argument("--license", default=cog_catalog.DEFAULT_LICENSE)
    p.add_argument(
        "--numobs",
        type=Path,
        default=Path("artifacts/aster_numobs.tif"),
        help="ASTER GED clear-sky counts, for the gap share of land",
    )
    p.add_argument(
        "--land-geometry-uri",
        type=Path,
        default=masks.DEFAULT_LAND_GEOMETRY_URI,
        help="the buffered land geometry the published tiles were masked with",
    )
    p.add_argument(
        "--strict-land-geometry-uri",
        type=Path,
        default=masks.STRICT_LAND_GEOMETRY_URI,
        help="the unbuffered land geometry the coverage counts divide by",
    )
    p.add_argument(
        "--gap-buffer-cells",
        type=int,
        default=masks.GAP_BUFFER_CELLS,
        help="how far the reported ASTER gap region grows. It has to match the "
        "value the tiles were built under, which their processing:lineage "
        "states",
    )
    p.add_argument(
        "--tile",
        action="append",
        metavar="TILE",
        help="recount only: restrict to this tile, repeatable. Without it every "
        "published item is recounted",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="recount only: print each item's old and new coverage block and "
        "write nothing",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="sync only: upload. Without it sync lists what it would send",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="sync only: upload every publishable file, whatever the listing "
        "says. Use it after changing a content type, which a listing does "
        "not report",
    )
    a = p.parse_args()
    if a.step in ("plan", "copy") and not a.runs:
        p.error(f"--runs is required for {a.step}")
    if a.dest is None:
        a.dest = configured_dest(a.fleet_dir)
    return {
        "plan": cmd_plan,
        "copy": cmd_copy,
        "recount": cmd_recount,
        "finish": cmd_finish,
        "sync": cmd_sync,
    }[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())

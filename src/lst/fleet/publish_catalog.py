"""Publish finished tiles from their run prefixes into one public catalog.

A run writes a single-item catalog under its own prefix, because one graph
composites one tile and the instance that ran it has no view of the others.
The published collection has to describe all of them. `cog_catalog` already
derives the collection from the items on disk, so this brings the items
together and calls that.

`plan`, `copy`, and `finish` move nothing large through this machine. The
rasters are copied inside the bucket, and the thumbnail reads them through
`/vsis3` from their published home, so those three transfer a few hundred
kilobytes of JSON. `recount` is the exception, and its docstring says what it
reads.

Four steps, separately runnable, because the first is reversible and the rest
change a public address:

    uv run lst-publish-catalog plan    --runs <uri> --dest <uri>
    uv run lst-publish-catalog copy    --runs <uri> --dest <uri>
    uv run lst-publish-catalog recount --dest <uri>
    uv run lst-publish-catalog finish  --dest <uri>

`plan` lists what `copy` would write and what it would replace, and writes
nothing. `copy` moves each tile's item directory into place. `recount` rebuilds
each item's coverage block against the land geometry, touching no raster.
`finish` rebuilds the collection, the root catalog, the thumbnail, the item
mirror, and both Markdown files from every item then present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lst import cog_catalog, masks
from lst.lst_qa import LST_NODATA_DN

ITEM_FILES = ("lst_p95.tif", "qa_count.tif")

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


def s3(*args: str, capture: bool = True) -> str:
    """One `aws s3` call, with the profile the caller's environment selects."""
    out = subprocess.run(
        ["aws", "s3", *args], check=True, capture_output=capture, text=True
    )
    return out.stdout if capture else ""


def split(uri: str) -> tuple[str, str]:
    rest = uri.removeprefix("s3://").rstrip("/")
    bucket, _, key = rest.partition("/")
    return bucket, key


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
    before = {name: properties.get(name) for name in names}
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

    This is the one step that reads pixels. It reads each `lst_p95` once,
    through `/vsis3`, which is 33 MB to 471 MB per tile on the five published in
    September 2026. It writes item JSON and nothing else: no raster is
    rewritten, and no composite runs.

    `--dry-run` prints each tile's old and new block and writes nothing.
    `--tile` narrows it to named tiles, so the first run against a real catalog
    can read 33 MB rather than a gigabyte.
    """
    bucket, key = split(a.dest)
    prefix = f"{key}/{a.collection}".strip("/")
    items = published_items(bucket, prefix)
    if not items:
        print("no items at the destination", file=sys.stderr)
        return 1
    wanted = set(getattr(a, "tile", None) or ())
    if wanted:
        items = [k for k in items if Path(k).parent.name in wanted]
        missing = wanted - {Path(k).parent.name for k in items}
        if missing:
            print(
                f"not published at the destination: {', '.join(sorted(missing))}",
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
    work = Path(tempfile.mkdtemp(prefix="recount-"))
    print(f"land geometry {a.strict_land_geometry_uri}, sha256 {digest[:16]}")
    print(f"gap buffer    {a.gap_buffer_cells} GED cell(s)")
    print(f"{len(items)} item(s) to recount\n")
    changed = 0
    for item_key in sorted(items):
        tile = Path(item_key).parent.name
        local = work / f"{tile}.json"
        s3("cp", f"s3://{bucket}/{item_key}", str(local), capture=False)
        item = json.loads(local.read_text())
        fresh = recount_item(item, lst_uri_for(bucket, prefix, tile), a)
        if not rewrite_coverage(item, fresh, sentence, dry_run=a.dry_run):
            continue
        changed += 1
        if a.dry_run:
            continue
        local.write_text(json.dumps(item, indent=2) + "\n")
        s3("cp", str(local), f"s3://{bucket}/{item_key}", capture=False)
    shutil.rmtree(work, ignore_errors=True)
    if a.dry_run:
        print(f"\n{changed} item(s) would change. Nothing was written.")
        return 0
    print(
        f"\n{changed} item(s) rewritten, rasters untouched. Run `finish` to "
        f"rebuild the collection from them."
    )
    return 0


def _show(value) -> str:
    """A coverage value for the diff, or a dash when the item did not carry it."""
    if value is None:
        return "—"
    return f"{value:,}" if isinstance(value, int) else f"{value:.6f}"


def cmd_finish(a) -> int:
    """Rebuild every collection-level document from the items now published."""
    bucket, key = split(a.dest)
    prefix = f"{key}/{a.collection}".strip("/")
    work = Path(tempfile.mkdtemp(prefix="publish-"))
    root = work / "catalog"
    (root / a.collection).mkdir(parents=True)
    items = published_items(bucket, prefix)
    if not items:
        print("no items at the destination", file=sys.stderr)
        return 1
    for k in items:
        tile = Path(k).parent.name
        (root / a.collection / tile).mkdir(exist_ok=True)
        s3(
            "cp",
            f"s3://{bucket}/{k}",
            str(root / a.collection / tile / f"{tile}.json"),
            capture=False,
        )
    print(
        f"read {len(items)} item(s): "
        f"{', '.join(sorted(Path(k).parent.name for k in items))}"
    )

    def lst_uri(item_id: str) -> str:
        return lst_uri_for(bucket, prefix, item_id)

    cog_catalog.rebuild_collection(
        root, a.collection, license_id=a.license, lst_uri=lst_uri
    )
    for name in COLLECTION_FILES:
        s3(
            "cp",
            str(root / a.collection / name),
            f"s3://{bucket}/{prefix}/{name}",
            capture=False,
        )
    root_prefix = "/".join(part for part in (bucket, key) if part)
    for name in ("catalog.json", "README.md", "AGENTS.md"):
        s3("cp", str(root / name), f"s3://{root_prefix}/{name}", capture=False)
    shutil.rmtree(work, ignore_errors=True)
    print(
        "collection, catalog, thumbnail, mirror and docs rebuilt from the "
        "published items"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("plan", "copy", "recount", "finish"))
    p.add_argument("--runs", help="prefix holding the finished run catalogs")
    p.add_argument("--dest", required=True, help="published catalog root")
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
    a = p.parse_args()
    if a.step in ("plan", "copy") and not a.runs:
        p.error(f"--runs is required for {a.step}")
    return {
        "plan": cmd_plan,
        "copy": cmd_copy,
        "recount": cmd_recount,
        "finish": cmd_finish,
    }[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())

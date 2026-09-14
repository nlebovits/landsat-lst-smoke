# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "boto3",
#   "matplotlib",
#   "numpy",
#   "pyarrow>=16",
#   "rasterio",
# ]
# ///
"""Publish finished tiles from their run prefixes into one public catalog.

A run writes a single-item catalog under its own prefix, because one graph
composites one tile and the instance that ran it has no view of the others.
The published collection has to describe all of them. `cog_catalog` already
derives the collection from the items on disk, so this brings the items
together and calls that.

Nothing large moves through this machine. The rasters are copied inside the
bucket, and the thumbnail reads them through `/vsis3` from their published
home, so a five-tile publish transfers a few hundred kilobytes of JSON.

Two steps, separately runnable, because the first is reversible and the second
is the one that changes a public address:

    uv run publish_catalog.py plan   --runs <uri> --dest <uri>
    uv run publish_catalog.py copy   --runs <uri> --dest <uri>
    uv run publish_catalog.py finish --dest <uri>

`plan` lists what `copy` would write and what it would replace, and writes
nothing. `copy` moves each tile's item directory into place. `finish` rebuilds
the collection, the root catalog, the thumbnail, the item mirror, and both
Markdown files from every item then present.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import cog_catalog  # noqa: E402

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


def cmd_finish(a) -> int:
    """Rebuild every collection-level document from the items now published."""
    bucket, key = split(a.dest)
    prefix = f"{key}/{a.collection}".strip("/")
    work = Path(tempfile.mkdtemp(prefix="publish-"))
    root = work / "catalog"
    (root / a.collection).mkdir(parents=True)
    listing = s3("ls", "--recursive", f"s3://{bucket}/{prefix}/")
    items = [
        k
        for k in (line.split()[-1] for line in listing.splitlines())
        if k.endswith(".json") and Path(k).stem == Path(k).parent.name
    ]
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
        return f"/vsis3/{bucket}/{prefix}/{item_id}/{ITEM_FILES[0]}"

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
    p.add_argument("step", choices=("plan", "copy", "finish"))
    p.add_argument("--runs", help="prefix holding the finished run catalogs")
    p.add_argument("--dest", required=True, help="published catalog root")
    p.add_argument("--collection", default="lst-p95-2021-2025")
    p.add_argument("--license", default=cog_catalog.DEFAULT_LICENSE)
    a = p.parse_args()
    if a.step in ("plan", "copy") and not a.runs:
        p.error(f"--runs is required for {a.step}")
    return {"plan": cmd_plan, "copy": cmd_copy, "finish": cmd_finish}[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())

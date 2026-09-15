"""Check every published raster over the network, before anyone depends on it.

`cog_catalog._verify_cog` already checks a raster, on the instance, against a
local path, at the moment it is written. That catches a bad writer. It cannot
catch anything that happens afterwards, and what happens afterwards is the
upload.

An upload cut part way leaves a file whose header is complete and whose tail is
missing. It opens. It reports the right shape, the right block size, and the
right overviews. Every check that reads only the header passes. The pixels at
the bottom of the image are not there. MEASURED on 2026-09-15: a driver bug
terminated an instance while its uploader was still running, and the only
reason `N00W045` kept its rasters is that they had gone up seconds earlier.

So this reads pixels. It opens each raster through `/vsis3`, the way a client
does, and pulls a window from the far corner at full resolution and the whole
of the smallest overview. A file missing its tail fails on the corner read.

    uv run lst-fleet-verify --runs s3://BUCKET/PREFIX/runs \
        --tiles-file artifacts/south_america.txt
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lst.cog_catalog import BLOCK_SIZE, LST_FILENAME, QA_FILENAME
from lst.fleet.publish_catalog import find_tiles
from lst.fleet.waves import load_tiles

#: Pixels read from the far corner of the full-resolution image.
#:
#: One block is enough. GDAL fetches the byte range that block occupies, and a
#: file whose tail never arrived cannot answer it. Reading more would cost
#: bandwidth to learn the same fact.
CORNER = BLOCK_SIZE

#: How many rasters are checked at once. Each check is a handful of range
#: requests against S3 and spends its time waiting, not computing.
THREADS = 16


def check_raster(uri: str) -> str | None:
    """Read enough of one raster to prove it is whole. Returns a fault or None.

    The order matters. Structure is checked first, because a wrong block size
    is a writer fault and says nothing about the upload. The corner read comes
    last, because it is the only check that can fail on a complete file for a
    reason worth reporting differently.
    """
    import rasterio

    try:
        with rasterio.Env(GDAL_PAM_ENABLED="NO", AWS_NO_SIGN_REQUEST="NO"):
            with rasterio.open(uri) as src:
                if src.block_shapes[0] != (BLOCK_SIZE, BLOCK_SIZE):
                    return f"internal tiles are {src.block_shapes[0]}"
                if max(src.height, src.width) > BLOCK_SIZE and not src.overviews(1):
                    return "larger than one block and carries no overviews"
                levels = src.overviews(1)
                if levels:
                    # The smallest overview lives at the end of a COG, after
                    # every full-resolution block. Reading all of it is cheap
                    # and touches the last bytes of the file.
                    src.read(
                        1,
                        out_shape=(
                            1,
                            src.height // levels[-1] or 1,
                            src.width // levels[-1] or 1,
                        ),
                    )
                # A slice pair rather than `Window(...)`: rasterio accepts
                # both, and `Window` is an attrs class whose generated
                # signature the type checker cannot read.
                rows = (max(src.height - CORNER, 0), src.height)
                cols = (max(src.width - CORNER, 0), src.width)
                src.read(1, window=(rows, cols))
    except Exception as err:
        return f"{type(err).__name__}: {err}".replace("\n", " ")[:200]
    return None


def check_tile(tile: str, prefix: str) -> tuple[str, list[str]]:
    """Both rasters of one tile. Returns `(tile, faults)`."""
    faults = []
    for name in (LST_FILENAME, QA_FILENAME):
        uri = f"{prefix.rstrip('/')}/{name}".replace("s3://", "/vsis3/", 1)
        fault = check_raster(uri)
        if fault:
            faults.append(f"{name}: {fault}")
    return tile, faults


def verify(
    runs_uri: str,
    collection_id: str,
    *,
    tiles: list[str] | None = None,
    threads: int = THREADS,
    say=print,
) -> dict[str, list[str]]:
    """Check every tile found under `runs_uri`. Returns the faults, by tile.

    A tile named in `tiles` that the bucket does not hold is a fault of its
    own. Reporting only on what was found would call a missing tile clean.
    """
    found = find_tiles(runs_uri, collection_id)
    if tiles is not None:
        missing = [t for t in tiles if t not in found]
        found = {t: p for t, p in found.items() if t in set(tiles)}
    else:
        missing = []

    faults: dict[str, list[str]] = {t: ["no item in the bucket"] for t in missing}
    say(f"checking {len(found)} tile(s), {len(missing)} missing")
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for tile, bad in pool.map(
            lambda item: check_tile(*item), sorted(found.items())
        ):
            if bad:
                faults[tile] = bad
                say(f"  {tile:9} FAULT  {'; '.join(bad)}")
            else:
                say(f"  {tile:9} ok")
    return faults


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, help="prefix holding the run catalogs")
    p.add_argument("--collection", default="lst-p95-2021-2025")
    p.add_argument(
        "--tiles-file",
        type=Path,
        help="only these tiles, and report any of them the bucket lacks",
    )
    p.add_argument(
        "--tiles",
        nargs="+",
        help="tile ids on the command line, in place of --tiles-file",
    )
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument(
        "--profile",
        help="AWS profile for the reads. Defaults to the caller's environment",
    )
    a = p.parse_args(argv)

    if a.profile:
        os.environ["AWS_PROFILE"] = a.profile
    from lst.fleet.launch import parse_tiles

    if a.tiles_file and a.tiles:
        p.error("pass at most one of --tiles-file or --tiles")
    tiles = None
    if a.tiles_file:
        tiles = load_tiles(a.tiles_file)
    elif a.tiles:
        tiles = parse_tiles(a.tiles)
    faults = verify(a.runs, a.collection, tiles=tiles, threads=a.threads)

    if not faults:
        print("\nevery raster opened and read to its last block")
        return 0
    print(f"\n{len(faults)} tile(s) need attention:")
    for tile in sorted(faults):
        print(f"  {tile:9} {'; '.join(faults[tile])}")
    print("\nRerun them:")
    print("  " + " ".join(sorted(faults)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

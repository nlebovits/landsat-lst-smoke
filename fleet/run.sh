#!/bin/bash
# The whole pipeline for one tile, on the instance. Takes a SHA, not a branch.
#
# Every phase writes a marker, and the uploader pushes `markers.txt`
# continuously, so a watcher on another machine sees progress without SSH.
set -o pipefail
set -x
exec > >(tee -a /mnt/nvme/run/run.log) 2>&1

TILE="${1:?tile id}"
COMMIT="${2:?40-character commit sha}"
REPO=/home/ubuntu/landsat-lst-smoke
RUN=/mnt/nvme/run
ART="${ART_URI:?artifacts s3 uri}"
export PATH="$HOME/.local/bin:$PATH"

mark() { echo "MARKER $1 rc=${2:-0} $(date -u +%FT%TZ)" >> "$RUN/markers.txt"; }
die()  { mark "$1" "$2"; echo "FAILED at $1 rc=$2"; exit "$2"; }

cd "$REPO"
git fetch --all --prune --tags
git checkout --detach "$COMMIT" || die checkout $?
git rev-parse HEAD | tee "$RUN/commit.txt"
test "$(git rev-parse HEAD)" = "$COMMIT" || die commit_mismatch 1
uv sync || die uv_sync $?

# boto3 through `uv`, not the `aws` CLI. The AMI has no CLI installed, and
# adding one is a second way to do what the run's own dependencies already do.
# The artifacts prefix is public, so the download is unsigned and needs no
# credentials at all.
mkdir -p "$REPO/artifacts"
uv run python - "$ART" "$REPO/artifacts" <<'PY' || die artifacts $?
import sys, pathlib, boto3
from botocore import UNSIGNED
from botocore.config import Config

uri, dest = sys.argv[1], pathlib.Path(sys.argv[2])
bucket, _, prefix = uri.removeprefix("s3://").partition("/")
s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))
for name in ["tile_scene_inventory.parquet", "land_tiles.parquet",
             "aster_numobs.tif", "aster_numobs_manifest.json",
             "land_buffered.gpkg", "land_buffered_sha256.txt"]:
    out = dest / name
    s3.download_file(bucket, f"{prefix}/{name}", str(out))
    print("got", name, out.stat().st_size, flush=True)
PY
mark setup_done

nohup sar -o "$RUN/sar.bin" 5 > /dev/null 2>&1 &

# A liveness signal, because the phases are long and silent. Staging runs about
# 19 minutes between `prep_start` and `prep_done` and writes no marker, so a
# watcher reading markers alone cannot tell a staging tile from a wedged one.
# Appending rather than rewriting is deliberate: the uploader re-sends a file
# whose size changed, and a rewritten timestamp is the same size every time.
( while :; do
    echo "BEAT $(date -u +%FT%TZ)" >> "$RUN/heartbeat.txt"
    sleep 60
  done ) > /dev/null 2>&1 &
HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

# `--stage-dir` on the instance store, `--target-memory-gib` so a configuration
# that cannot fit stops before the fetch, `--stage-threads 128` because MEASURED
# staging runs at 321 MB/s there against 233 at the default 64.
mark prep_start
uv run tile_prep.py --tile "$TILE" \
    --stage-dir /mnt/nvme/stage \
    --out-dir "$RUN/prep" \
    --block 512 --workers 64 --threads-per-worker 1 \
    --stage-threads 128 \
    --target-memory-gib 256
RC=$?; mark prep_done $RC; [ $RC -eq 0 ] || exit $RC
test -f "$RUN/prep/tile-prep.npz" || die no_prep_artifact 1

# Four flags whose defaults are wrong for an instance. `--engine fused` is not
# the default and changes both the speed and the memory model. `--tile-prep`
# omitted composites the pooled percentile and leaves the WRS seam in a
# finished, wrong raster. `--keep-staged` lets the two passes share one fetch.
mark composite_start
uv run shard_lst_p95.py --tile "$TILE" \
    --tile-prep "$RUN/prep" \
    --engine fused --chunk 360 \
    --workers 48 --threads-per-worker 1 --memory-limit-gib 5 \
    --stage-dir /mnt/nvme/stage --keep-staged \
    --stage-threads 128 \
    --sample-interval 0.5 \
    --out-dir "$RUN/tile" \
    --catalog-dir "$RUN/tile/catalog"
RC=$?; mark composite_done $RC

sadf -d "$RUN/sar.bin" -- -u -r -b -n DEV > "$RUN/sar.txt" 2>/dev/null || true
mark all_done $RC
exit $RC

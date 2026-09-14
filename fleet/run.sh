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

mkdir -p "$REPO/artifacts"
for f in tile_scene_inventory.parquet land_tiles.parquet aster_numobs.tif \
         aster_numobs_manifest.json land_buffered.gpkg land_buffered_sha256.txt; do
  aws s3 cp --no-sign-request "$ART/$f" "$REPO/artifacts/$f" || die "artifact_$f" $?
  echo "got $f $(stat -c %s "$REPO/artifacts/$f")"
done
mark setup_done

nohup sar -o "$RUN/sar.bin" 5 > /dev/null 2>&1 &

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

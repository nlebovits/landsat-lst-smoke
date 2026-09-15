# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["boto3"]
# ///
"""Push a run's directory to object storage, continuously, while it runs.

Started before the pipeline, so a box that dies halfway still leaves its prep
artifact, its logs, and its markers behind. Earlier runs kept results only on
the instance, and terminating one destroyed them.

It writes `_MANIFEST.json` last, with `complete: true`. That flag is the only
proof a run finished uploading, and it is what `watch.py` reads.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXCLUDE = ("*.staging.tif", "*.lock", "*.tmp", "stage/*")


def wanted(rel: str, exclude=DEFAULT_EXCLUDE) -> bool:
    """Whether a path under the run directory is a result rather than scratch.

    The previous uploader excluded only `stage/`, so every run shipped
    `qa_count.staging.tif` at 4.08 GB and `lst_p95.staging.tif` at 0.68 GB.
    `composite.cleanup_staging` deletes both on the success path, so they are
    an intermediate the bucket was storing as if it were a product: about 85%
    of every run's uploaded bytes.

    `*.tmp` came from the first run that used this uploader. It polls every 30
    seconds and caught GDAL part way through building overviews, so
    `qa_count.tif.ovr.tmp` reached the bucket at 199 MB on one tile. GDAL then
    renamed it away on the instance and the copy stayed. A publish copies a
    prefix wholesale, so a temp file left here becomes a temp file in the
    public catalog.
    """
    posix = Path(rel).as_posix()
    return not any(
        fnmatch.fnmatch(posix, pat) or fnmatch.fnmatch(Path(posix).name, pat)
        for pat in exclude
    )


def scan(root: Path, exclude=DEFAULT_EXCLUDE) -> dict[str, int]:
    out = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if wanted(rel, exclude):
            out[rel] = path.stat().st_size
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True, help="runs prefix plus run id")
    p.add_argument("--profile", default=None)
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--done-marker", default="all_done")
    a = p.parse_args()

    import boto3

    s3 = boto3.Session(profile_name=a.profile).client("s3", region_name=a.region)
    sent: dict[str, int] = {}

    def push(rel: str, size: int) -> None:
        s3.upload_file(str(a.run_dir / rel), a.bucket, f"{a.prefix}/{rel}")
        sent[rel] = size
        print(f"up {rel} {size}", flush=True)

    finished = False
    while True:
        here = scan(a.run_dir)
        for rel, size in here.items():
            if sent.get(rel) != size:
                try:
                    push(rel, size)
                except Exception as err:  # noqa: BLE001  keep pushing
                    print(f"retry {rel}: {err}", flush=True)
        markers = a.run_dir / "markers.txt"
        if markers.is_file() and a.done_marker in markers.read_text():
            if finished:
                break
            # One more pass, because the last writes land after the marker.
            finished = True
        time.sleep(a.interval)

    manifest = {
        "prefix": f"s3://{a.bucket}/{a.prefix}",
        "files": sent,
        "complete": True,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(
        Bucket=a.bucket,
        Key=f"{a.prefix}/_MANIFEST.json",
        Body=json.dumps(manifest, indent=1).encode(),
    )
    print(f"complete {len(sent)} files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

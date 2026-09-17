"""Delete run prefixes and published catalogs, one named prefix at a time.

Nothing in this repository has ever deleted an S3 object. `publish_catalog.py`
prints orphans and tells the operator to remove them by hand, and
`fleet/README.md` says the same about the two divergent artifact copies. Doing
that by hand across 41 run prefixes is how a keeper gets deleted.

So this module deletes, under three rules that a hand-typed `aws s3 rm` does
not enforce.

**One prefix per call.** A single recursive delete over `runs/` with `--exclude`
patterns is one typo away from taking everything. Each run prefix is deleted on
its own, by its full name.

**A keeper is refused, not excluded.** Keepers are named on the command line
and checked against the delete list before anything runs. A keeper that appears
in the delete list stops the whole command. An exclude pattern that silently
matches nothing cannot do that.

**Nothing runs without `--yes`.** The default prints what it would delete, with
object counts and bytes, and exits.

    uv run lst-fleet-prune runs --uri s3://BUCKET/PREFIX/runs \
        --keep lst-S30W065-20260915-142626 \
        --keep lst-S50W075-20260915-142626 \
        --keep lst-S35W055-20260915-142626
"""

from __future__ import annotations

import argparse
import subprocess
import sys

#: Prefixes this command refuses to delete whatever the caller asks, because
#: every instance downloads them and losing them stops the fleet rather than
#: costing a rerun. Matched against the key, not the whole URI.
PROTECTED = ("/artifacts/",)


def s3(*args: str) -> str:
    """One `aws s3` call, with the profile the caller's environment selects."""
    out = subprocess.run(
        ["aws", "s3", *args], check=True, capture_output=True, text=True
    )
    return out.stdout


def split(uri: str) -> tuple[str, str]:
    rest = uri.removeprefix("s3://").rstrip("/")
    bucket, _, key = rest.partition("/")
    return bucket, key


def listing(uri: str) -> list[tuple[str, int]]:
    """Every object under `uri`, as `(key, size)`.

    An `aws s3 ls` line is `date time size key`. A prefix with no objects
    returns an empty list rather than raising, because deleting something
    already gone is not an error.
    """
    try:
        text = s3("ls", "--recursive", uri.rstrip("/") + "/")
    except subprocess.CalledProcessError:
        return []
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[2].isdigit():
            continue
        rows.append((fields[-1], int(fields[2])))
    return rows


def run_prefixes(runs_uri: str) -> list[str]:
    """The immediate child prefixes of a runs root, sorted.

    Reads the recursive listing rather than a delimited one, so a prefix that
    holds objects only several levels down is still found.
    """
    bucket, root = split(runs_uri)
    names = set()
    for key, _ in listing(runs_uri):
        tail = key.removeprefix(root + "/")
        head, sep, _ = tail.partition("/")
        if sep and head:
            names.add(head)
    return sorted(names)


def delete_prefix(uri: str, *, dry_run: bool) -> None:
    """Remove every object under one prefix.

    Refuses a protected prefix outright. `--dryrun` is passed to `aws` as well
    as honoured here, so a dry run cannot delete even if this function is
    called wrongly.
    """
    _, key = split(uri)
    for guard in PROTECTED:
        if guard.strip("/") in key.split("/"):
            raise ValueError(f"refusing to delete a protected prefix: {uri}")
    args = ["rm", uri.rstrip("/") + "/", "--recursive"]
    if dry_run:
        args.append("--dryrun")
    subprocess.run(["aws", "s3", *args], check=True)


def plan_runs(runs_uri: str, keep: frozenset[str]) -> tuple[list[str], list[str]]:
    """Split the run prefixes under `runs_uri` into delete and keep.

    Raises:
        ValueError: A name in `keep` is not a prefix under `runs_uri`. A
            misspelled keeper would otherwise be deleted in silence.
    """
    found = run_prefixes(runs_uri)
    missing = keep - set(found)
    if missing:
        raise ValueError(
            f"--keep names {len(missing)} prefix(es) that {runs_uri} does not "
            f"hold: {', '.join(sorted(missing))}"
        )
    doomed = [name for name in found if name not in keep]
    return doomed, sorted(keep)


def measure(uri: str) -> tuple[int, int]:
    """`(objects, bytes)` under one prefix."""
    rows = listing(uri)
    return len(rows), sum(size for _, size in rows)


def gib(n: int) -> str:
    return f"{n / (1 << 30):,.2f} GiB"


def cmd_runs(a) -> int:
    keep = frozenset(a.keep or [])
    doomed, kept = plan_runs(a.uri, keep)

    root = a.uri.rstrip("/")
    total_objects = total_bytes = 0
    print(f"runs root      {root}")
    print(f"delete         {len(doomed)} prefix(es)")
    for name in doomed:
        objects, size = measure(f"{root}/{name}")
        total_objects += objects
        total_bytes += size
        print(f"  {name:<44} {objects:>5} obj  {gib(size):>12}")
    print(f"  {'total':<44} {total_objects:>5} obj  {gib(total_bytes):>12}")
    print(f"keep           {len(kept)} prefix(es)")
    for name in kept:
        objects, size = measure(f"{root}/{name}")
        print(f"  {name:<44} {objects:>5} obj  {gib(size):>12}")

    if not a.yes:
        print("\nnothing deleted. Pass --yes to delete.")
        return 0
    for name in doomed:
        delete_prefix(f"{root}/{name}", dry_run=False)
    print(f"\ndeleted {len(doomed)} prefix(es), {total_objects} objects")
    return 0


def cmd_prefix(a) -> int:
    objects, size = measure(a.uri)
    print(f"prefix         {a.uri.rstrip('/')}")
    print(f"objects        {objects}")
    print(f"bytes          {gib(size)}")
    if not objects:
        print("\nalready empty, nothing to do.")
        return 0
    if not a.yes:
        print("\nnothing deleted. Pass --yes to delete.")
        return 0
    delete_prefix(a.uri, dry_run=False)
    print(f"\ndeleted {objects} objects")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="step", required=True)

    runs = sub.add_parser("runs", help="delete run prefixes, keeping named ones")
    runs.add_argument("--uri", required=True, help="the runs root")
    runs.add_argument(
        "--keep",
        action="append",
        metavar="PREFIX",
        help="a run prefix name to keep, repeatable",
    )
    runs.add_argument("--yes", action="store_true", help="actually delete")
    runs.set_defaults(func=cmd_runs)

    prefix = sub.add_parser("prefix", help="delete one prefix wholesale")
    prefix.add_argument("--uri", required=True)
    prefix.add_argument("--yes", action="store_true", help="actually delete")
    prefix.set_defaults(func=cmd_prefix)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

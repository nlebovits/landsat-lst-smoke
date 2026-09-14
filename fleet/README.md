# Running tiles on EC2

Everything a run needs, in one place. Until 2026-09-14 this lived in one
laptop's shell history and in an untracked directory. A later session searched
the repository, found nothing, and nearly rebuilt a launcher that already
existed.

## One run, start to finish

```bash
SHA=$(git rev-parse origin/fleet-sizing)

uv run fleet/launch.py --dry-run --tiles N40W080 --commit "$SHA"
uv run fleet/launch.py --tiles N40W080 --commit "$SHA"

fleet/drive.sh ~/.landsat-lst-run/run-<id>.json N40W080

uv run fleet/watch.py --manifest ~/.landsat-lst-run/run-<id>.json
uv run fleet/teardown.py --manifest ~/.landsat-lst-run/run-<id>.json

uv run publish_catalog.py plan --runs s3://.../runs --dest s3://.../landsat-lst
uv run publish_catalog.py copy --runs s3://.../runs --dest s3://.../landsat-lst
uv run publish_catalog.py finish --dest s3://.../landsat-lst
```

`--dry-run` prints the `run-instances` call and creates nothing. Run it first.
`AGENTS.md` also asks for a rehearsal with the real flags before any EC2
minute: `uv run shard_lst_p95.py --rehearse 6 --tile N40W080 ...`.

## The credential model

The account has two identities and they are not interchangeable.

| identity | kind | reads | expires |
|---|---|---|---|
| `radiant-earth` | SSO role | EC2, and `s3://usgs-landsat` which is requester-pays | about one hour |
| `source-coop` | static IAM user | the results bucket | no |

The role **cannot call `iam:PassRole`**, so an instance gets no instance
profile and `drive.sh` copies both identities to the box. The SSO half expires
in about an hour against a tile that takes most of one. So `drive.sh` copies
credentials immediately before it starts the run, not during setup. A run that
began 51 minutes after its credentials were minted stopped part way through
with `The provided token has expired`.

## Guard the key: ssh is the only way into a running instance

`ec2-instance-connect`, `ssm`, and the serial console are all denied. **Lose
the key and the instance is unreachable.** `fleet/launch.py` enforces both
rules in code rather than trusting them:

- The private key goes to `~/.ssh`, never a temp directory. A key under a
  session scratchpad was cleared by a workstation restart while its instance
  kept running, and nobody could reach the run again.
- The key file must be non-empty before any instance is created.

`pricing:GetProducts` and `ce:GetCostAndUsage` are also denied, so no figure
here has ever been checked against a bill.

## Terminating

Terminating is often refused for an agent by the permission layer.
`fleet/teardown.py` prints the exact command for a human when that happens.
**Do not leave it.** One instance idled for over an hour after finishing, which
wasted more than the tile's own compute cost.

The cost report runs first. AWS answers queries about a terminated instance
for only about an hour, and after that nobody can recover its lifetime.

## The artifacts

Six files, about 229 MB, that every instance downloads in region from
`nlebovits/landsat-lst-test/artifacts`. Pushing them from the laptop instead
made four concurrent instances share one uplink and quadrupled setup for no
reason.

`tile_scene_inventory.parquet`, `land_tiles.parquet`, `aster_numobs.tif`,
`aster_numobs_manifest.json`, `land_buffered.gpkg`, `land_buffered_sha256.txt`.

Copy the set together. `check_mask_inputs` and `check_manifest` refuse a run
whose artifacts disagree with each other. Rebuild them with the commands in
`FINDINGS.md`; `aster_ged.py` needs a NASA Earthdata login, once, off-instance.

> There are currently two divergent copies in the bucket,
> `landsat-lst/inputs/` and `landsat-lst-test/artifacts/`. Only the second
> carries `land_buffered_sha256.txt`, so only that one is checkable. Pick one
> and delete the other.

## Watching a run

`fleet/watch.py` polls object storage, never SSH, because the uploader pushes
`markers.txt` every interval. It names four states:

| state | meaning |
|---|---|
| `running` | markers advancing |
| `hung` | no marker for longer than the stall window |
| `failed` | a marker with a non-zero rc |
| `gone` | the instance ended before `all_done` |

`gone` exists because one instance of a four-instance fleet kept running and
billing after the other three self-terminated, and the watcher of the day
reported nothing about it for eight minutes.

Do not use `get-console-output`. It was polled eight times and returned nothing
while three of four instances had already failed.

## Where results go

The instance pushes to `runs/<name>/` while it runs. An instance that stops
part way through has already written its prep artifact, its logs and its
markers to the bucket. `_MANIFEST.json` with `complete: true` is the only
proof an upload finished.

Promotion to the published catalog at `landsat-lst/lst-p95-2021-2025/` is a
separate step, `publish_catalog.py`. It is separate because it changes a public
address and the run does not need to know about it.

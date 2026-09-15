# Running tiles on EC2

Everything a run needs, in one place. Until 2026-09-14 this lived in one
laptop's shell history and in an untracked directory. A later session searched
the repository, found nothing, and nearly rebuilt a launcher that already
existed.

## One run, start to finish

```bash
SHA=$(git rev-parse origin/fleet-sizing)

uv run lst-fleet-launch --dry-run --tiles N40W080 --commit "$SHA"
uv run lst-fleet-launch --tiles N40W080 --commit "$SHA"

fleet/drive.sh ~/.landsat-lst-run/run-<id>.json N40W080

uv run lst-fleet-watch --manifest ~/.landsat-lst-run/run-<id>.json
uv run lst-fleet-teardown --manifest ~/.landsat-lst-run/run-<id>.json

uv run lst-publish-catalog plan --runs s3://.../runs --dest s3://.../landsat-lst
uv run lst-publish-catalog copy --runs s3://.../runs --dest s3://.../landsat-lst
uv run lst-publish-catalog finish --dest s3://.../landsat-lst
```

`--dry-run` prints the `run-instances` call and creates nothing. Run it first.
`AGENTS.md` also asks for a rehearsal with the real flags before any EC2
minute: `uv run lst-shard --rehearse 6 --tile N40W080 ...`.

## Tiles, zones, and a partial launch

`--tiles` takes either spelling and any mixture of the two.

```bash
uv run lst-fleet-launch --tiles S45W075,S50W075 --commit "$SHA"
uv run lst-fleet-launch --tiles S45W075 S50W075 --commit "$SHA"
```

The launcher checks every id against the 5 degree grid before anything
launches. A malformed id, an off-grid id, and a repeated id each stop the run.
One wrong id costs a whole instance otherwise. That box launches and clones the
repository. It downloads the artifacts and fails several minutes into billing.

A launch that meets `InsufficientInstanceCapacity` tries the next availability
zone, in the order `config.toml` lists under `instance.availability_zones`. The
configured subnet goes first. Every other AWS error stops the run at the first
zone and puts the message on stderr. Retrying a wrong security group in four
zones turns one clear failure into four and reads as a full region.

The launcher writes the manifest before the first key pair exists. It rewrites
the file after each step that creates something. A key, an instance id, and an
address are all on disk as soon as they exist. Every write goes to a temporary
file and an atomic rename.

So a launch that stops on its third tile leaves the first two readable by
`watch.py` and `teardown.py`, with their key paths. The previous launcher wrote
the file once, after the last tile. A capacity refusal then left two running
instances whose ids existed only in a scrollback.

## The credential model

The account has two identities and they are not interchangeable.

| identity | kind | reads | expires |
|---|---|---|---|
| `radiant-earth` | SSO role | EC2, and `s3://usgs-landsat` which is requester-pays | about one hour |
| `source-coop` | static IAM user | the results bucket | no |

The role **cannot call `iam:PassRole`**. An instance gets no instance
profile and `drive.sh` copies both identities to the box. The SSO half expires
in about an hour against a tile that takes most of one. So `drive.sh` copies
credentials immediately before it starts the run, not during setup. A run that
began 51 minutes after `drive.sh` issued its credentials stopped part way
through with `The provided token has expired`.

## Guard the key: ssh is the only way into a running instance

`ec2-instance-connect`, `ssm`, and the serial console are all denied. **Lose
the key and the instance is unreachable.** `fleet/launch.py` enforces both
rules in code rather than trusting them:

- The private key goes to `~/.ssh`, never a temp directory. A key under a
  session scratchpad vanished when the workstation restarted, and its instance
  kept running with nobody able to reach it.
- `fleet/launch.py` checks that the key file holds bytes before it creates any
  instance.

`pricing:GetProducts` and `ce:GetCostAndUsage` are also denied, so nobody has
ever checked a figure here against a bill.

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

Do not use `get-console-output`. The watcher polled it eight times and got
nothing back while three of four instances had already failed.

## Where results go

The instance pushes to `runs/<name>/` while it runs. An instance that stops
part way through has already written its prep artifact, its logs and its
markers to the bucket. `_MANIFEST.json` with `complete: true` is the only
proof an upload finished.

Promotion to the published catalog at `landsat-lst/lst-p95-2021-2025/` is a
separate step, `publish_catalog.py`. It is separate because it changes a public
address and the run does not need to know about it.

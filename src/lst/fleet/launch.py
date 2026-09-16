"""Launch one instance per tile, at a pinned commit, and drive each one.

One command for a whole run. `--dry-run` prints the `run-instances` call it
would make and creates nothing, which is the rehearsal `AGENTS.md` asks for
before any EC2 minute.

Three rules are enforced here rather than trusted, because each one was bought
the expensive way:

- The commit is a full SHA. The previous mechanism took a branch name and ran
  `git pull --ff-only` on the instance, so the code a tile was built from
  depended on when the box happened to wake up. A published tile has to name a
  commit that means something.
- The private key lands in `~/.ssh` and is checked non-empty before any
  instance exists. A key under a session scratchpad was cleared by a
  workstation restart while its instance kept running, and this account's role
  cannot call `ec2-instance-connect`, `ssm`, or the serial console.
- Every instance carries the tags `lst-cost-report --tag` needs. An untagged
  instance cannot be priced and cannot be found by teardown.
- The manifest is rewritten after every step of every instance, not once at the
  end. A four-tile launch that lost its third tile to capacity used to exit
  before writing anything, leaving two running instances that teardown could
  not read and nobody had a key path for.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

SHA = re.compile(r"^[0-9a-f]{40}$")

#: Where the deployment assets live, relative to the repository root.
#:
#: `config.toml` and `user-data.sh` are not package data. `fleet/drive.sh`
#: copies `run.sh` and `upload.py` out of the same directory with `scp`, so all
#: five have to stay one ordinary directory that shell can address.
#:
#: This used to be `Path(__file__).parent`, which worked only while this module
#: sat beside them. An installed console script has no repository to be
#: relative to, so the directory is now a flag with a stated default rather
#: than an assumption about where the code lives.
DEFAULT_FLEET_DIR = Path("fleet")

#: The one AWS error a launch may answer by moving to another zone.
CAPACITY_ERROR = "InsufficientInstanceCapacity"

#: The errors that mean the account is already running as much as it may, not
#: that this zone is full. They are account-and-region facts, so trying another
#: zone cannot help and the launch stops at the first one.
#:
#: `servicequotas:GetServiceQuota` is denied for this role, so the concurrent
#: vCPU ceiling cannot be read before a launch. Meeting one of these is the
#: only way to discover it. A caller placing a wave catches `QuotaExhausted`,
#: keeps the instances it did place, and learns its own width from where the
#: refusal landed.
QUOTA_ERRORS = (
    "VcpuLimitExceeded",
    "InstanceLimitExceeded",
    "MaxSpotInstanceCountExceeded",
)

#: The errors that mean "you are asking too fast", not "you may not have this".
#:
#: MEASURED on 2026-09-16: placing 80 instances through 8 threads, minutes
#: after terminating 60, drew `RequestLimitExceeded` on `ec2:RunInstances`.
#: The launcher treated it as fatal, so it killed the run after 62 instances
#: and the teardown terminated all of them. About $31 for nothing.
#:
#: A throttle is the one AWS error that asks for exactly one thing: wait and
#: try again. The AWS CLI already retries twice on its own, so the wait here
#: starts above the wait it has already done.
THROTTLE_ERRORS = (
    "RequestLimitExceeded",
    "Throttling",
    "ThrottlingException",
    "RequestThrottled",
)

#: How many times one zone is asked again after a throttle, and the first wait.
#: The wait doubles each time with jitter, so six attempts span about two
#: minutes. Beyond that the tile goes back to the queue and another one tries.
THROTTLE_RETRIES = 6
THROTTLE_BACKOFF_SECONDS = 2.0


class LaunchRefused(RuntimeError):
    """AWS would not place this instance, and the run may continue without it.

    Both subclasses leave the caller in a good state. The key pair is deleted,
    the private key is removed, and the manifest entry is dropped, so the tile
    is exactly as it was before the attempt. The caller decides whether to wait,
    run narrower, or give the tile back to the queue.

    Distinct from `SystemExit`, which every other launch error still raises. A
    wrong security group or an expired token is not something a retry fixes.
    """

    def __init__(self, tile: str, code: str, detail: str):
        super().__init__(f"{tile}: {code}: {detail}")
        self.tile = tile
        self.code = code
        self.detail = detail


class QuotaExhausted(LaunchRefused):
    """The account already runs as many instances as it may.

    Account-and-region wide, so every other pending launch would meet it too.
    A caller stops asking for more and runs what it has.
    """


class Throttled(LaunchRefused):
    """AWS is asking this account to make fewer requests per second.

    Recoverable by construction. The tile goes back to the queue and the driver
    tries it again on a later poll, by which time the burst has drained.
    """


class CapacityExhausted(LaunchRefused):
    """Every configured zone is out of this instance type right now.

    A region fact, not an account fact, and a temporary one. MEASURED on
    2026-09-15: single zones refused `m6id.16xlarge` repeatedly while other
    zones placed it seconds later.

    This used to be a `SystemExit`. At width 20 it never fired. At width 60 or
    more it becomes likely, and killing a five-hour run over one tile that
    could be relaunched a minute later is the wrong trade. The tile goes back
    to the queue instead.
    """


#: The error code out of an AWS CLI failure. Matched at the position the CLI
#: prints it, not anywhere in the text: a message that merely mentions capacity
#: is not a capacity failure, and treating one as such would retry a malformed
#: request in four zones and report the region as full.
ERROR_CODE = re.compile(r"An error occurred \(([A-Za-z0-9.]+)\)")

#: The tile grid, repeated from `lst.land_tiles` rather than imported, because
#: importing it would pull geopandas into a module that otherwise needs none.
#:
#: This module now lives inside the package, so the saving is import time
#: rather than the empty dependency block it used to have. `lst.fleet.planner`
#: imports the real grid and is where a launch's tile list is validated against
#: it; this is a shape check on an operator's typing.
TILE_ID = re.compile(r"^([NS])(\d{2})([EW])(\d{3})$")
TILE_SIZE_DEGREES = 5
LATITUDE_LIMIT = 60


def fleet_asset(name: str, fleet_dir: Path | None = None) -> Path:
    """One deployment asset, or a refusal that names the flag to fix it.

    Raises:
        SystemExit: when the asset is absent. A bare `FileNotFoundError` on
            `fleet/user-data.sh` reads as a broken install; what it usually
            means is that the command ran somewhere other than the repository
            root.
    """
    path = (fleet_dir or DEFAULT_FLEET_DIR) / name
    if not path.is_file():
        msg = (
            f"no {name} at {path}. The fleet's deployment assets live in the "
            f"repository's fleet/ directory, and this command looks for them "
            f"relative to the working directory. Run it from the repository "
            f"root, or pass --fleet-dir."
        )
        raise SystemExit(msg)
    return path


def load_config(path: Path | None = None, fleet_dir: Path | None = None) -> dict:
    return tomllib.loads((path or fleet_asset("config.toml", fleet_dir)).read_text())


def on_a_remote(sha: str, repo: Path) -> bool:
    """Whether a clone of this repository would contain `sha`.

    An instance clones the repository and checks the commit out. A commit that
    exists only on this workstation is not there to check out, so `git
    checkout` exits 128 and the tile dies about four minutes into billing,
    having done nothing.

    MEASURED on 2026-09-16: 60 instances were launched at an unpushed commit.
    Every one of them wrote `MARKER checkout rc=128` and was requeued, and the
    mistake cost about $42 before the run was stopped. At the full 665 tiles it
    would have run the whole queue twice and spent most of a four-figure budget
    on `git checkout`.

    `git branch -r --contains` is the exact question: a clone fetches every
    remote branch, so a commit reachable from one is a commit the instance can
    check out. One fetch is attempted before answering no, because the local
    copy of the remote refs may be older than the push.
    """

    def contains() -> bool:
        out = subprocess.run(
            ["git", "-C", str(repo), "branch", "-r", "--contains", sha],
            capture_output=True,
            text=True,
        )
        return out.returncode == 0 and bool(out.stdout.strip())

    if contains():
        return True
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet"],
        capture_output=True,
        text=True,
    )
    return contains()


def resolve_commit(value: str, repo: Path | None = None) -> str:
    """The 40-character SHA this run pins, or an explanation of the refusal.

    A branch name is refused rather than resolved. Resolving one here would
    read this machine's idea of the branch, which is not what the instance
    would clone, and the difference is invisible in the published item.

    A SHA the remote does not carry is refused for the same reason, one step
    further on. See `on_a_remote`.
    """
    if SHA.fullmatch(value):
        if repo is not None and not on_a_remote(value, repo):
            raise SystemExit(
                f"--commit {value} is not on any remote branch.\n"
                f"Every instance clones this repository and checks that commit "
                f"out, so an unpushed commit fails on every tile and bills for "
                f"the attempt. Push it first."
            )
        return value
    hint = ""
    if repo is not None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", value],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if SHA.fullmatch(out):
                hint = f" It resolves here to {out}, which you can pass instead."
        except subprocess.CalledProcessError:
            pass
    raise SystemExit(
        f"--commit needs a full 40-character SHA, not {value!r}.{hint}\n"
        f"A branch tip moves, and a published tile has to name the code that "
        f"built it."
    )


def parse_tiles(values) -> list[str]:
    """The tiles to launch, from either spelling, checked against the grid.

    `--tiles S45W075 S50W075` and `--tiles S45W075,S50W075` mean the same
    thing, and so does any mixture of the two. The comma form is what a person
    copies out of a ticket, and argparse's `nargs="+"` turned it into one token
    that reached `run-instances` as a key name and a tag.

    A tile id is checked, not trusted. Every wrong id here costs a full
    instance: the box launches, clones, downloads the artifacts, and fails in
    `tile_bounds` or against the inventory, several minutes into billing. The
    grammar and the 5 degree grid are both cheap to check before that.

    A repeat is refused rather than collapsed. Two machines on one tile write
    the same prefix from two runs, and the second upload wins by arriving
    later. Someone who names a tile twice meant something else.

    Returns:
        The ids in the order given, upper-cased.

    Raises:
        SystemExit: on an empty list, a malformed id, an off-grid id, or a
            repeat, naming the token as it was typed.
    """
    tokens = [
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    ]
    if not tokens:
        raise SystemExit("--tiles named no tile.")

    tiles: list[str] = []
    for token in tokens:
        tile = token.upper()
        match = TILE_ID.fullmatch(tile)
        if not match:
            raise SystemExit(
                f"{token!r} is not a tile id. The form is N40W080 or S45W075: "
                f"N or S, two digits of latitude, E or W, three digits of "
                f"longitude."
            )
        ns, lat, ew, lon = match.groups()
        north = int(lat) * (1 if ns == "N" else -1)
        west = int(lon) * (1 if ew == "E" else -1)
        if north % TILE_SIZE_DEGREES or west % TILE_SIZE_DEGREES:
            raise SystemExit(
                f"{token!r} is off the {TILE_SIZE_DEGREES} degree grid. A tile "
                f"names its north edge and its west edge, and both are "
                f"multiples of {TILE_SIZE_DEGREES}."
            )
        if not -LATITUDE_LIMIT + TILE_SIZE_DEGREES <= north <= LATITUDE_LIMIT:
            raise SystemExit(
                f"{token!r} lies outside the latitude limit. North edges run "
                f"from {LATITUDE_LIMIT} down to "
                f"{-LATITUDE_LIMIT + TILE_SIZE_DEGREES}."
            )
        if not -180 <= west <= 180 - TILE_SIZE_DEGREES:
            raise SystemExit(
                f"{token!r} lies outside the grid. West edges run from -180 to "
                f"{180 - TILE_SIZE_DEGREES}, and no tile crosses the "
                f"antimeridian."
            )
        if tile in tiles:
            raise SystemExit(
                f"{token!r} is named twice. Two instances on one tile write "
                f"the same prefix and the later upload wins, so a repeat is "
                f"refused rather than collapsed."
            )
        tiles.append(tile)
    return tiles


def subnet_candidates(cfg: dict) -> list[tuple[str, str]]:
    """The subnets a launch may try, in order, each with its zone name.

    `instance.subnet` goes first, because it is the one every published tile
    ran in and the one the dry run prints. The configured zones follow in the
    order the config lists them, and the subnet already tried is not tried
    twice.

    A config with no zone list yields the one subnet, which is what the
    launcher did before the fallback existed.
    """
    inst = cfg["instance"]
    zones = list(inst.get("availability_zones", ()))
    named = {z["subnet"]: z["zone"] for z in zones}
    first = inst["subnet"]
    out = [(named.get(first, "the configured subnet"), first)]
    out.extend((z["zone"], z["subnet"]) for z in zones if z["subnet"] != first)
    return out


def error_code(text: str) -> str | None:
    """The AWS error code in a CLI failure, or None if the text has no code."""
    match = ERROR_CODE.search(text or "")
    return match.group(1) if match else None


def key_path(key_dir: str, name: str) -> Path:
    """Where the private key goes. Never a temp directory.

    `Path.expanduser` is the whole rule: a relative or temp path is refused so
    that a workstation restart cannot strand a running instance.
    """
    base = Path(key_dir).expanduser()
    if not base.is_absolute():
        raise SystemExit(f"key_dir must be absolute, not {key_dir!r}")
    for bad in ("/tmp", "/var/tmp", "/dev/shm"):
        if str(base) == bad or str(base).startswith(bad + "/"):
            raise SystemExit(
                f"key_dir {base} is cleared on reboot. A key lost while its "
                f"instance runs cannot be recovered: this role has no "
                f"ec2-instance-connect, no ssm, and no serial console."
            )
    return base / f"{name}.pem"


def tag_spec(cfg: dict, name: str, tile: str) -> str:
    """The `--tag-specifications` value, with every key teardown and costing need."""
    tags = {"Name": name, "tile": tile, **cfg["tags"]}
    pairs = ",".join(f"{{Key={k},Value={v}}}" for k, v in tags.items())
    return f"ResourceType=instance,Tags=[{pairs}]"


def render_user_data(cfg: dict, out_dir: Path, fleet_dir: Path | None = None) -> Path:
    """`user-data.sh` with the deadline substituted in.

    The deadline lives in `config.toml` because it is the only thing that
    bounds what a hung run can cost, and a number buried in a shell script is a
    number nobody revises.
    """
    text = fleet_asset("user-data.sh", fleet_dir).read_text()
    minutes = int(cfg["instance"]["deadline_minutes"])
    if "__DEADLINE__" not in text:
        raise SystemExit("user-data.sh lost its __DEADLINE__ placeholder")
    out = out_dir / "user-data.rendered.sh"
    out.write_text(text.replace("__DEADLINE__", str(minutes)))
    return out


def run_instances_argv(
    cfg: dict, name: str, tile: str, user_data: Path, subnet: str | None = None
) -> list[str]:
    """The exact call. Built in one place so `--dry-run` cannot drift from it.

    `subnet` overrides the configured one for a capacity retry. Everything else
    is identical between attempts, so a tile that lands in the fourth zone ran
    the same request as one that landed in the first.
    """
    inst, aws = cfg["instance"], cfg["aws"]
    subnet = subnet or inst["subnet"]
    return [
        "aws",
        "ec2",
        "run-instances",
        "--profile",
        aws["profile"],
        "--region",
        aws["region"],
        "--image-id",
        inst["ami"],
        "--instance-type",
        inst["type"],
        "--key-name",
        name,
        "--security-group-ids",
        inst["security_group"],
        "--subnet-id",
        subnet,
        "--associate-public-ip-address",
        "--block-device-mappings",
        f"DeviceName=/dev/sda1,Ebs={{VolumeSize={inst['root_volume_gb']},"
        f"VolumeType=gp3,DeleteOnTermination=true}}",
        "--tag-specifications",
        tag_spec(cfg, name, tile),
        "--instance-initiated-shutdown-behavior",
        "terminate",
        "--user-data",
        f"file://{user_data}",
        "--count",
        "1",
        "--query",
        "Instances[0].InstanceId",
        "--output",
        "text",
    ]


def aws_try(argv: list[str]) -> tuple[int, str, str]:
    """One AWS call that may fail. Returns `(returncode, stdout, stderr)`.

    The caller decides what a failure means. `run-instances` answers a capacity
    refusal by moving to another zone, and everything else stops the run.
    """
    out = subprocess.run(argv, capture_output=True, text=True)
    return out.returncode, out.stdout.strip(), out.stderr.strip()


def fail(argv: list[str], code: int, detail: str) -> SystemExit:
    """Put the AWS message on stderr and build the exit that carries it.

    `check=True` under `capture_output` raises a `CalledProcessError` whose
    text is the argv and nothing else, so the reason a call failed is lost
    exactly when it is needed. A dry run validates the request but not capacity
    or quota, so the message is often the only way to tell a malformed call
    from a full region. Twice on 2026-09-14 this swallowed the answer.

    The message goes to stderr here as well as into the exception, so it is
    visible under a driver that reports only the exit status.
    """
    print(detail, file=sys.stderr, flush=True)
    return SystemExit(f"aws {' '.join(argv[1:3])} failed with {code}:\n{detail}")


def aws(argv: list[str]) -> str:
    """One AWS call whose failure stops the run, with its message intact."""
    code, stdout, stderr = aws_try(argv)
    if code != 0:
        raise fail(argv, code, stderr or stdout)
    return stdout


def wait_out_throttling(argv: list[str], tile: str, say=print) -> tuple[int, str, str]:
    """One AWS call, asked again while AWS is only asking us to slow down.

    A throttle is not a refusal. It carries no information about capacity or
    quota, so moving to another zone answers a question nobody asked and
    spends another request against the same rate limit. The only useful reply
    is to wait, and each wait is longer than the last.

    Jitter matters because the threads that met the limit met it together.
    Backing off in lockstep rebuilds the burst that caused it.

    Raises:
        Throttled: when AWS is still throttling after `THROTTLE_RETRIES`.
    """
    wait = THROTTLE_BACKOFF_SECONDS
    for attempt in range(THROTTLE_RETRIES):
        code, stdout, stderr = aws_try(argv)
        detail = stderr or stdout
        if code == 0 or error_code(detail) not in THROTTLE_ERRORS:
            return code, stdout, stderr
        if attempt == THROTTLE_RETRIES - 1:
            raise Throttled(tile, error_code(detail) or "Throttling", detail)
        pause = wait * (0.5 + random.random())
        say(f"{tile}  AWS is throttling, waiting {pause:.0f}s")
        time.sleep(pause)
        wait *= 2
    raise AssertionError("unreachable")


def run_instances(cfg: dict, name: str, tile: str, user_data: Path, say=print) -> dict:
    """One instance, trying each configured zone while capacity is the reason.

    Only `InsufficientInstanceCapacity` moves the launch along. Any other error
    stops the run at the first zone with its message on stderr, because
    retrying a wrong security group or an expired token in four zones turns one
    clear failure into four and reports the region as full.

    Returns:
        `{instance_id, zone, subnet, capacity_refusals}`, where the refusals
        are the zones that had no room. The manifest keeps them, so a run that
        took an hour to place four instances says why.

    Raises:
        QuotaExhausted: when the account may not run another instance.
        CapacityExhausted: when every configured zone is out of this type.
        SystemExit: on any other error, which a retry would not fix.
    """
    refusals = []
    candidates = subnet_candidates(cfg)
    for zone, subnet in candidates:
        argv = run_instances_argv(cfg, name, tile, user_data, subnet=subnet)
        code, stdout, stderr = wait_out_throttling(argv, tile, say=say)
        if code == 0:
            return {
                "instance_id": stdout,
                "zone": zone,
                "subnet": subnet,
                "capacity_refusals": refusals,
            }
        detail = stderr or stdout
        found = error_code(detail)
        if found in QUOTA_ERRORS:
            # Account-wide, so the next zone would refuse it too.
            raise QuotaExhausted(tile, found, detail)
        if found != CAPACITY_ERROR:
            raise fail(argv, code, detail)
        refusals.append(zone)
        say(
            f"{tile}  {zone} has no {cfg['instance']['type']} capacity, "
            f"trying the next zone",
        )
    zones = ", ".join(zone for zone, _ in candidates)
    raise CapacityExhausted(
        tile,
        CAPACITY_ERROR,
        f"every configured zone refused a {cfg['instance']['type']} for "
        f"{tile}: {zones}. Nothing was launched for this tile. Instances "
        f"already launched are in the manifest.",
    )


class RunManifest:
    """The record of what exists, rewritten after every step of every instance.

    The previous launcher built the whole entry list and wrote the file once,
    after the last tile. A capacity failure on the third of four tiles exited
    before that line, so two running instances existed with their ids on stdout
    and nowhere else. `watch.py` and `teardown.py` both read the manifest, so
    neither could see them, and the keys were named in a scrollback.

    Every write is a temporary file and an `os.replace`, which is atomic on one
    filesystem. A reader either sees the previous complete manifest or the next
    one, and a launcher killed mid-write leaves neither truncated.
    """

    def __init__(self, path: Path, header: dict):
        self.path = Path(path)
        self.header = dict(header)
        self.instances: list[dict] = []
        # Reentrant, because `add` and `remove` both flush while holding it.
        #
        # The launcher used to place instances one at a time, so nothing here
        # was ever touched by two threads. Placing them in parallel makes every
        # method below a critical section twice over: the list is mutated, and
        # `write` stages through one fixed temporary path that a second writer
        # would overwrite mid-flight. Either race publishes a manifest missing
        # an instance that is running and billing.
        self.lock = threading.RLock()
        self.write()

    def add(self, entry: dict) -> dict:
        """Record one instance and flush. Returns the entry, to be mutated."""
        with self.lock:
            self.instances.append(entry)
            self.write()
        return entry

    def remove(self, entry: dict) -> None:
        """Drop an entry that never became an instance, and flush.

        A key pair reaches the manifest before `run-instances` is called, so a
        quota refusal leaves an entry naming a key and no instance. `watch.py`
        would report it `gone` forever and `teardown.py` would skip it, so the
        entry is removed once its key is deleted.
        """
        with self.lock:
            self.instances = [e for e in self.instances if e is not entry]
            self.write()

    def write(self) -> None:
        with self.lock:
            payload = dict(self.header) | {"instances": self.instances}
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            os.replace(tmp, self.path)


def delete_key_pair(cfg: dict, name: str) -> bool:
    """Remove one key pair from EC2. Returns whether the call succeeded.

    Never raises. A key pair that outlives its instance costs nothing and
    blocks nothing, so failing to delete one must not fail the caller that was
    cleaning up after something that matters.
    """
    a = cfg["aws"]
    code, _, _ = aws_try(
        [
            "aws",
            "ec2",
            "delete-key-pair",
            "--profile",
            a["profile"],
            "--region",
            a["region"],
            "--key-name",
            name,
        ]
    )
    return code == 0


def launch_one(
    cfg: dict,
    tile: str,
    run_id: str,
    dry_run: bool,
    user_data: Path,
    manifest: RunManifest | None = None,
) -> dict:
    """One key pair and one instance, recorded at every step that creates one.

    The order is the whole point. The entry reaches the manifest as soon as the
    key exists, the instance id lands in it before anything waits on the
    instance, and each later fact is another flush. A run killed at any line
    after `run-instances` leaves a manifest naming a billing instance and the
    key that opens it.
    """
    name = f"lst-{tile}-{run_id}"
    pem = key_path(cfg["paths"]["key_dir"], name)
    if dry_run:
        argv = run_instances_argv(cfg, name, tile, user_data)
        print(f"\n# {tile}: key -> {pem}")
        print(" ".join(argv))
        zones = ", ".join(zone for zone, _ in subnet_candidates(cfg))
        print(f"# {tile}: capacity fallback would try {zones}")
        return {"tile": tile, "name": name, "pem": str(pem), "dry_run": True}

    a = cfg["aws"]
    pem.parent.mkdir(parents=True, exist_ok=True)
    material = aws(
        [
            "aws",
            "ec2",
            "create-key-pair",
            "--profile",
            a["profile"],
            "--region",
            a["region"],
            "--key-name",
            name,
            "--query",
            "KeyMaterial",
            "--output",
            "text",
        ]
    )
    pem.write_text(material)
    pem.chmod(0o600)
    if pem.stat().st_size == 0:
        raise SystemExit(
            f"{pem} is empty. Refusing to launch an instance there is no way back into."
        )

    entry = {"tile": tile, "name": name, "pem": str(pem), "state": "key_created"}
    if manifest is not None:
        manifest.add(entry)

    try:
        placed = run_instances(cfg, name, tile, user_data)
    except LaunchRefused:
        # The key opens nothing. Leaving it behind accumulates one dead key
        # pair per refusal, in EC2 and in ~/.ssh, across every wave.
        delete_key_pair(cfg, name)
        pem.unlink(missing_ok=True)
        if manifest is not None:
            manifest.remove(entry)
        raise
    entry |= {
        "instance_id": placed["instance_id"],
        "zone": placed["zone"],
        "subnet": placed["subnet"],
        "capacity_refusals": placed["capacity_refusals"],
        "state": "launched",
    }
    if manifest is not None:
        manifest.write()

    aws(
        [
            "aws",
            "ec2",
            "wait",
            "instance-running",
            "--profile",
            a["profile"],
            "--region",
            a["region"],
            "--instance-ids",
            entry["instance_id"],
        ]
    )
    ip = aws(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--profile",
            a["profile"],
            "--region",
            a["region"],
            "--instance-ids",
            entry["instance_id"],
            "--query",
            "Reservations[0].Instances[0].PublicIpAddress",
            "--output",
            "text",
        ]
    )
    entry |= {"ip": ip, "state": "running"}
    if manifest is not None:
        manifest.write()
    print(
        f"{tile}  {entry['instance_id']}  {ip}  {placed['zone']}  key {pem}",
        flush=True,
    )
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tiles",
        nargs="+",
        required=True,
        help="tile ids, space separated or comma separated or both. Checked "
        "against the 5 degree grid before anything launches",
    )
    p.add_argument(
        "--commit", required=True, help="full 40-character SHA the instances check out"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the call and create nothing"
    )
    p.add_argument(
        "--fleet-dir",
        type=Path,
        default=DEFAULT_FLEET_DIR,
        help="the repository's fleet/ directory, holding config.toml and "
        "user-data.sh (default: %(default)s, relative to the working directory)",
    )
    p.add_argument(
        "--config", type=Path, default=None, help="override config.toml alone"
    )
    p.add_argument(
        "--manifest-dir", type=Path, default=Path.home() / ".landsat-lst-run"
    )
    a = p.parse_args()

    cfg = load_config(a.config, a.fleet_dir)
    tiles = parse_tiles(a.tiles)
    # The repository this command was run from, which is the one whose SHA the
    # instances check out. Resolved from the working directory rather than from
    # this module's own path, which after installation is a site-packages
    # directory and not a checkout at all.
    commit = resolve_commit(a.commit, repo=Path.cwd())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print(
        f"run {run_id}  commit {commit}  {len(tiles)} tile(s) "
        f"on {cfg['instance']['type']}"
    )
    a.manifest_dir.mkdir(parents=True, exist_ok=True)
    user_data = render_user_data(cfg, a.manifest_dir, a.fleet_dir)

    if a.dry_run:
        for tile in tiles:
            launch_one(cfg, tile, run_id, True, user_data)
        print(
            f"\nDry run. Nothing was created. Deadline would be "
            f"{cfg['instance']['deadline_minutes']} minutes per instance."
        )
        return 0

    # The manifest exists before the first key pair does, and is named now
    # rather than at the end. A launch that dies on the third tile has already
    # written the first two, and the operator has the path to read.
    path = a.manifest_dir / f"run-{run_id}.json"
    manifest = RunManifest(
        path, {"run_id": run_id, "commit": commit, "tiles": tiles, "config": cfg}
    )
    print(f"manifest {path}")
    entries = [launch_one(cfg, t, run_id, False, user_data, manifest) for t in tiles]

    print(
        f"\nnext: uv run fleet/drive.sh per tile, then "
        f"uv run lst-fleet-watch --run {run_id}"
    )
    for e in entries:
        print(f"  fleet/drive.sh {path} {e['tile']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

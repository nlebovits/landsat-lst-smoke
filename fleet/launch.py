# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = []
# ///
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
- Every instance carries the tags `cost_report.py --tag` needs. An untagged
  instance cannot be priced and cannot be found by teardown.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA = re.compile(r"^[0-9a-f]{40}$")


def load_config(path: Path | None = None) -> dict:
    return tomllib.loads((path or HERE / "config.toml").read_text())


def resolve_commit(value: str, repo: Path | None = None) -> str:
    """The 40-character SHA this run pins, or an explanation of the refusal.

    A branch name is refused rather than resolved. Resolving one here would
    read this machine's idea of the branch, which is not what the instance
    would clone, and the difference is invisible in the published item.
    """
    if SHA.fullmatch(value):
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


def render_user_data(cfg: dict, out_dir: Path) -> Path:
    """`user-data.sh` with the deadline substituted in.

    The deadline lives in `config.toml` because it is the only thing that
    bounds what a hung run can cost, and a number buried in a shell script is a
    number nobody revises.
    """
    text = (HERE / "user-data.sh").read_text()
    minutes = int(cfg["instance"]["deadline_minutes"])
    if "__DEADLINE__" not in text:
        raise SystemExit("user-data.sh lost its __DEADLINE__ placeholder")
    out = out_dir / "user-data.rendered.sh"
    out.write_text(text.replace("__DEADLINE__", str(minutes)))
    return out


def run_instances_argv(cfg: dict, name: str, tile: str, user_data: Path) -> list[str]:
    """The exact call. Built in one place so `--dry-run` cannot drift from it."""
    inst, aws = cfg["instance"], cfg["aws"]
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
        inst["subnet"],
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


def aws(argv: list[str]) -> str:
    """One AWS call, with its error message intact.

    `check=True` under `capture_output` raises a `CalledProcessError` whose
    text is the argv and nothing else, so the reason a call failed is lost
    exactly when it is needed. A dry run validates the request but not capacity
    or quota, so the message is often the only way to tell a malformed call
    from a full region. Twice on 2026-09-14 this swallowed the answer.
    """
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(
            f"aws {' '.join(argv[1:3])} failed with {out.returncode}:\n"
            f"{out.stderr.strip() or out.stdout.strip()}"
        )
    return out.stdout.strip()


def launch_one(
    cfg: dict, tile: str, run_id: str, dry_run: bool, user_data: Path
) -> dict:
    name = f"lst-{tile}-{run_id}"
    pem = key_path(cfg["paths"]["key_dir"], name)
    argv = run_instances_argv(cfg, name, tile, user_data)
    if dry_run:
        print(f"\n# {tile}: key -> {pem}")
        print(" ".join(argv))
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

    instance_id = aws(argv)
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
            instance_id,
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
            instance_id,
            "--query",
            "Reservations[0].Instances[0].PublicIpAddress",
            "--output",
            "text",
        ]
    )
    print(f"{tile}  {instance_id}  {ip}  key {pem}", flush=True)
    return {
        "tile": tile,
        "name": name,
        "pem": str(pem),
        "instance_id": instance_id,
        "ip": ip,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles", nargs="+", required=True)
    p.add_argument(
        "--commit", required=True, help="full 40-character SHA the instances check out"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the call and create nothing"
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--manifest-dir", type=Path, default=Path.home() / ".landsat-lst-run"
    )
    a = p.parse_args()

    cfg = load_config(a.config)
    commit = resolve_commit(a.commit, repo=HERE.parent)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print(
        f"run {run_id}  commit {commit}  {len(a.tiles)} tile(s) "
        f"on {cfg['instance']['type']}"
    )
    a.manifest_dir.mkdir(parents=True, exist_ok=True)
    user_data = render_user_data(cfg, a.manifest_dir)
    entries = [launch_one(cfg, t, run_id, a.dry_run, user_data) for t in a.tiles]

    if a.dry_run:
        print(
            f"\nDry run. Nothing was created. Deadline would be "
            f"{cfg['instance']['deadline_minutes']} minutes per instance."
        )
        return 0

    manifest = a.manifest_dir / f"run-{run_id}.json"
    manifest.write_text(
        json.dumps(
            {"run_id": run_id, "commit": commit, "config": cfg, "instances": entries},
            indent=1,
        )
    )
    print(f"\nmanifest {manifest}")
    print(
        f"next: uv run fleet/drive.sh per tile, then "
        f"uv run fleet/watch.py --run {run_id}"
    )
    for e in entries:
        print(f"  fleet/drive.sh {manifest} {e['tile']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

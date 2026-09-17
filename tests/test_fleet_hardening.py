"""The launcher hardening, driven entirely offline.

`tests/test_fleet_launch.py` pins the rules the mechanism already had. This
file covers what the four-tile run of 2026-09-14 cost: a capacity refusal on
the third tile exited before the manifest was written, and two running
instances existed with their ids on stdout and nowhere else. Neither `watch.py`
nor `teardown.py` reads stdout.

Nothing here reaches AWS. `launch.aws_try` is the one seam every call goes
through, so a fake in its place drives the whole mechanism, capacity fallback
and all, for nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from lst.fleet import launch
from lst.fleet import teardown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fleet"))


SHA = "9e2b703abec9756945686d5e8037788a666d28e6"

CAPACITY_STDERR = (
    "An error occurred (InsufficientInstanceCapacity) when calling the "
    "RunInstances operation (reached max retries: 2): We currently do not have "
    "sufficient m6id.16xlarge capacity in the Availability Zone you requested."
)

AUTH_STDERR = (
    "An error occurred (UnauthorizedOperation) when calling the RunInstances "
    "operation: You are not authorized to perform this operation."
)

GROUP_STDERR = (
    "An error occurred (InvalidGroup.NotFound) when calling the RunInstances "
    "operation: The security group 'sg-09717d989ae0dc8f0' does not exist."
)


@pytest.fixture(scope="module")
def cfg():
    return launch.load_config()


class FakeAws:
    """Stands in for `launch.aws_try`, scripted per subcommand.

    `run_instances` is a list read one entry per call, so a test says "the
    first zone is full and the second takes it" by writing that list. Every
    other call succeeds, which is what the real ones do once an instance
    exists.
    """

    def __init__(self, run_instances=None):
        self.run_instances = list(run_instances or [(0, "i-0abc", "")])
        self.calls: list[list[str]] = []
        self.subnets: list[str] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        action = argv[2]
        if action == "run-instances":
            self.subnets.append(argv[argv.index("--subnet-id") + 1])
            if not self.run_instances:
                return (0, f"i-{len(self.subnets):04d}", "")
            return self.run_instances.pop(0)
        if action == "create-key-pair":
            # Not a PEM header. `launch_one` only checks that the file it
            # writes is non-empty, and the real header trips the
            # detect-private-key gate on this very file.
            return (0, "not-a-real-key, for the size check alone\n", "")
        if action == "describe-instances":
            return (0, "203.0.113.7", "")
        return (0, "", "")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """A launcher whose AWS calls are scripted and whose keys land in tmp.

    `key_path` is replaced rather than pointed at `tmp_path`, because pytest's
    temporary directory is under `/tmp` and the real `key_path` refuses that
    for the reason `TestTheKeyOutlivesAReboot` records. The naming rule is kept
    so the manifest assertions still read a real key file name, and the refusal
    itself stays covered in `tests/test_fleet_launch.py`.
    """

    def build(run_instances=None):
        fake = FakeAws(run_instances)
        monkeypatch.setattr(launch, "aws_try", fake)
        keys = tmp_path / "keys"
        keys.mkdir(exist_ok=True)
        monkeypatch.setattr(
            launch, "key_path", lambda key_dir, name: Path(key_dir) / f"{name}.pem"
        )
        config = json.loads(json.dumps(launch.load_config()))
        config["paths"]["key_dir"] = str(keys)
        return fake, config

    return build


def header() -> dict:
    return {"run_id": "20260915-120000", "commit": SHA, "config": {}}


def quiet(*args, **kwargs) -> None:
    """A `say` that prints nothing, so a capacity walk does not fill the log."""


class TestTheTileListIsParsedBeforeAnythingLaunches:
    """Every wrong id costs a whole instance. The box launches, clones,
    downloads the artifacts, and fails minutes into billing.
    """

    def test_the_space_separated_form_still_works(self):
        assert launch.parse_tiles(["S45W075", "S50W075"]) == ["S45W075", "S50W075"]

    def test_the_comma_separated_form_is_accepted(self):
        """What a person copies out of a ticket. `nargs="+"` turned it into one
        token that reached run-instances as a key name and a tag."""
        assert launch.parse_tiles(["S45W075,S50W075"]) == ["S45W075", "S50W075"]

    def test_the_two_forms_mix(self):
        assert launch.parse_tiles(["S45W075,S50W075", "N40W080"]) == [
            "S45W075",
            "S50W075",
            "N40W080",
        ]

    def test_whitespace_around_a_comma_is_ignored(self):
        assert launch.parse_tiles(["S45W075, S50W075"]) == ["S45W075", "S50W075"]

    def test_a_trailing_comma_names_no_extra_tile(self):
        assert launch.parse_tiles(["S45W075,"]) == ["S45W075"]

    def test_lower_case_is_normalised(self):
        assert launch.parse_tiles(["s45w075"]) == ["S45W075"]

    def test_the_order_given_is_the_order_launched(self):
        assert launch.parse_tiles(["N40W080,S45W075"]) == ["N40W080", "S45W075"]

    @pytest.mark.parametrize(
        "bad", ["S45W75", "SW075", "45W075", "S45N075", "S45W0755", "X45W075"]
    )
    def test_a_malformed_id_is_refused(self, bad):
        with pytest.raises(SystemExit, match="is not a tile id"):
            launch.parse_tiles([bad])

    def test_the_refusal_quotes_the_token_as_typed(self):
        with pytest.raises(SystemExit) as err:
            launch.parse_tiles(["S45W75"])
        assert "'S45W75'" in str(err.value)

    @pytest.mark.parametrize("bad", ["S44W075", "S45W074", "N41W080"])
    def test_an_off_grid_id_is_refused(self, bad):
        """A tile names its north edge and its west edge, both multiples of 5."""
        with pytest.raises(SystemExit, match="off the 5 degree grid"):
            launch.parse_tiles([bad])

    @pytest.mark.parametrize("bad", ["N65W075", "S60W075"])
    def test_a_tile_outside_the_latitude_limit_is_refused(self, bad):
        with pytest.raises(SystemExit, match="latitude limit"):
            launch.parse_tiles([bad])

    def test_the_limit_itself_is_inside(self):
        """`iter_grid` runs north edges from 60 down to -55 inclusive."""
        assert launch.parse_tiles(["N60W075", "S55W075"]) == ["N60W075", "S55W075"]

    def test_a_tile_past_the_antimeridian_is_refused(self):
        with pytest.raises(SystemExit, match="antimeridian"):
            launch.parse_tiles(["N40E180"])

    def test_the_last_western_tile_is_inside(self):
        assert launch.parse_tiles(["N40E175"]) == ["N40E175"]

    def test_a_repeat_is_refused_rather_than_collapsed(self):
        """Two machines on one tile write the same prefix and the later upload
        wins, so a repeat is someone meaning something else."""
        with pytest.raises(SystemExit, match="named twice"):
            launch.parse_tiles(["S45W075,S45W075"])

    def test_a_repeat_in_a_different_case_is_still_a_repeat(self):
        with pytest.raises(SystemExit, match="named twice"):
            launch.parse_tiles(["S45W075", "s45w075"])

    @pytest.mark.parametrize("empty", [[], [""], [","], ["  "]])
    def test_an_empty_list_is_refused(self, empty):
        with pytest.raises(SystemExit, match="named no tile"):
            launch.parse_tiles(empty)

    def test_the_two_tiles_this_ticket_authorises_parse(self):
        assert launch.parse_tiles(["S45W075,S50W075"]) == ["S45W075", "S50W075"]


class TestTheCapacityFallback:
    """One zone's refusal is not the region's. The account holds one zone's
    worth of 64 vCPU quota at a time, and a zone that refuses one tile often
    takes the next.
    """

    def test_the_configured_subnet_is_tried_first(self, cfg):
        assert launch.subnet_candidates(cfg)[0][1] == cfg["instance"]["subnet"]

    def test_every_configured_zone_is_reachable(self, cfg):
        assert {s for _, s in launch.subnet_candidates(cfg)} == {
            "subnet-016d236cc30caf7de",
            "subnet-01c63ebdb73672acd",
            "subnet-07907d608259168a1",
            "subnet-0bb0dc44a35231652",
        }

    def test_no_subnet_is_tried_twice(self, cfg):
        subnets = [s for _, s in launch.subnet_candidates(cfg)]
        assert len(subnets) == len(set(subnets))

    def test_the_configured_subnet_keeps_its_zone_name(self, cfg):
        assert launch.subnet_candidates(cfg)[0][0] == "us-west-2b"

    def test_a_config_with_no_zone_list_yields_the_one_subnet(self, cfg):
        bare = json.loads(json.dumps(cfg))
        bare["instance"].pop("availability_zones")
        assert len(launch.subnet_candidates(bare)) == 1

    def test_the_zone_array_did_not_swallow_the_instance_keys(self, cfg):
        """A `[[instance.availability_zones]]` block written above the scalar
        keys of `[instance]` captures every key after it. Adding the fallback
        did exactly that, and `deadline_minutes` became a field of the fourth
        zone, so a launch would have had no deadline and no root volume."""
        for key in ("ami", "type", "subnet", "security_group"):
            assert isinstance(cfg["instance"][key], str), key
        assert isinstance(cfg["instance"]["deadline_minutes"], int)
        assert isinstance(cfg["instance"]["root_volume_gb"], int)
        for zone in cfg["instance"]["availability_zones"]:
            assert set(zone) == {"zone", "subnet"}, zone

    def test_a_full_zone_moves_the_launch_to_the_next(self, offline, tmp_path):
        _, cfg = offline([(255, "", CAPACITY_STDERR), (0, "i-0abc", "")])
        placed = launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert placed["instance_id"] == "i-0abc"
        assert placed["zone"] == "us-west-2a"
        assert placed["capacity_refusals"] == ["us-west-2b"]

    def test_it_walks_every_zone_before_giving_up(self, offline, tmp_path):
        fake, cfg = offline([(255, "", CAPACITY_STDERR)] * 4)
        with pytest.raises(
            launch.CapacityExhausted, match="every configured zone refused"
        ):
            launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert len(fake.subnets) == 4
        assert len(set(fake.subnets)) == 4

    def test_giving_up_says_nothing_was_launched_for_that_tile(self, offline, tmp_path):
        _, cfg = offline([(255, "", CAPACITY_STDERR)] * 4)
        with pytest.raises(launch.CapacityExhausted) as err:
            launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert "Nothing was launched for this tile" in str(err.value)

    def test_giving_up_does_not_end_the_run(self, offline, tmp_path):
        """A full region used to raise `SystemExit`.

        At width 20 that never fired. At width 60 or more it becomes likely,
        and one tile meeting a full region would end a five-hour run that had
        600 tiles left. The tile goes back to the queue instead.
        """
        _, cfg = offline([(255, "", CAPACITY_STDERR)] * 4)
        with pytest.raises(launch.LaunchRefused) as err:
            launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert not isinstance(err.value, SystemExit)

    def test_the_first_zone_succeeding_tries_no_others(self, offline, tmp_path):
        fake, cfg = offline([(0, "i-0abc", "")])
        launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert len(fake.subnets) == 1

    @pytest.mark.parametrize("stderr", [AUTH_STDERR, GROUP_STDERR])
    def test_another_error_stops_at_the_first_zone(self, offline, tmp_path, stderr):
        """Retrying a wrong security group in four zones turns one clear
        failure into four and reports the region as full."""
        fake, cfg = offline([(255, "", stderr), (0, "i-0abc", "")])
        with pytest.raises(SystemExit) as err:
            launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert len(fake.subnets) == 1
        assert stderr in str(err.value)

    def test_a_non_capacity_error_reaches_stderr(self, offline, tmp_path, capsys):
        _, cfg = offline([(255, "", AUTH_STDERR)])
        with pytest.raises(SystemExit):
            launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert "UnauthorizedOperation" in capsys.readouterr().err

    def test_an_error_with_no_code_is_not_a_capacity_error(self, offline, tmp_path):
        """A message the CLI format changed under is not a licence to retry."""
        fake, cfg = offline([(255, "", "connection reset by peer"), (0, "i-1", "")])
        with pytest.raises(SystemExit):
            launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        assert len(fake.subnets) == 1

    def test_the_word_alone_is_not_the_code(self):
        """A message that mentions capacity is not a capacity refusal."""
        text = "An error occurred (UnauthorizedOperation) ... instance capacity"
        assert launch.error_code(text) == "UnauthorizedOperation"

    def test_the_code_is_read_where_the_cli_prints_it(self):
        assert launch.error_code(CAPACITY_STDERR) == launch.CAPACITY_ERROR

    def test_no_code_reads_as_none(self):
        assert launch.error_code("broken pipe") is None
        assert launch.error_code("") is None

    def test_only_the_subnet_differs_between_attempts(self, offline, tmp_path):
        fake, cfg = offline([(255, "", CAPACITY_STDERR), (0, "i-0abc", "")])
        launch.run_instances(cfg, "lst-T-1", "T", tmp_path / "u.sh", say=quiet)
        first, second = (c for c in fake.calls if c[2] == "run-instances")
        for argv in (first, second):
            at = argv.index("--subnet-id")
            del argv[at : at + 2]
        assert first == second


class TestTheManifestSurvivesAPartialLaunch:
    """A capacity failure on the third of four tiles exited before the manifest
    was written. Both `watch.py` and `teardown.py` read the manifest.
    """

    def test_it_writes_before_the_first_instance_exists(self, tmp_path):
        path = tmp_path / "run.json"
        launch.RunManifest(path, header())
        assert json.loads(path.read_text())["instances"] == []

    def test_the_header_survives_every_flush(self, tmp_path):
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        manifest.add({"tile": "A"})
        assert json.loads(path.read_text())["commit"] == SHA

    def test_each_instance_lands_as_it_is_created(self, tmp_path):
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        manifest.add({"tile": "A", "instance_id": "i-1"})
        assert len(json.loads(path.read_text())["instances"]) == 1
        manifest.add({"tile": "B", "instance_id": "i-2"})
        assert len(json.loads(path.read_text())["instances"]) == 2

    def test_a_mutated_entry_is_flushed(self, tmp_path):
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        entry = manifest.add({"tile": "A", "state": "key_created"})
        entry["instance_id"] = "i-1"
        manifest.write()
        assert json.loads(path.read_text())["instances"][0]["instance_id"] == "i-1"

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        """A reader sees the previous complete manifest or the next one."""
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        manifest.add({"tile": "A"})
        assert not list(tmp_path.glob("*.tmp"))
        assert json.loads(path.read_text())

    def test_a_capacity_failure_leaves_the_earlier_instances_readable(
        self, offline, tmp_path
    ):
        """The failure this whole class exists for.

        The refused tile now leaves nothing behind. Its key pair is deleted and
        its entry is dropped, the same cleanup a quota refusal gets. An entry
        naming a key that no longer exists and an instance that never did would
        read `gone` to `watch.py` forever, and `teardown.py` would skip it.
        """
        fake, cfg = offline([(0, "i-0001", "")] + [(255, "", CAPACITY_STDERR)] * 4)
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        launch.launch_one(cfg, "N40W080", "1", False, tmp_path / "u.sh", manifest)
        with pytest.raises(launch.CapacityExhausted):
            launch.launch_one(cfg, "S45W075", "1", False, tmp_path / "u.sh", manifest)

        written = json.loads(path.read_text())["instances"]
        assert [e["tile"] for e in written] == ["N40W080"]
        assert written[0]["instance_id"] == "i-0001"
        assert written[0]["state"] == "running"

    def test_teardown_reads_a_partial_manifest(self, offline, tmp_path):
        """`teardown.py` filters on the key, so a keyed-but-unlaunched entry
        must not break the list of what to terminate."""
        _, cfg = offline([(0, "i-0001", "")] + [(255, "", CAPACITY_STDERR)] * 4)
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        launch.launch_one(cfg, "N40W080", "1", False, tmp_path / "u.sh", manifest)
        with pytest.raises(launch.CapacityExhausted):
            launch.launch_one(cfg, "S45W075", "1", False, tmp_path / "u.sh", manifest)

        run = json.loads(path.read_text())
        ids = [e["instance_id"] for e in run["instances"] if e.get("instance_id")]
        assert ids == ["i-0001"]
        assert teardown.terminate_argv(launch.load_config(), ids)[-1] == "i-0001"


class TestNoInstanceIsLeftUntracked:
    """A terminal failure after `run-instances` must leave the id on disk."""

    def test_the_id_is_recorded_before_anything_waits_on_it(
        self, offline, tmp_path, monkeypatch
    ):
        """`wait instance-running` can time out on a box that is billing."""
        fake, cfg = offline([(0, "i-0abc", "")])

        def die_on_wait(argv):
            if argv[2] == "wait":
                return (255, "", "Waiter InstanceRunning failed: Max attempts")
            return fake(argv)

        monkeypatch.setattr(launch, "aws_try", die_on_wait)
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        with pytest.raises(SystemExit):
            launch.launch_one(cfg, "N40W080", "1", False, tmp_path / "u.sh", manifest)

        written = json.loads(path.read_text())["instances"][0]
        assert written["instance_id"] == "i-0abc"
        assert written["state"] == "launched"

    def test_a_failed_describe_still_leaves_the_id(
        self, offline, tmp_path, monkeypatch
    ):
        fake, cfg = offline([(0, "i-0abc", "")])

        def die_on_describe(argv):
            if argv[2] == "describe-instances":
                return (255, "", "An error occurred (RequestExpired)")
            return fake(argv)

        monkeypatch.setattr(launch, "aws_try", die_on_describe)
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        with pytest.raises(SystemExit):
            launch.launch_one(cfg, "N40W080", "1", False, tmp_path / "u.sh", manifest)
        assert json.loads(path.read_text())["instances"][0]["instance_id"] == "i-0abc"

    def test_the_key_path_is_recorded_with_it(self, offline, tmp_path):
        """An instance whose key path is only in a scrollback is unreachable.
        This role has no instance-connect, no ssm, and no serial console."""
        _, cfg = offline([(0, "i-0abc", "")])
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        launch.launch_one(cfg, "N40W080", "1", False, tmp_path / "u.sh", manifest)
        written = json.loads(path.read_text())["instances"][0]
        assert written["pem"].endswith("lst-N40W080-1.pem")
        assert Path(written["pem"]).exists()

    def test_the_zone_it_landed_in_is_recorded(self, offline, tmp_path):
        _, cfg = offline([(255, "", CAPACITY_STDERR), (0, "i-0abc", "")])
        path = tmp_path / "run.json"
        manifest = launch.RunManifest(path, header())
        launch.launch_one(cfg, "N40W080", "1", False, tmp_path / "u.sh", manifest)
        written = json.loads(path.read_text())["instances"][0]
        assert written["zone"] == "us-west-2a"
        assert written["capacity_refusals"] == ["us-west-2b"]

    def test_a_dry_run_creates_nothing_and_records_nothing(self, offline, tmp_path):
        fake, cfg = offline()
        entry = launch.launch_one(cfg, "N40W080", "1", True, tmp_path / "u.sh")
        assert entry["dry_run"] is True
        assert fake.calls == []

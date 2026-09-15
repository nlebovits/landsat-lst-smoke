"""Every argparse help string has to survive argparse's own %-formatting.

`tile_prep.py` carried `"Staging is about 45% of a tile"` in a `help=` string.
argparse runs every help string through `%` formatting, so a bare `%` is a
format specifier. What it costs depends on the interpreter:

    3.12    `--help` raises `TypeError: %o format: an integer is required,
            not dict`. An ordinary run is unaffected.
    3.14    every invocation raises `ValueError: badly formed help string`,
            because 3.14 validates inside `add_argument` and `parse_args`
            calls that before it reads a flag.

`fleet/run.sh:67` runs `tile_prep.py` first on every instance, and its inline
block allows `<3.15`. `.python-version` pins 3.12 inside the repository, which
is why no gate saw it.

This check is static on purpose. Running a parser would only catch the fault on
an interpreter that rejects it, and which interpreter a subprocess gets here
depends on how the suite was launched: `uv` picks 3.14 for a fresh checkout
from a plain shell and 3.12 when pytest itself was started under `uv run`. A
guard that fires only under one of those is not a guard. Reading the syntax
tree fires on every interpreter, costs no subprocess, and needs no download.

`%(default)s` and `%%` are both legal and stay legal.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Directories that hold no first-party source: virtualenvs, agent worktrees,
#: and the retained measurement evidence.
SKIP_DIRS = frozenset(
    {
        ".claude",
        ".git",
        ".venv",
        "artifacts",
        "ec2-results",
        "evidence",
        "fulltile",
        "qa-parity",
        "s3-requests",
        "shard-run",
        "smoke",
        "smoke2",
        "split",
        "styles",
        "sweep",
    }
)

#: The argparse calls that take prose argparse will later %-format.
PARSER_CALLS = frozenset({"add_argument", "add_parser", "ArgumentParser"})

#: The keyword arguments argparse %-formats.
FORMATTED_KWARGS = frozenset({"help", "description", "epilog"})

#: A `%` that does not open a `%(name)s` mapping key. Escaped `%%` pairs are
#: removed before this runs. A lookahead cannot do that job on its own: in
#: `45%%`, the first `%` is excused by the `%` after it and the second is then
#: judged on the character after *it*, so the pair reads as one bare `%`.
BARE_PERCENT = re.compile(r"%(?!\()")


def has_unformattable_percent(text: str) -> bool:
    """Whether argparse's `%`-formatting of this string would raise."""
    return bool(BARE_PERCENT.search(text.replace("%%", "")))


def source_files() -> list[Path]:
    """Every first-party `.py` file, wherever in the tree it sits.

    `rglob` rather than a list of names, so this keeps finding the sources
    after they move into a package.
    """
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts)
    )


def unformattable_strings(path: Path) -> list[tuple[int, str, str]]:
    """Help strings in one file that argparse cannot format.

    Returns:
        One `(line, keyword, text)` per offending literal.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in PARSER_CALLS:
            continue
        for keyword in node.keywords:
            if keyword.arg not in FORMATTED_KWARGS:
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                # An f-string or a name. argparse still formats it, but its
                # text is not knowable here.
                continue
            if isinstance(value, str) and has_unformattable_percent(value):
                found.append((node.lineno, keyword.arg, value))
    return found


def test_the_sweep_finds_source_to_read():
    """The check is worthless if the file list is empty.

    A move that broke `source_files` would otherwise turn every parametrised
    case below into a pass over nothing.
    """
    files = source_files()
    assert len(files) > 20, f"only found {len(files)} source files under {ROOT}"


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_argparse_help_string_carries_a_bare_percent(path):
    offenders = unformattable_strings(path)
    assert not offenders, "\n".join(
        f"{path.relative_to(ROOT)}:{line} {kwarg}= has a bare % that argparse "
        f"reads as a format specifier. Write %% or reword. Text: {text!r}"
        for line, kwarg, text in offenders
    )


class TestTheCheckItself:
    """The guard has to fail on the shape it exists to catch."""

    def test_it_catches_a_bare_percent(self, tmp_path):
        target = tmp_path / "bad.py"
        target.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            'p.add_argument("--x", help="staging is about 45% of a tile")\n'
        )
        assert unformattable_strings(target) == [
            (3, "help", "staging is about 45% of a tile")
        ]

    def test_it_accepts_an_escaped_percent(self, tmp_path):
        target = tmp_path / "escaped.py"
        target.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            'p.add_argument("--x", help="staging is about 45%% of a tile")\n'
        )
        assert unformattable_strings(target) == []

    def test_it_accepts_a_mapping_key(self, tmp_path):
        """`%(default)s` is argparse's own documented substitution."""
        target = tmp_path / "mapping.py"
        target.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            'p.add_argument("--x", default=4, help="threads (default: %(default)s)")\n'
        )
        assert unformattable_strings(target) == []

    def test_a_lone_percent_after_an_escaped_pair_is_still_bare(self, tmp_path):
        """`%%%` is an escaped pair and then a bare one.

        This is the case the first version of the check got wrong, in the
        opposite direction: it read the second half of every `%%` as bare.
        """
        target = tmp_path / "triple.py"
        target.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            'p.add_argument("--x", help="a %%% b")\n'
        )
        assert [kwarg for _, kwarg, _ in unformattable_strings(target)] == ["help"]

    def test_it_reads_description_and_epilog_too(self, tmp_path):
        target = tmp_path / "both.py"
        target.write_text(
            "import argparse\n"
            'argparse.ArgumentParser(description="100% coverage", epilog="50% done")\n'
        )
        found = unformattable_strings(target)
        assert {kwarg for _, kwarg, _ in found} == {"description", "epilog"}

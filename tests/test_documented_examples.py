"""Every Python block in a published document runs against the live bucket.

A documented example is a promise. The reader pastes it and expects data. The
`AGENTS.md` decode example was published for months reading `da.rio.scales[0]`,
an attribute rioxarray has never had. Anyone who ran it got:

    AttributeError: 'RasterArray' object has no attribute 'scales'

No gate could have caught it. The blocks were generated strings, and a string
holding broken Python is still a valid string. Nothing executed them.

So these tests execute them. Each block runs in a subprocess against
`https://data.source.coop/nlebovits/landsat-lst/`, with an assertion appended
that checks what the block claims to produce. A block that raises, or that
decodes to something outside a physical temperature range, fails the build.

Marked `network`, so `uv run pytest` stays offline. CI runs them as:

    uv run pytest -m network tests/test_documented_examples.py

Scoped to this file. A bare `-m network` also collects `test_load_parity.py`,
which carries the `s3` mark as well and reads `s3://usgs-landsat`
requester-pays, and `test_inventory_parity.py`, which queries the USGS bulk
metadata service. Neither belongs in a gate on every pull request. The bucket
these blocks read is public and they issue ranged GETs against two tiles, so
this file costs wall clock and no money.

MEASURED 2026-09-17: the README block reads a 360 by 360 window and finishes
in 8 s. The AGENTS block opens lazily and its window read finishes in 16 s.
Neither downloads a whole tile.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Every document whose Python blocks have to run.
#:
#: The canonical README and its two published copies carry the same block, and
#: both published `AGENTS.md` copies carry the same block. They are listed
#: separately rather than deduplicated here, because a copy that has drifted
#: from its source is itself the failure this catches.
DOCUMENTS = (
    "README.md",
    "catalog/README.md",
    "catalog/AGENTS.md",
    "catalog/lst-p95-2021-2025/README.md",
    "catalog/lst-p95-2021-2025/AGENTS.md",
)

_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)

#: The physical range the compositor enforces, widened.
#:
#: `cog_catalog.LST_OUTPUT_MIN_C` and `LST_OUTPUT_MAX_C` are the real bounds.
#: These are deliberately looser. The point is to catch a decode that produced
#: raw DN, or Kelvin, or a scale applied twice, and each of those lands
#: thousands of degrees away. A bound that tracked the constants would instead
#: fail whenever someone tuned them.
COLDEST_C = -60.0
HOTTEST_C = 100.0

#: What each documented block has to produce, keyed by a substring that
#: identifies it. A block matching no key fails `test_every_block_is_checked`,
#: which is what stops a new example shipping unexecuted.
ASSERTIONS: dict[str, str] = {
    "rasterio.windows": """
import numpy.ma as ma
assert isinstance(lst_celsius, ma.MaskedArray), type(lst_celsius)
values = lst_celsius.compressed()
assert values.size > 0, "the documented window decoded to no data at all"
assert {coldest} <= values.min(), values.min()
assert values.max() <= {hottest}, values.max()
print("OK", values.size, round(float(values.min()), 2),
      round(float(values.max()), 2))
""",
    "rioxarray": """
import numpy as np
assert celsius.dtype.kind == "f", celsius.dtype
# A centred window, not the whole tile. The example is lazy on purpose and
# reading all 18000 by 18000 pixels would move 1.3 GB to check a range.
window = celsius.isel(band=0, y=slice(8000, 8512), x=slice(8000, 8512))
values = window.values[np.isfinite(window.values)]
assert values.size > 0, "the documented tile decoded to no data at all"
assert {coldest} <= values.min(), values.min()
assert values.max() <= {hottest}, values.max()
print("OK", values.size, round(float(values.min()), 2),
      round(float(values.max()), 2))
""",
}


def blocks_in(relative: str) -> list[str]:
    text = (ROOT / relative).read_text(encoding="utf-8")
    return [match.group(1) for match in _BLOCK.finditer(text)]


def assertion_for(block: str) -> str | None:
    for marker, tail in ASSERTIONS.items():
        if marker in block:
            return tail.format(coldest=COLDEST_C, hottest=HOTTEST_C)
    return None


def documented_blocks() -> list[tuple[str, int, str]]:
    found = []
    for relative in DOCUMENTS:
        for index, block in enumerate(blocks_in(relative)):
            found.append((relative, index, block))
    return found


CASES = documented_blocks()
IDS = [f"{relative}#{index}" for relative, index, _ in CASES]


class TestTheDocumentsCarryBlocksWeCheck:
    """Structure, checked offline. These run in the default suite."""

    def test_every_document_carries_at_least_one_block(self):
        for relative in DOCUMENTS:
            assert blocks_in(relative), f"{relative} has no Python block"

    def test_every_block_is_checked(self):
        """A new example has to come with an assertion.

        Without this, adding a block to `_agents_md` would publish code that
        no test ever runs, which is how the `rio.scales` example survived.
        """
        for relative, index, block in CASES:
            assert assertion_for(block) is not None, (
                f"{relative} block {index} matches no key in ASSERTIONS. Add "
                f"one stating what the block produces."
            )

    def test_the_published_readme_block_matches_the_canonical_one(self):
        """The three README copies carry one block, not three that drifted."""
        canonical = blocks_in("README.md")
        for relative in ("catalog/README.md", "catalog/lst-p95-2021-2025/README.md"):
            assert blocks_in(relative) == canonical

    def test_both_published_agents_blocks_match(self):
        assert blocks_in("catalog/AGENTS.md") == blocks_in(
            "catalog/lst-p95-2021-2025/AGENTS.md"
        )

    def test_every_block_parses(self):
        """Cheap, offline, and it would have caught nothing.

        `da.rio.scales[0]` parses. This only rules out the failures that never
        reach the interpreter. `test_the_documented_block_runs` is the test
        that means something.
        """
        for relative, index, block in CASES:
            ast.parse(block, filename=f"{relative}#{index}")

    def test_no_block_reads_a_relative_path(self):
        """A published block runs from anywhere or it runs nowhere.

        `catalog/AGENTS.md` is served from a bucket prefix, so a reader who
        copies a block out of it has no working directory in common with it.
        Every href in a documented block is absolute for that reason.

        Read through `ast` rather than a regular expression. The README block
        splits its URL across two adjacent string literals, which the parser
        folds into one constant and a regular expression reports as a bare
        relative path.
        """
        for relative, index, block in CASES:
            for node in ast.walk(ast.parse(block)):
                if not isinstance(node, ast.Constant):
                    continue
                text = node.value
                if isinstance(text, str) and text.endswith((".tif", ".json")):
                    assert text.startswith("https://"), (
                        f"{relative} block {index} opens {text!r}, which "
                        f"resolves against the reader's working directory"
                    )


@pytest.mark.network
@pytest.mark.timeout(300)
@pytest.mark.parametrize(("relative", "index", "block"), CASES, ids=IDS)
def test_the_documented_block_runs(relative, index, block, tmp_path):
    """Run the block verbatim, then assert what it claims to produce.

    A subprocess, not `exec`. The blocks import GDAL-backed readers and set
    global state in them, and a clean interpreter is what makes the result
    the reader's result rather than the test suite's.

    stderr is not asserted on. Both readers emit shutdown-time noise from
    GDAL that says nothing about whether the example worked.
    """
    script = tmp_path / f"block_{index}.py"
    script.write_text(block + assertion_for(block), encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
    )
    assert finished.returncode == 0, (
        f"{relative} block {index} failed:\n{finished.stdout}\n{finished.stderr}"
    )
    assert finished.stdout.startswith("OK "), finished.stdout

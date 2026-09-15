"""Exercise every repository-owned Vale rule, and the packages around them."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".vale.ini"

pytestmark = pytest.mark.skipif(
    shutil.which("vale") is None,
    reason="Vale is not installed; the prose CI job runs these tests",
)


def _words(count: int) -> str:
    return " ".join(["word"] * count) + "."


# One failing and one passing example per repository-owned rule. Add both when
# you add or change a rule, because test_rule_inventory_matches_the_cases
# fails otherwise.
CASES = {
    "Landsat-Docs.Sentence30": (_words(31), _words(30)),
    "Landsat-Mechanics.Ellipsis": ("Wait...", "Wait."),
    "Landsat-Mechanics.EmDash": ("word—word", "word — word"),
    "Landsat-Mechanics.EmDashDensity": ("a — b — c — d — e", "a — b — c — d"),
    "Landsat-Mechanics.Headings": ("# Bad Heading Here", "# Good heading"),
    "Landsat-Mechanics.Oxford": (
        "Choose red, blue and green options.",
        "Choose red, blue, and green options.",
    ),
    "Landsat-Mechanics.Quotes": ("“quoted”", '"quoted"'),
    "Landsat-Terms.Casing": ("Geotiff data.", "GeoTIFF data."),
    "Landsat-Terms.Hype": ("A seamless tool.", "A direct tool."),
    "Landsat-Voice.AffirmativeNegativeEcho": (
        "It reads the policy. It does not read the answers.",
        "Although it reads the policy, the answers remain unavailable.",
    ),
    "Landsat-Voice.ChatbotResidue": (
        "I hope this helps.",
        "The command prints the result.",
    ),
    "Landsat-Voice.ClosingTail": (
        "In conclusion, publish the files.",
        "Publish the files.",
    ),
    "Landsat-Voice.ConsequenceCadence": (
        "It is indexed, so people can search. It is open, so people can query.",
        "It is indexed, so people can search. People query the open files.",
    ),
    "Landsat-Voice.ContrastSlogan": (
        "It is not just a report, but a complete transformation.",
        "The report includes the measured results.",
    ),
    "Landsat-Voice.DramaticColon": (
        "Remember: this changes the final result.",
        "Remember that this changes the result.",
    ),
    "Landsat-Voice.Filler": ("It basically works.", "It works."),
    "Landsat-Voice.Mirrored": (
        "Data published by anyone, discoverable by everyone.",
        "Data published once and searchable from one place.",
    ),
    "Landsat-Voice.Passive": (
        "The file was written yesterday.",
        "The publisher wrote the file yesterday.",
    ),
    # max is 2. Two inline lists in a paragraph are ordinary prose, and three
    # stacked in a row are the cadence the rule exists to catch.
    "Landsat-Voice.SerialListCadence": (
        "It reads red, blue, and green files. It writes one, two, and three."
        " It logs alpha, beta, and gamma.",
        "It reads red, blue, and green files. It writes one, two, and three.",
    ),
    "Landsat-Voice.SoYouCan": (
        "It is open, so you can read it.",
        "Because it is open, you can read it.",
    ),
    "Landsat-Voice.StockTransitions": (
        "In today's landscape, catalogs matter.",
        "Catalogs make distributed data searchable.",
    ),
}

# Rules this repository turns off in .vale.ini, because every match here is a
# domain term or a literal figure. One test per rule, so a package bump that
# renames a rule fails loudly rather than switching the rule back on.
DISABLED_PACKAGE_RULES = {
    "ai-tells.FigurativeNouns": "The composite removes the seam between two WRS paths.",
    "ai-tells.FigurativeCarries": "Each row carries a scene identifier.",
    "ai-tells.FigurativeShape": "The block keeps the shape of the output array.",
    "ai-tells.FigurativePays": "Each output pixel costs 11 bytes.",
    "ai-tells.EmDashUsage": "The offset — the bulk deviation — shifts the scene.",
    "ai-tells.SemicolonUsage": "It reads the file; it writes the answer.",
    "ai-tells.FormalRegister": "The script implements the rule.",
    "ai-tells.ColonUsage": "**Note:** the guard runs before staging.",
}


def _checks(tmp_path: Path, text: str, level: str = "suggestion") -> set[str]:
    target = tmp_path / "page.md"
    target.write_text(text + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "vale",
            "--config",
            str(CONFIG),
            "--minAlertLevel",
            level,
            "--output",
            "JSON",
            target.name,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in {0, 1}, completed.stderr
    parsed = json.loads(completed.stdout or "{}")
    return {alert["Check"] for alerts in parsed.values() for alert in alerts}


def test_rule_inventory_matches_the_cases() -> None:
    rules = {
        f"{path.parent.name}.{path.stem}"
        for path in (ROOT / "styles").glob("Landsat-*/*.yml")
    }
    assert rules == set(CASES)


@pytest.mark.parametrize(("check", "examples"), CASES.items())
def test_each_rule_reports_bad_and_accepts_good(
    tmp_path: Path, check: str, examples: tuple[str, str]
) -> None:
    bad, good = examples
    assert check in _checks(tmp_path, bad)
    assert check not in _checks(tmp_path, good)


def test_google_package_is_active(tmp_path: Path) -> None:
    assert "Google.Semicolons" in _checks(tmp_path, "It reads; it writes.")


def test_microsoft_package_is_active(tmp_path: Path) -> None:
    assert "Microsoft.Wordiness" in _checks(tmp_path, "Tools utilize the cached file.")


def test_ai_tells_package_is_active(tmp_path: Path) -> None:
    found = _checks(tmp_path, "Let's dive into the paradigm shift. Here's the thing.")
    assert {check for check in found if check.startswith("ai-tells.")}


def test_the_three_anthropomorphic_rules_gate_at_error(tmp_path: Path) -> None:
    # These three are why the ai-tells pin moved to v1.37.0.
    # AnthropomorphicAdjectives does not exist before that release.
    text = (
        "The default is benign and the parser is brittle.\n\n"
        "The spec wants a retry and the release teaches the linter a new format.\n\n"
        "The guard hands the merge a clean start, and the cache pays for itself.\n"
    )
    found = _checks(tmp_path, text, level="error")
    assert "ai-tells.AnthropomorphicAdjectives" in found
    assert "ai-tells.AnthropomorphicCognition" in found
    assert "ai-tells.AnthropomorphicJustification" in found


@pytest.mark.parametrize(("check", "text"), DISABLED_PACKAGE_RULES.items())
def test_disabled_package_rules_stay_off(tmp_path: Path, check: str, text: str) -> None:
    assert check not in _checks(tmp_path, text)


def test_only_selected_readability_metrics_are_active(tmp_path: Path) -> None:
    sentence = (
        "Administrative interoperability documentation complicates implementation."
    )
    dense = " ".join([sentence] * 20)
    readability = {
        check for check in _checks(tmp_path, dense) if check.startswith("Readability.")
    }
    assert readability == {
        "Readability.AutomatedReadability",
        "Readability.FleschReadingEase",
    }


def test_passive_ignores_intransitives_and_un_adjectives(tmp_path: Path) -> None:
    # "gone" has no passive form, and "unverified" is an adjective. Both used
    # to report, which is why Passive gates at error only after the narrowing.
    for text in (
        "The saving is gone.",
        "That rate is unverified.",
        "The pixel is untouched.",
    ):
        assert "Landsat-Voice.Passive" not in _checks(tmp_path, text)


def test_filler_allows_a_concrete_not_just_contrast(tmp_path: Path) -> None:
    assert "Landsat-Voice.Filler" not in _checks(
        tmp_path, "The scanner checks content, not just paths."
    )


def test_dramatic_colon_does_not_treat_a_stage_label_as_prose(tmp_path: Path) -> None:
    assert "Landsat-Voice.DramaticColon" not in _checks(
        tmp_path,
        "**Stage 3: field matching.** Apply the matching rule.",
    )

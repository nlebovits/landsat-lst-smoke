# Prose style

Write for a technical reader in a hurry. Lead with the result. Give the
mechanism and the limits after it, so the reader can judge the number as well
as read it.

Prefer concrete verbs. Say which component acts. Address the reader as
"you" in an instruction. Keep a technical term when a plain substitute would
lose meaning.

Use sentence-case headings. Keep every example runnable and every claim
testable. State a cost, a limit, or an unknown plainly. This repository already
labels its claims `MEASURED`, `DERIVED`, and `UNKNOWN`. Keep that habit.

## Avoid formulaic prose

These rules apply to anyone writing here, including an LLM that drafts or edits
a document. Before committing generated prose, cut the conversational residue,
the unearned enthusiasm, and the repeated sentence frame.

Vary sentence length and structure. Prefer syntax that states how two ideas
relate over syntax that lists one after the other.

Avoid enumerative parataxis. It shows up as:

- repeated inline lists
- compound predicates
- balanced coordinate clauses
- affirmative-negative sentence pairs

Common frames include `It does A, B, and C`, `X does A and does B`, and
`It does A. It does not do B`. Use subordination when the ideas have a causal,
conditional, temporal, concessive, or purposive relation. Keep an inline list
for items that form a real set.

Replace the resultative frame `X, so you can Y` with the relation that makes Y
possible:

```text
Because the repository commits the fixtures, you can reproduce the check.
```

Do not repeat a grammatical frame across adjacent sentences or paragraphs. Cut
stock transitions, filler, chatbot closings, dramatic setup clauses, and
contrasts that turn two facts into a slogan. Limit the em dash rather than
reaching for it as a default transition. End on the last substantive point.

The rules identify text patterns, not authorship. A person can write formulaic
prose. An LLM can write clean prose.

## The gate

[Vale](https://vale.sh/) parses Markdown, then applies the repository-owned
rules in `styles/` and four pinned packages. CI and the `prek` commit hook both
run `--minAlertLevel=error`. Anything below that level reports without
blocking.

`styles/` holds four repository-owned directories:

| style | covers |
|---|---|
| `Landsat-Terms` | product names, and the hype words this repository refuses |
| `Landsat-Mechanics` | the em dash, quotes, headings, the Oxford comma |
| `Landsat-Voice` | passive voice, filler, chatbot residue, formulaic frames |
| `Landsat-Docs` | the 30-word sentence ceiling |

`Landsat-Voice.Passive` and `Landsat-Docs.Sentence30` gate at error. Both ran
at suggestion until now, where 163 findings accumulated across the three
documents without blocking a commit. Say which component acts, and split a
sentence that runs past 30 words.

`Landsat-Voice.Passive` matches a copula followed by a past participle. It
skips the intransitive verbs that form no passive, and a participle carrying an
`un-` prefix, because "is gone" and "is unverified" are not passive
constructions. `tests/test_prose_styles.py` guards both exclusions.

The pinned packages supply the rest. `Microsoft` and `Google` carry the
developer-documentation baseline. `Readability` reports the Automated
Readability Index and Flesch Reading Ease as suggestions, which makes a trend
visible without blocking a commit. Compare a document before and after an edit
rather than chasing a target, because a technical name raises both scores on
sound prose.

[`ai-tells`](https://github.com/tbhb/vale-ai-tells) carries 137 rules that read
the patterns the `Landsat-*` styles cannot match: figurative verbs, AI
vocabulary, hollow contrasts, and personified tools. Three of them matter most
here, and all three gate at error:

- `ai-tells.AnthropomorphicAdjectives` catches a mechanism graded for
  character. Say what the thing does or measures.
- `ai-tells.AnthropomorphicCognition` catches a tool handed a mind. State what
  the tool checks or produces.
- `ai-tells.AnthropomorphicJustification` catches the merit cliché.

Their reports look like this:

```text
AnthropomorphicAdjectives     a benign default, a brittle parser, a noisy target
AnthropomorphicCognition      the spec wants a retry, the release teaches the linter
AnthropomorphicJustification  pays for itself, pulls its weight, deserves a look
```

One repository-owned rule covers causal language that the package misses:

- `Landsat-Voice.CausalWriter` catches a condition, rule, or threshold used
  as the subject of a writer verb. State which component writes the output,
  or say that the condition causes it to write.

`.vale.ini` pins `ai-tells` at v1.37.0. `AnthropomorphicAdjectives` does not
exist before that release.

## Rules this repository turns off

`.vale.ini` disables a package rule only where every match is a domain term or
a literal figure, and every disable states its reason inline. The domain
disables:

| rule | why |
|---|---|
| `ai-tells.FigurativeNouns` | a seam here is the WRS-2 discontinuity this repository removes |
| `ai-tells.FigurativeCarries` | a row carries a column value, which is literal containment |
| `ai-tells.FigurativeShape` | every match is an array shape |
| `ai-tells.FigurativePays` | every match is a price in dollars or a size in bytes |

`tests/test_prose_styles.py` guards each one. A package bump that renames a
rule fails the suite rather than switching the rule back on.

## Running the checks

Run the blocking commit-stage gates:

```bash
uv run prek run --all-files --show-diff-on-failure
```

Run the advisory checks through their pinned hook environments. `prek` hides
output from a hook that exits zero. The Vale audit needs verbose mode to show
suggestion-level findings:

```bash
uv run prek --verbose run vale-audit \
    --all-files --hook-stage manual
uv run prek run proselint \
    --all-files --hook-stage manual
```

Both commands read every tracked document. During an edit, replace
`--all-files` with `--files path/to/doc.md`. proselint exits non-zero on an
advisory finding, and that status does not make the manual hook a gate.

[proselint](https://github.com/amperser/proselint) reports selected clichés,
hedges, redundant phrases, mixed metaphors, and commercial language.
`proselint.json` turns off the checks that need no judgment here. Its findings
stay advisory, because acting on one takes an editorial decision.

The gate covers handwritten Markdown. It skips generated `AGENTS.md` context,
the `.claude/` directory, the measurement directories such as `evidence/ec2-results/`
and `evidence/fulltile/`, and vendored trees.

## Suppress a false positive

Prefer a narrow suppression, and say why the prose has to keep its form.

```markdown
<!-- vale Landsat-Mechanics.Headings = NO -->
## An External Name That Uses Title Case
<!-- vale Landsat-Mechanics.Headings = YES -->
```

Use `<!-- vale off -->` only for a whole block that Vale cannot parse. Never
suppress a finding to make the check pass.

## Rule ownership

This repository owns the `Landsat-*` rules and their tests. They come from the
Portolan prose rules at the commit named in `styles/NOTICE`. `vale sync`
downloads the pinned packages into ignored directories.

Each custom rule has one failing and one passing example in
`tests/test_prose_styles.py`. Add both when you add or change a rule, because
`test_rule_inventory_matches_the_cases` fails otherwise.

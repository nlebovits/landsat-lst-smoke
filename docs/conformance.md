# Portolan conformance

This catalog targets Portolan v0.2.0 on STAC 1.1.0. `rashid` is the gate, and
`tests/test_cog_catalog.py` runs it over `catalog/` on every pull request.

Reproduce the check locally through the project environment, so the version
the test finds is the version you read about here:

```bash
.venv/bin/rashid check catalog/ \
  --schema --data --data-scope local
```

MEASURED 2026-09-17 with `rashid` 0.1.8 over 771 files: 0 errors, 1 warning.
The live probe against `https://data.source.coop/nlebovits/landsat-lst/`
reports the same, which covers `PTL-LIV-000` through `PTL-LIV-005`.

A `rashid` on `PATH` may be older. Version 0.1.6 carries no v0.2.0 schema and
reports `PTL-SCH-000` in place of validating, so it passes trees this gate
rejects.

## Accepted deviations

The test fails on any error-severity finding whose rule id is absent from this
table. Widening the table without an entry here fails a second test, which
parses these rule ids and compares them with the allow-list in the code.

| Rule | Where | Why | Tracking |
|---|---|---|---|
| `PTL-CAT-001` | `lst-p95-2021-2025/collection.json` | The 769 tile directories are the published URL contract, and `README.md` names one of them in its worked example. Regrouping them under subcatalogs moves every item document and both COGs: 769 x 3 objects, and roughly 400 GB of server-side copies. | [#35](https://github.com/nlebovits/landsat-lst-smoke/issues/35) |

## Findings this catalog does not carry, and why that is deliberate

Two recommendations pull against each other, and this catalog follows
Portolan rather than `stac-check`.

- **No `self` link on the root catalog.** `PORTO-CORE-081` states a SHOULD,
  and `rashid` 0.1.8 reports nothing. A static catalog can be mirrored or
  moved, and a tracked file whose correct content depends on the deployment
  target produces diff noise on every copy. `stac-check` recommends the
  opposite. Do not add one to silence it.
- **`vcs` and `issues` links on the root catalog.** Neither is a Core
  requirement and `rashid` checks neither. `build_root_catalog` in
  `src/lst/cog_catalog.py` emits both, because a reader who finds a metadata
  error needs somewhere to send the fix.

## Rules to leave alone

Four of the checks depend on decisions that are easy to undo by accident:

- `PTL-LNK-002` wants every contained object reachable through a `child` or
  `item` link. The 769 item links in `collection.json` are what satisfy it.
  `items.parquet` is an addition, never a replacement for them.
- `PTL-MIR-001` and `PTL-MIR-002` want `items.parquet` declared as a
  collection asset, typed `application/vnd.apache.parquet`, with the role
  `collection-mirror`.
- `PTL-VIZ-001` wants a thumbnail on a geospatial collection.
  `thumbnail.png` is 60 KB and it is tracked in git for that reason.
- `PTL-FIL-001` through `PTL-FIL-005` want `README.md` and `AGENTS.md` in
  every catalog and collection directory, linked with `rel:agents` and
  `rel:describedby`, with licence and provenance stated in the README. The
  repository `README.md` carries `## License` and `## Provenance`, and
  `_published_readme` copies that body into all four places.

Falsify a green run before trusting it. Delete
`catalog/lst-p95-2021-2025/AGENTS.md`, confirm that `PTL-FIL-001` and
`PTL-FIL-002` fire, then restore the file.

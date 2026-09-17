# Rules for agents working in this repository

Read this before FINDINGS.md. FINDINGS.md is the measurement history and it
records its own mistakes; the merged pull requests carry the current numbers.
When the two disagree, the pull request wins.

## The layout

The code is one installed package, `lst`, under `src/`. Read
`src/lst/__init__.py` first: it says what each of the three layers is for.

- `lst.*` is the run path. Nothing in it may import `lst.fleet` or
  `lst.measure`, and nothing an instance runs may import `lst.stac_reference`.
  `.importlinter` states both, and `uv run lint-imports` checks them.
- `fleet/` holds deployment assets and no importable code: `run.sh`,
  `drive.sh`, `user-data.sh`, `config.toml`, and `upload.py`. `upload.py` keeps
  a PEP 723 inline block because `drive.sh` copies it to an instance outside
  the checkout and runs it there. Nothing else in the repository has one.
- Dependencies are declared once, in `[project] dependencies`. An instance
  installs with `uv sync --frozen --no-dev`. Never `--only-group`: that
  installs a group instead of the project, so the console scripts vanish.
- Commands are console scripts. `uv run lst-prep`, `uv run lst-shard`,
  `uv run lst-fleet-plan`, and the rest are in `[project.scripts]`. Never
  `uv run <path>.py` for anything in the package.

  DERIVED from the package metadata, development requires Python 3.12, 3.13, or
3.14 and [uv](https://docs.astral.sh/uv/). Install the package and its development
dependencies from the repository root:

```bash
uv sync
```

Run package commands through their console scripts. Do not invoke files under
`src/lst` by path.

### Rehearse locally

This small run uses synthetic scenes, skips the production output mask, and
writes a local catalog:

```bash
uv run lst-shard \
  --bbox=-65.0,-32.5,-64.5,-32.0 \
  --rehearse 6 \
  --pixels-per-degree 120 \
  --chunk 30 \
  --workers 2 \
  --threads-per-worker 1 \
  --no-output-mask \
  --out-dir ./rehearsal
```

The command prefixes output with `REHEARSAL:`, and `summary.json` records
`"synthetic": true`. A rehearsal proves only the execution path, not real-data
output, capacity, runtime, or cost.

### Prepare production inputs

A real tile needs the complete scene inventory, ASTER GED raster, and both land
geometries. The repository commits small test slices rather than the production
artifacts.

Build the land and inventory artifacts first:

```bash
uv run lst-land-tiles \
  --out artifacts/land_tiles.parquet \
  --write-geometry artifacts/land_buffered.gpkg \
  --write-strict-geometry artifacts/land_strict.gpkg

uv run lst-inventory \
  --land-tiles artifacts/land_tiles.parquet \
  --out artifacts/tile_scene_inventory.parquet
```

The ASTER GED build needs a NASA Earthdata login:

```bash
uv run python -c "import earthaccess; earthaccess.login(persist=True)"
uv run lst-aster-ged --out artifacts/aster_numobs.tif
uv run lst-fleet-plan --out artifacts/fleet_plan.json
```

### Run one real tile

Build the seam-correction artifact before the composite. Use the same tile,
window, inventory, and stage directory for both commands:

```bash
uv run lst-prep \
  --tile S30W065 \
  --stage-dir ./stage \
  --out-dir ./tile-prep

uv run lst-shard \
  --tile S30W065 \
  --tile-prep ./tile-prep \
  --engine fused \
  --stage-dir ./stage \
  --out-dir ./composite-run
```

A full tile requires production-scale memory, storage, and requester-pays S3
access. Run `lst-prep` and `lst-shard` with `--dry-run` before allocating that
capacity. The [fleet runbook](fleet/README.md) covers EC2 launch, monitoring,
teardown, and catalog promotion.

### Check changes

```bash
uv run ruff check .
uv run ty check --extra-search-path tests --extra-search-path fleet
uv run lint-imports --no-logo
uv run pytest
vale --minAlertLevel=error README.md FINDINGS.md docs/PROSE.md
```

The package keeps three layers separate. `lst.*` holds the run path,
`lst.fleet` holds deployment operations, and `lst.measure` holds one-off
measurements. Read `src/lst/__init__.py` before changing those boundaries.

## The pipeline

The composite is a block plan and one submitted task per block
(`composite.submit_blocks`), run on a frisky cluster by `lst.shard_lst_p95`,
after `lst.tile_prep` has fitted the seam correction. Read
`lst/composite.py`'s module docstring first.

- **Two engines, and `fused` is the default.** `--engine graph` selects
  `composite.build_graph`, one lazy dask-xarray graph over the whole tile.
  Both apply every numeric rule through the same `reduce_block` and
  `finalize_block`, and `tests/test_composite_fused.py` asserts they write
  identical bytes block for block. Keep it that way. The graph is what makes
  the default checkable, so a change that breaks the comparison is a change
  that removes the only second opinion on the pixels a run publishes.
- The time axis is never chunked. `odc.stac.load` is called with
  `chunks={"time": -1, ...}`, `open_stack` asserts it, and `apply_ufunc`
  refuses a time-chunked input. A percentile over an axis may not chunk that
  axis. `DataArray.quantile` is not used anywhere: on dask it rechunks time
  silently and resets the spatial chunks.
- Task count scales with the block count, not the scene count.
  `tests/test_composite_graph.py` asserts it for the graph and
  `tests/test_block_plan.py` for the plan. If either fails, the change
  reintroduced per-scene tasks.
- One implementation per numeric rule. The kernels in `lst.lst_qa` and
  `lst.destripe` are numpy and run inside `composite.reduce_block`. Do not add
  a lazy twin of any of them.
- The driver does no per-block work before or during submission. Correction
  weights, masks, and statistics are lazy arrays or lazy reductions in the
  same graph.
- Output is the two COGs. No `.npy`, no `.npz`, no part files, no merge.
- Two land geometries, and only one of them masks. `artifacts/land_buffered.gpkg`
  is Natural Earth grown by 25 km and it decides pixels.
  `artifacts/land_strict.gpkg` is the same polygons unbuffered and it decides
  nothing: every published share of land divides by it, through
  `masks.land_split` and `masks.coverage`. MEASURED 2026-09-15 at 3600 pixels
  per degree, `S40W065` is 73,254,945 pixels of mask and 45,407,126 of land, so
  the two are not interchangeable. Never name a count `land` when it came from
  the buffered geometry.
- `qa_count` does not say whether a pixel was imaged. It counts observations
  that passed `not_fill AND qa_clear AND in_trusted_range`, so a source fill and
  a rejected observation both read zero. A scene bounding rectangle is not
  evidence either: it describes a file, and MEASURED at 26.49 S 61.64 W, 60 of
  60 source reads inside 295 overlapping footprints were fill. Splitting
  `lst:empty_land_pixels` needs a source-presence reduction in
  `composite.reduce_block`, and no arithmetic on the published rasters
  substitutes for it.

## Measuring and reporting

- Every number you report is labelled MEASURED, DERIVED, or UNKNOWN, in the
  sentence that states it. An estimate written in the voice of a measurement
  is the single most repeated mistake in this repository's history.
- A rehearsal (`--rehearse N`) proves the path, not the cost. Its output is
  tagged `REHEARSAL:` on every line and `"synthetic": true` in `summary.json`.
  Never call a rehearsal verified, passing, ready, or done in a report about
  real data.
- Before any change to the reduction, the graph, or the writer, run
  `uv run pytest tests/test_composite_graph.py tests/test_composite_run.py`.
  Before any EC2 minute, run a rehearsal on the laptop with the exact flags.

## Frisky

- `observe.enable()` is the first statement of every driver. Tracing in the
  driver process is off until then and `record_span` drops silently.
- Driver phases go through `observe.phase(name)`. A span name of your own
  never crosses to the scheduler; the client-phase event does, and the
  dashboard shows it live.
- Collect with `observe.collect(cluster, out_dir)` before `cluster.close()`.
  Nothing frisky holds survives the scheduler. Read a run afterwards with
  `frisky observe overview <out-dir>/spans.json` and `frisky observe prefixes`.
- `frisky.LocalCluster(processes=True, memory_limit=<bytes>)`. Bytes, not a
  string. `client.close()` before `cluster.close()`.

## Prose

`docs/PROSE.md` has the full rules. Before you finish any change that touches
a Markdown file, run the gate on it:

```bash
vale --minAlertLevel=error README.md FINDINGS.md docs/PROSE.md
```

Anything it reports blocks the commit and CI. The findings you will hit most:

- **Passive voice is an error.** Say which component acts. Write "the graph
  reduces the block", not "the block is reduced".
- **A sentence over 30 words is an error.** Split it.
- **`, so X does Y` is an error.** End the sentence at the comma and state the
  effect as its own claim.
- **Do not grade a mechanism's character.** No benign default, no brittle
  parser, no noisy target. Say what it does or measures.
- **Do not give a tool a mind.** A spec does not want, a release does not
  teach, a parser does not get confused. Say what it checks or produces.
- **No merit clichés.** Nothing pays for itself, pulls its weight, does the
  heavy lifting, or deserves a closer look.
- **No hype words, no filler, no chatbot closings.** The rules name them.

Never suppress a finding to make the check pass. If a rule is wrong for this
domain, disable it in `.vale.ini` with a stated reason and add a test to
`tests/test_prose_styles.py`, the way the four existing domain disables do.

## Working habits

- Run anything over 30 seconds in the background and keep talking.
- Retry a blocked command once before building a workaround.
- Read a file before editing it, and edit tracked Python through the Edit
  tool. Not through a heredoc, and not through `sed -i`. See the exception
  below, which is narrow and has conditions.
- Work in the worktree you were given. Never `git stash` on the shared stack.
- Do not add a module under `lst.measure` in the same change as a pipeline
  change. A measurement whose subject moved in the same commit is evidence
  about neither version.

### The one exception to editing Python by hand

A change that is mechanical, affects more files than anyone will read, and
fails loudly when wrong may be applied by a script. The move to `src/lst`
rewrote 172 imports across 44 files that way. Doing it by hand would not have
been safer: its failure mode is a missed site at edit 97, and a reviewer has
to trust 172 opaque operations instead of reading one script.

The conditions are the point, and all four hold or the rule stands:

1. **AST-scoped, not textual.** These module names appear in docstrings, skip
   reasons and assertion messages throughout this repository. A regex cannot
   tell an import from a sentence; `ast.Import` and `ast.ImportFrom` can.
2. **Dry run first, and read it.** The dry run for that change surfaced three
   bugs before anything was written: a rewrite that silently rebound a module
   name, a rename that shadowed a module with a local, and a per-line
   substitution that clobbered itself when one line held two references.
3. **Audit the diff mechanically.** Assert that every changed line is the kind
   of line you meant to change, and read whatever the audit flags. The Edit
   tool proves each edit matched; it does not prove the set was complete or
   that nothing else moved. This replaces that guarantee with a stronger one.
4. **Every gate green afterwards**, and the script kept where the reviewer can
   read it.

Convenience is not a condition. If the change needs judgment per site, it is
not mechanical, and it goes through the Edit tool however many sites there
are.

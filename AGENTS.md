# Rules for agents working in this repository

Read this before FINDINGS.md. FINDINGS.md is the measurement history and it
records its own mistakes; the merged pull requests carry the current numbers.
When the two disagree, the pull request wins.

## The pipeline

The composite is one lazy dask-xarray graph per tile (`composite.py`), run
on a frisky cluster by `shard_lst_p95.py`, after `tile_prep.py` has fitted
the seam correction. Read `composite.py`'s module docstring first.

- The time axis is never chunked. `odc.stac.load` is called with
  `chunks={"time": -1, ...}`, `open_stack` asserts it, and `apply_ufunc`
  refuses a time-chunked input. A percentile over an axis may not chunk that
  axis. `DataArray.quantile` is not used anywhere: on dask it rechunks time
  silently and resets the spatial chunks.
- The graph's task count scales with the block count, not the scene count.
  `tests/test_composite_graph.py` asserts it. If a change makes that test
  fail, the change reintroduced per-scene tasks.
- One implementation per numeric rule. The kernels in `lst_qa.py` and
  `destripe.py` are numpy and run inside `composite.reduce_block`. Do not add
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
  tool, never through heredocs or `sed -i`.
- Work in the worktree you were given. Never `git stash` on the shared stack.
- Do not add a `measure_*.py` script in the same change as a pipeline change.

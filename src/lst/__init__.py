"""A p95 composite of land surface temperature, from Landsat Collection 2.

Read `lst.composite`'s module docstring first. It holds the graph, and every
numeric rule the pipeline applies runs inside one of its kernels.

Three layers, and imports run one way through them:

`lst.*`
    The run path. One tile in, two COGs out. `lst.tile_prep` fits the seam
    correction, `lst.shard_lst_p95` drives the composite on a frisky cluster,
    and everything else is a library either of them reaches.

`lst.fleet`
    Planning, launching, watching, tearing down, and pricing an EC2 run, plus
    publishing finished tiles. Nothing in the run path imports it.

`lst.measure`
    One-shot measurement scripts. Each answers a question `FINDINGS.md` records
    the answer to. Nothing imports them either.

The deployment assets live outside this package, in `fleet/` at the repository
root: `run.sh`, `drive.sh`, `user-data.sh`, `config.toml`, and `upload.py`.
`upload.py` keeps its own inline dependency block on purpose. `drive.sh` copies
it to the instance outside the checkout and runs it from there, so a box whose
clone is broken still uploads what it produced.

`.importlinter` states these directions as contracts rather than as prose.
"""

"""Planning, running, watching, and pricing a fleet of EC2 instances.

These run on a workstation, never on an instance. The instance-side half is
`fleet/run.sh` and `fleet/upload.py` at the repository root, which are
deployment assets rather than importable code.

`lst.fleet.launch` reads two of those assets, `config.toml` and `user-data.sh`.
It locates them through `--fleet-dir` rather than relative to its own file: an
installed console script has no repository to be relative to, and resolving
them from the current working directory would make the result depend on where
the operator happened to be standing.

Nothing in `lst`'s run path imports this package.
"""

"""One-shot measurements. Each answers a question `FINDINGS.md` records.

These are instruments, not pipeline. They read what the pipeline wrote, or run
one fixed sample against a real bucket, and they print a number with its
provenance. Nothing in the run path imports them, and `.importlinter` says so.

`AGENTS.md` asks that a new measurement script not arrive in the same change as
a pipeline change. A measurement whose subject moved in the same commit cannot
be read as evidence about either version.
"""

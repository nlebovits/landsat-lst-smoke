"""Vulture whitelist. Empty, and that is the finding.

A whitelist names symbols vulture flags that are reached by something it
cannot see: a framework, a plugin interface, a fixture. Every name this file
carried was copied from `nlebovits/landsat-lst`, a package with Pydantic
models and Click commands, and none of them existed here. A whitelist of
absent names silences nothing and hides the next real one.

Run the gate as `prek.toml` and `.github/workflows/ci.yml` run it. At
`--min-confidence 80` it currently reports nothing over the checked modules,
so there is nothing to suppress. Add an entry here only with the reason the
symbol is reached, next to it.
"""

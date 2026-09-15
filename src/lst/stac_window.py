"""The composite's time window, and the cache identity that follows from it.

The composite is a five-year P95 over Landsat Collection 2 Level 2 scenes,
2021 through 2025. Every entry point reads the two constants below, so the
window is stated once. It cannot drift between the production run, the
measurement scripts, and the dry runs.

`DEFAULT_END` carries a time of day on purpose. A STAC `datetime` range is
closed at both ends, so a bare `2025-12-31` drops the scenes acquired that day.
The `2020-01-01/2025-01-01` window this repository used before dropped all of
2025 and included all of 2020.

A cached STAC item list is named after the query that produced it, and the
window is part of that query. A cache written for `2020-01-01/2025-01-01`
hashes to a different file than a `2021-01-01/2025-12-31T23:59:59Z` request, so
it can never satisfy one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: First instant of the composite window.
DEFAULT_START = "2021-01-01"
#: Last instant of the composite window. Closed interval, so 31 December counts.
DEFAULT_END = "2025-12-31T23:59:59Z"

DEFAULT_COLLECTION = "landsat-c2-l2"
DEFAULT_PLATFORMS = "landsat-8,landsat-9"
DEFAULT_CLOUD_COVER_LT = 100


def datetime_range(start: str = DEFAULT_START, end: str = DEFAULT_END) -> str:
    """The `datetime` argument a STAC search takes, as `start/end`."""
    return f"{start}/{end}"


def query_digest(
    bbox,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    collection: str = DEFAULT_COLLECTION,
    platforms: str = DEFAULT_PLATFORMS,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
) -> str:
    """A short stable hash of everything one STAC item list depends on.

    Two searches that agree on all six parts return the same items, so they may
    share a cache entry. Two that differ on any part may not.
    """
    parts = (
        collection,
        ",".join(f"{float(v):.6f}" for v in bbox),
        start,
        end,
        platforms,
        str(cloud_cover_lt),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def items_cache_path(
    bbox,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    cache_dir: Path | str = Path("/tmp"),
    collection: str = DEFAULT_COLLECTION,
    platforms: str = DEFAULT_PLATFORMS,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
) -> Path:
    """Where a cached STAC item list for this query belongs.

    The window appears twice: readably in the stem, and inside the digest that
    also covers the bbox, the collection, and the query filters. The stem is
    for a human reading `ls`; the digest is what makes reuse safe.
    """
    stem = f"{start}_{end}".replace(":", "").replace("-", "")
    digest = query_digest(
        bbox,
        start,
        end,
        collection=collection,
        platforms=platforms,
        cloud_cover_lt=cloud_cover_lt,
    )
    return Path(cache_dir) / f"stac_items-{stem}-{digest}.pkl"

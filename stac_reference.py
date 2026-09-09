# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["pystac-client", "pystac"]
# ///
"""Earth Search, kept as an oracle and never as a production path.

The sharded pipeline reads a precomputed inventory. This module holds the
catalogue query that inventory replaces, for three jobs and no others:

* `tests/test_inventory_parity.py` checks the inventory against it.
* `usgs_inventory` calls `fetch_items_by_id` for the handful of scenes whose
  acquisition straddles a month boundary, which the bulk file's whole-second
  timestamps cannot resolve on their own. That runs during precompute, on a
  laptop, for a few items.
* `measure_s3_requests` and the dry-run scripts use it to describe what the
  old path did.

Nothing here is reachable from `shard_lst_p95`'s production path.
`tests/test_no_stac_at_runtime.py` fails if that changes. The name says
`reference` rather than `search` so that an import of it in a runtime module
reads as a mistake.
"""

from __future__ import annotations

from stac_window import (
    DEFAULT_CLOUD_COVER_LT,
    DEFAULT_END,
    DEFAULT_PLATFORMS,
    DEFAULT_START,
    datetime_range,
)

STAC_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
STAC_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
SOURCES = {
    "earth-search": STAC_EARTH_SEARCH,
    "planetary-computer": STAC_PLANETARY_COMPUTER,
}
COLLECTION = "landsat-c2-l2"


def search_items(
    bbox,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    platforms: str = DEFAULT_PLATFORMS,
    cloud_cover_lt: int = DEFAULT_CLOUD_COVER_LT,
    source: str = "earth-search",
):
    """One STAC search. Returns the items and their bboxes.

    The query the precomputed inventory reproduces: the collection, the bbox,
    the closed datetime range, `eo:cloud_cover` below the threshold, and the
    platform list.
    """
    import pystac_client

    query: dict = {"eo:cloud_cover": {"lt": cloud_cover_lt}}
    plats = [p.strip() for p in platforms.split(",") if p.strip()]
    if plats and platforms.strip().lower() != "all":
        query["platform"] = {"in": plats}
    cat = pystac_client.Client.open(SOURCES[source])
    items = list(
        cat.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=datetime_range(start, end),
            query=query,
        ).items()
    )
    # pystac types Item.bbox as list[float] | None. Every item a bbox search
    # returns has one.
    return items, [tuple(i.bbox) for i in items]  # ty: ignore[invalid-argument-type]


def fetch_items_by_id(item_ids, *, source: str = "earth-search") -> dict:
    """Fetch named items, for a bounded lookup during precompute.

    Args:
        item_ids: STAC item ids, without the processing date.

    Returns:
        The raw item dicts, keyed by id. An id the catalogue does not hold is
        absent from the result rather than an error, so the caller decides
        what a miss means.
    """
    import pystac_client

    ids = list(item_ids)
    if not ids:
        return {}
    cat = pystac_client.Client.open(SOURCES[source])
    found = {}
    # `ids` is the only filter, so the search is cheap and needs no window.
    for batch in (ids[i : i + 50] for i in range(0, len(ids), 50)):
        for item in cat.search(collections=[COLLECTION], ids=batch).items():
            found[item.id] = item.to_dict()
    return found

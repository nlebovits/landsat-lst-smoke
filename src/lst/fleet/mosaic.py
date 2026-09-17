"""One URL that draws the whole collection in QGIS.

A published collection is one COG per tile. Opening it means adding tiles by
hand, one layer each, and a continent is a hundred of them. Nobody does that,
so nobody looks at the data.

A GDAL VRT is a small XML file that names every COG and the window each one
fills. GDAL treats it as a single raster and reads only the tiles a view
touches, so the whole collection opens as one layer and draws at any zoom
without downloading anything it does not show.

The sources are absolute HTTPS URLs under `/vsicurl/`, not relative paths and
not `/vsis3/`. A relative path breaks the moment the file is opened from
anywhere but its own directory, which is the only way anyone will open it.
`/vsis3/` asks for credentials the reader does not have. MEASURED on
2026-09-15: `https://s3.us-west-2.amazonaws.com/<bucket>/<key>` serves these
rasters unauthenticated, and `https://data.source.coop/<key>` does not.

    uv run lst-fleet-mosaic --dest s3://BUCKET/PREFIX --upload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from lst.cog_catalog import LST_FILENAME, QA_FILENAME
from lst.fleet.publish_catalog import find_tiles, s3, split

#: The host that serves this bucket without credentials.
S3_HTTPS = "https://s3.{region}.amazonaws.com/{bucket}/{key}"

#: Degrees per tile. The grid `land_tiles.py` builds, restated here because a
#: VRT needs the geotransform and nothing in the item JSON gives it cheaply.
TILE_DEGREES = 5


def https_url(bucket: str, key: str, region: str) -> str:
    return S3_HTTPS.format(region=region, bucket=bucket, key=key)


def tile_origin(tile: str) -> tuple[float, float]:
    """The `(west, north)` corner of a tile, from its name."""
    north = int(tile[1:3]) * (1 if tile[0] == "N" else -1)
    west = int(tile[4:7]) * (1 if tile[3] == "E" else -1)
    return float(west), float(north)


def build_vrt(
    sources: list[tuple[str, str]],
    *,
    pixels_per_degree: int,
    dtype: str,
    nodata: float | None,
    scale: float | None = None,
    offset: float | None = None,
) -> str:
    """A VRT over tiles of one grid. `sources` is `(tile_id, url)`.

    The grid does the arithmetic. Every tile is the same size and anchored to
    whole degrees, so the mosaic's extent and each tile's window follow from
    the tile names without opening a single raster. Opening 101 rasters over
    the network to learn what the grid already states would take minutes and
    could fail on any one of them.
    """
    if not sources:
        raise ValueError("no sources")
    corners = {tile: tile_origin(tile) for tile, _ in sources}
    west = min(w for w, _ in corners.values())
    north = max(n for _, n in corners.values())
    east = max(w for w, _ in corners.values()) + TILE_DEGREES
    south = min(n for _, n in corners.values()) - TILE_DEGREES
    side = TILE_DEGREES * pixels_per_degree
    width = int(round((east - west) * pixels_per_degree))
    height = int(round((north - south) * pixels_per_degree))
    step = 1.0 / pixels_per_degree

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, "SRS").text = "EPSG:4326"
    ET.SubElement(
        root, "GeoTransform"
    ).text = f"{west:.10f}, {step:.12f}, 0.0, {north:.10f}, 0.0, -{step:.12f}"
    band = ET.SubElement(root, "VRTRasterBand", dataType=dtype, band="1")
    if nodata is not None:
        ET.SubElement(band, "NoDataValue").text = repr(nodata)
    if scale is not None:
        ET.SubElement(band, "Scale").text = repr(scale)
    if offset is not None:
        ET.SubElement(band, "Offset").text = repr(offset)

    for tile, url in sorted(sources):
        tile_west, tile_north = corners[tile]
        col = int(round((tile_west - west) * pixels_per_degree))
        row = int(round((north - tile_north) * pixels_per_degree))
        src = ET.SubElement(band, "ComplexSource")
        ET.SubElement(src, "SourceFilename", relativeToVRT="0").text = f"/vsicurl/{url}"
        ET.SubElement(src, "SourceBand").text = "1"
        ET.SubElement(
            src,
            "SrcRect",
            xOff="0",
            yOff="0",
            xSize=str(side),
            ySize=str(side),
        )
        ET.SubElement(
            src,
            "DstRect",
            xOff=str(col),
            yOff=str(row),
            xSize=str(side),
            ySize=str(side),
        )
        if nodata is not None:
            ET.SubElement(src, "NODATA").text = repr(nodata)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def collect(dest_uri: str, collection: str, filename: str, region: str):
    """Every published tile holding `filename`, as `(tile_id, url)`.

    Lists the objects rather than reading the item JSONs. The raster is the
    thing the VRT points at, and an item that names a raster the bucket does
    not hold would put a dead URL in the mosaic.
    """
    bucket, prefix = split(dest_uri)
    base = f"{prefix}/{collection}"
    listing = s3("ls", "--recursive", f"s3://{bucket}/{base}/")
    found = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        key = fields[-1]
        if not key.endswith("/" + filename):
            continue
        tile = key.rsplit("/", 2)[-2]
        found.append((tile, https_url(bucket, key, region)))
    return sorted(set(found))


#: GDAL data type names, keyed by the numpy dtype rasterio reports.
GDAL_DTYPE = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


def read_encoding(url: str) -> dict:
    """The decoding rule, read off a published raster rather than assumed.

    An earlier version of this module carried `scale=0.01, offset=200.0` as
    literals. The real offset is -50.0, which `lst_qa.LST_OFFSET` states and
    every raster carries in its header. The VRT would have drawn every
    temperature 250 C too high, and it would have looked like data the whole
    way: plausible numbers, a plausible gradient, a plausible map.

    `cog_catalog._verify_cog` already says why this matters. A dropped or
    wrong scale leaves a DN looking like a temperature. So the mosaic asks the
    file, and if it cannot, it writes nothing.
    """
    import rasterio

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(f"/vsicurl/{url}") as s:
        dtype = s.dtypes[0]
        if dtype not in GDAL_DTYPE:
            raise SystemExit(f"{url}: unsupported dtype {dtype}")
        return {
            "dtype": GDAL_DTYPE[dtype],
            "nodata": s.nodata,
            "scale": s.scales[0],
            "offset": s.offsets[0],
            "pixels_per_degree": int(round(s.width / TILE_DEGREES)),
        }


def collect_runs(runs_uri: str, collection: str, filename: str, region: str):
    """The same list, drawn from the runs prefix before anything is published.

    `find_tiles` already knows that a tile which ran twice appears under two
    run prefixes, and that the newest object wins. Reusing it keeps one rule
    for which run is current, rather than a second one here that could pick
    differently and draw a mosaic the catalog does not match.
    """
    found = find_tiles(runs_uri, collection)
    bucket, _ = split(runs_uri)
    out = []
    for tile, prefix in found.items():
        _, key = split(f"{prefix}/{filename}")
        out.append((tile, https_url(bucket, key, region)))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", required=True, help="published catalog root")
    p.add_argument(
        "--runs",
        help="build the mosaic from the runs prefix instead of the published "
        "one, so a collection can be looked at before it is published",
    )
    p.add_argument("--collection", default="lst-p95-2021-2025")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument(
        "--upload",
        action="store_true",
        help="put the VRTs beside the collection, so one URL opens the mosaic",
    )
    a = p.parse_args(argv)

    plans = [(LST_FILENAME, "lst_p95.vrt"), (QA_FILENAME, "qa_count.vrt")]
    written = []
    for filename, vrt_name in plans:
        if a.runs:
            sources = collect_runs(a.runs, a.collection, filename, a.region)
        else:
            sources = collect(a.dest, a.collection, filename, a.region)
        if not sources:
            print(f"no {filename} under {a.dest}", file=sys.stderr)
            return 1
        enc = read_encoding(sources[0][1])
        print(
            f"{filename}: {enc['dtype']} nodata={enc['nodata']} "
            f"scale={enc['scale']} offset={enc['offset']} "
            f"{enc['pixels_per_degree']} px/degree"
        )
        text = build_vrt(
            sources,
            pixels_per_degree=enc["pixels_per_degree"],
            dtype=enc["dtype"],
            nodata=enc["nodata"],
            scale=enc["scale"],
            offset=enc["offset"],
        )
        path = a.out_dir / vrt_name
        path.write_text(text)
        written.append((path, len(sources)))
        print(f"{path}  {len(sources)} tile(s)")

    if a.upload:
        base = f"{a.dest.rstrip('/')}/{a.collection}"
        for path, _ in written:
            s3("cp", str(path), f"{base}/{path.name}", capture=False)
        bucket, _ = split(a.dest)
        _, prefix = split(f"{base}")
        print("\nOpen in QGIS, one layer, whole collection:")
        for path, _ in written:
            print(f"  {https_url(bucket, f'{prefix}/{path.name}', a.region)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The one-URL mosaic, and the decoding rule it must never invent.

A VRT that names the wrong scale or offset draws a map that looks right. The
gradient is plausible, the coastlines are in the right place, and every number
is wrong. An earlier version of this module carried `offset=200.0` as a
literal against a real offset of -50.0, which would have published a mosaic
reading 250 C too high.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest
from lst.fleet import mosaic
from lst.lst_qa import LST_OFFSET, LST_SCALE


class TestTheGridPlacesEveryTile:
    """The tile names carry the geometry, so no raster has to be opened."""

    def test_a_tile_names_its_north_west_corner(self):
        assert mosaic.tile_origin("S30W065") == (-65.0, -30.0)
        assert mosaic.tile_origin("N40E010") == (10.0, 40.0)
        assert mosaic.tile_origin("N00W035") == (-35.0, 0.0)

    def test_two_tiles_side_by_side_share_an_edge(self):
        text = mosaic.build_vrt(
            [("S30W065", "https://h/a.tif"), ("S30W060", "https://h/b.tif")],
            pixels_per_degree=100,
            dtype="UInt16",
            nodata=0.0,
        )
        root = ET.fromstring(text)
        assert root.attrib["rasterXSize"] == "1000"
        assert root.attrib["rasterYSize"] == "500"
        rects = {}
        for src in root.iter("ComplexSource"):
            name, dst = src.find("SourceFilename"), src.find("DstRect")
            assert name is not None and name.text and dst is not None
            rects[name.text] = dst.attrib
        west = rects["/vsicurl/https://h/a.tif"]
        east = rects["/vsicurl/https://h/b.tif"]
        assert west["xOff"] == "0" and east["xOff"] == "500"
        assert west["yOff"] == east["yOff"] == "0"

    def test_the_geotransform_starts_at_the_north_west_corner(self):
        text = mosaic.build_vrt(
            [("S30W065", "https://h/a.tif")],
            pixels_per_degree=3600,
            dtype="UInt16",
            nodata=0.0,
        )
        node = ET.fromstring(text).find("GeoTransform")
        assert node is not None and node.text
        gt = [float(v) for v in node.text.split(",")]
        assert gt[0] == -65.0
        assert gt[3] == -30.0
        assert gt[5] < 0  # north up

    def test_no_sources_is_refused(self):
        with pytest.raises(ValueError):
            mosaic.build_vrt([], pixels_per_degree=100, dtype="UInt16", nodata=0.0)


class TestTheDecodingRuleIsCarried:
    """DN times scale plus offset. Getting it wrong publishes a wrong map."""

    def test_scale_and_offset_reach_the_vrt(self):
        text = mosaic.build_vrt(
            [("S30W065", "https://h/a.tif")],
            pixels_per_degree=100,
            dtype="UInt16",
            nodata=0.0,
            scale=LST_SCALE,
            offset=LST_OFFSET,
        )
        root = ET.fromstring(text)
        scale, offset = root.find(".//Scale"), root.find(".//Offset")
        assert scale is not None and scale.text
        assert offset is not None and offset.text
        assert float(scale.text) == LST_SCALE
        assert float(offset.text) == LST_OFFSET

    def test_the_published_offset_is_negative_fifty_not_two_hundred(self):
        """The literal this module used to carry, named so it cannot come back."""
        assert LST_OFFSET == -50.0

    def test_a_band_with_no_scale_carries_none(self):
        """qa_count is a count. A scale on it would be an invention."""
        text = mosaic.build_vrt(
            [("S30W065", "https://h/a.tif")],
            pixels_per_degree=100,
            dtype="Byte",
            nodata=None,
        )
        root = ET.fromstring(text)
        assert root.find(".//Scale") is None
        assert root.find(".//Offset") is None
        assert root.find(".//NoDataValue") is None


class TestTheUrlsWork:
    """A relative path breaks anywhere but its own directory."""

    def test_every_source_is_an_absolute_vsicurl_url(self):
        text = mosaic.build_vrt(
            [("S30W065", "https://h/a.tif")],
            pixels_per_degree=100,
            dtype="UInt16",
            nodata=0.0,
        )
        src = ET.fromstring(text).find(".//SourceFilename")
        assert src is not None and src.text
        assert src.attrib["relativeToVRT"] == "0"
        assert src.text.startswith("/vsicurl/https://")

    def test_the_host_is_the_one_that_serves_without_credentials(self):
        """MEASURED: data.source.coop refuses these; the S3 host serves them."""
        url = mosaic.https_url("us-west-2.opendata.source.coop", "a/b.tif", "us-west-2")
        assert url == (
            "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/a/b.tif"
        )


class TestAnUnknownDtypeStops:
    def test_an_unsupported_dtype_is_refused(self):
        assert "complex64" not in mosaic.GDAL_DTYPE

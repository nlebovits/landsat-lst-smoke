# landsat-lst-smoke

## Known issues

### Hot pixels left by ASTER GED coverage gaps

Where ASTER GED caught no clear sky between 2000 and 2008, USGS interpolates
emissivity from the neighbouring cells and retrieves a temperature anyway, and
some of those retrievals fail upward. `masks.py` drops a pixel only where two
things are true together: its GED cell reports zero observations or lies one
cell from such a cell, and the pixel reads 70 C or hotter.

A gap cell records where ASTER missed the ground. It says nothing about the
retrieval that consumed the interpolated emissivity, so the geometry alone
removes far more than it should.
MEASURED on S30W065: 524 of the 605 gap cells have no pixel at or above 70 C.
In the 81 that do, the hot pixels are 4.77% of the cell. Masking the geometry
alone removes 701,839 valid pixels to remove 4,588 bad ones. The pair removes
5,432 and reaches more of the tail.

503 hot pixels survive on that tile, in cells with observations. The 70 C
threshold is a screen calibrated on one tile with no published source, so it
makes no claim about the hottest land surface, and a pixel above it outside a
gap is kept. `nlebovits/landsat-lst` applied the geometry alone, measured
2,799,286 pixels removed for 2,582 artifacts, and replaced it with this pair.

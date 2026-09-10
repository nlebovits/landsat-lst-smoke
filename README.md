# landsat-lst-smoke

## Known issues

### Hot pixels left by ASTER GED coverage gaps

Where ASTER GED holds no clear-sky observation, Collection 2 has no emissivity
and some surface temperature retrievals fail upward, so `masks.py` removes
every pixel whose GED cell reports zero observations. MEASURED on S30W065: that
rule takes 77.30% of the pixels at or above 70 C for 0.2167% of the valid ones,
and 1,347 hot pixels survive it. Widening the rule by one GED cell would raise
the first figure to 91.52% and remove 2.1 million ordinary pixels with it, so
the mask stops at the cell itself and those pixels stay in the output.

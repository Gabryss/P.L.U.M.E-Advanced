# Reference morphology data

PLUME uses two complementary public sources. They answer different questions
and must not be pooled as though they were the same measurement.

## Pyroduct Digital Catalog v2

The [Pyroduct Digital Catalog v2](https://doi.org/10.5281/zenodo.17750755)
contains more than 1,200 digitized cross-sections from more than 94 terrestrial
lava tubes. It constrains passage-section width, height, area and shape. It does
not provide enough planform information to calibrate branching or room spacing.

The repository freezes 76 caves for calibration and 19 caves for final
confirmation in `paper/splits/`. Never tune generator parameters from the
evaluation list. Download the v2 archive and extract the TXT collection below
`data/reference/pdc_v2/extracted/`; public reference archives are gitignored.

## Valentine Cave LiDAR

The [USGS NASA TubeX Valentine Cave data release](https://doi.org/10.5066/P14AC3J5)
contains georeferenced Riegl VZ-400 point clouds. The lightweight analysis uses
`Valentine_TUBE_UTM_10cm.copc.laz` (about 7.6 MB). The full-resolution cave
cloud is available from the same release but is not required for network-scale
measurements.

Valentine is an independent case study for localized widening, split/rejoin
geometry, and longitudinal heterogeneity. The PCA-aligned spans emitted by the
analysis script are multi-route envelopes. Where parallel passages overlap in
plan view they are not equivalent to a single PDC cross-section.

Run the analysis without generating a cave mesh:

```bash
uv sync --extra paper
.venv/bin/python scripts/analyze_lava_tube_references.py
```

Artifacts are written to `outputs/reference_morphology/`:

- PDC calibration section distributions;
- cave-wise longitudinal autocorrelation and adjacent-change diagrams;
- Valentine planform and longitudinal width/height envelope;
- an explicitly labelled PDC-versus-Valentine scale comparison;
- CSV tables and a machine-readable summary.

The PDC loader uses numeric station ordering (`1, 2, ..., 10`) rather than
lexical filename ordering (`1, 10, 2`), which is required before computing any
longitudinal statistic.

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

## Current calibration observations

The analysis intentionally computes tuning statistics from the 76 calibration
caves only. Of 1,558 candidate contours, 1,286 are valid simple sections. Their
width is 5.52 m at the median, 18.39 m at the 95th percentile, and 26.31 m at
the maximum. The median width/height ratio is 1.80. Within individual caves,
the median largest adjacent-section width ratio is 2.70, confirming that real
tubes can change size much faster than a single globally smoothed profile.

The 10 cm Valentine cloud contains 436,439 points. Its PCA-aligned planform
envelope is about 221.8 m long and 24.6 m wide. One-metre longitudinal bins
have a median envelope width of 11.6 m and a 95th percentile of 22.8 m, while
the median height envelope is 3.16 m. These are multi-route envelopes rather
than true orthogonal passage sections, but they support a useful design
constraint: exceptional rooms should be broad and comparatively low, not
scaled-up round tunnels.

## Generator interpretation

Stage B selects one to three seeded `drained_lava_pool` sites from high-flux,
low-grade locations with a slope break, confinement loss, or route
coalescence. It never places one on a grade-separated underpass. Each selected
room stores its cause, centre, outlet width, length, width and bounded depth;
the occupancy diagnostic paints its flow-aligned elliptical footprint.

Stage C converts that metadata into a smooth longitudinal widening with a
flatter, quieter floor and a modest arched roof. The ordinary-passage cap stays
at 10 m; only explicitly labelled rooms may use the Earth room endmember of
28 m. Stage D consumes the same metadata to form an elongated, seeded lobate
volume instead of a spherical chamber.

`drained_pool_max_width_m` is deliberately independent of the ordinary
`chamber_radius`, so increasing the empirical room endmember does not inflate
every structural junction in the network.

Generate a review package without invoking Stage D or the rest of the mesh
pipeline:

```bash
.venv/bin/python scripts/generate_network_diagnostics.py
```

The package includes Stage-B topology and emplacement plots, Stage-C section
gradients, a pool-specific four-panel figure, a calibration-only PDC dashboard,
and a seeded `network_density` sweep. Its `summary.json` explicitly records the
Stage A–C-only scope.

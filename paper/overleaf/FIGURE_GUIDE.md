# Figures already included

All ten figures are bundled and referenced directly by `main.tex`. Upload the corrected ZIP and recompile; there is nothing to download or place manually for this version.

| Figure | File in the project | Content | Evidence status |
|---|---|---|---|
| 1 | `figures/f01_pipeline.pdf` | Staged PLUME data flow and export architecture | Original vector schematic |
| 2 | `figures/f02_host_fields.png` | Terrain elevation, cover, competence, routing cost | Saved Earth inspection scenario, seed 4; reconstructed host fields |
| 3 | `figures/f03_network.png` | Network birth phases and longest-segment floor/roof profile | Saved Earth inspection scenario, seed 4 |
| 4 | `figures/f04_sections.png` | Six transverse section envelopes | Selected Stage-C samples from Earth seed 4 |
| 5 | `figures/f05_roof_screen.png` | Roof-screen geometry and gravity sensitivity | Analytical model illustration with fixed material properties |
| 6 | `figures/f06_events.png` | Event locations, volume section, actual interior rock render, measured mesh sizes | Current Earth seed-4 base with structural events and 649 Rocky prop meshes |
| 7 | `figures/f07_floor_atlas.png` | Final floor elevation, clearance, procedural labels, intrinsic atlas bands | Same current seed-4 structural-event illustration as Figure 6 |
| 8 | `figures/f08_surface_relief.png` | Matched before/after views of geometric surface relief | Saved isolated-reach development study |
| 9 | `figures/f09_morphometry.png` | Six reference/generated descriptor distributions | Measured: 100 Earth worlds, 19 evaluation caves, 200 reference contours |
| 10 | `figures/f10_scalability.png` | Time and memory for all 40 resource attempts | Measured: 35 complete, five dense 5 km memory-limit failures |

## Source and caption notes

Figures 2–4 were regenerated from the saved seed-4 scenario with their overall titles and subtitles removed; panel labels and axes remain. Their source images are also updated in the repository's `docs/figures/readme/` directory. Figures 5 and 8 retain their previous images. The documentation manifest is included as `source_snapshot/readme_figure_provenance.json`. Figures 6–7 replace the legacy Earth documentation diagnostics with newly computed current-geometry results. Exact sources and asset hashes are recorded in `provenance.json`; the old `figure_candidates/` images are reference material only.

Figures 2–4 illustrate one saved scenario. They are not parameter interventions, ensemble statistics, or final-mesh validation. The cross-section panels use metric coordinates, but their displayed limits vary; the longitudinal profile uses unequal horizontal and vertical scaling.

Figure 5 varies gravity while holding density, tensile strength, and safety factor fixed. It illustrates the stated beam-screen equation, not observed planetary cave-size distributions or a full preset comparison.

Figures 6–7 use the current Earth seed-4 base at 0.20 m voxel spacing. Optional collapse, choke, infill, rock, and boulder events are enabled. Rocky generates 641 rock meshes and eight boulder meshes, with the configured population multiplier of 1.0. One choke and one infill modifier are applied; the collapse candidate is rejected by the volume safeguards. The base atlas has 635 cells; relifting retains 630 across 25 segment bands and invalidates five. The contour panel samples actual before/after volumes at infill event 386. Event IDs change when the added props are sorted into the complete event list. The original no-event inspection GLB is unchanged.

The interior panel renders the exported cave and rock nodes, with ochre used to distinguish props from the gray cave; these are diagnostic colors, not material textures. The size histogram measures each prop's maximum world-X/Y mesh extent. All 649 rock meshes have finite vertices, valid face indices, and closed surfaces. Grounding records describe placement on the base floor; these checks are not a final collision or traversability certificate.

These counts describe one development scenario, not a confirmatory campaign. Atlas cells are filtered by a 1 m minimum overhead clearance; separate prop meshes are not subtracted from that density-based clearance. Breakdown labels retain collapse-influence metadata despite that volume modifier being rejected. Saved reports and figure data are bundled in `source_snapshot/events_rocks_seed4/`; the preceding no-rock snapshot is retained separately as historical context.

To replot Figures 6–7 in the installed PLUME repository, run:

```bash
python scripts/generate_paper_event_figures.py \
  --output paper/overleaf/source_snapshot/events_rocks_seed4 --render-only
```

This reads the saved figure data without generating geometry. Full recomputation needs the original base checkpoint and matching configuration; the snapshot records their hashes but does not bundle the large checkpoint. See the snapshot README for the generation command and scope.

Figure 8 shows a controlled development comparison of the same isolated passage with matching view and rendering settings. Its artificial closed ends and separate provenance are identified in the caption. It is not a natural cave survey or a confirmatory experiment.

## Measured campaign figures

Figures 9 and 10 come from the dated evaluation campaign. Figure 9 uses saved
section descriptors, including the measured reference contours, with exact
cumulative mass at each displayed ECDF support point. Up to 4,000 support points
are displayed per curve; all records contribute to the table statistics.
Advanced and ellipse curves coincide for width, height and aspect ratio by
construction. Figure 10 includes failed resource measurements as crosses at the
cost when stopped. They are not completed-mesh timing observations.

The source and image hashes are recorded with the campaign, along with the
original package freeze and the separately documented control-evaluator
amendment. These figures do not validate final-mesh cross-sections or application
imports. Standard mesh benchmarks use 0.6 m voxels, distinct from the 0.2 m
Figures 2–4 and 6–7 inspection scenario.

## Evidence still outside this campaign

- A spacing sweep at a common error threshold, with dedicated Stage-C timings,
  is needed for a sampling-efficiency claim.
- Final-mesh section cuts and convergence tests are needed to assess narrow
  passage fidelity and the causes of disconnected components.
- Actual application imports, poses, materials and dynamic contacts need separate
  verification; emitted packages are not equivalent evidence.
- Matched planetary preset comparisons need declared material and formation
  assumptions. The Earth results do not establish measured planetary interiors.

The external PDC source dataset is not required to compile this package. Its
version and source-content identity are preserved in campaign records. Figures
use the existing cave-level partition; whole-catalog exploration before that
partition is disclosed in the manuscript.

## Replacing an included image later

Replace the file at its existing path and recompile, keeping the caption consistent with the new image. If changing file type or name, update the corresponding `\includegraphics` path in `main.tex`. Prefer vector PDF for new plots and high-resolution PNG for renders. Recheck readability at the final printed size and update source/seed/version information.

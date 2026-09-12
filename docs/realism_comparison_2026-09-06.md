# Lava-tube realism comparison — 6 September 2026

> Historical report: the measurements and implementation observations below describe the dated run, not the current release. Some output folders were subsequently removed. Unavailable artifacts are identified by their former paths; source links point to the maintained files, whose line numbers have changed. See [current reliability and verification procedures](reliability.md).

The project has an appropriate staged architecture for the stated goal: physical host → formation network → longitudinal cross-sections → mesh. Its current output is a plausible procedural cave, but it is not yet a validated predictor of lava-tube morphology across celestial bodies. The most valuable next work is in the physical conditioning and geometric representation of passages. A small meshing check suggests that replacing marching cubes would not address the principal discrepancies found here.

This assessment combines inspection of the current source, fresh Stage A–C generation, a bounded Stage D check, and published observations. No generation parameters or production code were changed for this comparison. Earlier working-tree fixes remain in place.

**What was compared.** The user’s [Valentine Cave Sketchfab model](https://sketchfab.com/3d-models/valentine-cave-lava-tube-5cm-8e8cd77139b54f3d8912c00881af7214) is a point cloud: the page reports zero triangles, about 1.1 million vertices, and 5 cm point spacing. Its colors represent laser intensity rather than natural surface color; the author describes the floor as smooth and the roof as rough. Intensity alone should not be treated as a calibrated roughness measurement.

Quantitative Valentine measurements here use the separate local 10 cm COPC product from the [USGS NASA TubeX Valentine LiDAR release](https://www.usgs.gov/data/nasa-tubex-valentine-cave-2018-valentine-lidar), not the downloadable Sketchfab asset. The survey releases differ in date and resolution; identical acquisition or coverage is not assumed. The local cloud has 436,439 points and a 221.84 m span along its principal horizontal axis. That is a projected scan extent, not surveyed passage length or the complete cave’s length.

The cross-section reference is the [Pyroduct Digital Catalog v2](https://zenodo.org/records/17750755). Only the repository’s calibration partition was used: 1,286 valid contours from 76 caves. The 19 reserved evaluation caves were not used. Digitized survey sections have uneven station spacing and unequal representation among caves, so pooled counts do not establish the frequency of shapes in nature.

**Fresh current result.** Using `config/project.toml`, seed 1, Earth, standard quality and the configured development extent produced 57 segments, 4,975.65 m of total network length, 11 graph loops, three recorded vertical levels, two drained pools and 848 section samples. This is a larger, densely branched scenario; it is not a reconstruction of Valentine Cave. Its eight sources and network density of 3 are scenario settings, not measurements of Valentine.

The table measures the actual closed Stage C contours using the same morphometry function as the reference analysis. It does not substitute the nominal height control for the contour’s measured height.

| Measurement | Current Earth run | PDC calibration |
|---|---:|---:|
| Median width | 7.46 m | 5.52 m |
| Width, 5th–95th percentiles | 4.61–12.70 m | 1.80–18.39 m |
| Median height | 4.37 m | 3.05 m |
| Height, 5th–95th percentiles | 2.99–7.11 m | 0.84–14.55 m |
| Median width/height | 1.65 | 1.80 |
| Smallest width | 4.40 m | 0.35 m |
| Smallest height | 2.62 m | 0.29 m |

The current run has larger typical passages but a narrower range of sizes. About 35.0% of calibration contours are narrower than its smallest passage, and 44.2% are lower than its lowest contour. These are descriptive comparisons with this one run, not a general failure rate or a calibrated realism score. The source explains part of the restriction: the default minimum width is 44% of the body’s ordinary-passage cap, and the minimum height control is 30% of that cap. These are useful scale controls, but they exclude small branches and crawlways before any meshing occurs. See [config.py](../src/plume_advanced/config.py).

Historical figure: Measured contour distributions and a same-scale geometric example (unavailable local artifact: `outputs/realism_comparison/comparison.png`)

The bottom panels are illustrative, not a matched cross-section error test. The Valentine slice is a one-metre-thick plane perpendicular to a global principal axis, while the generated contour is locally normal to a passage. The selected scan slice visibly contains two distinct passages. Their combined span includes intervening rock. Its approximately 25 m envelope must not become the target width of one empty tunnel.

The widest generated contour is 27.96 m wide and 7.28 m high; its nominal height control is 8.31 m. Valentine’s sampled one-metre envelopes have a median height of 3.16 m and a 95th percentile of 3.88 m. These measurements suggest that broad, low spaces deserve more representation, but cannot establish a universal maximum room height. Scan coverage, plane orientation, separated passages and roof changes all matter.

**What Valentine reveals beyond width and height.** The [USGS cave survey, reproduced in Bulletin 1673](https://npshistory.com/publications/geology/bul/1673/sec3.htm), describes pillars dividing and reuniting flow, small distributaries, floor cascades, benches, lavacicles and compound pools. Particularly significant is a floor that drops while the roof continues above it, and a pooled area whose roof height differs substantially across adjacent parts. This supports independent floor and roof evolution and preservation of solid remnants inside compound junctions. It does not support simply enlarging every cross-section around one centerline.

| Aspect | Current support | Remaining limitation |
|---|---|---|
| Formation network | Directed flow conservation, breakouts, reoccupation, split/rejoin paths, captures and stacked routes | Process rates and branch opportunities are largely heuristic; the reference catalogue cannot validate topology |
| Local widening | Explicit drained-pool metadata and elongated room geometry | Room width must distinguish individual voids from compound envelopes containing pillars |
| Sections | Correlated shape changes, skew, separate upper/lower curves, floor and roof controls | The common parametric family does not explicitly construct preserved lava-level benches, overhangs or multiple boundary loops within a compound section |
| Vertical morphology | Buried centerlines, levels, clearance rules and junction blending | Floor drops, independent roof continuity, erosion and deposition are not recovered from an evolving physical floor/roof state |
| Surfaces | Shape relief, spatial roughness, texture displacement and rock assets | Detailed surface families are not conditioned on formation history; floor roughness has an inappropriate bias for the Valentine example |
| Physical host | A downstream `HostField` object carries terrain and structural/process fields | The default path synthesizes 2D proxy fields; there is no standard physical DEM/stratigraphy import workflow with units, uncertainty and missing-data validation |

The `benched` morphology label should not be mistaken for an explicit bench model. It is selected from shape biases, and the final section still comes from the common upper/lower profile function. A separate bench ledge tied to a former lava level would add geological meaning beyond that label. See [section_field.py](../src/plume_advanced/stages/section_field.py) and its [profile construction](../src/plume_advanced/stages/section_field.py).

There is also a concrete surface-control discrepancy: Stage D multiplies roughness by `1 + 0.55 * floor_weight`, adding up to 55% more amplitude toward the floor. This is a term-level bias, not a measured claim that every final floor is rougher than every roof. It is nonetheless opposite to the smooth-floor emphasis of the linked Valentine example. Floor, wall and ceiling roughness should be controlled separately and conditioned on intact pahoehoe, collapse debris, erosion and roof accretion. See [geometry.py](../src/plume_advanced/stages/geometry.py).

**What the meshing check establishes.** The longest generated segment, ID 40, is 259.84 m long. It was isolated with its original section samples and meshed at the resolved 0.6 m voxel size. Junction stamps, events, export smoothing and textures were excluded. The resulting mesh has 27,856 vertices and 55,708 faces and is watertight. Three interior normal cuts differ from their corresponding Stage C width and height by less than 3%; the largest dimensional discrepancy is approximately 0.18 m.

Historical figure: Original contours versus cuts through the generated mesh (unavailable local artifact: `outputs/realism_comparison/mesh_section_check.png`)

This supports the use of the existing mesher for these passage envelopes. It does not validate full-network connectivity, junction pillars, a finished textured scene, or centimetre-scale features. A 0.6 m base grid cannot reproduce the fine geometry resolved by a 5 cm cloud. Texture detail can improve appearance, but does not restore missing physical ledges, thin remnants or collision geometry. Full-scene sampling should eventually be checked again after events and export smoothing.

**How body dependence currently works.** The built-in choices are Earth, Mars and Moon. Physical scalar overrides exist, but unknown body/material names are rejected; Io, Europa, Enceladus and Titan are not supported presets. See [world.py](../src/plume_advanced/world.py) and [body validation](../src/plume_advanced/world.py).

| Resolved default | Earth | Mars | Moon |
|---|---:|---:|---:|
| Gravity | 9.81 m/s² | 3.71 m/s² | 1.62 m/s² |
| Minimum section width | 4.4 m | 22 m | 44 m |
| Ordinary-passage width cap | 10 m | 50 m | 100 m |
| Room width cap | 28 m | 100 m | 200 m |
| Minimum height control | 3 m | 15 m | 30 m |
| Standard voxel size | 0.6 m | 1.2 m | 2.4 m |
| Fresh median contour width | 7.46 m | 37.85 m | 95.19 m |
| Fresh median contour height | 4.37 m | 19.46 m | 50.04 m |

The last two rows come from fresh Stage A–C runs with the same project configuration and seed, changing only the body through the configuration loader. They demonstrate a response to the presets, not agreement with measured extraterrestrial interiors. The resolved settings and measurements are saved in body_comparison.json (unavailable local artifact: `outputs/realism_comparison/body_comparison.json`).

Gravity and rock properties already influence a roof-stability proxy, while body scales change the host and section settings. These are meaningful dependencies. Nevertheless, the initial lava temperature, linear cooling coefficient and nominal flow speed remain the same under the body-only override in this project configuration: 1,450 K, 0.025 K/m and 0.35 m/s. Flow is conserved through the graph, but flux weights and geometric response are procedural surrogates. There is no compositional viscosity law or environment-dependent energy balance from which the section shape is solved. The code explicitly describes the eruption controls as dimensionless surrogates in [world.py](../src/plume_advanced/world.py).

Formation and survival should be separated. An admissible roof span under low gravity does not predict which tube actually formed. [Blair et al.’s structural calculations](https://digitalcommons.usf.edu/kip_articles/5195/) depend on width, roof thickness and initial stress state; they are stability constraints rather than observed size distributions. [Keszthelyi’s thermal-budget study](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/95JB01965) provides a physical basis for accounting for heat losses, crystallization and lava cooling. A practical reduced model can use these dependencies without requiring a full three-dimensional fluid simulation.

Evidence should also be labeled by body. Earth provides directly measured interiors. For Mars, [Crown et al.](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022JE007263) characterize systems using orbital morphology and topography, including collapse chains and ridges; that is not an interior section catalogue. For the Moon, [Carrer et al.](https://www.nature.com/articles/s41550-024-02302-y) infer a subsurface conduit from radar at Mare Tranquillitatis. That constrains a particular site, not a universal lunar cross-section distribution.

The material and formation mechanism must accompany the body name. [Io has active lava volcanism](https://science.nasa.gov/jupiter/jupiter-moons/io/), whereas [Enceladus ejects water vapor and ice from an ocean beneath its icy crust](https://science.nasa.gov/mission/cassini/science/enceladus/). [Europa’s possible surface flows](https://europa.nasa.gov/resources/95/flow-like-features-on-europa/) are interpreted in an icy context. Generating their conduits requires distinct material/process assumptions; changing gravity in a terrestrial basalt model is insufficient. Basaltic cave interiors on those worlds should not be presented as validated outputs.

**Recommended development order.**

1. Establish Earth morphology targets before adding more body labels. Remove the coupling of minimum passage size to the maximum body cap; make traversal requirements an optional application filter. Fit joint width–height–shape distributions and their changes along passages. Keep an ordinary passage distinct from a compound room and its solid pillars.
2. Extend section state with separate floor and roof elevations, preserved lava-level benches, locally incised channels and controlled asymmetry. Permit multiple contours or solid inclusions where the network implies separated passages or surviving pillars. Preserve the existing graph and SDF interfaces where possible.
3. Replace selected heuristic controls with a reduced physical model: discharge in physical units, lava composition and temperature-dependent viscosity, slope/head, crystallization and cooling, crust growth, erosion/deposition and roof load. Body/environment profiles should supply boundary conditions and uncertainty, not just maximum dimensions. Formation-state changes should feed back into the host and network where they alter later flow paths.
4. Separate floor, wall and roof surface processes. Generate broad shapes first, explicit geological features next, then fine surface detail at a resolution appropriate to the required scale. Test junctions for unintended joins and lost rock pillars after meshing.
5. Validate on multiple seeds and separate observables: graph connectivity and branch geometry; locally normal sections; longitudinal floor/roof changes; multi-scale roughness; and final mesh fidelity. Weight caves and physical station spacing deliberately. Reserve the 19 evaluation caves for a frozen confirmatory experiment. Valentine has already informed design choices, so it is a calibration case study, not an untouched test of predictive accuracy.
6. Extend to Mars and Moon using explicit scenario priors and observational bounds, reporting which constraints are measured, inferred or assumed. Add other worlds only with an appropriate material/formation model.

**Reproduction and artifacts.** Fresh network diagnostics are in current_earth (unavailable local artifact: `outputs/realism_comparison/current_earth/summary.json`). Numerical comparisons, config/source fingerprints and the isolated meshing check are in comparison_metrics.json (unavailable local artifact: `outputs/realism_comparison/comparison_metrics.json`); all generated contour measurements are in generated_contour_metrics.csv (unavailable local artifact: `outputs/realism_comparison/generated_contour_metrics.csv`). The bounded mesh is isolated_segment.ply (unavailable local artifact: `outputs/realism_comparison/isolated_segment.ply`).

Run `scripts/generate_network_diagnostics.py --output outputs/realism_comparison/current_earth --density-sweep ''` using the project environment, followed by analyze_comparison.py (unavailable local artifact: `outputs/realism_comparison/analyze_comparison.py`). These are exploratory analyses, not a rerun of the full confirmatory benchmark or all export targets.

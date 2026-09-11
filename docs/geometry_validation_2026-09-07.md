# Lava-tube geometry validation — 7 September 2026

The current Earth tube is a useful visual prototype. This pass adds evidence about resolution, surveyed morphology and body-dependent behavior. It does not establish a calibrated physical model of real caves.

The accepted [full Earth tube](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/earth_tube_relief_seed2/lava_tube_geometry.glb) is preserved byte for byte. New fine meshes are independently capped inspection regions; they have **not** been stitched into a replacement full-network mesh. No rocks were generated.

## Shallow sections

Six shallow sections were regenerated with all original neighboring sweeps and junction influences at 20, 10, 5, 2.5 and 1.25 cm grid spacing. The 20 cm local cuts reproduce the full exported mesh within 0.1 mm. Every measured contour is closed and clear of the artificial inspection boundaries.

Five of six sections pass all final-refinement criteria. All six pass the height criterion; the largest final height change is 1.65 mm. Section 197 still changes by 5.66 cm in width and 5.82 cm in contour distance, concentrated at its thin side tips. Its height changes by 1.65 mm and its area by 0.11%. We retain this as an unresolved width measurement.

| Section | Height at 20 cm grid | Height at 1.25 cm grid | Final height change | All criteria |
|---|---:|---:|---:|---|
| 197 | 0.242 m | 0.308 m | 1.65 mm | Width/contour unresolved |
| 804 | 0.395 m | 0.377 m | 1.26 mm | Pass |
| 196 | 0.327 m | 0.334 m | 0.14 mm | Pass |
| 203 | 0.292 m | 0.337 m | 0.55 mm | Pass |
| 806 | 0.411 m | 0.390 m | 0.31 mm | Pass |
| 195 | 0.428 m | 0.421 m | 0.09 mm | Pass |

![Measured shallow sections](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/shallow_sections.png)

Criteria were declared before the study: height change ≤2 cm, width change ≤3 cm, area change ≤3%, and symmetric vertex-to-polyline distance ≤3 cm between the last two levels. The distance is a sampled contour measure, not an exact continuous Hausdorff distance. Heights are cross-sectional extents, not guaranteed walking clearances.

This varies the resolution of the whole configured pipeline, including voxel-dependent legacy roughness, fillets and smoothing. It measures practical output stability; it is not a pure discretization-error bound for one fixed geological surface.

The new generation report examines both profile width and height. It flags 233 of 830 original profiles at the full mesh’s 20 cm grid under an initial eight-sample criterion. That criterion is a screening heuristic before relief, not a claim that eight samples ensure convergence.

Inspect the [finest section 197 mesh](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/resolution_final/sample_197/voxel_0.0125m/lava_tube_geometry.glb), [all measurements](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/resolution_final/summary.json), or [resolution screening](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/resolution_screen.json). The region meshes use metres and glTF Y-up coordinates; their box caps are inspection boundaries, not natural termini.

## Surveyed morphology

The comparison uses 1,286 valid sections from 76 calibration caves in [Pyroduct Digital Catalog v2](https://zenodo.org/records/17750755). Coordinates from the 19 reserved evaluation caves were not opened. Reference caves receive equal weight; generated profiles are weighted by represented arc length.

| Descriptor | Surveyed calibration median | Earth scenario median |
|---|---:|---:|
| Width (m) | 5.687 | 7.496 |
| Height (m) | 3.039 | 1.947 |
| Width / height | 1.863 | 3.683 |
| Solidity | 0.9446 | 0.9602 |
| Normalized roof asymmetry | 0.01802 | 0.003508 |

The Earth scenario is broader and flatter, with less roof asymmetry and less variation in concavity. These statistics describe input profiles before 3-D relief. They are not measurements of every exported mesh section.

About 82.1% of represented network length has input section heights between 1 and 3 m. The accepted scenario settings were retained. The survey catalogue contains a broader range of Earth caves, so fitting its global size distribution would change that scenario. No parameter fitting was performed.

The station-order changes are exported, but PDC generally lacks physical distance between stations. These data cannot calibrate a longitudinal correlation length in metres. Floor residual and roof asymmetry were measured with equal boundary sampling; both describe section shape, not centimetre-scale wall texture.

[Distribution plots](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/calibration/comparison.png) · [Calibration measurements and provenance](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/calibration/summary.json)

## Surface variation

The local 10 cm Valentine LiDAR product was compared with the matched before/after 64 m generated reach. The [USGS data release](https://www.usgs.gov/data/nasa-tubex-valentine-cave-2018-valentine-lidar) documents the 2018 acquisition. Each source uses 1,500 sampled patch centers, equal-radius neighborhoods and 24 points per fitted plane.

| Neighborhood radius | Before relief: median plane residual | After relief | Valentine scan |
|---|---:|---:|---:|
| 0.4 m | 0.60 cm | 1.54 cm | 2.07 cm |
| 0.8 m | 1.81 cm | 3.07 cm | 4.47 cm |

The added relief moves the small-scale variation toward the reference. This supports the visual improvement, but the measurements combine surface relief, curvature and edges. Point coverage differs between scan and synthetic sampling. The 1.6 m neighborhoods can include opposite roof/floor surfaces in low passages; their larger residuals must not be read as material roughness. Roof, floor and wall feature proportions remain uncalibrated.

[Surface comparison plots](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/surface_scales/comparison.png) · [Raw patch measurements and limitations](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/surface_scales/summary.json)

## Body and seed checks

All nine runs passed network connectivity, source reachability, positive finite dimensions, non-self-intersecting profiles, and the local mesh export/cut checks. Each run generates the complete host, network and section fields; it meshes one junction region. This is not a whole-mesh test of nine networks.

Four initial inspection boxes clipped the selected contours. Their regions were expanded, and every accepted cut now lies more than two voxels from the box boundary. Initial clipped exports remain in their `initial_clipped_region` folders for audit. At joins, a transverse plane may also run along another branch; its full extent is not an individual tube width or an unsupported roof span.

| Body | Seed | Network segments | Original profiles | Median profile height |
|---|---:|---:|---:|---:|
| Earth | 1 | 60 | 847 | 2.13 m |
| Earth | 2 | 69 | 830 | 1.91 m |
| Earth | 3 | 64 | 835 | 2.01 m |
| Mars | 1 | 60 | 520 | 8.57 m |
| Mars | 2 | 65 | 522 | 7.66 m |
| Mars | 3 | 69 | 564 | 8.13 m |
| Moon | 1 | 133 | 669 | 15.34 m |
| Moon | 2 | 81 | 569 | 17.52 m |
| Moon | 3 | 78 | 540 | 17.20 m |

These runs share the Earth-scenario controls with each built-in body preset’s overrides. Their dimensions are outputs of those assumptions, not predictions validated against extraterrestrial interiors. Only Earth, Mars and Moon presets were tested.

A separate gravity counterfactual holds a 20 m wide, 6 m high passage, 1 m roof, 2,900 kg/m³ density, 3 MPa effective tensile strength and safety factor 1.5 fixed. The implemented beam screen gives demand ratios of 4.27 for Earth, 1.61 for Mars and 0.705 for the Moon. It closes the Earth/Mars fixture and preserves the Moon fixture. This verifies the implementation’s gravity response, not real collapse thresholds.

[Nine-run checks](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/body_matrix/summary.json) · [Clipping audit](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/body_matrix/clip_audit.json) · [Gravity counterfactual](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/geometry_validation/body_matrix/collapse_counterfactual.json)

## Changes and verification

Generation now writes a resolution report that considers the smallest profile dimension. Local inspection retains neighboring network sweeps and rejects contours that touch artificial box caps. PDC loading can filter by cave ID before reading coordinates, preserving the reserved evaluation partition. Four scripts reproduce the resolution, profile-reference, surface-scale and body/seed studies.

The full test run passed 270 tests and 31 subtests, with one skipped. After adding the artificial-boundary guard, all 10 focused local-geometry/PDC tests passed. Linting, type checking of 74 source files and source/wheel package builds passed.

The next work before calling the generator physically realistic is calibration of section asymmetry and shape variation, followed by integrating local refinement into a continuous full-network export. Section 197’s width remains unresolved at the declared tolerance. Additional decorative roughness is not the main priority.

Reproduction from the repository root (using the existing trusted local checkpoint for the resolution study):

```bash
.venv/bin/python scripts/validate_tube_resolution.py --checkpoint /tmp/plume-relief-tube.pkl --output outputs/geometry_validation/resolution_final --resume
.venv/bin/python scripts/compare_tube_calibration.py
.venv/bin/python scripts/compare_surface_scales.py
.venv/bin/python scripts/validate_body_matrix.py
```

To produce a fresh checkpoint, run `scripts/generate_tube_only.py` with `config/earth_tube_only.toml`, a new output directory and `--checkpoint` pointing to a local file. Never load an untrusted pickle checkpoint. Detailed inputs, hashes, thresholds and raw measurements are stored alongside each study.

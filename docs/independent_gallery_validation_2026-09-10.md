# Independent-gallery validation — 10 September 2026

The final inspection run is `outputs/earth_independent_gallery_seed20260910`, generated with `config/earth_independent_gallery.toml`. Its release record contains the exact production source hash, accepted seed, mesh hashes and links to evidence. This validates a bounded procedural scenario, not geological truth or every PLUME mode.

## Network and sections

The main run accepted candidate 16 (zero-based attempt 15), seed 2109121975, with no coordinate repair. It passes all 51 network/section checks. There are three independently seeded sources, three merges, one split and rejoin around a surviving rock island, two blind breakouts, four formation phases, two reoccupied branches and one drained pool. Main route target: 380 m; combined centreline length: approximately 605 m.

The fixed-host matrix covers network seed labels 20260910–20260912 with two and three sources. All six cases pass within the configured 48-candidate limit, including repairs where necessary. These are A–C cases only; they do not contain six additional meshes. Reports are in `outputs/independent_gallery_validation/`. The seed labels use the validation script's documented derivation, so they are not equivalent to setting the root seed in the full preset.

Two fresh processes with Python hash seeds 11 and 37 reproduce equal network, section and shape hashes, plus byte-identical candidate-decision reports. Evidence is in `outputs/independent_gallery_reproducibility/`. The final full-run, matrix and reproduction reports are checked against the same production source digest.

## Mesh and imports

The first mesh failed additional inspection: it contained two components, including a roughly 0.51 m³ detached blind tip, and an extra tiny surface handle. The accepted graph and sections alone did not detect these discretization artefacts. The preset now enables a radius-one grayscale density closing after surface relief and before roof-stability assessment. This operation can close gaps up to two cells wide; its spatial limits are documented in the generation guide. It reconnects the tip instead of deleting its detached mesh.

The corrected raw mesh has 227,014 vertices and 454,028 triangles. The exported mesh has the same connectivity after welding duplicated seam positions. Both have one connected watertight component, consistent winding, Euler characteristic zero and genus one, matching the graph's single cycle. The voxel plan contains exactly one island, about 465.44 m², and no remaining tiny plan holes.

All 16 transverse spot checks produce closed contours containing the requested section centres. One confluence plane required the complete mesh: the initial local triangle crop truncated an oblique adjacent passage. The checker now retries the full plane before reporting a failed contour. These checks are not a continuous clearance survey. Of 616 section profiles, 350 have a minor dimension below eight 0.20 m voxels; shallow tips remain resolution-limited.

All 27 portable-asset checks pass. Actual Blender 4.0.1 import preserves triangle count, scale and dimensions within 2 mm. Interior camera locations have measured vertical clearances of approximately 1.64 m and 1.94 m. The native scene and renders are reused only when a byte-identical final GLB and identical section geometry are verified. Source and history summaries include the new breakout records.

Unity and Unreal packages use byte-identical GLBs, with no rocks or separate collision meshes. Native Unity and Unreal imports were not executed. The grey geometry previews have no photographic textures; the extra geometry-lighting images emphasize surface shape.

## Tests and remaining issue

- Network/configuration/section/continuity regression batch: **172 tests and 26 subtests passed**.
- Focused voxel repair, surface relief, configuration and gallery batch: **58 tests and 22 subtests passed**.
- Final gallery/history-summary regression: **17 tests passed**.
- Additional geometry/continuity/tiled batch: **31 passed, one failed**. These batches overlap; their counts should not be added.
- Lint checks on the changed production modules and new tests passed.

The failing broader test is `tests/test_geometry.py::GeometryTests::test_geometry_stage_stamps_voxels_and_builds_isosurface_mesh`. It loads the older general `config/project.toml` preset and expects one component. That preset produces 11 voxel components and five mesh components at 0.60 m resolution. Repeating it with the new fissure-repair call replaced by a checked no-op reproduces the failure; independent-gallery mode is disabled there. No roof-collapse event accounts for those fragments. This general-preset connectivity problem remains unresolved and is not hidden by changing the test's expectation. It is separate from the passing new-gallery mesh checks.

This report does not claim that every PLUME feature is integrated or that the complete repository test suite is green. The supported single-level history mechanisms are integrated; stacked levels, vertical capture and legacy cell-lobe/deposition feedback remain outside this mode.

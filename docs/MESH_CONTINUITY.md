# Continuous cave meshes and junctions

Chunks are processing units. The exported cave wall is assembled and welded before visual smoothing, UV charting, and displacement. A shared carved voxel volume alone does not guarantee a connected mesh.

## Changes

- Simplified paths retain branch attachment cells on straight parent routes before graph extraction. In the seed-1 example, this changes five disconnected graph groups into one graph with 68 segments. Physical intersections are no longer the only connection between those groups.
- Route centerlines are resampled in metres and curved after graph extraction. Shared nodes retain their exact positions, and cubic approach regions follow a common parent direction. Host elevation, cover, and growth cost are sampled again along the curved paths before flux and thermal state are assigned.
- Junction clusters are bounded by passage width and total cluster diameter. Distant confluences no longer collapse transitively into a single oversized junction region.

- Chunk vertices are matched by Euclidean distance, including matches across coordinate-rounding buckets. This prevents micrometre-sized numerical differences from becoming centimetre-sized cracks during smoothing.
- Profile sweeps include longitudinal distance and rounded endpoints. Their allocation boxes no longer determine where the cave ends. Endpoint caps are no longer stamped over every multi-sample segment.
- Ordinary junctions use the incident tunnel volumes with bounded, local blending. Explicit chambers retain their chamber volumes. Grade-separated crossing influences do not receive junction fillets.
- Segments sharing a graph endpoint approach a common direction, profile, and floor elevation, with gradual changes in section size and body-specific width limits. Changes fade out over a bounded connection region. Frames are recomputed from the shaped centerline while retaining the shared floor; underpasses keep their separate interior grades.
- Density stamping reserves bounds for the actual profile, relief, and smooth-union distance band. The empty-field value lies outside that band so omitted stamps cannot perturb neighboring tiles. Small isolated solid pockets of at most eight face-connected voxel samples are removed before meshing. Halo evaluation keeps this cleanup consistent at tile boundaries; larger rocks and connected dividers remain.
- Meshing fails before export when assembled edges are open or nonmanifold. Portable validation checks physical edge topology after welding exact UV-chart copies, so legitimate UV seams are distinguished from cracks.
- Collision clustering is accepted only when its result remains watertight and consistently wound. Otherwise the canonical mesh is exported as the collider. This can increase collision cost; a future topology-preserving decimator can reduce it safely.

This remains a procedural, host-driven network, not a fluid-dynamics simulation. Rounded natural termini and deliberate blind branches remain part of the generated network. Grade-separated paths retain their explicit attachment semantics.

## Regenerate

Existing exports do not change automatically. Generate into a fresh directory:

```bash
.venv/bin/plume-generate --output outputs/new_environment/stage_b_cave_network.png
```

The normal project configuration still controls the world, seed, geological events, loose rocks, and target packages. Previously generated checkpoints are invalidated by production-source changes.

The inspection run in `outputs/natural_final/` uses seed 1, the standard Earth resolution of 0.6 m, and structural events. Loose rock props were disabled for this run to expose the wall geometry; they remain enabled in the user's normal configuration. The all-target export includes Blender, UE5, Unity, Gazebo, and Omniverse packages. Use the GLB as the visual asset and its collision sidecar for physics. The wall remains an inward-facing void boundary, not a finite-thickness rock shell.

## Regression coverage

`tests/test_network.py` checks restored attachment cells, graph connectivity, exact endpoints, rounded grid corners, and conserved flow. `tests/test_mesh_continuity.py` covers distance welding across rounding buckets, translated meshes at nonbinary voxel spacing, smoothing without seams, finite narrowing tube ends, shared junction floors, independent underpasses, dense/tiled junction surface and cleanup equivalence, physical holes versus UV seams, and collision fallback for narrow tubes. The existing geometry test checks one connected surface with closed manifold edges and inward-oriented surfaces and retained route centres.

## Verification notes

The geometry-only inspection mesh in `outputs/natural_v3/` has one connected, watertight wall with 879,032 triangles. `mesh_comparison.png` compares it with the original output using identical views and visual smoothing. `route_comparison.png` shows the centerline change. The mesh image crops the network to two regions; lines ending at the image crop are not physical tube termini.

The focused network, geometry, section, sampling, export, and validation tests passed, including the seed-1 network test that failed before route repair. Additional regressions cover rounded routes and identical junction surfaces at different tile sizes. Ruff and the whitespace check passed.

The final structural-event run and portable-asset validation results are recorded under `outputs/natural_final/`.

The structural-event realization contains a continuous main cave boundary (881,858 triangles) and one detached solid rock island (812 triangles, approximately 85 cubic metres), both closed. The combined wall object has 882,670 triangles. The second surface component is a resolved rock created by a geological event, not a disconnected passage or chunk seam. The untextured `outputs/natural_v3/cave_shape_preview.glb` omits structural events and has exactly one surface component.

Verification completed: 46 distinct focused tests and five subtests passed across the main run and added regressions. The final Blender GLB passed all 40 portable-asset checks. All four OBJ colliders (Blender, UE5, Unity, and Gazebo) are watertight and consistently wound, with 882,670 triangles each; the exporter retained canonical collision geometry when clustering could not preserve closure. See `validation/validation_report.json` and `collision_verification.json` in the final output directory. The Omniverse export uses the same prepared collision geometry.

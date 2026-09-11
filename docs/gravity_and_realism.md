# Gravity-dependent roofs and Earth morphology

PLUME separates **formation size controls** from **survival constraints**.
Stage B proposes the formation network. Stage C constructs profiles and marks
their roof stability after all frame, elevation and junction adjustments.
Stage D checks the section envelopes and its enlarged junction volumes before
meshing. A failed roof becomes a blocked region of breakdown material. Thus the
final accessible cave may have different connectivity from the formation graph.

## Observed passage heights versus structural limits

The earlier figure's roughly 14 m height was a **conditional screening bound**,
computed with an arbitrarily selected floor depth of 15 m. For its 20 m wide
example, the lunar bound was 14.30 m and the Earth bound was 10.73 m. Neither
value is a prediction or a target for ordinary passage height. This beam model
screens roof bending; it cannot establish a universal height limit, including
the stability of tall sidewalls. The updated figure shows required roof rock
thickness directly, removing the illustrative floor-depth assumption.

For the supplied Valentine reference, the [TubeX study](https://doi.org/10.1029/2019JE006138)
describes a roughly 1–3 m cave scale and explicitly discusses its approximately
3 m height. The [USGS survey](https://npshistory.com/publications/geology/bul/1673/sec3.htm)
records an entrance height of 8–10 ft (2.44–3.05 m), smaller distributaries
3–5 ft (0.91–1.52 m) high, and a compound pool with a locally higher roof
10–20 ft (3.05–6.10 m) above its floor. These describe different passage types.
Larger Earth examples also exist: [NPS describes Indian Tunnel](https://www.nps.gov/crmo/learn/education/site-tour-3.htm)
as over 30 ft (9.14 m) high. Three metres is therefore not a universal cap.

The saved generated Earth contours have a **median height of 4.27 m**, a
95th percentile of 7.03 m and a maximum of 10.61 m. Passing the roof screen
does not validate this distribution against Valentine. Its ordinary passages
still need height calibration, with low passages and compound rooms treated
separately. The pooled PDC height tail also cannot establish how frequently
such rooms should occur: survey sampling and cave representation are uneven.

## Coupled width and height limits

The screening model treats the roof as a simply supported beam of unit breadth
under its own weight. Let `w` be unsupported width, `h` cavity height, `d` the
depth of the floor below the local surface, `t = d - h` roof thickness, `rho`
rock density, `g` gravity, `sigma` effective rock-mass tensile strength and `F`
the strength safety factor. In consistent SI units:

```text
required_roof_thickness = 3 F rho g w² / (4 sigma)
demand_ratio            = required_roof_thickness / t
maximum_width          = sqrt(4 sigma t / (3 F rho g))
maximum_height         = max(d - required_roof_thickness, 0)
```

The intact candidate passes if `t > 0` and the demand ratio is at most one.
Height is bounded at a **fixed floor elevation**: making the tube taller
consumes roof cover. The two maxima are conditional views of the same
constraint, not independent per-planet numbers. Increasing cover or effective
strength permits wider roofs; increasing gravity, density or the safety factor
reduces the admissible span. At equal roof thickness and rock properties,
maximum width scales as `1/sqrt(g)`.

The coefficient `3/4` follows the maximum bending stress in a simply supported
beam under uniform self-weight. The default safety factor is 1.5. Effective
strength already includes the material profile’s quality and weathering
reduction; those reductions are not applied a second time.

This is a deliberately conservative, interpretable procedural constraint.
It does not model arching, regional stresses, layered roofs, fracture
propagation, lateral confinement, sidewall buckling, or individual support
pillars. In particular, it does not assert that real lunar caves have the
computed universal maximum sizes. [Blair et al. (2017)](https://doi.org/10.1016/j.icarus.2016.10.008)
model a much richer structural problem with sensitivity to stress state and
roof geometry. Their structural limits also do not establish the sizes that
lava flow can actually form. The project’s broader body-dependent formation
model remains a procedural surrogate.

![Required roof cover and independent floor features](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/realism_improvements/gravity_and_sections.png)

## Collapse behavior

- Stage C uses the actual transformed profile’s floor and ceiling elevations,
  rather than assuming that half the nominal height is the crown elevation.
- Stage D reassesses the supplied geometry, including configured profile
  enlargement, so stale section metadata cannot bypass the screen. Enlarged
  junctions are checked too, including the bounded
  lateral expansion used by their shape noise and refinement stencil.
- Failed envelopes are replaced locally by solid ellipsoidal breakdown plugs.
  The plugs cover the section and its adjacent sampling interval, or the
  enlarged junction. They represent a conservative **blocked collapse end
  state**, not a dynamically simulated rubble distribution or an assertion
  that every collapse opens a skylight. The breakdown footprint can extend
  into neighboring stable sections.
- These modifications run after passage/junction unions and before floor
  sampling. They are independent of optional event density, event kinds,
  rock props, and the optional events stage’s route-preservation rollback.
  A completely failed candidate can yield an empty mesh.
- Both dense and tiled density grids apply the same model. The finalization
  path retains the stability records, including when no optional events run.

The screening is local to sampled section envelopes and explicit junction
volumes. It is not a pointwise structural analysis of the final smoothed and
displaced export. Those later surface operations and unresolved geometry must
be considered when using the cave for engineering analysis.

## Configuration and reports

World profiles supply gravity, density, effective tensile strength and the new
`world.roof_safety_factor`. The loader passes these to the section model and
the optional event ranking. Direct Stage C clients can set
`gravity_m_s2`, `rock_density_kg_m3`, `effective_tensile_strength_pa`, and
`roof_safety_factor` on `SectionFieldConfig`. Values must be finite and
positive, with safety factor at least one. The existing body passage/room
maximums remain independent formation controls.

Bundled body presets currently cover Earth, Mars and the Moon. Other moons
need appropriate body, material and formation profiles; substituting gravity
alone does not establish realistic silicate or cryovolcanic tube formation.

Section samples and their NPZ export include `floor_world_z`, `roof_world_z`,
`maximum_stable_width_m`, `maximum_stable_height_m`, `roof_demand_ratio` and
`collapse_required`. The latter describes the candidate section, before
breakdown. Summary fields include `unstable_section_count` and
`maximum_roof_demand_ratio`.

Stage D’s `stability_records` identify each section or junction, its dimensions,
roof requirement, conditional width/height maxima, demand ratio and outcome
(`intact` or `blocked_by_breakdown`). `stability_collapse_count` counts failed
screened envelopes, not statistically independent geological collapse events.
`preserved_pillar_column_count` counts grid columns, not individual pillars.

## Morphology improvements

The project-loader defaults for minimum width and height are 0.5 m and 0.35 m,
independent of the body’s maximum passage size. These are numerical/configuration
floors, not empirical universal minimums. A run is not guaranteed to sample all
admissible sizes. Choose sufficient mesh resolution for the smallest desired
feature; lowering these controls does not automatically refine the entire grid.

`bench_strength` and `floor_incision_ratio` add preserved lava-level ledges and
a narrow floor channel. Their strength varies with the existing correlated
morphology and flow-maturity fields. Profile construction preserves the roof
curve while adjusting the floor; later connection harmonization still aligns
shared endpoints. These features remain heuristic interpretations of formation
history, not results of an erosion or crystallization solver.

The Earth project configuration lowers the height-ratio minimum and raises its
maximum to admit flatter and taller sections, and reduces baseline floor
relief. Stage D’s `floor_roughness_scale` and `roof_roughness_scale` default to
0.25 and 1.0. These gains control the same noise field at equal spatial
coordinates; they do not promise a measured final roughness ratio everywhere.

Before room/junction unions, the mesher identifies solid columns enclosed by
the projected existing passage footprint. It restores those remnants after the
unions so broad room stamps cannot erase existing loop pillars. Columns crossed
by a passage on any level are excluded, preventing an invented pillar from
blocking an underpass. Dense and tiled storage use the same global projection.
Open-ended gaps are not inferred as pillars, and these columns are not credited
as supporting individual roof beams in the conservative stability screen.

Drained-pool stamping also stops forcing the room width to at least twice the
already expanded median section width. The requested pool width and incident
envelope set the lower bound instead. This avoids double widening.

## Verification

The final full regression run passed: **243 tests and 25 subtests passed,
1 test skipped**, with **82.34% coverage** (70% required). Ruff, mypy,
compilation checks, and the offline source/wheel build also passed.

`tests/test_stability.py` checks analytical threshold crossing in both width
and height, the response to gravity and rock strength, invalid physical inputs,
actual versus nominal roof cover, mandatory collapse in dense and tiled grids,
complete closure with optional events disabled, enlarged-junction failure,
preserved pillars and underpasses, independent floor shaping, and separate
floor/ceiling roughness. Existing section, geometry, export and pipeline tests
provide integration coverage.

Tiled geometry reconciles shared density samples before meshing, using the
same sample ownership as density queries. Regression tests cover faces, edges,
corners and missing neighboring tiles. Mesh welding also preserves distinct
vertices within a chunk to avoid deleting thin surface triangles.

The seed-1 Earth comparison is stored in
[validation_metrics.json](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/realism_improvements/validation_metrics.json).
It uses actual contours from the saved before/after runs. It is an exploratory
regression check; no reserved PDC evaluation caves were used for tuning.

The full seed-1 Earth surface with the compact integration-test event
population is saved as
[verified_cave_surface.ply](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/realism_improvements/current_earth/verified_cave_surface.ply).
It contains 363,627 vertices and 727,334 triangles, with zero boundary edges
and zero nonmanifold edges. This verifies mesh integrity, not geological
accuracy. The illustrative
[cutaway](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/outputs/realism_improvements/verified_mesh_review.png)
removes the ceiling for inspection; its crop boundaries are artificial.

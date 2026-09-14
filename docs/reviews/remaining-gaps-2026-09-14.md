# Remaining gaps and completion plan

Assessment date: 14 September 2026. Reviewed implementation: **`1f8017f`**,
following the [validated cleanup](cleanup-2026-09-14.md). This is an inspection
and proposed implementation plan, not a claim that the work below has run.

PLUME has an integrated, reproducible generation/inspection/repair pipeline.
It is not yet a uniformly configured, performance-qualified simulator asset
generator across all its presets, lengths, bodies and optional features.
The most useful next change is to make the intended acceptance requirements
explicit and consistent, then close the geometry and native-application gaps
against those requirements.

## Evidence and priority

The [V3 campaign](mobility-repair-campaign-2026-09-14.md) covers six textured,
rock-free Earth designs at a 250 m dominant-route target, plus six cold replays.
All passed their declared numerical and native checks. Its source is `c5d8922`;
the cleanup has separate regression evidence. Historical larger campaigns remain
valuable, but were run against earlier code and different resolutions.

| Priority | Gap | Current evidence | Completion condition |
|---|---|---|---|
| 1 | Preset and acceptance consistency | Only 4/16 shipped configurations enable required clearance; only 1 enables collision; none enable refinement or export-size budgets | A declared simulation profile requires the requested checks; a neutral research profile clearly reports their absence |
| 1 | Resolution and retained geometric detail | All six V3 cases have input-resolution warnings; two multi-source cases accept zero added relief and one accepts 25% | A strict quality profile rejects unverified resolution or excessive detail loss, with bounded repairs and measured evidence |
| 1 | Collision behavior inside the engines | Native fixtures check imported counts, positions and vertical rays | Native finite-body sweeps and overlap checks agree with the accepted source routes, including deliberate obstruction failures |
| 2 | Long-cave cost and visual mesh size | V3 visual meshes have 0.75–1.40 million triangles; GLBs are 92.0–116.5 MB; no current visual LOD exporter | Short and kilometre-scale exports meet a declared runtime/import/storage budget on the target hardware |
| 2 | Material validation beyond successful import | Many planned views pass, but Unity and Unreal brightness differs; native fixture requires one rock-free 4K primitive | Stable material controls, broader input coverage and separate material/lighting diagnostics are automated |
| 2 | Broader seed and feature coverage | Current native campaign has three root seeds in each of two modes, no rocks/events, and Earth only | A frozen stratified campaign covers current supported modes and preserves all failures, budgets and cold-replay results |
| 3 | Scientific scope and other bodies | Earth reference comparisons exist; body presets are Earth/Mars/Moon and use a simplified roof screen | Claims match measured morphology and structural assumptions; additional bodies have documented material/formation models and their own validation scope |

The [resolved preset inventory](remaining-gaps-preset-inventory-2026-09-14.json)
records the exact values and configuration hashes behind the first row. It is a
read-only inventory, not a generation campaign. Disabled features in intentional
research presets are not themselves defects; inconsistent expectations between
an inspection example and a simulation-ready delivery are the problem.

## 1. Define acceptance profiles before another broad campaign

Relevant code: [`GeometryConfig`](../../src/plume_advanced/stages/geometry_types.py),
[`ExportConfig`](../../src/plume_advanced/world.py),
[`complete_inspection`](../../src/plume_advanced/pipeline/inspection.py),
[`reliability campaign`](../../src/plume_advanced/evaluation/reliability.py),
and the maintained [`config`](../../config) directory.

The short single preset currently requests no required capsule corridor and no
dedicated collider. Even the full interconnected presets disable collision.
The V3 campaign supplied additional settings; selecting a similarly named preset
does not reproduce that campaign's acceptance contract. All 16 configurations
leave the available triangle and byte budgets at zero, which means unlimited.

Add a separate acceptance-policy configuration, rather than overloading
`run.quality` (which already affects resolution selection). The policy should
state whether clearance, collision, resolution convergence, detail retention,
texture checks and native checks are required. Resolve it into the manifest
before work starts. Keep scientific crawlway generation possible through an
explicit research policy; do not silently enlarge all optional branches.

For the requested inspection profile, require 0.5 m height × 0.5 m width with the
existing 0.02 m margin on declared routes. Preserve the current upright-capsule
interpretation; equal height and width gives a sphere, not a box or a vehicle.
Expose each requirement's result as passed, failed, not requested or unavailable.
Keep informational repairs distinct from unmet mandatory requirements. A repair
may change effective settings only within the chosen policy, and cannot change
the policy itself to manufacture acceptance.

**Tests and exit criteria:** load every advertised preset and check its resolved
policy; reject contradictory requirements in preflight; require equivalent
policy evaluation in `plume-generate`, `plume-check` and resumed runs; inject
missing collision, absent native reports and excessive relief loss and verify
that strict publication fails. These checks should inspect resolved values and
delivered artifacts, rather than just look for TOML key names.

## 2. Preserve detail and demonstrate resolution

Relevant code: [`_accept_base_surface`](../../src/plume_advanced/stages/geometry.py),
[`surface_relief`](../../src/plume_advanced/stages/surface_relief.py),
[`surface_defects`](../../src/plume_advanced/stages/surface_defects.py),
[`resolution`](../../src/plume_advanced/pipeline/resolution.py), and
[`mesh_inspection`](../../src/plume_advanced/stages/mesh_inspection.py).

Local relief repair exists, but bounded global candidates still include scales
1, 0.5, 0.25 and 0. Zero removes the added accretion layer, while retaining section
shape and texture. A textured render can hide that geometric loss. Current
reports warn about it, but do not enforce a minimum detail requirement.

First record how much surface area each repair affects and its displacement
amplitude relative to the requested relief. Establish a quality-policy limit
from controlled examples and reference measurements, not an arbitrary universal
rock-roughness number. Improve localization of offending features before
lowering relief globally. If local repair cannot satisfy both detail and geometry
requirements, preserve the failure and try only the already bounded upstream
recovery; a globally smooth candidate must not pass a policy that requires relief.

The existing refinement path uses a consistent grid per candidate and compares
floor and roof separately along common probes. That is useful, but does not
measure complete contour convergence. Extend the comparison to widths, profile
boundaries, junction openings and deliberately selected high-curvature/low-cover
regions. Keep topology and finite-body clearance checks at every accepted level.
Matching total passage height cannot substitute for matching both floor and roof.
Closed, oriented topology alone also does not exclude all self-intersections;
add spatial triangle-pair checks after surface processing, with explicit shared-edge
and shared-vertex exclusions and bounded diagnostic output.

Start with consistent tiled-grid refinement on representative short reaches.
Profile allocation and repeated copying before attempting finer full kilometre
grids. Adaptive patch stitching would be a new mesher feature: it needs shared
boundary samples, compatible extraction, normal continuity and junction tests.
Do not treat independent diagnostic patches as a repaired full mesh.

**Tests and exit criteria:** known convergent and deliberately nonconvergent
cavities; unequal floor/roof errors; between-probe constrictions; intersecting
surface patches with otherwise valid edge counts; seam-crossing repairs; unchanged
remote detail; stable identifiers on cold replay; and explicit rejection at voxel,
memory, time or detail limits. A strict short-cave campaign must have no unresolved
mandatory fidelity checks. A resolution warning can be closed by its declared
convergence evidence, not by deleting the warning or relaxing a clearance limit.

## 3. Validate the imported collision mesh with a finite body

Relevant code: [`route_clearance`](../../src/plume_advanced/stages/route_clearance.py),
[`route_placement`](../../src/plume_advanced/stages/route_placement.py),
[`collision exporter`](../../src/plume_advanced/exporters/collision.py),
[`native runner`](../../scripts/check_native_engines.py), and the
[`Unity`](../../tests/fixtures/unity/PlumeNativeCheck.cs) /
[`Unreal`](../../tests/fixtures/unreal/native_check.py) fixtures.

Export the accepted repaired path, its dimensions, units and source identity as
an interchange artifact. The native runner should consume that exact path and
check the independently imported dedicated collider. Sweep every path edge and
test initial overlap, endpoints, junction connectors and stationary cases. Use
the sphere case explicitly when capsule height equals width. Verify metre to
centimetre conversion, backface behavior and collision-layer selection.

Unity's capsule cast does not report an already overlapping collider, so a cast
alone is insufficient; pair it with an overlap check. Unreal provides a capsule
sweep with a complex-collision option. These APIs are starting points for the
fixture, not evidence that PLUME has already exercised them.
[Unity CapsuleCast](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Physics.CapsuleCast.html),
[Unreal Capsule Trace](https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Collision/CapsuleTraceByChannel).

**Tests and exit criteria:** all required paths pass on the cooked collider in
both engines; a thin transverse wall, an initially overlapping body, a blocked
corner, wrong unit scaling and an intentionally omitted collider each fail.
Retain failing positions and source/cooked asset receipts. Do not count another
ray-only check as completion of this work.

Grounded robot or character motion is a separate contract. It needs a declared
footprint, step height, slope limit, support/contact requirements and time step.
The current vertical search can find a floating geometric corridor; it cannot
establish wheel contact. Its seven height bands and fixed XY route also mean
search exhaustion is inconclusive about all possible paths. A later bounded
lateral search can reduce false rejections, but must preserve route connectivity,
honor the same query budget and independently recheck every returned path.

## 4. Make kilometre-scale exports practical

Relevant code: [`geometry storage`](../../src/plume_advanced/stages/geometry_types.py),
[`prepared scene`](../../src/plume_advanced/exporters/scene.py),
[`target exports`](../../src/plume_advanced/exporters/targets.py), and
[`collision reduction`](../../src/plume_advanced/exporters/collision.py).

Current collider reduction saves 40–80% of collision triangles in V3, but leaves
the visual triangle counts unchanged. Tiled density storage is not runtime mesh
streaming: the final cave is assembled and exported as a whole. Add a separate
visual LOD and chunk-export path while retaining the accepted master surface.
Use deterministic spatial chunk IDs, shared boundary positions/normals and a
single reusable material set. Keep the validated collider independent of visual
LOD. Chunk seams must not create invisible barriers or unload the floor beneath
the inspection body.

Unity LOD groups manage supplied detail levels; Unreal World Partition streams
spatial actors. PLUME must still produce suitable mesh assets, bounds and material
references for either integration. Splitting one huge scene into names without
separate geometry does not provide that benefit.
[Unity LOD Group](https://docs.unity3d.com/6000.0/Documentation/Manual/class-LODGroup.html),
[Unreal World Partition](https://dev.epicgames.com/documentation/unreal-engine/world-partition-in-unreal-engine).

**Tests and exit criteria:** seamless reconstruction at maximum detail; declared
LOD error bounds; conserved metre scale and topology at chunk boundaries; no
texture reset at chunk origins; native traversal across load/unload boundaries;
and measured cold import, cooking, peak RAM/VRAM, bytes, visible triangles and
frame-time percentiles. Extend budgets to the delivered package and collider,
not only the primary visual asset. A coarse mesh that loses a required passage
cannot be accepted merely because it is smaller.

Use the existing 3600 s / 8192 MiB worker limits as the initial comparison
baseline, with one native editor at a time. Log budget exhaustion; do not raise
limits automatically. For runtime testing, a proposed initial target is p95 frame
time at most 33.3 ms at 1920×1080 on the recorded RTX 3070 machine, with fixed
render settings and a repeated route. This is a proposed engineering target,
not a measured result or the user's confirmed simulator specification. Profile
short and 1–3 km scenes separately before setting final package/VRAM limits.

## 5. Separate material correctness from illumination

Relevant code: [`material adapters`](../../src/plume_advanced/material_assets),
[`native views`](../../src/plume_advanced/evaluation/native_views.py), and
[`native runner`](../../scripts/check_native_engines.py).

The current fixture requires one cave primitive, three 4096×4096 maps and a
dedicated collider. It is valuable but deliberately narrower than all supported
exports. Both engines can halve light intensity per view to avoid clipping;
those records do not establish matched lighting or appearance.

Add controlled unlit albedo, normal-orientation and roughness checks before lit
comparison. Use fixed cameras, tile scale, a neutral reference patch and explicit
exposure/color settings for the latter. Preserve the existing exposure-adjusted
survey as a separate visibility check. Compare seams in projection-transition
regions, shallow floors, junctions and chunk boundaries; include a deliberately
broken projection as a negative control. Check object movement/scaling against
the declared projection-coordinate convention.

Generalize the fixture by declared capabilities: neutral/PBR, multiple image
sizes and multiple primitives. Unsupported cases should report unsupported,
never a successful empty check. Add native Blender acceptance to the same
evidence format, and attach native results to a campaign only after matching
the export receipts. Engine integration remains opt-in when tools are absent;
a policy requiring it must remain incomplete until those checks run.

**Tests and exit criteria:** declared material sizes/conventions import correctly;
normal inversion and color-space errors are detected; the seam control fails;
ordinary and boundary views pass repeatable criteria. Keep rendering tolerances
separate from deterministic source/mesh identities. Defer Gazebo/Omniverse native
claims until comparable application checks exist for their own material paths.

## 6. Run a staged, reproducible campaign on the resulting version

Freeze the source, inputs, acceptance policies, seeds, hardware, limits and
success criteria before execution. Retain the earlier problematic seeds 0, 1,
17, 42, 20260912 and 4294967295 as regressions. Add fresh seeds from a recorded
independent seed schedule; do not select only attractive results.

| Stage | Proposed coverage | Purpose |
|---|---|---|
| Fast ensemble | Earth/Mars/Moon × single/interconnected × short/long × 25 seeds = 300 network/section cases | Host containment, topology, coupled stability, source interactions and repeatability before expensive meshing |
| Full geometry | Four predeclared seeds from each of those 12 groups = 48 complete meshes, each with a cold replay | Final topology, body clearance, resolution, detail and resource requirements |
| Full textured Earth | Six seeds × single/interconnected × short/long = 24 originals, each with a cold replay | Whole repair/export contract with reusable 4K maps and dedicated collision |
| Native validation | Every original in the 24-case textured set, plus targeted negative fixtures | Materials, cooked collision, finite-body motion and runtime budgets |
| Optional-feature matrix | Declared cases covering history/layout modes, stacked crossings, structural events, props/providers, dense/tiled storage, neutral/PBR and export targets | Exercise live paths excluded from the rock-free campaign, including expected provider/input failures |

These are proposed counts, not completed evaluations. Use body-appropriate host
and resolution recipes; changing only an Earth preset's body label is not a
controlled test when metric overrides suppress body scaling. Deduplicate any
overlap between the full-geometry and textured sets in the frozen inventory.
Cold replays test reproducibility and are not additional independent worlds.
The optional-feature inventory needs explicit cases before launch, including
which event-induced topology changes are legitimate.

Advance in stages; stop promotion when a required gate fails. Report unchanged
acceptance, local repair, replacement network, correct bounded rejection,
unexpected exception, timeout and incomplete work separately. Report denominators
and uncertainty per group; aggregate success can hide a failing multi-source or
long-cave group. A constrained case rejected without publishing an invalid mesh
has a different meaning from a programming crash.

Use the earlier measured 10–31 minute short-case times only for rough scheduling.
They do not predict kilometre-case cost. Plan disk space before retaining cold
replays and native projects. This plan does not start the campaign automatically.

## 7. Bound the scientific claims

Relevant code and evidence: [`world`](../../src/plume_advanced/world.py),
[`stability`](../../src/plume_advanced/stability.py),
[`evaluation`](../../src/plume_advanced/evaluation),
[`reference data`](../reference_morphology_data.md), and the
[`claim/evidence matrix`](../paper/CLAIM_EVIDENCE_MATRIX.md).

Repeat terrestrial morphology comparisons for the changed generator using
width/height distributions, asymmetry, benches, bottleneck persistence, and
network split/merge persistence. Compare like spatial scales and account for
survey coverage. The PDC supplies terrestrial cross-section data; the Valentine
LiDAR release includes scan-spacing and occlusion qualifications. Neither one
alone establishes a universal network shape or a universal Earth height limit.
[PDC v2](https://zenodo.org/records/17750755),
[USGS Valentine LiDAR](https://www.usgs.gov/data/nasa-tubex-valentine-cave-2018-valentine-lidar).

The repository has 76 calibration caves and 19 confirmation caves. The historical
paper campaign already reports results against the 19-cave set. Reusing it is
useful regression evidence, but is not a new untouched holdout. Freeze any new
confirmatory data before further tuning and report uncertainty at cave level,
not as if every section were an independent cave.

Earth, Mars and Moon are the only current body presets. Additional Jovian or
Saturnian moons require an explicit material/formation domain and parameter
provenance, not just another gravity constant. Separate rock and any proposed
ice-host models. Keep empirical constraints, inferred behavior and artistic
controls identifiable; unavailable extraterrestrial interior surveys cannot be
replaced by a claim of validation from Earth data.

The roof calculation is a local beam-based screen. A fuller structural claim
needs an independently validated model covering its declared loading, material,
fracture and geometric assumptions. Sensor or collision interaction with an
external rock mass also needs a finite-host-shell/entrance representation; the
current cavity boundary should not be described as that feature. These are
extensions to the scientific and simulation scope, not prerequisites for honest
use of the present interior procedural model.

## Recommended implementation order

1. Acceptance-policy results and consistent simulation presets, with publication
   failure tests. This makes every later completion criterion enforceable.
2. Native finite-body checks using the accepted repaired routes; run the existing
   failure fixtures and six known designs through them.
3. Detail-retention limits, improved local repair, contour convergence and bounded
   intersection diagnostics on short examples; profile their costs.
4. Visual chunk/LOD export and native performance checks for long caves.
5. Generalized material/native checks and the staged current-version campaign.
6. Recalibration and additional body/structural models as separately scoped work.

A defensible release claim is that supported profiles either publish an asset
meeting their recorded checks or stop with reproducible evidence within finite
budgets. No finite test suite proves that every possible seed generates an
acceptable or scientifically realistic cave. The remaining work should tighten
the checked contract and measure its scope, rather than promise universal success.

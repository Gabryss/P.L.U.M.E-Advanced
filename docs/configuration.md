# Configuration

[← Project overview](../README.md) · [Generation](usage.md) · [How the pipeline works](architecture.md)

## One small recipe, one resolved configuration

The user-facing file is [config/project.toml](../config/project.toml). Its core is:

```toml
recipe_version = 1
preset = "preview"
procedural_seed = 3

[world]
body = "earth"

[run]
render_diagnostics = true

[export]
target = "neutral"
```

A recipe expands **in memory**: shared calibration → named preset → your TOML
overrides → explicit CLI/evaluation overrides → body/stage defaults and validation.
It never rewrites another config file. The internal calibration is stored once
in [presets.json](../src/plume_advanced/presets.json); users do not need to edit it.
The same loader serves generation, evaluation and qualification.

| Everyday control | Meaning |
|---|---|
| `recipe_version = 1` | Compact recipe format; do not combine with `schema_version` |
| `preset` | Selects a coherent host domain, topology, resolution and acceptance policy |
| `procedural_seed` | Initial nonnegative root seed; controls all named stage streams and sampled ranges. A search may accept a later seed |
| `world.body` | `earth`, `mars` or `moon`; an edit selects that body's default rock material unless explicitly overridden |
| `run.max_seed_attempts` | Total full-generation attempts, including the initial seed; default `8`, `0` disables this limit |
| `run.max_attempt_seconds` | Optional time limit per seed including repairs/export; default `0` means no timer |
| `run.render_diagnostics` | Produces stage figures in addition to mandatory inspection evidence |
| `export.target` | Target package; changing it selects the corresponding format unless one is supplied |
| `acceptance.profile` | `research`, `inspection` or `simulation`; requirements are explained in [Architecture](architecture.md#embedded-inspection-and-repair) |
| `acceptance.require_ground_routes` | `false` by default in every preset; explicitly set `true` to require reference-robot geometric qualification |

Relative asset paths are resolved against **the recipe's directory**. Paths in
the bundled textured examples assume `config/` beside `texture/`. If you move a
recipe, update its paths or use absolute paths. Unknown keys, invalid versions,
non-finite numbers and contradictory acceptance controls fail explicitly.
Tables merge recursively; arrays such as `[min, max]` replace the entire array.
`--show-config` displays the resulting values before any allocation or generation.
`--seed NUMBER` overrides the root seed, including stage streams and sampled
ranges. It does not edit the TOML file. The accepted seed and settings are written
to `seed_attempts.json` and `resolved_project_config.json`.

## Focused overrides

Add only the tables you need. For example, change the mesh resolution and
triangle budget while retaining the calibrated network:

```toml
[geometry]
resolution_policy = "fixed"
voxel_size = 0.10

[export]
target = "unity"
max_visual_triangles = 2000000
max_asset_bytes = 268435456
```

Edit an existing table rather than declaring it twice in TOML. `run.quality`
selects resolution only with `geometry.resolution_policy = "body"`; it does not
supersede a fixed voxel size. `run.dev_mode` crops a body-scaled domain rather than
shrinking passage dimensions. Use `short-*` / `long-*` for the supplied Earth
extents. Changing only `network.target_route_length_m` does not resize the host:
longer or wider systems also need sufficient grid extent and source placement.

The specialised `trunk`, `gallery`, `gallery-long` and `interacting` recipes expose
distinct active network models. `body-study` uses body-scaled host conditions;
`research` retains the general scientific scenario and optional events used by
`src/plume_advanced/evaluation/resources/experiments.toml`. Short/long inspection domains are explicitly metric:
a `world.body` edit changes physics and passage defaults but does not automatically
widen those fixed host grids. Inspect the resolved domain for large Moon/Mars tubes.

For a ready-made visual showcase, use [config/showcase.toml](../config/showcase.toml).
It extends `short-multi` with fine meshing, 8K maps, Rocky debris, geological events
and all exports. The [showcase guide](usage.md#showcase-generation) explains its
detail settings, dependencies and finite resource budgets.

## Appearance, rocks and route requirements

| Settings | Effect / units |
|---|---|
| `events.enabled`, `events.include_rock_props` | Geological modifications and separate loose-rock props; off in the first-cave recipe, enabled in the showcase |
| `events.use_rocky_meshes` | Requests the optional Rocky provider; otherwise built-in prop meshes are used |
| `geometry.cave_diffuse_texture`, `cave_normal_texture`, `cave_roughness_texture` | PBR source paths; empty paths select a neutral material |
| `geometry.embedded_texture_max_size` | Maximum image edge in pixels; 4K has one quarter the pixels of 8K |
| `geometry.cave_texture_scale_m` | Metres per repeated tile, independent of cave length |
| `geometry.cave_normal_scale` | Fine shading strength; does not change collision or passage clearance |
| `geometry.cave_normal_convention` | `opengl` or explicitly declared `directx`; no guess from filenames |
| `geometry.texture_repair_attempts` | `0` inspects only; `1` permits bounded map repair and one package rebuild |
| `acceptance.route_height_m`, `route_width_m`, `route_margin_m` | Required inspection envelope, metres; defaults 0.5 × 0.5 with 0.02 margin |
| `acceptance.require_ground_routes` | Adds floor support, slope, step, chassis and junction checks; see [generation modes and retry limits](usage.md#ordinary-generation-or-robot-qualification) |
| `acceptance.repair_ground_routes` | Allow measured ramp and cross-slope repairs for simulation; default `false`, requires ground-route qualification |
| `geometry.ground_ramp_max_change_m` | Maximum vertical floor edit during grading; default 0.5 m, must be > 0 and ≤ 1 m |
| `acceptance.robot_length_m`, `robot_max_slope_deg`, `robot_max_step_m` | Reference chassis limits: 0.7 m, 20°, 0.10 m |
| `acceptance.robot_height_m`, `robot_ground_clearance_m` | Optional body height separate from passage height, and declared underbody gap above sampled floor; defaults to passage height and zero clearance. The safety margin inflates the chassis into that gap. |
| `acceptance.robot_support_spacing_m` | Floor support sampling spacing in metres; default 0.10 |

The supplied 4K recipes reuse [Poly Haven Dark Rock](https://polyhaven.com/a/dark_rock)
from the repository's LFS master textures. This is an appearance asset, not a
measured or body-specific lava-tube calibration. Color is sRGB; normal and
roughness maps are linear data. In portable GLB, roughness is packed into green
and metallic into blue. Native continuous shaders decode raw normal RGB themselves;
use the installers' import settings rather than automatic normal-map swizzling.

Robot checks are controlled by `acceptance.require_ground_routes`; an advanced
`geometry.ground_robot_length_m` alone does not enable them. Selecting the
`simulation` profile or a `simulation-*` preset does not enable them either.
The default qualification envelope is **0.7 m long × 0.5 m wide × 0.5 m high**,
with 0.02 m margin, 20° maximum slope and 0.10 m maximum step. Search never relaxes
these limits to obtain a passing cave.

Seed-attempt and time limits do not enlarge repair, memory or query budgets.
See [retry and stopping behavior](usage.md#retry-limits-and-stopping),
[replay and resume](usage.md#seed-history-replay-and-resume), and
[export budget overrides](usage.md#retry-an-export-without-regenerating) for
the corresponding commands. Changing a wall-clock limit affects when work stops,
not the deterministic seed sequence or repair design.

## Traversability maps

Maps are enabled for full generations. Add `[traversability]` only when overriding
the defaults below. These settings are independent of `acceptance`: they label
terrain without qualifying, repairing or rejecting a cave.

| Control | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Export terrain arrays, classification, separate measurement views and overviews |
| `resolution_m` | `0.25` | Square cell size in metres |
| `robot_length_m`, `robot_width_m`, `robot_height_m` | `0.7`, `0.5`, `0.5` | Explicit reference envelope in metres |
| `margin_m` | `0.02` | Additional footprint and headroom safety margin |
| `max_slope_deg` | `20.0` | Maximum fitted floor inclination |
| `max_step_m` | `0.10` | Maximum detrended step/roughness over the footprint |
| `max_cells` | `12000000` | Maximum cells in the shared world XY extent; allocation guard |

The cell size must be at most half the smaller footprint dimension; the footprint
radius may span at most 32 cells. Choose finer sampling for small robots and assess
resolution convergence. The default envelope is an example, not a claim about the
robot that will be used. `run.render_diagnostics = false` does not suppress map
deliverables; use `traversability.enabled = false` for that.

[Map fields, coordinate conventions and regenerating maps](traversability.md).

## Advanced configuration map

A standalone advanced file starts with `schema_version = 4` instead of
`recipe_version = 1`. It exposes the full experiment interface; ordinary use
only needs a compact recipe. Unsupported schemas and unknown keys are rejected.

| Table / controls | Role | Implementation reference |
|---|---|---|
| `world.gravity_m_s2`, `bulk_density_kg_m3`, strength/quality fields | Explicit physical scenario parameters | [world.py](../src/plume_advanced/world.py) |
| `flow_regime` | Supply, duration, inflation, distributary tendency and cooling; dimensionless | [world.py](../src/plume_advanced/world.py) |
| `host_field.grid`, `ranges`, `wave_ranges` | Domain size/resolution and seeded terrain/geology variation | [host_field.py](../src/plume_advanced/stages/host_field.py) |
| `host_field.routing_weights` | Relative influence of slope, cover, fracture, capacity and stability | [host_field.py](../src/plume_advanced/stages/host_field.py) |
| `network.topology`, `systems`, `interconnection` | Network style, independently growing systems, parallel persistence and interactions | [network_topology.py](../src/plume_advanced/stages/network_topology.py), [network_interconnected.py](../src/plume_advanced/stages/network_interconnected.py) |
| `network.emplacement_history`, `lobe_growth` | Formation phases, reuse, retirement, breakout and pool controls | [network.py](../src/plume_advanced/stages/network.py) |
| `network.quality.max_attempts`, `repair_passes` | Finite pre-mesh morphology search; repeatable candidate streams | [network_quality.py](../src/plume_advanced/stages/network_quality.py) |
| `section_field` | Sampling spacing, asymmetry, floor/roof shape, longitudinal variation and cover limits | [section_field.py](../src/plume_advanced/stages/section_field.py) |
| `floor_map`, `events` | Floor sampling, structural events and debris placement | [floor_map.py](../src/plume_advanced/stages/floor_map.py), [events.py](../src/plume_advanced/stages/events.py) |
| `geometry.storage_mode`, `chunk_size`, `max_dense_voxels` | Dense versus tiled volume storage and allocation controls | [geometry_types.py](../src/plume_advanced/stages/geometry_types.py) |
| `geometry.recovery_local_attempts`, `recovery_network_attempts` | Upstream local adjustments and replacement networks in the same host | [recovery.py](../src/plume_advanced/pipeline/recovery.py) |
| `geometry.resolution_refinement_attempts`, `resolution_max_allocated_voxels` | Explicit bounded grid refinement; never an unlimited memory retry | [resolution.py](../src/plume_advanced/pipeline/resolution.py) |
| `geometry.surface_*`, `cave_smoothing_iterations`, `cave_displacement_scale_m` | Geometric relief, smoothing and image displacement; reinspection required | [geometry.py](../src/plume_advanced/stages/geometry.py) |
| `geometry.collision_target_reduction`, `collision_max_error_m` | Collider reduction target and measured error budget | [collision.py](../src/plume_advanced/exporters/collision.py) |
| `export.max_visual_triangles`, `max_asset_bytes`, `visual_max_error_m` | Published asset cost and reduction error limits | [scene.py](../src/plume_advanced/exporters/scene.py) |

Defaults, types and units are documented next to the consuming dataclasses.
Some are derived from the body or acceptance policy, so a dataclass default alone
is not the effective run setting. Always inspect the resolved configuration.
`network.source_count` places inlet samples; `network.systems.count` is the number
of independently interacting systems. They are not interchangeable.

## Optional network detail

[detailed-network.toml](../config/detailed-network.toml) enables local refinement
after the regional graph passes inspection. Set `network.detail.enabled = true`
in an existing regional recipe, or start from this minimal configuration:

```toml
recipe_version = 1
preset = "short-multi"

[network.topology]
generation_mode = "regional_growth"

[network.detail]
enabled = true
```

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `false` | Refine an accepted regional network; requires quality inspection |
| `strength` | `0.5` | Local displacement/envelope strength, from 0 to 1; zero retains geometry |
| `feature_scale_m` | `24` | Base feature scale, 4–200 m; adjusted by passage supply and role, with a six-width floor before drawing feature lengths |
| `sampling_error_m` | `0.025` | Maximum per-coordinate and width interpolation error against the dense reference, 0.001–0.1 m |
| `maximum_samples` | `50000` | Refinement reference/output sample budget; exhaustion preserves the accepted geometry |

Existing sources, outlets and node approaches remain fixed. Local plan changes
are capped at `length-weighted mean width × strength`; width adjustments at
`40% × strength`. Layer relief is capped at
`min(1.5 m, 0.35 × mean width) × strength`. The example recipe uses `strength = 0.8`;
lower values reduce feature amplitude, while their catalogue and spacing stay fixed.
Feature lengths range from 0.85–3 times the effective scale, with quiet gaps of
0.45–3 times that scale. Extra burial occurs only at local roof weakening in
layered networks. These are proposal bounds: inspection can reduce
or reject an edit without changing the seed.

Compatible local overlaps can add shared junctions and subdivide existing routes.
Connecting touching envelopes can also adjust their local approaches, independently
of the initial variation displacement. Flow allocation and source lineage are then
rebuilt. Overpasses remain separate, and a new node never exempts remote overlap
from inspection. See [encounter rules](networks.md#optional-network-detail).

Short segments and protected approaches may remain unchanged. A passage may
also remain unchanged when no complete feature and its quiet gap fit. Single-layer
networks retain the host elevation convention; independent vertical relief
requires explicit layers.

This remains a **network-only** experiment. It does not add wall roughness,
cross-sections or robot qualification. See the [refinement architecture](networks.md#optional-network-detail)
and [paired evaluation](evaluation.md#paired-network-detail-evaluation).

## Regional network controls

Use [config/regional-network.toml](../config/regional-network.toml) with
`plume-network`. It inherits the 400 m multi-system host and selects
`network.topology.generation_mode = "regional_growth"`. The original presets
retain their existing generation modes.

The optional `[network.regional]` table controls the network experiment:

| Parameter | Default | Meaning |
|---|---:|---|
| `branch_growth` | `"detour"` | `"detour"` chooses a downstream destination; `"front"` grows locally and discovers possible merges after separation |
| `planning_step_m` | 6 | Coarse host-routing spacing in metres; independent of mesh resolution |
| `correlation_length_m` | 80 | Spatial scale of seeded routing preferences |
| `route_variation` | 0.6 | Preference strength; zero removes this random cost perturbation |
| `secondary_scale_weight` | 0 | Strength of an additional seeded cost field at one-quarter of the correlation length; 0 disables it, up to 1 |
| `branch_localization` | 0 | Concentrate branch starts in seeded local zones; 0 is uniform, 1 is strongly localized. Zone size follows `correlation_length_m` |
| `hierarchy_strength` | 0 | Unequal route capacities and bounded supply-dependent widths, up to 1. With the statistical width field disabled, 0 retains uniform width |
| `width_log_sigma` | 0 | Log-width field amplitude, from 0 to 1; 0 preserves the previous width model. Nonzero replaces the repeating two-wave modulation |
| `width_correlation_m` | 10 | Positive spatial correlation length of that field in metres; independent of routing correlation length |
| `blind_branch_fraction` | 0 | Detour mode only: probability of retaining a separated blind prefix. Front mode derives termination from its growth and travel budget |
| `source_stagger_m` | 0 | Seeded downstream offset of each inlet, from 0 to this distance; 0 keeps all sources on one cross-section |
| `source_lateral_jitter_m` | 0 | Maximum seeded lateral offset from each regularly spaced inlet, in metres, before snapping to the planning grid. Proposals preserve inlet order, host bounds and separate routing lanes; 0 preserves regular lateral spacing |
| `outlet_count` | 1 | Number of distinct downstream termini, from 1 to 8. Values above 1 enable multiple outlets in single-layer regional growth |
| `outlet_band_width_m` | 0 | Total lateral width of the downstream selection band. With one outlet, 0 keeps the axial endpoint; with multiple outlets, 0 derives a band from inlet spread and required terminal separation |
| `branches_per_km` | 8 | Detour mode: requested additions per kilometre. Front mode: mean Poisson birth opportunities per kilometre, capped by `maximum_branches` |
| `minimum_branch_length_m` | 70 | Minimum length of newly added passage |
| `maximum_branch_length_m` | 220 | Detour mode: target span before reconnecting (detours may be longer). Front mode: maximum coarse growth distance; a seeded travel budget may stop growth earlier |
| `attempts_per_branch` | 12 | Detour proposal budget per requested addition. Front mode tries at most `min(4, attempts_per_branch)` proposals per sampled birth |
| `maximum_branches` | 64 | Maximum requested additions or sampled births, depending on mode |
| `extra_connections` | 0 | Optional extra links after branch growth, from 0 to 8. Layered runs try a descending ramp first, then alternate with same-level connections. This is an upper bound, not a quota |
| `maximum_grid_cells` | 250000 | Planning-grid allocation limit; exceeding it fails explicitly |

`network.systems.count` and `source_spacing_widths` still set the source layout.
`minimum_independent_length_widths` reserves separate inlet corridors before
merging. Combine `source_stagger_m` and `source_lateral_jitter_m` for irregular
inlet positions in both plan directions. Lateral placement tries at most 64
layouts per network candidate, then reports failure through the bounded network
search. These controls place feeder inlets; their distribution is not calibrated
to volcanic vent locations. Large staggering also shortens some feeder routes
to the downstream terminal band.
Setting `branches_per_km = 0` disables optional branch births; set
`network.systems.require_split = false` if splits are not required. Continuations
needed to supply requested termini still run. Set `extra_connections = 0` as well
to disable optional extra links.

[multi-outlet-network.toml](../config/multi-outlet-network.toml) combines six
irregular inlets with three downstream termini on a 400 m, single-layer host.
Set `outlet_count = 1` to restore a common outlet. Host routing costs select the
terminal cells; lateral separation is at least three reference passage widths
or two planning steps, whichever is larger. If nearest-terminal feeder routes
leave a requested outlet unused, a sustained downstream fork supplies it.
Each missing terminus tests at most 64 fork sites before the normal bounded
candidate search reports failure. All requested termini must remain present
after repair; a smaller count is never silently accepted.

Termini are ends of the modelled network, **not automatically surface openings**.
All passages must belong to **one connected network**, while retaining every
requested terminus. Separated feeder groups are joined through bounded
host-routed forks and confluences after feeder geometry inspection. The search uses
at most 512 interior sites, 64 searches and 8,192 expanded heading states per
required connection. It retains the host constraints and checks departure and
receiving directions. If routing or subsequent geometry repair fails, the
candidate is rejected; disconnected output is not accepted.

Connection proposals also pass metric geometry inspection before being kept.
The repair preserves existing passages and tries another proposal if widths or
bends create a conflict. Its audit records component counts, accepted connections,
rejected geometry proposals and search work in the network quality report.
Multiple outlets require `network.quality.enabled = true`.

Multiple outlets are currently a single-layer option. For separate outlets on retained layers, use
`layers.preserve_layer_trunks` instead; combining the two options is rejected
explicitly.
Retained-layer routing builds a connected network through descending ramps and
continues each upper trunk to its own outlet. Layered generation also runs the
connectivity repair: separated groups can join through existing host-screened
ramps between adjacent levels. Every proposed connection must pass the actual
3D grade, junction continuity and rock-separation checks. Overlapping XY
projections alone do not create a junction. Existing connected networks need no
additional connection, and the same finite search limits apply across layers.

[earth-survey-network.toml](../config/earth-survey-network.toml) is an optional
400 m, three-source **Earth width scenario**. Its base radius, log-width amplitude
and correlation length come from the [survey fitting workflow](evaluation.md#earth-network-width-calibration).
The recipe keeps layers and additional detail off so those uncalibrated effects
are not confused with the width fit. Multi-source merging and splitting remain
enabled. Width bounds and gradient checks still apply.
[earth-survey-full.toml](../config/earth-survey-full.toml) carries the same network
settings into a full single-layer run with sections, textured meshes and no rocks.
It uses a 0.15 m voxel grid and requires texture validation; it does not require
robot qualification. Additional network detail remains network-only; the full
layered recipe is described below.
Its source layout, branch frequency, host field and layer geometry are not fitted.
Do not transfer its metre-scale parameters to Mars or the Moon as a validated model.
The regional mode defaults to one layer and a shared downstream outlet.
The varied-network recipe uses a 30 m source stagger and a 160 m exit band.
Outlet selection uses the host routing graph, including its feasibility checks
and seeded preference field. Each retained layer selects its own exit at its
configured downstream extent. This is a routing rule, not a pressure solution.
`network.quality.max_attempts` and `repair_passes` bound candidate
search and repair. Accepted graphs record actual branches and rejected proposals
in their provenance; requested density may not be attainable on the supplied host.
Layered feeders and front-mode feeders use both existing repair allowances (up to
`2 * repair_passes`) before branch growth, so a successfully repaired feeder can
continue. An incomplete feeder is rejected without another outer repair cycle.
`feeder_repair_history` records the checks before and after each repair.

For a 3 km experiment, change `preset = "short-multi"` to `preset = "long-multi"`
in a copy of the recipe. This also supplies the larger host; changing only the
network length does not resize its substrate. Keep metre-based routing spacing
when increasing extent, and inspect the planning-cell budget.
Long runs also write `network_windows.png`, with 600 m views for closer inspection.

### Optional layers

Add this table to a regional recipe, or use
[regional-multilayer.toml](../config/regional-multilayer.toml):

```toml
recipe_version = 1
preset = "short-multi"

[network.topology]
generation_mode = "regional_growth"

[network.layers]
enabled = true
count = 2
spacing_m = 12.0
```

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `false` | Explicit opt-in; when false, the existing one-layer routing is unchanged |
| `preserve_layer_trunks` | `false` | Continue each upper layer to its own downstream outlet after a descending fork; distribute branch proposals across levels |
| `minimum_extent_fraction` | 1 | Upper levels end at seeded distances between this fraction and the full route extent; 0.5–1. The deepest level retains the full target |
| `spacing_variation` | 0 | Seeded differences in adjacent-level spacing; 0–1. The minimum rock gap is always retained |
| `count` | 2 | Requested levels, from 2 to 4; needs at least this many total `network.systems.count` sources |
| `spacing_m` | 12 | Vertical centreline spacing between adjacent levels |
| `passage_height_m` | 3 | Reference passage height for network clearance screening |
| `minimum_rock_m` | 3 | Required roof, basal and inter-passage rock thickness in the reference envelope |
| `maximum_connection_grade` | 0.25 | Maximum absolute vertical rise/run after smoothing; 0.25 is 25% grade, about 14° |
| `connection_opportunities_per_km` | 6 | Candidate ramp-bank density; minimum planning coverage is retained for each level, and only selected feasible connections enter the network |
| `connection_variation` | 0 | Seeded variation in ramp length and lateral destination, from 0 to 1; every candidate still passes the same host and grade checks |

Spacing must accommodate height plus rock thickness. Level 1 starts at a depth
of `minimum_rock_m + passage_height_m / 2`; subsequent levels add `spacing_m` by default. With `spacing_variation`, each gap
varies by at most 40% of that strength, bounded below by height plus minimum rock.
Larger stacks or gentler ramps need longer routes and sufficient host thickness.
Sources are distributed over the levels rather than multiplied by their count.
The original `emplacement_history.stacked_lobe_fraction` stays zero; this is a
separate regional feature. Turning `enabled` back to `false`
restores the one-layer model without requiring other settings to be removed.

[branching-layers-full.toml](../config/branching-layers-full.toml) runs all stages
with three layers, six sources, textures and no rocks. It starts with a 0.25 m
tiled grid and exports all supported application packages. The `research`
acceptance profile requires collision, textures and export budgets explicitly,
without imposing a robot body or ground-route requirement. Mesh validity and
network/section checks still apply. Narrow passages may be under-resolved; check
`section_resolution_report.json` before using the result for quantitative work.

Layered sections retain network depth and stay within `passage_height_m` and the
local network width. `section_field.minimum_roof_thickness` must also fit the
accepted layer layout (3 m in this recipe). Section depth preferences do not
relocate the layers. A contradictory roof requirement is rejected by section
inspection. `network.detail.enabled` must remain false for full generation.

For a denser experiment, use [complex-network.toml](../config/complex-network.toml).
It provides an 800 m downstream target on a wider host, three sources, three
optional layers with their own outlets, 60–300 m bypass targets and 16 requested
branch additions. Set `network.layers.enabled = false` to compare the same
settings in one layer. Requested branches are a budget, not a guarantee: only
screened additions are retained, and `summary.json` / `campaign.json` record
the number actually accepted. These recipes run with `plume-network` only.

For localized complexity, use [varied-network.toml](../config/varied-network.toml).
It keeps the 800 m host and three sources, uses `branch_growth = "front"`, and
samples births with mean 12 (capped at 24). It combines localized opportunities,
unequal route capacities and optional layers with different downstream extents.
A front follows local host cost, heading and downstream potential. After it has
grown independently, it may encounter and merge into a compatible nearby passage;
it can also stop without merging or give rise to later branches. Birth count is
not a required final branch count. `growth_events` records each decision, including
when capture was first considered, `capture_searches` and `capture_expanded_states`.
Capture uses at most four 256-state searches per proposal; it may complete beyond
the sampled lifetime, but never beyond `maximum_branch_length_m` in coarse distance.
New same-level routes also undergo bounded cost-field relaxation before fitting:
fixed junctions, at most two passage widths of displacement, and 40 optimizer
iterations. No extra configuration is required. `route_relaxation` records the
objective change and work for these proposals; its `accepted` field means the
proposal passed the preliminary host screen, not full-network qualification.
Rejected branch events retain their relaxation audit too.
`local_repair_history` records bounded bend
and clearance searches, their work counts and whether full inspection accepted them.
`local_search_expanded_states` counts accepted and rejected search work against
`local_search_state_limit` (`2400 × repair_passes` per candidate). Exhaustion
prevents more search expansion, while cheap direct fits remain available.
Set `network.layers.enabled = false` for a single-layer comparison. The ordinary
full-generation presets are unaffected. All these controls are procedural
approximations, not fitted geological distributions.

For a denser layered scenario, [branching-layers.toml](../config/branching-layers.toml)
keeps the same 800 m host with six staggered inlets across three optional layers.
It retains one outlet per layer and uses a smaller passage scale to leave space
for independent growth. The original recipes keep their settings.

| Control | Value in this recipe | Purpose |
|---|---:|---|
| `network.systems.count` | 6 | Two sources per layer |
| `network.regional.source_stagger_m` / `source_lateral_jitter_m` | 80 / 10 m | Irregular inlet positions |
| `network.base_passage_radius` | About 3.67 m | Reference passage scale before local width modulation |
| `network.regional.width_log_sigma` / `width_correlation_m` | 0.2 / 12 m | Spatially coherent width variation |
| `network.regional.correlation_length_m` / `secondary_scale_weight` | 110 m / 0.35 | Route variation at two scales |
| `network.regional.branches_per_km` | 20 | Mean branching opportunity density, capped at 24 births |
| `network.regional.extra_connections` | 2 | At most one additional descending ramp and one same-level link |

Branch lengths remain 60–300 m, and the normal bend, clearance, host and ramp
checks remain enabled. More sources do not guarantee more loops: retained side
branches may have blind ends. This is an exploratory morphology recipe, not a
calibration of multi-layer lava-tube geology.

Extra connections are tried on the already grown network. They must join
different passage chains, keep away from inlets, outlets and existing junctions,
and remain separated from other extra-link endpoints by at least the larger of
six reference widths or the minimum branch length. The original host constraints,
grade limits and rock gap still apply. A valid cave remains usable when an extra
link cannot be fitted: it is skipped without a seed retry.

Each optional slot permits at most 64 searches and 8,192 expanded heading states.
`system_provenance.extra_connections` records accepted links, rejected checks,
construction errors, skipped slots and work counts. Setting the value to zero
preserves the earlier generation path. The option requires inspected
`regional_growth`; it never silently bypasses network quality checks.

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

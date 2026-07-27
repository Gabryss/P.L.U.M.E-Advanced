# PLUME-Advanced

`PLUME-Advanced` is a staged procedural pyroduct / lava-tube prototype.

The major architecture upgrade is active on
`feature/major-procedural-upgrade`. Its implementation roadmap, acceptance
criteria, and dependency policy are in
[`docs/MAJOR_UPGRADE_PLAN.md`](docs/MAJOR_UPGRADE_PLAN.md). Blender is not a
runtime dependency: generation and export use Python libraries and open
interchange formats.

The current implementation focuses on the full inspectable cave-shape pipeline:
build a readable terrain substrate, derive a cave-network skeleton, generate a
geometry-ready section field around that skeleton, stamp that network into a
voxel density grid, and mesh the carved volume. Later surface/texturing stages
are still intentionally incomplete.

## Pipeline

| Stage | Status | Purpose | Current Output |
|---|---|---|---|
| A. Host Field | Implemented | Build terrain and structural layers | `outputs/stage_a_host_field.png` |
| B. Cave Network | Implemented | Generate a host-driven braided cave-network skeleton | `outputs/stage_b_cave_network.png` |
| C. Section Field | Implemented | Build adaptive lava-tube cross-sections around the skeleton | `outputs/stage_c_section_field.png` |
| D. Geometry | Implemented | Stamp the cave network into a voxel grid, polygonize it, and build a globally welded render mesh | `outputs/stage_d_geometry.png`, target export package |
| E. Geological Events | Implemented | Placed mesh-stage rocks, boulders, collapse debris, choke points, and infill | `outputs/stage_e_geological_events.png` |
| F. Surface Detail / Texturing | Placeholder | Wall detail, floor variation, material masks | TODO |

## Current Outputs

### Stage A: Host Field

![Stage A Host Field](outputs/stage_a_host_field.png)

Stage A produces the terrain substrate and the main scalar layers used by later
stages: elevation, slope, cover thickness, roof competence, and growth cost.

### Stage B: Cave Network

![Stage B Cave Network](outputs/stage_b_cave_network.png)

Stage B generates the current default output: a host-driven braided cave-network
skeleton with split/rejoin structure, islands, chamber-like expansions, and
segment metadata for later geometry stages.

### Stage C: Section Field

![Stage C Section Field](outputs/stage_c_section_field.png)

Stage C generates geometry-ready cross-section samples along the network:
adaptive sample spacing, underground centerline placement, 3D local frames,
lava-tube profile controls, and junction-aware blending through split/merge
regions.

### Stage D: Geometry

![Stage D Geometry Diagnostics](outputs/stage_d_geometry.png)

![Stage D Chunk Diagnostics](outputs/stage_d_geometry_chunks.png)

![Stage D Geometry Presentation](outputs/stage_d_geometry_presentation.png)

Stage D converts the Stage-C samples into a carved density field. It stamps
capsule tunnels and widened junction/chamber regions into a voxel grid, then
meshes the zero-density isosurface in chunks. Stage-E rocks, boulders, and
larger geological event meshes are placed as separate mesh objects. The
diagnostic render focuses on footprint alignment, longitudinal continuity,
chunk coverage, and section slices. The chunk render isolates chunk coverage,
face-count distribution, and Y/Z chunk spans. The presentation render gives a
cleaner plan/mesh preview.

### Stage E: Geological Events

![Stage E Geological Events](outputs/stage_e_geological_events.png)

Stage E places deterministic mesh-stage events from the Stage-C section field:
scattered rocks, larger boulders, collapse debris, choke points, and floor
infill. These events are generated as separate placed meshes before texturing.
Stage F can later use their `material_hint` metadata for texture and material
masks.

## How It Works

### Stage A: Host Field

Implemented in `src/stages/host_field.py`.

The terrain is not a full volcanic edifice. It is a seed-driven,
pyroduct-oriented host slab. The project config defines ranges for the broad
geological controls, and `procedural_seed` resolves those ranges into a
concrete host field for the run:

- a sampled source region and dominant flow direction
- a sampled large-scale downhill grade
- a sampled broad host corridor
- a sampled fracture corridor and roof-competence structure
- a sampled set of low-frequency directional terrain waves

Conceptually:

```text
terrain = large-scale directional grade
        - corridor depression
        + low-frequency waves
```

Once the terrain exists, the remaining host-field layers are either derived from
it or generated from the same seed-resolved structural controls.

| Layer | Built From | Used Now | Intended Later Use |
|---|---|---|---|
| `elevation` | directional grade + corridor + waves | terrain profile, downhill direction | surface interaction |
| `gradient_x`, `gradient_y` | `np.gradient(elevation)` | downhill steering for graph growth | path-cost and event logic |
| `slope_degrees` | gradient magnitude | cover thickness, diagnostics, growth cost | gating unstable or unrealistic zones |
| `cover_thickness` | base thickness + relief bonus - slope penalty | growth cost, graph diagnostics | collapse and surface-stability rules |
| `roof_competence` | structural bands + fracture corridor + edge weathering | growth cost, graph diagnostics | ceiling roughness, collapse, material masks |
| `growth_cost` | weighted slope, cover, and competence penalties | visualization, summaries | future explicit path scoring |

The `HostField` API currently exposes:

- `sample(x, y)`: bilinear sample of all fields
- `contains(x, y, margin=0.0)`: map bounds check
- `downhill_direction(x, y, fallback_angle_degrees=None)`: normalized downhill vector

### Stage B: Cave Network

Implemented in `src/stages/network.py`.

Stage B now builds the cave skeleton directly instead of starting from a single
trunk. Its branch/split/merge grammar is seed-sampled from
`[network.braid_grammar]`, then fitted to the generated host field. The
generator:

- uses the host field and the configured `procedural_seed`
- traces a downhill backbone with host-guided branch motifs
- samples localized asymmetric braid zones, branch counts, lateral offsets, ladders, and underpasses
- supports `backbone`, `island_bypass`, `chamber_braid`, `ladder`, `spur`, and `underpass` segment kinds
- records graph metadata such as `z_level`, `merge_behavior`, `crossing_group_id`, `island_id`, and `chamber_id`
- clusters morphologically meaningful split/merge/crossing regions into explicit junction objects
- derives occupancy and graph summaries from the resulting network

The Stage B visualization includes a longitudinal diagnostics panel. It reads
left to right along the main flow direction: the filled step trace shows how
many parallel channels are present at each slice, the green line/band shows
mean and min/max tube width, and vertical markers indicate junction regions.

### Stage C: Section Field

Implemented in `src/stages/section_field.py`.

Stage C wraps a lava-tube-shaped section field around the Stage-B skeleton.
The generator:

- resamples each segment adaptively based on curvature, width gradient, and junction proximity
- moves the section centerline below the host surface using cover thickness and roof-thickness heuristics
- builds geometry-ready local frames (`tangent`, `normal`, `binormal`)
- derives smooth section controls such as width, height, floor flattening, roof arch, and lateral skew
- uses explicit Stage-B junction regions to blend split/merge morphology without hard jumps at nodes
- records per-sample junction influences for later split/merge volume construction
- stores closed local 2D section contours and scalar profile controls for the voxel geometry stage

### Stage D: Geometry

Implemented in `src/stages/geometry.py`.

Stage D turns the section field into a voxel-first mesh. The generator currently:

- stamps Stage C section-profile densities along every segment, with higher density meaning carved space
- adds controlled seeded wall roughness near the isosurface for less capsule-like walls
- stamps widened junction/chamber regions into the same field
- processes the field in 3D chunks
- polygonizes chunk meshes with `scikit-image`
- validates/exports the assembled OBJ-ready mesh with `trimesh`

## Configuration

The single source of truth is:

```text
config/project.toml
```

It is loaded by `src/config.py`, which converts TOML sections into dataclass
configs for the generators.

Execution flow:

1. load `config/project.toml`
2. resolve the celestial body, rock material, named stage seeds, run mode, and export target
3. build `HostFieldConfig`, `CaveNetworkConfig`, `SectionFieldConfig`, `GeologicalEventConfig`, and `GeometryConfig`
4. generate and render stages A–E
5. build the globally assembled cave geometry
6. export through the selected target adapter
7. write both source and resolved configuration metadata in `outputs/`

`procedural_seed` is expanded into stable named seeds for host, network,
sections, events, geometry, surface, and export. Consequently, disabling
events cannot perturb the network.

### World, run, and export config

| Section | Purpose |
|---|---|
| `[world]` | Select `earth`, `mars`, or `moon`, plus a rock-material preset and optional physical overrides |
| `[run]` | Select preview/standard/production quality and development extent caps |
| `[export]` | Select neutral, Blender, UE5, Unity, Gazebo, or Omniverse output intent |

Body presets supply gravity, atmosphere/erosion context, default rock material,
passage and room caps, route-length guidance, and production resolution
guidance. Earth defaults to 10 m passages and 20 m rooms; Moon defaults to
100 m passages and 200 m rooms. Explicit TOML values can override a preset.
All internal geometry remains right-handed, Z-up, and metre-based.

Development mode shortens the host extent and braid count but does not shrink
the selected body's passages:

```toml
[world]
body = "moon"
material = "mare_basalt"

[run]
dev_mode = true
dev_max_route_length_m = 1500.0
dev_max_braid_zones = 2

[events]
enabled = false
enabled_kinds = []

[export]
target = "omniverse"
format = "usd"
```

### Host Field Config

| Key Group | Purpose |
|---|---|
| `[host_field.grid]` | map dimensions and sample resolution |
| `[host_field.ranges]` | `[min, max]` ranges for sampled source, terrain, corridor, cover, roof, and fracture controls |
| `[host_field.wave_ranges]` | sampled wave count and `[min, max]` ranges for low-frequency terrain deformation layers |

The runtime `HostFieldConfig` remains concrete: `src/config.py` samples the
range blocks with `procedural_seed` before stage generation starts. Fixed
legacy values and `[[host_field.waves]]` are still accepted for targeted tests
or hand-authored scenarios, but the default project config is range-driven.

### Network Config

| Key Group | Purpose |
|---|---|
| `source_*`, `sink_margin`, `trace_max_steps` | control network source/sink setup and trace extent |
| `growth_cost_weight`, `roof_weight`, `cover_weight`, `slope_penalty_weight`, `corridor_weight` | bias path selection through the host field |
| `occupancy_smoothing_passes` | clean occupancy artifacts |
| `chamber_*`, `base_passage_radius` | control chamber detection and occupancy painting |
| `spur_*`, `channel_count_samples` | control terminal spur generation and braid sampling |
| `[network.braid_grammar]` | `[min, max]` ranges and probabilities for sampled braid zones, branch counts, offsets, ladders, and underpasses |

### Section Field Config

| Key Group | Purpose |
|---|---|
| `base_height_ratio`, `minimum_height_ratio`, `maximum_height_ratio` | control the default lava-tube width/height relationship |
| `minimum_sample_spacing`, `maximum_sample_spacing` | bound adaptive section-sample spacing |
| `curvature_spacing_weight`, `width_gradient_spacing_weight`, `junction_spacing_weight` | make sampling denser where the skeleton or morphology changes faster |
| `profile_resolution` | control local section contour resolution |
| `floor_flatness_*`, `roof_arch_*`, `lateral_skew_amplitude` | shape the lava-tube profile |
| `centerline_wobble_*` | add bounded centerline meander to avoid unnaturally straight tube runs |
| `junction_*_gain` | control how strongly junction regions widen or stay tight through splits/merges |

### Geometry Config

| Key Group | Purpose |
|---|---|
| `voxel_size`, `density_margin` | control field resolution and padding around the stamped cave network |
| `chunk_size` | controls how much of the density grid is polygonized at once |
| `iso_level` | defines the density threshold used for the cave wall surface |
| `tunnel_radius_scale`, `junction_radius_scale`, `chamber_radius_scale` | control how section samples widen while stamping |
| `use_section_profiles` | use Stage C's closed cross-section polygons instead of circular capsule stamps |
| `wall_roughness_*` | add seeded near-wall roughness before marching cubes |
| `junction_irregularity_*` | deform junction/chamber volumes so they blend less like perfect ellipsoids |
| `minimum_radius`, `weld_tolerance` | keep thin passages meshable and weld repeated isosurface vertices |

### Event Config

| Key Group | Purpose |
|---|---|
| `rock_density_per_100m`, `boulder_density_per_100m` | control loose debris density along sampled cave length |
| `geological_event_density_per_100m` | control larger collapse/choke/infill event density |
| `collapse_event_fraction`, `choke_event_fraction`, `infill_event_fraction` | split larger geological events by type |
| `*_radius_range` | bound event sizes before they are clipped to local tube dimensions |
| `minimum_event_spacing` | keep major event centers from clustering too tightly |
| `max_lateral_floor_fraction` | keep floor debris inside the local tube profile |
| `mesh_latitude_segments`, `mesh_longitude_segments` | control generated event mesh resolution |
| `enabled`, `enabled_kinds` | export an empty tube or enable selected event families |
| `use_rocky_meshes`, `strict_optional_provider` | control the optional Rocky provider without importing it when disabled |

## Project Layout

- `config/`: project configuration
- `docs/MAJOR_UPGRADE_PLAN.md`: phased architecture and release gates
- `scripts/`: stage entrypoints
- `src/config.py`: TOML loader
- `src/world.py`: celestial body, material, run, export, and seed profiles
- `src/exporters/`: Blender-independent target adapters
- `src/stages/`: stage implementations
- `src/visualization/`: stage visualizations
- `outputs/`: generated images
- `tests/`: smoke tests

## Run

Install dependencies:

```bash
uv sync --group dev
```

Or with standard Python tooling:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest pillow
```

The project intentionally uses external libraries where they improve the core
algorithm or developer workflow:

- `numpy`: scalar fields, density grids, and vectorized stamping
- `scipy`: voxel connected-component labeling
- `scikit-image`: marching-cubes isosurface extraction
- `trimesh`: mesh validation metadata and OBJ export
- `matplotlib`: stage visualizations
- `rich`: CLI progress bars

Generate the current cave network with the single entrypoint:

```bash
.venv/bin/python scripts/generate_cave.py
```

The generator prints progress bars for configuration loading, stages A-C,
Stage-E geological event placement, detailed Stage-D voxel/mesh generation,
Stage-D visualization, and the selected target export.
Geometry progress reports stamp counts, chunk meshing status, assembled face
counts, and final component counts.

That one command produces:

- `outputs/stage_a_host_field.png`
- `outputs/stage_b_cave_network.png`
- `outputs/stage_c_section_field.png`
- `outputs/stage_e_geological_events.png`
- `outputs/stage_d_geometry.png`
- `outputs/stage_d_geometry_chunks.png`
- `outputs/stage_d_geometry_presentation.png`
- `outputs/resolved_project_config.json`
- `outputs/export_<target>/...`

Optional:

```bash
.venv/bin/python scripts/generate_cave.py \
  --config config/project.toml \
  --output outputs/stage_b_cave_network.png \
  --host-output outputs/stage_a_host_field.png \
  --section-output outputs/stage_c_section_field.png \
  --event-output outputs/stage_e_geological_events.png \
  --geometry-output outputs/stage_d_geometry.png \
  --geometry-chunk-output outputs/stage_d_geometry_chunks.png \
  --geometry-presentation-output outputs/stage_d_geometry_presentation.png \
  --geometry-glb-output outputs/stage_d_geometry.glb
```

Optional host-field debug render:

```bash
.venv/bin/python scripts/render_host_field.py
```

Both scripts read `config/project.toml` by default.

## Upgrade roadmap

The authoritative roadmap is
[`docs/MAJOR_UPGRADE_PLAN.md`](docs/MAJOR_UPGRADE_PLAN.md). It covers mesh and
export correctness, SDF-grounded events, a causal host model, a flux-conserving
multi-source network, sparse tiled geometry, procedural PBR surfaces, target
packages, and validation/performance gates.

## Existing stages and deferred work

These are placeholders for the next implementation passes.

### Stage D: Geometry

Implemented as a voxel-density meshing pass.

What exists now:

- one density grid containing the full stamped tunnel network
- chunked isosurface generation for review/progress visualization
- diagnostic, chunk-focused, and presentation Stage-D render artifacts
- OBJ export of the cave mesh plus separate placed Stage-E mesh objects

Still deferred:

- watertight versus blend-ready export modes
- seam-aware UV atlas generation and production surface cleanup
- higher-quality event-specific cleanup for rocks, boulders, collapse, choke points, and infill

### Stage E: Geological Events

Implemented as a mesh-stage event pass.

What exists now:

- deterministic placement from Stage-C samples
- scattered rocks and larger boulders
- collapse debris, choke points, and floor infill
- separate placed meshes for every Stage-E event
- event visualization and summary metrics
- voxel integration before Stage-D meshing

Still deferred:

- richer event-specific mesh cleanup controls
- explicit gameplay/navigation metadata
- final material-mask export for Stage F

### Stage F: Surface Detail / Texturing

Placeholder.

Planned role:

- add wall and floor detail
- derive texturing masks from competence, events, and geometry
- avoid using detail noise to define topology

## Summary

The current project state is intentionally narrow:

- Stage A builds the terrain and structural substrate
- Stage B builds the current braided cave-network skeleton
- Stage C builds adaptive lava-tube cross-sections around that skeleton
- Stage E places geological mesh events including rocks and boulders
- Stage D stamps the network into a voxel density field, meshes the isosurface, and exports the placed Stage-E meshes alongside it
- Stage F and watertight/detail work remain for the next passes

That keeps the pipeline inspectable while still leaving a clear path toward the
final pyroduct mesh and texture stages.

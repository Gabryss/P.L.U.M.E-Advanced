# PLUME-Advanced

`PLUME-Advanced` is a staged procedural pyroduct / lava-tube prototype.

The major architecture upgrade is active on
`feature/major-procedural-upgrade`. Its implementation roadmap, acceptance
criteria, and dependency policy are in
[`docs/MAJOR_UPGRADE_PLAN.md`](docs/MAJOR_UPGRADE_PLAN.md). Blender is the
default target, but is not a runtime dependency: generation and export use
Python libraries and open interchange formats.

The current implementation focuses on the full inspectable cave-shape pipeline:
build a readable terrain substrate, derive a cave-network skeleton, generate a
geometry-ready section field around that skeleton, stamp that network into a
voxel density grid, and mesh the carved volume. Later surface/texturing stages
are still intentionally incomplete.

## Pipeline

| Stage | Status | Purpose | Current Output |
|---|---|---|---|
| A. Host Field | Implemented | Build terrain and structural layers | `outputs/stage_a_host_field.png` |
| B. Cave Network | Implemented | Generate a host-driven, body-scaled braided cave-network skeleton | `outputs/stage_b_cave_network.png`, `outputs/stage_b_network_report.json` |
| C. Section Field | Implemented | Build adaptive lava-tube cross-sections around the skeleton | `outputs/stage_c_section_field.png` |
| D. Geometry | Implemented | Stamp the cave network into a voxel grid, polygonize it, and build a globally welded render mesh | `outputs/stage_d_geometry.png`, target export package |
| E. Geological Events | Implemented | Ground rocks/boulders on the base cave and apply collapse/choke/infill as structural modifiers | `outputs/stage_e_geological_events.png` |
| F. Surface Detail / Texturing | Placeholder | Wall detail, floor variation, material masks | TODO |

## Current Outputs

### Stage A: Host Field

![Stage A Host Field](docs/figures/celestial_bodies/earth/stage_a_host_field.png)

Stage A produces the terrain substrate and the main scalar layers used by later
stages: elevation, slope, cover thickness, roof competence, and growth cost.

### Stage B: Cave Network

![Stage B Cave Network](docs/figures/celestial_bodies/earth/stage_b_cave_network.png)

Stage B generates a host-driven, multi-source braided cave network. Source
feeders merge into the main route, flux is conserved through graph
splits/merges, and every point carries flux, temperature, and lava age.

### Stage C: Section Field

![Stage C Section Field](docs/figures/celestial_bodies/earth/stage_c_section_field.png)

Stage C generates geometry-ready cross-section samples along the network:
adaptive sample spacing, underground centerline placement, 3D local frames,
lava-tube profile controls, and junction-aware blending through split/merge
regions.

### Stage D: Geometry

![Stage D Geometry Diagnostics](docs/figures/celestial_bodies/earth/stage_d_geometry.png)

![Stage D Chunk Diagnostics](docs/figures/celestial_bodies/earth/stage_d_geometry_chunks.png)

![Stage D Geometry Presentation](docs/figures/celestial_bodies/earth/stage_d_geometry_presentation.png)

Stage D converts the Stage-C samples into a carved density field. It stamps
capsule tunnels and widened junction/chamber regions into a voxel grid, then
meshes the zero-density isosurface in chunks. Stage-E rocks and boulders remain
separate editable props; collapse, choke, and infill events modify the cave
density before the final isosurface is generated. The
diagnostic render focuses on footprint alignment, longitudinal continuity,
chunk coverage, and section slices. The chunk render isolates chunk coverage,
face-count distribution, voxel size, minimum passage sampling, and Y/Z chunk
spans. The presentation render gives a cleaner plan/mesh preview.

### Stage E: Geological Events

![Stage E Geological Events](docs/figures/celestial_bodies/earth/stage_e_geological_events.png)

Stage E places deterministic events from the Stage-C section field after the
base cave density exists. Rocks and boulders raycast to the actual floor,
align with the inward surface normal, and embed slightly to avoid floating.
Collapse, choke, and infill events are smooth solid SDF intersections, so they
change the final render and collision topology instead of adding decorative
ellipsoids. A configurable share of loose debris is sampled from floor cells
around collapse regions, with relaxed intra-cluster spacing and explicit
parent-collapse metadata.

## Celestial-body comparison

The same seed and base configuration produce materially different underground
environments when the physical world preset changes:

| View | Earth | Mars | Moon |
|---|---|---|---|
| Floor atlas | ![Earth floor atlas](docs/figures/celestial_bodies/earth/stage_c_floor_map.png) | ![Mars floor atlas](docs/figures/celestial_bodies/mars/stage_c_floor_map.png) | ![Moon floor atlas](docs/figures/celestial_bodies/moon/stage_c_floor_map.png) |
| Geometry diagnostics | ![Earth geometry diagnostics](docs/figures/celestial_bodies/earth/stage_d_geometry.png) | ![Mars geometry diagnostics](docs/figures/celestial_bodies/mars/stage_d_geometry.png) | ![Moon geometry diagnostics](docs/figures/celestial_bodies/moon/stage_d_geometry.png) |
| Chunk diagnostics | ![Earth chunk diagnostics](docs/figures/celestial_bodies/earth/stage_d_geometry_chunks.png) | ![Mars chunk diagnostics](docs/figures/celestial_bodies/mars/stage_d_geometry_chunks.png) | ![Moon chunk diagnostics](docs/figures/celestial_bodies/moon/stage_d_geometry_chunks.png) |
| Geometry presentation | ![Earth generated cave](docs/figures/celestial_bodies/earth/stage_d_geometry_presentation.png) | ![Mars generated cave](docs/figures/celestial_bodies/mars/stage_d_geometry_presentation.png) | ![Moon generated cave](docs/figures/celestial_bodies/moon/stage_d_geometry_presentation.png) |
| Geological events | ![Earth geological events](docs/figures/celestial_bodies/earth/stage_e_geological_events.png) | ![Mars geological events](docs/figures/celestial_bodies/mars/stage_e_geological_events.png) | ![Moon geological events](docs/figures/celestial_bodies/moon/stage_e_geological_events.png) |

The complete eight-figure pipeline comparison for every body is in
[`docs/CELESTIAL_BODY_GALLERY.md`](docs/CELESTIAL_BODY_GALLERY.md).

Body selection changes more than passage radius. It scales the host
correlation lengths, fracture structure, vertical relief, route target,
branch persistence, section spacing, and floor-map resolution. Production
route guidance is 5 km for Earth, 15 km for Mars, and 30 km for the Moon;
development mode creates shorter representative versions while preserving
each body's passage dimensions.

## How It Works

### Stage A: Host Field

Implemented in `src/plume_advanced/stages/host_field.py`.

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

Once the terrain exists, independent low-frequency process fields are combined
through an explicit routing formula. The network consumes that formula once;
it no longer adds slope, cover, and competence again on top of an opaque cost.

| Layer | Built From | Used Now |
|---|---|---|---|
| `elevation`, gradients, slope | directional grade + corridor + waves | downhill steering and routing |
| `emplacement_thickness`, `cover_thickness` | flow corridor, lobe variation, erosion, deposits | routing and roof demand |
| `lithology_quality`, `fracture_intensity`, `cooling_index` | material profile and independent structural/thermal bands | competence, capacity, stability |
| `deposit_thickness`, `erosion_index` | body/material weathering proxies | cover and flow capacity |
| `flow_capacity` | corridor, emplacement, fractures, deposits | routing |
| `roof_stability` | competence plus density, gravity, span, cover, strength | routing and diagnostics |
| `routing_cost` (`growth_cost` compatibility alias) | named slope/cover/fracture/capacity/stability contributions | network routing |

The `HostField` API currently exposes:

- `sample(x, y)`: bilinear sample of all fields
- `contains(x, y, margin=0.0)`: map bounds check
- `downhill_direction(x, y, fallback_angle_degrees=None)`: normalized downhill vector
- `routing_influence_summary()`: variance, contribution, and routing-cost correlation per term

### Stage C2: Cave-Floor Atlas

After the base cave volume is stamped, PLUME raycasts a placement atlas for
events. Once collapse, choke, and infill modifiers have changed the volume,
the same stable cell addresses are relifted against the final cave. Blocked
cells are invalidated rather than projected through solid rock. Atlas
coordinates are
`(segment_id, distance_along_m, lateral_offset_m)`; every cell also stores its
exact world XYZ, surface normal, clearance, and vertical graph level.

This is deliberately richer than a top-down occupancy image: underpasses and
stacked branches can occupy the same world XY without becoming the same map
cell. Rocks and boulders are selected from atlas cells and therefore use
prevalidated floor contacts. Each run writes:

- `stage_c_floor_map.png`: final elevation, clearance, geology, chambers,
  termini, events, and intrinsic segment atlas
- `stage_c_floor_map.npz`: liftable cells plus occupancy, floor-height,
  clearance, geology, debris, and vertical-level-count rasters for procedural
  generation and external inspection
- `stage_c_floor_map.json`: schema, units, configuration, and summary

### Stage B: Cave Network

Implemented in `src/plume_advanced/stages/network.py`.

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
- keeps parallel centrelines separated in both host-corridor units and full passage widths
- lengthens braid persistence with body spatial scale instead of only inflating passage radius
- reserves chambers for explicit junction/confluence semantics rather than painting high-flux blobs into occupancy
- derives occupancy and graph summaries from the resulting network

The Stage B visualization includes a longitudinal diagnostics panel. It reads
left to right along the main flow direction: the filled step trace shows how
many skeleton channels are present at each slice, the red dashed trace shows
how many remain visibly separate after passage widths are applied, the green
line/band shows mean and min/max tube width, and vertical markers indicate
junction regions. The same data, per-kind length/persistence statistics, and
the network summary are written to `stage_b_network_report.json`.

### Stage C: Section Field

Implemented in `src/plume_advanced/stages/section_field.py`.

Stage C wraps a lava-tube-shaped section field around the Stage-B skeleton.
The generator:

- resamples each segment adaptively based on curvature, width gradient, and junction proximity
- moves the section centerline below the host surface using cover thickness and roof-thickness heuristics
- builds geometry-ready local frames (`tangent`, `normal`, `binormal`)
- turns underpass levels into smooth, clearance-constrained vertical grades
- derives smooth section controls such as width, height, floor flattening, roof arch, and lateral skew
- uses explicit Stage-B junction regions to blend split/merge morphology without hard jumps at nodes
- records per-sample junction influences for later split/merge volume construction
- stores closed local 2D section contours and scalar profile controls for the voxel geometry stage

### Stage D: Geometry

Implemented in `src/plume_advanced/stages/geometry.py`.

Stage D turns the section field into a voxel-first mesh. The generator currently:

- stamps Stage C section-profile densities along every segment, with higher density meaning carved space
- adds controlled seeded wall roughness near the isosurface for less capsule-like walls
- stamps widened junction/chamber regions into the same field
- processes the field in 3D chunks
- switches from a dense field to overlapping sparse tiles when the configured
  dense-voxel budget would be exceeded
- polygonizes chunk meshes with `scikit-image`
- validates/exports the assembled OBJ-ready mesh with `trimesh`

## Configuration

The single source of truth is:

```text
config/project.toml
```

It is loaded by `src/plume_advanced/config.py`, which converts TOML sections into dataclass
configs for the generators.

Execution flow:

1. load `config/project.toml`
2. resolve the celestial body, independent flow regime, rock material, named stage seeds, run mode, and export target
3. build `HostFieldConfig`, `CaveNetworkConfig`, `SectionFieldConfig`, `GeologicalEventConfig`, and `GeometryConfig`
4. generate host, network, and section stages
5. stamp the base cave density
6. raycast a base floor atlas and place grounded props/events
7. apply structural modifiers and polygonize the final cave
8. relift and classify the final geological floor atlas
9. export through the selected target adapter
10. write source, resolved configuration, and floor-map metadata in `outputs/`

`procedural_seed` is expanded into stable named seeds for host, network,
sections, events, geometry, surface, and export. Consequently, disabling
events cannot perturb the network.

### World, run, and export config

| Section | Purpose |
|---|---|
| `[world]` | Select `earth`, `mars`, or `moon`, plus a rock-material preset and optional physical overrides |
| `[flow_regime]` | Control supply, duration, inflation, distributary tendency, and cooling independently of the selected body |
| `[run]` | Select preview/standard/production quality and development extent caps |
| `[export]` | Select Blender (default), neutral, UE5, Unity, Gazebo, or Omniverse output intent |

Body presets supply gravity, atmosphere/erosion context, default rock material,
passage and room caps, route-length guidance, and production resolution
guidance. Earth defaults to 10 m passages and 20 m rooms; Moon defaults to
100 m passages and 200 m rooms. Explicit TOML values can override a preset.
All internal geometry remains right-handed, Z-up, and metre-based.

The body says where the tube forms; the flow regime describes the eruption
that formed it. This separation avoids equating low gravity with one fixed
network shape:

```toml
[flow_regime]
supply_rate_scale = 1.0
duration_scale = 1.0
inflation = 0.50
distributary_tendency = 0.65
cooling_rate_scale = 1.0
```

Higher sustained supply relative to cooling expands host correlation lengths.
Longer duration increases the route target, inflation increases junction-room
widening, and distributary tendency controls lateral-branch abundance.

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
target = "blender"
format = "glb"
```

### Host Field Config

| Key Group | Purpose |
|---|---|
| `[host_field.grid]` | map dimensions and sample resolution |
| `[host_field.ranges]` | `[min, max]` ranges for sampled source, terrain, corridor, cover, roof, and fracture controls |
| `[host_field.wave_ranges]` | sampled wave count and `[min, max]` ranges for low-frequency terrain deformation layers |

The runtime `HostFieldConfig` remains concrete: `src/plume_advanced/config.py` samples the
range blocks with `procedural_seed` before stage generation starts. Fixed
legacy values and `[[host_field.waves]]` are still accepted for targeted tests
or hand-authored scenarios, but the default project config is range-driven.

### Network Config

| Key Group | Purpose |
|---|---|
| `source_*`, `sink_margin`, `trace_max_steps` | control network source/sink setup and trace extent |
| `growth_cost_weight`, `corridor_weight` | bias path selection using the explicit host routing cost and broad corridor prior |
| `occupancy_smoothing_passes` | clean occupancy artifacts while retaining the graph skeleton |
| `minimum_branch_offset_widths` | keep parallel centrelines visibly separate after their physical widths are applied |
| `chamber_*`, `base_passage_radius`, `paint_flux_chambers` | control explicit junction rooms and optionally enable legacy flux-blob painting |
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

### Floor Map Config

| Key | Purpose |
|---|---|
| `lateral_spacing_m` | spacing between sampled floor lanes |
| `maximum_lateral_fraction` | excludes wall-adjacent floor where geological props would intersect the walls |
| `plan_resolution_m` | grouping resolution used to report top-down overlaps |
| `minimum_clearance_m` | rejects cells without useful measured headroom |

### Geometry Config

| Key Group | Purpose |
|---|---|
| `resolution_policy` | select body/quality-aware resolution or a legacy fixed voxel size |
| `voxel_size`, `density_margin` | set fixed-policy resolution and padding around the stamped cave network |
| `storage_mode`, `max_dense_voxels` | select dense/tiled storage or automatically enforce a dense-memory budget |
| `chunk_size` | controls how much of the density grid is polygonized at once |
| `iso_level` | defines the density threshold used for the cave wall surface |
| `tunnel_radius_scale`, `junction_radius_scale`, `chamber_radius_scale` | control how section samples widen while stamping |
| `use_section_profiles` | use Stage C's closed cross-section polygons instead of circular capsule stamps |
| `wall_roughness_*` | add seeded near-wall roughness before marching cubes |
| `junction_irregularity_*` | deform junction/chamber volumes so they blend less like perfect ellipsoids |
| `minimum_radius`, `weld_tolerance` | keep thin passages meshable and weld repeated isosurface vertices |

With `resolution_policy = "body"`, Stage D resolves the voxel size from the
selected body and `[run].quality`:

| Quality | Earth | Mars | Moon |
|---|---:|---:|---:|
| `preview` | 1.0 m | 2.0 m | 4.0 m |
| `standard` | 0.7 m | 1.4 m | 2.8 m |
| `production` | 0.5 m | 1.0 m | 2.0 m |

The policy guarantees at least ten nominal samples across an Earth preview
passage and records both nominal and actual minimum-section sampling in the
geometry summary. To request an exact resolution, use
`resolution_policy = "fixed"` together with `voxel_size`; mixing a fixed size
with the body policy is rejected instead of silently choosing one.

### Event Config

| Key Group | Purpose |
|---|---|
| `rock_density_per_100m`, `boulder_density_per_100m` | control loose debris density along sampled cave length |
| `geological_event_density_per_100m` | control larger collapse/choke/infill event density |
| `collapse_event_fraction`, `choke_event_fraction`, `infill_event_fraction` | split larger geological events by type |
| `*_radius_range` | bound event sizes before they are clipped to local tube dimensions |
| `minimum_event_spacing` | keep major event centers from clustering too tightly |
| `minimum_rock_spacing`, `minimum_boulder_spacing` | use denser small-rock scatter without crowding major obstacles |
| `clustered_debris_fraction` | target share of props preferentially sampled around collapses |
| `collapse_cluster_radius_scale`, `collapse_cluster_spacing_scale` | control collapse-debris reach and local packing |
| `max_lateral_floor_fraction` | keep floor debris inside the local tube profile |
| `mesh_latitude_segments`, `mesh_longitude_segments` | control generated event mesh resolution |
| `enabled`, `enabled_kinds` | export an empty tube or enable selected event families |
| `use_rocky_meshes`, `strict_optional_provider` | control the optional Rocky provider without importing it when disabled |

## Project Layout

- `config/`: project configuration
- `docs/MAJOR_UPGRADE_PLAN.md`: phased architecture and release gates
- `docs/CELESTIAL_BODY_GALLERY.md`: generated Earth, Mars, and Moon figure gallery
- `scripts/`: stage entrypoints
- `src/plume_advanced/config.py`: TOML loader and path-aware schema validation
- `src/plume_advanced/world.py`: celestial body, material, run, export, and seed profiles
- `src/plume_advanced/exporters/`: Blender-independent target adapters
- `src/plume_advanced/stages/`: stage implementations
- `src/plume_advanced/visualization/`: stage visualizations
- `outputs/`: generated images
- `tests/`: unit, regression, exporter, and compact end-to-end tests

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
.venv/bin/plume-generate
```

The repository wrapper `.venv/bin/python scripts/generate_cave.py` remains
available. An installed wheel includes a compact default configuration, so
`plume-generate` also works outside the source checkout. A local
`config/project.toml` takes precedence when present.

Generation checks every destination directory before writing. If a destination
already contains files, an interactive run asks for confirmation and a
non-interactive run stops safely. For a deliberate unattended/debug overwrite,
use either:

```toml
[run]
overwrite_outputs = true
```

or:

```bash
.venv/bin/python scripts/generate_cave.py --force-overwrite
```

Use `--body earth`, `--body mars`, or `--body moon` to override the configured
body for one run. The override also selects that body's default geological
material.

Regenerate all documentation figures in one batch:

```bash
.venv/bin/python scripts/generate_body_figures.py
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
- `outputs/stage_c_floor_map.png`
- `outputs/stage_c_floor_map.npz`
- `outputs/stage_c_floor_map.json`
- `outputs/stage_a_host_influence.json`
- `outputs/stage_b_network_report.json`
- `outputs/stage_d_geometry.png`
- `outputs/stage_d_geometry_chunks.png`
- `outputs/stage_d_geometry_presentation.png`
- `outputs/resolved_project_config.json`
- `outputs/run_manifest.json` with running/complete/failed status, current or
  failed stage, dependency versions, hashed inputs and outputs, source state,
  and elapsed time
- `outputs/export_<target>/...`

### Import the default Blender package

The default configuration writes a validated Blender package to
`outputs/export_blender/`. Import the cave through Blender's import menu; do
not use `File > Open`, which is for Blender project files:

1. Choose `File > Import > glTF 2.0 (.glb/.gltf)`.
2. Select `outputs/export_blender/stage_d_geometry.glb`.

Every Blender GLB package also contains:

- `stage_d_geometry_fallback.obj`, a geometry fallback imported through
  `File > Import > Wavefront (.obj)`;
- `stage_d_geometry_import_blender.py`, which tries GLB first and OBJ second
  when run from Blender's Scripting workspace;
- `README_IMPORT_BLENDER.txt`, containing the same package-local instructions;
- `stage_d_geometry.blender_validation.json`, proving that the generated asset
  passed an independent parse and finite-bounds check before export completed.

The OBJ fallback prioritizes dependable geometry interchange. Use the GLB when
you want the generated material and texture data.

Optional:

```bash
.venv/bin/python scripts/generate_cave.py \
  --config config/project.toml \
  --output outputs/stage_b_cave_network.png \
  --host-output outputs/stage_a_host_field.png \
  --section-output outputs/stage_c_section_field.png \
  --floor-map-output outputs/stage_c_floor_map.png \
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

The remaining placeholders focus on surface synthesis and target-native scene
features.

### Stage D: Geometry

Implemented as a voxel-density meshing pass.

What exists now:

- one density grid containing the full stamped tunnel network
- chunked isosurface generation for review/progress visualization
- diagnostic, chunk-focused, and presentation Stage-D render artifacts
- target export of the final cave plus separate grounded rock/boulder props

Still deferred:

- watertight versus blend-ready export modes
- seam-aware UV atlas generation and production surface cleanup
- higher-quality event-specific cleanup for rocks, boulders, collapse, choke points, and infill

### Stage E: Geological Events

Implemented as a two-pass surface-query and structural-modifier stage.

What exists now:

- deterministic placement from Stage-C samples and named event seed
- base-density floor raycasts and contact normals for prop placement
- surface-aligned, slightly embedded rock and boulder props
- collapse, choke, and floor-infill SDF modifiers
- collapse-centred clustered rock/boulder sampling with parent-event metadata
- class-specific spacing and broader candidate coverage
- event visualization and summary metrics
- voxel integration before Stage-D meshing
- post-event floor relifting and geology masks for sediment, breakdown,
  debris, and constrictions

Still deferred:

- area-aware blue-noise floor sampling
- cached rock prototype families and instance export
- direct consumption of floor geology masks by Stage-F surface materials

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
- Stage D1 stamps the base network into a voxel density field
- Stage E grounds prop meshes and creates structural density modifiers
- Stage D2 meshes the final cave and exports grounded props alongside it
- Stage F and watertight/detail work remain for the next passes

That keeps the pipeline inspectable while still leaving a clear path toward the
final pyroduct mesh and texture stages.

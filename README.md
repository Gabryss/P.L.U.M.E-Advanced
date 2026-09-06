# PLUME-Advanced

`PLUME-Advanced` is a staged procedural pyroduct / lava-tube prototype.

The network reliability and procedural-flow upgrade is active on
`codex/network-reliability-overhaul`. Its implementation roadmap, acceptance
criteria, and dependency policy are in
[`docs/MAJOR_UPGRADE_PLAN.md`](docs/MAJOR_UPGRADE_PLAN.md). The configured
export builds packages for Blender, UE5, Unity, Gazebo, and NVIDIA Omniverse
from one canonical scene using Python libraries and open interchange formats.

The current implementation focuses on the full inspectable cave-shape pipeline:
build a readable terrain substrate, derive a cave-network skeleton, generate a
geometry-ready section field around that skeleton, stamp that network into a
voxel density grid, and mesh the carved volume. The portable visual-surface
path includes xatlas UVs, normals/tangents, PBR packaging, smoothing, and
displacement baking; geology-conditioned material synthesis, visual LODs, and
finite wall shells remain deferred.

## Watch the complete pipeline

[![Watch the PLUME pipeline animation: overview and stages A–F](docs/media/full_pipeline_preview.jpg)](docs/media/FullPipeline.mp4)

[Open or download the full pipeline video](docs/media/FullPipeline.mp4)
— **3 min 36 s · 1080p · 30 fps · seven chapters**. Click the preview above
to follow the host fields through network growth, adaptive sections, meshing,
rock placement, and surface preparation.

| Start | Chapter |
|---|---|
| 0:00 | Pipeline overview |
| 0:41 | A · Host field: elevation map to rotating 3D terrain |
| 1:38 | B · Semantic network: lava-age flow and colored segment roles |
| 1:57 | C · Adaptive sections: branch sampling and profile shapes |
| 2:13 | D · Geometry: density, edge crossings, and marching cubes |
| 2:43 | E · Grounded rocks and boulders |
| 2:59 | F · Surface preparation and export |

The animation uses frozen stage artifacts for explanation, not a physical
lava-flow simulation. Stage E demonstrates rock and boulder placement only.
Stage F's rotating surface uses illustrative styles, not exported texture
channels or a final textured-cave fly-through.

The overview and each stage also remain separate video assets. See the
[Manim video project](paper/media/video/README.md) for individual animations,
setup, rendering, and lossless assembly instructions. The README video and
preview live in `docs/media/`; working renders and caches remain Git-ignored.

## Pipeline

| Stage | Status | Purpose | Current Output |
|---|---|---|---|
| A. Host Field | Implemented | Build terrain and structural layers | `outputs/stage_a_host_field.png` |
| B. Cave Network | Implemented | Grow a seeded, multi-phase lava-lobe network with competition, reoccupation, retirement, and coalescence | `outputs/stage_b_cave_network.png`, `outputs/stage_b_emplacement_history.png`, `outputs/stage_b_network.json`, diagnostics report |
| C. Section Field | Implemented | Build adaptive/uniform/reference lava-tube cross-sections around the skeleton | `outputs/stage_c_section_field.png`, `outputs/stage_c_sections.{json,npz}` |
| D. Geometry | Implemented | Stamp the cave network into a voxel grid, polygonize it, and build a globally welded render mesh | `outputs/stage_d_geometry.png`, geometry report, portable scene package |
| E. Geological Events | Implemented | Ground rocks/boulders on the base cave and apply parameterized collapse/choke/infill structural modifiers | visualization and event report |
| F. Surface Detail / Texturing | Partial | Coherent UVs/tangents, embedded PBR maps, smoothed and displacement-baked visual wall | `outputs/export_neutral/plume_cave_scene.glb` |

## Current Outputs

### Stage A: Host Field

![Stage A Host Field](docs/figures/celestial_bodies/earth/stage_a_host_field.png)

Stage A produces the terrain substrate and the main scalar layers used by later
stages: elevation, slope, cover thickness, roof competence, and growth cost.

### Stage B: Cave Network

![Stage B Cave Network](docs/figures/celestial_bodies/earth/stage_b_cave_network.png)

Stage B generates a host-driven, multi-source lava-tube network. A downhill
backbone follows a spatially correlated perturbation of the host terrain, while
seeded lobe fronts diverge, cool, retire, or coalesce into downstream channels.
Source feeders join the resulting graph, flux is conserved through every
split/merge, and every point carries flux, temperature, and lava age.

### Stage C: Section Field

![Stage C Section Field](docs/figures/celestial_bodies/earth/stage_c_section_field.png)

Stage C generates geometry-ready cross-section samples along the network:
adaptive sample spacing, underground centerline placement, 3D local frames,
lava-tube profile controls, and junction-aware blending through split/merge
regions. Flux, temperature, lava age, and derived flow maturity remain attached
to every sample, so later morphology and deposits follow the same formation
history as the network.

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
Candidate ranking also follows the inherited flow state: mature, cooled, and
lower-flux reaches preferentially accumulate infill and transported lag, while
structural controls continue to govern collapse and choke placement.

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

Stage B builds the cave skeleton through a hybrid process model configured by
`growth_model = "hybrid_lobe"`, `network_density`, and
`[network.lobe_growth]`, and `[network.emplacement_history]`. It combines an
active-lobe routing model, a DOWNFLOW-style spatially correlated terrain
perturbation, and a lightweight thermal/flux budget. The generator:

- uses the host field and the configured `procedural_seed`
- traces a downhill backbone over correlated, seed-controlled terrain uncertainty
- scores overflow, margin-avulsion, bend-overflow, and seeded-blockage events along the supplying route
- allocates each accepted breakout a finite fraction of its parent phase's flux, so neighboring lobes compete instead of behaving as independent paths
- balances momentum, perturbed slope, downstream potential, early channel avoidance, and later channel reuse at every growth step
- returns only the surviving fraction of a coalesced branch to the downstream parent and retires launches below the viable-flux threshold
- inflates a temporary emplacement surface after every accepted lobe, allowing later fronts to respond to earlier deposition
- processes the seeded breakout queue in chronological eruptive phases so deposited relief and established passages persist into later pulses
- limits every phase to a finite source budget and records source-budget utilization separately from downstream recirculation
- permits downstream-compatible later pulses to reoccupy an established lobe and form a supply-piracy breakout without teleporting between graph locations
- retires exposed lobes when their thermal budget falls below the configured threshold
- emits `anastomosis` segments when lobes coalesce and `abandoned_lobe` or `stalled_lobe` segments when they terminate
- constructs the preserved network over seeded emplacement phases, recording route birth, retirement, duty cycle, and peak formation flux
- places a controlled fraction of old and young lobes on upper and lower levels, with smooth vertical capture at their attachments
- distinguishes ordinary confluences from the subset of energetic coalescences that enlarge into chambers
- classifies local breakout regime and preserved roof state (`intact_tube`, `partial_roof`, `skylight_prone`, or `open_channel`)
- records each lobe's path id, topological branch order, causal breakout and loop mechanism, parent/branch/returned flux, reoccupation target, surface feedback, termination reason, final temperature, and lateral separation
- clusters morphologically meaningful split/merge regions into explicit junction objects
- solves the completed directed graph for exactly conserved flow and monotonic cooling/age
- reserves chambers for explicit junction/confluence semantics rather than painting high-flux blobs into occupancy
- derives occupancy and graph summaries from the resulting network

`growth_model = "legacy_braid"` remains available for reproduction of older
assets; only that compatibility mode reads `[network.braid_grammar]`.

`network_density` is the high-level artistic control for natural topology.
`0.0` disables lobe branches, `0.5` produces a sparse network, `1.0` is the
calibrated baseline, and values up to `3.0` progressively increase lobe
launches and the resulting opportunities for loops and anastomoses. The value
also adjusts anchor spacing so additional paths can form without collapsing
into regularly spaced split zones.

The Stage B visualization includes longitudinal channel/width diagnostics, a
seeded lobe-lifetime diagram, a persistence-versus-sinuosity morphospace, a
finite-flux allocation plot, and a breakout-cause/survival plot. A separate
`stage_b_emplacement_history.png` diagnostic shows per-phase active paths,
finite source-budget allocation/return, and the survival timeline without
running the section or mesh stages.
The longitudinal panel reads
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
- turns early upper-level routes into shelf-like, clearance-constrained profiles that descend into younger tubes without symmetric cable-shaped sags
- derives smooth section controls such as width, height, floor relief, roof arch, and lateral skew
- evaluates seeded node-anchored morphology fields so width, height ratio, arch, floor, skew, and roughness drift continuously without shape jumps at junctions
- interpolates Stage-B flux, temperature, and lava age into every section and derives a normalized flow-maturity field
- lets progressive cooling and age subtly lower the profile, flatten the floor, and change bounded wall relief
- adds bounded multi-harmonic wall/floor relief while keeping every local profile finite, simple, and closed
- uses explicit Stage-B junction regions to blend split/merge morphology without hard jumps at nodes
- records per-sample junction influences for later split/merge volume construction
- stores closed local 2D section contours and scalar profile controls for the voxel geometry stage

The Stage C diagnostic contains plan and longitudinal vertical views, a
dominant-route size/elevation trace, ten actual-scale cross sections sampled
through the route, morphology controls, and a width/aspect/floor morphospace.

### Stage D: Geometry

Implemented in `src/plume_advanced/stages/geometry.py`.

Stage D turns the section field into a voxel-first mesh. The generator currently:

- stamps Stage C section-profile densities along every segment, with higher density meaning carved space
- adds zoned, multi-scale seeded wall relief near the isosurface, including
  stronger floor terrain variation, so smooth flow-lined regions alternate
  with rougher rocky regions
- stamps bounded, asymmetric split/merge transitions over one to three local
  diameters, using incident width and flux without producing spherical hubs
- leaves grade-separated plan-view crossings as independent volumes unless
  Stage B explicitly marks a coalescence, chamber, or vertical capture
- processes the field in 3D chunks
- switches from a dense field to overlapping sparse tiles when the configured
  dense-voxel budget would be exceeded
- polygonizes chunk meshes with `scikit-image`
- cancels opposite-wound internal seam faces and verifies that the assembled
  surface remains closed and manifold
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
sections, events, and geometry. Each generator then derives labeled sub-seeds
for independent domains such as route grammar, inlet strength, per-segment
morphology, junction relief, event placement, ground contacts, and exported
surface detail. The streams are based on labels rather than call order, so
adding a detail draw or disabling events cannot perturb the network. Direct
generator use without an explicit seed resolves to the reproducible baseline
seed `0`; it never falls back to operating-system entropy or disables
procedural variation.

### World, run, and export config

| Section | Purpose |
|---|---|
| `[world]` | Select `earth`, `mars`, or `moon`, plus a rock-material preset and optional physical overrides |
| `[flow_regime]` | Control supply, duration, inflation, distributary tendency, and cooling independently of the selected body |
| `[run]` | Select preview/standard/production quality and development extent caps |
| `[export]` | Select all application packages, Blender, neutral, UE5, Unity, Gazebo, or Omniverse output intent |

Body presets supply gravity, a default rock material, passage and room caps,
route-length guidance, and production resolution guidance. Earth defaults to
10 m passages and 20 m rooms; Moon defaults to
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

Development mode shortens the host extent and active-lobe count but does not shrink
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
enabled = true
include_rock_props = false # cave walls, but no separate rocks or boulders

[export]
target = "all"
format = "auto"
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
| `growth_model` | select the default `hybrid_lobe` process or the reproducibility-only `legacy_braid` grammar |
| `network_density` | high-level `[0, 3]` multiplier for natural lobe, loop, and anastomosis abundance; `1` is the calibrated default |
| `channel_count_samples` | control longitudinal network diagnostics sampling |
| `[network.lobe_growth]` | control active-lobe population, persistence, correlated terrain uncertainty, routing forces, branch flux, cooling, retirement, and coalescence |
| `breakout_*` within `[network.lobe_growth]` | weight capacity overflow, confinement loss, curvature, and seeded blockage when selecting causal branch events |
| `minimum_viable_flux_fraction`, `coalescence_flux_return_fraction` | control branch survival and how much discharge rejoins the parent after coalescence |
| `deposition_feedback_m`, `deposition_spread_cells` | control how accepted lobes modify the temporary surface seen by later growth |
| `[network.emplacement_history]` | control seeded emplacement phases, route lifetimes, stacked-level abundance, vertical-capture chamber formation, and roof preservation |
| `phase_flux_budget_fraction`, `retirement_flux_threshold` | bound each pulse's source discharge and retire flux-starved paths independently of thermal retirement |
| `reoccupation_probability`, `breakout_probability` | control later-pulse passage reuse and the subset that exits through a supply-piracy breakout |
| `[network.braid_grammar]` | legacy-only ranges and probabilities used when `growth_model = "legacy_braid"` |

### Section Field Config

| Key Group | Purpose |
|---|---|
| `base_height_ratio`, `minimum_height_ratio`, `maximum_height_ratio` | control the default lava-tube width/height relationship |
| `minimum_sample_spacing`, `maximum_sample_spacing` | bound adaptive section-sample spacing |
| `curvature_spacing_weight`, `width_gradient_spacing_weight`, `junction_spacing_weight` | make sampling denser where the skeleton or morphology changes faster |
| `profile_resolution` | control local section contour resolution |
| `width_scale_*`, `width_longitudinal_variation` | broaden passage sizes smoothly within resolved world limits |
| `height_ratio_*`, `minimum_tube_height` | control aspect-ratio diversity without under-resolved vertical pinches |
| `floor_flatness_*`, `floor_relief_*`, `wall_roughness_*`, `roof_arch_*`, `lateral_skew_amplitude` | shape deterministic irregular wall, roof, and floor profiles |
| `morphology_gradient_strength`, `morphology_correlation_length` | control continuous node-anchored section drift and its physical wavelength |
| `centerline_wobble_*` | add bounded centerline meander to avoid unnaturally straight tube runs |
| `vertical_level_spacing`, `maximum_uphill_grade`, `level_transition_fraction` | control stacked-route separation and physically bounded capture profiles |
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
| `cave_normal_scale` | control tangent-space normal-map strength |
| `cave_smoothing_iterations` | remove marching-cubes terraces from the visual mesh without changing collision |
| `cave_displacement_scale_m`, `cave_displacement_midlevel` | bake bounded height relief into portable visual-mesh positions |

With `resolution_policy = "body"`, Stage D resolves the voxel size from the
selected body and `[run].quality`:

| Quality | Earth | Mars | Moon |
|---|---:|---:|---:|
| `preview` | 1.0 m | 2.0 m | 4.0 m |
| `standard` | 0.6 m | 1.2 m | 2.4 m |
| `production` | 0.5 m | 1.0 m | 2.0 m |

The policy guarantees at least ten nominal samples across an Earth preview
passage and records both nominal and actual minimum-section sampling in the
geometry summary. To request an exact resolution, use
`resolution_policy = "fixed"` together with `voxel_size`; mixing a fixed size
with the body policy is rejected instead of silently choosing one.

### Event Config

| Key Group | Purpose |
|---|---|
| `include_rock_props` | include separate rock/boulder meshes; set `false` for a wall-only scene while retaining structural cave events |
| `rock_population_multiplier` | scale background rocks, boulder anchors, rubble-cluster frequency, and collapse fragments together; the project default is `10.0` |
| `debris_density_basis` | use `floor_area` so wider galleries receive proportionally more debris, or `length` for legacy projects |
| `rock_density_per_100m2`, `boulder_density_per_100m2` | control sparse unassociated debris and true boulder density over integrated gallery floor area |
| `rock_density_per_100m`, `boulder_density_per_100m` | legacy length-based density controls |
| `geological_event_density_per_100m` | control larger collapse/choke/infill event density |
| `collapse_event_fraction`, `choke_event_fraction`, `infill_event_fraction` | split larger geological events by type |
| `*_radius_range` | bound the heavy-tailed size family before applying the local gallery capacity |
| `gallery_width_size_fraction`, `gallery_clearance_size_fraction`, `boulder_max_height_fraction`, `roof_block_size_fraction` | derive local fragment limits; boulders may reach two-thirds of available cave height by default |
| `minimum_event_spacing` | keep major event centers from clustering too tightly |
| `minimum_rock_spacing`, `minimum_boulder_spacing`, `background_contact_spacing` | combine a small absolute floor with footprint-aware separation |
| `clustered_debris_fraction` | target share of props preferentially sampled around collapses |
| `collapse_cluster_radius_scale`, `collapse_cluster_spacing_scale` | control collapse-debris reach, size decay, and talus contact packing |
| `max_lateral_floor_fraction`, `edge_accumulation_strength`, `placement_jitter_m` | favour natural wall-side deposition while removing discrete floor-atlas rows |
| `rover_width_m`, `rover_side_margin_m`, `rover_max_lateral_slope`, `preserve_rover_route` | inflate obstacles and preserve a continuous rover-width interval through each segment |
| `enable_debris_families`, `boulder_satellite_count_range`, `boulder_halo_radius_range_m` | surround each large boulder with a compact, size-biased family of rubble, companions, and runout fragments |
| `collapse_fragment_count_range`, `collapse_talus_radius_range_m` | derive larger talus-family populations from collapse volume and distribute them in anisotropic fans |
| `minor_cluster_density_per_1000m2`, `minor_cluster_count_range`, `minor_cluster_radius_range_m` | replace uniform pebble scatter with compact parent-and-child rubble patches separated by clean floor |
| `clean_floor_fraction`, `debris_patch_length_m` | retain coherent clean lava-floor patches instead of uniform salt-and-pepper coverage |
| `wall_scree_fraction`, `transported_lag_fraction` | split background debris between wall margins, low/flat transported deposits, and general scatter |
| `mesh_latitude_segments`, `mesh_longitude_segments` | control generated event mesh resolution |
| `enabled`, `enabled_kinds` | export an empty tube or enable selected event families |
| `rock_size_bias`, `boulder_size_bias` | bias deterministic size sampling toward abundant small debris while retaining occasional large obstacles |
| `rocky_texture_dir`, `rocky_resolution_scale`, `rocky_max_subdivisions` | select Rocky material inputs and geometric detail |
| `use_rocky_meshes`, `strict_optional_provider` | enable Rocky and prevent silent fallback to the legacy low-resolution mesh |

Rocks and boulders can be generated by
[Gabryss/Rocky](https://github.com/Gabryss/Rocky), pinned to a tested commit in
the optional `rocks` dependency group in `pyproject.toml`. The base installation
does not install or import Rocky. `include_rock_props = false` disables separate
rock and boulder props; `use_rocky_meshes = false` keeps props enabled but uses
the built-in mesh generator. Stage E deterministically selects rounded, angular,
vesicular, slab, ropy-lava, and eroded families; scales them to the sampled
physical dimensions; aligns local up with the final floor normal; embeds the
base slightly; and clusters part of the debris near collapse events. Placement
is joint rather than independent: the sampled floor position determines the
maximum plausible fragment size, footprint-aware spacing permits dense talus,
and fragment size decreases away from its collapse source. The floor combines
coherent clean patches, sparse unassociated debris, wall scree, transported
lag deposits, compact minor rubble patches, boulder aprons, and volume-scaled
collapse fans. Boulder aprons mix tiny fragments with a substantial visible
10–65 cm companion population and a sparse runout tail. Minor patches use a
parent-and-child process: an identifiable 28–70 cm anchor is surrounded by
8–18 smaller fragments within 1.2–3.0 m, leaving broad clean intervals instead
of a uniform pebble grid. Family ids, anchor event ids, and debris roles are
exported as GLB node extras and USD custom metadata. A continuous lateral
jitter is raycast back onto the final floor, and elongated fragments follow
runout/downhill direction instead of receiving unconstrained yaw.

`rock_population_multiplier = 10.0` targets ten times the complete population,
not merely ten times the sparse background layer. Because accepted placements
must still fit the cave, avoid intersections, and preserve the rover route,
Stage E replenishes rejected, tightly packed family slots as 3–12 cm
distributed micro debris in active debris patches. This maintains the global
density target without forcing unsafe overlap around an individual boulder or
collapse. Dense scenes increase generation time and scene size approximately
linearly.

The default rover envelope is 1.0 m wide with 0.1 m clearance per side. Every
accepted prop must leave a connected 1.2 m corridor through its tunnel segment;
this can be disabled for intentionally impassable scenario generation. Normal
maps improve RGB shading but do not change ordinary depth or collision output,
so rover-relevant fragments remain mesh geometry. Rocky UVs and material maps
are retained on the separate prop nodes in GLB and USD exports. With
`strict_optional_provider = true`, a missing or incompatible Rocky installation
stops generation instead of quietly restoring the smooth fallback spheres.

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

Install the optional Rocky provider only when detailed rock/boulder props are
required:

```bash
uv sync --group dev --extra rocks
```

Or with standard Python tooling:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest pillow
```

With pip, use `-e '.[rocks]'` instead of `-e .` to install Rocky.

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

Interrupted deterministic runs can resume from validated stage checkpoints:

```bash
.venv/bin/plume-generate --resume
```

Checkpoints default to `.plume-checkpoints` beside the output directory and may
be relocated with `--checkpoint-directory`. A checkpoint is reused only when
its resolved configuration, source inputs, production Python sources, Python
version, and payload checksum still match. Checkpoints are local pickle files;
only resume from a trusted directory created by PLUME.

Use `--body earth`, `--body mars`, or `--body moon` to override the configured
body for one run. The override also selects that body's default geological
material.

Regenerate all documentation figures in one batch:

```bash
.venv/bin/python scripts/generate_body_figures.py
```

The generator prints progress bars and ETAs for configuration loading, stages
A-C, Stage-E geological event placement, detailed Stage-D voxel/mesh
generation, Stage-D visualization, and the selected target export. Stage E
reports boulder anchors, individual requested prop slots with running accepted
and rejected totals plus the current debris role, and final Rocky mesh
construction. Its total can grow while minor-cluster child counts are resolved.
It can grow again if rejected local-family slots need a distributed
micro-debris recovery pass.
Geometry progress reports stamp counts, chunk meshing status, assembled face
counts, and final component counts.

That one command produces:

- `outputs/stage_a_host_field.png`
- `outputs/stage_b_cave_network.png`
- `outputs/stage_b_emplacement_history.png`
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

### Import the complete portable scene

The default configuration writes
`outputs/export_neutral/plume_cave_scene.glb`. This is the final scene rather
than a Stage-D-only intermediate: it contains the cave wall after structural
events, separate editable rock/boulder nodes, vertex normals, tangents, UVs,
embedded base-colour/metallic-roughness/normal textures, and visual
displacement baked into vertex positions. No source texture files are required
after copying the GLB.

The neutral package also writes `plume_cave_scene_fallback.obj` with its MTL,
using the same processed wall positions, UVs, and normals as the GLB. The
separate `plume_cave_scene_collision.obj` remains the simplified physics mesh;
do not use that collision sidecar as the rendered cave.

Import the GLB through the target application's glTF importer. Blender uses
`File > Import > glTF 2.0`; UE5 uses Interchange, while Unity requires a
compatible glTF package such as glTFast. Gazebo receives a native SDF/OBJ model
package instead. A single interchange file cannot prescribe
an engine's physics settings, so the package also provides
`plume_cave_scene_collision.obj` for consumers that require a separate,
lower-complexity collision mesh.

Select `target = "all"` with `format = "auto"` to create Blender, UE5, Unity,
Gazebo, and Omniverse packages together under `outputs/export_all/`. The
aggregate manifest records every primary asset and sidecar. Per-target exports
remain available with `target = "blender"`, `"ue5"`, `"unity"`, `"gazebo"`,
or `"omniverse"` and their native format (`glb`, `obj`, or `usd`). These
adapters do not change the canonical cave.

The final wall surface, event meshes, texture payload, and simplified collision
mesh are prepared once and shared by every selected adapter. Each export is
first written to a sibling staging directory and then published as one atomic
directory replacement. A failed adapter therefore leaves the previous package
intact. Treat `outputs/export_<target>/` as package-owned: a successful export
replaces that complete directory, including unrelated files placed inside it.

### Configuration schema

The current project schema is `schema_version = 3`. Schema 1 and 2 projects are
migrated on load when removed export scaffolding was inactive. Configurations
that requested the unimplemented `generate_lods`, `generate_wall_shell`, or
`generate_visual = false` switches fail with an actionable migration error
instead of silently promising assets that are not produced.

The Gazebo adapter targets the current Gazebo Jetty LTS stack: Gazebo Sim 10,
sdformat 16, and SDF 1.12. It writes a relocatable model directory, a launchable
world file, copied material textures, and the exact `GZ_SIM_RESOURCE_PATH`
command. UE5, Unity, Blender, and Omniverse packages each include import
guidance alongside their visual and collision assets.

The cave GLB and USD outputs use xatlas to split the final visual wall into
topology-aware conformal charts. The atlas result is rescaled so one repeated
rock-texture tile represents approximately 8 m in either UV direction; atlas
normalization therefore cannot stretch one tile over the complete cave.
xatlas duplicates vertices where charts require seams, including around
branches and openings. Displacement samples from every copy of a seam vertex
are averaged before moving the original welded surface, preventing UV seams
from opening physical cracks. Smooth geometric normals are recomputed on that
welded displaced surface and copied to the chart vertices, while tangents are
derived from the final xatlas UV orientation for correct normal-map shading.
The visual mesh receives a light global cleanup followed by spatially varying
extra smoothing: rough zones keep geometric relief, while smooth lava-flow
zones receive more cleanup. Collision geometry retains the original
conservative isosurface. Cave faces are oriented toward the traversable
interior and exported single-sided because the generated surface is a void
boundary, not an exterior rock shell. The
`geometry.cave_normal_scale`, `geometry.cave_smoothing_iterations`,
`geometry.cave_displacement_scale_m`, and
`geometry.cave_displacement_midlevel` settings control normal-map strength,
visual smoothing, and portable vertex displacement. Their defaults are
`2.0`, `4`, `0.12 m`, and `0.5`. Displacement is visual-only: collision keeps
the undisplaced surface for stable simulation contact.

OBJ export now uses the same smoothed and displacement-baked cave surface as
GLB. It writes mesh-bound `vt` UV coordinates, smooth `vn` vertex normals, and
an MTL that references the colour, roughness, and normal maps. OBJ cannot embed
those images, so copy its texture dependencies with the OBJ/MTL; prefer GLB
when a single drag-and-drop file is required.

Surface processing happens only after Stage-D geometry and Stage-E structural
events are complete. The order is visual smoothing, xatlas chart generation,
metric UV rescaling, seam-consistent UV-driven displacement, final
normal/tangent recomputation, and material binding. The diffuse, roughness,
and normal materials therefore do not participate in cave generation or
pre-smoothing geometry.

UVs do not increase polygon resolution or change the silhouette. They define
where the material and tangent-space normal map are sampled. A coherent UV
layout and tangent basis can remove shading seams and make small rock detail
look better, but geometric normals improve only when they are recomputed from
the final displaced mesh—as this exporter does—or when the mesh itself becomes
denser. The baked normal map adds sub-polygon shading detail; the 0.6 m standard
voxel resolution and baked vertex displacement provide the actual geometry.

Validate the generated asset with visible phase progress:

```bash
.venv/bin/plume-validate outputs/export_neutral/plume_cave_scene.glb
```

Run the complete Python regression suite first, with an individual-test
progress bar, and then validate the asset:

```bash
.venv/bin/plume-validate \
  outputs/export_neutral/plume_cave_scene.glb \
  --run-tests
```

Validation writes `outputs/validation/validation_report.json` and
`outputs/validation/validation_summary.md`. Checks cover the GLB container,
embedded maps, topology, normals/tangents, UV continuity, collapsed UV
triangles, localized 95th/99th-percentile distortion, event completeness,
baked displacement, and run-manifest hashes.

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
  --geometry-glb-output outputs/plume_cave_scene.glb
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

## Scientific evaluation

The installed `plume-evaluate` command provides the separate, resumable paper
workflow documented in [`paper/README.md`](paper/README.md). It includes PDC
v2.0 auditing, cross-section morphometry, matched controllability and host
ablations, adaptive-sampling fidelity, child-process scalability measurement,
export consistency, determinism, saved-data figures, and generated LaTeX.
Normal generation also emits stable semantic graph/section artifacts, geometry
and event reports, semantic hashes, and per-stage timings.

These instruments support falsifiable claims about a process-informed,
host-conditioned environment generator. They do not turn the host model into a
full lava-emplacement simulation, and the Moon/Mars presets remain controlled
scenario envelopes rather than physically validated cave distributions.

## Existing stages and deferred work

Remaining work focuses on geology-conditioned surface synthesis and
target-native scene features.

### Stage D: Geometry

Implemented as a voxel-density meshing pass.

What exists now:

- one density grid containing the full stamped tunnel network
- chunked isosurface generation for review/progress visualization
- diagnostic, chunk-focused, and presentation Stage-D render artifacts
- target export of the final cave plus separate grounded rock/boulder props

Still deferred:

- watertight versus blend-ready export modes
- higher-quality event-specific cleanup for rocks, boulders, collapse, choke points, and infill
- explicit visual LOD generation and finite wall-shell modes

### Stage E: Geological Events

Implemented as a two-pass surface-query and structural-modifier stage.

What exists now:

- deterministic placement from Stage-C samples and named event seed
- flow-aware placement using inherited flux, temperature, age, and maturity
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

Partially implemented. The portable visual path generates xatlas charts,
metric UVs, normals/tangents, PBR texture bindings, smoothing, and
displacement-baked wall geometry. Direct use of host/floor geology masks,
event-specific cleanup, and explicit visual LODs remain future work. Detail
continues to affect representation rather than defining network topology.

## Summary

The current project state is intentionally narrow:

- Stage A builds a seeded terrain and structural substrate
- Stage B grows a seeded, host-guided lobe network with natural divergence, retirement, coalescence, conserved flow, and independent inlet strengths
- Stage C builds seeded adaptive sections while preserving flux, cooling, and age along every tube
- Stage D1 stamps the base network into a voxel density field
- Stage E grounds prop meshes and creates structural density modifiers
- Stage D2 meshes the final cave and exports grounded props alongside it
- Stage F's geology-conditioned synthesis, explicit visual LODs, and specialized
  watertight/wall-shell modes remain for later passes

That keeps the pipeline inspectable while still leaving a clear path toward the
final pyroduct mesh and texture stages.

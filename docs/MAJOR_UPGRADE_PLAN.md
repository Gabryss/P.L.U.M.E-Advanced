> Historical design plan. The current interface is schema 4; see [the cleanup review](legacy-cleanup.md). Superseded migration and braid proposals below are retained as design history.

# PLUME-Advanced major upgrade plan

Status: active on `major-procedural-upgrade`.

## Product constraints

- Generation must run from Python and installable native/Python libraries after
  cloning the repository. Blender is not a runtime dependency.
- The repository is an underground-environment generator. Robotics stacks,
  sensors, autonomy, path planning, SLAM, and ROS integrations are out of
  scope; exported maps remain engine-neutral geological artifacts.
- The canonical generated world uses metres and a documented right-handed,
  Z-up coordinate system.
- Celestial-body selection changes geology and generation. Export target
  selection changes representation, axes, units, packaging, and collisions,
  but never changes the underlying cave. Explicit visual LODs remain deferred.
- Every stage is deterministic from a named sub-seed. Disabling events or an
  exporter cannot change the host field or cave network.
- Production quality may take longer, but processing should be tiled and
  restartable so long tubes do not require one dense world-sized voxel array.

## Engineering hardening status

- Runtime code now lives under the installable `plume_advanced` namespace;
  scripts and tests no longer mutate `sys.path`.
- Cave GLBs and USD assets use xatlas conformal charts generated from the final
  visual wall, metric UV rescaling, seam-consistent displacement,
  UV-derived tangents, inward single-sided surfaces, non-shrinking visual
  smoothing, and configurable normal-map strength.
- The default complete scene is a neutral self-contained GLB. Height relief is
  baked into metre-scale vertex positions for cross-simulator portability;
  the conservative collision sidecar remains undisplaced.
- `plume-validate` runs phased container, material, geometry, normal/tangent,
  UV, event, displacement, and reproducibility checks with Rich progress and
  emits JSON plus Markdown reports.
- Wheels include a compact, texture-free default project so the
  `plume-generate` entry point works outside a source checkout.
- Nested TOML validation reports complete setting paths, and configured
  texture paths resolve relative to the configuration file.
- The run manifest is written atomically at every major stage. It records
  running, complete, or failed state, the current/failed stage, and SHA-256
  metadata for available inputs and completed outputs.
- Schema 3 is current. Schema 1 and 2 inputs migrate when removed export
  scaffolding was inactive; enabled LOD/wall-shell promises are rejected.
- Stage artifacts are checkpointed atomically and `--resume` validates the
  resolved configuration, inputs, production sources, Python version, and
  payload checksum before reuse.
- Export adapters consume one prepared canonical scene and publish complete
  packages through recoverable staging-directory swaps.
- Rocky and all separate rock/boulder generation are optional. The base
  package does not depend on Rocky and disabled props do not import it.
- CI checks Python 3.12 and 3.13, import ordering, an expanded typed core,
  compilation, a 70% repository coverage floor, and wheel construction.

## Target pipeline

1. **Host** — terrain, emplacement thickness, lithology, rock-mass quality,
   fracture tensor, thermal history, deposits, erosion, and stability fields.
2. **Flow network** — directed multi-source graph carrying conserved lava flux,
   temperature, age, grade, and vertical level.
3. **Sections** — continuous profiles with world/material samples attached to
   every section.
4. **Base SDF** — sparse, tiled cave volume with explicit entrances, portals,
   and skylights.
5. **Events** — structural modifiers alter the SDF; props are grounded and
   collision-validated against the final surface.
6. **Surface** — globally coherent normals, tangents, UVs or triplanar
   coordinates, geological masks, and baked PBR material sets.
7. **Scene asset** — one prepared semantic model currently containing the
   render mesh, simplified collision, event meshes, materials, transforms,
   units, and metadata. Visual LODs remain future work.
8. **Export adapters** — GLB/OBJ, Blender-compatible USD, UE5, Unity, Gazebo,
   and Omniverse packages.

## Phase 0 — correctness and configuration

- Add a versioned TOML schema.
- Resolve Earth, Mars, and Moon presets plus separately configurable
  geological material profiles.
- Add development mode that limits extent and graph complexity without
  changing body dimensions.
- Add global and per-kind geological-event switches.
- Add export target, format, quality, and collision controls. LOD and finite
  wall-shell controls are intentionally absent until those assets exist.
- Derive stable named sub-seeds for host, network, sections, events, and
  geometry. Surface/export preparation is deterministic from geometry and does
  not own a redundant seed.
- Validate dimensions, enum values, ranges, and incompatible target/format
  selections with actionable errors.
- Emit a resolved configuration/manifest beside every generated asset.

Exit criteria:

- Switching body changes passage/room limits and stability inputs.
- Switching events off yields a valid empty event field and does not load
  optional rock generators.
- Development mode caps physical route extent and grammar complexity.
- Existing schema-v1 and schema-v2 configurations migrate to schema 3 when
  they do not request removed, unimplemented export capabilities.

## Phase 1 — mesh and export correctness

- Export the globally welded cave render mesh, not the internal marching-cubes
  processing chunks.
- Compute angle-weighted vertex normals after welding.
- Generate seam-aware UVs and MikkTSpace-compatible tangents.
- Prepare the canonical visual/collision scene exactly once; adapters serialize
  it without rebuilding UVs, displacement, event geometry, or collision.
- Add a neutral standards-compliant GLB writer and complete OBJ materials,
  normals, and texture coordinates.
- Add asset checks for finite values, bounds, winding, manifold state, normals,
  UVs, tangents, materials, units, and a one-metre reference fixture.
- Keep texture masters separate from runtime texture budgets; never silently
  replace an 8K source with an undocumented 1K embedded texture.

Exit criteria:

- GLB contains `POSITION`, `NORMAL`, and coherent `TEXCOORD_0`; tangent data is
  present whenever a normal texture is exported.
- A one-metre validation cube imports at one metre in neutral/Unity/Omniverse
  packages and 100 centimetres in UE5.
- Chunk boundaries do not create lighting or normal seams.

## Phase 2 — event and structural geology

Status: foundational two-pass implementation complete. Base-volume raycasts,
surface-aligned rock/boulder props, footprint-aware spacing, and collapse/choke/
infill density modifiers are active. Collapse-centred debris families, local
gallery/roof-conditioned size caps, floor-area density, continuous placement
jitter, stable runout orientation, and a one-metre-rover route constraint are
active. Layered clean-floor/background/scree/lag populations, compact
boulder-satellite families, volume-scaled collapse talus fans, exported family
metadata, and post-event geological floor masks are also active. Cached
instances and full mesh-versus-wall collision rejection remain.

- Split events into props (`rock`, `boulder`), modifiers (`collapse`, `choke`,
  `infill`, `skylight`), and material masks.
- Query the final base SDF for floor/roof contact; embed grounded props by a
  configurable fraction and reject intersections.
- Replace candidate ranking with area-aware blue-noise placement and explicit
  clustered processes around collapses.
- Make collapse likelihood use span, roof thickness, density, gravity,
  effective fractured-rock strength, joints, and weathering.
- Structural modifiers change visual and collision SDFs instead of adding
  decorative ellipsoids.
- Integrate optional rock generators through an explicit package interface;
  production runs fail clearly when a requested provider is unavailable.
- Cache prototype rock families and export them as instances.

Exit criteria:

- Ground-contact tests pass for every prop.
- Structural event toggles produce measurable topology/clearance changes.
- Empty-tube export contains no event geometry or event dependencies.

## Phase 3 — host-field causal model

Status: causal-field foundation complete. Independent emplacement, lithology,
fracture, cooling, capacity, deposit, erosion, and gravity-aware stability
layers are active. Routing uses one inspectable composite exactly once and
reports term influence. Adaptive local sampling and propagation of every raw
field through sections/material masks remain.

- Replace elevation-derived cover with an emplacement-thickness field.
- Add lithology, joint/fracture tensor, rock-mass quality, thermal/cooling,
  lava capacity, regolith/sediment, erosion, and impact layers.
- Remove `growth_cost` as an independently weighted input. Consumers calculate
  one named cost from raw layers and log each contribution.
- Use a coarse regional host plus continuous/adaptive local sampling around
  Earth-scale passages.
- Attach host values to graph points, section samples, material masks, and
  event probabilities.
- Add layer ablation and sensitivity reports. Remove a layer if it has no
  unique consumer or measurable effect.

Exit criteria:

- Every layer has a documented consumer and a tested effect.
- No raw field is counted both directly and inside an opaque composite cost.
- Body/material selection measurably affects network, stability, events, and
  surface masks.

## Cross-cutting floor-map stage

Status: topology-aware post-event implementation complete.

- Raycast lateral floor lanes from Stage-C sections against the Stage-D base
  volume for event placement.
- Address cells intrinsically by segment, distance along, and lateral offset;
  retain graph `z_level` and world XYZ for lossless lifting.
- Store measured clearance and the actual inward surface normal.
- Use atlas cells as rock/boulder candidates instead of inventing offsets from
  section centerlines.
- Relift stable cell addresses against the final post-modifier volume and
  explicitly report invalidated cells.
- Classify final floor cells as bare basalt, sediment, breakdown, debris, or
  constriction and export continuous event/debris influence fields.
- Mark generated chambers and termini without claiming that termini are
  physically opened entrances.
- Export NPZ/JSON for downstream environment tooling and a geological PNG with
  world elevation, clearance, classified events, and overlap-safe intrinsic
  views.

Next refinements are explicit entrance/portal carving, material-mask
consumption, area-aware debris sampling, and multi-level geological rasters.

## Phase 4 — physical network

Status: multi-source physical topology foundation complete. Production route targets now
drive host extent; branch length scales with the host; separation is enforced
in passage-width and host-corridor units; decorative flux-chamber painting is
disabled; skeleton-versus-visible channel diagnostics and a JSON network
report are active. Configured sources now create explicit feeder entries; flux,
temperature, and age propagate through the graph.

- [x] Use all configured sources, each with its own initial flux and temperature.
- [x] Conserve flux at splits and merges and derive width from flux, cooling, and
  material capacity.
- Treat braid grammar as a prior; host conditions decide whether and where a
  split, merge, room, or bypass is viable.
- Make chambers consequences of ponding, confluence, low grade, high flux, and
  stable roof conditions.
- [x] Implement real vertical grades and clearances for underpasses so graph edges
  do not accidentally fuse in the SDF.
- Make occupancy, chamber diagnostics, and exported geometry derive from the
  same graph and section source of truth.
- Validate entry-to-exit connectivity, curvature, grade, clearance, unintended
  intersections, flux conservation, and branch statistics.

Implemented foundation:

- Earth, Mars, and Moon have independent horizontal, vertical, and fracture
  host scales plus 5/15/30 km production route guidance.
- `[flow_regime]` separates supply, duration, inflation, distributary
  tendency, and cooling from celestial-body physics.
- Development mode shortens each body proportionally while keeping real
  passage widths, rather than forcing all bodies into the same footprint.
- Braid lengths grow with spatial scale and branch offsets clear both the full
  passage diameter and a host-corridor-relative threshold.
- Occupancy always retains the generated skeleton, and chamber expansion is
  tied to explicit structural junctions.
- Stage B reports skeleton channels, visibly distinct occupied channels, and
  width-normalized primary-branch persistence.

Exit criteria:

- `source_count` changes the generated flow graph.
- Underpasses have physical vertical separation.
- No diagnostic-only chamber or flux layer exists.

## Phase 5 — scalable geometry and surface

Status: body/quality-aware resolution and sparse tiled storage foundation
complete. The generator automatically switches to overlapping active tiles
when the dense-voxel budget is exceeded; tile-local events and meshing keep
working-set memory bounded. Segment and junction bounds now index only the
tiles they can affect, including edge and corner neighbors during sampling.
Surface-material work and persistent tile-cache refinements remain.

- [x] Replace the dense world bounding box with tiled sparse narrow-band SDFs or an
  adaptive octree implementation.
- Use a resolution policy based on minimum tube diameter; production targets
  roughly 12–20 samples across the smallest passage. **Foundation complete:**
  body presets and run quality now resolve the dense-grid voxel size and expose
  sampling diagnostics.
- Cache or analytically evaluate 2D profile SDFs instead of calculating
  point-to-polygon distance repeatedly.
- Preserve one high-resolution master, then derive boundary-aware visual LODs,
  simplified collision LODs, and streaming tiles.
- Generate an optional finite wall shell by extracting inner/outer surfaces
  and bridging explicit portals. Validate thickness and self-intersections.
- Build multiscale procedural PBR materials from global coordinates and host
  masks. Bake Base Color, Normal, ORM, and optional Height with seam margins.

Exit criteria:

- A development generation is representative and fast.
- A multi-kilometre production run has bounded per-tile memory.
- Render and collision assets share topology semantics but have independent
  complexity budgets.

## Phase 6 — target packages

The generator stays Blender-independent at runtime. Blender is the first
supported and default interactive target.

- **Blender:** validated GLB output plus an OBJ fallback, package-local import
  instructions, and a convenience script that tries GLB then OBJ. Blender is
  not invoked during generation.
- **UE5:** GLB Interchange package, centimetre convention metadata, collision
  naming, and import guidance. Nanite/streaming specialization remains future
  target-native work.
- **Unity:** metre/Y-up package with material guidance and separate mesh
  colliders. Prefabs and LOD groups remain future target-native work.
- **Gazebo:** relocatable `model.config` + `model.sdf`, DAE/OBJ visual meshes,
  simplified collision meshes, and PBR texture paths.
- **Omniverse:** USD/USDC with `metersPerUnit`, `upAxis`, payload tiles,
  instances, Preview Surface/MaterialX materials, and collision APIs.

Each adapter consumes the same `SceneAsset` and writes a target validation
report. Optional format libraries must have pure command-line or Python APIs
and documented installation extras.

## Testing and release gates

- Unit tests: configuration, seed stability, profile resolution, cost terms,
  grounding, transforms, normals, UVs, and manifests.
- Property tests: bounds, connectivity, flux conservation, clearances,
  deterministic output, and event independence.
- Golden fixtures: tiny caves for each body and exporter.
- Import smoke tests: headless or SDK-based checks where target software
  permits them; otherwise strict format validators.
- Performance gates: peak memory and time per generated metre at preview and
  production quality.
- Reproducibility manifest: schema version, resolved config, stage seeds,
  dependency versions, target convention, source hashes, and output checksums.

## Dependency policy

Core generation remains installable through Python packaging. Candidate
libraries will be evaluated behind small interfaces:

- sparse volumes/SDF: OpenVDB Python bindings or a chunked NumPy/SciPy backend;
- UV atlas: xatlas Python bindings (integrated for the final visual wall);
- mesh processing/LOD: trimesh plus meshoptimizer or pymeshlab where licensing
  and distribution are suitable;
- tangents: a MikkTSpace-compatible binding or a tested local implementation;
- USD: Pixar OpenUSD Python bindings;
- FBX: prefer target-side GLB/USD where possible because Autodesk FBX SDK
  redistribution complicates a clone-and-install workflow.

No optional dependency may be imported when its feature is disabled.

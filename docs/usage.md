# Generation and materials

[← Project overview](../README.md) · [Installation](installation.md) · [Configuration](configuration.md) · [Simulator imports](simulators.md)

Run commands from the repository root after installation. Start with one recipe;
textures, rocks and robot qualification are optional choices within the same pipeline.

## First cave

```bash
uv run plume-generate --output outputs/first_cave/network.png
```

This uses [config/project.toml](../config/project.toml): an Earth cave with a
250 m route target, one system, no image textures and no loose rocks. Its plain
material is expected. The run includes inspection, bounded repair, exports and
stage figures, and is labelled **not robot-qualified**.

`--output` names the network figure; its parent directory holds the complete run.
Use a separate directory for each generation. Check `run_manifest.json` for
`status: "complete"`, then open the asset listed in its export records.

## Traversability ground truth

Full runs also save `export_<target>/traversability/`: one map set per layer and
connecting ramp. Open `layer_0_overview.png`, then inspect separate elevation,
clearance, slope, roughness, obstacle and rejection-reason views. Numeric arrays
retain the measurements. These maps do not impose robot qualification.

To add maps to an existing completed GLB run:

```bash
uv run plume-traversability --source outputs/first_cave
```

See [map files, coordinates, layer connections and experimental limits](traversability.md).

## Inspect networks in 3D

`plume-network` generates only the host and network, and includes an interactive
`viewer.html` beside its figures. Open that file directly in a browser on any
computer; it needs no server, internet connection or Python installation.

```bash
uv run plume-network --config config/varied-network.toml --output outputs/my_network
```

To combine existing networks in one page, supply their run folders (or evaluation
campaign folders). Every centerline sample and actual layer elevation is retained.

```bash
uv run plume-network-view --source outputs/my_network --output outputs/network-viewer.html
```

| Control | Use |
|---|---|
| Drag / right-drag / scroll | Orbit / pan / zoom |
| Top, Side, 3D, Fit all | Change or reset the view |
| Layers and inter-layer connections | Isolate levels and connecting ramps |
| Click a passage | Read its length, width, elevations and connections |
| Vertical exaggeration | Inspect shallow elevation changes; default **1× true scale** |
| Download / Save PNG | Transfer the complete offline page / save the current view |

The circular passage shapes illustrate network widths; they are **not cave meshes
or validated cross-sections**. Network acceptance does not establish robot clearance.

For a measured Earth passage-width scenario with three interacting sources:

```bash
uv run plume-network --config config/earth-survey-network.toml --output outputs/earth_survey
```

For six irregularly placed inlets and three downstream termini, use:

```bash
uv run plume-network --config config/multi-outlet-network.toml --output outputs/multi_outlet
```

This recipe stops at the network stage. The [outlet controls](configuration.md#regional-network-controls)
allow a common outlet or several separated termini in one connected network;
connectivity and all requested termini must pass inspection.
Open `outputs/multi_outlet/viewer.html` to inspect the network.

The Earth survey recipe fits width statistics only; none of these recipes certifies
geological topology. See [what was measured and what remains procedural](evaluation.md#earth-network-width-calibration).

To take the three-source Earth survey model through sections, meshing, inspection and textured
exports, use the full-run recipe:

```bash
uv run plume-generate --config config/earth-survey-full.toml --output outputs/earth_survey_full/network.png
```

This uses three interacting sources, a 400 m route target, a 0.15 m voxel grid,
4K texture maps and all five export packages. Rocks are disabled and stage figures
are included. Robot qualification is not required. `network.detail` still
requires the network-only command.

For a complete three-layer cave with six sources, sparse additional connections,
textures and no rocks:

```bash
uv run plume-generate --config config/branching-layers-full.toml --output outputs/multilayer_full/network.png
```

This preserves the network's actual elevations through sections and meshing.
It checks geometry and exported packages without imposing robot dimensions.
See [layer controls and resolution limits](configuration.md#optional-layers).

For an HTTP network viewer, add `--serve` to `plume-network-view`; it listens on `127.0.0.1:8765` by default. Set
`--bind` to a trusted LAN/VPN interface only when remote access is intended. This
small server serves just the page and has no authentication. A private authenticated
static host is another option for access from outside that network.

## Textured caves

Complete the one-time [texture setup](installation.md#texture-dependencies), then run:

```bash
uv run plume-generate --config config/simulator-check.toml --output outputs/textured_cave/network.png
```

This is a 120 m, single-system, rock-free cave with 1K maps and all five export
packages. Change `geometry.embedded_texture_max_size` to `4096` for 4K maps;
texture resolution is independent of mesh resolution.

Open `outputs/textured_cave/export_all/blender/plume_cave_scene.glb` in Blender
with **File → Import → glTF 2.0**, then select **Material Preview**. Follow
[Simulator imports](simulators.md) for lighting, continuous materials and the
other applications.

### Add textures to an existing cave

For a neutral inspection run, create a separate material revision:

```bash
uv run python scripts/texture_inspection.py outputs/first_cave --config config/short-multi.toml --output outputs/first_cave_textured
```

The script uses only the recipe's appearance settings and preserves the passage
geometry and network count. It writes visual packages for Blender, Unity and
Unreal plus `material_revision.json`. It does not repeat collision checks or
qualify the result for simulation.

## Choose a recipe

| Recipe | Route target | Systems | Initial voxel size | Appearance / policy |
|---|---:|---|---:|---|
| [project](../config/project.toml), preset `preview` | 250 m | One | 0.20 m | Neutral; inspection |
| [simulator-check](../config/simulator-check.toml) | 120 m | One | 0.20 m | 1K PBR; all-target import example |
| [short-single](../config/short-single.toml) | 400 m | One | 0.20 m | Neutral; inspection |
| [short-multi](../config/short-multi.toml) | 400 m | Three interacting | 0.08 m | 4K PBR; inspection |
| [long-single](../config/long-single.toml) | 3 km | One | 0.20 m | Neutral; inspection |
| [long-multi](../config/long-multi.toml) | 3 km | Three interacting | 0.20 m | 4K PBR; inspection |
| [earth-survey-full](../config/earth-survey-full.toml) | 400 m | Three sources, one layer | 0.15 m | 4K PBR, no rocks; inspection |
| [branching-layers-full](../config/branching-layers-full.toml) | 800 m | Six sources, three layers | 0.25 m | 4K PBR, no rocks; research inspection |
| [simulation-single](../config/simulation-single.toml) / [simulation-multi](../config/simulation-multi.toml) | 250 m | One / three | 0.04 m | 4K PBR; simulation policy; robot opt-in |
| [showcase](../config/showcase.toml) | 300 m | Three interacting | 0.04 m | 8K PBR, rocks, rough terrain; inspection |

Run any recipe with the same command pattern:

```bash
uv run plume-generate --config config/short-multi.toml --output outputs/multi/network.png
```

**Simulation recipes are evaluation inputs, not prequalified caves.** Match voxel
size explicitly when comparing mesh costs: the short multi preset is finer than
the short single preset. For physical settings, larger domains and specialist
network models, see [Configuration](configuration.md).

## Showcase generation

After [texture setup](installation.md#texture-dependencies), generate the detailed
multi-network scene with the optional Rocky dependency:

```bash
uv run --extra rocks plume-generate --config config/showcase.toml --output outputs/showcase/network.png
```

| Feature | Recipe setting |
|---|---|
| Network | Three systems in one host, persistent parallel passages, required merges and splits; 300 m downstream target |
| Geometry | 4 cm grid, 96-point cross-sections, 0.5–1.5 m section spacing |
| Terrain | Wall, roof, floor and crust relief; at least 75% relief retained |
| Debris | Rocky rocks/boulders and clustered debris; collapse, choke and infill events enabled |
| Materials | Repeated 8K PBR tiles at a 4 m scale, plus native continuous-material adapters |
| Delivery | Five application packages, checked collider, stage figures and inspection reports |
| Budgets | Two seed attempts; no wall-clock timer; 30 million visual triangles and 4 GiB per file |

This is an expensive visual master: generation and export can take hours. Visual
reduction is disabled, and an over-budget asset stops for review. The allocation
cap of 500 million density samples is not a total RAM limit. UVs, meshes,
textures, inspections and serialization need additional memory.

One 2 cm refinement is permitted within the allocation budget. Halving voxel
spacing can require roughly eight times the volume samples and four times the
surface triangles. To reduce cost, use an 8 cm grid and 4K textures; `run.quality`
does not override an explicitly fixed voxel size.

Rough terrain and obstacles are intentional. Robot qualification and floor grading
are off; mesh, clearance, texture and export checks remain required. Event
settings permit features rather than guaranteeing a particular count for every
seed. Import the **full scene** to retain the rock objects. See the
[showcase in all five applications](simulators.md).

## Ordinary generation or robot qualification

The robot is an optional application example. Ordinary generation validates the
cave and its exports under the selected policy; a robot's inability to traverse
the terrain does not invalidate that cave. Terrain maps and later simulator
showcases do not trigger seed retries or floor repair.

| Recipe setting | Required result | Successful export label |
|---|---|---|
| Flag omitted, or `acceptance.require_ground_routes = false` | Cave passes its configured geometry, material and export checks | `qualified: false`, `status: "not_requested"` |
| `acceptance.require_ground_routes = true` | Those checks plus reference-robot geometric checks | `qualified: true`, `status: "qualified"` |

To request robot qualification, add or edit this table:

```toml
[acceptance]
require_ground_routes = true
```

No preset enables this flag implicitly. PLUME tries bounded local repair, then
retries eligible failures using deterministic seeds. Only an accepted candidate
is published. The reference robot is **0.7 m long × 0.5 m wide × 0.5 m high**,
with a 0.02 m margin, 20° slope limit and 0.10 m step limit. Search never relaxes
these limits. Underbody clearance defaults to **zero**; set
`acceptance.robot_ground_clearance_m` explicitly for the intended robot.
Wheel/track dynamics still need simulator evaluation. These checks qualify an
application-specific route, not the generator or its geological realism.

Set `acceptance.repair_ground_routes = true` to permit measured floor grading.
It tries bounded ramp and cross-slope edits, preserving roof and host constraints,
then reinspects the mesh and exports. The default maximum vertical change is
0.5 m. These are recorded engineering edits for simulation, not natural formation.
See [repair mechanics](architecture.md#embedded-inspection-and-repair) and
[route settings](configuration.md#appearance-rocks-and-route-requirements).

### Retry limits and stopping

```toml
[run]
max_seed_attempts = 8
max_attempt_seconds = 0
```

These are the defaults: eight total seeds and no wall-clock timer. `1` evaluates
only the initial seed and its local repairs; `0` disables the corresponding limit.
Attempt counts include attempts already recorded when resuming. A positive time
limit is checked at work boundaries and cannot interrupt an in-flight native call.
**Ctrl+C** stops the run and preserves its journal.

| Situation | Pipeline response |
|---|---|
| Retryable network or mesh failure after local repair | Record rejection and try the next seed within the attempt limit |
| Required robot checks fail | Repair when enabled, then retry without changing the robot limits |
| Export geometry/triangle budget fails | Preserve completed geometry checkpoints for re-export |
| Final file-size check fails | Keep the checked package in the reported `*.size-rejected-*` directory |
| Seed limit or explicit time limit reached | Stop with diagnosis and seed history; exit status `2` |
| Invalid configuration, missing assets, resource exhaustion or unexpected error | Stop with an error; do not retry seeds |

A partial output directory is not a completed generation. `--debug` includes a
traceback. Time, memory and export budgets are independent; disabling one does
not remove the others or guarantee that a passing cave exists.

## Seed history, replay and resume

`seed_attempts.json` records the request identity, each root/stage seed, outcome,
diagnosis and accepted seed. Its sequence is deterministic, not clock-based.

| Task | Command |
|---|---|
| List presets | `uv run plume-generate --list-presets` |
| Inspect settings without generation | `uv run plume-generate --config config/project.toml --show-config` |
| Start from seed 42 | `uv run plume-generate --config config/project.toml --seed 42 --output outputs/replay_seed_42/network.png` |
| Resume the first cave | `uv run plume-generate --config config/project.toml --output outputs/first_cave/network.png --resume` |

Resume requires the original recipe, CLI overrides, code, dependencies and inputs.
It skips rejected seeds, retries an interrupted seed and rechecks a completed
winner. An exhausted attempt limit remains exhausted. Only completed stages are
checkpointed; a partly built volume restarts that stage.

To reproduce one rejected seed without searching past it, use `--seed NUMBER`
and `run.max_seed_attempts = 1` in a separate recipe/output directory, retaining
its physical settings and asset paths. The winner's effective settings are in
`resolved_project_config.json`. Keep the original inputs and journal as well.
Checkpoints contain trusted local Python pickle data; use exported assets for exchange.

## Retry an export without regenerating

If generation completed its final geometry checkpoint, re-export with the
original recipe and a fresh destination:

```bash
uv run --extra rocks plume-reexport --config config/showcase.toml --source outputs/showcase --output outputs/showcase_export
```

This reuses the geometry, repeats export preparation and inspection, and saves
`reexport.json`. Source code, dependencies, assets and generation settings must
match the checkpoints. Use the original `--body` override if one was supplied.

| Optional flag | Effect |
|---|---|
| `--max-visual-triangles N` | Explicit visual triangle budget |
| `--max-asset-bytes N` | Per-file byte budget |
| `--visual-max-error-m N` | Permitted visual reduction deviation |
| `--max-seconds N` | Export-only cooperative timer; default `0` is unlimited |
| `--checkpoint-directory PATH` | Location of the matching saved checkpoints |

Re-export has no timer by default, including when generation had one. Partial UV
work is not checkpointed, so preparation restarts. Progress is saved beside the
destination as `<output>.progress.jsonl` and remains available after a failure.
Application folders are written directly under the destination; the default
asset basename is `plume_cave`.

### Recover a file-size rejection

When only the final byte limit fails, use the retained directory printed by the
exporter and choose an explicit revised limit:

```bash
uv run plume-reexport --recover-package outputs/showcase_export.size-rejected-XXXXXXXX --output outputs/showcase_recovered --max-asset-bytes 4294967296
```

Replace `XXXXXXXX` with the actual suffix. Recovery verifies saved checksums,
copies the checked files and updates the size-limit metadata without repeating
meshing or serialization. It preserves qualification status and the source copy.
The receipt and all package files must be intact. This mode cannot change geometry
or textures; the revised byte limit must still accommodate every file.

### Refine local network geometry

```bash
uv run plume-network --config config/detailed-network.toml --output outputs/detailed-network
```

This generates the host and network only, using localized host-guided features
(`strength = 0.8`) and a 3D viewer. Compatible encounters become shared junctions.
Set `[network.detail] enabled = false` to retain coarse geometry, or
add the table to another regional recipe. See [detail controls](configuration.md#optional-network-detail)
and [paired before/after evaluation](evaluation.md#paired-network-detail-evaluation).

## Inspect the output

| Artifact | Purpose |
|---|---|
| `run_manifest.json` | Completion/failure status, effective configuration, identities and exported paths |
| `seed_attempts.json` | Every attempted seed, rejection diagnosis and winner |
| `resolved_project_config.json` | Effective settings of the accepted candidate |
| `pipeline_quality_report.json`, `pipeline_recovery.json` | Required checks, repair actions and outcomes |
| `pipeline_inspection.png`, `section_resolution_report.json` | Measured passages and sampling evidence |
| `export_TARGET/` | Visual assets, collider, materials and import instructions |
| `export_TARGET/robot_qualification.json` | Qualification status and reference limits |
| `texture_recovery.json` inside the export | Material inspection/repair journal |
| `progress.jsonl` | Operation trace, seed/attempt context and timings |

For `export.target = "all"`, use `export_all/`; the qualification label sits beside
its five application folders. `not_requested` means robot checks were not required,
not that generation failed. A `qualified` label covers PLUME's geometric contract;
[simulator qualification](evaluation.md#native-engine-qualification) is separate.

See [Simulator imports](simulators.md) for asset paths, shaders, lighting and
texture troubleshooting. Changing the target alone does not enable textures.

## Diagnostics and figures

| Progress field | Meaning |
|---|---|
| Attempt, root seed and mode | Candidate being evaluated and its configured attempt limit |
| Stage and operation | Completed stages, work units and elapsed time |
| Export target and filename | Application package and asset being prepared |
| Mesh rows / GLB bytes | Serialization progress for vertices, triangles and binary data |
| Last work report | Time since an internal update; elapsed time alone does not prove forward progress |
| Ground checks / repairs | Sampled routes, floor contacts, chassis sweeps and repair trials |
| Query budget | Resource consumption, not a percentage of the cave inspected |

Operations without internal counters show no percentage or ETA. A native call
can pause terminal refresh while holding Python's interpreter lock. Finishing
one substep does not finish the export; inspect the operation label and trace.

`run.render_diagnostics = true` enables stage figures. Mandatory inspection reports
are produced independently of this flag. To regenerate the documentation's host,
network and section diagrams without meshing:

```bash
uv run python scripts/generate_readme_figures.py
```

Figures and their [provenance](figures/readme/current_provenance.json) live in
`docs/figures/readme/`; logos, animations and simulator previews also live under
`docs/`. Clearing `outputs/` does not remove documentation images.

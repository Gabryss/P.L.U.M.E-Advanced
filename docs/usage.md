# Generation and materials

[← Project overview](../README.md)

Run commands from the repository root after [installation](../README.md#installation).
This page contains optional workflows; choose the section you need.

## First cave

```bash
uv run plume-generate --output outputs/first_cave/network.png
```

This reads [config/project.toml](../config/project.toml): a small Earth cave with a
250 m route target, no image textures and no loose rocks. The GLB has a plain
material, so a grey/brown appearance is expected. Stage figures and inspection
reports are saved beside `network.png`.

The `--output` argument names a figure; its parent directory holds the complete run.
Use a new output directory for each cave. Exact asset names are listed in `run_manifest.json`.

## Textured caves

Install [Git LFS](https://git-lfs.com/) and ImageMagick with EXR support once.
On Ubuntu the ImageMagick package provides the `convert` command. Fetch the
repository's source images from the project root:

```bash
git lfs install
git lfs pull
```

Then generate a small textured cave with exports for every supported application:

```bash
uv run plume-generate --config config/simulator-check.toml --output outputs/textured_cave/network.png
```

This recipe has a 120 m route target, one source, no rocks and 1K maps for a quick
import check. Increase `geometry.embedded_texture_max_size` to `4096` for 4K maps
and the Unity/Unreal validation harness. Texture size is separate from mesh resolution.

Open `outputs/textured_cave/export_all/blender/plume_cave_scene.glb` with Blender's
**File → Import → glTF 2.0**, then use **Material Preview**. The maps are embedded.
See [Simulators](simulators.md) for the other applications and seamless native materials.

### Add textures to an existing cave

After fetching the source images above:

```bash
uv run python scripts/texture_inspection.py outputs/first_cave --config config/short-multi.toml --output outputs/first_cave_textured
```

Only the recipe's appearance settings are used. The passage geometry and number
of networks remain unchanged. The result contains a separate textured export;
changing `export.target` alone does not add textures.

## Choose a recipe

| Recipe | Route target | Systems / topology | Initial voxel size | Appearance / policy |
|---|---:|---|---:|---|
| [project](../config/project.toml), preset `preview` | 250 m | One gallery, local bypass | 0.20 m | Neutral; inspection |
| [short-single](../config/short-single.toml) | 400 m | One gallery | 0.20 m | Neutral; inspection |
| [short-multi](../config/short-multi.toml) | 400 m | Three persistent interacting systems | 0.08 m | 4K PBR; inspection |
| [long-single](../config/long-single.toml) | 3 km | One gallery | 0.20 m | Neutral; inspection |
| [long-multi](../config/long-multi.toml) | 3 km | Three persistent interacting systems | 0.20 m | 4K PBR; inspection |
| [simulation-single](../config/simulation-single.toml) / [simulation-multi](../config/simulation-multi.toml) | 250 m | One / three systems | 0.04 m, bounded 0.02 m retry | 4K PBR; simulation + reference ground robot |

**Simulation recipes are evaluation inputs, not prequalified environments.**
Their fine grids can be expensive. The short multi preset is also more detailed
than the short single preset; match `geometry.voxel_size` explicitly when comparing
mesh costs between them.

Run a chosen recipe with the same command pattern:

```bash
uv run plume-generate --config config/short-multi.toml --output outputs/multi/network.png
```

Textured recipes need the source maps described above. Recipes with Rocky props
also need the [optional rocks dependency](installation.md#optional-dependencies).

## Configuration and resume

| Task | Command |
|---|---|
| List presets | `uv run plume-generate --list-presets` |
| Show effective settings | `uv run plume-generate --show-config` |
| Resume the first cave | `uv run plume-generate --output outputs/first_cave/network.png --resume` |

Resume requires matching inputs, source code and dependencies. It verifies
checkpoints and repeats downstream inspections. A failed required check stops
publication and preserves its diagnosis. Checkpoints contain trusted local Python
pickle data; they are not interchange assets.

## Inspect the output

| Artifact in a successful run | Purpose |
|---|---|
| `resolved_project_config.json` | Effective physical and procedural settings |
| Stage A–C data and figures | Host, accepted network, section contours and floor samples |
| `pipeline_quality_report.json`, `pipeline_recovery.json` | Required checks, repairs and effective settings |
| `pipeline_inspection.png`, `section_resolution_report.json` | Measured passages and input sampling evidence |
| `export_TARGET/` | Target visual asset, collider, import instructions and material evidence |
| `texture_recovery.json` inside the export | Map normalization and package repair journal |
| `run_manifest.json`, `progress.jsonl` | Completion/failure status, identities, timings and progress |

Exact output names are recorded in the manifest. Stage figures are optional;
inspection reports are still produced when `run.render_diagnostics = false`.

| Target | Recipe setting | How to inspect |
|---|---|---|
| Blender | `export.target = "blender"` | Import the GLB; use Material Preview or Rendered shading |
| Unity | `export.target = "unity"` | Use a glTF importer and the supplied URP material installer |
| Unreal | `export.target = "ue5"` | Use Interchange import and the supplied material builder |
| Gazebo | `export.target = "gazebo"` | Follow the generated model/SDF instructions |
| Omniverse | `export.target = "omniverse"` | Use the generated USD package |
| All supported targets | `export.target = "all"` | Creates each target package from the same canonical scene |

Recipes choose a compatible default format when the target is overridden.
For complete PBR GLBs, `continuous_material/` contains shared maps and **native
blended projection shaders**. Apply them to avoid directional texture seams at UV
chart boundaries. Ordinary GLB import uses its portable UV material; custom
shaders are not transferred automatically.

| Application | Continuous material setup |
|---|---|
| Blender | Run the bundle's `apply_blender_material.py` on the cave; images are packed |
| Unity URP | Copy the bundle under `Assets`; use **Tools → PLUME → Create continuous rock material (URP)** |
| Unreal | Enable Python Editor Script Plugin; execute the bundle's `unreal/create_material.py`, then assign the material |

Detached export bundles include `SETUP.txt` with application-specific details.
For an existing run, `scripts/create_blender_inspection.py` can create a lit
inspection scene; choose `--mapping triplanar` for the continuous material.
A plain grey render may be the neutral package or Solid shading. Persistent
sparkles can be sampling noise: inspect the material and lighting, enable
appropriate Cycles denoising and allow sufficient samples before blaming UVs.

## Diagnostics and figures

The terminal shows operations, elapsed time and repair attempts. `progress.jsonl`
keeps the detailed record. Enable `run.render_diagnostics` for stage figures.

The README's topology, host and section diagrams can be regenerated without meshing:

```bash
uv run python scripts/generate_readme_figures.py
```

[Current figure provenance](figures/readme/current_provenance.json) records seeds
and settings; [older geometry provenance](figures/readme/provenance.json) covers
the retained geometry illustration.

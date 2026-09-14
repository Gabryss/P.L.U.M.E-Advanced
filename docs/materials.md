# Portable cave materials

The earlier short and long interconnected inspection packages intentionally had
empty texture paths. Their GLBs contained no images, and Blender and Unity showed
the neutral fallback material correctly. A no-rocks export can still be textured:
rock props and cave-wall materials are separate settings.

## Recommended starting point

Use one seamless rock PBR tile, repeated at a physical scale across the cave.
The full interconnected presets use the existing
[Poly Haven Dark Rock](https://polyhaven.com/a/dark_rock) asset at **4096 × 4096**,
with a **4 m tile** and normal strength **1.0**. This is an appearance baseline;
the material is not a measured lava-tube surface or a body-specific calibration.
The master images are 8K, but the export resolution is controlled separately.

A 4K image has one quarter as many pixels as an 8K image. Repeating the same tile
keeps image memory independent of cave length. Geometry and draw costs still grow
with the mesh. Texture compression, mipmaps and residency in the target engine
also affect runtime memory; compressed PNG file size is not GPU memory use.

The geometry provides passage shape and larger relief. The normal map provides
fine shading detail; it does not change collision, clearance or the silhouette.
Avoid increasing polygon count just to represent detail that a normal map can show.

The original PLUME screenshot has strong fine surface detail. A Blender procedural
material can be baked into color, roughness and tangent-space normal images, or
recreated separately in each engine. Arbitrary Blender node graphs do not transfer
as a universal shader through glTF. Portable PBR images are the practical common
format here. See the [Blender glTF material documentation](https://docs.blender.org/manual/en/4.0/addons/import_export/scene_gltf2.html).

## Configuration

Set these keys **inside the existing `[geometry]` table**. Paths below are relative
to a configuration stored in the repository's `config/` directory:

```toml
[geometry]
strict_texture_loading = true
embedded_texture_max_size = 4096
cave_texture_scale_m = 4.0
cave_normal_scale = 1.0
cave_diffuse_texture = "../texture/dark_rock_8k/textures/dark_rock_diff_8k.jpg"
cave_normal_texture = "../texture/dark_rock_8k/textures/dark_rock_nor_gl_8k.exr"
cave_roughness_texture = "../texture/dark_rock_8k/textures/dark_rock_rough_8k.exr"
cave_displacement_texture = ""
cave_displacement_scale_m = 0.0
```

`cave_texture_scale_m` sets metres per repeated tile. Smaller values give finer,
more frequent detail; larger values make the rock features larger. The portable
GLB uses metric UV charts; directional textures can reveal chart boundaries even
when every triangle has valid UVs. Use the continuous native materials below to
remove those material boundaries. They reuse the same tile and do not change
geometry. Judge ordinary passages, floors and junctions before choosing 8K.

Install Git LFS and fetch the master assets (`git lfs pull`). Pillow reads PNG/JPEG;
EXR inputs also require ImageMagick's `convert` command with EXR support. The exporter
embeds PNG images in the GLB, so importing the delivered GLB does not require EXR
support, ImageMagick, or access to the original `texture/` directory. Explicitly
configured missing or undecodable maps fail the full pipeline's texture gate, even
if low-level `strict_texture_loading` is disabled.

The general default keeps its 1024-pixel export budget and 8 m tile for compatibility.
The two full interconnected presets request 4K/4 m. Geometry-only and Stage A–C
presets remain useful without texture dependencies.

## Automatic texture inspection and repair

Ordinary generation and `plume-check --scope full` run the same texture stage
inside `export_target_asset`, before accepting the export. Resumed geometry
checkpoints do not bypass texture inspection.
Keep original texture files outside the export directory that will be replaced;
the exporter rejects destinations that would erase their own inputs.

```toml
[geometry]
texture_repair_attempts = 1
cave_normal_convention = "opengl"
```

The budget is **0 or 1**. The default permits one normal-map repair pass per
distinct source and one package rebuild. Zero keeps inspection enabled and
rejects maps that need repair. `directx` explicitly declares a cave normal map
whose green channel must be inverted to OpenGL convention. The pipeline never
guesses this from filenames or appearance. Rocky event normals retain their
OpenGL contract.

- Color, normal and roughness maps become shared, portable PNG copies, respecting
  aspect ratio and the configured size limit without upscaling. Different map
  resolutions are allowed. Dark 16-bit roughness maps are scaled by their storage
  range, not auto-contrasted.
- Usable normal vectors outside the length tolerance are normalized after
  resizing, then checked again. Undefined vectors, missing maps and corrupt files
  stop acceptance; no substitute material is invented. Color is sRGB; normals
  and roughness are linear data. There is no automatic color-space inference or
  tile-border painting.
- Displacement is decoded and hashed without quantizing its master. The existing
  visual-surface loop can reduce or disable its amplitude when geometry fails.
  UV-collapse repair and the portable UV/tangent checks also remain active.
- GLB map pixels, bindings, repeat samplers, normal strength and roughness
  channel packing are compared with the accepted maps. OBJ references must
  resolve to packaged copies. Complete PBR GLBs require the Blender/Unity/Unreal
  continuous-material bundle, correct settings, exact embedded images and
  current adapter files.
- A texture-package failure triggers at most one reserialization from the same
  accepted geometry and maps, followed by reinspection. It never changes the
  network seed or weakens geometry, memory or file-size limits. If it still
  fails, the previous published export stays in place.

Each successful export contains `texture_recovery.json`: source/content hashes,
map measurements, repairs, rejected package attempts and accepted files. The
run's `pipeline_quality_report.json` includes this evidence; campaign cold
replays compare the texture journal's hash. Failures carry the same diagnostics
in the quality report's inspection field. Progress messages distinguish decoding,
map repair, binding checks and package rebuilds.
The version fingerprint includes the shipped shader and material adapter files,
so changing them invalidates earlier checkpoint/campaign compatibility.

This verifies material data and packaging. A valid directional tile can still
show **UV chart boundaries**: apply the continuous material below in Blender,
Unity or Unreal. Standard GLB imports use the portable UV material. The gate does
not certify lighting, geological appearance, tile seamlessness or native shader
compilation. Material-only revisions made with `scripts/texture_inspection.py`
retain their separate material-revision report; they are not full pipeline runs.

## Texture an existing inspection run

Use the original neutral GLB and a current project configuration. This does not
rerun network generation, mesh extraction, smoothing or displacement:

```bash
uv run --no-sync python scripts/texture_inspection.py \
  outputs/earth_long_interconnected \
  --config config/earth_long_interconnected_full.toml \
  --output outputs/earth_long_interconnected/textured_4k

blender --background --threads 6 --python-exit-code 1 \
  --python scripts/create_blender_inspection.py -- \
  outputs/earth_long_interconnected/textured_4k --mapping triplanar
```

The output directory must be new. The helper supports one neutral `cave_wall`
primitive with existing UVs and tangents, without rock props. To try another tile,
start again from the neutral source and choose another output directory. It rejects
geometry displacement settings rather than silently ignoring them.

The three application folders receive byte-identical GLBs. A new native Blender
scene packs the imported images. `material_revision.json` records source and output
hashes, map hashes, image dimensions and exact equality of positions, indices,
normals and tangents. UVs are uniformly rescaled to the requested tile size. The
original run, its stage figures and its generation evidence remain available.
`blender_import_check.json` records the actual native import, including packed
images and color spaces. These checks do not constitute a Unity or Unreal import test.

With `--mapping triplanar`, the native scene is instead named
`plume_continuous_inspection.blend`; `blender_continuous_check.json` explicitly
records that its native shader differs from the portable GLB. Its images go to
`previews/continuous/`, preserving the original UV comparison images and report.

## Continuous materials for Blender, Unity and Unreal

Complete PBR GLB target exports now include one `continuous_material/` bundle
alongside the portable asset. All-target exports share one bundle. It extracts
the embedded PNG bytes directly, so there is no EXR conversion, additional atlas,
image resampling or geometry copy. Three object-space projections are blended
with `abs(normal)^4` weights. Normal-map slopes are oriented in each projection's
frame, blended, and applied relative to the unperturbed surface normal. This
avoids the atlas seams and incorrect normal directions seen in floor renders.

For an existing textured GLB:

```bash
uv run --no-sync python scripts/package_continuous_materials.py \
  outputs/RUN/export_blender/plume_cave_scene.glb \
  --output outputs/RUN/continuous_material --tile-size-m 4
```

Use a new output directory. Follow the bundle's [material setup guide](../src/plume_advanced/material_assets/README.md):

- Blender: run its `apply_blender_material.py` on the open cave, or give separate
  `--source` and `--output` paths to preserve the original scene. Images stay packed.
- Unity URP: copy the bundle inside `Assets`, then use its Tools > PLUME menu to
  create a material with the correct texture import settings. Assign it to the cave.
  A Custom Function is also included for URP/HDRP Lit Shader Graph integration.
- UE5: execute `unreal/create_material.py` with the Python Editor Script Plugin.
  It creates a new content folder and connected Default Lit material. Check
  compilation and assign the material to the imported cave.

The GLB keeps its standard material. A custom shader does not transfer through
an ordinary glTF import automatically; its [texture bindings](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
refer to mesh UV coordinates. The native adapters share the same shading method,
but importer axis conventions and application lighting can affect appearance.
The included URP shader uses light probes and forward lighting, with shadow and
depth passes; advanced renderer features need the Lit Shader Graph integration.

Cost is nine map samples instead of three. Texture count and polygon count are
unchanged. The normal map must remain **raw linear RGB** in these custom engine
materials; their shaders decode it explicitly. This differs from the engine's
usual normal-map import/compression setting. The installers configure this.
Projection blending does not repair actual topology, intersections or displacement
already baked into geometry.

Regression tests cover image-byte preservation, neutral-material handling,
determinism, invalid settings, native Blender normal orientation on floors,
ceilings and oblique walls, roughness channel interpretation, and a nonconstant
texture's invariance to broken UV charts. Optional HLSL checks compile the shared
body and the actual URP passes against official Unity headers. Native Unity and
Unreal editor/render verification is a separate, opt-in check. The repeatable
runner below exercises installed editors rather than inferring compatibility
from shader source or package contents.

## Native engine inspection

Run `scripts/check_native_engines.py` against an accepted **attempt directory**
containing `pipeline_quality_report.json` and `export/plume_cave.glb`. This focused
Linux fixture supports a rock-free cave with three 4K maps and at least 50 passage
samples. It creates new projects; it does not open or edit your simulation project.

```bash
uv run --no-sync python scripts/check_native_engines.py \
  outputs/CAMPAIGN/case_0000/attempt_0000 \
  --output outputs/native_inspection \
  --unity /path/to/Unity \
  --unreal /path/to/UnrealEditor
```

Either editor argument can be omitted. The tested toolchain is Unity 6000.6 with
URP 17.6 and glTFast 6.20, plus Unreal 5.8. The isolated Unity project pins those
packages; its first launch needs registry access and an active Unity license.
Both checks need a working GPU/display environment. The timeout defaults to
30 minutes per editor invocation and can be set with `--timeout`.

The command imports the actual GLB, calls the shipped native material installers,
checks geometry counts and passage collisions, verifies map resolution/color
spaces, and renders two interiors plus a disabled-normal-map control. Unity also
compares every imported vertex and deliberately corrupts UVs to check that the
continuous material is independent of UV charts. Unreal checks mesh bounds and
compiled shader availability. The inspection light stays at the verified passage
centre. Each view allows up to eight trials, halving light intensity until at
most 0.5% of pixels clip to white. Trial images and accepted light intensities are
retained. Blank, missing, or more than 1% white-clipped final images fail the
runner. This catches blown-out floor details that a contrast-only image check
could accept. Images still need visual review; these checks do not certify
geological realism, exposure agreement between engines, or every viewpoint.

Results, editor logs and images remain in the chosen output directory. Open
`unity_project/Assets/PLUME_Inspection.unity` or the Unreal project and its
`/Game/PLUME_Run01/PLUME_Inspection` map. `native_summary.json` records acceptance;
an editor error or failed check makes the command exit unsuccessfully.

Unreal 5.8 enables Nanite during GLB import by default. Its fallback mesh can be
much coarser and affects complex collision. This inspection disables Nanite and
uses full-resolution complex collision to compare with the accepted source.
Texture streaming is disabled only in the isolated inspection project so captures
use resident maps. Evaluate Nanite, collision simplification, streaming and texture
compression separately before choosing simulation performance settings.

## Import settings

- **Blender:** import the GLB, or open `plume_textured_inspection.blend` in the
  material revision directory. It starts in Material Preview shading at a lit
  interior camera. Press F12 for the camera render. Solid viewport shading hides
  image materials even when they are packed correctly. The earlier
  `plume_full_inspection.blend` in the original export directory remains neutral.
- **Unity:** import the GLB with a compatible glTF importer such as
  [Unity glTFast](https://docs.unity.com/en-us/asset-transformer-sdk/2026.4/manual/sdktips/export-guidelines).
  Keep imported materials and normals/tangents. The importer must support the active
  render pipeline. A gray object can also indicate an overridden material; a pink
  object typically indicates a shader problem.
- **Unreal:** use the provided GLB/Interchange guide and preserve normals/tangents.
  Check the imported material and normal convention in the installed engine version.

The standard material bindings follow [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html):

| Map | Interpretation |
| --- | --- |
| Base color | sRGB, with a neutral white multiplying factor |
| Normal | Linear data, tangent space, OpenGL/+Y convention |
| Metallic–roughness | Linear data: roughness in green, metallic in blue (zero for this rock) |

Do not connect the glTF metallic–roughness image directly to Unity's
metallic–smoothness input when making a material manually: channel layout and
roughness/smoothness semantics differ. Prefer the importer's material conversion.
The cave material uses repeat wrapping, and the GLB includes tangent vectors.
Normal-map data must not receive an sRGB color conversion.

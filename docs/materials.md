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
more frequent detail; larger values make the rock features larger. UV charts use
metric area scaling, so local distortion and chart seams can remain. It is not a
seamless triplanar projection. Judge the material inside ordinary passages and at
junctions before choosing 8K. A different scanned basalt tile can be substituted
without changing generation. Broader color variation or triplanar engine shaders
are possible future improvements if repetition or chart seams prove distracting.

Install Git LFS and fetch the master assets (`git lfs pull`). Pillow reads PNG/JPEG;
EXR inputs also require ImageMagick's `convert` command with EXR support. The exporter
embeds PNG images in the GLB, so importing the delivered GLB does not require EXR
support, ImageMagick, or access to the original `texture/` directory. Explicitly
configured missing or undecodable maps fail when strict loading is enabled.

The general default keeps its 1024-pixel export budget and 8 m tile for compatibility.
The two full interconnected presets request 4K/4 m. Geometry-only and Stage A–C
presets remain useful without texture dependencies.

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
  outputs/earth_long_interconnected/textured_4k
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

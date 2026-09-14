# Floor material review — 12 September 2026

The short interconnected cave's floor showed abrupt changes in the rock pattern
and apparent relief. Controlled renders of the same mesh, camera and point light
isolated the cause: boundaries remained with the normal map disconnected and
vanished with a plain material. The normal map amplified independently oriented,
packed xatlas chart boundaries. Valid UV triangles do not imply a visually
continuous texture. The earlier collapsed-triangle repair addresses a different,
localized defect and does not solve this problem.

## Change and delivery

`outputs/earth_short_interconnected_seed20260912_fixed/continuous_material_revision/`
contains a separate `export_blender/plume_continuous_inspection.blend`, two rendered
views, and `engine_materials/`. The original scene, source GLB and generation
artifacts are preserved. No network, mesh or rocks were regenerated.

The native material samples the original three 4096-pixel images in object metres.
Three signed projections use smoothly normalized `abs(N)^4` weights. Normal-map
slopes are transformed into each projection's frame and blended tangentially to
the unperturbed normal, avoiding UV-frame discontinuities. The shader preserves
fine texture detail; geometry is unchanged. A 4 m tile and normal strength 1 are
retained. A floor-detail camera and point light make the revised floor inspectable.

Blender has a node material and a standalone application script. Unity has a URP
lit shader, an editor installer, and a Shader Graph custom function for URP/HDRP.
Unreal has an editor Python builder that creates a Default Lit material and image
assets. Both engine shaders use the same HLSL body, raw linear RGB OpenGL normal
maps, and the green channel of the packed metallic–roughness image. The UE builder
passes vertex normals through a Vertex Interpolator before pixel shading and
converts centimetres to metres. The full target exporter includes one shared
material bundle when a complete PBR GLB is present; neutral exports avoid this
extra material parsing. Shared files are recorded for all-target exports.

The standard GLB still uses UVs. These are native material adapters, not a custom
shader encoded as a portable glTF feature. Importer axis conversion may change
pattern orientation between applications. The original small unintended wall
surface loop is also unchanged: the source geometry still has `needs_review`
status. A material correction does not clear its topology gate.

## Validation

- **60 targeted tests passed**, plus 5 subtests: geometry/export compatibility,
  material revisions, file budgets, UV repair, new packages and native shaders.
- **27 native Blender normal cases** cover six axis directions, oblique surfaces,
  rotated objects, zero strength and nonzero signed normal slopes. Linear EXR
  samples match expected normals within `2e-4` per component. Denoising is disabled
  for these numerical tests; it would alter the measurement.
- A nonconstant image exposes the old UV material's sensitivity to changed chart
  coordinates while the projected material render is identical. Roughness is
  checked independently against the image's green channel. Images are reused.
- Both PNG-origin conventions compile in the shared HLSL test using glslang 15.1.
- **30 URP shader entry/variant compilations pass** using DXC 1.8.2505.32 and the
  official Unity Graphics `2022.3/staging` headers: forward, shadow and depth
  vertex/fragment entry points with base, shadows, instancing/fog, screen-space
  shadows and Forward+ variants. HLSL 2018 and legacy macro expansion match these
  Unity headers. Warnings about float-to-half conversion are retained by DXC;
  compilation is not an in-editor rendering test.
- The actual 4,324,056-triangle Blender scene is saved, independently reopened,
  and checked for exact geometry, UV, normal and object-transform preservation.
  Three 4K maps remain packed. Floor and second-interior renders are inspected.
- Project-wide Ruff checks and targeted mypy checks pass. A built wheel includes
  all adapter resources; the GLSL/DXC tools and Unity headers are temporary test
  dependencies, not runtime dependencies or bundled source.

The HLSL source uses the engine's regular lighting/shadow support, but neither
Unity nor Unreal is installed in this workspace. Their native editor import,
material compilation and rendered appearance are **not yet verified**. The URP
shader uses SH/light probes; baked lightmaps, deferred G-buffer output, motion
vectors and a depth-normal SSAO pass need the Lit Shader Graph integration.

## Repeating checks

```bash
PLUME_BLENDER_BINARY=/path/to/blender \
PLUME_GLSLANG_BINARY=/path/to/glslangValidator \
PLUME_DXC_BINARY=/path/to/dxc \
PLUME_UNITY_HEADERS=/path/to/Unity-Graphics-root \
uv run --no-sync pytest tests/test_projected_materials.py tests/test_blender_material.py
```

Compiler/native tests explicitly skip when their optional tools are absent.
Do not interpret a skipped test as a pass. The Unity include root must contain
`Packages/com.unity.render-pipelines.universal` and `Packages/com.unity.render-pipelines.core`.
The test does not download dependencies or modify engine headers.

## Cost and references

Projection requires nine map samples instead of three, while image count,
geometry count and the number of cave material slots remain unchanged. Profile
GPU shading and image compression on the simulation hardware. A larger unique
bake is unnecessary for this correction. Repetition, projection blending and the
rock scan's appearance remain artistic limitations.

Implementation references: [glTF material and texture specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html),
[Unity URP shader source](https://github.com/Unity-Technologies/Graphics/tree/2022.3/staging/Packages/com.unity.render-pipelines.universal),
[Unreal custom material expressions](https://dev.epicgames.com/documentation/unreal-engine/custom-material-expressions-in-unreal-engine),
and [Unreal vertex normals](https://dev.epicgames.com/documentation/unreal-engine/vector-material-expressions-in-unreal-engine).

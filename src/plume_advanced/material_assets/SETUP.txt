# Continuous rock material

This bundle removes UV chart boundaries from material sampling. It uses the same
three images embedded in the source GLB, projected in three object-space directions
and smoothly blended. Normal-map slopes are oriented into the same frame before
blending; connecting a box-projected normal texture to a UV Normal Map node would
reintroduce incorrect shading.

The GLB is unchanged and retains its portable UV material. These are additional
native shaders; they are not carried automatically by an ordinary GLB import.
The shared HLSL body supplies both Unity and Unreal. `settings.json` records the
source hash, exact image hashes, tile size and normal strength.

## Blender

With the cave scene open, run `apply_blender_material.py` in the Scripting workspace.
It replaces only `cave_wall`'s material, packs the same images and leaves saving to
you. To produce a new scene without changing the original:

```bash
blender --background --python apply_blender_material.py -- \
  --source /path/to/original.blend --output /path/to/new_continuous.blend
```

The output must not exist. A companion `.material.json` checks exact geometry,
UVs, normals and object-transform preservation. Mesh coordinates are metres;
apply object scale before selecting a physical tile size.

## Unity URP

Copy this bundle (including the `unity`, `textures`, shared HLSL and settings files)
into a new folder under `Assets`. The supplied lit shader targets URP 14/17 APIs.
Use **Tools > PLUME > Create continuous rock material (URP)** and select this
bundle's `settings.json`. Assign the new material to the imported cave renderer.
The installer configures repeat, mipmaps, sRGB color and linear data maps.
**The normal PNG uses Default texture type with sRGB off**, preserving raw RGB;
this shader performs its own normal decoding. Do not let Unity swizzle/compress it
as a Normal Map texture. Packed roughness is sampled from green and converted to
smoothness. Check shader errors in Console before assignment.

This forward lit shader includes additional lights, main/additional shadows, fog,
shadow casting and a depth-only pass. It uses ambient SH/light probes. Baked
lightmaps, deferred G-buffer output, motion vectors and an SSAO depth-normal pass
are not supplied; use a Lit Shader Graph integration when those features matter.
URP's deferred renderer can draw this material with its forward-only pass.
The images are initially uncompressed for correctness. Select a suitable RGB data
compression format later and compare the normal result on the target hardware.

### Shader Graph (URP or HDRP)

In a Lit graph add a **Custom Function**, File mode, `unity/PlumeTriplanar.hlsl`,
name `PlumeTriplanar`, **Float** precision. Inputs in order:

- `ColorMap`, `NormalMap`, `RoughnessMap`: Texture2D, repeat samplers.
- `PositionObject`, `NormalObjectInput`: Vector3 from Position/Normal Vector nodes,
  both set to Object space. Use the unperturbed normal.
- `TileSize`, `NormalStrength`, `BlendExponent`: floats, defaults from settings and 4.

Outputs: `BaseColor` Vector3, `Smoothness` Float, `NormalObject` Vector3. Connect
color/smoothness to the Lit blocks. Transform `NormalObject` from Object to World
as a normal and connect to the Normal (World Space) block. Metallic is zero.
Create a material from that graph and bind the same three images. The default
axis conversion assumes glTFast's X reflection of glTF Y-up geometry; a different
importer can rotate the pattern but does not restore UV seams. Native HDRP graph
compilation/rendering must be checked in your project.

## Unreal Engine 5

Enable **Python Editor Script Plugin**. Choose **Tools > Execute Python Script**
and select `unreal/create_material.py`. It imports the three PNGs and creates
`/Game/PLUME_Continuous/M_PLUME_Continuous`. It refuses an existing destination;
for another bundle call `create_material(bundle_root, destination)` with a new
content folder. It does not change actors or assign materials automatically.
Open the material, check compilation, then assign it to the imported cave mesh.
The builder stops if the engine reports shader compilation errors. Its transform
connections use the unnamed first input exposed by Unreal's material scripting API.

For initial geometry/collision validation in UE 5.8, disable **Build Nanite** on
import (or disable Nanite on the imported mesh and rebuild). The default Nanite
fallback can be substantially reduced and is used for complex collision. Compare
clearances before selecting a reduced fallback or collider for simulation.
Allow texture streaming and shader compilation to finish before judging a render.

The builder creates a Default Lit, one-sided material. It uses object position
converted from centimetres to metres, an unperturbed local normal, the shared
projection code, and a world-space Normal output (`Tangent Space Normal` disabled).
Raw linear RGB normal data is retained using VectorDisplacementmap compression;
do not switch that texture to Unreal's normal-map sampler/compression convention.
The PNG sampler's vertical convention is handled explicitly in the shared code.
Keep object scale uniform; apply scale to the mesh if necessary. Importer axis
conversions can rotate/mirror the pattern relative to Blender; scale, continuous
mapping and normal orientation are independent of UV charts.

## Cost and limits

The shader samples three maps along three axes: nine texture samples versus three
for the UV material. It reuses three textures, adds no polygons, and requires no
unique texture atlas per kilometre. RGB(A) 4K texture memory plus mipmaps is still
substantial; PNG file size is not GPU memory. Profile shading on the actual
simulation hardware. This is not a fix for holes, intersections, baked displacement
or topology errors. Texture repetition, the chosen rock scan's geology, and some
soft blending between projections remain visible.

Bundling is not native engine validation. Blender renders and shader compiler tests
are reported separately. Unity/Unreal editor import and final appearance require
those applications; no claim of native validation is made by these files alone.

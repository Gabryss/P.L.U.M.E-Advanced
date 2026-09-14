# Short full-generation inspection — 12 September 2026

This follow-up exercises root seed **20260912** with the full Earth interconnected
preset: 400 m downstream target, three source systems, fixed 8 cm voxels, 4K PBR
maps, and no optional events, rocks or separate collision mesh. It retains the
host-field-driven network and actual cross sections. This is one intended cave,
with an interrupted first build and a rebuilt inspection version.

## What the full-resolution run exposed

The initial run on `dce5168` passed all 50 network/section checks but contained a
detached air pocket: 45 density samples, with lattice extent 6 × 11 × 1, creating
a 248-triangle shell measuring roughly 0.54 × 0.90 × 0.056 m. Its checkpoint and
failure evidence remain in `outputs/earth_short_interconnected_seed20260912`.
That run is explicitly marked rejected; its original manifest is preserved.

The bounded surface cleanup now removes isolated, one-sample-thick air sheets
spanning at most 16 samples per axis, in addition to the existing eight-sample
speck rule. It protects every refined route centre and domain boundaries,
preserves resolved cavities, and runs after surface filtering but before roof
stability. Dense and tiled cases agree across seams. The actual failing density
crop is retained as a 6 KiB regression fixture.

An in-memory check of the original volume changed exactly the 45 detached air
samples and restored one component. Rebuilding the production pipeline also
moves pocket cleanup after fissure closing: the resulting density differs at
194 unique samples (69 change phase). These are distinct measurements; the
fresh full rebuild must not be described as changing only 45 samples.

## Verification performed

- 52 targeted geometry, relief, tile, voxel-topology and stability tests passed.
- The full suite passed: 593 tests and 59 subtests; two optional tests were skipped
  (native Blender was not enabled in that sweep; Manim is not installed).
- Project-wide Ruff passed; the geometry module passed mypy.
- The optional native Blender material regression was then enabled separately
  and passed, including its actual Cycles render. Manim remains unavailable.
- Fresh-process host/network/section replay passed with different Python hash
  seeds, matching the full run and the original rejected run exactly.
- The rebuilt raw mesh has one component, consistent winding and a watertight
  surface: 2,162,026 vertices and 4,324,056 triangles.
- The primary route is approximately 410.7 m; total passage length is 989.95 m.
  Three source systems interact through three merges and two splits.

## Remaining defect — do not report an unconditional pass

The raw surface has genus **2**, while the accepted network has cycle rank **1**.
A tree/cotree decomposition locates the extra microscopic handle at approximately
`(5.7, -98.8, 168.3)` metres in PLUME's Z-up coordinates. Its short loop has bounds
of roughly 0.24 × 0.19 × 0.27 m. The other large surface cycles follow the intended
network island. The small handle is an unintended surface artifact, although the
mesh remains connected and watertight.

General opening and wider closing were evaluated on **in-memory copies only**.
Opening fragmented the volume and introduced further handles; wider closing
retained the extra handle. Neither experiment changed production code, the
saved checkpoints, or the delivered geometry. The isolated-pocket cleanup above is the only change to volume geometry in
this follow-up.

The rebuilt inspection output is
`outputs/earth_short_interconnected_seed20260912_fixed`. Its
`inspection_status.json` marks it **needs_review**. Completing export is an
operational success, not a claim that its topology matches the network. The
portable, section-cut and native-import reports beside the asset provide the
remaining inspection evidence. This run does not establish reliability for all
seeds or continuous traversability.

## Atlas repair after export validation

The first GLB passed 40/41 portable checks. Atlas face 89615 had the same UV
for all three vertices despite a valid spatial area of about 0.24 cm². The
exporter now checks its final float32 UVs and gives collapsed triangles local
metric charts before computing tangent frames. Healthy charts and all spatial
corners are preserved; shared UV vertices are split only where necessary.

The existing inspection GLB was repaired with the same production function.
Its failed chart used three isolated vertices, so no vertices, indices or
positions needed changing. Only three UVs and tangent entries changed; packed
images and all other attributes were verified byte-identical. Matching fallback
OBJ texture coordinates were updated. Original files, the prior manifest,
source snapshot and checksums remain in `validation/uv_repair`. The run manifest
records this postprocessing separately from the generation source.

The repaired portable asset passes **41/41 checks**. The export follow-up suite
passes **47 tests and five subtests**, with its optional native test skipped;
project-wide lint and module type checks pass. This follow-up is distinct from
the earlier 593-test full suite. The unresolved surface genus mismatch remains
flagged and was not hidden by the UV repair.

## Final inspection artifacts

- Textured GLB: approximately 217 MiB, 4,324,056 triangles, 2,315,463
  vertices including UV seams; no separate collider or rock meshes.
- Blender 5.2.0 LTS reimport preserved triangle count and bounds within 2 mm.
  All three 4096 × 4096 maps are packed, with linear normal/roughness data and
  connected shader inputs. The saved `.blend` was reopened for preview rendering.
- 34 exported-mesh section cuts returned closed contours, including two rooms
  and eight low-clearance samples. These are spot checks.
- Unity and UE5 packages carry byte-identical GLBs; native engine imports were
  not executed.
- Both raw and exported surface meshes are connected, watertight and consistently
  wound. Their genus remains 2 instead of the network's cycle rank 1, so the
  overall inspection status is **needs_review**.

The native scene and all four initial renders were saved before a background
audio-shutdown stall. That isolated process was interrupted. A second Blender
process reopened the saved scene and rendered readable exterior previews with
+4 stops exposure, then exited cleanly. The native file was not modified.

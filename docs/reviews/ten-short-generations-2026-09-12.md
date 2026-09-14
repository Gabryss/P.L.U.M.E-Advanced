# Ten short Earth caves: generation and regression review

The inspection set contains five single-source caves and five interconnected
multi-source caves. Each requests a 250 m main route at 0.10 m voxel spacing,
using root seeds 0, 17, 42, 20260912 and 4294967295. All branches count toward the
separately reported total passage length. Structural events and rock props are
disabled, and each delivered cave uses the same reusable 4K material.

The generated [inspection index](../../outputs/ten_cave_review/README.md),
[visual gallery](../../outputs/ten_cave_review/index.html) and
[machine-readable checks](../../outputs/ten_cave_review/summary.json) contain the
per-case results and direct links to the final assets. The final inspection
files are under `outputs/ten_cave_review/final`; exploratory, rejected,
interrupted and replay runs are retained separately as evidence.

![Section-envelope comparison](../../outputs/ten_cave_review/network_comparison.png)

The comparison uses Stage-C section-envelope unions, with a common metric
scale and a centred lateral axis. It is a network/section diagram, not a
silhouette extracted from the final mesh. Actual Blender previews, both
textured and with a neutral material, are provided in each case.

## Final results

All ten delivered cases passed the specified mesh, export and native inspection
checks, and all ten fresh-process geometry replays matched exactly. Blender
checked **4,919 saved section centres**; **815 transverse cuts** produced closed
contours around their sampled axes. There are **65 stage figures**, plus exterior,
interior and junction/chamber previews. No rock props were generated.

| Case | Main / all passages (m) | Triangles | GLB (MiB) | Added relief scale | Surface attempts |
|---|---:|---:|---:|---:|---:|
| single_0 | 256.8 / 387.3 | 1,126,612 | 101.8 | 1 | 1 |
| single_17 | 254.3 / 324.6 | 1,164,088 | 102.9 | 0 | 4 |
| single_42 | 252.2 / 325.5 | 1,087,404 | 99.8 | 1 | 1 |
| single_20260912 | 253.7 / 384.4 | 1,128,616 | 101.4 | 1 | 1 |
| single_4294967295 | 255.6 / 388.8 | 1,210,300 | 104.7 | 0.5 | 2 |
| multi_0 | 270.4 / 641.5 | 1,760,660 | 124.6 | 1 | 1 |
| multi_17 | 284.8 / 697.2 | 2,100,730 | 136.6 | 0 | 6 |
| multi_42 | 295.8 / 605.1 | 1,880,368 | 128.3 | 0 | 4 |
| multi_20260912 | 250.7 / 634.4 | 1,855,880 | 127.8 | 1 | 1 |
| multi_4294967295 | 279.0 / 648.4 | 2,025,570 | 133.7 | 0 | 6 |

Four cases omit the added accretion relief and one uses half its requested
amplitude. The accepted cross-section morphology and base wall variation remain;
this detail reduction is explicit, not an assertion that every case retains the
full requested relief. Both full and reduced-detail cases still use the same 4K
material. Bright distant patches in some textured views are illuminated by the
second inspection torch; neutral views use only the active camera's torch.

The standalone GLBs are approximately **100–137 MiB each**, with **1.09–2.10
million triangles**. Diagnostic checkpoints, fallback formats and replay evidence
make the complete output directory larger than the importable model itself.
Use one shared material/texture set when loading several caves together.

## Defects found and corrected

The first diagnostic pass rejected four of the five multi-source meshes.
Tighter centre checks and corrected clearance handling also exposed cases
that needed detail reduction. The rejected files and logs remain available;
they have not been replaced by successful results from different root seeds.

1. **A root seed did not necessarily vary the whole host scenario.** Reseeding
   an already resolved configuration retained the original sampled host range
   values. The loader now accepts the override before resolving ranges, and
   the reliability runner uses that path. Its tests compare the complete result
   with an explicitly edited TOML seed. The separate scientific experiment
   helper that deliberately holds resolved host values fixed retains that role.
2. **A closed mesh could still contain detached shells or unintended handles.**
   The surface gate now checks component count and Euler characteristic against
   the accepted connected graph. For the event-free tube volume, surface genus
   must equal `edges - nodes + 1`. This check runs before floor sampling/export,
   and is repeated on the final exported mesh after joining only identical
   seam positions. It complements the existing closed-manifold check.
3. **Relief on a thin branch borrowed clearance from a nearby tall gallery.**
   Clearance propagation is now confined to the same vertical air run. Detached
   air regions without any sampled route centre are removed before fissure
   closing. Every seeded component is retained, so cleanup cannot conceal a
   disconnected real branch by keeping only the largest cavity.
4. **Some valid networks still developed grid-scale connections during volume
   construction.** Surface acceptance now tries a bounded sequence on the same
   immutable swept volume: the requested relief, then half, quarter and zero
   added relief; omission of requested closing; and finally a one-voxel
   grayscale opening to remove unresolved air bridges. Each candidate must
   retain every refined route centre, satisfy roof screening and pass the actual
   mesh topology check. A candidate that disconnects a real passage is rejected.
   Unexpected programming exceptions are not swallowed as random bad seeds.
5. **The initial single-source export in this campaign was neutral.** The
   geometry preset intentionally has no maps. The completion step now applies
   the maintained 4K material to both modes. Already exported neutral geometry
   was reused for a verified material-only revision, preserving positions,
   normals, tangent frames and indices. The failed neutral exports and revision
   provenance are retained. Shared prepared maps were checked pixel-for-pixel
   against a directly converted export before reuse.
6. **Interior previews were too dark for inspection.** The native inspection
   scene uses +3 EV exposure for textured interior cameras and zero exposure for
   exterior views. The dark textured outer surface was unsuitable for judging
   plan geometry, so the gallery uses neutral-material exterior views. Separate
   neutral-material interior previews expose the geometry
   without the normal map; these use only the active camera's inspection light.

The final opening step was exercised by the seed-17 and maximum-uint32
multi-source cases. Reports retain every rejected surface attempt and the
selected detail/closing/opening settings. A zero relief scale omits the added
accretion layer; it preserves cross-section morphology, base wall variation and
material detail. The generator does not silently replace the root seed, switch
to a different accepted network or relax a topology threshold to obtain a pass.

Some export workers ended with SIGTERM (exit 143), without a Python
geometry exception. Their logs are preserved; the cause of the termination was
not established. The last case uses an inspection-only checkpoint wrapper that
memoizes unchanged xatlas calls by exact input arrays, xatlas version and source
identity. It preserves completed UV batches across interruption and still passes
the normal export validators. This does not change the network or mesh.

A further viewpoint check found that extending a local tangent by six metres
could aim a camera into the floor after a dip. New inspection cameras target
actual neighbouring section positions and verify their sight line against the
imported mesh. The affected maximum-seed multi-source view was regenerated;
the cave geometry was unchanged. Large-junction close-ups provide additional
textured and neutral views, with their camera coordinates and clearance saved.

## Regression coverage

The repository's density fixtures reproduce a thin branch affected by relief and
an unintended connection near a merge. Their artificial crop caps isolate
surface topology; they are not independent roof-stability measurements. Tests
cover deterministic recovery, finite exhaustion, immutability of the input
volume, exception propagation, preservation of all sampled centres, and
matching dense/tiled behaviour across grid boundaries. Opening is checked for
anti-extensivity: it cannot enlarge or join air regions. Resolved rock islands
survive the local filter, and any lost graph cycle still fails the surface gate.

Two old event-placement and floor-atlas smoke tests used a coarse general
preview that obstructed a sampled centre. They now use a resolved short gallery
while retaining their event, grounding, atlas and topology assertions. The
packaged minimal CLI example also uses a resolved short gallery. The general
body-scaled preview remains a coarse study: physically unresolved configurations
can fail acceptance and require a finer resolution or different dimensions.

The new sparse component counter replaces Python bookkeeping per mesh corner.
Base meshes that already pass acceptance are reused when structural events do
not modify density, avoiding a second polygonization. Opening and connectivity
cleanup operate on tiles with consistent halos rather than assembling a global
dense volume.

The full suite passed **627 tests and 59 subtests**, with **84.59% line
coverage**; 20 optional native/compiler cases were skipped in that portable run.
A separate enabled Blender/material/UV run passed **31 tests**, including the
optional native render and shader compilation checks. Ruff and mypy also passed.
These are reported separately because the material run overlaps the full suite.

## What the campaign verifies

- Fresh host generation, deterministic candidate selection, network/section
  quality, conserved flow and multi-source interaction checks.
- Finite density/mesh values, a single closed connected surface, expected genus,
  and preservation of refined centreline samples before export.
- Both floor-map passes, no generated rock objects, and portable PBR bindings.
- Closedness, winding, nondegenerate spatial triangles and topology of the
  exported, smoothed mesh; 41 portable checks per completed textured export.
- Actual Blender import, scale/bounds, triangle-count preservation, packed
  textures, shader connections, and roof/floor rays at every saved section
  centre. Additional transverse spot checks include ordinary reaches, wide
  rooms and shallow branches.
- Fresh-process A–C and raw-mesh replays with a different `PYTHONHASHSEED`.
  These compare exact mesh arrays and saved network/section data. They are not
  ten byte-identical GLB replays.

The 0.10 m grid is an inspection resolution. The input-envelope screen flags
729 of 4,919 profiles below its eight-samples-across-height heuristic. These
include thin side passages and tips. Actual centre and contour checks are
reported separately; a passing topology gate is not evidence that all small
features have converged with resolution. Floor-atlas cells also require 1 m
clearance, so their count excludes many shallow locations. Wide or tall
transverse cuts near confluences can include an oblique neighbouring passage;
they are not direct one-to-one errors against a single input profile.

This was a development and repair campaign, not a frozen-source timing
benchmark. Per-stage source identities and interrupted workers are retained;
fresh-process geometry replays record the final source independently. The source
snapshot includes the runtime, lockfile, maintained presets and inspection
scripts. Timing under concurrent meshing, chart generation and rendering should
not be used to rank implementations or hardware.

The single-source preset intentionally stays within a trunk-and-island style;
its five results are not five different geological endmembers. Multi-source
cases keep persistent parallel galleries with merges and subsequent splits.
A finite Earth campaign cannot establish success for every seed, resolution or
celestial body, or quantitative realism against a cave survey. Centre rays and
transverse cuts are sampled checks, not a continuous traversability or exhaustive
self-intersection certificate. Unity/Unreal material code is included and shader
checks were run; native editor imports still require those applications.

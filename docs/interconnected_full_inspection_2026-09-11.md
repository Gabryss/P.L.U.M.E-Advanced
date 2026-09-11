# Full interconnected inspection — 11 September 2026

Two complete Earth caves were generated from the interconnected-system model, with
three persistent systems per host and no loose rocks. This follows the
[six-case A–C campaign](interconnected_validation_2026-09-11.md). It adds full-volume,
surface, exported-asset and Blender checks; the initial six cases were previews.

## Accepted assets

| Case | Downstream extent (m) | Combined passages (m) | Merges / splits | Graph cycles | Triangles | Closed mesh cuts |
|---|---:|---:|---:|---:|---:|---:|
| Short | 400 | 1,015.6 | 2 / 2 | 0 | 4,525,300 | 30 |
| Long | 3,000 | 7,185.3 | 12 / 11 | 10 | 4,808,630 | 66 |

[Short inspection](../outputs/earth_short_interconnected/README.md) ·
[Long inspection](../outputs/earth_long_interconnected/README.md) ·
[Machine-readable batch audit](../outputs/interconnected_full_inspection.json)

Both cases pass 50 network and section checks, one-component/watertight/winding
checks on raw and seam-welded exported meshes, graph-cycle versus surface-genus
and plan-island checks, and 27 portable-asset checks. The sampled mesh cuts above
include ordinary, shallow and wide sections. Actual Blender import preserves
triangle count and bounds within 2 mm; both interior cameras have verified floor
and roof hits. All three engine GLBs are byte-identical. Native Unity and Unreal
imports were not run.

Each cave has 12 stage figures drawn from its saved checkpoints. The mesh overview
uses the actual Blender-imported mesh, and the long cave also has six consecutive
500 m mesh views. Figure rendering checks that protected generation artifacts do
not change. PNG decoding, local guide links and representative figures are checked
separately during delivery review.

## Rejections and effective settings

The full-mesh gate rejected four intermediate meshes before export: the original
short case at 20 cm (seven excess handles), and the three original long preview
variants at 20 cm (five, one and one excess handles respectively). Passing the A–C
envelope checks was insufficient. Rejected checkpoints remain under
`tmp/interconnected_full_20260911`; none is listed as an inspection release.

The short cave retains the original primary host, network and sections, with 8 cm
voxel spacing. Its final mesh preserves the expected topology. The high resolution
was selected for this case's shallow interiors; it is not a convergence certificate
for every seed or a guarantee of continuous passage clearance.

The long cave uses a new accepted preflight of
`config/earth_long_interconnected_full.toml`. Its host is identical to the original
long host. Effective changes are network seed 2302742253, section width-scale median
0.9 (previously 1.0), and minimum section-height control 1.2 m (previously 0.9 m).
Candidate 6, seed 233986284, passes without repair. Voxel spacing remains 20 cm and
extraction isolevel remains zero. The explicit mesh radius scale of 1.0 equals the
already resolved Earth default; no 20% radius correction was applied. Experimental
local repairs and alternative isolevels were diagnostic only and were discarded.

The final configuration change and bounded network search produced this accepted
long mesh. It does not establish a universal repair for near-junction overlap or
guarantee that arbitrary future seeds mesh correctly. Continue to apply the
independent full-mesh gate to every new asset.

## Reproduction and scope

Fresh full-generation processes reproduce their respective saved host, network,
sections and complete acceptance report exactly. The revised long preflight also
passes a separate replay with a different Python hash seed. Full mesh byte-level
reproducibility across machines was not tested. Input and resolved configs, seeds,
checkpoints, reports and asset hashes are saved beside each cave.

The [source/configuration archive](../outputs/interconnected_generation_source_2026-09-11.zip)
contains the exact production source, presets, standard inspection scripts and
dependency lock. Its [manifest](../outputs/interconnected_source_snapshot.json)
records every bundled source-file hash and the installed package versions. Third-party
dependencies and Blender are not included in the archive.

The production package is unchanged from the initial preview campaign, with SHA-256
`6a2bdbdcb89b5b3158af9b2a32684083c659395a54d9ab28b3d1d754836dc0a3`.
Only full-generation presets and inspection/report utilities were adjusted here.
The mesh and section checkers now cache their compressed section arrays, avoiding
thousands of identical decompressions on long networks; acceptance conditions are
unchanged. The full long check processes were already running with their original
loading loops, so these changes affect future check runs rather than their results.

These are single-level procedural Earth inspection assets. Surface detail is
limited by the selected voxel spacing. The section-height control precedes taper,
surface relief and meshing. The checks do not certify geological equivalence to
Valentine Cave, continuous human/rover clearance, or structural pillar stability.
Texture maps and separate collision meshes are not supplied.

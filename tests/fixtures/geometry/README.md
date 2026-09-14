# Full-resolution air-pocket regression

`seed20260912_air_sheet.npz` is a 6 KiB local density crop from a rejected
400 m Earth interconnected run (root seed 20260912, 8 cm voxels). It contains
part of the main passage and an isolated 45-sample air component with lattice
extent 6 × 11 × 1. The latter produced a detached 248-triangle shell, only
5.6 cm thick. The original run used production commit `dce5168`.

The test checks the actual density samples: the unresolved pocket closes,
and every other sample in the crop remains unchanged. Synthetic companion
tests cover tile seams, different isovalues and scales, retained resolved
cavities, route support, connected shelves, domain boundaries and bounded
repair extent. This is a numerical resolution repair, not a geological
collapse model or permission to discard separated routed passages.

`relief_branch_seed20260912.npz` is an 8 m local density crop from a 400 m,
three-system diagnostic with root seed 20260912 and 8 cm voxels. It includes a
thin breakout beside a taller passage. Artificial solid caps close the crop;
these caps are test boundaries, not cave geometry. The archive stores the exact
float32 stamped density, its origin, relevant relief configuration and interior
section centres. At full relief the local surface acquires extra handles; the
bounded acceptance pass accepts half relief, preserves its input, and reproduces
the same mesh in a second run. Roof stability for this artificial region is
outside that regression's scope; full-cave cases exercise it separately.

`merge_neck_seed17.npz` is a capped, grid-aligned crop of the stamped
250 m interconnected Earth case with fresh-host root seed 17. The raw
closed crop has one unintended handle near a merging thin branch; one
voxel of grayscale opening removes that handle. All 25 interior sampled
centres remain in air. Crop caps are artificial, so the test isolates topology
and does not treat this fixture as a roof-stability study.

`collider_float32_seed1.npz` contains the local triangles that lost area or
orientation when an accepted double-precision collider was imported as float32.
The test preserves triangle count, repairs within a 1 mm displacement budget,
checks metre and centimetre representations, and repeats exactly. Its adjacent
JSON records the source and extraction scope. It does not replace whole-mesh
topology, clearance and deviation checks on production exports.

`route_placement_seed42.npz` contains a 6,338-triangle open patch from a rejected
250 m multi-source Earth case. The source surface was reconstructed with exactly
the rejected surface hash. Four edges of the midpoint inspection path collide
with a step, but a local vertical path can carry the same 0.5 m capsule with a
0.02 m margin. Every triangle intersecting the candidate-body bounding box plus
0.5 m is retained, without clipping triangles. The adjacent JSON records bounds,
source path/stations and identity. This patch tests continuous distances and
bounded path search, not closed-surface topology or roof stability. Synthetic
closed-prism companions exercise the complete mesh acceptance gate.

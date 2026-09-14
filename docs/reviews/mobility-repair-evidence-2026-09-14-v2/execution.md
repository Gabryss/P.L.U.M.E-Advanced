# Clearance and repair campaign execution

This revision uses the same six designs as the previous textured campaign:
Earth, a 250 m dominant-route target, single and three-source configurations,
and root seeds 1, 42 and 4294967295. The host remains unchanged for each case.
The required dominant-route inspection body is an upright 0.5 m high × 0.5 m
wide capsule with 0.02 m margin. Optional passages are not enlarged solely to
meet this requirement. This is a geometric corridor check, not vehicle dynamics.

Production uses the original 0.12 m voxel grid and three reusable 4096² PBR maps
with 4 m tile size. Rocks/events and baked image displacement remain disabled.
Two independent groups run concurrently; editors run sequentially on the GPU.
Each original and cold replay has a 3600 s wall-time limit and an 8192 MiB address
space ceiling. Replays use different Python hash seeds and no original checkpoint.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/plume-check \
  --configs docs/reviews/mobility-repair-evidence-2026-09-14-v2/single_250m_4k.toml \
  --seeds 1 42 4294967295 --scope full --timeout 3600 --memory-limit-mib 8192 \
  --output outputs/mobility_repair_campaign_20260914_v2/single

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/plume-check \
  --configs docs/reviews/mobility-repair-evidence-2026-09-14-v2/multi_250m_4k.toml \
  --seeds 1 42 4294967295 --scope full --timeout 3600 --memory-limit-mib 8192 \
  --output outputs/mobility_repair_campaign_20260914_v2/multi
```

`run_native_campaign.py` inspects each accepted original in isolated Unity and
Unreal projects. The adapter requires source artifact receipts and checks the
dedicated collider independently of the visual mesh. Every planned branch view
is required. It records setup, mesh cooking, fixed ray-query timings and process
peak memory, not application FPS. Shader and lighting behavior can differ between
engines; image guards detect missing/blank/clipped captures rather than proving
geological realism or checking every texel.

## Preserved failure and export-only regression

The first revision is retained in `outputs/mobility_repair_campaign_20260914`.
Its first single-source case passed generation and Unity, but Unreal imported
477,480 collider triangles instead of 477,482. A valid double-precision mesh
contained triangles that collapsed or inverted during float32 conversion.
Simply returning the original full triangulation was also unsafe: it had its
own float32 degeneracies. These defects were not fixed by disabling Unreal's
degenerate removal or weakening the import-count check.

The final production repair checks metre and centimetre float32 representations,
tries bounded local neighbour relaxation, and repeats full mesh, clearance and
surface-error inspection. Collider OBJ coordinates round-trip exactly. V1
workers were stopped explicitly when this defect was confirmed; incomplete runs
are not counted as finished replays or accepted campaign cases.

`precision_regression.json` describes an export-only test using V1's accepted
geometry and unchanged visual material. It is not a fresh campaign generation.
The corrected collider retains 477,482 triangles and passes actual Unreal import,
357 collision probes and 46 material views. The small extracted defect fixture
is retained in `tests/fixtures/geometry/collider_float32_seed1.npz` for exact
replay and metre/centimetre orientation regression tests.

## Separate resolution study

`resolution_fixture.py` builds a real 20 m tube, fits the same required capsule,
and exercises an actual 0.2 m → 0.1 m refinement. It also requires rejection
when the permitted refinement cannot meet a 1e-12 m convergence tolerance and
when an eight-sample allocation budget is exceeded. Original/replay JSON files
must match exactly under Python hash seeds 11 and 37. This measures floor and
roof changes at fixed dense probes; it is not full-surface Hausdorff convergence.

A separate 250 m development study reached the expensive 0.03 m candidate and
was stopped before completion. It provides no claim of convergence. The six
production cases retain 0.12 m resolution to keep the comparison and simulation
asset sizes meaningful; refinement remains explicitly opt-in.

## Test execution notes

The complete Python suite and native material checks run against the frozen
production source in `protocol.json`. An initial DXC invocation used the
`Packages` folder as the include root; those 15 invocations failed before shader
compilation. Correcting the include root to its parent allowed all 15 actual
URP shader-variant tests to pass without source changes. The Blender fixture
passed its 27 normal-frame cases and channel/UV checks. Missing optional
glslang checks are not reported as executed.

`audit_campaign.py` verifies preserved recipe and source hashes, generated
artifact receipts, exact replay identities, clearance and collider requirements,
native evidence and separate resolution outcomes. The source remains frozen
throughout the fresh campaign. Failed seeds are retained, never replaced with
easier seeds.

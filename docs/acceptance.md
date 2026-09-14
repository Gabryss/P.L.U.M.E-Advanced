# Acceptance profiles

An acceptance profile declares what a full run must demonstrate before PLUME
publishes its target package. It is independent of `run.quality`, which selects
generation settings. A higher rendering or resolution setting alone does not
enable the corresponding acceptance requirements.

Both `plume-generate` and `plume-check --scope full` enforce the same policy.
The separate `plume-evaluate export-consistency` experiment also preserves the
selected policy and size limits when packaging its canonical cave for all targets.
The resolved policy is included in configuration manifests and checkpoint
fingerprints. Repairs cannot change it. A rejected staged export leaves an
existing published package intact.

## Choose the contract

| Requirement | `research` | `inspection` | `simulation` |
|---|---|---|---|
| Raw, visual and serialized mesh checks | Required | Required | Required |
| Integrity of declared textures/material package | Required | Required | Required |
| Required-route body clearance | Optional | Required | Required |
| Dedicated, inspected collider | Optional | Required | Required |
| Finite visual-triangle and per-file limits | Optional | Required | Required |
| Input-resolution screen or measured refinement | Optional | Optional | Required |
| Minimum procedural relief factor | No extra limit | No extra limit | 1.0 by default |
| Complete diffuse/normal/roughness set | Optional | Optional | Optional |
| Native engine validation within publication | Unavailable | Unavailable | Unavailable |

“Optional” means the profile does not require it; explicit geometry/export
controls still run their existing checks. Texture integrity is mandatory even
for a partial material. A deliberately neutral export passes that integrity
check without claiming that PBR textures were tested.

The packaged `default_project.toml`, the short/long single/multi comparison
presets, and the two `*_interconnected_full.toml` presets declare `inspection`.
Other maintained presets explicitly declare `research` for their studies. A
schema-4 file that omits `[acceptance]` resolves to `research`; existing studies
do not silently acquire stronger requirements. Frozen historical campaign
configurations and their reported results are unchanged.

Start from a maintained preset and add or edit its **one** acceptance table:

```toml
[acceptance]
profile = "inspection"
require_textures = true  # Enable only with all three PBR source maps configured.
```

For the stronger numerical simulation checks:

```toml
[acceptance]
profile = "simulation"
minimum_relief_scale = 1.0

# Edit these keys in the existing geometry/export tables, not duplicate tables.
[geometry]
resolution_refinement_attempts = 2

[export]
generate_collision = true
max_visual_triangles = 2000000
max_asset_bytes = 268435456
```

These are edits to a complete scenario, not a replacement for its host, network,
section and material settings. Use a fresh output directory when changing the
scenario or policy. Run `plume-check --configs YOUR_CONFIG --seeds YOUR_SEED
--scope full --preflight` before a costly generation. Preflight checks controls
and inputs; it cannot predict whether that seed's final asset will pass.

## Defaults and explicit contradictions

When a required control is **absent**, the loader supplies:

- Clearance: 0.5 m total height, 0.5 m width, and 0.02 m clearance margin.
- Dedicated collision export: enabled.
- Export limits: 2,000,000 visual triangles and 256 MiB per exported file.
- Required resolution: up to two refinement attempts.

Explicit settings remain explicit. For example, `generate_collision = false`,
zero export budgets, or 0.4 m required width conflict with `inspection` and fail
configuration loading. A geometry clearance larger than the policy minimum is
allowed. The policy dimensions are metres and are not scaled by the selected
celestial body. These limits are starting budgets, not evidence that every
preset fits the intended simulator. A long or fine-resolution run can exceed
them and be rejected.

The named profiles' mandatory flags cannot be disabled. `simulation` may choose
a positive `minimum_relief_scale` below 1.0 as an explicit permitted reduction;
zero is rejected. Optional requirements can also be enabled on `research`.

| Acceptance key | Default / meaning |
|---|---|
| `profile` | `research`, `inspection`, or `simulation` |
| `require_clearance` | Require actual raw/visual/collider route evidence at the policy dimensions |
| `require_collision` | Require a dedicated collider and successful inspection/serialization |
| `require_export_budgets` | Require positive `export.max_visual_triangles` and `export.max_asset_bytes` |
| `require_resolution` | Require the resolution evidence described below |
| `require_textures` | Require configured diffuse, normal and roughness maps and successful material inspection |
| `require_native` | Request integrated native validation; currently stops full runs as unavailable |
| `route_height_m` | 0.5; positive total height of the upright body |
| `route_width_m` | 0.5; positive width, no larger than height |
| `route_margin_m` | 0.02; nonnegative clearance margin |
| `minimum_relief_scale` | 0.0 for research/inspection; 1.0 for simulation; range 0–1 |

## What the requirements establish

**Clearance.** The declared route polylines must pass the existing continuous
upright-capsule checks on actual raw and prepared visual triangles and, when
required, the collider. Evidence must cover every required route and segment;
an empty, incomplete or undersized check does not pass. Equal height and width
describe a sphere, not a 0.5 m cube. This verifies collision-free body placement
and sweeps along those routes. It does not establish wheel contact, traction,
turning constraints or navigation through all optional branches.

**Resolution.** Every input profile must pass a screen of at least eight
samples across its smallest dimension, or the bounded refinement journal must
show successful floor/roof convergence with unchanged topology and nonempty,
unblocked comparison probes. Missing, empty or failed evidence blocks export.
An explicit `resolution_refinement_attempts = 0` is allowed: a sufficiently
resolved input can still pass; an under-resolved one cannot. The existing
allocation and refinement budgets still apply. This is a screening/convergence
criterion on input envelopes and base-mesh floor/roof measurements, not a proof
of convergence of every contour, event feature or texture displacement.

**Relief retention.** The lower bound is the global procedural relief factor
times the smallest local repair factor. Candidates below the configured bound
are recorded and skipped before expensive density copies. If no allowed surface
candidate passes, the existing bounded upstream recovery can try sections or
another deterministic realization in the same host. An exhausted recovery
fails. Export checks the factors again, including geometry supplied directly
through the API. No relief requirement is claimed when no procedural relief
was requested. This measures settings, not affected surface area or actual
roughness: clipping, density filters, mesh smoothing and texture displacement
need separate fidelity measurements. Visual displacement/smoothing reductions
remain visible in repair reports and are not governed by this particular bound.

**Export size.** The exporter enforces actual visual-triangle counts and file
sizes inside the publication transaction. It does not automatically decimate
the cave to fit. The byte limit is per exported file, not total package size,
GPU memory, texture mip memory or engine runtime cost. Dedicated collider size
and runtime performance still need their own measured budgets.

**Native checks.** `require_native = true` deliberately fails full-run preflight
with an `unavailable` result. Unity/Unreal/Blender checks exist as separate tools,
but are not integrated into the atomic publication transaction. An old editor
report cannot satisfy this flag. Keeping it false reports `not_requested`; it
does not imply an engine test passed. Integrating finite-body native sweeps and
binding their receipts to the current asset remains separate work.

## Read the result and resume safely

`export_TARGET/pipeline_inspection.json` and the run's
`pipeline_quality_report.json` contain an `acceptance` object:

- `policy`: the fully resolved immutable requirements.
- `checks`: each requirement's `required`, `status`, `detail` and evidence.
- `passed`: true only when every required row passed.

Statuses are `passed`, `failed`, `unavailable` or `not_requested`. A failure or
missing capability in a required row stops publication. On failure, the quality
report's `inspection` field carries the acceptance error when that gate caused
the failure; upstream topology/clearance failures retain their own diagnostic
reports instead. Neither is a passing asset.

Completion rechecks current requirements and compares the saved policy and
evidence. Missing or changed acceptance records fail completion. Normal resume
only reuses checkpoints with matching configuration, inputs, source and runtime,
then repeats export and completion checks. Campaign receipts additionally hash
delivered evidence. Reports are integrity checks for local workflow artifacts,
not signatures for untrusted downloads.

`plume-check --scope network` and `--scope sections` report the full contract as
`not_evaluated`; their success means those stages passed. The geometry-only
`generate_tube_only.py` research helper refuses full acceptance policies and
directs users to `plume-generate`. Low-level Python generation/export APIs
default to research; callers requiring a policy must pass it explicitly through
the supported pipeline/export APIs.

See [reliability](reliability.md) for repair budgets and
[remaining gaps](reviews/remaining-gaps-2026-09-14.md) for the broader validation
work. These profiles make acceptance explicit; they do not turn procedural
approximations into geological or simulator certification.

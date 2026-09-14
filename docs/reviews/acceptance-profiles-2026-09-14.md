# Acceptance profiles: implementation and verification

This change follows the gap assessment at baseline `535b987`. It makes the
publication contract explicit without claiming that all remaining scientific or
simulator validation gaps are closed.

## Implemented behavior

- Research, inspection and simulation profiles share one immutable policy.
- The packaged default and six main comparison presets require 0.5 m height,
  0.5 m width, 0.02 m margin, a checked collider and finite export limits.
- Simulation additionally requires resolution evidence and a positive procedural
  relief-factor bound (1.0 by default). Disallowed global/local reductions are
  rejected inside bounded surface recovery and checked again at export.
- Generation, full campaign workers and the export-consistency experiment retain
  the declared policy. Completion/resume rechecks policy and recorded evidence;
  changed policies invalidate checkpoint identities.
- Texture integrity stays mandatory. Complete PBR input requirements are explicit.
- Required native validation is reported unavailable and blocks full generation;
  existing separate editor checks are not silently promoted into this contract.
- Stage-only campaign results identify the full profile as not evaluated. The
  geometry-only helper accepts unrestricted research and directs other policies
  to the full pipeline.

The [profile guide](../acceptance.md) describes controls, defaults and limitations.
The [resolved inventory](acceptance-preset-inventory-2026-09-14.json) contains all
16 maintained configurations: seven inspection and nine research presets. This
is a configuration audit, not a claim that each preset fits its runtime budgets.

## Verification

The frozen-source full suite passed **976 tests and 59 subtests**, with 20 optional
checks skipped, in 1,265.79 seconds. Statement coverage was **85.94%**, above the
70% required gate. **74 new acceptance tests** were added, and the CLI recovery
integration test also now runs with both research and inspection policies.

Ruff passed across the repository; mypy passed on 111 source files; compilation
and dependency checks passed. The wheel and source distribution built, and the
wheel installed into an isolated temporary target. Its production-source hash
matches the tested checkout; the inspection default and all eight material
support resources are present. The wheel is 458,724 bytes; the source archive
is 557,782 bytes.

The real all-target export test produced 314,324 visual triangles and 62,864
collider triangles. Five target packages passed the inspection contract, and
matching-case reuse passed. This exercises portable exports, not native editors.


The new tests include real box-passage export with two required routes and a
collider, deterministic export replay, a real 20 m volume refinement and export,
a saved branch whose local relief reduction is refused by the strict policy,
a full simulation worker, CLI recovery/resume under research and inspection,
and real tiny PBR maps packaged with material adapters. Fixed proposals in
orchestration tests exercise production screening and meshing; they are not
new stochastic campaign coverage.

Negative tests cover missing/empty/undersized route evidence, omitted paths and
wrong segment identity, unavailable native validation, low/missing resolution
thresholds, global/local relief bounds, stale acceptance evidence, contradictory
configuration, immutable policy, changed checkpoint identity, and real triangle
or byte-budget overruns. Rejected staged exports preserve the previous package.
An unavailable native requirement also respects output-overwrite refusal.

Machine-readable verification is retained in
[acceptance-verification-2026-09-14.json](acceptance-verification-2026-09-14.json).
The previously interrupted development run changed source during an export-reuse
test and correctly invalidated its cache; only the subsequent frozen-source run
is used as passing full-suite evidence.

## Remaining limits

This is a numerical acceptance contract. Resolution screening and measured
floor/roof convergence do not establish complete contour convergence. Relief
factors do not measure actual roughness or affected area; surface filtering,
smoothing and displacement still need independent fidelity criteria. Capsule
clearance does not establish ground-contact vehicle motion.

No broad new seed campaign or native editor campaign was run for this feature.
The earlier V3 results remain evidence for their original source/configuration.
Next work is integrated native finite-body route validation tied to the exported
asset, followed by measured fidelity and runtime/asset-size studies. Additional
bodies, structural events and long networks still need stratified evaluation.

The earlier baseline CI run at `535b987` failed three checks (backend validity,
network seed 5 plan crossings, and stacked-section separation); the preceding
cleanup run passed. The cause of that differing behavior is not diagnosed by
this feature. It remains an open reproducibility/CI investigation; see the
baseline CI details in the verification JSON.

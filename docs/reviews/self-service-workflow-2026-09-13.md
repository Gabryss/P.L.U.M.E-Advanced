# Self-service generation and diagnostics — 13 September 2026

The supported `plume-check` command extends the existing reliability engine with
preflight, immutable campaign plans, process locking, validated stage resume,
file-integrity receipts, an offline HTML report and actionable failure categories.
The operating instructions are in [the self-service guide](../self-service.md).
No new procedural seed-selection or relaxed geometry thresholds were introduced.

## Verified in this change

- 152 targeted regression tests and five subtests passed in 115.87 seconds,
  including the new resume/fault-injection tests and existing campaign, checkpoint,
  seed override, export, output guard, progress, CLI and configuration tests.
- The real checkpoint integration test injected disk failure after section
  generation, then recovered host, network and sections without recomputation.
  A cold invocation was separately confirmed not to use those checkpoints.
- Fault tests covered interruption, missing/modified artifacts and receipts,
  changed identities and inputs, replay failure, concurrent-process locking,
  malformed worker results, invalid requests, preserved prior failures and
  explicit one-attempt-per-resume behavior.
- Ruff passed across source, scripts and tests. Mypy passed all 98 production
  source files. `git diff --check` passed.
- Ten short network/section cases passed, with ten independent cold replays:
  single and interconnected presets, seeds 0, 17, 42, 20260912 and 4294967295.
  This was an A–C sweep, not another ten full exported caves.
- One complete neutral packaged smoke case, seed 3, passed generation and portable
  export validation. Its cold replay matched host/network/section identities,
  exact raw mesh hash and GLB bytes. It contained 335.40 m of combined passages,
  388,776 triangles and a 14,913,548-byte GLB, without rocks.
- That full case explicitly reported two of 45 input profiles below the eight
  voxel-samples heuristic. It was not represented as a convergence certificate.
- A deliberately coarse 2 m version of the same case failed at the base-surface
  gate. The retained `SurfaceTopologyError` identifies obstructed sampled route
  centres and gives resolution-focused next steps. This is an expected negative
  test; it is not counted as a passing generation.
- Resuming the ten-case sweep and auditing the full export verified existing
  receipts without starting another worker. All 64 generated report links resolved.

Local evidence is under `outputs/self_service_validation`: `sections/report.html`,
`full/report.html`, `rejected_coarse/report.html` and `checks/`. The earlier ten
fully rendered caves remain their own historical campaign; their source identities
were not rewritten to claim validation of the new orchestration code.

## Scope and limitations

The 152-test run was targeted, not a rerun of the entire project suite. Its
whole-package coverage report is a record of that narrower invocation. Earlier
full-suite coverage is historical evidence for the preceding version.

The full smoke case was neutral. Existing material/export tests ran, but this
change did not perform new native Blender, Unity or Unreal imports. Triplanar
material bundles remain available in configured textured exports.

Receipts are integrity checks for trusted local files. Checkpoints contain pickle
payloads and are not safe imports from third parties. Interrupted export atlas
work is recomputed; partial atlas caching remains outside the supported workflow.
Known geometric repairs are automatic and bounded. Unknown programming defects
are diagnosed and retained, not automatically edited away or retried under new
seeds. Numerical acceptance alone cannot establish natural appearance, continuous
walkability or scientific validity.

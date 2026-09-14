# All-repair evaluation campaign — 13 September 2026

Use **`final/`** for final acceptance evidence. It uses frozen production source `07c063f95f21234ad8c563381d0053e8192fc95803d04fa34cb9f6c8fd1978fc`.

The first attempt (source `60074a71b013236cbec71132ca08d4d736187e313d0a1361d677592c3b3116a7`) remains at this directory's top level. Expanded integration testing found a USD coordinate precision defect. Its serialization gate correctly blocked publication. The full runners were interrupted, USD output/inspection were corrected, and **all six full cases and all 24 screen cases were restarted** in `final/` under `protocol_final.json`. The original protocol and failure evidence remain available; do not combine incomplete initial results with final results.

## Prespecified runs

- Six full Earth cases, each followed by a new process without checkpoints: single 80 m seeds 0, 17 and 4294967295; single 250 m seed 17; three-source 250 m seeds 1 and 73.
- Twenty-four network/section cases, also with cold replays: single and three-source 250 m recipes, seeds 0, 1, 2, 3, 7, 17, 42, 73, 101, 255, 65535 and 4294967295.
- Fixed 12 cm voxels, real PBR materials (five 4K cases, one inherited 1K case; see the resolution audit), collision enabled, rocks/events disabled in full cases. Targets refer to the dominant route, not summed branch length.
- Full workers: 3600-second timeout, 8192 MiB address-space limit. Screen workers: 600-second timeout, same memory limit. At most two full workers run concurrently. Resident memory is reported separately.
- Original/replay Python hash seeds are 11/37. Do not edit code, recipes, dependencies or source textures during execution. Deliberate fault tests use isolated temporary data.

The exact recipes, protocol and audit script are retained in `docs/reviews/all-repairs-evidence-2026-09-13/`. Source maps and production repair settings are unchanged.

## Reproduce

From the repository root, use a **new** output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-mpl \
  .venv/bin/plume-check --configs docs/reviews/all-repairs-evidence-2026-09-13/single_80m_4k.toml \
  --seeds 0 17 4294967295 --scope full --timeout 3600 --memory-limit-mib 8192 --output outputs/repeat_single80

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-mpl \
  .venv/bin/plume-check --configs docs/reviews/all-repairs-evidence-2026-09-13/single_250m_4k.toml \
  --seeds 17 --scope full --timeout 3600 --memory-limit-mib 8192 --output outputs/repeat_single250

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-mpl \
  .venv/bin/plume-check --configs docs/reviews/all-repairs-evidence-2026-09-13/multi_250m_4k.toml \
  --seeds 1 73 --scope full --timeout 3600 --memory-limit-mib 8192 --output outputs/repeat_multi250

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-mpl \
  .venv/bin/plume-check --configs docs/reviews/all-repairs-evidence-2026-09-13/single_250m_4k.toml \
  docs/reviews/all-repairs-evidence-2026-09-13/multi_250m_4k.toml \
  --seeds 0 1 2 3 7 17 42 73 101 255 65535 4294967295 --scope sections \
  --timeout 600 --memory-limit-mib 8192 --output outputs/repeat_screen
```

The recorded final execution groups the four single-source full cases into one runner via `final/run_group.py`; this does not alter per-case generation. `plume-check --output <existing-group> --resume --report-only` rechecks saved evidence without generating again. Use `--resume` without `--report-only` to finish interrupted cases with compatible checkpoints. Source changes require a new campaign, not reuse of the initial version's checkpoints.

## Tests and retained diagnostics

`final/tests.xml` / `final/tests.log` are the final full test run, excluding separately marked performance and paper experiments: 777 passed, 18 skipped, 3 deselected, and 59 subtests passed. Native Blender integration is enabled with `PLUME_BLENDER_BINARY=/home/gabriel/Software/blender-4.0.1-linux-x64/blender`. `final_tests.log` at the top level is the interrupted pre-fix integration run.

`repair_tests.*` retain the initial focused run, which exposed an incorrect assumption in a newly added width-repair test: its synthetic elevation data did not match the host. Production correctly refused to lose original route centres. The test now checks both safe rejection of that inconsistent fixture and successful repair of a host-sampled fixture. `initial_test_sources/` preserves the initial test; `supplemental_tests.*` records the corrected cases and surface-relief tests. No production gate was relaxed.

`coverage_collection_failure.*` retain an optional coverage-instrumentation attempt that failed during NumPy import (`cannot load module more than once per process`). The final suite runs normally without coverage instrumentation; no coverage percentage is claimed and no dependency was changed during generation.

`native_blender/` contains a separate copy of the seed-0 80 m accepted asset, a packed Blender scene, a native roof/floor survey, and two rendered interiors using the continuous projection material. Its `source_receipt.json` links copies to the initial accepted original. `final/native_verification.json` additionally confirms the rendered GLB is byte-identical to the final version's seed-0 asset and that the native material adapter is unchanged. It does not modify sealed case evidence. Unity/Unreal package checks do not constitute native editor/compiler execution.

## Acceptance scope

Network geometry, section constraints, host preservation, mesh topology and sampled passages, bounded recovery, UVs, visual geometry, collision fallback, source texture integrity, repaired normals, material bindings and packaged adapters, serialized assets, budgets, reproducibility and artifact receipts are checked. Fault tests additionally exercise exhaustion, resource/programming errors, interrupted runs and damaged evidence.

Finite tests are not proof for every possible seed. Neither package checks nor a few renders certify seamless appearance everywhere, geological realism, continuous navigation, exhaustive self-intersection absence or simulator frame rate. Retained warnings remain part of the result.

## Texture resolution audit

The frozen `single_250m_4k.toml` recipe omits `embedded_texture_max_size`, so its actual prepared maps are 1024×1024 (the project default), despite the filename. This case is reported as 1K. The three 80 m single-source cases and both 250 m multi-source cases use 4096×4096 maps. The recipe was not altered during replay. Final report tables use measured prepared-map dimensions.

# Running and diagnosing PLUME yourself

Use ordinary `plume-generate`: evaluation, actual-mesh inspection and bounded
repair are already part of generation. You do not need an assistant or a second
command to trigger them. The separate `plume-check` command adds multi-seed
campaigns, process limits and cold reproducibility replays when you want broader
regression evidence.

Texture inspection and bounded repair are also automatic in full runs. See
[the material repair contract](materials.md#automatic-texture-inspection-and-repair).
Inspect `export_TARGET/texture_recovery.json` for map repairs and package retries;
the run quality report includes it. A missing source image needs to be restored,
not a different network seed.

## Generate with built-in checks

```bash
uv run --no-sync plume-generate \
  --config config/earth_short_interconnected_full.toml \
  --output outputs/my_checked_cave/network.png
```

Use a fresh output directory. The normal pipeline screens the network, evaluates
section resolution, inspects generated triangles, checks the visual mesh after
smoothing/displacement, checks collision simplification if enabled, and verifies
serialized geometry before publishing the package. Known failures trigger a
finite, repeatable repair sequence. A successful run includes
`pipeline_quality_report.json`, `pipeline_recovery.json`, `pipeline_inspection.png`,
`section_resolution_report.json` and a complete `run_manifest.json`. These checks
and the measured-passage figure run even with optional stage figures disabled.

Read warnings and effective repair settings in the quality report.
`pipeline_recovery.json` records whether the original network passed, was locally
repaired, or was replaced by another deterministic network in the same host.
The default two local attempts and two replacement networks need no manual
intervention. Configure them under `[geometry]`; see
[upstream recovery](reliability.md#upstream-recovery-and-deterministic-regeneration)
for budgets, route protections and reproducibility. If these attempts are exhausted,
the failed case remains available for diagnosis and no export is published. On failure,
it records the failing stage and error, plus detailed inspection evidence when
available. `progress.jsonl` identifies the last completed operation. The normal
command's `--resume` reuses compatible checkpoints and repeats export inspection;
it never treats an old successful report as proof about newly prepared geometry.
An export rejected during staging does not replace a previously published one.

See [embedded pipeline acceptance](reliability.md#embedded-pipeline-acceptance)
for the exact gates and repair limits. Sampled numerical checks do not certify
geological realism or native-engine rendering. Unknown programming defects still
require a code fix and regression test; no finite campaign proves all seeds valid.

## Install or update the command

From the repository root:

```bash
uv sync --locked --group dev --extra rocks --extra paper
uv run --no-sync plume-check --help
```

The core command does not require the test or optional dependencies. For a
minimal installation use `uv sync --locked --no-dev`. The equivalent entry point
without reinstalling is `python -m plume_advanced.evaluation.reliability` in your
existing project environment.

## Start with configuration and input checks

```bash
uv run --no-sync plume-check \
  --configs config/earth_short_single.toml config/earth_short_interconnected_full.toml \
  --seeds 0 17 42 20260912 4294967295 --scope full --voxel-size 0.10 \
  --preflight
```

This checks configuration loading, required input files and EXR conversion-tool
availability. It reports disabled export budgets. It does not generate a network,
measure clearances, check image contents or import into an engine. Fix any missing
texture files before running the full textured scenario.

## Generate and check ten small caves

```bash
uv run --no-sync plume-check \
  --configs config/earth_short_single.toml config/earth_short_interconnected_full.toml \
  --seeds 0 17 42 20260912 4294967295 --scope full --voxel-size 0.10 \
  --timeout 3600 --memory-limit-mib 8192 \
  --output outputs/my_ten_case_check
```

The two configurations and five seeds produce ten distinct cases: five single
source and five interconnected. Their configured route targets are a few hundred
metres. These presets disable rocks; the single-source preset is neutral and the
interconnected full preset includes its configured material maps. The command
respects these differences, rather than silently assigning another material.

Every successful case is also generated in a fresh process with a different
Python hash seed. **Ten cases plus ten verification replays means twenty worker
runs**, not twenty independent requested environments. Replays can approximately
double generation and storage costs. They must remain cold to test repeatability.
Use `--no-replay` only for exploratory runs; the report says repeatability was not
checked.

Workers run sequentially to bound peak memory. Each has a time limit. The Linux
address-space limit includes virtual allocations and is different from physical
RAM usage; other platforms record that this ceiling is unavailable. Memory limits
never increase automatically. Resolution refinement is explicitly opt-in and
bounded; collider reduction must pass its configured topology, clearance and
surface-error checks. Neither mechanism weakens acceptance to force success.

During long operations, the terminal shows the current stage, work unit, measured
counts and elapsed worker time every ten seconds. Detailed events are retained in
`progress.jsonl`. There is no invented percentage for an operation with unknown
work count. The HTML report is updated after each case.

## Open the report

Open `outputs/my_ten_case_check/report.html` in a browser. It works offline and
links each case to its error, evidence, portable asset and next action. The JSON
summary supports other tools and CI.

| Result | Meaning |
|---|---|
| Checks passed | Every planned case passed its recorded checks and required replay, with unchanged inputs and code. Read any warnings. |
| Checks failed | At least one case, artifact-integrity check or replay failed. The failure remains in the report; later cases still run. |
| Incomplete | The campaign was interrupted or contains cases that have not run. Resume it. |
| Warning | A check passed with a limitation, such as reduced surface relief or fewer than eight voxel samples across a narrow input profile. This needs inspection. |

For full cases, checks include the production network and base-surface gates,
finite closed oriented geometry, floor-map processing, portable material/UV/normal
and provenance checks, and an additional genus/component check on the actual
exported cave when structural events have not changed its topology. The original
and replay compare host/network/section semantic identities, exact raw mesh
array hashes and GLB bytes. Structural events retain their event-specific checks;
the original network genus is not imposed on a physically changed surface.

When enabled, `required_route_height_m` and `required_route_width_m` require a
continuous upright capsule corridor on the dominant route. The packaged default,
general project scenario and full interconnected presets use 0.5 m × 0.5 m plus
0.02 m margin; other configurations need these controls explicitly. Bounded vertical placement precedes geometry changes, and
every proposed path is rechecked. Optional side branches may remain narrower.

**These are numerical checks, not a claim that the shape looks natural.** The
capsule check does not certify vehicle dynamics or ground contact. The pipeline
does not certify every surface self-intersection or scientific fidelity; optional
resolution checks measure convergence only at their declared probes. Inspect thin
passages and junctions. Native
Blender rendering and Unity/Unreal imports remain separate application checks;
packaging the material adapters is not evidence that an engine imported them.
See [materials](materials.md) and [reliability](reliability.md) for those workflows.

## Resume safely

After a timeout, interruption or resource problem:

```bash
uv run --no-sync plume-check --output outputs/my_ten_case_check --resume --timeout 7200
```

The saved plan supplies configurations, seeds, scope, resolution and replay policy.
Only the resource limits may change within the same campaign. The command:

1. Verifies source, runtime, configuration and external input identities.
2. Verifies receipts for completed results and their files, including material bundles.
3. Skips intact successful work. Rebuilds missing or modified successful results.
4. Retries each unfinished/failed case **once per explicit resume**, retaining its
   earlier attempt and diagnosis. It never loops indefinitely on the same defect.
5. Reuses compatible host, initial network/section, accepted network-section-base
   triple and base-floor checkpoints
   for original cases. Later event/export/validation work is repeated as needed.
   An interrupted UV atlas batch is currently recomputed; it has no partial cache.
6. Replays without checkpoints, so cache reuse cannot conceal nondeterminism.

Changing code, dependency versions, input file contents or generation settings
requires a **new output directory**. Keep the same seed to verify a correction.
Existing hand-built inspection folders and older campaign schemas do not have this
resume contract. Do not relabel them as new campaigns.

An OS lock prevents two processes from writing the same campaign and releases on
process exit. Checkpoints are hash-checked local pickle artifacts, not a safe
format for untrusted downloads. File receipts detect accidental changes; they are
not cryptographic signatures against a malicious editor.

To verify saved receipts and refresh the report **without any generation**:

```bash
uv run --no-sync plume-check --output outputs/my_ten_case_check --resume --report-only
```

This checks the saved evidence's integrity, not another native render or a fresh
geometry computation. Failed cases keep their earlier diagnostic.

## What to do with a campaign failure

The filenames below are campaign-worker artifacts. For ordinary generation, use
the same reasoning with `pipeline_quality_report.json`,
`network_quality_report.json`, `section_resolution_report.json`,
`stage_d_geometry_report.json` and `progress.jsonl`.

| Category | Action |
|---|---|
| Configuration or missing input | Correct the named value/path or restore the dependency. Preflight again. Changed inputs require a new campaign. |
| Host domain | Enlarge the host or reduce requested extent/source spacing. More seeds cannot fit an impossible domain. |
| Network rejected | Read `network_quality.json`: all configured candidates were tried. Correct the formation constraints without disabling the quality checks. |
| Surface rejected | Read `pipeline_recovery.json`, `resolution.json` and the traceback. If upstream recovery is exhausted, all configured local and replacement attempts failed. Retain the seed and failed evidence; test finer voxels or corrected constraints in a new campaign. |
| Timeout | Read the last progress operation, then resume with more time if appropriate. |
| Memory or disk space | Correct the resource constraint, then resume. Keep the failed attempt's evidence. |
| Export budget | Reduce extent or use a separately validated resolution. Raise the configured budget only when the simulator can support it. |
| Export validation | Read `asset_checks.json`. Fix the failing material, geometry or provenance condition before using the asset. |
| Replay mismatch | Compare original and replay results. Treat this as a reproducibility defect, not a seed to discard. |
| Unexpected error/native crash | Preserve the traceback and fixed-seed recipe. Correct the code/environment and add a regression test. Automatic geometric retries do not repair arbitrary code defects. |

Known surface repairs keep the accepted network and root seed. They try lower
added relief and bounded voxel filtering, then recheck the same topology and
route-centre constraints. Effective values and rejected attempts are in
`surface_quality.json` for completed surfaces. Exhausted repairs retain their
rejection reasons in the worker traceback. A repair that would violate a gate is
rejected rather than hidden.

## Verify a code fix before using it

Run the targeted regression first, followed by the maintained checks:

```bash
uv run --no-sync ruff check src scripts tests
uv run --no-sync mypy src/plume_advanced
uv run --no-sync pytest -q --cov=plume_advanced --cov-report=term-missing --cov-fail-under=70
```

Then rerun the exact failed seed in a new campaign, followed by the ten-case sweep.
Do not replace the failing seed with an easier one. Repository CI runs the suite
on supported Python versions on pushes and pull requests; native application
checks require their separately configured runtimes.

CLI exit codes are `0` for passed checks, `1` for failed checks, `2` for invalid or
incompatible campaign requests, and `130` for an interrupted command. A numerical
pass can still contain explicit inspection warnings.

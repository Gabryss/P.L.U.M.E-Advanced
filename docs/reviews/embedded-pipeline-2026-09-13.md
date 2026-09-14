# Embedded evaluation, inspection and repair — 13 September 2026

The normal `plume-generate` command now performs acceptance inside generation.
The separate campaign command remains an optional regression tool. A normal run
needs no additional inspection command or diagnostic-render flag.

## What changed

The existing deterministic network and immutable-density surface repair loops
remain in use. Actual-triangle inspection now participates in base-surface
acceptance and checks the final event surface. It measures vertical floor/roof
intersections at refined route centres, verifies manifold topology and preserves
protected route points. Structural events can intentionally obstruct secondary
passages; their measurements are reported without requiring the original graph
genus after the event.

Shared export preparation repeats inspection on the actual float32 mesh after
smoothing, UV splitting and displacement. Known inspection failures try a finite
sequence of reduced displacement and, finally, no smoothing/displacement. The
root seed, accepted network and original requested configuration remain intact;
the prepared scene records its effective settings. Unsafe collision clustering
falls back to the already inspected original collider. No arbitrary programming
exception becomes a geometry retry.

Serialized GLB content is validated in staging, and visual geometry is compared
with the inspected arrays for GLB, OBJ and PLUME's ASCII USD schema. A rejected
package cannot replace an existing export. Final pipeline acceptance verifies the
serialized file hashes and unchanged input/source fingerprint before the run is
marked complete. Resumed geometry still passes through export inspection.

Normal runs write `section_resolution_report.json`,
`pipeline_quality_report.json` and `pipeline_inspection.png`; packages include
`pipeline_inspection.json`. Optional stage figures remain separately controlled.
The terminal and progress trace now cover twelve overall stages and include
measured-mesh, repair and serialized-asset work.

## Regression evidence

Focused suites cover **237 distinct passing tests, 5 subtests, 18 skipped, 1 deselected**.
The first selections passed 231 tests; the final material selection passed 64
(including six additional tests), repeating already covered tests.
The 28 new inspection tests cover actual geometry independent of the density
field, torus rock islands, separated vertical air layers, damaged surfaces,
exact UV seam welding versus real cracks, intentional and protected event
obstructions, repeatable visual repairs, exhausted repair preservation,
programming-error propagation, unsafe collider fallback, corrupted GLB/OBJ/USD,
changes after staging, participation in existing surface repairs, valid partial materials and missing
configured texture bindings.

The CLI integration test runs the real pipeline and requires all twelve stages,
always-on reports/figure and valid output receipts. The focused selection also
covers geometry with structural events, export targets, checkpoints, simulation
budgets, materials, seed overrides and the campaign engine. The 18 skipped tests
require separately configured native shader/application tools. One long event
placement test was deselected; structural-event consumption was exercised.
This was not a rerun of the entire repository suite. Ruff, mypy (101 source files)
and `git diff --check` passed.

Commands used (with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and a writable
`MPLCONFIGDIR`):

```bash
pytest -q tests/test_embedded_inspection.py tests/test_cli.py tests/test_export_targets.py tests/test_validation.py tests/test_surface_acceptance.py tests/test_surface_topology.py tests/test_checkpoints.py tests/test_geometry.py tests/test_events.py -k 'not test_event_stage_places_seeded_mesh_events'
pytest -q tests/test_progress.py tests/test_export_budgets.py tests/test_geometry_export.py tests/test_projected_materials.py tests/test_uv_repair.py tests/test_evaluation_geometry_report.py
pytest -q tests/test_reliability_resume.py tests/test_config_reliability.py tests/test_seed_override.py tests/test_local_geometry.py tests/test_reliability_campaign.py
```

Validation evidence is retained in `outputs/embedded_pipeline_validation/`.
The single-source case uses the unchanged packaged root-seed-3 configuration.
Its second ordinary CLI run is cold, with `PYTHONHASHSEED=7919`: raw and visual
mesh hashes, measurements, repair choices and GLB bytes match exactly.
The multi-source recipe is retained in `recipes/multi.toml`; it derives from the
maintained full interconnected preset with root seed 0, a 250 m route target and
0.10 m voxels. Asset paths are made absolute to preserve their meaning after
moving the recipe. Rocks and events remain disabled in these generation cases;
the multi-source case retains its 4K material maps.

Production source SHA-256 for the initial single/replay and full multi runs: `4d53e2af8fc7435eb1e12d9d4164defcdd5d7987297d630f51a3a46ff63eb23d`.

## Scope

These are sampled numerical and serialization checks. They do not certify
geological realism, continuous rover navigation, exhaustive triangle
self-intersections or native Blender/Unity/Unreal shader behavior. The inspection
figure plots measured geometry, not a rendered material. OBJ/USD checks cover
PLUME's own visual geometry schema; separate material-bundle geometry is outside
that round trip. Narrow-profile resolution warnings remain explicit and do not
silently change voxel size or acceptance thresholds. Irrecoverable failures still
require correcting the recorded input, environment or code defect.

## Final material compatibility check

The full multi run exposed no missing textures. An additional synthetic partial-
material regression did expose an overly strict new checker: albedo-only
materials were incorrectly required to include normal and roughness maps. The
checker now honors configured map slots and validates their bindings and image
indices; missing configured maps still fail. Partial diffuse-only, normal-only
and roughness-only materials and deliberate binding removal are covered.

The final changes affect validation and texture progress reporting, not geometry
or material generation. Texture loading and EXR conversion now identify the
current file and size limit. The full multi GLB subsequently passed 41 portable
checks and the final configured-map binding guard. A further cold ordinary CLI
run, `single_final`, exercises the final source version.

Final production source SHA-256: `e13b81b73d95a250640cbc95cd3bb69919c29bdc818bfed3c71b1ee758ad8f75`.

## Completed ordinary generation runs

| Case | Main route | Combined network | Triangles | Inspected route samples | GLB |
|---|---:|---:|---:|---:|---:|
| single_final | 252.2 m | 335.4 m | 388,776 | 185 | 14.2 MiB |
| multi | 270.4 m | 641.5 m | 1,760,660 | 633 | 124.6 MiB |

Both cases passed their hard gates. The single case flags 2/45 input profiles
below eight samples across their smallest dimension; the multi case flags
203/632. These are retained resolution/convergence warnings, not repaired or
certified passage shapes. The multi network has three systems, two merges and
one split, and accepted network candidate 11 after rejecting ten candidates.
Its full run took about 17.4 minutes, largely UV/material export; single runs took
95–109 seconds. Some checks ran concurrently, so these are observations rather
than controlled performance benchmarks.

The final-source single run completed all twelve stages and retained identical
GLB bytes to both earlier cold runs. All manifest output receipts were verified
for the four runs (two distinct caves, two additional single-case verification
runs). The single and multi assets passed 38 and 41 external portable checks,
respectively, including final run provenance. The inspection figures were
visually checked for readable measurements and labels. No native engine render
is claimed for these cases.

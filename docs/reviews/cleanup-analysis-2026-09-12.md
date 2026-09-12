# PLUME Advanced cleanup analysis — 12 September 2026

PLUME needs a focused cleanup, beginning with validation and reproducibility contracts. The earlier cleanup removed substantial legacy implementation, but some inactive calculations remain and several workflows have diverged. Removing every file that looks old would lose supported functionality and scientific evidence.

This review examines the **working tree**, including the uncommitted cleanup and material changes, against checkpoint `32cbcee331e496deebb34002002ae82e09b05cf2` on `lava-emplacement-history`. It does not change generation, configuration, tests, delivered caves, or frozen experiments. Review probes, build products and fresh verification logs are under `tmp/cleanup_review_20260912/`; durable evidence accompanies this report in [cleanup-analysis-2026-09-12.json](cleanup-analysis-2026-09-12.json).

## Scope and method

| Area | Inspected scope |
|---|---|
| Maintained package | 85 Python files, 35,071 physical lines |
| Repository utilities | 29 Python scripts, 5,512 lines |
| Tests | 50 Python files, 9,766 lines |
| Configuration | All 15 repository presets plus the packaged default; all load as schema 4 |
| Configuration surface | 405 annotated fields across classes ending in `Config`, including nested and resolved settings |
| Documentation | README, documentation tree, paper entry point, local links and selected generation/validation claims |
| Delivery | Dependency declarations, lockfile, CI, source distribution and installed-wheel import behavior |
| Research/history | Tracked snapshots and archive boundaries, rather than rewriting historical source |

The review combines syntax-tree inventories, definition/reference searches, configuration loading, duplicate-body detection, targeted call-path inspection, bounded reproductions, packaging checks and the full regression suite. Static references are not proof that a feature executes, and missing static references are not proof that a callback is dead. For example, the three apparently unreferenced pytest methods in `validation.py` are framework hooks and must stay.

No new kilometre-scale caves, scientific evaluation campaign, or native engine renders were generated for this audit. Those are separate checks of behavior and visual quality. Coverage does not establish geological validity.

## Findings by priority

P1 means fix before relying on cleanup validation or reproduction. P2 means a concrete cleanup or maintenance issue. P3 means optional simplification or workspace housekeeping.

| ID | Priority | Finding | Recommended treatment |
|---|---|---|---|
| C01 | P1 | CI omits a dependency imported during test collection | Correct test environment |
| C02 | P1 | Evaluation and production use different acceptance scopes | Share an explicit acceptance policy |
| C03 | P1 | Checkpoint identity ignores numerical dependency changes | Include runtime identity |
| C04 | P1 | Provenance validation does not bind the selected asset | Verify the actual asset and revision chain |
| C05 | P2 | Source/provenance helpers disagree and can report misleading values | Consolidate source and dependency identity |
| C06 | P2 | `level_transition_fraction` is accepted but unused | Retire the ineffective setting |
| C07 | P2 | Obsolete route-texture calculations still execute | Retain route centres; remove unused frame payload |
| C08 | P2 | Inspection/export/validation workflows duplicate pipeline behavior | Extract shared services and validation profiles |
| C09 | P2 | Large modules and untyped cross-module calls impede safe changes | Extract coherent responsibilities incrementally |
| C10 | P2 | Presets duplicate thousands of lines and mix inputs with resolved metadata | Clarify the configuration contract and scenario catalog |
| C11 | P2 | Current documentation contradicts itself and includes lost local evidence | Correct current claims; label historical records |
| C12 | P2 | Render-quality problems lack a repeatable regression check | Preserve a controlled visual test alongside numeric checks |
| C13 | P3 | Generated material dominates local disk use | Separate disposable caches from inspection/research assets |
| C14 | P3 | Primitive geometry remains an explicit legacy option | Decide its supported scope before removing it |

### C01 — CI cannot collect the full suite with its declared dependencies

**Evidence:** [CI workflow](../../.github/workflows/ci.yml), test job installation; [test_evaluation_campaign.py](../../tests/test_evaluation_campaign.py), line 6; [pyproject.toml](../../pyproject.toml), `paper` extra and development group.

The job installs `--group dev --extra rocks`. `test_evaluation_campaign.py` imports `psutil` unconditionally, but `psutil` is declared only in the `paper` extra. Neither the core nor Rocky dependency graph supplies it. The local environment already has it, so a local green suite does not catch the mismatch.

**Reproduction:** with only the `psutil` import made unavailable, collecting that test module exits with code 2 and `ModuleNotFoundError`. This simulates the missing dependency; it is not a claim that GitHub Actions was run during this review.

**Cleanup:** install the paper extra in the full-suite CI job, or declare the needed test dependency in the development environment. Keep a separate minimal-install job. Do not solve this by silently skipping the campaign tests in the principal test job. Verify collection in the exact declared CI environment.

### C02 — Evaluation can accept a network that the main pipeline would reject

**Evidence:** [evaluation/experiments/common.py](../../src/plume_advanced/evaluation/experiments/common.py), lines 46–53; [cli.py](../../src/plume_advanced/cli.py), around line 410; [network_acceptance.py](../../src/plume_advanced/stages/network_acceptance.py), lines 188 and 221–225.

The main pipeline passes `section_config` into network generation, enabling candidate rejection based on the sampled passages. Evaluation's `generate_network()` does not pass it. `generate_sections()` then samples sections only after the network has been selected, without reapplying the section acceptance gate. Morphometry, export consistency and determinism use this helper.

When quality screening is enabled, the same scenario can therefore have a different acceptance scope depending on its entry point. A bounded call inspection confirms evaluation forwards no keyword arguments to `generate()`.

**Cleanup:** define a shared generation service with an explicit acceptance scope. Production-equivalent section/export evaluations should use the production gate. Network-only experiments and sampling ablations may deliberately use a narrower scope, but must record that policy and avoid claiming production equivalence. Preserve frozen campaign code and results; changing acceptance is a methodological change, not a behavior-preserving refactor.

### C03 — Checkpoints can be reused after numerical dependencies change

**Evidence:** [pipeline/checkpoints.py](../../src/plume_advanced/pipeline/checkpoints.py), lines 23–49 and 66–74; [cli.py](../../src/plume_advanced/cli.py), checkpoint construction around line 344.

The fingerprint includes resolved configuration, supplied input files and package source. Checkpoint loading additionally checks the Python version and payload digest. It does not include installed NumPy, SciPy, scikit-image, xatlas, Trimesh or Rocky versions. The lockfile is not a fingerprint input in the normal CLI path.

**Reproduction:** changing a lockfile beside otherwise identical package source leaves the fingerprint unchanged. Installed dependency identity is also absent from the implementation. This creates a route to mixing cached stages with stages rebuilt under different numerical libraries.

**Cleanup:** share a relevant runtime/dependency identity between checkpoint creation and loading; include optional backends when used. Test dependency-only invalidation. Separately consider stage-specific fingerprints later: the current all-package fingerprint unnecessarily invalidates host/network work after an unrelated visualization or export edit. Preserve conservative invalidation until dependency boundaries are explicit.

### C04 — Asset provenance validation can pass for an unrecorded GLB

**Evidence:** [validation.py](../../src/plume_advanced/validation.py), `_reproducibility_checks()` at line 771 and ancestor-manifest discovery at line 151; [texture_inspection.py](../../scripts/texture_inspection.py), revision report creation.

The validator checks the files listed by a discovered run manifest, but never requires `self.asset_path` to be one of those files. A different GLB can inherit a completed ancestor run's successful hash check. Material revisions have their own source/output hashes, yet the generic validator does not follow `material_revision.json`.

**Reproduction:** a minimal GLB not listed in a completed manifest receives two passing reproducibility checks when the manifest lists an unrelated intact file. Changing the GLB leaves both checks passing. This probe exercises the provenance phase only, not geometry validation.

**Cleanup:** require a matching digest for the selected asset. For revisions, validate the revised GLB, its recorded source GLB, and the link to the original run. Do not require a material-only revision to pretend it generated another network. Add cases for unrecorded, modified, copied and legitimately revised assets.

### C05 — Reproducibility helpers have drifted

**Evidence:** [run_manifest.py](../../src/plume_advanced/run_manifest.py), lines 51–72 and 115–143; [evaluation/provenance.py](../../src/plume_advanced/evaluation/provenance.py), lines 71–152; [network_acceptance.py](../../src/plume_advanced/stages/network_acceptance.py), source hashing.

There are several separate implementations of source identity, hashing, dependency lists and Git inspection. Two Git helpers have identical bodies. Both turn a failed Git command into an empty string, which callers convert to `dirty = false`. The result should be unknown, not clean.

The generation manifest hashes files under the supplied working/source root. When an installed wheel is used from an empty directory, it records SHA-256 of empty input (`e3b0c442…b855`), rather than the executing package. This was reproduced using the wheel built during this review. Evaluation already hashes the executing package. The generation dependency list also omits xatlas and Rocky, although both can affect delivered assets.

**Cleanup:** centralize executing-package identity, file hashing and installed dependency reporting. Keep repository revision, working-tree status, runtime identity and input/config hashes as separate fields. Represent unavailable Git information explicitly. Version any changed identity format; do not retroactively rewrite old manifests.

### C06 — Remove an ineffective section setting

**Evidence:** [section_field.py](../../src/plume_advanced/stages/section_field.py), line 75; [config.py](../../src/plume_advanced/config.py), lines 1203–1204; all 15 repository presets.

`level_transition_fraction` is declared, accepted, serialized and range-validated, but has no consumer in maintained generation code. Users can tune it without affecting a passage. It is a confirmed inactive input, not merely a setting unused by the current interconnected scenario.

**Cleanup:** remove it from active presets, the input contract and validation, with an explicit retired-key error and migration note. Do not invent new behavior for it as part of cleanup. Preserve semantic section outputs in before/after checks; resolved configuration hashes will legitimately change.

`body_vertical_scale`, `body_fracture_scale` and `target_samples_across_passage` need a different treatment: they record resolved information. The review does not classify them as dead generation algorithms.

### C07 — Route-local texture machinery survives its consumer

**Evidence:** [geometry.py](../../src/plume_advanced/stages/geometry.py), `_surface_texture_frames()` at line 533 and `_texture_node_longitudinals()` at line 635; [geometry_types.py](../../src/plume_advanced/stages/geometry_types.py), `SurfaceTextureFrame`; [geometry_export.py](../../src/plume_advanced/stages/geometry_export.py), lines 455 and 845–882.

Generation still computes graph-geodesic distances, transported frame axes, profile points and perimeters for a route-aware texture representation. The current exporter generates UVs with xatlas. Its frame consumer only reads `frame.center` to choose the global face orientation. Searches found no production reads of `longitudinal_m`, `longitudinal_rate` or `profile_perimeter_m`; tests still exercise the obsolete frame expectations.

**Cleanup:** replace this payload with the route centres actually needed for orientation. Remove the geodesic traversal and unused frame calculations. Update tests to verify the real orientation/UV contract. Preserve the orientation behavior and compare exported positions, winding, normals, tangents and UVs on representative fixtures. Old pickles need their recorded source version; they are not a reason to retain an unused mapping implementation indefinitely.

### C08 — Inspection utilities have become alternate pipeline implementations

**Evidence:** [generate_tube_only.py](../../scripts/generate_tube_only.py), line 83; [package_inspection_exports.py](../../scripts/package_inspection_exports.py); [texture_inspection.py](../../scripts/texture_inspection.py); [validate_inspection_asset.py](../../scripts/validate_inspection_asset.py), line 29; [render_run_diagnostics.py](../../scripts/render_run_diagnostics.py), lines 38–156.

The tube-only exporter independently smooths geometry, assigns normals, converts axes and serializes a GLB, while the main exporter uses the prepared visual-surface contract. Neutral validation selects private methods because the main validator unconditionally requires texture maps. Packaging and texturing scripts both call private target-descriptor/guide helpers. The saved-run figure script contains sparse-grid and plot corrections that the normal diagnostic path does not share.

These tools are useful and are not all redundant commands. Their implementation ownership is the problem: material, shading, validation and figure fixes need to propagate consistently.

**Cleanup:** keep thin task-specific commands over shared APIs for geometry-only visual preparation, target packaging, asset selection, named validation profiles, and dense/tiled plotting. Make intentional differences such as double-sided inspection materials and omitted stages explicit. Preserve output safeguards: the old tube-only utility currently allows an existing output directory, whereas the main command checks overwrite authorization and the material revision requires a new directory.

### C09 — Extract responsibilities before attempting broad code simplification

| File | Lines | Main extraction opportunities |
|---|---:|---|
| `stages/network.py` | 4,772 | Data types, emplacement backends, general growth, graph/flow operations |
| `stages/geometry.py` | 3,000 | Profile sweeps, junctions, structural stamping, dense/tiled assembly |
| `stages/events.py` | 2,460 | Structural events, population placement, Rocky adapter |
| `stages/section_field.py` | 2,092 | Sampling, vertical placement, profile construction, junction blending |
| `stages/geometry_export.py` | 1,845 | Surface preparation, material loading, GLB writing, OBJ writing |
| `config.py` | 1,395 | Input parsing, body/flow resolution, stage-specific validation |

These six files contain about 44% of maintained package lines. `_run_pipeline()` alone is 556 lines; `_build_lobe_growth_paths()` is 526; `_validate_pipeline_configs()` is 483.

Mypy currently passes, but all 11 functions in `network_gallery_growth.py`, all 9 in `network_interconnected.py`, and all 8 in `network_topology.py` lack parameter and return annotations. Cross-module `generator._...` calls therefore remain weakly specified despite the green type check. The optional Rocky integration also imports private Rocky helpers; the pinned revision limits immediate drift but does not make that a stable adapter boundary.

**Cleanup:** extract one coherent responsibility at a time, starting with shared graph types/flow operations and export preparation. Add typed contracts at those boundaries. Avoid a new class hierarchy for every small helper, and avoid rewriting growth mathematics while moving code. Preserve numeric operation order and random-seed labels where output preservation is intended.

### C10 — Preset duplication obscures which controls actually apply

**Evidence:** `config/` contains 4,559 lines across 15 complete presets, typically 205–264 explicit leaf settings each. The packaged default is another separate scenario. All 16 configurations load successfully.

The short/long, single/multi, interconnected and full-export distinctions are valid, but large copied tables make material, quality and retired-key updates repetitive. Some dataclass fields exposed by the permissive per-table key set are derived values that the loader overwrites rather than user controls—for example geometry resolution metadata in [config.py](../../src/plume_advanced/config.py), lines 752–756.

**Cleanup:** publish a compact scenario catalog covering topology, independent growth, system count, physical extent, resolution, events and material policy. Distinguish accepted user inputs from resolved output metadata. Prefer concise documented overrides or reproducibly generated explicit presets over hand-maintained copies. Preserve resolved manifests in a comparison test; do not casually introduce inheritance that changes relative asset-path resolution or random sampling order.

### C11 — Correct current documentation while preserving dated reports

**Evidence:** [README](../../README.md), line 5 advertises schema 3 while line 402 requires schema 4. The local-link audit found 24 missing output targets and 7 absolute editor-style source links in the selected documentation scope. The README's checked local targets themselves exist.

Several dated reports link to output folders subsequently deleted during clean generation runs. Those results should be marked as historical/unavailable locally, with durable evidence or a regeneration recipe where possible. Do not rewrite historical results as current measurements.

The xatlas helper's docstring calls independent chart packing “harmless,” while [materials.md](../materials.md) correctly explains that chart seams and distortion can remain. These descriptions should agree: repeat wrapping does not establish visual continuity across independently oriented/offset charts. The `SurfaceTextureFrame` documentation also describes a mapping role it no longer has.

**Cleanup:** correct the schema header, add clear current-versus-historical document navigation, use repository-relative links, and connect claims to retained evidence. Keep the previous cleanup report as a dated record instead of updating its old test counts to today's result.

### C12 — Preserve the recent rendering defects as a separate regression target

**Evidence:** [create_blender_inspection.py](../../scripts/create_blender_inspection.py), lines 18–34 and 92–98; [validation.py](../../src/plume_advanced/validation.py), `_normal_tangent_checks()`; [test_material_revision.py](../../tests/test_material_revision.py).

The current code checks map binding, packed images, unit normals and orthogonal tangent frames. Those checks do not detect all chart seams, face-versus-shading-normal disagreement, grazing-light artifacts or a poor inspection-light setup. The Blender inspection script has fixed render settings, including 24 final samples, rather than a named inspection-quality preset. Recent user inspection demonstrated why successful material loading is not sufficient evidence of a good render.

**Cleanup:** consolidate inspection settings and add a small native Blender regression scene with fixed geometry, camera, light, maps and renderer version. Include neutral and textured comparisons and a deterministic CPU reference with tolerant visual thresholds. Add a numeric shading-normal/face-normal diagnostic without treating every angular deviation as a defect. Keep this optional/native check separate from fast portable-asset tests. Investigate the floor artifacts as a rendering issue; neither deleting texture support nor increasing samples establishes a fix.

### C13 — Most local storage is generated material, not removable production code

Approximate allocated sizes at the beginning of this audit were: `outputs/` 17 GiB, `tmp/` 12 GiB, `paper/` 2.7 GiB, `.vscode/` 1.5 GiB, and `.git/` 4.1 GiB. The whole workspace was about 38 GiB. These are local disk measurements, not repository download size, and verification itself adds temporary files.

There are 501 tracked Python files under `paper/`, mostly frozen campaign and Overleaf copies. They are outside the installed package. The new wheel contains only the 85 maintained package modules plus package metadata/default configuration; the source distribution contains no frozen or old `build/` tree.

**Cleanup:** stale ignored `build/` copies, tool caches and confirmed-disposable diagnostics are housekeeping candidates. User inspection scenes, the old PDF under `output/`, research outputs and editor state require an explicit retention policy. If duplicated frozen bundles are moved to release archives, retain their hashes and retrieval instructions first. Do not purge `.git`, `texture/`, research evidence or Blender work to make a source cleanup look larger.

### C14 — The primitive geometry branch is legacy, but reachable

**Evidence:** [geometry.py](../../src/plume_advanced/stages/geometry.py), lines 1992–2031 and 2408–2496; [test_mesh_continuity.py](../../tests/test_mesh_continuity.py), lines 310 and 433; [README](../../README.md), line 657.

All 16 maintained configurations select profile sweeps. `use_section_profiles = false` still activates capsule/ellipsoid stamping and is documented and tested. It is a legitimate candidate for retiring an old product option, but it is not unreachable code.

**Recommendation:** if profile-based geometry is now the sole supported model, retire this switch and its exclusive branches in a separate change with an explicit migration error. First distinguish primitive helpers used exclusively by this mode from any shared structural-event/junction operations. This should not be bundled invisibly into ordinary refactoring.

## Keep deliberately

- General lobe growth, trunk layouts, independent gallery growth and interconnected systems have distinct live dispatch paths. A no-rocks interconnected preset does not make the other modes dead.
- Dense and tiled meshing, physical roof screening, structural events, optional rock populations, collision and application exports remain active features.
- DOWNFLOW-reference and external Flowy support are reachable comparison backends. Removal would change supported scientific comparisons.
- Frozen campaign source, inputs, tables and manifests are experiment records. Their older schemas must remain tied to their recorded implementation.
- Pytest hooks, package entry points, protocol methods and public APIs should not be deleted based only on a name-count scan.
- Small repeated progress emitters do not justify a new general-purpose abstraction. Consolidation should remove real drift or a meaningful maintenance burden.

## Recommended cleanup sequence

1. **Establish a trustworthy baseline:** resolve C01–C05 with targeted regressions for dependency-only invalidation, installed-wheel identity, Git failure, selected-asset hashes and acceptance scope. Keep methodological changes separate from refactors.
2. **Remove confirmed inactive work:** retire `level_transition_fraction`; simplify the unused surface-frame payload while preserving route centres. Record semantic before/after results and the intended checkpoint incompatibility.
3. **Unify inspection services:** share visual preparation, validation profiles, packaging and sparse plotting. Preserve useful command entry points and their explicit scopes.
4. **Refactor the largest modules incrementally:** extract typed graph/flow and export boundaries first. Preserve mode behavior, seed labels, flux/lineage, section clearances, mesh winding and output contracts.
5. **Update configuration and documentation:** maintain scenario identities, correct current claims, and label dated evidence. Make any primitive-mode retirement a deliberate separate change.
6. **Apply workspace retention separately:** prune only agreed disposable artifacts. Source cleanup and storage cleanup have different acceptance criteria.

For behavior-preserving changes, use representative fixed-seed cases covering general single-system, general multi-system, trunk layout, independent growth and interconnected growth. Compare host/network/section semantic outputs, candidate acceptance decisions, source lineage and discharge. Exercise both dense and tiled geometry, structural events and no-events cases, plus neutral and textured exports. Package source hashes will change during cleanup; geometric/semantic equality must be compared independently of those hashes.

## Verification

Fresh verification results are recorded in the accompanying JSON evidence. Verification ran on Python 3.13.5.

| Check | Result |
|---|---|
| Full regression suite | 413 passed, 59 subtests passed, 1 optional Manim module skipped; 703.03 seconds |
| Package line coverage | 84.15%; 13,327 of 15,838 executable statements covered |
| Ruff | Passed for maintained source, scripts and tests |
| Mypy | Passed for all 85 package modules, subject to the untyped-boundary limitation in C09 |
| Compilation | Passed for package, scripts, tests and the video scene |
| Presets | All 15 repository presets and the packaged default loaded successfully |
| Lockfile | Offline consistency check passed |
| Installed dependencies | Compatibility check passed for 42 installed packages |
| Utility entry points | All 28 non-Blender scripts passed `--help`; Blender script compiled |
| Distribution | Source distribution and wheel built; wheel contains 85 package modules and its default TOML, without frozen research or stale build trees |
| Built-wheel import | Imported the extracted wheel's package and loaded its schema-4, no-rocks default |
| Targeted reproductions | Confirmed C01–C05 contract gaps; no production code changed |
| Source preservation | All 85 package-file hashes match the start of this audit |

The first build attempts encountered a read-only shared cache and an absent build backend in the runtime environment. The successful build used the already cached setuptools backend with a writable audit cache and no network or runtime dependency changes. This was an environment adjustment, not a project build fix.

Coverage is uneven: the three specialized network visualization modules have 0% measured coverage, general geometry visualization has 12.9%, and CLI orchestration has 48.1%. These figures identify verification gaps, not dead code. Subprocess execution can also fall outside collected coverage, so the scalability worker's 0% is not sufficient evidence that it never runs. The proposed shared inspection services should gain representative tests as their behavior is consolidated.

The local environment includes the paper dependencies and therefore does not represent the incomplete CI install in C01. Native Unity/Unreal imports, new Blender renders, external Flowy execution and a new scientific campaign are outside this cleanup review. No findings above have been silently marked fixed by the existence of passing tests.

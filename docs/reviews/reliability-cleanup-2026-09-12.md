# Reliability and cleanup implementation — 12 September 2026

This change implements the cleanup audit and strengthens seed screening, progress reporting, export budgets and regression coverage. The pre-change state is saved in commit `89d7f6a02f8e5822cdd45ab7f01104b3525b23b7`. The final package SHA-256 is `dbc1abcf0a2a433d1e56ef6f908eea3d32e3d23bc3223c2385f6cda7f58db09c`. The generation campaign completed on `22f29272488ff42553ef4a53b5110f6306e98712253d1336eb404ad1edc36d2d`, before the final display-only correction described below.

The [reliability guide](../reliability.md) gives commands, limits and interpretation. The [preset catalog](../config-presets.md) distinguishes the supported scenarios. This report supersedes recommendations in the earlier [cleanup analysis](cleanup-analysis-2026-09-12.md); its historical measurements are unchanged.

## What changed

### Seed acceptance and repairs

Production and section-level evaluation now share the same network-and-section acceptance gate. Numerical or geometric candidate failures advance through a bounded, deterministically derived sequence. Impossible host dimensions fail immediately with a domain explanation. Programming errors, missing inputs and exhausted resources remain visible failures.

The evaluation work found actual defects:

| Defect | Correction and protection |
|---|---|
| Smoothing shortened route distances and made width gradients worse | Enforce a non-expanding width envelope against the repaired physical distances; test slope limits, duplicated stations and repeatability |
| Repeated repairs compounded terminal tapering | Apply an absolute terminal envelope |
| Long gallery noise could make large lateral excursions | Smoothly bound correlated variation; test multiple lengths and seeds |
| Independent source systems produced compound islands in the trunk style | Form local islands after source confluence; interconnected mode retains compound interactions |
| Breakouts crossed nearby passages | Screen ranked branch sites against host bounds and other passage chords before adding them |
| A missing island identifier was counted as an island | Count only populated identifiers |
| Confluence transitions fused arms prematurely | Bound trunk-gallery transitions to four base widths |
| Section repair smoothed otherwise valid island routes closed | Preserve accepted routes and narrow conflicting width envelopes, then repeat clearance, topology and flow checks |

The previously failing long-multi seeds `0`, `2` and `3` are included in the campaign. Seeds `0` and `2` also have direct regression tests. No acceptance threshold was relaxed to pass these cases. Geometry can differ from earlier revisions because these are algorithm corrections; reproducibility is tied to the recorded implementation, configuration, inputs and runtime.

### Reproducibility and failure evidence

The new isolated campaign runner records every case, including timeouts and native crashes, and continues with subsequent cases. Each successful case is repeated in a fresh process using a different Python hash seed. It compares host, network and section identities; full cases also compare GLB bytes. Linux workers have a recorded address-space ceiling. A source edit during a campaign invalidates the overall pass flag.

Run manifests and evaluation share executing-package and runtime identity helpers. An installed wheel used outside a checkout hashes the actual imported package. Unavailable Git information is unknown rather than falsely clean. Checkpoint schema v2 binds configuration, inputs, source, the lockfile where available, numerical dependencies and Python identity; payload integrity is checked before reuse.

Portable validation now verifies that the selected asset is registered with its actual digest, or is a verified material revision of a registered source. A nearby successful manifest cannot attest an unrelated GLB. Intentionally untextured assets have an explicit neutral validation profile.

### Progress and performance

The command-line display now shows eleven overall stages and the current work unit. Host layers, candidate decisions, section sampling, voxel tiles, mesh work, floor raycasts, revalidation, UV batches, normal/tangent batches, collision, serialization and checkpoint work feed the display and `progress.jsonl`. Unknown totals show elapsed time without an invented percentage. A real command-line integration test verifies all eleven stage completions, the final manifest and output hashes, and proper progress closure.

Normal and tangent accumulation uses bounded triangle batches with scalar-reference agreement tests, including degenerate triangles/UVs and handedness. In a controlled 20,000-triangle helper benchmark, normal calculation fell from 0.846 s to 0.0114 s and tangent calculation from 0.417 s to 0.00912 s. These approximately 74× and 46× figures apply to those helpers only. UV unwrapping remains a substantial export cost. [Benchmark measurements](reliability-evidence-2026-09-12/shading-benchmark.json).

Exports now report visual triangles, vertices after UV seams, mesh-buffer bytes, collider triangles and file sizes. Optional triangle limits are checked before expensive surface preparation; file limits fail within atomic staging, before publishing a partial package. Disabling collision skips its computation. Material revisions reuse the existing UV buffer. Diagnostic figures sample at most 100,000 displayed triangles without changing the exported geometry.

### Cleanup and inspection

Removed the primitive/capsule/ellipsoid generation path, analytic room stamping, unused transported texture frames and geodesic/perimeter calculations. Route centres needed for inward surface orientation remain. Retired settings are rejected instead of silently accepted: `section_field.level_transition_fraction`, `geometry.use_section_profiles`, and both `geometry.junction_irregularity_*` controls. Remove them from old TOML files; historical checkpoints require their historical code.

Tube-only export, target descriptors, validation profiles and dense/tiled diagnostic plots now use shared package services. Blender inspection has preview/standard/high quality settings and denoising. A native Blender 5.2.0 LTS regression imports embedded maps, verifies shader connections, linear data maps and packing, and renders a controlled two-color fixture to catch grey/pink/missing-texture failures.

Deleted 500 archived Python copies totaling 6,512,023 uncompressed bytes. Their exact paths and recovery checkpoint are in the [retirement inventory](retired-source-inventory-2026-09-12.json). Research inputs, measured results, figures, tables, textures and user inspection scenes remain. The final package wheel is 371,221 bytes and contains the 90 maintained modules and packaged default; archived research source is excluded. [Distribution check](reliability-evidence-2026-09-12/distribution.json).

Large active generation modules remain. Shared progress, identity, shading and inspection responsibilities were extracted without rewriting every growth backend. Live general, layout, independent, interconnected, stacked, dense/tiled, structural-event and optional-rock paths remain supported. Historical metric readers remain active consumers of retained scientific data. The cleanup does not claim every public interface is fully typed or that all future refactoring is complete.

## Regression verification

All **591 current tests** have passed across the final full sweep and subsequent targeted runs, with **59 passing subtests**. The optional Manim module was skipped. The full sweep recorded 581 passes and one sediment-fixture failure; after investigating the failure, the corrected test and the nine newly added cases passed. This is a combined verification result, not a claim that the first full sweep exited successfully.

The infill proposal in that fixture was rejected while two collapses and a choke were applied. Demanding sediment for the rejected proposal was incorrect. The corrected assertion checks applied geometry; seven focused tests cover accepted/rejected infill and choke, surviving floor margins, and retained collapse talus. Connectivity, watertightness, floor contact and prop validity assertions remain. [Observed event evidence](reliability-evidence-2026-09-12/rejected-infill.json).

| Check | Result |
|---|---|
| Full sweep | 581 passed, one subsequently corrected test, 59 passing subtests; 1,064.92 s |
| Corrected event and related floor checks | 9 passed; 97.78 s |
| Two additional pinned failing-seed regressions | 2 passed; 66.18 s |
| Final display correction and affected integration checks | 93 passed; 49.92 s, including the real CLI, native Blender, seed regressions and shading/export checks |
| Package line coverage | 13,608 / 16,087 statements, **84.59%** in the full sweep |
| Ruff / mypy / compilation | Passed; mypy covers 90 package modules |
| Lock / installed dependency consistency | Passed; 42 installed packages compatible |
| Native Blender | Import, shader-binding, image-packing and CPU-render checks passed |
| Distribution | Wheel/sdist built; imported extracted wheel from an empty working directory, loaded its default and verified executing-source identity |

The wheel probe used the available installed dependencies; it was not a fresh online installation. CI now declares the paper extra needed during collection, retains the base-install job, and runs against Python 3.12 and 3.13. Remote CI itself was not executed locally.

Retained logs: [full sweep](reliability-evidence-2026-09-12/full-suite.txt), [event correction](reliability-evidence-2026-09-12/event-correction.txt), [pinned seeds](reliability-evidence-2026-09-12/pinned-seeds.txt), [coverage by module](reliability-evidence-2026-09-12/coverage-summary.json).

The last display test exposed Rich's `total=None` semantics: resetting a task retains its old total. The fix replaces the active task when the work unit or total changes. The test now inspects the displayed task during known/unknown transitions, rather than checking only the trace. This fix was applied after the campaign completed. A per-module hash comparison confirms that only `progress.py` changed; the other 89 modules are identical. The affected 93 tests passed, and an additional fresh-process single-network case plus replay matched the earlier campaign's semantic hashes. [Display-change identity and replay](reliability-evidence-2026-09-12/display-fix.json), [final test log](reliability-evidence-2026-09-12/display-release-tests.txt).

## Seed and export evaluation

All **61 cases and all 61 fresh-process replays passed**: 59 network-and-section cases and two complete mesh/export cases. This represents 122 isolated executions, not 122 full caves. The additional display-change smoke replay is separate. [Complete case records, runtime, identities and timings](reliability-evidence-2026-09-12/seed-campaign.json).

| Scenario | Cases | Scope |
|---|---:|---|
| Short Earth single / multi / interconnected | 8 each, 24 total | Network and sections |
| Long Earth single / multi / interconnected | 8 each, 24 total | Network and sections |
| Body-scaled project preset on Mars / Moon | 4 each, 8 total | Network and sections |
| Interacting systems on Moon, previously timed-out seeds | 3 | Network and sections |
| Short Earth single seed 0 / interconnected seed 17 | 1 each, 2 total | Full geometry, both floor passes, export and portable validation |

The eight Earth root seeds are `0, 1, 2, 3, 17, 42, 20260910, 4294967295`; body-scaled cases use `0, 1, 17, 4294967295`. The additional Moon cases use `17, 20260910, 4294967295`. Each root seed also varies the host through derived stage seeds. The worst accepted long-multi case needed candidate 46 of its configured 64, demonstrating why both retry coverage and an explicit exhaustion outcome matter.

The main 59-case matrix used up to six concurrent workers, 600 seconds and 8 GiB of address space per execution. Two additional Moon rechecks used 900 seconds. Source identity remained unchanged throughout each campaign. Timings under concurrent load are observations, not controlled speed benchmarks.

Both full cases used an explicitly recorded 0.22 m voxel override, neutral materials, and disabled events/collision. They do not claim full rock-population or textured planetary evaluation. Both meshes were finite, closed, consistently wound and nondegenerate; each passed all 38 portable-asset checks and reproduced identical GLB bytes.

| Full case | Combined passage length | Triangles | GLB | OBJ fallback | Worker peak RSS |
|---|---:|---:|---:|---:|---:|
| Single, seed 0 | 528.6 m | 326,596 | 12.65 MB | 39.55 MB | 893.9 MiB |
| Interconnected, seed 17 | 1,033.8 m | 611,426 | 23.65 MB | 75.87 MB | 1,511.4 MiB |

Lengths include all branches; they are not the preset's longitudinal target. File sizes use decimal MB; process memory uses binary MiB. Generation RSS is not simulation runtime memory. Detailed size reports: [single](reliability-evidence-2026-09-12/single-export-size.json), [interconnected](reliability-evidence-2026-09-12/interconnected-export-size.json). Local generated assets remain under `tmp/reliability_cleanup_20260912/confirmed_matrix/case_056/case_0000/export/` and `case_057/case_0000/export/`; binary caves are not committed as test fixtures.

## Interpretation and remaining limits

Passing procedural tests is not geological validation against a surveyed cave. A finite seed sample cannot guarantee every integer seed or incompatible configuration succeeds. The generator must still be allowed to reject an impossible domain or exhaust a bounded search with its quality report.

Mesh budgets describe exported files and buffers. Engine allocations, decoded textures and mipmaps consume additional memory. There is no automatic LOD, decimation, streaming system or guarantee of native Unity/Unreal import. Those application checks remain separate. The native Blender fixture detects missing material delivery, but does not prove every generated cave is free of UV seams, grazing-light artifacts or viewport sampling noise.

Earlier failed and interrupted development campaigns remain separately recorded in [development-failures.json](reliability-evidence-2026-09-12/development-failures.json). They are not counted as final passes. Earth-sized fixed hosts overridden to Mars/Moon were incompatible, and a development full-export case exceeded its 300-second limit while UV batches were still advancing. Larger time allowances address measured cost; they do not waive geometry or reproducibility checks.

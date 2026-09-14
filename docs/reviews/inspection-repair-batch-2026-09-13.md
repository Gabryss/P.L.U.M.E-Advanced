# Inspection and repair evaluation — 13 September 2026

**Result: 9/10 full cases passed; four required surface repair. Two cold replays matched exactly. A collider fallback-reporting defect was fixed and verified, but multi-source seed 1 still fails geometry acceptance at both tested resolutions.**

This evaluation exercises the embedded pipeline, including its failure paths. A rejected case demonstrates detection, not successful repair. Passing a finite batch cannot guarantee that all seeds or configurations work.

## Reproduction and scope

Evidence is retained in `outputs/inspection_repair_batch_20260913/`. The offline `report.html` indexes the per-case quality reports, repair decisions, measured-passage figures and exported assets. `evaluation_summary.json` and `integrity_audit.json` support automated inspection. Keep the output directory to retain these measurements; it is not tracked in Git.

The two recipes derive from the maintained short single-source and interconnected presets. Each uses root seeds **1, 7, 17, 42 and 20260913**, Earth, a shared host field, no rocks or structural events, neutral materials, collision export, fixed 0.12 m voxels, 4 m density padding and a 12-million-voxel dense-storage threshold. Export budgets are five million visual triangles and 350 MB per asset. These are recorded evaluation overrides, not the unchanged production presets. In particular, the maintained interconnected full preset uses 0.08 m voxels.

The first ten requests used a 150 m route target. All were rejected during network screening because the preset requirements could not fit: island/branch spacing for the single-source cases and persistent parallel/merge/split opportunities for the interconnected cases. Their bounded candidate and repair histories remain in the `single` and `multi` directories. No mesh was published. They are failed feasibility trials, not ten generated caves.

The full follow-up changes only the route target to **250 m**, keeping the same ten root-seed/mode combinations and quality thresholds. Its recipes, exact hashes and source identity are recorded in `evaluation_plan_250m.json`. The target describes route length; combined branch lengths are larger. Campaign workers have 1,800-second wall-time and 8,192 MiB address-space limits. The two campaigns ran concurrently, so timings are operational measurements, not controlled speed benchmarks.

Only root seed 17 in each mode receives a predefined cold replay, with `PYTHONHASHSEED=7919` instead of 11 and no checkpoint reuse. Other cases have no cold-replay claim. Replays compare host, network and section identities, exact mesh and GLB hashes, surface-repair decisions and export-inspection records.

Executing production-source SHA-256:

```text
7c370b361bba69f37b3e2b5363228a4e94dd5fea141ae46425a03036c14c564b
```

## Controlled regressions

**117 focused tests passed**, covering actual-mesh containment and topology; genuine cracks versus duplicate UV-seam vertices; structural-event route protection; configured material bindings; corruption of serialized GLB/OBJ/USD; visual repair; collision fallback; exhausted repair rejection; preservation of previous exports; source changes; timeouts; and campaign resumption/integrity. The exact selection and results are in `controlled_checks.log` and `controlled_checks.xml`.

Two captured local density regressions were also rebuilt twice. The branch regression required half-strength relief; the merge-neck regression required a one-voxel opening. Both ended with accepted topology and route containment, identical repeated meshes/repair decisions, and unchanged input density. Only host-cover enforcement is omitted for these artificial crops, whose cut boundaries have no physical host-cover meaning. These local controls are not two additional full cave generations.

## Pipeline integration corrected during this evaluation

Full evaluation workers already used the production generation and export gates, but omitted the shared final run-inspection step. They now call it, retain the quality report and measured-passage figure, and include their hashes in the run manifest. Worker failures also retain a failed quality report. Material validation requires configured maps while supporting intentionally partial materials. No acceptance threshold was weakened to make the batch pass.

## Interpretation limits

A pass covers finite, closed, consistently oriented geometry; expected topology; sampled route-centre containment; visual/collision preparation checks; serialized geometry and configured material bindings; file budgets; and recorded artifact integrity. Resolution warnings remain visible. The batch does not render textures or certify Blender/Unity/Unreal behavior, geological appearance, continuous rover clearance, every triangle self-intersection or resolution convergence. It cannot establish a population-wide failure rate from five selected seeds per mode.

## Remaining seed-1 mesh failure

At 0.12 m, the interconnected seed-1 case exhausted all six base-surface candidates. Its expected genus was zero. The requested surface had genus five; removing relief still left two components and incorrect topology. The final one-voxel opening produced one component but genus four. No export was published.

A separate base-only resolution study reused the exact accepted network and sections (their semantic identities were verified) at **0.08 m**. It also exhausted six candidates. With full relief the mesh had genus eleven; without relief it retained two components and genus three. The last opening candidate produced one component but still genus one. This rules out claiming that an 8 cm setting alone fixes the case. It does not establish convergence or identify every local cause: 312/641 input profiles fall below the eight-voxel heuristic at 0.12 m, and 143/641 still do at 0.08 m.

The inspection correctly catches this defect, but the current bounded repairs cannot resolve it. A subsequent geometry correction should retain this seed, localize the unwanted connections/pockets, add a captured regression, and rerun both resolutions. Raising the allowed genus or dropping required routes would conceal the problem. The study took about 705 seconds; see `resolution_probe_1_0.08_v2/report.json`. An earlier auxiliary probe was interrupted to correct its progress logger and is explicitly excluded from completed evaluation counts.

## Full 250 m results

**9/10 cases passed; 1 were rejected. 4 passing cases needed base-surface repair. Both predefined cold replays matched exactly.**

All completed-case file audits passed. Both campaigns completed under unchanged source and inputs. The single-source campaign passed overall; the multi-source campaign correctly failed its overall gate because a case was rejected. The two cold replays are additional verification runs, not additional distinct caves.

| Mode | Root seed | Result | Base attempts | Relief scale | Route samples | GLB MB | Package MB | Collider triangles |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| single | 1 | Passed | 1 | 1 | 357 | 29.9 | 160.7 | 795,184 |
| single | 7 | Passed | 1 | 1 | 325 | 29.4 | 157.8 | 782,028 |
| single | 17 | Passed | 3 | 0.25 | 323 | 30.2 | 162.6 | 804,012 |
| single | 42 | Passed | 1 | 1 | 327 | 28.3 | 152.5 | 751,584 |
| single | 20260913 | Passed | 1 | 1 | 401 | 31.4 | 168.7 | 833,420 |
| multi | 1 | Rejected | 6 | None accepted | — | — | — | — |
| multi | 7 | Passed | 2 | 0.5 | 671 | 50.0 | 270.6 | 1,319,432 |
| multi | 17 | Passed | 2 | 0.5 | 647 | 55.9 | 302.7 | 1,477,056 |
| multi | 42 | Passed | 3 | 0.25 | 597 | 49.1 | 265.9 | 1,298,406 |
| multi | 20260913 | Passed | 1 | 1 | 611 | 49.7 | 269.4 | 1,316,474 |

MB is decimal. Package size includes GLB, collider OBJ, fallback OBJ and supporting files; decoded engine memory is additional. All measured collider counts equal their original mesh counts in this batch. The pipeline preserved integrity, but this is not evidence of an efficient physics collider. All accepted cases retained section-resolution warnings.

### Repair-path coverage

| Path | Evidence |
|---|---|
| Network candidates and repairs | Real full cases, including multi-source seed 17: candidate index 25 (26th candidate), repair pass 2; identical cold replay decisions. |
| Surface detail reduction | Real full cases plus captured branch regression; original and replay decisions match. |
| Density opening | Captured merge-neck regression passes twice; failed seed 1 exhausts this repair too. |
| Visual smoothing/displacement fallback | Controlled tests; every passing full case accepted its first visual candidate. |
| Collider fallback | Real full cases retained their original collider; the reporting bug below hid that fact in the original flags. |
| Serialized corruption and preservation | Controlled GLB/OBJ/USD corruption tests reject publication; full-case serialized checks and independent file hashes pass. |

## Collider reporting correction after the frozen batch

The evaluation found a real observability bug: `simplified_collision_arrays` could reject its clustered mesh internally and return the original arrays, while the surrounding export report still said `used_raw_fallback=false`. A new regression reproduced that exact mismatch before the fix (`collision_reporting_before.log`). The original exported meshes were preserved correctly; the repair evidence was incomplete.

The fix passes fallback evidence from the simplifier to the shared export inspection, records a reason, and emits a progress event. New tests cover both scene preparation and serialization into the final run-quality report. The expanded focused selection passed **120 tests**, including the pre-existing direct simplifier compatibility test; Ruff, Mypy and whitespace checks passed. These are 120 distinct post-fix tests, not 117 plus 120 different tests.

After the campaigns finished, collision preparation was repeated against all **9 accepted saved meshes**. Each now reports its fallback and reason, and every collider array hash matches the original published inspection. No old export or original quality report was rewritten. `post_fix_collision_audit.json` records this limited stage replay and its source identity; it is not another full generation or cold replay.

Post-fix source SHA-256:

```text
55fa02f0350c763cee4de6e4058600e504f0c2a4e47702efdaf019e27ce05c49
```

The two full cold-replay claims above apply to the frozen batch source. The later reporting correction is supported by the new regression tests and identical collision-array hashes. The seed-1 geometry defect remains unresolved; a truthful rejection is not a successful repair.

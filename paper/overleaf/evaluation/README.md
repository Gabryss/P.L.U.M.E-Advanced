# PLUME evaluation campaign, 7 September 2026

> Historical code was removed during the 12 September cleanup. Restore commit `89d7f6a` to run the commands described below. Inputs, results and figures in this document remain historical evidence.

**Completed:** all 2,076 declared cases were attempted. There are 2,071 completed
cases and five dense 5 km memory-limit failures. Read the [measured results](REPORT.md)
and the machine-readable `report.json`. All case configuration identities and
both source snapshots passed the final provenance audit.

The results are mixed: selected shape descriptors improve relative to matched
ellipses, while sizes remain mismatched; tiled storage completes the resource
suite within budget; adaptive sampling has greater point-set discrepancy in all
30 pairs; distributary tendency has no median cycle-rank response. All 33 A–C
repeatability checks and all 15 visual/collision package-presence checks pass.
No application imports were performed. These outcomes are not a certificate of
geological accuracy, final-mesh fidelity or robotics performance.

This campaign executes the full declaration in `paper/experiments.toml` against
the modified project state preserved under `frozen/`. The source and configuration
were copied before reading the evaluation partition's morphology. No generator
parameters are adjusted using this campaign's results. Earlier whole-catalog
exploration is disclosed in `paper/splits/README.md`; the evaluation partition is
not historically unseen.

Before any controllability cases ran, a static review found an outdated sweep
implementation: manually editing selected derived settings missed current lobe
consumers and could resize the duration grid around a source left outside it.
`controls_amendment.json` records the correction and `frozen_controls_v2/` preserves
its source. The sweep now resolves every stage from the actual flow control, as
an edited project file would do. All 12 settings were tested against literal TOML
edits, and the default configuration hash is identical between revisions. This
amendment does not tune the generator to morphology results. The other six
experiments use the original snapshot. The control sweep uses six workers; other
network experiments use four and resource benchmarks remain sequential. One
driver restart retained 100 morphology and 280 routing case records; only
unfinished routing work was restarted.

## Scope and counts

| Experiment | Declared cases | Measurement |
| --- | ---: | --- |
| Morphometry | 100 | Earth Stage-C profiles versus the 19-cave PDC evaluation partition and dimension-matched ellipses |
| Routing ablation | 700 | 100 seeds, full model and six routing interventions |
| Scalability | 40 | Five seeds, four requested lengths, dense and tiled reconstruction |
| Controllability | 1,200 | 100 seeds, four controls, three levels each |
| Sampling | 30 | Adaptive and approximately count-matched uniform profiles against a 1 m reference |
| Determinism | 3 | Repeats, checkpoints and downstream-setting invariance of A-C artifacts |
| Export consistency | 3 | Five target packages per development-size world |

These are 2,076 experiment cases, not 2,076 fully meshed caves. Only the scalability
and export experiments reconstruct meshes. Optional rock/event generation is
outside these two experiments. The existing rock illustration remains a separate
development example. Automatic package checks do not substitute for application
imports or collision/contact tests in a simulator.

The standard Earth configuration resolves to 0.6 m voxels. The benchmark records
profile-resolution warnings; this spacing is not claimed to resolve narrow
passages or reproduce the separate 0.2 m inspection illustration. Scalability
timing includes host, network, sections, density/roof screening, final meshing and
welding, plus child-process startup. It excludes appearance preparation, optional
events, exports and application import. Cases run individually with a 7,200 s
timeout and a monitored 12 GiB process-tree RSS limit. Failed and timed-out cases
remain in the denominator; time and memory medians use completed cases only.

## Corrections before the freeze

- The scalability worker now actually creates the final mesh; previously it
  measured only density construction.
- Routing effects now use matched seeds and paired bootstrap intervals.
- The morphology aggregate has its own joint cave/world bootstrap interval.
  Cached empirical distributions preserve the exact repeated-sample estimator
  while avoiding repeated sorting. Distances remain section-weighted.
- Worker failures, timeouts, memory-limit failures and non-finite metrics are
  retained as explicit case outcomes. Independent network/profile work uses
  four processes (six for the amended control sweep), with one numerical-library thread each. Resource benchmarks
  remain sequential.

Validation before freezing: 32 evaluation tests passed; focused type checking
and lint checks passed. A 500 m tiled standard-quality preflight completed through
meshing. A separate small packaged-default fixture failed at graph validation;
it is not included in campaign results and was not used to tune the generator.

## Running and inspecting

From the project root, with the project and optional `paper` dependencies installed:

```bash
.venv/bin/python paper/campaigns/2026-09-07/run_campaign.py
```

The driver selects the preserved package for each experiment via `PYTHONPATH`. It runs all declared
experiments even if an experiment reports failed cases. A completed command is
not silently retried, including commands with unsuccessful cases. `execution.json`
records command status, timestamps and log locations. `logs/` holds full output;
`paper/outputs/*/cases/` holds per-case results. The original declaration and the
effective declaration are both retained; only the output destination differs.

`freeze_manifest.json` contains hashes of all preserved files and the referenced
texture assets. `requirements_frozen.txt` records installed dependency versions.
The source snapshot includes preexisting uncommitted changes, so its file hashes
are the authoritative source identity, alongside the base commit. Texture files
are referenced through a symlink and separately hashed; they must also accompany
a portable reproduction bundle. The external PDC dataset is identified by its
content hash in the results and is not redistributed with the paper.

Once every experiment has produced the declared number of case records:

```bash
.venv/bin/python paper/campaigns/2026-09-07/build_report.py
```

This derives manuscript Tables IV-VI and the campaign report from saved results.
The report distinguishes completed measurements, failed cases, failed checks,
and unperformed application imports.

## Auditing and replotting saved measurements

`audit_provenance.py` reconstructs each experiment's provenance and all resolved
case configurations without regenerating a network or changing raw records. It
refuses to save an audit if a case identity differs. Run it with the same frozen
source, thread environment and worker count recorded for that experiment.
Verified configurations and full provenance are stored in `verified_provenance/`.
For benchmark failures, the actual project configuration is reconstructed from
frozen inputs; completed benchmarks additionally match the hash emitted by the
worker. The raw benchmark record's `resolved_config_sha256` refers to its
experiment declaration, while `project_config_sha256` identifies the worker's
resolved project. The audit makes this distinction explicit.

`plot_morphometry.py` creates Figure 9 from saved section descriptors.
`plot_scalability.py` creates Figure 10 from all 40 resource records, retaining
stopped cases. The latter loads `reporting_source/plotting.py`, a preserved
postprocessing correction that suppresses overlapping logarithmic tick labels
and includes failures omitted by the original plotter. Neither script changes
experimental measurements. Their figure manifests record input and image hashes.
After the campaign driver's generic `figures` command, run these scripts to
produce the manuscript figures:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-evaluation-mpl \
  .venv/bin/python paper/campaigns/2026-09-07/plot_morphometry.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/plume-evaluation-mpl \
  .venv/bin/python paper/campaigns/2026-09-07/plot_scalability.py
```

Once all 40 isolated resource measurements had finished, the three sequential
export cases were started alongside the remaining parameter sweep.
`concurrent_exports.json` and `logs/exports-concurrent.log` record this overlap.
They use the original frozen source and the same export provenance environment.
The main driver later encounters and reuses matching completed export cases;
its quick replay is not an export-runtime measurement. This overlap does not
apply to any scalability case. `run_exports_concurrent.py` preserves the helper.

After exports finished, `run_auxiliary_concurrent.py` started the 30 sampling
cases and then the three repeatability cases alongside the remaining control
sweep. Each uses the original frozen source and four configured workers;
`concurrent_auxiliary.json` records actual execution times and separate logs.
As with exports, the main driver later reuses matching completed records. Only
non-benchmark experiments overlap; resource costs were already fully recorded.

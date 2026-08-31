# Claim-to-evidence matrix

| Paper claim | Experiment | Metric | Raw source | Status |
|---|---|---|---|---|
| Host conditioning affects routing | E5 | paired centerline displacement and exposure | `paper/outputs/host_ablation/raw_results.*` | Instrumented; campaign pending |
| Host conditioning affects topology | E5 | cyclomatic/split/merge deltas | same | Instrumented; campaign pending |
| Networks are topology-rich | E2/E5 | cycles, branches, underpasses, z levels | evaluation cases | Metrics implemented |
| Earth profiles have morphometric plausibility | E1 | clustered normalized W1 and KS | `paper/outputs/morphometry/*` | Calibration pilot improved; confirmatory campaign pending |
| Adaptive sampling saves cost at matched error | E6 | surface error vs count | `paper/outputs/sampling_ablation/*` | Claim conditional |
| Tiled geometry improves scalability | E7 | peak RSS and wall time | scalability cases | Campaign pending |
| Canonical scale survives packaging | E9 | digest/bbox/collision | export cases | Checks pending |
| Runs are deterministic and isolated | E10 | semantic hash equality | `paper/outputs/determinism/*` | Instrumented |

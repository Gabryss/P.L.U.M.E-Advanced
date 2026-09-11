# Claim-to-evidence matrix

| Paper claim | Experiment | Metric | Raw source | Status |
|---|---|---|---|---|
| Host conditioning affects routing | E5 | paired centerline displacement and exposure | `paper/outputs/host_ablation/raw_results.*` | 100 paired seeds completed; single-term median shifts 4.79-8.34 m; constant cost 12.39 m |
| Host conditioning affects topology | E5 | cyclomatic/split/merge deltas | same | 100 paired seeds completed; individual median cycle changes -1 to +1, intervals include zero; constant cost +1.5 [1, 2] |
| Formation graphs contain cycles | E2/E5 | cycle rank | evaluation cases | Full routing model has median cycle rank 12 across 100 seeds; not a final-void connectivity or robotics benefit claim |
| Inputs control their declared readouts | E2 | paired high-minus-low primary metrics | `paper/outputs/controllability/*` | 1,200 cases complete: duration +3.810 km [3.682, 4.929]; inflation ratio +0.0318 [0.0162, 0.0490]; distributary cycle change 0 [0, 0]; supply readout is an input-resolution check |
| Earth profiles have morphometric plausibility | E1 | clustered normalized W1 and KS | `paper/outputs/morphometry/*` | 100 worlds / 19 evaluation caves completed; aggregate distance 0.254 versus ellipse 0.840; size and variability mismatches remain |
| Adaptive sampling lowers sampled-point discrepancy at approximately matched count | E6 | paired boundary-point discrepancy and count | `paper/outputs/sampling_ablation/*` | Not supported: 30/30 pairs have larger adaptive mean discrepancy; paired increase 0.439 m [0.432, 0.454], with 29.5 fewer sections. No continuous-surface or timing conclusion |
| Tiled geometry lowers memory for these cases | E7 | peak RSS, wall time and completion | `paper/outputs/scalability/*` | 35/40 complete; tiled 20/20, dense 15/20; five 5 km dense cases hit 12 GiB cap. No uniform time advantage or mesh-fidelity claim |
| Target packages contain visual and collision assets | E9 | file presence and extent diagnostics | `paper/outputs/export_consistency/*` | 15/15 packages present across three worlds; nine parsed GLBs, max extent difference 0.174% including visual preparation; zero application imports |
| A–C repeats, checkpoints and downstream-setting isolation preserve semantics | E10 | semantic hash equality | `paper/outputs/determinism/*` | All 33 checks pass across three seeds; no final-mesh, container-byte or cross-platform claim |

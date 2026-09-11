# Evaluation campaign results

Source freeze: `985011be995430320281772e3b149adb59e998471605f3381b2dea8dac85f051`.

All declared cases were attempted. Completion means the experiment ran; it does not mean every scientific hypothesis or quality criterion was supported.

| Experiment | Planned | Completed | Other outcomes |
| --- | ---: | ---: | --- |
| morphometry | 100 | 100 | None |
| controllability | 1200 | 1200 | None |
| host_ablation | 700 | 700 | None |
| sampling_ablation | 30 | 30 | None |
| scalability | 40 | 35 | failed: 5 |
| export_consistency | 3 | 3 | None |
| determinism | 3 | 3 | None |

## Cross-section morphology

Reference: 200 usable sections from 19 evaluation caves. Generated: 118257 sections from 100 worlds.

Normalized Wasserstein distance (lower is better); Advanced intervals resample caves and worlds.

| Descriptor | Ellipse | Advanced | Advanced 95% interval |
| --- | ---: | ---: | --- |
| Width | 0.3536 | 0.3536 | [0.326, 0.6949] |
| Height | 0.4175 | 0.4175 | [0.2907, 0.7956] |
| Aspect ratio | 0.2572 | 0.2572 | [0.2287, 0.3521] |
| Area | 0.554 | 0.5379 | [0.3202, 1.142] |
| Compactness | 0.9808 | 0.4409 | [0.3454, 0.7019] |
| Floor residual | 1.057 | 0.232 | [0.1674, 0.4587] |
| Roof asymmetry | 1.065 | 0.08456 | [0.07294, 0.3909] |
| Selected aggregate | 0.8401 | 0.2537 | [0.2267, 0.4172] |

The aggregate averages aspect ratio, compactness, floor residual and roof asymmetry. Width, height and aspect ratio match the ellipse by construction. These are Stage-C comparisons, not validation of final meshes or planetary distributions.

Absolute sizes remain mismatched: generated/reference median width is 8.16/5.97 m, and median height is 4.64/3.50 m.

## Routing interventions

Effects are paired medians relative to the full model; brackets are 95% bootstrap intervals.

| Condition | Pairs | Mean centerline displacement (m) | Cycle-rank change |
| --- | ---: | --- | --- |
| No slope term | 100 | 8.005 [7.189, 8.856] | 1 [0, 2] |
| No cover term | 100 | 4.793 [4.042, 5.436] | 0 [0, 0] |
| No fracture term | 100 | 8.336 [7.572, 9.595] | -1 [-1, 0] |
| No capacity term | 100 | 6.929 [5.865, 8.269] | 0 [-1, 0] |
| No stability term | 100 | 5.706 [4.997, 6.554] | 0 [-1, 0] |
| Constant cost | 100 | 12.39 [11.44, 12.99] | 1.5 [1, 2] |

## Resource measurements

Time and RSS medians include completed cases only. Completion denominators include failures.

| Requested route (km) | Mode | Completed | Median time (s) | Median peak RSS (GiB) |
| --- | --- | ---: | ---: | ---: |
| 0.5 | dense | 5/5 | 29.21 | 1.716 |
| 0.5 | tiled | 5/5 | 34.94 | 0.4973 |
| 1 | dense | 5/5 | 49 | 3.979 |
| 1 | tiled | 5/5 | 51.47 | 0.7353 |
| 2 | dense | 5/5 | 73.32 | 10.9 |
| 2 | tiled | 5/5 | 73.72 | 1.028 |
| 5 | dense | 0/5 | Unavailable | Unavailable |
| 5 | tiled | 5/5 | 126.7 | 1.65 |

## Parameter response and sampling

High-minus-low control effects and adaptive-minus-uniform sampling effects are paired by seed. Brackets contain pointwise 95% bootstrap intervals.

| Control | Primary readout | Levels (low to high) | Pairs | Median change [95% interval] |
| --- | --- | --- | ---: | --- |
| Distributary | Cycle rank | 0.2 to 0.8 | 100 | 0 [0, 0] |
| Duration | Main-route length (m) | 0.5 to 1.5 | 100 | 3810 [3682, 4929] |
| Inflation | Junction / passage width | 0.2 to 0.8 | 100 | 0.03176 [0.01621, 0.04903] |
| Supply | Resolved host scale (input check) | 0.7 to 1.3 | 100 | 0.3035 [0.3035, 0.3035] |

The supply readout is a resolved input scale, not an independent geometry response.

| Sampling readout | Pairs | Adaptive minus uniform [95% interval] |
| --- | ---: | --- |
| Mean point discrepancy (m) | 30 | 0.4392 [0.4323, 0.4541] |
| 95th-percentile point discrepancy (m) | 30 | 2.107 [2.066, 2.169] |
| Section count | 30 | -29.5 [-32, -27] |

Counts are approximately matched; point-cloud discrepancies do not establish continuous-surface accuracy or savings at matched error.

Adaptive mean discrepancy is larger in 30/30 completed pairs. This comparison does not support an adaptive-sampling advantage under its point-set metric.

## Determinism and export checks

| Determinism check | Passed for every completed case |
| --- | --- |
| events off host equal | Yes |
| events off network equal | Yes |
| events off sections equal | Yes |
| export changed host equal | Yes |
| export changed network equal | Yes |
| export changed sections equal | Yes |
| resume checkpoint reused | Yes |
| resume sections equal | Yes |
| same run host equal | Yes |
| same run network equal | Yes |
| same run sections equal | Yes |

Export experiment: 3/3 cases completed. Visual/collision package-presence checks: passed. Manual application imports: 0.

Completed export cases mean the experiment ran. Its pass flag checks visual/collision file presence; it does not certify orientation, materials, final mesh identity after import, or simulator contact.

## Failure inventory

| Experiment | Failure class | Cases |
| --- | --- | ---: |
| scalability | memory_limit | 5 |

Tables IV-VI are generated under `paper/overleaf/results/`.

## Scope



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


Full measurements, realized geometry ranges, supplementary paired effects and summary hashes are in `report.json`.

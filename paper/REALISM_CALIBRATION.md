# Cross-section realism calibration log

The initial 100-world whole-catalog run was exploratory and showed narrow size
and aspect-ratio distributions, overly compact profiles, and floors that were
too uniform. A deterministic 76-cave calibration / 19-cave confirmatory split
was frozen before parameter changes. Seventy-two self-intersecting calibration
sections were excluded by the predeclared simple-polygon rule; the confirmatory
partition contains no detected self-intersections.

The matched three-seed calibration pilot uses seeds 0, 1, and 2 and 1,286 valid
reference sections. Lower normalized Wasserstein distance is better.

| Metric | Original pilot | Frozen realism pilot |
|---|---:|---:|
| Aggregate selected metrics | 0.911 | 0.477 |
| Aspect ratio | 0.679 | 0.560 |
| Compactness | 0.856 | 0.608 |
| Floor residual | 1.557 | 0.462 |
| Roof asymmetry | 0.554 | 0.278 |

The frozen pilot's matched-ellipse aggregate is 0.898, so the procedural profile
is closer on the selected shape descriptors in calibration. No confirmatory
partition metric was evaluated during parameter selection. The pilot is model
development evidence, not the paper's final result; the final evaluation must
run from a clean commit with `reference_partition = "evaluation"`.

The implementation adds bounded, deterministic segment-scale width and height
variation, smooth longitudinal modulation, multi-harmonic wall relief, floor
relief, and a broader asymmetry field. Height variation is tapered at segment
endpoints, and absolute width/height limits preserve connected voxel and mesh
geometry.


## Subsequent evaluation campaign

The frozen 7 September 2026 campaign has now been completed. Its [results](campaigns/2026-09-07/REPORT.md)
use 100 Earth worlds and the 19-cave evaluation partition. They do not change the
historical calibration results above. Selected shape distances improve against
matched ellipses, while absolute sizes and variability remain mismatched. No
parameters were retuned using the evaluation outcomes.

# Interconnected-system validation — 2026-09-11

This campaign screens the shared-host interconnected mode through Stages A–C. It does not generate or validate cave meshes, rocks, Unity scenes or Unreal scenes. The thresholds are procedural morphology targets, not geological certification.

## Results

| Preset | Network seed offset | Accepted candidate (1-based) | Repair passes | Parallel coverage | Merges / splits | Total passage length |
|---|---:|---:|---:|---:|---:|---:|
| short | 0 | 1 | 0 | 100.0% | 2 / 2 | 1,015.6 m |
| short | 1 | 1 | 1 | 100.0% | 2 / 1 | 1,068.3 m |
| short | 2 | 1 | 0 | 74.4% | 3 / 2 | 917.8 m |
| long | 0 | 2 | 3 | 83.0% | 11 / 11 | 7,260.0 m |
| long | 1 | 4 | 2 | 70.9% | 13 / 11 | 6,322.7 m |
| long | 2 | 3 | 1 | 90.8% | 11 / 9 | 7,038.6 m |

The short preset spans 400 m downstream; the long preset spans 3,000 m. All six distinct network/section realizations passed 50 acceptance checks each. Independent seed variants share an identical physical host within each scale. The accepted networks include sustained, physically separated parallel routes and interactions distributed downstream. One of the short realizations has fewer junctions and a different shared-passage arrangement; acceptance does not require identical loop counts.

Both offset-zero presets were replayed in separate interpreters with Python hash seeds 11 and 37. Host, network and section semantic hashes matched, and the complete acceptance reports were byte-identical. The replay directories contain repeated verification results, not additional distinct cave designs.

The selected regression suite passed 169 tests and 26 subtests. Following the last metric/phase refinements, all 15 focused interconnected tests passed. The focused suite covers deterministic replay, host immutability, source/host sensitivity, impossible-host rejection, local event spacing, profile fusion, inlet-only branching, junction-preserving repairs, starved split phases and invalid controls. Lint and whitespace checks passed.

## Reproduce

```bash
uv run --no-sync python scripts/validate_interconnected.py \
  --config config/earth_short_interconnected.toml \
  --output outputs/interconnected_short_validation --seed-offsets 0 1 2

uv run --no-sync python scripts/validate_interconnected.py \
  --config config/earth_long_interconnected.toml \
  --output outputs/interconnected_long_validation --seed-offsets 0 1 2
```

Use new empty output directories. The script stores the base network seed, selected retry seed, resolved configuration, unchanged host identity and every rejected candidate/repair. Rejected layouts in this campaign included sharp junctions, route crossings, uphill sections, nonlocal overlaps, footprint cycle mismatches and long event-free reaches. No failed final candidate was delivered as accepted.

Production package source SHA-256: `6a2bdbdcb89b5b3158af9b2a32684083c659395a54d9ab28b3d1d754836dc0a3`.

The [inspection index](../outputs/interconnected_validation/README.md) links all 24 verified stage PNGs and the per-case artifacts. The [machine-readable audit](../outputs/interconnected_validation/campaign_audit.json) contains case identities and metrics. These generated output links remain available only while the local output directory is retained; the reproducible presets and this report remain in the repository.

## Interpretation and limits

The new mode demonstrates the requested interconnected procedural topology under the two supplied Earth host scenarios. Parallel coverage is measured from Stage-C profile envelopes with a minimum intervening rock gap, excluding blind branches. It is not the older centreline-only single-channel metric, so the two percentages should not be compared as if measured identically.

Final voxel/mesh topology, continuous clearance, surface relief, rock-pillar structural behaviour and application imports still require downstream validation. Multi-level capture and thermofluid/deposition feedback are not implemented by this mode. See [the generation model and active controls](interconnected_systems.md).

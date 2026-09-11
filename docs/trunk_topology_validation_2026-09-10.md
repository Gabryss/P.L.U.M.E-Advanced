# Dominant-gallery topology validation — 2026-09-10

Scope: the Earth dominant-gallery preset, its network and sampled sections.
No full meshes, loose rocks, Blender/Unity/UE import checks, scan fitting or
independent geological validation were performed in this campaign.

## Evidence identity

- Config: [earth_valentine_topology.toml](../config/earth_valentine_topology.toml).
- Production source SHA-256: `44d6f8d3092e0f3ffaf908fca670cd6b6a90959ca3e81cf868dd5683e2c5e035`.
- Preview report: `outputs/valentine_topology_seed20260910/network_quality_report.json`.
- Matrix: `outputs/trunk_topology_validation/summary.json`, with every candidate's
  checks retained in the corresponding case directory.
- Fresh-process comparison: `outputs/trunk_topology_reproducibility/reproducibility.json`.
- Automated regressions: `tmp/trunk_regression.log`.

Generated outputs are local artifacts. The committed figure and this dated
record summarize those artifacts; reruns under changed code or dependencies
must create new evidence rather than assuming these results still apply.

## Preview

The top-level procedural seed is 20260910. The first candidate passed all
38 checks without repair. It has:

- 305.23 m of dominant route and
  461.51 m of total graph passage length;
- 2 enclosed rock islands and 3 blind branches;
- 72.8% of projected route extent with one channel;
- 1.92 m minimum tested island gap, above the 1.52 m threshold;
- one connected section footprint and exactly two surviving enclosed holes;
- 463 sampled sections and no sections flagged unstable
  by the existing roof model.

Mean section height is 1.68 m and maximum height
is 3.15 m. These are generated sample statistics,
not observations or a universal terrestrial height limit.

![Accepted centreline and section footprint](figures/readme/trunk_topology.png)

The main passage and local bypass connectivity resemble the requested class of
plan, while its outline remains smoother and simpler than the reference scan.
That visual comparison is qualitative; no calibrated survey outline was used.

## Nine-case campaign

All nine cases passed the configured B+C checks. Host and section seed stayed
fixed. Three seed labels derive network seeds with
`derive_subseed(label, "trunk-topology-validation")`, shared across inlet counts
so the count comparison is controlled. These labels are not top-level procedural
seed overrides. Every case used the same thresholds and a budget of 24 candidate
attempts with up to three repairs each.

| Inlets | Network seed label | Accepted candidate (1-based) | Repairs | Checks | Dominant gallery fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 20260910 | 2 | 0 | 38 | 0.667 |
| 1 | 20260911 | 2 | 0 | 38 | 0.667 |
| 1 | 20260912 | 1 | 0 | 38 | 0.662 |
| 2 | 20260910 | 4 | 0 | 38 | 0.657 |
| 2 | 20260911 | 2 | 0 | 38 | 0.661 |
| 2 | 20260912 | 3 | 0 | 38 | 0.667 |
| 3 | 20260910 | 4 | 1 | 38 | 0.661 |
| 3 | 20260911 | 3 | 0 | 38 | 0.655 |
| 3 | 20260912 | 3 | 0 | 38 | 0.667 |

Rejected proposals were retained in the reports. Rejection reasons included
insufficient space for the requested features, excessive lateral extent,
insufficient island clearance, terminal taper and width-gradient constraints. The first
passing candidate is selected; rejected shapes are never exported as successes.
This small fixed-host campaign does not estimate general acceptance rates across
host fields, all bodies or arbitrary topology settings.

## Reproduction and regression checks

Two separate Python processes, using hash seeds 11 and 37, produced identical
network and section hashes. Their full acceptance reports were byte-identical.
The comparison returned `passed: true`.

The regression selection passed **170 tests and 26 subtests**, covering network
generation, deterministic acceptance, distributed systems, topology controls,
config parsing, section generation, world contracts, mesh continuity, tiled
geometry, checkpoints and network metrics. The 24 topology tests include
adversarial distant bypasses, filled islands, disconnected section fragments,
invalid profiles, lost source lineage, impossible feature density, and repeated
repairs of terminal tapers. Mesh regressions exercise existing fixtures; they
are not full-mesh verification of this new preview.

To repeat the tests:

```bash
uv run --no-sync pytest -q tests/test_network_topology.py tests/test_network_quality.py \
  tests/test_network_systems.py tests/test_network.py tests/test_config.py \
  tests/test_section_field.py tests/test_world_contracts.py \
  tests/test_mesh_continuity.py tests/test_tiled_geometry.py \
  tests/test_checkpoints.py tests/test_evaluation_network_metrics.py
```

The [model and controls](trunk_topology.md) give the campaign commands and explain
what the footprint checks do and do not establish.

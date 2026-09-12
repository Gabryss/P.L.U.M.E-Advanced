# Dominant galleries and local rock islands

For independently growing sources with phase history, use [the integrated gallery mode](independent_gallery_growth.md). This document describes the earlier `generation_mode = "layout"` construction unless stated otherwise.

The Valentine Cave top-down reference motivates a specific procedural topology:
one broad passage, local splits around enclosed rock islands, uneven widths, and
a few short side branches. Long separated corridors are a different scenario.
PLUME now exposes this choice as `network.topology.style = "trunk_dominated"`.
The default remains `general`, preserving the earlier generation modes.

This is a qualitative reference-informed grammar, not a scan reconstruction or
a calibrated model of cave formation. The supplied screenshot has no scale bar.
The [NPS Valentine Cave description](https://www.nps.gov/places/valentine-cave.htm)
lists 498 m of total cave length and describes both large passages and lower
sections. That length does not establish the extent or scale of the screenshot.
The preset's 300 m route is an explicit scenario choice.

## Generate and inspect

```bash
uv run --no-sync python scripts/generate_network_diagnostics.py \
  --config config/earth_valentine_topology.toml \
  --output outputs/valentine_review --density-sweep ""
```

This generates stages A–C: host, network and cross sections. It creates no cave
mesh or loose rocks. `stage_bc_topology_footprint.png` shows the network and the
occupied plan together. The JSON network records each segment's `topology_role`,
`island_id`, source history, topology controls and measured morphology. The
quality report records all attempts, repairs, rejection reasons and seeds.

![Network and passage footprint from the Earth preset](figures/readme/trunk_topology.png)

The top panel shows the network connections. The bottom panel unions the actual
world-space section envelopes; white holes are the enclosed rock islands. Both
panels preserve equal metre scales. Width variation comes from the generated
sections, not a constant-width stroke drawn around the network.

To subsequently build a mesh, pass the same config to `plume-generate`, choosing
a fresh directory. The output argument is a diagnostic PNG path:

```bash
uv run --no-sync plume-generate --config config/earth_valentine_topology.toml \
  --output outputs/valentine_full/stage_b_network.png
```

The preset disables optional events, rocks and texture assets. The existing body
dimension controls and roof screening still apply. The new topology preview and
validation campaign do not establish final mesh continuity or application import
quality; those require inspection of the generated mesh.

## How the grammar works

1. Construct a smooth longitudinal route biased by the host routing cost.
   Named seeded variations affect its lateral position.
2. Allocate separated island intervals, leaving intact gallery between them.
   Replace each interval with two unequal curved arms that share split and rejoin
   nodes. There is no third passage through the island.
3. Place independent blind branches on intact gallery. Their lengths stay in the
   requested range and within a budget that preserves gallery dominance. Widths
   taper once; later repairs and section generation impose taper caps rather
   than repeatedly multiplying the same narrowing.
4. Vary passage widths along the gallery. Solve source contributions and flow
   allocation over the resulting directed graph. With multiple inlets, short
   upstream feeders merge before the island splits. Mixed source identities
   remain traceable on both arms after a split.
5. Run network checks, then construct the cross sections and check their actual
   footprint. Accept the first passing candidate, or try the next deterministic
   candidate. Stop with a rejection report if the configured budget is exhausted.

The width envelope is a morphology control rather than a hydraulic diameter
derived from flux. Split allocation uses width-based capacity weights. Flow at
blind leaves represents the existing model's terminal allocation, so the source
balance check sums **all graph leaves**, not just the main outlet. This is not a
steady fluid-flow simulation.

## Controls

All `_widths` distances below multiply the resolved base passage width
`2 * network.base_passage_radius`. They are dimensionless morphology controls,
not measurements of Valentine Cave. Ordinary TOML values remain in metres.

| Setting in `[network.topology]` | Default | Meaning |
|---|---|---|
| `style` | `"general"` | Set to `"trunk_dominated"` to activate this grammar. |
| `island_count` | `[1, 3]` | Inclusive integer range for enclosed island events. |
| `island_length_widths` | `[4.5, 7.0]` | Longitudinal length range of each split/rejoin interval. |
| `island_half_span_widths` | `[0.85, 1.20]` | Nominal arm excursion; each side also gets independent shape variation. |
| `side_branch_count` | `[1, 3]` | Inclusive integer range for blind branches. |
| `side_branch_length_widths` | `[2.5, 4.5]` | Nominal branch reach; actual curved arclength differs slightly. |
| `lateral_variation_widths` | `0.75` | Lateral variation of the preferred main route. |
| `correlation_length_widths` | `10.0` | Longitudinal scale of that variation. |
| `width_variation` | `0.36` | Amplitude of the Stage-B longitudinal width modulation. |
| `minimum_trunk_fraction` | `0.65` | Dominant gallery route length divided by all gallery passage lengths; upstream feeders are excluded from numerator and denominator. |
| `minimum_single_channel_fraction` | `0.55` | Fraction of projected inlet-to-outlet extent crossed by exactly one centreline, ignoring blind branches. Feeders remain included here. |
| `maximum_bypass_fraction` | `0.22` | Maximum island-arm length divided by projected inlet-to-outlet extent. |
| `maximum_lateral_span_widths` | `5.0` | Maximum centreline lateral span of the whole network, including feeders and branches. |
| `minimum_island_clearance_widths` | `0.20` | Minimum section-envelope gap sampled across the middle 38–62% of each island interval. |

The Earth preset overrides island count to `[2, 2]`, blind-branch count to `[2, 3]`,
branch reach to `[2.5, 4.0]` and the candidate budget to 24. It keeps three repair
passes per candidate. It does not weaken the general bend, grade, crossing,
width, flow or section-profile checks.

`network.systems.count` selects the number of upstream feeders in this mode.
The distributed system spacing, capture/release and persistence controls apply
only to the `general` topology. Lobe, history and density controls are also
inactive here; the diagnostics script skips the unrelated density sweep and
emplacement-history figures. Internal emplacement is required. Extra sources
still require room in the host; accepting counts up to eight in the configuration
does not guarantee that eight will fit a particular compact scenario.

## Acceptance and reproducibility

In addition to the shared network and 3D section checks, this grammar checks
gallery dominance, parallel-route extent, lateral spread, requested feature
counts, valid two-arm island connections, source identities, transported lineage
and total terminal flow allocation. It then checks that the section footprint:

- forms one connected component;
- retains the requested number of enclosed islands;
- leaves a minimum gap between the two arms of each island.

The footprint is a raster union of projected section-envelope strips. Its
baseline spacing is `max(0.08 m, 0.025 × base passage width)`, enlarged if needed
to bound raster area. Holes smaller than `0.08 × base passage width²` are ignored
as sampling-scale defects. This screening targets a single gallery layer; it is
not a volume-intersection, mesh-watertightness or accessible-floor proof.

The first candidate uses the resolved network seed. Subsequent candidates use
the existing named seed derivation, and every repair is deterministic. Fixed
inputs, code and dependency versions are part of the reproduction contract.
The host stays fixed across a candidate search; no wall-clock randomness or
unbounded "try until it looks good" loop is used.

Run the targeted seed/inlet campaign and a fresh-process reproduction check:

```bash
uv run --no-sync python scripts/validate_trunk_topology.py \
  --output outputs/trunk_topology_validation

uv run --no-sync python scripts/validate_network_reproducibility.py \
  --config config/earth_valentine_topology.toml \
  --output outputs/trunk_topology_reproducibility
```

The campaign keeps the host and section seed fixed and varies three named network
seeds across one, two and three inlets. It records failed cases as well as passes;
these are nine network/section cases, not nine full meshed lava tubes. The
reproduction command compares two separate processes with different Python hash
seeds, including profile hashes and byte-identical acceptance reports.

See the [dated validation results](trunk_topology_validation_2026-09-10.md).

# Reproducible network acceptance

PLUME screens candidate networks before allocating the cave volume. A connected,
watertight mesh can still represent a dogleg, an unintended crossing, or a
full-width capped branch. Mesh integrity alone is insufficient.

Acceptance is a **heuristic morphology screen**, not geological certification.
Its thresholds describe the supported procedural style, not universal limits
for Earth, Mars or the Moon. Gravity-dependent roof stability remains separate.

## Selection and repair

1. Generate the host once; keep that input fixed throughout the search.
2. Generate a network with the requested seed and measure its defects.
3. Attempt up to three deterministic repairs, evaluating after each pass.
   Repairs smooth routes in metres while retaining graph nodes, reduce width
   saturation with a bounded discharge response, and taper blind ends. Arc
   lengths, sampled host properties, flow, cooling, ages, junctions and occupancy
   are rebuilt after repairs. No branches are silently deleted.
4. When Stage B passes, generate and check the actual Stage-C sections, including
   final junction blends and elevations.
5. If either stage fails, try the next deterministically derived seed. Accept
   the **first passing** candidate, independent of elapsed time or worker order.
6. On exhaustion, raise `NetworkQualityError` and stop before meshing. Preserve
   the rejection report. Never export a rejected candidate as a fallback.

Feeders join an existing path at first contact instead of crossing it to reach
a distant prescribed junction. Stage C retains the accepted plan route during
junction harmonization; moving stations toward another branch's tangent could
otherwise reverse their order on short branches.

## Checks

| Scope | Checks |
|---|---|
| Graph and flow | Connectivity, source/outlet reachability, conservation and thermal ordering; unique IDs; acyclic flow; endpoint agreement |
| Route shape | Finite positive geometry; host bounds; heading changes; bend radius relative to width; sinuosity; small-scale heading oscillation; long transverse arterial excursions; minimum extent |
| Morphology | Width gradients; length-weighted width-cap saturation; tapered blind ends; near-identical independent loop motifs |
| Intersections | Self-intersections and unmodeled crossings; actual floor/roof clearance; severe nonlocal passage overlap outside junction regions |
| Final sections | Positive finite profiles; orthonormal frames; section-induced bends; width-scale grades; sustained uphill runs; shared-junction floor agreement |
| Interacting systems (when enabled) | Requested source identities, front membership at connections, transported source history, minimum passage persistence, requested merge/split events, total inlet/outlet discharge |

The [interacting-system mode](network_systems.md) adds seven checks to the 25
network/section checks. It also admits multiple outlets and applies loop-motif
screening to its split/rejoin arms.

The report records measured values, thresholds and affected segment IDs.
Repetition detection excludes near-straight fragments and pieces of the same
lobe. It does not compare the network with a surveyed cave population. The
overlap screen uses interpolated section bounds; it is not a triangle-level
collision or final mesh clearance test. Smooth natural branches and multiple
levels are allowed when the checks pass.

`CaveNetworkGenerator.generate(host)` screens Stage B. Pass `section_config=...`
to include Stage C in candidate selection. Both the main CLI and the network
diagnostics script do this. The geometry stage rechecks the supplied network
and sections before volume creation, so modifying sections after selection can
produce a new rejection rather than silently meshing the changed shape.

## Configuration

Screening is enabled by default. These defaults are explicit engineering
heuristics. Width-normalized distances scale with passage size.

```toml
[network.quality]
enabled = true
max_attempts = 8
repair_passes = 3
maximum_turn_degrees = 55.0
minimum_bend_radius_widths = 0.65
maximum_sinuosity = 3.0
maximum_width_cap_fraction = 0.55
maximum_width_gradient = 0.6
maximum_wiggle_degrees = 12.0
terminal_width_ratio = 0.45
maximum_similar_loop_pairs = 2
minimum_route_extent_fraction = 0.35
maximum_section_grade = 0.5
maximum_uphill_run_widths = 8.0
maximum_uphill_grade = 0.08
minimum_crossing_clearance_m = 1.0
minimum_passage_separation_widths = 0.70
maximum_transverse_run_widths = 4.0
minimum_downstream_alignment = 0.25
```

An attempt is one seeded topology. Three repair passes mean up to four
evaluations of that candidate. Limits never relax during search. Explicitly
setting `enabled = false` is available for raw-generator research and controlled
ablations; those outputs have no morphology acceptance claim.

## Reproduction and audit

`network_quality_report.json` records the base seed, candidate seeds, repairs,
checks, accepted shape hash, thresholds, resolved network/section configurations,
host hash, implementation hash and numerical runtime versions. It is updated
during search, including failures.

Candidate zero uses the requested seed. Candidate `i > 0` uses
`derive_subseed(base_seed, "network-quality-v1", i)`. An omitted seed means zero.
No clock, randomized Python hash, global RNG or worker completion order affects
selection. Reproduce a result using the **original base configuration**, same
host, code and numerical dependencies. The normal run manifest also fingerprints
production sources and inputs and invalidates stale checkpoints.

Acceptance changes the output distribution. Scientific evaluations must state
whether they analyze raw proposals or accepted outputs and report rejection
rates and attempt counts. Previous frozen evaluation campaigns are unchanged;
their results do not validate this acceptance policy.

## Review without meshing

```bash
uv run --no-sync python scripts/generate_network_diagnostics.py \
  --config config/earth_full_inspection_seed20260910.toml \
  --output outputs/network_review --density-sweep ''
```

This writes the report, network/section arrays and diagnostic figures. It
generates neither a cave mesh nor rocks.

To test reproduction in two fresh processes, with different Python hash seeds:

```bash
uv run --no-sync python scripts/validate_network_reproducibility.py \
  --config config/earth_full_inspection_seed20260910.toml \
  --output outputs/network_reproducibility
```

The command compares network and section semantic hashes and requires
byte-identical acceptance reports. It exits unsuccessfully if either generation
fails or the results differ. Both runs generate only Stages A-C.

## Known preset limitation

The development-sized Moon case derived from `config/project.toml` failed its
eight-candidate screening trial: its short terminal branches produced excessive
width gradients relative to the much larger lunar passage sizes. This is an
unresolved generation/preset limitation, not a reason to relax the acceptance
thresholds automatically. That scenario now stops before meshing and preserves
its rejection history for diagnosis. Passing an Earth or Mars case does not
establish that all bodies, densities, hosts or seeds will pass.

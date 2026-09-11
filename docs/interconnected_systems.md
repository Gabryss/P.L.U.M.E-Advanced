# Interconnected systems in one host

This mode addresses the earlier multi-source presets' convergence into one dominant
gallery. It is an opt-in procedural model, not a time-resolved lava simulation or a
reconstruction of Valentine Cave. Topology acceptance thresholds are design targets;
they are not measured geological limits.

## Generate and inspect

Use `config/earth_short_interconnected.toml` (400 m downstream extent) or
`config/earth_long_interconnected.toml` (3,000 m). Total passage length includes all
parallel routes and is consequently greater than the downstream extent.

```bash
uv run --no-sync python scripts/generate_network_diagnostics.py \
  --config config/earth_short_interconnected.toml \
  --output outputs/interconnected_short_preview --density-sweep ""

uv run --no-sync python scripts/validate_interconnected.py \
  --config config/earth_long_interconnected.toml \
  --output outputs/interconnected_long_validation --seed-offsets 0 1 2
```

The preview generates the host, graph and cross sections. The validation command
additionally saves resolved settings, acceptance/rejection reports, source lineage,
metric channel-count samples, stage figures and trusted local checkpoints. It does
not mesh the cave or generate rocks. Choose an empty directory: the campaign refuses
to replace an existing result. With the same preset, `plume-generate` uses the common
mesher and exporters after the network/section acceptance gate.

For full inspection meshes, use `config/earth_short_interconnected_full.toml` or
`config/earth_long_interconnected_full.toml`. The short full preset preserves the
primary preview and increases sampling resolution to 8 cm. The long full preset
uses the same host with a 0.9 section width-scale median and a 1.2 m minimum
section-height control, then validates a fresh network at 20 cm resolution. The
resolved mesh radius scale remains 1.0. The long seed is explicit in the config;
its new preflight is separate from the initial six-case preview campaign.

Run `scripts/check_mesh_topology.py` on each completed generation. This independent
check compares graph cycles with the actual raw and exported mesh, requiring one
watertight component, consistent winding, and matching surface genus and significant
plan islands. The normal CLI's network acceptance does not replace this mesh check.
The [output guide](../outputs/README.md) links packaged inspection assets and their
checks. Saved stage figures can be regenerated with
`scripts/render_run_diagnostics.py RUN_DIRECTORY`.

## Stage A: several possible routes, one physical field

The existing 2D host representation is unchanged: all fronts sample the same terrain,
cover, competence, fracture, capacity and routing cost. Generated hosts can include
several broad troughs whose centres vary independently at a low spatial frequency.
Their smooth, bounded union enters both terrain elevation and process-layer creation.
Overlapping troughs do not multiply the prescribed excavation depth. Terrain gradients
and routing penalties are then derived through the existing Stage-A pipeline.

Host corridors do not encode graph edges, merges or splits. Their seed is independent
of network retries. A supplied host is never rewritten to satisfy a topology target;
infeasible routing exhausts the bounded search and stops before meshing. A single
corridor remains the default, preserving existing preset terrain.

| Host control | Meaning |
|---|---|
| `corridor_count` | Number of generated broad troughs, 1–8; integer, default 1 |
| `corridor_width`, `corridor_depth` | Width and maximum depth of the combined trough envelope |
| `corridor_spacing` | Nominal transverse separation of trough centres in metres |
| `corridor_lateral_variation` | Amplitude of seeded centre-offset knots in metres; a cubic interpolant connects them |
| `corridor_correlation_length` | Approximate longitudinal distance between offset knots in metres |

With `apply_body_scaling = true`, corridor spacing and correlation lengths use the
same horizontal scaling as corridor width. The supplied Earth presets use explicit
metric dimensions. Host width limits the space available for the whole network;
individual tube widths still follow the body, material and stability configuration.

## Stage B: persistent front preferences and local interactions

Each source retains a named random stream, long-range lateral preference and a
heading. At downstream stations the router scores a bounded fan of forward
trajectories. It samples cost and elevation at intermediate locations along each
trajectory, penalizes unnecessary turns and rejects trajectories outside the host or
above the permitted signed uphill grade. Source offsets do not decay toward a common
trunk. This is a finite-lookahead routing heuristic, not a global optimal-path solver.

Source preference identities retain their lateral order using a deterministic
projection. The capture/release planner forms one graph passage per active group.
Neighboring groups may merge after independent travel; a shared group may split after
sustained separation of its onward preferences. Capture and release thresholds differ.
Residence and event-spacing rules apply to participating fronts, allowing unrelated
pairs to interact near the same downstream station. Connection proposals also check
direction compatibility, local substrate relief and a minimum prospective supply
fraction. The actual graph's split allocations are checked again after flow assignment.

Preference identities and transported provenance are distinct. After mixing, both
outgoing branches inherit the incoming contributors even though their routing front
IDs differ. A shared passage carries the combined supply once. Conservative graph
flow, weighted thermal state and the per-phase discharge ledger are reused.

The mode also retains finite blind breakouts, phase inactivity, reoccupation and
conditional drained pools. These operate on the grown arterial graph. A side-branch
count is a launch target, not a guaranteed count in this mode. Island counts,
minimum trunk fraction, minimum single-channel fraction and the gallery's maximum
lateral span are inactive. Primary merge/split opportunities scale with route length
through metric correlation and persistence controls; there is no fixed global cap
of eight primary loops.

Routes are smoothed in downstream coordinates with shared junction tangents. The
same constraint is reapplied after repairs, preventing smoothing from introducing
backtracking or sharp joins. Every altered candidate is reassessed.

## Acceptance after routing and cross-section generation

Set `network.topology.style = "interconnected"` and
`network.topology.generation_mode = "independent_growth"`. At least two systems,
internal emplacement, `hybrid_lobe` and zero stacked-lobe fraction are required.

| `network.interconnection` control | Acceptance/routing effect |
|---|---|
| `lookahead_widths` | Forward host-sampling horizon; default 6 passage widths |
| `maximum_junction_angle_degrees` | Direction compatibility for proposed and final connections; default 45° |
| `minimum_parallel_fraction` | Fraction of downstream reach with at least two separated passage envelopes; default 0.40 |
| `minimum_window_parallel_fraction` | Minimum parallel fraction within each downstream window; default 0.20 |
| `minimum_parallel_run_widths` | Minimum longest continuous parallel reach; default 8 widths |
| `maximum_single_run_fraction` | Longest uninterrupted single-channel reach divided by total extent; default 0.40 |
| `minimum_source_independent_length_widths` | Minimum total unshared arterial travel for every source preference; default 5 widths |
| `minimum_clearance_widths` | Rock gap needed for envelopes to count as distinct; default 0.25 widths |
| `interaction_window_m` | Maximum event-free reach and scale for parallel-coverage windows; default 600 m |
| `minimum_interactions_per_km` | Minimum merge/split density over downstream distance; default 2/km |

There are at least two coverage windows, even on short runs. The event-free reach is
also capped at 65% of the total extent. These constraints prevent several initial
feeders followed by one long trunk from satisfying the multi-network target. Their
combination may be infeasible on very short hosts; it does not force a connection.

Widths are multiples of `2 * network.base_passage_radius`. Stage B counts unions of
width envelopes. Stage C measures the actual profiles in world coordinates, so
coincident source labels or an inflated, fused cross section cannot count as separate
passages. Blind side branches are excluded from the parallel-coverage metric.

The existing graph connectivity, source lineage, flow, chronology, bends, width
gradients, crossings, grade, profile-frame, junction-floor and nonlocal-overlap checks
remain active. Additional checks cover signed host uphill grades, actual junction
angles, viable split supply, section-footprint connectivity and agreement between
resolved footprint holes and graph cycle rank. Source membership is not used as a
substitute for physical separation.

Split-supply validation covers both reference discharge and every stored formation
phase; a well-supplied average cannot hide a starved outgoing passage during a phase.

The footprint checks are plan-projection checks for this one-level model. They do not
certify the final triangle mesh or every intervening rock pillar's structural
stability. Final mesh topology, dimensions, clearance and application import still
need their existing validation when a full mesh is generated. Stacking, vertical
capture, deposition feedback and a calibrated thermofluid solver remain unsupported.

## Reproducibility and campaign

Candidate zero uses the resolved network seed. Subsequent attempts use the existing
named deterministic retry sequence. Routing choices, lateral-order projection and
event ordering have stable tie breaking. The first passing candidate is accepted;
exhaustion raises an error, including the rejection history. The same fixed host is
used for every retry.

The campaign offsets only the resolved network seed; host and section seeds remain
fixed. For each supplied scale it checks three offsets and reruns the first in a
fresh interpreter (`PYTHONHASHSEED=11` versus `37`). It compares host/network/section
semantic hashes and the complete acceptance report byte for byte. Every report
records resolved controls, source-code identity and numerical-library versions.
Different seed results must be inspected as an ensemble; one passing preview is not
a claim of geological validation.

The [2026-09-11 campaign](interconnected_validation_2026-09-11.md) records the initial
three-seed tests at each scale, successful fresh-process replays and the regression
scope.

The preview legend colours routing fronts, with shared passages in purple. Circles
are actual merge nodes and diamonds are split nodes. The background is the shared
routing cost; shapes are generated section envelopes. Long previews use consecutive
500 m reaches at equal physical axis scale, plus the full downstream channel-count
trace. They are not artist-drawn target layouts or cave-mesh renders.

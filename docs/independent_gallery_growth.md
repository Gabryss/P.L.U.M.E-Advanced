# Independent systems with a dominant gallery

`config/earth_independent_gallery.toml` combines independently seeded arterial routes with the compact-gallery acceptance rules. It replaces the fixed upstream feeder construction for this use case. The earlier `earth_valentine_topology.toml` and `earth_valentine_multi_inspection_seed20260910.toml` remain available as explicit layout examples.

This is a reproducible procedural formation model. Passing its checks does not establish that a particular cave formed this way, or constitute validation against a measured Valentine Cave survey.

## Generate and inspect

```bash
uv run --no-sync python scripts/generate_network_diagnostics.py \
  --config config/earth_independent_gallery.toml \
  --output outputs/independent_gallery_preview --density-sweep ""

uv run --no-sync plume-generate \
  --config config/earth_independent_gallery.toml \
  --output outputs/independent_gallery/stage_b_network.png
```

The first command produces host, network, sections and diagnostic figures, without a mesh. The second produces the complete cave mesh. This preset requests three sources, a 380 m route, one or two local rock islands and one or two blind branches. It uses Earth basalt settings, a metric host grid, four formation phases and 0.20 m voxels. Events, loose rocks, textures and collision generation are disabled. The dimensions are scenario choices, not dimensions measured from the user's reference image.

## How systems interact

Each source has its own named random stream and correlated lateral routing preferences. Preferences are biased by the host's routing cost and a shared corridor. The common corridor guides the systems without specifying the number or positions of confluences. An order-preserving projection prevents source identities from swapping when their unconstrained preferences cross.

The interaction planner advances all fronts through longitudinal stations. Neighboring fronts can merge after enough independent travel and when their preference centres approach the capture threshold. A merged front can separate only after the release threshold is exceeded for a sustained distance. Capture and release have distinct thresholds, minimum passage lengths and minimum event spacing. A merged front produces one shared graph passage. Source contributions remain recorded downstream of mixing and subsequent splits.

Splits that rejoin into two arms form candidate rock islands. The generator does not insert an island when its independent routes fail to produce one. It rejects that candidate. Complex arrangements that do not survive the gallery and section checks are rejected too.

## Formation history

Arterial passages participate in every configured phase. Finite blind breakouts can grow from shared passages where the sampled host has low capacity or weak confinement. A breakout follows seeded, host-biased steps with directional persistence. Available flux affects its width; exposure cooling, a finite pulse reach and a step limit bound its growth. Short or implausible results can still fail the final morphology checks.

A breakout remains active for its sampled phase span. Reoccupation requires at least one complete inactive phase. The generator retains the inactive gap, not just the first and last active phases. Each phase has its own conserved discharge ledger: dormant branches receive zero, active blind branches draw from upstream supply, and their combined allocation cannot exceed the phase breakout budget. Passage exposure influences Stage-C morphology through `active_phase_count`. Point-level `flux` remains the conserved reference discharge of the preserved graph; it is not the per-phase discharge.

Drained pools are selected on shared arterial passages using the existing host/grade/coalescence selector, weighted by their average phase discharge. They become local widening and floor-profile changes with bounded dimensions and graded transitions. They are distinct from loose rocks and from the old cylindrical chamber artefact.

The phase ledger and `interaction_events`/`phase_events` are exported in `stage_b_network.json`. The history figure displays actual per-phase discharge, including inactive gaps. Discharge is a procedural reference scale, not a calibrated measurement in cubic metres per second. Station-wise front progression and discrete history phases are not a time-resolved fluid simulation.

## Controls used in this mode

| Configuration | Effect |
|---|---|
| `network.topology.style = "trunk_dominated"` | Enables dominant-gallery and rock-island acceptance rules |
| `network.topology.generation_mode = "independent_growth"` | Selects independent routing; the default `layout` retains prescribed island/feeder construction |
| `network.systems.count` | Number of source systems; at least two in this mode |
| `source_spacing_widths` | Initial lateral inlet spacing |
| Systems `lateral_variation_widths`, `correlation_length_widths` | Independent routing variation and its longitudinal scale |
| `merge_distance_widths`, `split_distance_widths` | Capture/release thresholds; release must exceed capture |
| `minimum_shared_length_widths`, `minimum_independent_length_widths` | Required persistence of an arterial front run, including fragments separated by pool/branch nodes |
| `split_confirmation_widths`, `interaction_spacing_widths` | Sustained separation and minimum event spacing |
| Topology `lateral_variation_widths`, `correlation_length_widths` | Common corridor variation |
| `width_variation` | Correlated arterial width variation |
| `island_count`, `side_branch_count` | Acceptance ranges; these do not force a missing connection into a candidate |
| `side_branch_length_widths` | Finite pulse-reach range for blind breakouts |
| `minimum_trunk_fraction`, `minimum_single_channel_fraction`, `maximum_bypass_fraction`, `maximum_lateral_span_widths` | Limits on excessive parallel passage and lateral excursions |
| `minimum_island_clearance_widths` | Required rock separation between opposing section envelopes |
| History `phase_count`, `active_phase_span`, `breakout_probability`, `reoccupation_probability` | Chronological breakout and reuse controls |
| `phase_flux_budget_fraction`, `retirement_flux_threshold` | Combined phase budget for blind branches and minimum viable launch allocation |
| Lobe `branch_flux_fraction`, `minimum_viable_flux_fraction` | Requested breakout allocation and launch threshold |
| Lobe `maximum_steps`, `candidate_temperature`, `inertia_weight` | Bound and steer metric breakout propagation |
| Lobe `breakout_capacity_weight`, `breakout_confinement_weight` | Rank possible breakout sites from sampled host properties |
| `cooling_k_per_m`, lobe `exposed_cooling_multiplier`, `retirement_temperature_k` | Thermal stopping criterion |
| History `drained_pool_*` | Pool occurrence, ranking and dimensions; section stability limits still apply |

Controls ending in `_widths` are multiples of `2 * network.base_passage_radius`. `island_length_widths` and `island_half_span_widths` belong to the older layout construction; independent island geometry emerges from route separation and interaction timing.

## Capabilities and limits

| Capability | Independent gallery mode |
|---|---|
| Host field, body/material defaults, roof stability, adaptive sections, floor atlas, geometric surface relief and mesh export | Uses the common PLUME stages |
| Several independently seeded systems, merges, splits and source lineage | Integrated |
| Chronological activity, finite-budget blind breakouts, cooling retirement and passage reuse | Integrated as described above |
| Drained pools | Integrated; occurrence is conditional on suitable sites |
| Deterministic candidate rejection, repair and retry | Integrated; fails closed when the budget is exhausted |
| Legacy cell-based lobe grammar, braid recipes and deposition feedback | Not executed in this mode; remain part of the general single-system generator |
| Stacked levels and vertical capture | Unsupported here; nonzero `stacked_lobe_fraction` is rejected explicitly |
| Rocks, geological events, texture assets, collision packages | Separate optional stages; disabled in the supplied inspection preset |
| Thermofluid simulation or a physically proven formation history | Not implemented |

The generic configuration contains controls used by alternative generators. Their presence in the resolved configuration does not mean every control participates in this mode. The exported backend provenance lists the supported formation mechanisms and excluded mechanisms.

## Acceptance and reproducibility

The combined gate applies 51 network and section checks. Alongside graph connectivity, flow balance, bends, width changes, crossings and clearance, it checks source identity at each connection, minimum front persistence, surviving rock islands in the section footprint, phase chronology, dormant-phase flow, per-phase discharge conservation, the combined breakout budget, source seeds and interaction evidence. Repairs recompute reference flow, phase ledgers and sections.

Candidate zero uses the resolved network seed. Later candidates use the existing named retry sequence. The first passing candidate is accepted, with every failed attempt and repair recorded. The preset allows 48 candidates and three repairs per candidate. It never silently emits the last failed network. Reproduction requires the same resolved configuration, host, production source and dependency versions; reports record these identities.

The network/section gate does not prove mesh topology. Check the generated mesh separately for components, watertightness, holes and interior clearance. Very shallow tips can be close to the voxel resolution. Blender import and rendered interior checks validate Blender only; equivalent export bytes are not proof of a native Unity or Unreal import.

### Grid-scale fissure repair

The inspection preset enables `geometry.density_closing_voxels = 1`. After surface relief and before roof-stability assessment, a radius-one grayscale closing repairs solid fissures one or two grid cells wide. The default is `0` (disabled); only `0` and `1` are accepted. At this preset's 0.20 m resolution the operation can connect voids separated by up to about 0.40 m. It therefore requires explicit final topology checks and is unsuitable for preserving genuine features that small. It does not delete detached mesh components or change the network.

The initial inspection mesh revealed a separated blind tip and an additional tiny tunnel through the surface. This repair restored the tip connection in the development check while retaining the graph's large island. Dense/tiled equivalence, retention of a resolved rock island, preservation of distant separate cavities, and the disabled no-op are covered by tests.

Use `scripts/check_mesh_topology.py RUN_DIRECTORY` on trusted locally generated checkpoints to compare graph cycles, voxel-plan islands, raw mesh topology and the exported GLB after welding duplicate seam positions. The check writes `mesh_topology_check.json` and fails if the mesh does not have the expected connected, watertight topology. Use `scripts/check_tube_sections.py` and actual application inspection for additional clearance checks.

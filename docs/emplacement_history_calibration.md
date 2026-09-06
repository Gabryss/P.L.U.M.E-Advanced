# Emplacement-history calibration for Stage B (GEO-005)

This note defines scientific constraints for the next emplacement-history and
QA revisions.  It is a calibration specification, not a lava-flow solver.
Quantities marked **EMPIRICAL** are reported observations, **INFERRED** are
process interpretations supported by observations, and **HEURISTIC** are
transparent generator/QA choices.  No number below is a universal constant.

## Evidence that can constrain Stage B

| Source | Primary observation | Constraint relevant to PLUME |
|---|---|---|
| Kauahikaua et al. (1998), Kīlauea, [doi:10.1029/97JB03576](https://doi.org/10.1029/97JB03576) | Active tubes were observed through multiple phases; tube reoccupation after a pause and persistent tube activity were documented. Tube transport and morphology varied strongly with emplacement style and discharge. | **EMPIRICAL/INFERRED:** preserve episode identity and allow later pulses to reuse an older route; do not assume every pulse creates a new branch. |
| Kauahikaua et al. (1998), Hualālai synthesis in the open paper [PDF](https://pages.uoregon.edu/cashman/pdfs/Kauahikauaetal.pdf) | At least four of six ca. 1800 flows issued from pre-existing cones/tubes; high-rate examples were about 100 m³/s, while a low-rate hummocky flow was <5 m³/s and likely lasted months. Reoccupation styles ranged from vigorous fountaining to passive filling. | **EMPIRICAL:** reoccupation is real but site-specific. Use a configurable reoccupation opportunity, conditioned on pulse history and source state; never encode 4/6 as a global prior. |
| Biass et al. (2019), Kīlauea episode 61g, [doi:10.1029/2019JB017444](https://doi.org/10.1029/2019JB017444) | A 3-month time series grew ~12 × 10⁶ m³ at a near-constant 1.2–2.7 m³/s; an initial tube inflated and later modulated a second lobe/tube system. Episode 61f stalled about 2 km from source while 61g captured the supply. | **EMPIRICAL/INFERRED:** episodes compete; a surviving route can capture supply while a nearby branch stalls. Pulse persistence must be represented separately from geometric branch count. |
| Dietterich et al. (2014), Hawaiian channel networks, [doi:10.1002/2014JF003103](https://doi.org/10.1002/2014JF003103) | Breakouts/overflows arise from levee failure, blockages, or supply pulses; flow-front bifurcations record obstacle interactions. Flux divides at splits and sums at confluences; branches become slower/narrower after splitting, while confluences have the opposite tendency. Steeper slopes yield faster, thinner, narrower flows; cooling away from source tends to slow, thicken, and widen flow. | **INFERRED:** branch competition and survival must be flux- and slope-conditioned; branch creation cannot be independent random decoration. Preserve distributary versus tributary directionality. |
| Calvari & Pinkerton (1998), Etna, [doi:10.1029/97JB03388](https://doi.org/10.1029/97JB03388) | Inflated flows, ephemeral vents, secondary tubes, and tube-fed distal extension built a complex flow field. | **INFERRED:** an emplacement pulse may change the active route and spawn a secondary route; event order matters. |
| Calvari & Pinkerton (1999), Etna, [doi:10.1016/S0377-0273(99)00024-4](https://doi.org/10.1016/S0377-0273%2899%2900024-4) | Tubes form at successive levels; inflation and coalescence produce vertical/horizontal capture. Capture can cause abrupt level or flow-direction changes. | **EMPIRICAL/INFERRED:** distinguish grade-separated crossings from true captures; a crossing must not silently become a merge. |
| Calvari et al. (2024), Etna 1792–93, [doi:10.3389/feart.2024.1448187](https://doi.org/10.3389/feart.2024.1448187) | A mapped >4° field contains stacked, overlapping tube systems, captures, branches, benches, levees, falls, and tubes at different distances from the fissure. Declining discharge produced progressively proximal breakouts; the field contains at least three overlapping cavities in its proximal systems. | **EMPIRICAL:** phase, slope, and distance from source must be retained as stratification variables. Features are not evidence for a universal level count or breakout rate. |

The current Stage-B implementation already exposes phase count and active span,
birth/death phase, lobe path IDs, branch flux fractions, retired paths,
breakout trigger/score, parent and child flux, coalescence return, stacked
levels, capture/chamber flags, roof state, crossing groups, and anastomosis
counts.  The current configuration includes `phase_count=[3,5]`,
`active_phase_span=[1,3]`, `stacked_lobe_fraction=0.34`,
`retired_path_fraction=0.24`, `minimum_viable_flux_fraction=0.075`, and
`coalescence_flux_return_fraction=0.72`.  Treat these as tunable defaults, not
measurements.

## Actionable emplacement constraints

### Pulse persistence and tube reoccupation

1. **EMPIRICAL/INFERRED:** each arterial route has an episode identity and
   `birth_phase`, `death_phase`, and `formation_state`; a later pulse can choose
   `reoccupy`, `inflate_existing`, `branch_from_existing`, or `new_route`.
   Reoccupation must reference an existing segment family rather than creating
   a coincident duplicate edge.
2. **INFERRED:** persistent routes should be selected by accumulated support
   (integrated flux × active duration) and thermal/host suitability. A route
   receiving a later pulse may widen or change surface state without changing
   its graph identity.
3. **HEURISTIC QA:** require at least one route spanning all phases in a
   long-lived baseline world; require at least one nonpersistent pulse in a
   competition stress world. Report the phase-weighted fraction of segments
   reoccupied, newly formed, and abandoned. Do not force a fixed fraction across
   body presets.
4. **Caution:** Hualālai's 4/6 reoccupation count is a case study, not a global
   estimate. Keep `stacked_lobe_fraction`, pulse count, and reoccupation
   probability independently configurable.

### Abandonment, competition, and distributary hierarchy

1. **EMPIRICAL/INFERRED:** a split divides parent discharge; a daughter with
   insufficient flux, high exposed cooling, or poor downstream potential may
   stall/abandon. A surviving daughter may capture most of a later pulse (as in
   episode 61g versus 61f). Record the cause, not just `stalled_lobe`.
2. **INFERRED:** preserve directed hierarchy: parent order 0/arterial, then
   branch order +1 at each distributary split. At every split, child fluxes must
   sum to the parent flux up to explicitly recorded loss/retirement; at every
   coalescence, incoming fluxes must sum before any configured deposition loss.
3. **HEURISTIC QA:** over a seed ensemble, median child/parent flux and
   persistence should decrease with distributary order; test the sign and report
   confidence intervals rather than enforcing a fixed slope. Flag a world if
   >20% of split children gain both flux and width without a labelled capture or
   chamber.
4. **INFERRED:** `retired_path_fraction` controls opportunity, while
   `minimum_viable_flux_fraction`, cooling, and obstruction determine survival.
   A constant random death independent of flux is not an acceptable physical
   explanation.

### Breakout, overflow, obstacle bypass, and anastomosis

1. **EMPIRICAL:** branch creation is associated with flow-front bifurcation,
   levee failure/overflow, blockages, supply pulses, or obstacle interaction.
   `breakout_capacity_weight`, `breakout_confinement_weight`,
   `breakout_curvature_weight`, and `breakout_blockage_weight` should remain
   separately logged so QA can identify which mechanism is active.
2. **INFERRED:** a breakout removes finite flux from the parent and modifies the
   emplacement surface used by later fronts. It must carry `breakout_trigger`,
   `parent_flux_before_split`, `parent_flux_after_split`, and an initial branch
   flux. A branch that later coalesces may return only its surviving flux.
3. **INFERRED:** anastomosis/coalescence requires a spatially plausible approach
   and a compatible level. Do not merge paths merely because raster cells touch
   after smoothing; preserve `crossing_group_id` for grade-separated paths.
4. **HEURISTIC QA:** for each event family, report event rate per 100 m of
   arterial route, trigger-score distribution, and median parent/child flux.
   Compare seeds and phase strata; no universal event rate is currently
   defensible.

### Grade-separated crossings versus true coalescence

Use two separate graph relations:

* **Crossing:** plan-view proximity or intersection with distinct z-levels,
  no shared walkable volume, and no flux node. Keep `crossing_group_id`, level,
  vertical separation, and minimum clearance. Geometry must render two passages.
* **True coalescence/capture:** a documented horizontal or vertical connection
  with a shared junction volume, compatible levels, and a flux-conservation node.
  Record whether it is lateral coalescence, vertical capture, or chamber-forming
  coalescence.

**MUST-HAVE QA:** zero unlabelled crossings; zero flux merges at a crossing;
zero disconnected capture metadata; and no chamber enlargement for a crossing
alone. Calvari & Pinkerton's successive-level and capture observations support
the distinction, but do not specify a universal clearance or blend length.

### Discharge, width, slope, branch order, and survival

The following are directional constraints, not universal equations:

* **EMPIRICAL/INFERRED:** higher flux generally supports longer/faster routes;
  after a split, daughter flux and width should tend to be lower, while a
  confluence tends to increase both. Use signed Spearman correlations and
  partial correlations by phase and slope stratum.
* **EMPIRICAL/INFERRED:** steeper slopes tend to produce faster, thinner,
  narrower flows; cooling and crystallization away from source tend to slow,
  thicken, and widen them. Use grade and downstream distance as covariates;
  never apply a single width-vs-flux power law to all worlds.
* **INFERRED:** survival length should increase with integrated flux and active
  duration, and decrease with exposed cooling or blockage severity, all else
  equal. This is a ranking test, not a calibrated hazard probability.
* **HEURISTIC QA bands:** in ≥20 seeds, flag (not automatically fail) if fewer
  than 60% of worlds show positive width–flux association after phase/grade
  stratification, fewer than 60% show negative width–grade association, or
  branch-order medians do not decline in either flux or persistence. The metric
  report must include effect sizes and bootstrap intervals.

## Mapping to current configuration and diagnostics

| Scientific concept | Current variables / metadata | Required diagnostic |
|---|---|---|
| Pulse episodes | `emplacement_history.phase_count`, `active_phase_span`; `birth_phase`, `death_phase`, `emplacement_phase_count` | Phase occupancy, phase-weighted active length, route survival by phase |
| Reoccupation | `formation_state`, lobe/path IDs, `channel_reuse_weight`, `deposition_feedback_m` | Reoccupied/new-route/abandoned counts and reused length fraction |
| Competition and abandonment | `retired_path_fraction`, `minimum_viable_flux_fraction`, `retirement_temperature_k`, `exposed_cooling_multiplier` | Survival curves versus initial/integrated flux, cooling, slope, and blockage |
| Discharge hierarchy | `source_flux`, `branch_flux_fraction`, `parent_flux_before_split`, `parent_flux_after_split`, `coalescence_returned_flux` | Flux-conservation residual, child/parent ratios by order and phase |
| Breakout/overflow | Four `breakout_*_weight` controls; `breakout_trigger`, `breakout_score`, `breakout_event_count` | Trigger-stratified event rate and parent/child flux/length distributions |
| Obstacle bypass/anastomosis | host slope/curvature/confinement/blockage fields; `anastomosis_count`, `loop_probability`, `channel_avoidance_weight`, `channel_reuse_weight` | Distance to obstacle/curvature peak at branch birth; coalescence approach and level |
| Stacked levels/capture | `stacked_lobe_fraction`, `maximum_absolute_level`, `capture_probability`, `vertical_capture_chamber_probability`; `z_level`, `vertical_capture`, `crossing_group_id` | Crossing versus capture confusion matrix; vertical separation and capture length |
| Chambers and roof | `chamber_formation_probability`, `chamber_gain`, `roof_failure_probability`, `roof_state` | Chamber selection by flux quantile and event cause; roof-state by cover/span |
| Geometry response | segment width/length and `slope_degrees`, `flux`, `temperature_k`, `age_s` | Width/flux/grade/order correlations; no unlabelled discontinuities |

## Acceptance table for NET-005 and QA-005

| Class | Must-have physical constraint | Tunable artistic control (must remain explicit) |
|---|---|---|
| Phase history | Phase IDs, birth/death, route reoccupation versus new route; at least one persistent route in long-lived worlds | `phase_count`, active-span distribution, phase timing/seed phase offsets |
| Flux and hierarchy | Splits divide finite flux; coalescences sum it; child order has inspectable flux/persistence trend | Branch flux range, source count, launch rate, branch abundance |
| Survival | Abandonment is explainable by flux, cooling, blockage, or host cost; stalled paths remain labelled | Retirement fraction, temperature threshold, minimum viable fraction |
| Breakouts | Breakouts are triggered by capacity/confinement/curvature/blockage/supply context and alter later emplacement | Relative trigger weights, deposition feedback, breakout density |
| Crossings/captures | Grade-separated crossings are not merges; true captures have connection, level, and flux metadata | Capture probability, level span, chamber/capture gain |
| Slope/width response | Report stratified width–flux and width–grade trends; do not claim a universal law | Morphology width gain, base passage scales, longitudinal modulation |
| Validation | ≥20-seed ensemble, deterministic manifests, effect sizes and uncertainty; PDC confirmatory partition untouched | Seed list, bootstrap count, diagnostic presentation choices |

## Scientific cautions

* Observations mix pāhoehoe and ‘a‘ā, slope regimes, durations, and survey
  scales.  Use slope/body/flow-regime strata; do not pool them into one
  “natural” branch rate.
* Reoccupation evidence is strong qualitatively but sparse quantitatively.
  The Hualālai 4/6 example and Biass et al.'s two-system time series should
  motivate configurable histories, not calibrate a universal probability.
* Network observations are usually surface channels or exposed tubes.  A
  generated subsurface route can be process-plausible without being directly
  validated by those observations.  PDC cross-sections remain irrelevant to
  this phase-history calibration.
* All acceptance bands above are diagnostics for detecting implausible failure
  modes.  They must not be tuned against the frozen 19-cave confirmatory
  partition or presented as geological constants.


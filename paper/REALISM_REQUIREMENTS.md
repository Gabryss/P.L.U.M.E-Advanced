# Lava-tube realism requirements (GEO-001)

Status: implementation contract for the next Network, Passage Morphology,
Geometry, and QA revisions.  This is a scientific specification, not a claim
that PLUME is a thermofluid or structural-physics simulator.

## 1. Evidence and guardrails

The repository's current diagnostics expose the following Stage-B quantities:
node/edge and component counts, source-to-exit reachability, split/merge
counts, cyclomatic number, sinuosity, branch persistence, vertical levels and
captures, roof-state histogram, chamber count, host exposure, flux residuals,
and temperature/age monotonicity.  Stage C exposes per-sample width, height,
area, compactness, solidity, centroid offsets, floor residual, roof asymmetry,
and the closed local contour.  These are the quantities to report; do not add
an unlogged visual score.

The frozen development evidence is in `paper/REALISM_CALIBRATION.md`: 1,286
valid calibration contours from 76 caves (72 self-intersecting contours were
excluded by a predeclared rule), and the matched three-seed pilot improved the
aggregate selected-metric distance from 0.911 to 0.477.  The corresponding
aspect-ratio, compactness, floor-residual, and roof-asymmetry distances were
0.560, 0.608, 0.462, and 0.278.  These are calibration evidence only.  The
19-cave confirmatory partition is locked and must be evaluated once, after
 freezing code and configuration; no value from it may guide tuning.  No final
 confirmatory metrics are present in the base-commit tree, so this contract
 does not report a fabricated evaluation result.

### Source / claim / metric traceability

| ID | Primary source or authoritative dataset | Claim used here | Observable metric or test |
|---|---|---|---|
| S1 | Kauahikaua et al. (1998), *JGR* 103, 27303–27323, [doi:10.1029/97JB03576](https://doi.org/10.1029/97JB03576) | Kīlauea tubes range from 3–25 m wide and up to 20 m high proximally; typical tubes are about 5 m high × 3 m wide, with 1–2 m active stream depth; tubes may downcut (10 cm/day measured at one skylight). | Width/height distributions, stream-clearance proxy, longitudinal floor-elevation change, and grade-conditioned downcutting events. |
| S2 | Peterson et al. (1994), *Bull. Volcanol.* 56, 343–360, [doi:10.1007/BF00326461](https://doi.org/10.1007/BF00326461) | Sustained emplacement and crust growth produce tubes; formation history matters for dimensions and persistence. | Segment birth/retirement, age and temperature monotonicity, and persistence-length distributions. |
| S3 | Calvari & Pinkerton (1998), *JGR* 103, 27291–27301, [doi:10.1029/97JB03388](https://doi.org/10.1029/97JB03388) | Etna flow fields grow through inflated flows, ephemeral vents, secondary tubes, and tube-fed distal extension. | Event chronology, source/vent count, lobe duty cycle, branch birth distance, and connected downstream reach. |
| S4 | Calvari & Pinkerton (1999), *JVGR* 90, 263–280, [doi:10.1016/S0377-0273(99)00024-4](https://doi.org/10.1016/S0377-0273%2899%2900024-4) | Tube networks can occur at successive levels; inflation and coalescence produce vertical and horizontal captures and complex branching. | Distinct z-level count, capture count, overlap in plan, and split/merge junction taxonomy. |
| S5 | Allred & Allred (1997), “Development and Morphology of Kazumura Cave,” *J. Cave and Karst Studies* 59, [survey PDF](https://caves.org/wp-content/uploads/Publications/JCKS/v59/cave_59-02-fullr.pdf) | Falls form on moderate slopes (about 1.6–6.2°), with backcutting, plunge-pool widening, stepped floors, and a high/wide chamber below the fall; stacked passages can occur below a fall. | Grade-conditioned fall placement; local width/height gain; floor step, chamber, and downstream level-change metrics. |
| S6 | Cooper & Kauahikaua (1992), USGS OFR 92-352, [doi:10.3133/ofr92352](https://doi.org/10.3133/ofr92352) | Extinct Hawaiian tubes preserve morphology useful for inferring tube evolution, while collapse and later modification alter the original conduit. | Roof-state and breakdown-event histograms; compare intact versus modified segments without treating either as a universal template. |
| S7 | Calvari et al. (2024), *Frontiers in Earth Science* 12, 1448187, [doi:10.3389/feart.2024.1448187](https://doi.org/10.3389/feart.2024.1448187) | A mapped Etna field contains tubes at different distances and levels, with keyholes, captures, benches, levees, coatings, falls, and skylights; slope and eruption history change the network. | Per-feature provenance, level/capture counts, lining/bench/fall occurrence, and stratified (not pooled) slope distributions. |
| S8 | Romio et al. (2025), **Pyroduct Digital Catalog v2.0**, [Zenodo record](https://zenodo.org/records/17750755), [doi:10.5281/zenodo.14535885](https://doi.org/10.5281/zenodo.14535885) | >1,200 digitized/manual cross-sections from >94 terrestrial tubes; TXT sections are ordered approximately entrance-to-end. Catalog cells are 35 m × 35 m and source scales/measurement methods differ. | Calibration-only contour ECDFs and normalized Wasserstein distances for width, aspect, compactness, floor residual, and roof asymmetry; cave-level split, not section-level random split. |

Tags in the requirements below mean **EMPIRICAL** (directly observed in a
source), **INFERRED** (process interpretation supported by one or more
sources), or **HEURISTIC** (a transparent generator/QA choice that is not a
population estimate).

### Implementation hand-off matrix

| Area | Required implementation responsibility | Minimum QA artifact |
|---|---|---|
| Network | Generate process-labelled branches, loops, splits/merges, levels, captures, grades, and emplacement phases; expose density controls independently. | Seed-sweep table with topology, sinuosity, grade, levels, reachability, and conservation metrics. |
| Passage Morphology | Map flow/age/host state to smoothly varying, non-ellipse-only sections; create finite junction blends and event-conditioned floors/linings/benches. | Per-section descriptor table plus longitudinal first-difference and event-provenance checks. |
| Geometry | Preserve section contours through density stamping and meshing; keep underpasses separate, connect captures, and realize chambers/falls/breakdown as topology-changing volumes. | Native-vs-voxel descriptor comparison, manifold/watertight checks, and floor-contact report. |
| QA / Evaluation | Enforce frozen split, calibration-only tuning, cave-stratified uncertainty, and explicit failure labels; never substitute a visual score for a metric. | Reproducible manifest containing commit, config, seeds, partition, thresholds, and all pass/flag results. |

## 2. Per-world requirements (Network and QA)

The world-level report must include all metrics in Section 1, seed, world
preset, network density, and the level of every segment.  A run is invalid if
any of the following fail:

* **EMPIRICAL/INFERRED:** one connected component for the preserved cave
  system, every entry reaches an exit (unless explicitly labelled an
  abandoned/stalled lobe), and all nonterminal junctions have both incoming and
  outgoing segments.  Report disconnected and source-unreachable counts; do
  not silently repair them in QA.
* **EMPIRICAL (state conservation):** maximum relative split/merge flux
  residual ≤1e-6, zero zero-flux segments, no temperature increase along a
  segment, and no lava-age decrease.  These are consistency constraints, not
  geological calibration targets.
* **INFERRED:** topology must be process-labelled.  Every branch, anastomosis,
  capture, chamber, underpass, retirement, and roof state carries an event or
  emplacement phase; a generic high-flux blob is not a junction explanation.
* **HEURISTIC (density control):** `network_density=0` is a single backbone
  endmember; density 1 is the calibrated Earth-like baseline; density 3 is a
  deliberately complex endmember.  The mapping is monotonic in opportunities,
  not a promise of a natural prevalence.  Across ≥20 seeds, report medians and
  10–90% intervals for branch ratio, cyclomatic number, stacked-segment share,
  and junction count normalized by dominant-route length.  Keep anchor spacing,
  lobe launch rate, loop probability, capture probability, and chamber gain
  separately configurable so density 3 cannot hide one overloaded knob.
* **HEURISTIC (recommended envelope):** at density 1, target branch ratio
  0.05–0.35 and cyclomatic number 0–3 per 100 m of dominant route; at density
  3, target 0.15–0.65 and 1–8 per 100 m.  Treat these as QA envelopes for
  detecting a straight or over-braided failure, not as empirical frequencies.
  A preset may document a different envelope with a new calibration record.
* **INFERRED:** report host-exposure distributions (slope, cover, fracture,
  capacity, stability) and compare them with the host-field distribution.  A
  cave should not preferentially occupy the highest routing penalty decile
  unless the preset explicitly models a breakout or capture.

## 3. Per-segment requirements (Network, Passage Morphology, Geometry)

Metrics are computed on the centerline between consecutive samples, with
segment length used as the weight.  Source scales range from active-flow
observations to cave surveys, so the bands below are intentionally broad.

* **EMPIRICAL/INFERRED (grade and sinuosity):** retain downhill potential but
  permit local reversals at bends, captures, and falls.  Report grade in degrees
  and sinuosity `S = arc_length / endpoint_chord`; never round `S` to one.  Over
  ≥20 seeds, require at least 10% of nonterminal segments with `S ≥ 1.10` and
  no more than 30% with `S < 1.02` at density 1 (HEURISTIC anti-straightness
  envelope).  Stratify grade by preset; do not apply Etna's 10–20° proximal
  slopes to Hawaiian or planetary worlds.
* **INFERRED (longitudinal evolution):** width and height must vary smoothly
  within a segment and may taper or inflate with emplacement age/flux.  Report
  coefficient of variation and downstream Spearman trend for width, height,
  floor elevation, and clearance.  As a QA envelope, CV 0.10–0.80 and a
  nonzero trend in at least one size variable on ≥60% of long segments are
  acceptable; a new preset may justify another range.
* **EMPIRICAL (cross-section scale anchor):** use S1's observed 3–25 m widths,
  up-to-20 m heights, and 1–2 m active-stream depth as a plausibility check for
  Earth-like worlds.  Do not clamp every generated segment to these values;
  distal, small, inflated, and planetary cases are intentionally configurable.
* **INFERRED/HEURISTIC (splits and confluences):** transition morphology over
  a finite distance of 1–3 local passage diameters.  At a junction report
  incident widths, area ratio, angle, blend length, floor continuity, and roof
  clearance.  Target daughter-area sum / parent-area in 0.7–1.3 and junction
  maximum width / median incident width in 1.0–2.5.  Values outside are allowed
  only for a labelled chamber or collapse.  Grade-separated underpasses must
  not be filleted into a single volume.
* **EMPIRICAL/INFERRED (levels and captures):** assign upper/lower level and
  capture event explicitly.  For a capture, require a vertical transition,
  overlap in plan view, and a clearance-preserving connection; a z jump without
  a connecting event is invalid.  Report distinct levels, vertical separation,
  overlap count, and capture length.
* **EMPIRICAL (falls):** if a fall is generated, condition its candidate grade
  on the 1.6–6.2° survey envelope in S5, then apply a local step and a widened
  plunge-pool/chamber below.  Falls may occur outside this envelope only when a
  preset documents a different gravity/slope regime.  Never scatter falls
  independently of grade and longitudinal floor evolution.
* **INFERRED (breakdown and roof):** roof state (`intact_tube`,
  `partial_roof`, `skylight_prone`, `open_channel`) must correlate with cover,
  span, competence, and event history.  Breakdown changes collision topology
  and is not merely a texture decal; report event location, removed roof span,
  and resulting skylight/open interval.

## 4. Per-section requirements (Passage Morphology, Geometry, QA)

* **EMPIRICAL (PDC comparison):** sections are closed, finite, simple polygons;
  reject self-intersections using the predeclared contour rule.  Compare
  generated and PDC calibration ECDFs for width, height, aspect ratio, area,
  compactness, solidity, centroid offsets, floor residual, and roof asymmetry.
  Use cave-level bootstrap confidence intervals and normalized Wasserstein
  distances; never mix sections from the 19 evaluation caves into parameter
  selection.
* **HEURISTIC (interim QA bands):** until a clean confirmatory run exists,
  flag (not automatically fail) aggregate calibration distances >0.60,
  aspect >0.75, compactness >0.75, floor residual >0.80, or roof asymmetry
  >0.40.  These thresholds are deliberately looser than the pilot values
  (0.477, 0.560, 0.608, 0.462, 0.278) to avoid tuning to three seeds.
  Final pass/fail thresholds must be declared before evaluation and reported
  with uncertainty.
* **EMPIRICAL/INFERRED:** avoid an ellipse-only ensemble.  Include asymmetric,
  keyhole/inverted-keyhole, arched, triangular, rectangular, and elongated
  endmembers when process controls call for them (S1, S4, S7).  Report the
  fraction of sections whose compactness, solidity, floor residual, or roof
  asymmetry lies outside the matched-ellipse baseline; this is a diversity
  diagnostic, not a target fraction.
* **INFERRED (longitudinal continuity):** width, height, centroid, floor relief,
  and wall relief are continuous at sample joins and junction transitions.
  QA computes normalized first differences and flags jumps >0.25 local diameter
  per sample, except at a labelled fall, capture, collapse, or chamber edge.
* **EMPIRICAL/INFERRED (floors, linings, benches, levees):** floor relief must
  be nonuniform but connected; optional linings, benches/shelves, lateral
  levees, drips, and tube-in-tube false floors are event-driven and carry
  provenance.  Their occurrence and thickness remain configurable because
  PDC contours generally omit material stratigraphy and fine lining detail.
  A bench or lining must not seal the walkable aperture unless the event is
  explicitly a choke/infill.
* **HEURISTIC (resolution):** evaluate section descriptors at native contour
  resolution and at the geometry voxel resolution.  A feature smaller than two
  voxels is reported as unresolved rather than invented; mesh QA separately
  checks manifoldness, watertightness, and floor contact.

## 5. Priority order

**P0 (must land before the next evaluation):** process-labelled junctions and
finite blend lengths; non-straight but grade-aware centerlines; explicit level /
capture metadata; width/height/floor longitudinal variation; section ECDF QA
using calibration only; density-3 stress report; and hard state-consistency /
mesh-connectivity checks.

**P1 (next scientific increment):** grade-conditioned lava falls with plunge
pools; event-driven benches/levees/linings; breakdown linked to roof state and
cover; stratified topology summaries by emplacement phase; and uncertainty
intervals over ≥20 seeds.

**Deferred:** thermofluid inversion, structural collapse probability, material
composition/mineralogy, planetary validation from terrestrial contours,
fine-scale stalactite/stalagmite populations, and claims about natural event
frequencies.  These require data and physics not present in the current
generator or PDC.

## 6. Scope limitations and non-tuning statement

PDC is a terrestrial cross-section catalog, not a planform, longitudinal-profile,
junction, level, or breakdown dataset.  Agreement with PDC can support section
morphometry only; it cannot validate network topology, sinuosity, captures,
lava-fall placement, chamber geometry, or mesh continuity.  Catalog source
surveys, digitization methods, and physical scales differ, so all pooled values
must be accompanied by cave-stratified uncertainty and no false precision.

The 76/19 cave split in `paper/splits/` is frozen.  Calibration code may read
only `pdc_calibration_caves.txt`; the 19 confirmatory caves are read once from a
clean, frozen commit.  Any new threshold, preset, or density-3 envelope is a
new declared experiment, never a silent change to the evaluation partition.

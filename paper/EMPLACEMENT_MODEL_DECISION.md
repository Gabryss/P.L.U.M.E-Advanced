# Proposal-generator decision: DOWNFLOW versus MrLavaLoba/Flowy

Status: SCI-002 scientific decision record.  This compares external
**surface-lava emplacement** models as proposal generators for PLUME's buried
tube network.  Neither model is a lava-tube simulator and neither is allowed
to replace PLUME's process-labelled network, section, or geometry stages.

## Decision

Use a DOWNFLOW-style ensemble as the default *planform route prior* when a
lightweight, terrain-conditioned proposal is needed.  It supplies multiple
steepest-descent alternatives and an explicit topographic uncertainty scale,
but its paths are alternatives, not simultaneous tubes.  Use Flowy v1.0.0 as
an optional *footprint/thickness prior* when a case has a calibrated eruption
volume and observed surface-flow outline.  Flowy is preferable to the original
MrLavaLoba executable for an external experiment because it is a maintained
C++20 reimplementation with a pinned release and substantially lower runtime.

The proposal adapter must therefore:

1. import only planform, elevation/grade, corridor-occupancy, and uncertainty
   evidence;
2. preserve the external model name, revision, inputs, and seed in provenance;
3. convert proposals into candidate centreline edges before PLUME assigns
   flux, cooling, tube sections, levels, captures, chambers, or roof state; and
4. keep the adapter optional and process-separated.  Do not link either model
   into the PLUME library or vendor its source without a license review.

This is a conditional recommendation, not a claim that DOWNFLOW or Flowy
recovers buried topology.  PLUME's current hybrid-lobe model remains the
authoritative generator when no external proposal is requested.

## What the source models actually compute

### DOWNFLOW

Favalli et al. define DOWNFLOW as a stochastic hazard model.  Given a DEM and
vent (or vent ensemble), it follows the maximum-slope path, repeatedly
perturbing every elevation cell within `±Δh`.  `Δh` is tied to DEM vertical
accuracy or a characteristic lava-flow vertical scale.  The output is an
ensemble of possible flow paths / inundation areas.  The paper explicitly says
that DOWNFLOW does not solve thermal or rheological transport equations and is
for potential flow paths, not temporal emplacement ([Favalli et al., 2005, doi:10.1029/2004GL021718](https://doi.org/10.1029/2004GL021718)).

Legitimate PLUME inputs are the path polylines, sampled surface elevation and
grade, path occupancy frequency, and the perturbation/DEM uncertainty.  The
adapter may use these to bias candidate centreline direction, route length,
source reachability, and host-field exposure.  A single path must never be
labelled “the tube”.

### MrLavaLoba and Flowy

MrLavaLoba places a sequence of elliptical lava parcels/lobes on a DEM.  New
lobes bud from existing lobes with a probabilistic law influenced by local
steepest slope and user parameters.  The published model is explicitly intended
to estimate likely inundated area and final deposit thickness, not progression
through time ([de' Michieli Vitturi & Tarquini, 2018, doi:10.1016/j.jvolgeores.2017.11.016](https://doi.org/10.1016/j.jvolgeores.2017.11.016)).

Flowy is a C++20 reimplementation of MrLavaLoba ([official repository](https://github.com/flowy-code/flowy)); its v1.0.0 release is pinned to commit
`6beb3ba` ([release](https://github.com/flowy-code/flowy/releases/tag/v1.0.0)).
Official examples report approximately 50–100× lower runtime than the
reference implementation, while retaining the probabilistic method and
outputs.  Flowy/MrLavaLoba outputs that PLUME may consume are the final or
masked thickness raster, hazard/path-frequency raster, lobe footprint, and
source/volume metadata.  These are surface-flow priors, not tube radii or
cross-sections.

## Scientific legitimacy and non-inferences

| Quantity from an external run | Allowed PLUME use | Explicitly *not* inferred |
|---|---|---|
| DEM, local slope, curvature, flow direction | Host routing prior; candidate edge tangent; grade strata | Lava rheology, cooling, or subsurface conduit geometry |
| DOWNFLOW path ensemble and occupancy frequency | Route alternatives, uncertainty corridor, source-to-exit reachability | A simultaneous branching network, flux, tube age, or branch persistence |
| Flowy/MrLavaLoba thickness / masked thickness | Low-frequency emplacement prior; relative branch supply or chamber opportunity after calibration | Tube diameter, roof span, active-stream depth, or conservation of lava flux |
| Flowy/MrLavaLoba lobe footprint and lobe count | Candidate segment density and planform complexity prior | Number of preserved tubes, junction degree, or multilevel topology |
| Hazard/path-frequency raster | Candidate corridor ranking and uncertainty visualization | Probability of a buried tube or collapse risk |
| Source locations and total erupted volume | PLUME entry candidates and a world-scale volume budget | Segment-wise discharge history or thermal state |

Neither model can infer, without additional observations or a separate physics
model: (i) buried versus surface emplacement; (ii) simultaneous split/merge
topology; (iii) vertical levels, captures, and underpasses; (iv) tube
cross-sections, floor relief, linings, benches, levees, or lava falls; (v)
cooling/age fields and temperature-monotone flux; (vi) roof competence,
breakdown, skylights, or structural collapse; or (vii) mesh-continuity and
walkable-clearance constraints.  A raster-to-graph adapter must mark every
such quantity as PLUME-generated or heuristic.

## Parameter mapping into PLUME

| External parameter/output | PLUME field or operation | Mapping rule and caveat |
|---|---|---|
| DOWNFLOW DEM | `HostField.elevation`, slope and routing layers | Use the same horizontal frame/resolution; resample only with a recorded method. |
| DOWNFLOW vent(s) | entry nodes / source candidates | Treat multiple vents as source alternatives unless observations justify simultaneous sources. |
| DOWNFLOW `Δh` | route-uncertainty scale; candidate corridor width | Keep in metres and report DEM vertical accuracy; never reinterpret as tube height. |
| DOWNFLOW run count | proposal ensemble size | More runs reduce sampling noise; they do not increase network density or physical flux. |
| DOWNFLOW path length/termination | dominant-route target / terminal candidate | Preserve termination reason; do not add an artificial exit to force connectivity. |
| Flowy/MrLavaLoba DEM | host elevation/slope | Match DEM vertical datum and cell size; archive a checksum. |
| `n_flows` | external ensemble size | It is not PLUME's number of tubes. |
| `min_n_lobes`, `max_n_lobes` | candidate persistence-length prior | Convert lobe-chain length to metres using the raster cell/lobe scale; calibrate only on non-confirmatory data. |
| `lobe_area` | footprint area scale | Use only for corridor width/area opportunity; never set section area directly. |
| `lobe_exponent` | branch/divergence opportunity prior | Map monotonically to `network_density` only after an independent topology calibration; retain the source value. |
| `max_slope_prob` | downhill steering weight | Map to a routing prior, not a deterministic tangent. |
| `thickening_parameter` | width/occupancy prior | Feed low-frequency width variation; Passage Morphology still creates sections. |
| total erupted volume | world-scale occupancy/flux budget | Preserve units and uncertainty; PLUME allocates conserved segment flux independently. |
| thickness/hazard raster | normalized corridor and branch weights | Store as evidence layers with provenance, not as final cave density. |

## Ensemble and reproducibility protocol

This protocol uses only the repository's declared network seeds and mapped
surface-flow cases.  It does **not** read, tune, rank, or otherwise inspect the
frozen 19-cave PDC confirmatory partition.

1. Freeze the external revision (`DOWNFLOW` paper/code revision or Flowy
   `v1.0.0`, commit `6beb3ba`), compiler/interpreter, dependency lockfile,
   DEM checksum, cell size, vent coordinates, stopping rule, and unit system.
2. Run seeds `0..99` for each model and case.  Use seeds `0..79` for proposal
   parameter selection and retain `80..99` as an untouched model audit.  This
   split is independent of the PDC cave split.  Run every selected setting at
   least three times with the same seed to test determinism.
3. For each run archive raw rasters/polylines plus a manifest containing model
   revision, command line, seed, DEM checksum, `Δh` or lobe parameters, runtime,
   exit status, and output hashes.  Flowy's official v1.0.0 executable has
   already produced bit-identical ASC rasters for repeated `rng_seed=424242`
   smoke runs; retain this as a reproducibility gate, not a scientific result.
4. Convert outputs to a common metric space: metres, signed grade, connected
   corridor graph, occupancy probability, and (for Flowy) thickness normalized
   by total volume.  Do not compare pixel counts across different resolutions.
5. Compare distributions over seeds and cases, with bootstrap intervals.  A
   pretty single run or a single hazard map cannot select a generator.

## Quantitative selection criteria

The following are predeclared proposal-generator gates.  Thresholds marked
HEURISTIC are engineering tolerances, not estimates of natural prevalence.

| Gate | Metric and procedure | Acceptance target |
|---|---|---|
| G1, topology | Skeletonize planform proposals at native resolution; compare branch, confluence, cycle, and braiding-index distributions to held-out mapped surface-flow cases (not PDC). | **HEURISTIC:** normalized Wasserstein distance ≤0.25 for each reported topology family, or document the failed family and keep that output as a prior only. |
| G2, terrain response | Length-weighted correlation between candidate tangent and downhill direction; report grade and obstacle bypass rate by slope stratum. | **INFERRED/HEURISTIC:** median downhill alignment ≥0.70; no slope stratum may have zero surviving paths unless the source case is genuinely blocked. |
| G3, uncertainty | Occupancy coverage of the mapped outline by the 95% proposal corridor, with corridor overreach reported separately. | **HEURISTIC:** coverage ≥0.90 and overreach ≤0.50 of corridor area; do not trade all precision for coverage. |
| G4, branch semantics | For Flowy, test whether extracted lobe branches remain connected and whether thickness decreases at splits and increases at confluences; for DOWNFLOW, verify that alternative paths are not merged as simultaneous edges. | **EMPIRICAL/INFERRED:** 100% of imported edges carry source-model and ensemble provenance; zero unlabelled pseudo-junctions. |
| G5, reproducibility | Repeat same seed and command three times; compare bytes/hashes for rasters and canonicalized graph JSON. | **REQUIRED:** bit-identical outputs where the executable claims deterministic behavior; otherwise max coordinate deviation ≤1 raster cell and the cause documented. |
| G6, PLUME invariants | After proposal import, run PLUME's graph checks: one connected component when required, source reachability, conserved split/merge flux, monotone age/temperature, finite junction blends, and valid closed sections. | **REQUIRED:** zero state-consistency violations. External output cannot waive a PLUME invariant. |
| G7, cost/risk | Record wall time, peak memory, build success, dependency count, and license review status on all 100 seeds. | **HEURISTIC tie-break:** prefer Flowy when G1–G6 are tied and its speed advantage is material; never choose speed over failed science or license clearance. |

For a compact selection report, use the weighted score

`0.30·G1 + 0.15·G2 + 0.15·G3 + 0.15·G4 + 0.15·G5 + 0.10·G7`,

after converting each gate to a [0,1] pass score and assigning score 0 to any
model that fails G6.  Report the full vector as well as the score; a scalar is
not permission to conceal a topology or reproducibility failure.

## Licensing, dependencies, and reproducibility risks

* **Flowy:** the official repository is GPL-3.0 and builds with C++20/Meson,
  micromamba/Conda environment files, and NetCDF-related support.  A PLUME
  distribution should invoke a user-provided executable or a separately
  packaged optional adapter unless legal review approves GPL integration. Pin
  the v1.0.0 tag and record compiler, Meson, and dependency versions.  The
  paper reports large runtime gains, but those are machine- and case-dependent;
  benchmark locally rather than treating them as a guarantee.
* **Original MrLavaLoba:** the official repository includes a custom `COPYING`
  license requiring copyright retention, inclusion of the license, and express
  written consent for distribution of the package or modified package.  It also
  depends on Python/NumPy/Matplotlib plus bundled or third-party scripts for
  truncated-normal sampling and shapefile I/O.  Do not vendor or redistribute
  the original code without written permission.  The later MrLavaLoba2 mirror
  advertises Apache-2.0, but its lineage and exact implementation must be
  audited before substitution; treat it as a different revision.
* **DOWNFLOW:** the scientific paper is authoritative for the method, but the
  executable/repository provenance and license must be pinned by the adapter
  owner before redistribution.  Its minimal DEM/vent inputs are a dependency
  advantage, not evidence of tube realism.
* **Common risk:** stochastic output, raster resolution, DEM vertical error,
  compiler math, and hidden default parameters can change graph extraction.
  Every imported proposal must remain reproducible from its manifest and must
  be regenerable without network access after dependencies are cached.

## Bottom line for implementation

Network may consume DOWNFLOW occupancy as a route prior and Flowy occupancy /
thickness as a low-frequency branch or flux prior.  Passage Morphology must
continue to generate widths, heights, floors, roof asymmetry, and junction
blends from PLUME state.  Geometry must continue to construct levels, captures,
falls, breakdown, and mesh continuity.  QA must reject any adapter that turns
an ensemble of alternatives into unlabelled simultaneous tubes or that uses
PDC confirmatory data for selection.

### References

* Favalli, M. et al. (2005), “Forecasting lava flow paths by a stochastic
  approach,” *Geophysical Research Letters*, 32, [doi:10.1029/2004GL021718](https://doi.org/10.1029/2004GL021718).
* de' Michieli Vitturi, M. & Tarquini, S. (2018), “MrLavaLoba: A new
  probabilistic model for the simulation of lava flows as a settling process,”
  *Journal of Volcanology and Geothermal Research*, 349, [doi:10.1016/j.jvolgeores.2017.11.016](https://doi.org/10.1016/j.jvolgeores.2017.11.016).
* Sallermann, M. et al. (2024), “Flowy: High performance probabilistic lava
  emplacement prediction,” *Computer Physics Communications*, preprint and
  official repository: [arXiv:2405.20144](https://arxiv.org/abs/2405.20144),
  [flowy-code/flowy](https://github.com/flowy-code/flowy).
* Dietterich, H. R. et al. (2014), “Channel networks within lava flows,” *JGR
  Earth Surface*, [doi:10.1002/2014JF003103](https://doi.org/10.1002/2014JF003103).


# Emplacement proposal backend decision

Status: provisional engineering decision.  The frozen PDC evaluation partition
is not used here because it constrains passage cross-sections, not surface-flow
routing or network topology.

## Scope

These backends provide **route evidence for Stage B**.  They do not replace
PLUME's semantic graph construction, flux conservation, roofing, stacked
levels, captures, chambers, lobe retirement, or reachability checks.

## Compared models

### PLUME internal hybrid

The built-in model combines the Stage-A routing potential, a correlated
terrain perturbation, seeded lobe-front growth, and the existing network
grammar.  It is the only candidate designed to create a connected buried-tube
graph directly.  It is deterministic, dependency-free, and exposes the full
`network_density` control, but its emplacement rules are project heuristics
rather than a published surface-lava simulator.

### DOWNFLOW reference ensemble

Favalli et al. (2005) describe DOWNFLOW as a fast stochastic hazard model based
on many steepest-descent paths over independently perturbed DEMs
([doi:10.1029/2004GL021718](https://doi.org/10.1029/2004GL021718)).  PLUME's
`downflow_reference` backend implements that published idea transparently; it
does **not** claim to be an official DOWNFLOW library.  It returns a ranked,
seeded ensemble of persistent route corridors.  This is a strong lightweight
prior for topographic uncertainty, but it has no lava volume, deposit
thickness, lobe genealogy, tube roofing, or underground-network semantics.

### Flowy / MrLavaLoba

MrLavaLoba emplaces probabilistic elliptical parcels whose budding responds to
local slope, inertia, volume, and parent selection; its intended outputs are
inundation and final deposit thickness, not a time-resolved flow solution or a
buried conduit graph
([doi:10.1016/j.jvolgeores.2017.11.016](https://doi.org/10.1016/j.jvolgeores.2017.11.016)).
[Flowy](https://github.com/flowy-code/flowy) is the C++20 reimplementation.  Its
paper reports fidelity to MrLavaLoba with substantially lower runtime
([arXiv:2405.20144](https://arxiv.org/abs/2405.20144)).

PLUME invokes the official executable with a generated TOML file and ESRI
ASCII DEM, validates the thickness raster, and reconstructs route evidence
from the lobe-parent CSV.  It never treats every lobe edge as a lava tube.

## Validity and interpretation rules

1. All candidates receive the same Stage-A terrain and named network seed.
2. Repeated proposal cells are removed with chronological loop erasure before
   graph construction.
3. A proposal must contain at least three cells and make material progress in
   the configured downstream direction.  Trapped or upstream Flowy runs are
   explicit failures; PLUME never silently substitutes its internal route.
4. Every network report records backend name, version, executable or algorithm
   provenance, and seed namespace.
5. Backend comparison uses Stage B only.  Cross-section and mesh quality cannot
   be credited to a route proposer.

## Selection criteria

The operational default must first satisfy deterministic reproduction,
single-component connectivity, entry-to-exit reachability, positive flux, and
bounded runtime.  Secondary evidence includes downstream validity, topology
and sinuosity distributions, scientific relevance, portability, integration
cost, and license/dependency burden.

The default remains `internal` unless the matched-seed benchmark shows that an
external prior materially improves valid Stage-B distributions without
reducing success or control.  `downflow_reference` is expected to be the most
useful optional research prior.  `flowy` is expected to be the richer surface
emplacement experiment, but it carries a GPL-3.0 external C++ dependency and a
surface-deposit-to-buried-conduit interpretation gap.

This ordering is deliberately falsifiable: the generated benchmark JSON and
figures record the measurements used for the final decision.

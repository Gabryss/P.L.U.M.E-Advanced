# Scientific review and changes made

This review is based on the pasted manuscript, the current local PLUME-Advanced implementation, the declared evaluation workflow, and the cited primary research pages. The repository already contained substantial uncommitted development work. The initial manuscript-only review did not change the generator or experiments. The later evaluation revision corrected instrumentation and configuration resolution as documented below. The current figure revision also updates the repository's documentation figure builders and assets; its PDF build files are under `tmp/pdfs/`.

## Main assessment

The original draft has a coherent systems contribution, but several sentences imply stronger physical validation or experimental evidence than currently exists. The strongest defensible narrative is an inspectable, spatially conditioned lava-tube scenario generator that preserves intermediate representations and supports controlled robotics experiments. Quantitative superiority, final-mesh morphology, successful application imports, and robotics performance remain separate claims requiring evidence.

The expanded draft adds a problem formulation, implementation-level mechanics, statistical definitions, detailed protocols, ten inserted figures, and a more explicit discussion of validity. Tables IV–VI now use measured campaign data; the original reporting template has been replaced with the completed campaign results.

## Illustrated package correction

The first ZIP contained figure candidates without inserting most of them. This edition includes the eight method/development figures and two measured campaign figures directly in the manuscript, with no empty figure boxes. Captions identify seed-4 inspection outputs, analytical schematics, a current optional-event scenario on the seed-4 base, and a separate controlled surface-relief study. These eight illustrations support the methods description. Figures 9–10 separately show measured campaign outcomes. The ZIP requires no manual figure assembly.

The latest revision adds 641 Rocky rocks and eight boulders to the current seed-4 event illustration (Figures 6–7). Figure 6 includes an actual exported-scene interior view and measured prop sizes. Two of three structural candidates modify the volume; the refreshed atlas retains 630 of 635 cells. Captions distinguish accepted volume changes, separate rock geometry, and procedural labels, including collapse-influence metadata retained after a rejected collapse modifier. This is an illustrative development run, not completion of the declared evaluation campaign. The original no-event inspection asset remains unchanged. Figures 2–4 retain the removal of redundant overall headings and subtitles.

The appearance discussion also now describes the saved geometry-stage relief study separately from texture-only displacement. The former changes the density boundary; the latter is an appearance path whose collision implications must be handled separately.

## Substantive corrections

1. **Novelty positioning.** Host-conditioned cave networks are not new by themselves. Paris et al. already combine geological constraints, skeletons, and implicit cave geometry. The binary table marking broad capabilities “No” or “Limited” was replaced with a documented-scope table. The new contribution is framed around PLUME's integration and lava-tube/robotics semantics. [Primary paper](https://diglib.eg.org/bitstream/handle/10.1111/cgf14420/v40i7pp277-287.pdf).

2. **Current Stage B.** The old “braid grammar” description omitted the current lobe growth, finite phase supply, retirement, coalescence, reoccupation, and retained history. These are now described as procedural mechanisms. [Implementation](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/stages/network.py).

3. **Two different roof models.** The host routing prior remains `rho*g*Wc^2/(cover*strength)`. The newer geometry-specific beam screen uses actual profile envelopes and `3*F*rho*g*W^2/(4*strength)` as required roof thickness. They are distinct and are not substituted for each other. Conditional width/height limits and the restricted mechanical assumptions are explained. [Screen](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/stability.py), [integration notes](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/docs/gravity_and_realism.md).

4. **Formation versus surviving connectivity.** Local roof failure can insert blocking plugs independently of optional event settings. A formation edge may therefore exist when final free space is blocked. Pointwise floor clearance is also not a robot-feasibility test. This distinction now appears in the formulation, methods, evaluation, and discussion.

5. **Flux conservation at boundaries.** Conservation is stated for internal junctions. Sources inject supply; exits and retired terminals act as sinks in bookkeeping. Applying the equality to every vertex without boundary terms would be incorrect.

6. **Temperature and age at merges.** The implementation uses flux-weighted incoming states. Cooling and age growth hold within each segment, but a merge may be warmer than a cold inlet or younger than an old inlet. The original global monotonicity wording was corrected, and the actual width-squared allocation and segment propagation are specified.

7. **Ablation interpretation.** Removing one weight normally renormalizes the others. Removing the explicit fracture routing term does not remove indirect fracture effects in competence/capacity. The saved condition named `unconditioned` disables the routing-cost field, not all terrain and host dependencies. The paper now calls these routing-term interventions. [Routing](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/stages/host_field.py), [experiment](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/evaluation/experiments/host_ablation.py).

8. **Execution order and density sign.** The workflow constructs base density, applies structural changes, then meshes the final boundary. Density above the isovalue denotes void in the current implementation. The field is not asserted to be an exact signed-distance function. The schematic and equations now reflect these conventions.

9. **Atlas scope.** The final atlas is refreshed after structural events and retains invalidated cells. Junction charts can overlap, and separate rocks are not automatically represented by the density-based clearance. Planning must combine atlas information and the final collision scene. [Atlas](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/stages/floor_map.py).

10. **Morphometric statistics and leakage.** Added the existing 76-cave/19-cave partition, disclosure that a whole-catalog exploratory analysis preceded the split, cave/world cluster resampling, descriptor definitions, and distinction between dependence and unequal sampling weights. The Stage-C experiment validates profiles, not the event-modified final mesh. [Partition disclosure](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/paper/splits/README.md).

11. **Baseline interpretation.** Ellipses are matched to each generated section's width and height. Width, height, and aspect ratio are therefore identical by construction. Shape improvements against this baseline cannot establish improvement over the complete Pyroduct or original PLUME software. The aggregate descriptor set is named explicitly.

12. **Sampling claims.** The current experiment approximately matches section count and measures bidirectional nearest-neighbor distances between profile point clouds. It does not establish reduced sample count at matched continuous-surface error, nor Stage-C speedup. A stronger efficiency claim needs a spacing sweep, a shared error threshold, and timings. [Experiment](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/evaluation/experiments/sampling_ablation.py), [metric](/home/gabriel/Lab/python/P.L.U.M.E-Advanced/src/plume_advanced/evaluation/metrics/sections.py).

13. **Scalability limits.** Tiling avoids empty-space storage but cannot guarantee total memory independent of scene size. Global meshing, welding, UVs, and export can dominate. Requested and realized route lengths, timing scope, completion counts, and peak memory are now required in reporting.

14. **Export verification and LaTeX repair.** Corrected the malformed `\mathcal{A}*k` and `\epsilon*{...}` expression. The new formula first maps imported geometry back to canonical coordinates. Bounds alone cannot verify translation, topology, materials, or contact. Package emission and actual application imports are reported separately.

15. **Appearance and deferred work.** Added the existing UV/tangent/PBR and visual-displacement path, its separation from collision geometry, and the deferred geology-conditioned materials, LODs, and outer wall shell. These features are not presented as finished contributions.

## Reference review

The bibliography keeps the supplied IEEE-style inline bibliography in a separate `references.tex` file, so Overleaf does not require BibTeX. Verified DOI links were added where available. The following checks are especially relevant:

- **PDC:** pin the version-specific DOI `10.5281/zenodo.17750755` while retaining the catalog's recommended citation `10.5281/zenodo.14535885`. The public record's broad count and the repository's audited identifier count refer to different levels of precision. [Zenodo version record](https://zenodo.org/records/17750755).
- **Pyroduct:** Icarus volume 447, article 116904 is a 2026 issue, while its DOI contains 2025. These are not conflicting years. [Publisher page](https://www.sciencedirect.com/science/article/pii/S001910352500452X).
- **Cano/Gazebo:** the publisher's preferred citation is 2023, although its online publication date is November 2022. The 2023 citation is retained and pages/DOI are added. [Publisher citation](https://link.springer.com/chapter/10.1007/978-3-031-21062-4_26).
- **Lunar observations:** added Carrer et al.'s radar evidence for a conduit. This motivates the problem without implying that a dense interior survey is available. [Nature Astronomy](https://www.nature.com/articles/s41550-024-02302-y).
- **Volumetric terrain:** the final publisher entry includes Sergio Huerta, unlike the accessible manuscript copy; the bibliography follows the publisher's author list and 2014 issue year. [Publisher page](https://link.springer.com/article/10.1007/s00371-013-0909-y).
- **PLUME:** retained the supplied iSpaRo 2025 attribution and added the verified [arXiv preprint](https://arxiv.org/abs/2508.20926). Reconcile the final proceedings DOI and pagination with the author's publication record before submission; they were not guessed from secondary listings.
- **DAEDALUS and Antoniuk:** the existing entries were retained with typographic cleanup. The report and cited prior-work trail support them, but a full publisher-level bibliographic audit of every field remains advisable when adopting the final venue format.

The broad lunar-base construction review from the original bibliography was omitted because the revised argument no longer relies on base-construction claims. This is a scope edit, not a judgment on that article.

## Evaluation revision, 7 September 2026

The actual modified package and inputs were preserved before evaluation, and
Tables IV–VI use the completed morphology, routing and resource measurements.
The original baseline commit alone does not identify the evaluated source.

Corrections include final meshing in the scalability worker, preservation of
resource-limit failures, paired routing intervals, and a joint cave/world
bootstrap distribution for the morphology aggregate. Before any control cases
ran, the sweep was corrected to resolve every stage from each edited flow input.
A separate snapshot records that amendment; the default configuration hash is
unchanged. No model parameters were fitted to evaluation outcomes.

Morphology uses 100 Earth worlds and 200 contours from the 19 evaluation caves.
The selected normalized distance is 0.254, versus 0.840 for matched ellipses;
absolute sizes and variability remain mismatched. All 700 routing cases
completed. All 40 resource cases were attempted: tiled storage completed 20/20;
dense completed 15/20, with all five 5 km attempts exceeding the memory budget.
All 1,200 control cases, 30 sampling cases, three repeatability cases and three export cases also completed. Duration changed main-route length; inflation had a modest width-ratio response; distributary tendency had zero median cycle-rank change. Adaptive sampling had greater point-set discrepancy in every pair. All 33 A–C repeatability checks and 15 package-presence checks passed; no application imports were performed. The report records every suite's completed, failed and unperformed checks.

Resource costs cover host generation through final meshing at 0.6 m resolution.
Events, props, appearance, exports and application imports are excluded.
Under-resolved profiles and disconnected components prevent interpreting
completion as a fidelity certificate. The retained Figures 6–7 rock scenario
is separate from these experiments.

## What still needs evidence before submission

The principal remaining work outside the declared campaign is:
- Add final-mesh cross-section validation if the paper claims final scene morphology matches terrestrial data.
- Add an error-matched spacing/timing sweep if retaining a sampling-efficiency claim.
- Complete actual application imports, including scale, pose, material and dynamic-contact checks.
- Include a robotics task evaluation if making claims about improved autonomy or transfer.
- Restore the appropriate author block and adapt length and anonymity to the target venue.

The frozen configuration copies and hashes in this package document the review context. They are not a reproducible substitute for releasing the generator and its dependencies.

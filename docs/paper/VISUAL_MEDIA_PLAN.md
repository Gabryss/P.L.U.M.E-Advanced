# PLUME-Advanced visual media plan

## Purpose

This document defines the publication figures, supplementary visuals,
animation, and conference video for PLUME-Advanced. It is grounded in the
current repository rather than an aspirational feature list.

The central visual story is:

> A deterministic, host-conditioned pipeline turns an interpretable
> geological substrate into topology-rich, simulation-ready lava-tube scenes,
> while preserving scale and semantics across export targets.

The visuals must not imply that PLUME-Advanced is a thermofluid simulator or
that the Mars and Moon presets are externally validated geological models.

## Repository audit

### Ready now

- The inspectable A-E pipeline is implemented and has one high-resolution
  diagnostic image per stage for Earth, Mars, and the Moon.
- The repository contains 24 body/stage PNGs under
  `docs/figures/celestial_bodies/`.
- Stage D has a cleaner 16:9 presentation renderer in addition to engineering
  and chunk diagnostics.
- The generator emits semantic graph, section, floor-atlas, geometry, event,
  manifest, and portable scene artifacts that can drive new figures without
  scraping pixels from existing plots.
- The scientific evaluation package already defines morphometry,
  controllability, host ablation, sampling ablation, scalability, export
  consistency, and determinism experiments.
- Saved-result plotting already exports PNG and SVG for five experiment
  families.

### Not ready yet

- `paper/outputs/` is absent, so confirmatory results are not currently
  available for final quantitative figures.
- The current stage images are engineering dashboards. They are useful as
  source material and supplementary diagnostics but are too dense to serve as
  the main paper narrative unchanged.
- No camera-path, frame-sequence, animation, or video-compositing pipeline is
  implemented.
- No conference-specific page size, column width, duration, resolution, or
  accessibility requirements have been frozen.
- Geology-conditioned surface synthesis, visual LODs, and finite wall shells
  remain deferred and must not appear as completed contributions.

## Main-paper figure set

Target six main figures. Each figure must answer one question and map to a
defensible claim.

| ID | Figure | Scientific question | Panels | Source / gate | Priority |
|---|---|---|---|---|---|
| F1 | System overview | How does PLUME turn world parameters into a portable cave scene? | Inputs; A-E pipeline; canonical scene; export targets | Current stage artifacts and exporter metadata | Build now |
| F2 | Host-conditioned generation | How do interpretable host fields affect route formation? | Host layers; routing-cost composition; conditioned network overlay; unconditioned counterfactual | Stage A/B plus E5 pair | Build after one verified E5 pair |
| F3 | Topology and geometry construction | How does the semantic graph become continuous 3D geometry? | Braided graph; adaptive sections; SDF stamping/junction blend; final mesh/floor atlas | Stage B-D semantic artifacts | Build now |
| F4 | Morphometric evaluation | Are generated Earth cross-sections closer to the terrestrial reference than a matched ellipse? | Representative contours; descriptor ECDFs; normalized distance summary with confidence intervals | Confirmatory E1 only | Blocked on final campaign |
| F5 | Controllability and ablations | Do exposed controls and host terms produce measurable, interpretable effects? | Control-response small multiples; host-term ablation; adaptive-vs-uniform fidelity/cost | E2, E5, E6 | Blocked on final campaigns |
| F6 | Scale and portability | Does the system scale and preserve the same canonical scene across packages? | Time/RSS vs route length; dense/tiled comparison; canonical-to-target diagram; verified scale/bounds table | E7, E9, manual import protocol | Blocked on final campaign and imports |

F1 should be the visual abstract and first paper figure. It must use a single
left-to-right reading direction and a restrained schematic style rather than
placing screenshots inside boxes. F3 should be the technical centerpiece: the
same highlighted branch/junction must remain visually tracked across graph,
sections, density, mesh, and atlas panels.

## Supplementary figure set

| ID | Content | Existing basis |
|---|---|---|
| S1 | Full A-E pipeline for Earth | Existing eight-image Earth gallery |
| S2 | Controlled Earth/Mars/Moon scenario comparison | Existing body gallery; label as scenario extrapolation |
| S3 | Network grammar and junction taxonomy | Segment kinds and junction metadata from Stage B |
| S4 | Floor-atlas coordinate system and multi-level disambiguation | Stage C2 atlas JSON/NPZ and render |
| S5 | Geological event families and topology changes | Base/post-event density and event report |
| S6 | Dense/tiled chunk diagnostics and seam checks | Existing Stage-D chunk outputs plus E7 |
| S7 | Determinism and stage-seed isolation | Manifests and E10 semantic hashes |
| S8 | Export/package validation | Validation report and completed manual-import protocol |
| S9 | Failure cases and rejected cases | Preserved failed/timeout/invalid experiment records |

Failure cases are mandatory research evidence, not promotional outtakes.
Include at least one topology failure, one difficult morphology, and one
resource-limit case if these occur in the frozen campaign.

## Animation package

Produce one master 16:9 animation from reusable shots, then derive the video,
talk inserts, GIF/WebM teaser, and supplementary clip from it.

### Core sequence (about 55 seconds)

| Time | Shot | Visual action | On-screen message |
|---:|---|---|---|
| 0-4 s | Problem | Empty terrain transitions to an underground cutaway | Realistic, controllable lava-tube environments are difficult to author at scale |
| 4-11 s | Stage A | Host layers reveal one at a time and combine into routing influence | Interpretable host conditioning |
| 11-18 s | Stage B | Feeders grow, split, merge, braid, and underpass | Flux-carrying topology, not a single spline |
| 18-25 s | Stage C | Cross-sections sweep along a highlighted branch; sampling densifies near curvature/junctions | Adaptive, continuous morphology |
| 25-34 s | Stage D | Sections stamp into a density field; chunks polygonize and weld | Scalable voxel-to-mesh construction |
| 34-41 s | Stage E | Collapse/choke/infill alter the volume; grounded props appear | Geological events change geometry and traversability |
| 41-47 s | Floor atlas | World view flattens into intrinsic segment-distance-lateral coordinates | Topology-aware placement and inspection |
| 47-52 s | Canonical scene | Texture, collision, and semantics appear as separate layers | One canonical simulation-ready scene |
| 52-55 s | Export | The same scene fans out to Blender, UE5, Unity, Gazebo, and Omniverse | Representation changes; the cave does not |

### Animation implementation rules

- Render a numbered PNG frame sequence first; encode MP4/WebM only after
  frame-level review.
- Keep geometry identity stable across stages. Camera cuts must not make a new
  cave look like the continuation of the old one.
- Use semantic artifacts as animation inputs; do not animate screenshots.
- Use a cutaway, clipping plane, or inward-facing cave camera because the mesh
  represents a void boundary and is intentionally single-sided.
- Display scale bars and units whenever spatial extent is visible.
- Use overlays and captions that remain legible with audio muted.
- Preserve a clean master without captions so conference-specific variants can
  be composed without rerendering geometry.

## Conference video storyboard

Target a 2:30 master unless the selected venue specifies another limit.

| Time | Section | Content |
|---:|---|---|
| 0:00-0:12 | Hook | Fast underground reveal, research problem, one-sentence contribution |
| 0:12-0:30 | Why it matters | Need for large, controllable, inspectable environments; clarify scope |
| 0:30-1:25 | Method | Use the 55-second core animation with concise narration |
| 1:25-1:55 | Evidence | Morphometry, controllability/ablation, and scalability figures using frozen results |
| 1:55-2:15 | Outputs | Final textured scene, floor atlas, collision/semantic layers, verified target imports |
| 2:15-2:30 | Limits and takeaway | Terrestrial validation scope, planetary presets as scenarios, final contribution statement |

The video should contain no unverified superlatives. Replace terms such as
"realistic" with the measured property, for example "closer on the selected
cross-section descriptors," when the final result supports it.

## Visual system

- Use one color per pipeline concept across every medium: host/terrain ochre,
  network cyan, sections violet, density/geometry slate, geological events
  vermilion, evaluation green, and export targets neutral gray.
- Reserve vermilion for interventions, hazards, or failure emphasis; do not use
  it as a generic series color.
- Use a color-vision-safe categorical palette and verify every figure in
  grayscale.
- Label panels directly where possible; minimize legends and avoid rainbow
  colormaps.
- Plot physical quantities with units and report uncertainty, sample count,
  and excluded/failed cases in the caption or panel.
- Export paper diagrams and plots as PDF/SVG plus a 300 dpi PNG preview.
- Keep text editable in vector masters and use the paper's final typeface once
  the venue template is known.
- Maintain separate `diagnostic`, `paper`, `slides`, and `video` render themes.

## Proposed asset layout

```text
paper/media/
  style/
  figures/
    main/
    supplementary/
  animation/
    shots/
    frames/
    masters/
  video/
    storyboard/
    narration/
    captions/
    exports/
  provenance/
```

Every final asset should have a provenance record containing the commit,
resolved configuration, seed or experiment IDs, source artifact hashes,
render command, dimensions, and creation time. Final figures must consume
saved experiment outputs and must never silently rerun generation.

## Production order and gates

1. Freeze the target conference specifications and the paper's exact claim
   wording.
2. Create the shared visual theme and F1/F3 from current semantic artifacts.
3. Run and audit the confirmatory E1, E2, E5, E6, E7, E9, and E10 campaigns
   from a clean, recorded commit.
4. Replace the current generic evaluation plots with claim-focused composite
   figures including uncertainty and failure counts.
5. Complete manual target imports before showing or claiming cross-engine
   consistency.
6. Implement reusable animation primitives and render the 55-second silent
   master.
7. Record narration only after all quantitative statements and captions are
   frozen.
8. Produce conference, talk, poster, supplementary, and web derivatives from
   the same masters.
9. Perform a final scientific-claim audit, visual accessibility check, and
   playback test on the conference delivery machine.

## Immediate implementation milestone

The first implementation milestone should deliver:

- F1 system overview;
- F3 graph-to-geometry figure;
- a visual style module shared by paper plotting and stage presentation
  renderers;
- a 10-15 second proof-of-concept animation covering Stage B to Stage D;
- provenance sidecars for all four outputs.

This milestone is independent of the pending confirmatory campaigns and will
establish the visual language needed by every later asset.

# Seed-4 events with actual Rocky geometry

This is the active source snapshot for Figures 6 and 7. On the existing
Earth seed-4 base, Rocky produced 641 rocks and eight boulders. They remain
649 separately editable meshes in the exported scene. The cave voxel spacing
is 0.20 m; prop meshes are separate geometry and can represent smaller features.

The population multiplier stays at 1.0. Event settings change only `enabled`,
`include_rock_props`, `use_rocky_meshes`, and `enabled_kinds` from the base
configuration. All five event kinds are enabled. The structural candidates
match the preceding illustration in position, dimensions, and severity.
After the full event list is sorted, their IDs are 208 (choke), 226 (collapse),
and 386 (infill). Choke and infill modify the volume; collapse is rejected.

The atlas retains 630 of 635 cells across 25 segment bands. Its geology labels
are refreshed with the new props. Density-based overhead clearances do not
subtract separate rocks, so they cannot be used alone for collision checking.

## Replot the paper figures

In the installed PLUME repository, run:

```bash
python scripts/generate_paper_event_figures.py \
  --output paper/overleaf/source_snapshot/events_rocks_seed4 --render-only
```

The bundled `rock_interior.png` is an actual exported-scene render. Ochre and
gray are diagnostic colors used to identify props and cave geometry; material
textures are not displayed. Camera and node information is saved in
`rock_render_report.json`. The histogram measures the maximum world-X/Y
extent of each actual prop mesh, not its requested nominal radius.

## Recompute from the original trusted base checkpoint

From the repository root, with the core and `rocks` dependencies installed:

```bash
python scripts/generate_paper_event_figures.py \
  --checkpoint /tmp/plume-inspection-seed4.pkl \
  --config outputs/earth_inspection_seed4/generation_config.toml \
  --output outputs/paper_events_rocks_seed4 \
  --include-rocks --collect-only
python scripts/render_event_rocks.py outputs/paper_events_rocks_seed4
python scripts/generate_paper_event_figures.py \
  --output outputs/paper_events_rocks_seed4 --render-only
```

The large base checkpoint and full GLB are not bundled in the Overleaf ZIP.
Their identities are recorded in the provenance and export reports. The GLB
is available in the original run directory as `lava_tube_with_rocks.glb`.
The optional `--event-field` argument reuses a matching trusted event
checkpoint; it was used after switching the inspection export to neutral
materials to avoid unnecessary texture-atlas generation. Only the completed
neutral export is used in these figures. The script copies here are records;
they require an installed PLUME environment and are not a relocated runtime.

The base TOML's relative asset paths refer to its original run directory.
`resolved_config.json` includes the resolved event overrides. Load only trusted
pickle checkpoints. Non-finite optional report values are serialized as null.
This is one illustrative development scenario, not a confirmatory campaign.

# Current event and floor-atlas figure data

This snapshot supports Figures 6 and 7. It uses the current Earth seed-4
base geometry at 0.20 m voxels, with a separate optional-event finalization.
The original no-event inspection GLB is preserved. No rocks were generated.

`study.json` stores cell attributes, event records, graph plan coordinates,
counts, overrides, and original input hashes. `event_section.npz` stores
before/after density samples in the same source-section plane. The two atlas
exports, reports, and resolved configuration provide supporting records.
Non-finite optional diagnostic values are serialized as JSON null.

Replot from the repository root, using its installed PLUME environment:

```bash
python scripts/generate_paper_event_figures.py \
  --output paper/overleaf/source_snapshot/event_atlas_seed4 --render-only
```

The bundled script copy records the exact builder used. It requires PLUME
and its plotting dependencies; this snapshot is not a standalone generator.

The original data-generation command, run from the repository root, was:

```bash
python scripts/generate_paper_event_figures.py \
  --checkpoint /tmp/plume-inspection-seed4.pkl \
  --config outputs/earth_inspection_seed4/generation_config.toml \
  --output outputs/paper_event_atlas_seed4
```

The large checkpoint is not bundled. Load only a trusted checkpoint: pickle
files execute code when loaded. The saved base TOML records the original
configuration; its relative asset paths refer to its original run directory,
not this relocated snapshot. `resolved_config.json` includes event overrides.

One choke and one infill modifier were applied. The collapse candidate was
rejected by volume safeguards. Of 635 base atlas cells, 630 survive and five
are invalidated. The atlas uses a 1 m minimum overhead clearance; it is not
a complete free-space or robot-traversability representation. Breakdown
labels retain collapse-influence metadata and do not imply loose debris.
These are illustrative results from one scenario, not a confirmatory campaign.

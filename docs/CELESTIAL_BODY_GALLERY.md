# Celestial-body generation gallery

These figures are generated from the same flow-regime configuration and
procedural seed. Only the celestial-body preset and its default geological
material change. The preset changes host correlation lengths, vertical and
fracture scales, route guidance, passage/room limits, section sampling, and
floor-map resolution. This makes differences in gravity, stability, passage
scale, terrain, network growth, and geological events directly comparable
without merely inflating one shared footprint.

The gallery uses preview-quality body-aware geometry resolution: 1 m voxels
for Earth, 2 m for Mars, and 4 m for the Moon. This keeps the narrowest
generated sections sampled by approximately 8, 19, and 20 voxels respectively
while avoiding one wasteful shared resolution.

Refresh the complete gallery with:

```bash
.venv/bin/python scripts/generate_body_figures.py
```

If the gallery already exists, confirm the interactive prompt or use
`--force-overwrite` for an intentional unattended refresh.

## Earth

| Pipeline view | Figure |
|---|---|
| Host field | ![Earth host field](figures/celestial_bodies/earth/stage_a_host_field.png) |
| Cave network | ![Earth cave network](figures/celestial_bodies/earth/stage_b_cave_network.png) |
| Section field | ![Earth section field](figures/celestial_bodies/earth/stage_c_section_field.png) |
| Floor map | ![Earth floor map](figures/celestial_bodies/earth/stage_c_floor_map.png) |
| Geometry diagnostics | ![Earth geometry diagnostics](figures/celestial_bodies/earth/stage_d_geometry.png) |
| Chunk diagnostics | ![Earth chunk diagnostics](figures/celestial_bodies/earth/stage_d_geometry_chunks.png) |
| Geometry presentation | ![Earth geometry presentation](figures/celestial_bodies/earth/stage_d_geometry_presentation.png) |
| Geological events | ![Earth geological events](figures/celestial_bodies/earth/stage_e_geological_events.png) |

## Mars

| Pipeline view | Figure |
|---|---|
| Host field | ![Mars host field](figures/celestial_bodies/mars/stage_a_host_field.png) |
| Cave network | ![Mars cave network](figures/celestial_bodies/mars/stage_b_cave_network.png) |
| Section field | ![Mars section field](figures/celestial_bodies/mars/stage_c_section_field.png) |
| Floor map | ![Mars floor map](figures/celestial_bodies/mars/stage_c_floor_map.png) |
| Geometry diagnostics | ![Mars geometry diagnostics](figures/celestial_bodies/mars/stage_d_geometry.png) |
| Chunk diagnostics | ![Mars chunk diagnostics](figures/celestial_bodies/mars/stage_d_geometry_chunks.png) |
| Geometry presentation | ![Mars geometry presentation](figures/celestial_bodies/mars/stage_d_geometry_presentation.png) |
| Geological events | ![Mars geological events](figures/celestial_bodies/mars/stage_e_geological_events.png) |

## Moon

| Pipeline view | Figure |
|---|---|
| Host field | ![Moon host field](figures/celestial_bodies/moon/stage_a_host_field.png) |
| Cave network | ![Moon cave network](figures/celestial_bodies/moon/stage_b_cave_network.png) |
| Section field | ![Moon section field](figures/celestial_bodies/moon/stage_c_section_field.png) |
| Floor map | ![Moon floor map](figures/celestial_bodies/moon/stage_c_floor_map.png) |
| Geometry diagnostics | ![Moon geometry diagnostics](figures/celestial_bodies/moon/stage_d_geometry.png) |
| Chunk diagnostics | ![Moon chunk diagnostics](figures/celestial_bodies/moon/stage_d_geometry_chunks.png) |
| Geometry presentation | ![Moon geometry presentation](figures/celestial_bodies/moon/stage_d_geometry_presentation.png) |
| Geological events | ![Moon geological events](figures/celestial_bodies/moon/stage_e_geological_events.png) |

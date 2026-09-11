# Geometry for lava accretion and crust relief

The Earth tube study now includes physical relief on its walls, roof and floor.
These features are part of the voxel surface and exported vertices, with no
texture displacement or separate rock objects. The original section contours
remain the larger passage envelope.

## Morphological basis and limits

[USGS Bulletin 1673's description of Valentine Cave](https://npshistory.com/publications/geology/bul/1673/sec3.htm)
documents smooth sheared walls alongside complex accretion benches, pahoehoe
floors, and thick, short ceiling lavacicles. This motivates contrasting smooth
and rough patches and different forms on the floor, wall and roof. It does not
provide calibrated distributions for the procedural amplitudes below.

The model approximates inward accretion using broad lobes, broken wall layers,
localized floor ridges, scattered tapering roof projections and smaller crust
creases. It is not a fluid, solidification, fracture or erosion simulation.
Wall layers are approximately gravity-horizontal and warped in space; they
are not integrated along a reconstructed historical lava free surface. Fine
cracks and centimetre-scale drip textures remain outside this mesh resolution.

## Controls

All relief controls live in `[geometry]`. Their defaults are zero, preserving
other scenarios; the Earth tube study explicitly enables them. These Earth
settings are not assumed to describe the Moon, Mars or icy moons.

| Control | Earth study value | Meaning |
| --- | ---: | --- |
| `surface_wall_relief_m` | 0.55 m | Upper amplitude for wall lobes and broken layers |
| `surface_roof_relief_m` | 0.65 m | Upper amplitude for ceiling relief and projections |
| `surface_floor_relief_m` | 0.22 m | Upper amplitude for floor crust and ridges |
| `surface_crust_relief_m` | 0.10 m | Upper amplitude for the smaller crust octave |
| `surface_feature_scale_m` | 0.85 m | Base spatial scale; broader forms use multiples |
| `surface_normal_filter_voxels` | 0.75 | Density filter width for shading normals |

Amplitudes are bounds, not typical displacements. Patch strength, orientation,
resolution attenuation and a local clearance cap reduce the actual offset.
Relief is measured in metres rather than the earlier dimensionless density
roughness, which becomes physically weaker as the voxel size shrinks.

The finest octave is reduced when its requested scale is unresolved. The
full-network export uses 0.20 m voxels; the representative passage uses 0.10 m.
Both keep two surface smoothing iterations. Reducing the normal filter from
1.2 to 0.75 voxels preserves more of the resolved relief while continuing to
reconstruct gradients across neighboring tiles. Changing normals alone is
not counted as geometric detail.

## Pipeline and safeguards

The relief pass runs after passage unions and solid-pocket cleanup and before
mandatory roof stability screening. It decreases density only, so it cannot
increase the carved void or enlarge the input roof span. The local clearance
cap uses 40% of the nearby peak distance to the original wall. This reduces
accretion in low passages; it is not a proof that every route remains traversable.
Roof screening still uses the original, larger section/junction envelopes.

Rotated, seeded world-space value fields are continuous across section joins.
Tiled evaluation reads an immutable copy of the original neighboring density
samples, then synchronizes overlapping samples. No random sequence is restarted
at a chunk, section or branch boundary. Dense and tiled results are checked
against each other, including nonzero isovalues.

After accretion, an isolated-pocket repair removes solid flakes or closes air
specks of at most eight voxels. It preserves larger cavities and connected
dividers. This handles unresolved components created at tangential boundaries;
the repair runs before mandatory roof-collapse screening.

The envelope guarantee applies to the accretion operation, before pocket repair. Marching-cubes
discretization and subsequent mesh smoothing have their usual small numerical
offsets. A watertight export is not, by itself, a geological validation.

## Reproduction

```bash
.venv/bin/python scripts/inspect_surface_relief.py \
  --config config/earth_tube_only.toml \
  --output-directory outputs/surface_relief_study --voxel-size 0.10
.venv/bin/python scripts/generate_tube_only.py \
  --config config/earth_tube_only.toml \
  --output-directory outputs/earth_tube_relief_seed2
.venv/bin/python scripts/render_tube_views.py outputs/earth_tube_relief_seed2 \
  --cameras outputs/earth_tube_fixed_seed2/inspection_cameras.json
.venv/bin/python scripts/check_tube_sections.py outputs/earth_tube_relief_seed2 --shallow-count 6
```

The passage comparison holds resolution, camera, material, smoothing and normal
filter fixed, so its visible difference comes from the new geometric relief.
It uses an isolated reach of the generated segment with artificial closed ends;
the far closure can appear in the distance. It is not a complete separate cave.

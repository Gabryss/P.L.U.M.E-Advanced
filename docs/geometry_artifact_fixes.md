# Geometry artifact corrections

This revision addresses the drum-shaped chamber, abrupt level ramps, repeated
wall ribs and unwanted interior ledges observed in the Earth seed-2 model.
It retains the host → network → section field → mesh pipeline and gravity-based
roof screening. Rock props remain disabled in the tube study configuration.

## Causes and corrections

- **Chambers were constructed twice.** Stage C widened and shaped the passage,
  but Stage D added an independent room with steep sides and a shallow roof.
  With `use_section_profiles = true`, the section sweeps now define the room.
  Junction records retain a conservative stability envelope and identify the
  construction as `section_sweeps`; they no longer add an analytic room.
- **Pillar preservation damaged the distance field.** Setting the entire solid
  column to a fixed negative value also overwrote the narrow band beside the
  wall, imprinting the voxel lattice on the isosurface. Profile sweeps already
  preserve the intervening solid. The legacy primitive path restores the
  original negative values instead of a constant.
- **Adjoining sweeps disagreed on bends.** Independent projection onto each
  chord assigned different interpolated sections to a shared boundary point.
  Sweeps now meet on the shared section planes. Internal overlaps are small;
  rounded full-size caps are reserved for chain ends.
- **Profile interpolation was quantized.** Rounding the interpolation fraction
  to two decimals made 100 discrete steps in each sweep. Evaluation is now
  continuous, with bounded batches and conservative rejection outside the
  surface band to control memory and computation.
- **A helper capsule overrode shallow profiles.** Its minimum vertical radius
  could enlarge a passage beyond its supplied contour. Profile mode now uses
  the contour without that capsule or a second junction radius multiplier.
- **Vertical placement created decks and exit ramps.** Automatic separation
  uses passage height and intervening-rock clearance. A smooth displacement
  spans the branch, with amplitude constrained by cover and uphill grade;
  individual points are not clipped into flat plateaus. An explicit positive
  `vertical_level_spacing` remains a requested minimum separation.
- **Bends and wall turns were undersampled.** Shape-preserving centerline
  interpolation resolves bends while preserving the floor independently of
  section height. Cross-section vertices are concentrated near wall turns.
- **Triangle normals exaggerated residual grid banding.** Undisplaced cave
  surfaces now use reconstructed distance-field gradients for their shading
  normals. Neighboring tiles share the reconstruction stencil, preventing
  lighting seams. This changes the shading normals, not the vertex positions.
  Displaced textured surfaces retain geometric normals after displacement.

The legacy `use_section_profiles = false` mode retains analytic primitive
geometry for compatibility. These morphology corrections target the normal
profile-based pipeline.

## Reproduction and inspection

```bash
.venv/bin/python scripts/generate_tube_only.py \
  --config config/earth_tube_only.toml \
  --output-directory outputs/earth_tube_fixed_seed2
.venv/bin/python scripts/render_tube_views.py outputs/earth_tube_fixed_seed2
```

The geometry-only command runs fresh host, network, section and mesh stages,
keeps mandatory roof screening, and avoids rock generation and texture baking.
It exports a single, double-sided GLB in metres with the glTF Y-up convention.
The inspection renderer reads that exported GLB and records its cameras. It
uses neutral lighting; the images are mesh renders, not generated illustrations.

The initial corrected Earth study used 0.25 m voxels. Its sampled contours have a median height
of about 1.91 m; about 86% lie between 1 and 3 m. These are scenario parameters,
not universal Earth lava-tube limits. Final mesh cuts should be read separately
from the sampled contour statistics.

The current Earth study configuration adds [surface relief](surface_relief.md)
at 0.20 m resolution. Use `outputs/earth_tube_relief_seed2` for that revision;
the older `earth_tube_fixed_seed2` output retains its original files and resolved
configuration for comparison.

## Validation scope

Regressions cover continuous profile interpolation, agreement at a shared
section plane, shallow-section clearance, preserved wall distances around
solid remnants in dense and tiled storage, chamber roof/width bounds, smooth
level displacement and mandatory gravity-dependent roof failure. Export checks
cover manifold closure, winding, finite vertices, nondegenerate faces and the
absence of props. Direct interior and junction views are a separate review.

A requested vertical level may not fit within local cover and grade limits.
The smooth displacement does not certify every graph level as a physically
separated layer. Nor does the roof screen replace fracture, arching, thermal
history or full geomechanical analysis. The result remains a procedural model
requiring calibration against surveyed caves.

The reference morphology includes irregular chambers, benches, flow divides,
arched roofs and local floor drops described in
[USGS Bulletin 1673, Valentine Cave](https://npshistory.com/publications/geology/bul/1673/sec3.htm).
The interpolation uses the non-overshooting, continuously differentiable
[PCHIP implementation in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

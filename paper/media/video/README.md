# PLUME-Advanced Manim video project

This project renders scientific animation from saved, hash-bearing PLUME stage
artifacts. Rendering never invokes cave generation.

## Setup

Install the optional video dependencies:

```bash
.venv/bin/pip install -e '.[video]'
```

FFmpeg must be available on `PATH`. Manim may also require platform packages
for Cairo, Pango, and LaTeX depending on the scenes being rendered.

## Prepare one frozen hero run

```bash
.venv/bin/python scripts/prepare_manim_assets.py \
  --config config/project.toml \
  --body earth
```

The command runs only Stages A-C and writes frozen host-field JSON/NPZ,
network JSON, section JSON/NPZ, and a provenance manifest to
`paper/media/video/assets/hero/`. It
refuses to replace existing prepared inputs unless `--force-overwrite` is
provided.

## Render the prototype

Fast development render:

```bash
.venv/bin/python scripts/render_manim_video.py --quality low
```

1080p review render:

```bash
.venv/bin/python scripts/render_manim_video.py --quality high
```

All wrapper renders use 30 fps. A successful render writes a provenance JSON
beside the video with the Manim version, exact command, source hashes, selected
hero segment, and output hash. Render products and Manim's partial-movie cache
are local artifacts ignored by Git.

The `GraphToGeometryPrototype` scene tracks the same highlighted segment from
the Stage-B network through Stage-C profiles into a schematic Stage-D carved
volume. The rendered frame includes abbreviated semantic hashes for the input
network and section field.

Use `--hero-segment ID` to select a scientifically important branch or
junction after inspecting `stage_b_network.json`. Without the option, the
loader chooses the segment with the most Stage-C samples.

## Render standalone stage assets

```bash
.venv/bin/python scripts/render_manim_video.py --all-stages --quality high
```

This produces four independent videos:

- `StageAHostField.mp4` reveals the frozen host fields;
- `StageBSemanticFlow.mp4` draws directed segments according to propagated
  lava age;
- `StageCAdaptiveSections.mp4` provides a slower reading sequence along a long
  branch;
- `StageDMarchingCubes.mp4` explains profile-SDF stamping, voxel
  classification, Lewiner marching-cubes triangulation, and cross-chunk vertex
  welding.

An individual asset can be rendered with `--scene`, for example:

```bash
.venv/bin/python scripts/render_manim_video.py \
  --scene StageCAdaptiveSections \
  --quality high
```

## Design boundary

Manim owns explanatory animation, annotations, and result-plot composition.
The final textured cave fly-through remains a separate renderer shot. This
prototype intentionally represents Stage D schematically; it does not claim
to render the exported surface mesh.

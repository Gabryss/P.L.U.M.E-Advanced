# Scientific evaluation workflow

This directory freezes the declarations used to evaluate PLUME-Advanced for a
robotics paper. The generator is a deterministic, host-conditioned,
process-informed environment generator—not a thermofluid or structural-physics
simulator. Moon and Mars presets are controlled scenario extrapolations; PDC
supports terrestrial morphometric comparison only.

## Install and audit

```bash
uv sync --group dev --extra paper
uv run plume-evaluate --config paper/experiments.toml audit
export PLUME_PDC_ROOT=/absolute/path/to/PDC-v2
uv run plume-evaluate --config paper/experiments.toml pdc-audit
```

Manually inspect the PDC inventory and rejections. The frozen cave-level split
is documented in `paper/splits/README.md`; calibration may read only the 76-cave
calibration partition.

The bounded development pilot is:

```bash
uv run plume-evaluate --config paper/experiments.toml morphometry \
  --reference-partition calibration --max-seeds 3
```

Do not use `--reference-partition evaluation` while selecting parameters. The
default declared morphometry run uses the 19-cave confirmatory partition.

## Run

```bash
uv run plume-evaluate --config paper/experiments.toml morphometry
uv run plume-evaluate --config paper/experiments.toml controllability
uv run plume-evaluate --config paper/experiments.toml host-ablation
uv run plume-evaluate --config paper/experiments.toml sampling-ablation
uv run plume-evaluate --config paper/experiments.toml scalability
uv run plume-evaluate --config paper/experiments.toml export-consistency
uv run plume-evaluate --config paper/experiments.toml determinism
uv run plume-evaluate --config paper/experiments.toml aggregate
uv run plume-evaluate --config paper/experiments.toml figures
uv run plume-evaluate --config paper/experiments.toml latex
```

Each case has a stable run ID. Completed cases are skipped. Failed, timeout,
and invalid cases remain; `--force` archives the previous attempt. Figures
and LaTeX consume saved results and never rerun generation.

`plume-evaluate all` runs the complete declared campaign. It is intentionally
not a quick smoke test: the frozen seed sets include full-resolution geometry
and isolated scalability cases. Manual Blender, UE5, Unity, Gazebo, and
Omniverse imports remain a separate protocol in
`docs/paper/EXPORT_VERIFICATION_PROTOCOL.md`.

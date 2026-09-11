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

Each case has a stable run ID. A completed case is reused only when its
configuration, source contents, dependencies, input files, dataset identity,
and schema versions still match. Changed cases are recomputed and their old
attempts are archived, as are attempts replaced with `--force`. Legacy results
without a fingerprint are recomputed. Current campaign indexes exclude cases
from other fingerprints while preserving their case files.

Failed, timeout, and invalid cases remain visible. Experiment commands return
a nonzero exit code when cases fail, no cases complete, or reported checks
fail. Export consistency meshes the cave before packaging and cannot report
success for an empty or failed campaign. Figures and LaTeX consume saved
results and never rerun generation.

`plume-evaluate all` runs the complete declared campaign. It is intentionally
not a quick smoke test: the frozen seed sets include full-route geometry
and isolated scalability cases. Manual Blender, UE5, Unity, Gazebo, and
Omniverse imports remain a separate protocol in
`docs/paper/EXPORT_VERIFICATION_PROTOCOL.md`.

The standard Earth benchmark uses 0.6 m voxels and runs through final meshing
and welding. It excludes optional events, appearance and packaging. The declared
limits are 7,200 seconds and 12 GiB of monitored process-tree RSS per case.
This resolution is not a guarantee of narrow-passage fidelity; profile-resolution
warnings accompany the measurements.

Independent network/profile cases can use `PLUME_EVALUATION_WORKERS` on platforms
with process forking; the default is one. Resource benchmarks remain sequential.
Parameter sweeps resolve every stage from the changed flow-regime input, using
the same configuration loader as an ordinary project-file edit.

The [7 September 2026 campaign record](campaigns/2026-09-07/README.md) preserves
the source snapshots, configuration, dependencies, declared counts, evaluator
amendment and execution logs. Its driver retains completed cases on restart and
continues to other experiments when a suite reports failures. The report builder
requires all declared case records before producing the final campaign report.

The campaign is complete: **2,076 attempted, 2,071 completed, five dense 5 km
memory-limit failures**. The [results report](campaigns/2026-09-07/REPORT.md)
includes paired effects and negative findings. Source/configuration audits cover
all cases; the revised manuscript uses measured Tables IV–VI and Figures 9–10.

# Evaluation and simulator qualification

[← Project overview](../README.md)

## Seed campaigns

```bash
# Ten A-C cases, each with a separate cold reproducibility replay.
uv run plume-check \
  --configs config/short-single.toml config/short-multi.toml \
  --seeds 0 17 42 20260912 4294967295 --scope sections \
  --timeout 600 --memory-limit-mib 8192 --output outputs/section_campaign

# Full generation, material and export checks for one textured seed.
uv run plume-check \
  --configs config/short-multi.toml --seeds 17 --scope full \
  --timeout 7200 --memory-limit-mib 24576 --output outputs/full_campaign
```

Open the campaign's offline `report.html`. The JSON summary retains failures,
resource limits, repair decisions and replay results. Ten requested cases plus
cold replays means twenty worker runs. `--no-replay` is an explicit exploratory
tradeoff, recorded in the report. Workers run sequentially; memory and timeout
limits are not increased by repair.

```bash
uv run plume-check --output outputs/full_campaign --resume --timeout 10800
uv run plume-check --output outputs/full_campaign --resume --report-only
```

`--report-only` checks saved receipts without generating anything. A resumed
campaign verifies its original source/runtime/input identity; code or configuration
changes require a new campaign directory. A stage-only success says nothing about
meshing, floor contact, materials or native import.

## Development checks

Install the development dependencies with `uv sync --locked` before running:

```bash
uv run ruff check .
uv run mypy src/plume_advanced
uv run pytest -q
```

Tests include measured failure crops, synthetic geometry controls, deterministic
replays and policy failures. Optional native application tests require their
external tools; a skipped test is not evidence of engine compatibility.

## Native engine qualification

```bash
uv run python scripts/qualify_simulation.py \
  --config config/simulation-single.toml --seed 0 \
  --output outputs/simulation_trial \
  --unity /path/to/Unity --unreal /path/to/UnrealEditor
```

This source-checkout workflow runs numerical generation, a cold replay, native
material views and collision controls. Every supplied editor is mandatory.
**Only `ready/` is the promoted delivery**, created after all requested checks
pass. `qualification.json` explains a rejection; working results remain in
`generation/` and `native/`.

The maintained native harness targets Unity 6000.6 URP and Unreal 5.8 on Linux,
with GPU access. Unity's initial project setup also needs package-registry access
and a valid license. Its current input contract is one rock-free, untransformed
cave primitive with three 4K maps. The numerical API's `require_native = true`
remains unavailable; the separate qualifier supplies the native promotion gate.

To check an existing accepted 4K generation, including normal CLI
`export_blender/` or `export_all/blender/` output, run:

```bash
uv run python scripts/check_native_engines.py outputs/my_cave \
  --output outputs/my_cave_native \
  --unity /path/to/Unity --unreal /path/to/UnrealEditor
```

This imports the existing cave and records material and collision checks. It does
not perform the qualifier's cold replay or create a `ready/` delivery.

Use the checked native scene and its **separate static collider**. PLUME metadata
uses right-handed Z-up metres. The supplied glTFast adapter maps points to Unity
`(-x, z, -y)` metres; the Unreal adapter maps to `(100x, -100y, 100z)` centimetres.
The raw collider OBJ does not carry glTF conversion. Alternative importers require
their own axis/bounds check. Configure simulator gravity separately from importing
geometry. Nanite, collision reduction, texture streaming and compression require
performance validation in the target simulation.

## Scientific evaluation and reproducibility

The bundled [experiment declaration](../src/plume_advanced/evaluation/resources/experiments.toml) separates morphometry,
controllability, host/sampling ablations, scalability, export consistency and
reproducibility. Data analysis uses generated graph/profile measurements, not
screenshots as quantitative evidence.

| Reference | What it informs | What it cannot establish |
|---|---|---|
| [Pyroduct Digital Catalog v2](https://doi.org/10.5281/zenodo.17750755) | Terrestrial cross-section shape and size distributions | Full network topology or extraterrestrial calibration |
| [USGS/NASA TubeX Valentine Cave](https://doi.org/10.5066/P14AC3J5) | One cave's planform, local widening and surface comparisons | A universal morphology target; plan envelopes are not single-passage sections |

The [bundled PDC cave partitions](../src/plume_advanced/evaluation/resources/splits/) contain 76
calibration caves and 19 confirmatory caves. The split ranked SHA-256 of
`pdc-v2.0:20260831:<cave-id>` and reserved the first 19. Do not tune on the
confirmatory set. An exploratory whole-catalog summary predates the split, so it
is not claimed to have been historically unseen.

```bash
export PLUME_PDC_ROOT=/absolute/path/to/extracted/PDC-v2
uv run --extra paper plume-evaluate audit
uv run --extra paper plume-evaluate pdc-audit
uv run --extra paper plume-evaluate morphometry \
  --reference-partition calibration --max-seeds 3
```

External reference archives remain outside Git. The maintained loader retains
source identities and rejection reasons. Bundled experiments write results under
`outputs/evaluation/` in the working directory. Their research recipe resolves
relative assets as the public `config/research.toml` does: from `config/` in that
working directory, with rock maps under `texture/`. External textures are not
included in the Python wheel. The audit returns a failure and lists missing input
files when they are unavailable.

Pass `--config path/to/experiments.toml` before the subcommand to use a custom
declaration. Its paths resolve beside that file; the selected project recipe's
assets resolve beside the recipe. Optional `general.asset_directory` overrides
the base for relative texture and Rocky scratch paths without moving the recipe.
Experiment tables reject unknown keys, invalid types, unsupported choices and
nonfinite or out-of-range values before starting work.

Export evaluation checks every target's descriptor and visual bounds in metres.
Completed exports are reused only while the complete package still matches its
file hashes. Missing, modified or extra files trigger regeneration; the previous
attempt is retained, and a failed regeneration replaces its success claim.
`plume-evaluate --help` lists the full
experiment suite; the complete campaign is much more expensive than a smoke test.

A repeatable run requires the **seed, resolved configuration, source/material
resources, external inputs and dependency runtime**. The source identity includes
the shared preset catalog and shipped material adapters. Cold replays use a
separate process and a different Python hash seed. No finite test campaign proves
that every possible seed succeeds.

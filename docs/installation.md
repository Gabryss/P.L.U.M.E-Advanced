# Installation details

[← Project overview](../README.md)

## Recommended setup

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/Gabryss/P.L.U.M.E-Advanced.git
cd P.L.U.M.E-Advanced
uv sync --locked
```

PLUME requires Python 3.12 or newer. uv can download a compatible interpreter
when one is unavailable. The lockfile fixes the dependency versions and the
project is installed into `.venv`. No simulator or GPU is needed for the default
cave generation. The native simulator checks require their respective applications.

Continue with [Usage](../README.md#usage). Commands in the guides assume the
repository root is the working directory.

## What does uv do?

uv manages the Python environment. `uv run` runs PLUME inside that environment
and checks that its dependencies are up to date.

`--no-sync` is an **optional uv flag**, not a PLUME flag. It skips that environment
update and is not required for generation. The quickstart uses ordinary `uv run`
so a missing or outdated environment is handled automatically. CI uses `--no-sync` where the exact
installed environment is intentionally tested. See uv's
[running commands](https://docs.astral.sh/uv/concepts/projects/run/) and
[locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) guides.

## Optional dependencies

| Need | Setup |
|---|---|
| Core package only, without development tools | `uv sync --locked --no-dev`; run with `uv run --no-dev …` |
| Detailed loose-rock props | `uv run --extra rocks plume-generate …` |
| Scientific datasets and resource experiments | `uv run --extra paper plume-evaluate …` |
| All development and scientific tools | `uv sync --locked --all-extras` |
| Rock image textures | Follow [Textured caves](usage.md#textured-caves) |
| Native application imports | Follow [Simulators](simulators.md) |

Ellipses above stand for the arguments of your chosen workflow. Extras must also
be requested on `uv run` when that command needs them. Source texture masters are
stored in Git LFS; they are not bundled in the Python wheel.

The main dependencies are NumPy, SciPy, scikit-image and trimesh for geometry;
fast-simplification and xatlas for asset preparation; Pillow for images; and
Matplotlib and Rich for diagnostics. [pyproject.toml](../pyproject.toml) records
the complete dependency list and version constraints.

## Alternative: pip

With Python 3.12+ and an activated virtual environment, run `python -m pip install -e .`
from the checkout. You can then call `plume-generate` directly. This does not
reproduce the locked dependency set automatically.

## Clone without downloading texture masters

For a lightweight, untextured-only checkout, prefix the clone command with
`GIT_LFS_SKIP_SMUDGE=1`. Fetch textures later with `git lfs pull` if needed.

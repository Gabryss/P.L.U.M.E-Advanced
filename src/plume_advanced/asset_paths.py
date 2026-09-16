"""Locate delivered assets without importing generation or rendering dependencies."""

from pathlib import Path


def find_export_asset(
    run_directory: str | Path, *, target: str = "blender", filename: str | None = None,
) -> Path:
    """Find one asset in a single-target or all-target run; reject ambiguous exports."""
    root = Path(run_directory).resolve()
    directories = (root / f"export_{target}", root / "export_all" / target)
    candidates = sorted({
        path.resolve()
        for directory in directories
        for path in ([directory / filename] if filename else directory.glob("*.glb"))
        if path.is_file()
    })
    if not candidates:
        locations = ", ".join(str(directory) for directory in directories)
        raise FileNotFoundError(f"No {filename or 'GLB asset'} found in {locations}")
    if len(candidates) > 1:
        locations = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Ambiguous {target} exports; retain one matching asset: {locations}")
    return candidates[0]

"""Reproducibility manifest helpers for completed generation runs."""

from __future__ import annotations

from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Iterable

from config import ProjectConfig, project_config_manifest


def write_run_manifest(
    project_config: ProjectConfig,
    output_path: str | Path,
    *,
    outputs: Iterable[str | Path],
    elapsed_seconds: float,
    source_root: str | Path,
) -> Path:
    """Atomically write a completed-run manifest with hashes and versions."""

    root = Path(source_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "plume.run-manifest.v1",
        "status": "complete",
        "elapsed_seconds": float(elapsed_seconds),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "dependencies": {
            name: _package_version(name)
            for name in (
                "matplotlib",
                "numpy",
                "pillow",
                "rich",
                "scikit-image",
                "scipy",
                "trimesh",
            )
        },
        "source": {
            "revision": _git_value(root, "rev-parse", "HEAD"),
            "dirty": bool(_git_value(root, "status", "--porcelain")),
            "sha256": _source_hash(root),
        },
        "resolved_config": project_config_manifest(project_config),
        "outputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for path in sorted(
                {Path(value).resolve() for value in outputs if Path(value).is_file()},
                key=str,
            )
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(output)
    return output


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _git_value(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [root / "pyproject.toml", root / "config" / "project.toml"]
    candidates.extend(sorted((root / "src").rglob("*.py")))
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["write_run_manifest"]

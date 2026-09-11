"""Reproducibility manifest helpers for completed generation runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping

from plume_advanced.config import ProjectConfig, project_config_manifest


def write_run_manifest(
    project_config: ProjectConfig,
    output_path: str | Path,
    *,
    outputs: Iterable[str | Path],
    elapsed_seconds: float,
    source_root: str | Path,
    status: str = "complete",
    current_stage: str | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
    inputs: Iterable[str | Path] = (),
    timings: Mapping[str, float] | None = None,
) -> Path:
    """Atomically write a run manifest with hashes, versions, and status."""

    root = Path(source_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if status not in {"running", "complete", "failed"}:
        raise ValueError("status must be running, complete, or failed")
    payload = {
        "schema": "plume.run-manifest.v1",
        "status": status,
        "elapsed_seconds": float(elapsed_seconds),
        "timings": {
            **{name: float(value) for name, value in (timings or {}).items()},
            "total_s": float(elapsed_seconds),
        },
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
        "inputs": [
            _file_record(path, relative_to=output.parent)
            for path in sorted(
                {Path(value).resolve() for value in inputs if Path(value).is_file()},
                key=str,
            )
        ],
        "outputs": [
            _file_record(path, relative_to=output.parent)
            for path in sorted(
                {Path(value).resolve() for value in outputs if Path(value).is_file()},
                key=str,
            )
        ],
    }
    if current_stage:
        payload["current_stage"] = current_stage
    if status == "failed":
        payload["failure"] = {
            "stage": failed_stage or "unknown",
            "error": error or "unknown error",
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
    candidates = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / "config" / "project.toml",
    ]
    candidates.extend(sorted((root / "src").rglob("*.py")))
    candidates.extend(sorted((root / "scripts").rglob("*.py")))
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, str | int]:
    return {
        "path": os.path.relpath(path, relative_to),
        "bytes": path.stat().st_size,
        "sha256": _file_hash(path),
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["write_run_manifest"]

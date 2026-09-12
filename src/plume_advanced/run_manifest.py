"""Reproducibility manifest helpers for completed generation runs."""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from plume_advanced.config import ProjectConfig, project_config_manifest
from plume_advanced.identity import (
    dependency_versions,
    git_identity,
    package_source_hash,
    sha256_file,
)


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
        "dependencies": dependency_versions(),
        "source": {
            **git_identity(root),
            "sha256": package_source_hash(),
            "identity_schema": "plume.executing-package.v1",
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


def _file_record(path: Path, *, relative_to: Path) -> dict[str, str | int]:
    return {
        "path": os.path.relpath(path, relative_to),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


__all__ = ["write_run_manifest"]

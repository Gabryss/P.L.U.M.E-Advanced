"""Provenance and stable semantic-digest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from plume_advanced.identity import (
    dependency_versions,
    git_identity,
    package_source_hash,
    sha256_file,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def quantized_array(values: Any, tolerance: float = 1e-6) -> list[Any]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        array = np.rint(array.astype(float) / tolerance).astype(np.int64)
    return array.tolist()


def directory_identity(root: str | Path) -> dict[str, Any]:
    """Hash file paths and content without loading a dataset into memory."""

    data_root = Path(root)
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(candidate for candidate in data_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(data_root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
        file_count += 1
        byte_count += size
    return {
        "root": str(data_root.resolve()),
        "file_count": file_count,
        "byte_count": byte_count,
        "sha256": digest.hexdigest(),
    }


def capture_provenance(
    source_root: str | Path,
    *,
    resolved_config: Any | None = None,
    inputs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    root = Path(source_root)
    git = git_identity(root)
    config_hash = semantic_hash(resolved_config) if resolved_config is not None else ""
    provenance: dict[str, Any] = {
        "git_commit": git["revision"],
        "git_dirty": git["dirty"],
        "source_sha256": package_source_hash(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "resolved_config_sha256": config_hash,
        "dependencies": dependency_versions(),
        "inputs": [
            {
                "path": str(Path(path).resolve()),
                "bytes": Path(path).stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
            if Path(path).is_file()
        ],
        "thread_environment": {
            name: os.environ.get(name, "")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PLUME_EVALUATION_WORKERS",
            )
        },
    }
    provenance["identity_sha256"] = semantic_hash(provenance)
    return provenance


__all__ = [
    "canonical_json_bytes",
    "capture_provenance",
    "directory_identity",
    "quantized_array",
    "semantic_hash",
    "sha256_file",
]

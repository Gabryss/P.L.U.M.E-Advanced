"""Provenance and stable semantic-digest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    config_hash = semantic_hash(resolved_config) if resolved_config is not None else ""
    return {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "resolved_config_sha256": config_hash,
        "dependencies": {
            name: _version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-image",
                "trimesh",
                "matplotlib",
                "xatlas",
                "psutil",
            )
        },
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
            )
        },
    }


def _git(root: Path, *arguments: str) -> str:
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


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


__all__ = [
    "canonical_json_bytes",
    "capture_provenance",
    "directory_identity",
    "quantized_array",
    "semantic_hash",
    "sha256_file",
]

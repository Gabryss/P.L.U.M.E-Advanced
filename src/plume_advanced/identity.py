"""Shared source, runtime and file identities for generation and evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

DEPENDENCIES = (
    "numpy",
    "scipy",
    "scikit-image",
    "trimesh",
    "fast-simplification",
    "xatlas",
    "pillow",
    "rich",
    "matplotlib",
    "rocky",
    "laspy",
    "lazrs",
    "psutil",
)


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def runtime_identity() -> dict:
    return {
        "schema": "plume.runtime.v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "dependencies": dependency_versions(),
    }


def package_source_hash(package_root: Path | None = None) -> str:
    """Identify Python and shipped material resources independently of the CWD."""
    package = package_root if package_root is not None else Path(__file__).parent
    python_paths = set(package.rglob("*.py"))
    if not python_paths:
        raise ValueError(f"No Python source in package: {package}")
    resources = {
        path for path in (package / "material_assets").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    paths = sorted(python_paths | resources)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def git_identity(root: Path) -> dict[str, str | bool | None]:
    def query(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ("git", *arguments),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    revision = query("rev-parse", "HEAD")
    status = query("status", "--porcelain") if revision else None
    return {"revision": revision, "dirty": bool(status) if status is not None else None}


def identity_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

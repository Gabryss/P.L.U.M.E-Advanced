"""Recoverable staging-directory transactions for complete export packages."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic_output_directory(destination: Path) -> Iterator[Path]:
    """Yield a sibling staging directory and atomically publish it on success."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    backup: Path | None = None
    try:
        yield staging
        if destination.exists():
            if not destination.is_dir():
                raise NotADirectoryError(
                    f"Export destination exists and is not a directory: {destination}"
                )
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup-",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            destination.replace(backup)
        try:
            staging.replace(destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = ["atomic_output_directory"]

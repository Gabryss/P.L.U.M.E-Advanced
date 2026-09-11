"""Safeguards for generation directories that already contain artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, TextIO


class OutputOverwriteRefused(RuntimeError):
    """Raised when a generation is not allowed to overwrite existing output."""


def populated_output_directories(
    directories: Iterable[str | Path],
) -> tuple[tuple[Path, int], ...]:
    """Return unique non-empty output directories and their entry counts."""

    populated: list[tuple[Path, int]] = []
    paths = {
        Path(directory).expanduser().resolve()
        for directory in directories
    }
    for path in sorted(paths, key=str):
        if not path.is_dir():
            continue
        entry_count = sum(1 for _ in path.iterdir())
        if entry_count:
            populated.append((path, entry_count))
    return tuple(populated)


def require_output_overwrite_confirmation(
    directories: Iterable[str | Path],
    *,
    allow_overwrite: bool = False,
    interactive: bool | None = None,
    input_func: Callable[[str], str] = input,
    error_stream: TextIO | None = None,
) -> None:
    """Require explicit consent before writing into non-empty directories.

    ``allow_overwrite`` is intended for the project configuration and
    ``--force-overwrite`` CLI flag. Non-interactive executions fail safely
    unless that bypass is enabled.
    """

    populated = populated_output_directories(directories)
    if not populated or allow_overwrite:
        return

    if interactive is None:
        interactive = sys.stdin.isatty()
    stream = error_stream if error_stream is not None else sys.stderr
    details = ", ".join(
        f"{path} ({entry_count} entr{'y' if entry_count == 1 else 'ies'})"
        for path, entry_count in populated
    )

    if not interactive:
        raise OutputOverwriteRefused(
            "Refusing to overwrite non-empty generation output in a "
            f"non-interactive session: {details}. Pass --force-overwrite or set "
            "run.overwrite_outputs = true in the project configuration."
        )

    answer = input_func(
        "Generation output already exists and files may be overwritten:\n"
        f"  {details}\n"
        "Continue? [y/N] "
    )
    if answer.strip().lower() not in {"y", "yes"}:
        print("Generation cancelled; existing output was left unchanged.", file=stream)
        raise OutputOverwriteRefused("Output overwrite was not confirmed.")


__all__ = [
    "OutputOverwriteRefused",
    "populated_output_directories",
    "require_output_overwrite_confirmation",
]

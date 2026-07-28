#!/usr/bin/env python3
"""Generate the documentation figure gallery for each celestial body."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from plume_advanced.output_guard import (
    OutputOverwriteRefused,
    require_output_overwrite_confirmation,
)

BODIES = ("earth", "mars", "moon")
FIGURE_FILENAMES = (
    "stage_a_host_field.png",
    "stage_b_cave_network.png",
    "stage_c_section_field.png",
    "stage_c_floor_map.png",
    "stage_d_geometry.png",
    "stage_d_geometry_chunks.png",
    "stage_d_geometry_presentation.png",
    "stage_e_geological_events.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "project.toml",
        help="Base project configuration used for every body.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "celestial_bodies",
        help="Destination root for body-specific documentation figures.",
    )
    parser.add_argument(
        "--bodies",
        nargs="+",
        choices=BODIES,
        default=list(BODIES),
        help="Bodies to render in the requested order.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Refresh a non-empty documentation figure directory without prompting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_output_overwrite_confirmation(
            (args.docs_dir,),
            allow_overwrite=args.force_overwrite,
        )
    except OutputOverwriteRefused as error:
        print(error, file=sys.stderr)
        return 2

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plume-body-figures-") as temp_dir:
        workspace = Path(temp_dir)
        for body in args.bodies:
            run_output = workspace / body
            run_output.mkdir()
            command = (
                sys.executable,
                str(ROOT / "scripts" / "generate_cave.py"),
                "--config",
                str(args.config),
                "--body",
                body,
                "--output",
                str(run_output / "stage_b_cave_network.png"),
                "--force-overwrite",
            )
            print(f"Generating {body.title()} documentation figures...")
            subprocess.run(command, cwd=ROOT, check=True)

            body_destination = args.docs_dir / body
            body_destination.mkdir(parents=True, exist_ok=True)
            for filename in FIGURE_FILENAMES:
                source = run_output / filename
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Expected {body} documentation figure was not generated: "
                        f"{source}"
                    )
                destination = body_destination / filename
                if destination.exists():
                    destination.unlink()
                shutil.move(str(source), destination)
            print(
                f"Moved {len(FIGURE_FILENAMES)} {body.title()} figures to "
                f"{body_destination}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

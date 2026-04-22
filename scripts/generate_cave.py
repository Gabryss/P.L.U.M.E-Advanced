#!/usr/bin/env python3
"""Single entrypoint for generating the current cave-network output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(tempfile.gettempdir()) / "plume-advanced-cache"
MPL_CACHE = CACHE_ROOT / "matplotlib"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
sys.path.insert(0, str(ROOT / "src"))

from config import load_project_config
from stages.host_field import HostFieldGenerator
from stages.network import CaveNetworkGenerator
from stages.section_field import SectionFieldGenerator
from visualization.host_field import HostFieldPlotter
from visualization.network import CaveNetworkPlotter
from visualization.section_field import SectionFieldPlotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "project.toml",
        help="Path to the project TOML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "stage_b_cave_network.png",
        help="Path for the generated cave-network visualization.",
    )
    parser.add_argument(
        "--host-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated host-field visualization. "
            "Defaults to a sibling file named stage_a_host_field.png."
        ),
    )
    parser.add_argument(
        "--section-output",
        type=Path,
        default=None,
        help=(
            "Path for the generated section-field visualization. "
            "Defaults to a sibling file named stage_c_section_field.png."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_config = load_project_config(args.config)
    host_output = args.host_output or args.output.with_name("stage_a_host_field.png")
    section_output = args.section_output or args.output.with_name("stage_c_section_field.png")

    host_field = HostFieldGenerator(project_config.host_field).generate()
    host_output_path = HostFieldPlotter().render(host_field, host_output)
    cave_network = CaveNetworkGenerator(project_config.network).generate(host_field)
    network_output_path = CaveNetworkPlotter().render(host_field, cave_network, args.output)
    section_field = SectionFieldGenerator(project_config.section_field).generate(cave_network)
    section_output_path = SectionFieldPlotter().render(
        cave_network,
        section_field,
        section_output,
    )

    print("Generated cave pipeline artifacts.")
    print(f"Configuration: {args.config}")
    print(f"Stage A visualization: {host_output_path}")
    print(f"Stage B visualization: {network_output_path}")
    print(f"Stage C visualization: {section_output_path}")
    for key, value in host_field.summary().items():
        print(f"host_{key}: {value:.3f}")
    for key, value in cave_network.summary().items():
        print(f"network_{key}: {value:.3f}")
    for key, value in section_field.summary().items():
        print(f"section_{key}: {value:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

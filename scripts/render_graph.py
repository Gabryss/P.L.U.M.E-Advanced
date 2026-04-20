#!/usr/bin/env python3
"""Generate and visualize the stage-B trunk graph."""

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
from stages import HostFieldGenerator, TrunkGraphGenerator
from visualization import TrunkGraphPlotter


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
        default=ROOT / "outputs" / "stage_b_trunk_graph.png",
        help="Path for the generated visualization image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_config = load_project_config(args.config)

    host_field = HostFieldGenerator(project_config.host_field).generate()
    trunk_graph = TrunkGraphGenerator(project_config.graph).generate(host_field)
    output_path = TrunkGraphPlotter().render(host_field, trunk_graph, args.output)

    print("Generated stage B trunk graph.")
    print(f"Visualization: {output_path}")
    for key, value in trunk_graph.summary().items():
        print(f"{key}: {value:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

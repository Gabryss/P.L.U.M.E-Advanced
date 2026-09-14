#!/usr/bin/env python3
"""Package continuous Blender/Unity/Unreal materials for an existing textured GLB."""

from __future__ import annotations

import argparse
from pathlib import Path

from plume_advanced.exporters.projected_materials import write_projected_material_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size-m", type=float, required=True)
    parser.add_argument("--normal-strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new material bundle directory")
    paths = write_projected_material_bundle(
        args.asset, args.output, tile_size_m=args.tile_size_m, normal_strength=args.normal_strength
    )
    if not paths:
        parser.error("Source needs all three cave PBR images")
    print(f"Created {len(paths)} material files in {args.output}; source GLB unchanged.")


if __name__ == "__main__":
    main()

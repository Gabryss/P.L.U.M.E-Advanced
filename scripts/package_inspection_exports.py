#!/usr/bin/env python3
"""Package one completed, rock-free Blender export for engine inspection.

Copies the already prepared portable GLB, so target folders cannot drift into
different geometry realizations. Generation remains the responsibility of the
normal plume-generate command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from plume_advanced.exporters.atomic import atomic_output_directory
from plume_advanced.exporters.targets import (
    ExportResult,
    _write_engine_import_guide,
    _write_target_descriptor,
)
from plume_advanced.validation import GlbAsset
from plume_advanced.world import ExportConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    run_manifest = json.loads((root / "run_manifest.json").read_text())
    if run_manifest["status"] != "complete":
        raise ValueError("Finish generation before packaging inspection exports")
    assets = list((root / "export_blender").glob("*.glb"))
    if len(assets) != 1:
        raise ValueError("Expected one completed Blender GLB")
    source = assets[0]
    document = GlbAsset(source).document
    mesh_nodes = [n for n in document.get("nodes", []) if "mesh" in n]
    if len(mesh_nodes) != 1 or mesh_nodes[0].get("name") != "cave_wall":
        raise ValueError("Inspection asset must contain only the cave_wall mesh")
    manifest = source.with_suffix(".manifest.json")
    geometry_manifest = json.loads(manifest.read_text())
    if geometry_manifest.get("events") or geometry_manifest["summary"]["event_mesh_count"]:
        raise ValueError("Inspection asset unexpectedly contains events or props")
    with source.open("rb") as file:
        source_hash = hashlib.file_digest(file, "sha256").hexdigest()
    records = [{"target": "blender", "asset": str(source.relative_to(root))}]
    for target in ("unity", "ue5"):
        destination = root / f"export_{target}"
        config = ExportConfig(target=target, file_format="glb", generate_collision=False)
        with atomic_output_directory(destination) as staging:
            asset = staging / source.name
            shutil.copy2(source, asset)
            shutil.copy2(manifest, staging / manifest.name)
            result = ExportResult(target=target, primary_asset=asset, files=(asset,))
            _write_target_descriptor(result, config, staging, source.stem)
            guide = _write_engine_import_guide(target, staging, asset, collision_asset=None)
            if guide:
                with guide.open("a") as file:
                    file.write("\nThis inspection package contains one complete cave and no rocks.\n"
                               "Start with editor fly navigation; a separate collider is not supplied.\n"
                               "All three application GLBs are byte-identical.\n")
            with asset.open("rb") as file:
                assert hashlib.file_digest(file, "sha256").hexdigest() == source_hash
        records.append({"target": target, "asset": str((destination / source.name).relative_to(root))})
    report = {
        "schema": "plume.inspection-packages.v1",
        "generation_count": 1,
        "source_run_manifest": "run_manifest.json",
        "geometry_sha256": source_hash,
        "bytes_per_glb": source.stat().st_size,
        "mesh_nodes": [n["name"] for n in mesh_nodes],
        "rock_mesh_count": 0,
        "separate_collision_mesh": False,
        "packages": records,
        "scope": "Identical portable assets; destination application imports are checked separately.",
    }
    (root / "inspection_packages.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

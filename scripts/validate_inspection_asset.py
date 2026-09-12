#!/usr/bin/env python3
"""Check a neutral full-pipeline inspection GLB without requiring texture maps.

Uses the public portable-asset protocol, including UV checks, with a neutral
material profile. It does not claim geological validity or route-wide clearance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plume_advanced.validation import (
    PortableAssetValidator,
    ValidationCheck,
    write_validation_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    asset = next((root / "export_blender").glob("*.glb"))
    validator = PortableAssetValidator(asset, run_manifest_path=root / "run_manifest.json", material_profile="neutral")
    checks = validator.validate(progress=lambda step, total, detail: print(detail, flush=True))
    document = validator.glb.document
    mesh_nodes = [n.get("name") for n in document.get("nodes", []) if "mesh" in n]
    checks.append(ValidationCheck("scene", "Only cave wall; no rocks", mesh_nodes == ["cave_wall"],
                                  f"mesh_nodes={mesh_nodes}"))
    # Neutral materials intentionally have no map bindings. The common
    # geometry, shading, UV and provenance checks still apply.
    output = root / "validation"
    write_validation_reports(asset, checks, output)
    scope = {
        "included": ["container", "topology", "normals and tangents", "UV distortion/seam metrics", "scene and no-rock check",
                     "displacement metadata", "generation output hashes"],
        "excluded": ["texture map presence", "geological accuracy",
                     "route-wide clearance", "application imports"],
    }
    (output / "validation_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    failed = [c.name for c in checks if not c.passed]
    print(json.dumps({"passed": len(checks) - len(failed), "failed": failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write a neutral USD scene with partitioned static collision from adapter arrays.

Run with Isaac Sim's python.sh. Every face is used once in the visual and once
across the collider chunks; no geometry is synthesized by this step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=500_000)
    args, _ = parser.parse_known_args()
    if args.output.suffix != ".usdc" or args.max_faces < 1:
        parser.error("Use a .usdc output and a positive --max-faces")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import numpy as np
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

        with np.load(args.arrays) as archive:
            points = np.asarray(archive["points"], dtype=np.float32)
            faces = np.asarray(archive["faces"], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Expected triangular 3D adapter arrays")
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("Adapter face indices are out of range")
        stage = Usd.Stage.CreateNew(str(output))
        if stage is None:
            raise RuntimeError("Could not create USD stage")
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/PLUME_Cave")
        stage.SetDefaultPrim(root.GetPrim())
        wall = UsdGeom.Mesh.Define(stage, "/PLUME_Cave/CaveWall")
        wall.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        wall.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        wall.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        wall.CreateSubdivisionSchemeAttr("none")
        material = UsdShade.Material.Define(stage, "/PLUME_Cave/NeutralRock")
        shader = UsdShade.Shader.Define(stage, "/PLUME_Cave/NeutralRock/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(.52, .49, .45))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.85)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(material)
        source = stage.DefinePrim("/PLUME_Cave/CaveCollision", "Xform")
        source.SetActive(False)
        chunks = []
        for chunk_id, start in enumerate(range(0, len(faces), args.max_faces)):
            stop = min(start + args.max_faces, len(faces))
            used, inverse = np.unique(faces[start:stop].reshape(-1), return_inverse=True)
            part = UsdGeom.Mesh.Define(
                stage, f"/PLUME_Cave/CaveCollisionChunk{chunk_id:04d}")
            part.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[used]))
            part.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(np.full(stop-start, 3, dtype=np.int32)))
            part.CreateFaceVertexIndicesAttr(
                Vt.IntArray.FromNumpy(inverse.astype(np.int32)))
            part.CreateSubdivisionSchemeAttr("none")
            part.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(part.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr("none")
            chunks.append({"path": str(part.GetPath()), "faces": stop-start,
                           "vertices": len(used)})
            print(f"Collision chunk {chunk_id+1}: {stop-start:,} faces", flush=True)
        stage.GetRootLayer().Save()
        receipt = {
            "schema": "plume.neutral-usd-adapter.v1",
            "scope": "Derived neutral USD using the inspected adapter arrays; not the original textured five-target export.",
            "arrays_sha256": digest(args.arrays), "usd_sha256": digest(output),
            "visual_faces": len(faces), "collision_faces": sum(row["faces"] for row in chunks),
            "collision_chunks": chunks, "max_faces_per_chunk": args.max_faces,
            "meters_per_unit": 1., "up_axis": "Z",
        }
        if receipt["visual_faces"] != receipt["collision_faces"]:
            raise RuntimeError("USD collision partition lost or duplicated faces")
        output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "collision_chunks"},
                         indent=2), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()

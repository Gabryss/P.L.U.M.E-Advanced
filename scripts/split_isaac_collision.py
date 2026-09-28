#!/usr/bin/env python3
"""Partition one oversized static USD collider without changing its triangles.

Run with Isaac Sim's python.sh. The output is a stronger USD layer over the
original exported stage: it disables the one-piece collider and enables
smaller static triangle meshes. The source visual mesh and materials remain
referenced, not copied or altered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="Original PLUME USD stage")
    parser.add_argument("--output", type=Path, required=True, help="Derived .usdc stage")
    parser.add_argument("--max-faces", type=int, default=500_000)
    args, _ = parser.parse_known_args()
    if args.max_faces < 1 or args.output.suffix != ".usdc":
        parser.error("--max-faces must be positive and --output must end in .usdc")

    source = args.asset.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics, Vt

        original = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
        if original is None:
            raise RuntimeError("Cannot open original USD stage")
        mesh = UsdGeom.Mesh(original.GetPrimAtPath("/PLUME_Cave/CaveCollision"))
        if not mesh:
            raise RuntimeError("Source collider /PLUME_Cave/CaveCollision is missing")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if not np.all(counts == 3) or len(indices) != 3 * len(counts):
            raise ValueError("Only triangular collider meshes are supported")
        if indices.min() < 0 or indices.max() >= len(points):
            raise ValueError("Collider face indices are out of range")

        derived = Usd.Stage.CreateNew(str(output))
        if derived is None:
            raise RuntimeError("Cannot create derived USD stage")
        derived.GetRootLayer().subLayerPaths.append(os.path.relpath(source, output.parent))
        UsdGeom.SetStageMetersPerUnit(derived, UsdGeom.GetStageMetersPerUnit(original))
        UsdGeom.SetStageUpAxis(derived, UsdGeom.GetStageUpAxis(original))
        if original.GetDefaultPrim():
            derived.SetDefaultPrim(derived.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
        override = derived.OverridePrim("/PLUME_Cave/CaveCollision")
        # A disabled collision API is still cooked by Isaac/PhysX on stage load.
        # Inactivate the source prim so only the partitioned siblings enter physics.
        override.SetActive(False)
        UsdPhysics.CollisionAPI.Apply(override).CreateCollisionEnabledAttr(False)
        chunks = []
        for chunk_id, start in enumerate(range(0, len(counts), args.max_faces)):
            end = min(start + args.max_faces, len(counts))
            block = indices[3 * start:3 * end]
            used, remapped = np.unique(block, return_inverse=True)
            part = UsdGeom.Mesh.Define(derived, f"/PLUME_Cave/CaveCollisionChunk{chunk_id:04d}")
            part.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[used]))
            part.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(np.full(end - start, 3, dtype=np.int32))
            )
            part.CreateFaceVertexIndicesAttr(
                Vt.IntArray.FromNumpy(remapped.astype(np.int32, copy=False))
            )
            part.CreateSubdivisionSchemeAttr("none")
            part.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(part.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr("none")
            chunks.append({"prim": str(part.GetPath()), "triangles": end - start,
                           "vertices": len(used)})
            print(f"PLUME: collider chunk {chunk_id + 1}: {end - start} triangles", flush=True)
        derived.GetRootLayer().Save()
        receipt = {
            "schema": "plume.isaac-collider-partition.v1",
            "source_asset": str(source), "source_sha256": digest(source),
            "derived_asset": str(output), "derived_sha256": digest(output),
            "source_triangles": len(counts), "partition_triangles": sum(c["triangles"] for c in chunks),
            "chunk_count": len(chunks), "max_faces_per_chunk": args.max_faces,
            "chunks": chunks,
            "geometry_statement": "Every original triangle occurs once; vertices are only locally reindexed.",
        }
        if receipt["source_triangles"] != receipt["partition_triangles"]:
            raise RuntimeError("Partition lost or duplicated faces")
        output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "chunks"}, indent=2), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()

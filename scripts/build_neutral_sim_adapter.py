#!/usr/bin/env python3
"""Build an explicitly derived neutral Gazebo package from accepted PLUME geometry.

This bypasses the failed texture-atlas export. The saved float32 mesh is
reinspected under the same ground-route contract before it is serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import struct
from pathlib import Path

import numpy as np

from plume_advanced.exporters.collision import surface_deviation
from plume_advanced.stages.geometry_export import (
    _orient_faces_toward_cave_interior,
    _smooth_visual_surface,
)
from plume_advanced.stages.mesh_inspection import inspect_surface, route_inspection_arguments


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    if len(faces) >= 2**32:
        raise ValueError("Binary STL face count exceeds uint32")
    record = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)),
                       ("attribute", "<u2")])
    with path.open("wb") as output:
        output.write(b"PLUME accepted geometry; neutral simulator adapter".ljust(80, b"\0"))
        output.write(struct.pack("<I", len(faces)))
        for start in range(0, len(faces), 50_000):
            triangle = vertices[faces[start:start + 50_000]]
            normal = np.cross(triangle[:, 1]-triangle[:, 0],
                              triangle[:, 2]-triangle[:, 0])
            length = np.linalg.norm(normal, axis=1)
            normal /= length[:, None]
            block = np.zeros(len(triangle), dtype=record)
            block["normal"] = normal
            block["corners"] = triangle
            output.write(block.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Adapter needs a new output directory")
    output.mkdir(parents=True)
    with args.checkpoint.open("rb") as stream:
        geometry = pickle.load(stream)  # noqa: S301 - locally generated checkpoint only
    raw_vertices = np.asarray(geometry.assembled_vertices, dtype=np.float64)
    faces = np.asarray(geometry.assembled_faces, dtype=np.int32)
    print(f"Loaded {len(raw_vertices):,} vertices and {len(faces):,} faces", flush=True)
    smoothed = _smooth_visual_surface(
        raw_vertices, faces, iterations=2,
        variation_seed=geometry.config.random_seed,
        roughness_frequency=geometry.config.wall_roughness_frequency,
    ).astype(np.float32)
    faces = _orient_faces_toward_cave_interior(smoothed.astype(np.float64), faces,
                                                geometry.route_centers).astype(np.int32)
    arguments = dict(points=geometry.route_centers,
                     expected_genus=geometry.expected_surface_genus,
                     require_centers=True,
                     protected_points=geometry.protected_route_points,
                     **route_inspection_arguments(geometry))
    checked = inspect_surface(smoothed, faces, **arguments)
    deviation = surface_deviation((raw_vertices, np.asarray(geometry.assembled_faces, dtype=np.int32)),
                                  (smoothed, faces), .013, purpose="Neutral adapter")
    print(json.dumps({"inspection_passed": checked["passed"],
                      "inspection_failures": checked["failures"],
                      "deviation": deviation}, indent=2), flush=True)
    if not checked["passed"] or not deviation["passed"]:
        raise RuntimeError("Derived float32 mesh failed the declared route or 13 mm surface checks")
    print("Derived float32 mesh passed full route and surface checks", flush=True)
    mesh_file = output / "cave_mesh.npz"
    np.savez(mesh_file, points=smoothed, faces=faces)
    package = output / "gazebo" / "plume_simulation_cave"
    mesh_dir = package / "meshes"
    mesh_dir.mkdir(parents=True)
    stl = mesh_dir / "cave.stl"
    write_stl(stl, smoothed, faces)
    model_uri = "model://plume_simulation_cave/meshes/cave.stl"
    (package / "model.config").write_text(
        '<model><name>plume_simulation_cave</name><version>1.0</version>'
        '<sdf version="1.10">model.sdf</sdf></model>\n'
    )
    (package / "model.sdf").write_text(f'''<sdf version="1.10"><model name="plume_simulation_cave">
      <static>true</static><link name="cave">
        <collision name="cave_collision"><geometry><mesh><uri>{model_uri}</uri></mesh></geometry></collision>
        <visual name="cave_visual"><geometry><mesh><uri>{model_uri}</uri></mesh></geometry>
          <material><ambient>0.18 0.17 0.16 1</ambient><diffuse>0.30 0.29 0.27 1</diffuse>
          <specular>0.06 0.06 0.06 1</specular></material></visual>
      </link></model></sdf>\n''')
    route = checked["ground_traversal"]["paths"][0]
    receipt = {
        "schema": "plume.neutral-sim-adapter.v1",
        "scope": "Derived neutral visual/collision asset from an accepted PLUME checkpoint; original five-target textured export failed.",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": digest(args.checkpoint),
        "smoothing_iterations": 2,
        "visual_max_error_m": .013,
        "surface_deviation": deviation,
        "source_faces": len(faces),
        "derived_faces": len(faces),
        "float32_zero_area_faces": 0,
        "topology": checked["topology"],
        "route": {
            "passed": route["passed"], "samples": route["samples"],
            "failed_stations": len(route["failed_stations"]),
            "failed_edges": len(route["failed_edges"]),
            "robot": checked["ground_traversal"]["robot"],
        },
        "mesh_npz_sha256": digest(mesh_file), "gazebo_stl_sha256": digest(stl),
        "gazebo_model_sha256": digest(package / "model.sdf"),
    }
    (output / "adapter_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("topology", "surface_deviation")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Mesh export helpers for Stage D geometry review."""

from __future__ import annotations

from pathlib import Path

import trimesh

from stages.geometry_types import CaveGeometry


def export_geometry_obj(cave_geometry: CaveGeometry, output_path: str | Path) -> Path:
    """Write the assembled Stage-D mesh as a Wavefront OBJ file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        handle.write("# PLUME-Advanced Stage D geometry export\n")
        for key, value in cave_geometry.summary().items():
            handle.write(f"# {key}={value:.3f}\n")
        mesh = trimesh.Trimesh(
            vertices=cave_geometry.assembled_vertices,
            faces=cave_geometry.assembled_faces,
            process=False,
        )
        handle.write(f"# trimesh_is_watertight={float(mesh.is_watertight):.3f}\n")
        handle.write(f"# trimesh_euler_number={float(mesh.euler_number):.3f}\n")
        handle.write("o cave_wall\n")
        for vertex in cave_geometry.assembled_vertices:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in cave_geometry.assembled_faces:
            a, b, c = (index + 1 for index in face)
            handle.write(f"f {a} {b} {c}\n")
        vertex_offset = len(cave_geometry.assembled_vertices)
        for event_mesh in cave_geometry.event_meshes:
            handle.write(f"\no event_{event_mesh.event_id:04d}_{event_mesh.kind}\n")
            handle.write(f"# material_hint={event_mesh.material_hint}\n")
            for vertex in event_mesh.vertices:
                handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
            for face in event_mesh.faces:
                a, b, c = (index + vertex_offset + 1 for index in face)
                handle.write(f"f {a} {b} {c}\n")
            vertex_offset += len(event_mesh.vertices)

    return output

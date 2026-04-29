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
        handle.write(mesh.export(file_type="obj"))

    return output

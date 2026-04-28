"""Mesh export helpers for Stage D geometry review."""

from __future__ import annotations

from pathlib import Path

from stages.geometry_types import CaveGeometry


def export_geometry_obj(cave_geometry: CaveGeometry, output_path: str | Path) -> Path:
    """Write the assembled Stage-D mesh as a Wavefront OBJ file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        handle.write("# PLUME-Advanced Stage D geometry export\n")
        for key, value in cave_geometry.summary().items():
            handle.write(f"# {key}={value:.3f}\n")
        for x_coord, y_coord, z_coord in cave_geometry.assembled_vertices:
            handle.write(f"v {x_coord:.6f} {y_coord:.6f} {z_coord:.6f}\n")
        for a, b, c in cave_geometry.assembled_faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")

    return output

#!/usr/bin/env python3
"""Generate and export an untextured cave surface for geometry inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    export_geometry_report,
    export_network_artifact,
    export_section_artifact,
)
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_export import _smooth_visual_surface, density_surface_normals
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional local geometry checkpoint for inspection")
    args = parser.parse_args()
    config = load_project_config(args.config)
    if config.events.enabled or config.events.include_rock_props:
        parser.error("Use a configuration with events and rock props disabled")
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_update = 0.0

    def progress(stage: str, current: int, total: int, message: str) -> None:
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= 10 or current == total:
            print(f"{stage}: {current}/{total} {message}", flush=True)
            last_update = now

    host = HostFieldGenerator(config.host_field).generate()
    network = CaveNetworkGenerator(config.network).generate(
        host, section_config=config.section_field,
        quality_report_path=out / "network_quality_report.json",
        quality_progress=lambda message: print(message, flush=True),
    )
    sections = SectionFieldGenerator(config.section_field).generate(network)
    export_network_artifact(network, out / "network.json")
    export_section_artifact(sections, out / "sections.npz")
    resolution = section_resolution_report(sections, config.geometry.voxel_size)
    (out / "resolution_checks.json").write_text(json.dumps(resolution, indent=2) + "\n")
    if resolution["under_resolved_count"]:
        print(f"Resolution: {resolution['under_resolved_count']} shallow or narrow sections "
              "need local refinement before using their clearances.", flush=True)
    geometry = GeometryGenerator(config.geometry).generate(network, sections, progress=progress)
    export_geometry_report(geometry, out / "geometry_report.json")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_bytes(pickle.dumps((network, sections, geometry)))
    (out / "resolved_config.json").write_text(json.dumps(asdict(config), default=str, indent=2) + "\n")
    export_surface(geometry, out, elapsed_before_export=time.monotonic()-started)


def export_surface(geometry, out: Path, *, elapsed_before_export: float = 0.) -> None:
    """Export and reload-check a completed geometry, without repeating generation."""
    started = time.monotonic()
    vertices = np.asarray(geometry.assembled_vertices, dtype=np.float64)
    faces = np.asarray(geometry.assembled_faces, dtype=np.int64)
    if not len(faces):
        raise RuntimeError("No open cave surface survived stability screening")
    vertices = _smooth_visual_surface(vertices, faces, iterations=geometry.config.cave_smoothing_iterations)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.vertex_normals = density_surface_normals(
        geometry.voxel_grid, vertices, mesh.vertex_normals,
        filter_voxels=geometry.config.surface_normal_filter_voxels,
    )
    mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name="Neutral cave surface", baseColorFactor=[150,143,130,255],
        metallicFactor=0., roughnessFactor=.95, doubleSided=True,
    ))
    mesh.metadata = {"kind": "cave_wall", "units": "metres", "rock_props": 0}
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="cave_wall", geom_name="cave_wall")
    path = out / "lava_tube_geometry.glb"
    scene.export(path)
    loaded = trimesh.load(path, force="scene", process=False)
    parts = list(loaded.geometry.values())
    if len(parts) != 1:
        raise RuntimeError("Tube-only export must contain exactly one mesh")
    part = parts[0]
    part.merge_vertices()
    checks = {
        "watertight": bool(part.is_watertight),
        "consistent_winding": bool(part.is_winding_consistent),
        "finite_vertices": bool(np.isfinite(part.vertices).all()),
        "nondegenerate_faces": bool(np.all(part.area_faces > 0.)),
        "single_cave_mesh": len(parts) == 1,
        "no_rocks": not geometry.event_meshes,
        "finite_unit_normals": bool(np.isfinite(part.vertex_normals).all()
                                    and np.allclose(np.linalg.norm(part.vertex_normals, axis=1), 1., atol=2e-5)),
    }
    report = {
        "checks": checks, "vertices": len(part.vertices), "triangles": len(part.faces),
        "voxel_size_m": geometry.config.voxel_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "elapsed_seconds": elapsed_before_export + time.monotonic() - started,
        "scope": "Geometry and export checks; visual review is separate from geological validation.",
    }
    (out / "export_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError("Export validation failed; inspect export_checks.json")


if __name__ == "__main__":
    main()

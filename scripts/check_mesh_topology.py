#!/usr/bin/env python3
"""Compare graph cycles, voxel plan islands and raw/exported mesh topology.

Reads trusted local checkpoints written by the normal PLUME pipeline. Never
use Python pickle checkpoints received from an untrusted source.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_fill_holes, label


def mesh_metrics(mesh):
    mesh.remove_unreferenced_vertices()
    components = len(mesh.split(only_watertight=False))
    return dict(
        vertices=len(mesh.vertices),
        triangles=len(mesh.faces),
        components=components,
        watertight=bool(mesh.is_watertight),
        consistent_winding=bool(mesh.is_winding_consistent),
        euler_number=int(mesh.euler_number),
        genus=int((2 - mesh.euler_number) // 2)
        if components == 1 and mesh.is_watertight and mesh.is_winding_consistent
        else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory
    with (root / ".plume-checkpoints/final_geometry.pickle").open("rb") as stream:
        geometry = pickle.load(stream)
    network = json.loads((root / "stage_b_network.json").read_text())
    grid = geometry.voxel_grid
    if hasattr(grid, "density"):
        plan = np.any(grid.density >= grid.iso_level, axis=2)
    else:
        plan = np.zeros(grid.shape[:2], dtype=bool)
        for key, tile in sorted(grid.tiles.items()):
            start = np.array(key[:2]) * grid.tile_size
            size = tile.shape[:2]
            plan[start[0] : start[0] + size[0], start[1] : start[1] + size[1]] |= np.any(
                tile >= grid.iso_level, axis=2
            )
    holes, count = label(binary_fill_holes(plan) & ~plan)
    areas = sorted((np.bincount(holes.ravel())[1:] * grid.voxel_size**2).tolist(), reverse=True)
    expected = (
        len(network["segments"]) - len(network["nodes"]) + 1
    )  # The acceptance gate requires one connected graph.
    quality = json.loads((root / "network_quality_report.json").read_text())
    width = 2 * quality["generation_config"]["base_passage_radius"]
    area_threshold = 0.08 * width**2  # Same significance cutoff as the section footprint check.
    raw = trimesh.Trimesh(
        vertices=np.array(geometry.assembled_vertices),
        faces=np.array(geometry.assembled_faces),
        process=False,
    )
    raw_stats = mesh_metrics(raw)
    asset = next((root / "export_blender").glob("*.glb"))
    scene = trimesh.load(asset, force="scene", process=False)
    exported = next(iter(scene.geometry.values())).copy()
    # Export duplicates positions along normal/texture seams. Reconstruct
    # connectivity on positions only; no change is written back to the asset.
    exported.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
    export_stats = mesh_metrics(exported)
    significant = sum(a >= area_threshold for a in areas)
    with np.load(root / "stage_c_sections.npz") as sections:
        offsets = sections["profile_offsets"]
        profile_points = sections["profile_points"]
    spans = [
        np.ptp(p, axis=0).min()
        for p in (profile_points[a:b] for a, b in zip(offsets[:-1], offsets[1:]))
    ]
    report = dict(
        scope="One complete generated cave. Voxel plan projection and mesh topology; not a continuous clearance survey.",
        graph_cycles=expected,
        raw_mesh=raw_stats,
        exported_mesh_after_seam_weld=export_stats,
        voxel_size_m=grid.voxel_size,
        voxel_plan_components=int(label(plan)[1]),
        voxel_plan_hole_areas_m2=areas,
        significant_hole_area_threshold_m2=area_threshold,
        significant_voxel_plan_islands=significant,
        macro_islands_match_graph=significant == expected,
        surface_genus_matches_graph=raw_stats["genus"] == export_stats["genus"] == expected,
        section_count=len(spans),
        sections_with_minimum_dimension_under_8_voxels=int(
            sum(v < 8 * grid.voxel_size for v in spans)
        ),
    )
    report["passed"] = bool(
        report["macro_islands_match_graph"]
        and report["surface_genus_matches_graph"]
        and all(
            m["components"] == 1 and m["watertight"] and m["consistent_winding"]
            for m in (raw_stats, export_stats)
        )
    )
    (root / "mesh_topology_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

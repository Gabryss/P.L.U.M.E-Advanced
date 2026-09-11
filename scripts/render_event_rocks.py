#!/usr/bin/env python3
"""Render actual exported cave and Rocky meshes for the paper's event illustration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from render_tube_views import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sections", type=Path, default=Path("outputs/earth_inspection_seed4/sections.npz"))
    parser.add_argument("--candidate-count", type=int, default=3)
    args = parser.parse_args()
    out = args.directory
    study = json.loads((out / "study.json").read_text())
    props = [e for e in study["events"] if e["kind"] in {"rock", "boulder"}]
    structures = [e for e in study["events"] if e["kind"] not in {"rock", "boulder"}]
    positions = np.array([[e["x"], e["y"], e["z"]] for e in props])
    samples = np.load(args.sections)
    cameras = []
    # Prefer boulder families away from volume edits, with an upstream camera on
    # the same segment. Selection is for readable illustration, not inference.
    candidates = []
    for event in props:
        if event["kind"] != "boulder":
            continue
        point = np.array([event["x"], event["y"], event["z"]])
        if any(np.linalg.norm(point[:2]-[e["x"], e["y"]]) < 2*max(e["radius_x"], e["radius_y"])+8 for e in structures):
            continue
        section = study["prop_source_sections"][str(event["event_id"])]
        indices = np.flatnonzero(samples["segment_id"] == event["segment_id"])
        delta = section["segment_arc_length"] - samples["arc_length_m"][indices]
        upstream = indices[(delta > 3) & (delta < 10)]
        if not len(upstream):
            continue
        index = int(upstream[np.argmin(np.abs(samples["arc_length_m"][upstream]-(section["segment_arc_length"]-5)))])
        height = samples["roof_world_z"][index]-samples["floor_world_z"][index]
        if height < 1.25:
            continue
        eye = samples["center_xyz_m"][index].copy()
        eye[2] = samples["floor_world_z"][index]+min(1.4, .65*height)
        target = point.copy()
        nearby = int(np.count_nonzero(np.linalg.norm(positions-point, axis=1)<6))
        candidates.append((nearby, event, eye, target))
    candidates.sort(key=lambda item: item[0], reverse=True)
    assert candidates, "No suitable interior camera; inspect placement data before choosing a view"

    print("Reading exported cave and separately transformed rock nodes", flush=True)
    source = out / "lava_tube_with_rocks.glb"
    scene = trimesh.load(source, force="scene", process=False)
    z_up = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
    meshes, colors, node_info = [], [], []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy(include_cache=True)
        mesh.apply_transform(z_up @ transform)
        is_prop = node.startswith("event_")
        meshes.append(mesh)
        colors.append(np.tile([188., 126., 70.] if is_prop else [210., 204., 193.], (len(mesh.vertices), 1)))
        node_info.append({"name": node, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                          "is_prop": is_prop, "finite": bool(np.isfinite(mesh.vertices).all())})
    assert sum(n["is_prop"] for n in node_info) == len(study["prop_meshes"])
    assert all(n["finite"] for n in node_info)
    combined = trimesh.util.concatenate(meshes)
    combined.vertex_normals = np.vstack([m.vertex_normals for m in meshes])
    albedo = np.vstack(colors)
    for i, (nearby, event, eye, target) in enumerate(candidates[:args.candidate_count]):
        print(f"Rendering rock family around boulder {event['event_id']} ({nearby} nearby props)", flush=True)
        report = render(combined, eye, target, out/f"rock_interior_candidate_{i}.png", "",
                        far=35, width=1100, height=700, fov=70, annotate=False, vertex_colors=albedo)
        cameras.append({"candidate": i, "boulder_event_id": event["event_id"], "segment_id": event["segment_id"],
                        "nearby_props_within_6m": nearby, **report})
    (out/"rock_render_report.json").write_text(json.dumps({
        "source_glb_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Actual exported scene; camera-headlight shading; ochre marks props and neutral gray marks cave, independent of material textures.",
        "nodes": node_info, "cameras": cameras, "selected_candidate": 0,
    }, indent=2)+"\n")
    shutil.copy2(out/"rock_interior_candidate_0.png", out/"rock_interior.png")


if __name__ == "__main__":
    main()

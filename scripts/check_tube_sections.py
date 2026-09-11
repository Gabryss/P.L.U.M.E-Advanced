#!/usr/bin/env python3
"""Measure transverse cuts through the exported tube, including wide rooms."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from matplotlib.path import Path as Polygon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--regular-count", type=int, default=8, help="Number of ordinary passage cuts distributed across the saved section samples")
    parser.add_argument("--shallow-count", type=int, default=0, help="Also inspect this many low-clearance sections")
    args = parser.parse_args()
    if args.regular_count < 1 or args.shallow_count < 0:
        parser.error("regular-count must be positive and shallow-count nonnegative")
    root = args.directory
    scene = trimesh.load(root / "lava_tube_geometry.glb", force="scene", process=False)
    mesh = next(iter(scene.geometry.values())).copy()
    mesh.apply_transform(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]))
    with np.load(root / "sections.npz") as saved:
        p = {key: saved[key] for key in (
            "profile_offsets", "profile_points", "junction_influence", "center_xyz_m",
            "segment_id", "tangent", "normal", "binormal",
        )}
    off = p["profile_offsets"]
    profiles = [p["profile_points"][a:b] for a, b in zip(off[:-1], off[1:], strict=True)]
    heights = np.array([np.ptp(q[:, 1]) for q in profiles])
    widths = np.array([np.ptp(q[:, 0]) for q in profiles])
    eligible = np.flatnonzero((p["junction_influence"] < .1) & (heights > 1.) & (heights < 3.))
    if len(eligible) < args.regular_count:
        # Short galleries can be entirely inside overlapping junction
        # influence regions. Keep inspecting actual cuts there.
        eligible = np.flatnonzero(np.isfinite(heights) & (heights > 1.))
    if not len(eligible):
        raise RuntimeError("No section above one metre is available for the ordinary passage checks")
    chosen = list(eligible[np.linspace(0, len(eligible)-1, min(args.regular_count, len(eligible)), dtype=int)])
    rooms = []
    for index in np.argsort(widths)[::-1]:
        if all(np.linalg.norm(p["center_xyz_m"][index]-p["center_xyz_m"][previous]) > 60. for previous in rooms):
            rooms.append(int(index))
            if len(rooms) == 2:
                break
    chosen.extend(rooms)
    shallow = []
    for index in np.argsort(heights):
        if len(shallow) >= args.shallow_count:
            break
        if index not in chosen and all(np.linalg.norm(p["center_xyz_m"][index]-p["center_xyz_m"][previous]) > 8.
                                       for previous in shallow):
            shallow.append(int(index))
    chosen.extend(shallow)
    chosen = list(dict.fromkeys(map(int, chosen)))
    centers = mesh.triangles_center
    rows = []
    for index in chosen:
        center, tangent, normal, binormal = (p[key][index] for key in ("center_xyz_m", "tangent", "normal", "binormal"))
        local_faces = np.flatnonzero(np.max(np.abs(centers-center), axis=1) < 1.5*max(widths[index], 8.))
        relative = centers[local_faces]-center
        local_faces = local_faces[(abs(relative@tangent) < 1.) & (abs(relative@binormal) < max(heights[index]*2., 5.))]
        cut = mesh.section(plane_normal=tangent, plane_origin=center, local_faces=local_faces)
        candidates = []
        cut_scope = "local_faces"
        for full_mesh in (False, True):
            if full_mesh:
                # A transverse plane near a confluence can follow an oblique
                # neighbouring passage outside the local crop. Retry the
                # actual complete mesh before declaring a broken contour.
                cut = mesh.section(plane_normal=tangent, plane_origin=center)
                cut_scope = "full_mesh"
            if cut is not None:
                for points in cut.discrete:
                    q = np.column_stack(((points-center)@normal, (points-center)@binormal))
                    if len(q) >= 4 and np.linalg.norm(q[0]-q[-1]) < .01 and Polygon(q).contains_point((0., 0.)):
                        candidates.append(q)
            if candidates:
                break
        row = {"sample_index":int(index), "segment_id":int(p["segment_id"][index]),
               "kind":"chamber" if index in rooms else ("shallow" if index in shallow else "passage"),
               "junction_influence":float(p["junction_influence"][index]),
               "measured":bool(candidates), "cut_scope":cut_scope}
        if candidates:
            q = min(candidates, key=lambda q: np.linalg.norm(q.mean(axis=0)))
            row.update(profile_height_m=float(heights[index]), mesh_height_m=float(np.ptp(q[:, 1])),
                       height_difference_m=float(np.ptp(q[:, 1])-heights[index]),
                       profile_width_m=float(widths[index]), mesh_width_m=float(np.ptp(q[:, 0])))
        rows.append(row)
    report = {"scope":f"{len(chosen)} transverse spot checks on the exported, smoothed GLB, including {len(rooms)} rooms and {len(shallow)} low-clearance sections; not a complete clearance survey.",
              "all_measured":all(row["measured"] for row in rows), "measurements":rows}
    (root / "mesh_section_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_measured"]:
        raise RuntimeError("Some requested inspection cuts did not produce a closed contour")


if __name__ == "__main__":
    main()

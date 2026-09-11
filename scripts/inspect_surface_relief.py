#!/usr/bin/env python3
"""Compare actual passage geometry with and without accretion, at identical settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import trimesh
from generate_tube_only import export_surface
from PIL import Image
from render_tube_views import render

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import export_section_artifact
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.surface_relief import apply_surface_relief


def measure_comparison(out: Path) -> None:
    """Measure transverse contours on both exported meshes, not just their shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as Polygon

    p = np.load(out / "sections.npz")
    indices = list(range(2, len(p["center_xyz_m"])-2))
    measured = {}
    for name in ("before", "after"):
        scene = trimesh.load(out/name/"lava_tube_geometry.glb", force="scene", process=False)
        mesh = next(iter(scene.geometry.values())).copy()
        mesh.apply_transform(np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]]))
        contours = []
        for i in indices:
            center = p["center_xyz_m"][i]
            cut = mesh.section(plane_normal=p["tangent"][i], plane_origin=center)
            candidates = []
            if cut is not None:
                for points in cut.discrete:
                    q = np.column_stack(((points-center)@p["normal"][i], (points-center)@p["binormal"][i]))
                    if np.linalg.norm(q[0]-q[-1]) < .01 and Polygon(q).contains_point((0., 0.)):
                        candidates.append(q)
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one closed passage contour at {name} cut {i}")
            contours.append(candidates[0])
        measured[name] = contours
    rows = []
    for j, i in enumerate(indices):
        row = {"section_index": i, "arc_length_m": float(p["arc_length_m"][i])}
        for name in ("before", "after"):
            q = measured[name][j]
            row[name+"_height_m"] = float(np.ptp(q[:, 1]))
            row[name+"_width_m"] = float(np.ptp(q[:, 0]))
            row[name+"_area_m2"] = float(abs(np.sum(q[:-1, 0]*q[1:, 1]-q[1:, 0]*q[:-1, 1]))/2.)
        rows.append(row)
    (out / "mesh_cut_comparison.json").write_text(json.dumps({"all_cuts_closed": True,
        "scope": "Matched transverse spot checks on exported meshes; not a complete clearance survey.",
        "measurements": rows}, indent=2)+"\n")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    for ax, j in zip(axes, np.linspace(0, len(rows)-1, 3, dtype=int)):
        for name, color in (("before", "#a6a8aa"), ("after", "#815d39")):
            q = measured[name][j]
            ax.plot(q[:, 0], q[:, 1], label=name.title(), color=color, lw=1.6)
        ax.set_aspect("equal")
        ax.set(xlabel="Across passage (m)", ylabel="Height relative to axis (m)",
               title=f"Exported mesh cut at {rows[j]['arc_length_m']:.1f} m along segment")
        ax.grid(alpha=.15)
    axes[0].legend(loc="upper right")
    fig.suptitle("Real geometric relief — matched cuts before and after accretion")
    fig.savefig(out / "mesh_cut_comparison.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=55)
    parser.add_argument("--voxel-size", type=float, help="Override resolution for a close inspection")
    args = parser.parse_args()
    config = load_project_config(args.config)
    if config.events.enabled or config.events.include_rock_props:
        parser.error("Disable events and rock props for this comparison")
    network = CaveNetworkGenerator(config.network).generate(HostFieldGenerator(config.host_field).generate())
    sections = SectionFieldGenerator(config.section_field).generate(network)
    field = next(f for f in sections.segment_fields if f.segment_id == args.segment)
    # Keep a roughly 60 m reach, with the camera safely away from its end caps.
    selected = tuple(s for s in field.samples if 20. <= s.segment_arc_length <= 85.)
    if len(selected) < 4:
        selected = field.samples
    if len(selected) < 4:
        parser.error("Choose a segment with at least four section samples")
    network = replace(network, segments=tuple(s for s in network.segments if s.segment_id == args.segment), junctions=())
    sections = replace(sections, segment_fields=(replace(field, samples=selected, connected_junction_ids=()),),
                       dominant_route_segment_ids=(args.segment,))
    c = replace(config.geometry, density_margin=3., storage_mode="dense")
    if args.voxel_size is not None:
        if args.voxel_size <= 0.:
            parser.error("--voxel-size must be positive")
        c = replace(c, voxel_size=args.voxel_size,
                    target_samples_across_passage=c.characteristic_passage_width_m/args.voxel_size)
    base_config = replace(c, surface_wall_relief_m=0., surface_roof_relief_m=0.,
                          surface_floor_relief_m=0., surface_crust_relief_m=0.)
    base = GeometryGenerator(base_config).build_base_volume(network, sections)
    original_density = base.voxel_grid.density.copy()
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(asdict(c), indent=2)+"\n")
    export_section_artifact(sections, out / "sections.npz")
    eye_sample, target_sample = selected[2], selected[4]
    eye = np.array([eye_sample.x, eye_sample.y, eye_sample.z])
    eye[2] += min(z for _, z in eye_sample.profile_points) + .60*eye_sample.tube_height
    target = np.array([target_sample.x, target_sample.y, target_sample.z])
    target[2] += min(z for _, z in target_sample.profile_points) + .60*target_sample.tube_height
    camera = {"eye": eye.tolist(), "target": target.tolist(), "segment_id": args.segment}
    (out / "camera.json").write_text(json.dumps(camera, indent=2)+"\n")
    for name in ("before", "after"):
        folder = out/name
        folder.mkdir(exist_ok=True)
        if name == "after":
            apply_surface_relief(base.voxel_grid, c)
            GeometryGenerator._remove_small_solid_pockets(base.voxel_grid, include_void=True)
        geometry = GeometryGenerator(c).finalize(replace(base, config=c))
        export_surface(geometry, folder)
        scene = trimesh.load(folder / "lava_tube_geometry.glb", force="scene", process=False)
        mesh = next(iter(scene.geometry.values())).copy(include_cache=True)
        mesh.apply_transform(np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]]))
        render(mesh, eye, target, folder / "interior.png", name.title()+" | same passage, camera and shading", far=55.)
        # Inspect the nearby wall and ceiling too; a long axial view can hide
        # their depth, especially under nearly grazing inspection lighting.
        side_target = eye+3.*np.asarray(eye_sample.tangent)+3.*np.asarray(eye_sample.normal)
        side_target[2] += .45
        render(mesh, eye, side_target, folder / "wall_roof.png", name.title()+" | nearby wall and roof", far=20.)
        print(name, "rendered", flush=True)
    shrink_only = bool(np.all(base.voxel_grid.density <= original_density))
    (out / "comparison_checks.json").write_text(json.dumps({"void_only_shrunk": shrink_only,
        "same_resolution_camera_and_normal_filter": True, "textures": False, "rocks": False}, indent=2)+"\n")
    comparison = Image.new("RGB", (1920, 1200))
    for row, view in enumerate(("interior", "wall_roof")):
        comparison.paste(Image.open(out / f"before/{view}.png"), (0, row*600))
        comparison.paste(Image.open(out / f"after/{view}.png"), (960, row*600))
    comparison.save(out / "comparison.png")
    measure_comparison(out)


if __name__ == "__main__":
    main()

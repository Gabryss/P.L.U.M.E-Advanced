#!/usr/bin/env python3
"""Rebuild paper event/atlas figures on a saved current tube, optionally with Rocky props."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import export_event_report, export_geometry_report
from plume_advanced.stages.events import GeologicalEventGenerator
from plume_advanced.stages.floor_map import FloorMapGenerator, export_floor_atlas
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_export import _smooth_visual_surface, density_surface_normals

COLORS = {"collapse": "#b76b24", "choke": "#ac3751", "infill": "#7060a7"}
GEOLOGY = {"bare_basalt": "#6b7781", "sediment": "#c19640", "breakdown": "#8f3d25",
           "debris": "#b96c27", "constriction": "#7751ad"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 12,
                     "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False})


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def export_inspection_scene(geometry, out):
    """Export actual wall and prop geometry without an unnecessary texture atlas."""
    vertices = _smooth_visual_surface(np.asarray(geometry.assembled_vertices, dtype=float),
                np.asarray(geometry.assembled_faces), iterations=geometry.config.cave_smoothing_iterations)
    wall = trimesh.Trimesh(vertices=vertices, faces=geometry.assembled_faces, process=False)
    wall.vertex_normals = density_surface_normals(geometry.voxel_grid, vertices, wall.vertex_normals,
                         filter_voxels=geometry.config.surface_normal_filter_voxels)
    scene = trimesh.Scene()
    to_gltf = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    for name, mesh, metadata, color in [
        ("cave_wall", wall, {"kind": "cave_wall"}, [150,143,130,255]),
        *[(f"event_{m.event_id:04d}_{m.kind}", trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False),
           {"kind": m.kind, "source_generator": m.source_generator, "event_id": m.event_id,
            "debris_family_id": m.debris_family_id, "debris_role": m.debris_role}, [128,105,78,255])
          for m in geometry.event_meshes],
    ]:
        mesh.apply_transform(to_gltf)
        mesh.metadata = metadata
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=name, baseColorFactor=color, metallicFactor=0., roughnessFactor=.95, doubleSided=True))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(out / "lava_tube_with_rocks.glb")


def collect(checkpoint, config_path, out, include_rocks=False, event_field=None):
    config = load_project_config(config_path)
    kinds = ("collapse", "choke", "infill", "rock", "boulder") if include_rocks else ("collapse", "choke", "infill")
    config = replace(config, events=replace(config.events, enabled=True,
                     include_rock_props=include_rocks, use_rocky_meshes=include_rocks,
                     enabled_kinds=kinds))
    print("Loading the saved no-event tube", flush=True)
    network, sections, base = pickle.loads(checkpoint.read_bytes())
    assert base.config == config.geometry, "Checkpoint and configuration must describe the same geometry"
    assert not base.event_meshes and not base.structural_event_ids
    (out / "resolved_config.json").write_text(json.dumps(asdict(config), default=str, indent=2)+"\n")
    floor = FloorMapGenerator(config.floor_map)
    print("Raycasting the current base floor atlas", flush=True)
    before = floor.generate(network, sections, base)
    export_floor_atlas(before, out / "base_floor_atlas")
    if event_field:
        events = pickle.loads(event_field.read_bytes())
        assert events.config == config.events, "Saved events and current event settings must match"
    else:
        events = GeologicalEventGenerator(config.events).generate(
            sections, base, before,
            progress=lambda phase, i, n, message: print(f"events/{phase}: {i}/{n} {message}", flush=True),
        )
    assert all(e.kind in kinds for e in events.events)
    if include_rocks:
        assert events.meshes and all(m.source_generator == "rocky" for m in events.meshes)
        (out / "event_field.pkl").write_bytes(pickle.dumps(events))
    else:
        assert not events.meshes
    print(f"Generated {len(events.events)} events and {len(events.meshes)} rock meshes", flush=True)
    final = GeometryGenerator(config.geometry).finalize(
        base, events, progress=lambda phase, i, n, message: print(f"{phase}: {i}/{n} {message}", flush=True)
    )
    if include_rocks:
        print("Saving the actual cave and separately editable rock meshes", flush=True)
        export_inspection_scene(final, out)
    print("Relifting floor cells against the final event-modified volume", flush=True)
    after = floor.revalidate(network, sections, final, before, events)
    assert len(after.cells) + len(after.invalidated_cell_ids) == len(before.cells)
    export_floor_atlas(after, out / "final_floor_atlas")
    export_geometry_report(final, out / "geometry_report.json")
    export_event_report(events, out / "event_report.json", invalidated_floor_cells=len(after.invalidated_cell_ids))
    report_path = out / "event_report.json"
    report_path.write_text(json.dumps(finite_json(json.loads(report_path.read_text())), indent=2, allow_nan=False)+"\n")
    applied = [e for e in events.events if e.event_id in final.structural_event_ids]
    assert applied, "No structural event survived; an empty event comparison must not be illustrated as an effect"
    samples = {(s.segment_id, s.index): s for f in sections.segment_fields for s in f.samples}
    # Evaluate the actual before/after scalar volumes in the source section plane.
    # Select the applied event with the largest changed area on that plane.
    slices = []
    for event in applied:
        sample = samples[(event.segment_id, event.sample_index)]
        q = np.array(sample.profile_points)
        u = np.arange(q[:, 0].min()-1, q[:, 0].max()+1.01, .1)
        v = np.arange(q[:, 1].min()-1, q[:, 1].max()+1.01, .1)
        uu, vv = np.meshgrid(u, v)
        points = (np.array([sample.x, sample.y, sample.z]) + uu[..., None]*sample.normal
                  + vv[..., None]*sample.binormal)
        densities = [np.array([g.voxel_grid.sample_density(p) for p in points.reshape(-1, 3)]).reshape(uu.shape)
                     for g in (base, final)]
        inside = [d > g.voxel_grid.iso_level for d, g in zip(densities, (base, final), strict=True)]
        area = float(np.count_nonzero(inside[0] & ~inside[1])*.01)
        slices.append((area, event, sample, u, v, *densities))
    area, event, sample, u, v, density_before, density_after = max(slices, key=lambda x: x[0])
    np.savez_compressed(out / "event_section.npz", u=u, v=v, before=density_before, after=density_after,
                        iso=base.voxel_grid.iso_level)
    payload = {
        "body": config.world.body.name, "seed": config.procedural_seed,
        "voxel_size_m": config.geometry.voxel_size,
        "event_overrides": {"enabled": True, "include_rock_props": include_rocks, "use_rocky_meshes": include_rocks,
                            "enabled_kinds": list(kinds)},
        "before_summary": before.summary(), "after_summary": after.summary(),
        "events": [asdict(e) for e in events.events], "applied_event_ids": list(final.structural_event_ids),
        "before_cells": [asdict(c) for c in before.cells], "after_cells": [asdict(c) for c in after.cells],
        "invalidated_cell_ids": list(after.invalidated_cell_ids),
        "network": [{"segment_id": s.segment_id, "xy": [[p.x, p.y] for p in s.points]} for s in network.segments],
        "section_example": {"event_id": event.event_id, "kind": event.kind, "segment_id": sample.segment_id,
                            "sample_index": sample.index, "removed_slice_area_m2": area},
        "final_surface_component_count": final.component_count,
        "event_summary": events.summary(),
        "prop_meshes": [{"event_id": m.event_id, "kind": m.kind, "source_generator": m.source_generator,
                         "shape": m.source_shape_type, "vertex_count": m.vertex_count, "face_count": m.face_count,
                         "extent_xyz_m": np.ptp(np.asarray(m.vertices), axis=0).tolist()}
                        for m in events.meshes],
        "prop_source_sections": {str(e.event_id): asdict(samples[(e.segment_id, e.sample_index)])
                                 for e in events.events if e.kind in {"rock", "boulder"}},
        "scope": "Separate optional-event illustration on the existing seed-4 base; original no-event GLB preserved; not a confirmatory campaign.",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (out / "study.json").write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: payload[k] for k in ("before_summary", "after_summary", "applied_event_ids", "section_example")}, indent=2), flush=True)


def context(ax, study):
    for segment in study["network"]:
        xy = np.array(segment["xy"])
        ax.plot(xy[:, 1], xy[:, 0], color="#b0b9be", lw=.8, zorder=0)
    ax.set(xlabel="World Y (m)", ylabel="World X (m)")
    ax.set_aspect("equal", adjustable="datalim")


def render(out):
    study = json.loads((out / "study.json").read_text())
    applied = set(study["applied_event_ids"])
    events, before, after = study["events"], study["before_cells"], study["after_cells"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    context(axes[0, 0], study)
    props = study.get("prop_meshes", [])
    if props:
        for kind, color in (("rock", "#c18b42"), ("boulder", "#5d4737")):
            selected = [e for e in events if e["kind"] == kind]
            axes[0, 0].scatter([e["y"] for e in selected], [e["x"] for e in selected],
                               color=color, s=9 if kind == "rock" else 23, label=kind, zorder=2)
    for kind, color in COLORS.items():
        selected = [e for e in events if e["kind"] == kind]
        axes[0, 0].scatter([e["y"] for e in selected], [e["x"] for e in selected],
                           color=color, edgecolor="white", s=65, label=kind, zorder=3)
    axes[0, 0].set_title("(a) Rock and structural-event locations" if props else "(a) Structural-event locations")
    axes[0, 0].legend(loc="upper right", fontsize=10)
    with np.load(out / "event_section.npz") as cut:
        ax = axes[0, 1]
        ax.contour(cut["u"], cut["v"], cut["before"], levels=[float(cut["iso"])], colors=["#237f8a"], linewidths=1.8)
        ax.contour(cut["u"], cut["v"], cut["after"], levels=[float(cut["iso"])], colors=["#b75a39"], linewidths=1.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="Local lateral coordinate (m)", ylabel="Local vertical coordinate (m)",
               title=f"(b) {study['section_example']['kind'].title()} event {study['section_example']['event_id']} · volume section")
        ax.legend(handles=[Line2D([], [], color="#237f8a", label="Before events"),
                           Line2D([], [], color="#b75a39", label="After events")], fontsize=10)
    kinds = list(COLORS)
    x = np.arange(3)
    proposed = [sum(e["kind"] == k for e in events) for k in kinds]
    accepted = [sum(e["kind"] == k and e["event_id"] in applied for e in events) for k in kinds]
    axes[1, 0].bar(x-.18, proposed, width=.36, color="#9aa7b1", label="Generated")
    axes[1, 0].bar(x+.18, accepted, width=.36, color="#287d88", label="Applied to volume")
    axes[1, 0].set(xticks=x, xticklabels=kinds, ylabel="Event count", title="(c) Structural candidates and accepted modifiers")
    axes[1, 0].set_yticks(range(max(proposed)+2))
    axes[1, 0].legend(fontsize=10)
    lookup = {c["cell_id"]: c for c in before}
    h0 = np.array([lookup[c["cell_id"]]["clearance_m"] for c in after])
    h1 = np.array([c["clearance_m"] for c in after])
    ax = axes[1, 1]
    ax.scatter(h0, h1, s=15, color="#287d88", alpha=.5, edgecolors="none")
    bound = float(max(h0.max(), h1.max()))*1.08
    ax.plot([0, bound], [0, bound], ls="--", color="#8a949c", lw=1)
    ax.set(xlim=(0, bound), ylim=(0, bound), xlabel="Before-event clearance (m)", ylabel="After-event clearance (m)",
           title="(d) Paired surviving floor cells")
    ax.text(.04, .92, f"{len(after)} paired cells; {len(study['invalidated_cell_ids'])} invalidated", transform=ax.transAxes, fontsize=10)
    if props:
        # In the rock-enabled edition, show actual exported geometry and mesh
        # measurements instead of repeating the floor-atlas clearance plot.
        render_file = out / "rock_interior.png"
        if not render_file.exists():
            raise FileNotFoundError("Render the exported rock scene and select rock_interior.png before building the rock-enabled figure")
        ax = axes[1, 0]
        ax.clear()
        ax.imshow(plt.imread(render_file))
        ax.set_title("(c) Exported rock family inside the tube")
        ax.set_axis_off()
        ax = axes[1, 1]
        ax.clear()
        maximum = max(max(p["extent_xyz_m"][:2]) for p in props)
        minimum = min(max(p["extent_xyz_m"][:2]) for p in props)
        bins = np.geomspace(minimum*.95, maximum*1.05, 18)
        for kind, color in (("rock", "#c18b42"), ("boulder", "#5d4737")):
            values = [max(p["extent_xyz_m"][:2]) for p in props if p["kind"] == kind]
            ax.hist(values, bins=bins, histtype="stepfilled", color=color, alpha=.75, label=f"{len(values)} {kind}s")
        ax.set(xscale="log", xlabel="Maximum world-X/Y mesh extent (m)", ylabel="Mesh count",
               title="(d) Sizes of generated rock meshes")
        ax.legend(fontsize=10)
    fig.savefig(out / "f06_events.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, values, title, cmap, label in [
        (axes[0, 0], [c["z"] for c in after], "(a) Final floor elevation", "terrain", "World Z (m)"),
        (axes[0, 1], [c["clearance_m"] for c in after], "(b) Final overhead clearance", "viridis", "Clearance (m)"),
    ]:
        context(ax, study)
        p = ax.scatter([c["y"] for c in after], [c["x"] for c in after], c=values, cmap=cmap, s=11, edgecolors="none")
        ax.set_title(title)
        fig.colorbar(p, ax=ax, label=label, shrink=.82)
    ax = axes[1, 0]
    context(ax, study)
    for geology, color in GEOLOGY.items():
        cells = [c for c in after if c["geology_class"] == geology]
        if cells:
            ax.scatter([c["y"] for c in cells], [c["x"] for c in cells], c=color, s=16,
                       label=geology.replace("_", " "), edgecolors="none")
    rejected = [lookup[i] for i in study["invalidated_cell_ids"]]
    if rejected:
        ax.scatter([c["y"] for c in rejected], [c["x"] for c in rejected], marker="x", s=45,
                   color="#c43749", label="invalidated cell")
    ax.set_title("(c) Geology labels and invalidated cells")
    ax.legend(fontsize=9, loc="lower left")
    ax = axes[1, 1]
    p = ax.scatter([c["atlas_x_m"] for c in after], [c["atlas_y_m"] for c in after],
                   c=[c["clearance_m"] for c in after], cmap="viridis", s=11, edgecolors="none")
    if rejected:
        ax.scatter([c["atlas_x_m"] for c in rejected], [c["atlas_y_m"] for c in rejected], marker="x", s=35, color="#c43749")
    ax.set(xlabel="Distance along segment (m)", ylabel="Segment band + lateral offset (m)", title="(d) Intrinsic floor atlas")
    fig.colorbar(p, ax=ax, label="Clearance (m)", shrink=.82)
    fig.savefig(out / "f07_floor_atlas.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--collect-only", action="store_true", help="Save the scenario before selecting its interior illustration")
    parser.add_argument("--include-rocks", action="store_true", help="Generate actual Rocky rock/boulder meshes and export the full scene")
    parser.add_argument("--event-field", type=Path, help="Reuse a trusted event checkpoint with matching settings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        if not args.checkpoint or not args.config:
            parser.error("--checkpoint and --config are required for data generation")
        collect(args.checkpoint, args.config, args.output, args.include_rocks, args.event_field)
    if not args.collect_only:
        render(args.output)


if __name__ == "__main__":
    main()

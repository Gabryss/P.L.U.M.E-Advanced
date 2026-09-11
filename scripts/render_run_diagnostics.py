#!/usr/bin/env python3
"""Render stage figures from a completed run, without generating another cave.

Only use trusted local checkpoints created by PLUME: they contain Python pickle
objects. Stored digests check integrity, not the trustworthiness of their author.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plume-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from plume_advanced.evaluation.artifacts import export_section_artifact, network_semantic_hash
from plume_advanced.evaluation.provenance import sha256_file
from plume_advanced.evaluation.visualization.dashboard import render_diagnostic_dashboard
from plume_advanced.evaluation.visualization.emplacement import render_emplacement_phase_activity
from plume_advanced.pipeline.checkpoints import StageCheckpointStore
from plume_advanced.stages.network_topology import section_footprint
from plume_advanced.visualization.drained_pools import DrainedPoolPlotter
from plume_advanced.visualization.events import GeologicalEventPlotter
from plume_advanced.visualization.floor_map import FloorMapPlotter
from plume_advanced.visualization.geometry import GeometryPlotter
from plume_advanced.visualization.host_field import HostFieldPlotter
from plume_advanced.visualization.network import CaveNetworkPlotter
from plume_advanced.visualization.network_topology import render_topology_footprint
from plume_advanced.visualization.section_field import SectionFieldPlotter


class InspectionSectionPlotter(SectionFieldPlotter):
    """Reserve space for profile progression labels at true metric scale."""

    def _draw_plan_panel(self, *, ax, section_field):
        super()._draw_plan_panel(ax=ax, section_field=section_field)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if ymax - ymin > 8 * (xmax - xmin):
            # Keep a reference tick on narrow plans without stretching them.
            reference = 0.0 if xmin <= 0.0 <= xmax else float(f"{(xmin + xmax) / 2:.2g}")
            ax.set_xticks([reference])

    def _draw_cross_section_panel(self, *, ax, cave_network, section_field):
        super()._draw_cross_section_panel(
            ax=ax, cave_network=cave_network, section_field=section_field,
        )
        if ax.texts:
            low, high = ax.get_ylim()
            label_low = min(text.get_position()[1] for text in ax.texts)
            ax.set_ylim(min(low, label_low - .15 * (high - low)), high)
            ax.xaxis.labelpad = 14


class SavedGeometryPlotter(GeometryPlotter):
    """Project sparse tiles directly, without allocating a dense 3-D volume."""

    def __init__(self, geometry):
        super().__init__()
        grid = geometry.voxel_grid
        self.plan = np.zeros(grid.shape[:2], dtype=bool)
        self.profile = np.zeros(grid.shape[1:], dtype=bool)
        for start, tile in self.blocks(grid):
            carved = tile >= grid.iso_level
            x, y, z = start
            nx, ny, nz = tile.shape
            self.plan[x:x + nx, y:y + ny] |= carved.any(axis=2)
            self.profile[y:y + ny, z:z + nz] |= carved.any(axis=0)

    @staticmethod
    def blocks(grid):
        if hasattr(grid, "density"):
            yield (0, 0, 0), grid.density
        else:
            for key, tile in sorted(grid.tiles.items()):
                yield tuple(value * grid.tile_size for value in key), tile

    def _carved_footprint(self, cave_geometry):
        grid = cave_geometry.voxel_grid
        x, y, _ = grid.origin
        return self.plan, (x, x + grid.shape[0] * grid.voxel_size,
                           y, y + grid.shape[1] * grid.voxel_size)

    def _carved_profile(self, cave_geometry):
        grid = cave_geometry.voxel_grid
        rows = np.flatnonzero(self.profile.any(axis=1))
        occupied = self.profile[rows]
        low = occupied.argmax(axis=1)
        high = occupied.shape[1] - 1 - occupied[:, ::-1].argmax(axis=1)
        return (grid.origin[1] + rows * grid.voxel_size,
                grid.origin[2] + low * grid.voxel_size,
                grid.origin[2] + high * grid.voxel_size)

    def _draw_profile_panel(self, ax, cave_network, cave_geometry):
        # Stage-B elevations precede the Stage-C vertical section placement;
        # do not overlay them as if they were the final mesh centreline.
        y, low, high = self._carved_profile(cave_geometry)
        ax.fill_between(y, low, high, color="#0f766e", alpha=0.5)
        ax.set(title="Carved vertical envelope", xlabel="Y (m)", ylabel="Z (m)")
        ax.grid(alpha=0.2)

    def _draw_chunk_profile(self, ax, cave_geometry):
        from matplotlib.patches import Rectangle

        for mesh in cave_geometry.chunk_meshes:
            vertices = np.asarray(mesh.vertices)
            if not len(vertices):
                continue
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            ax.add_patch(Rectangle((low[1], low[2]), high[1] - low[1], high[2] - low[2],
                                   facecolor="#0f766e", edgecolor="#0f766e", alpha=0.15))
        ax.autoscale_view()
        ax.set(title="Mesh chunk bounds", xlabel="Y (m)", ylabel="Z (m)")
        ax.grid(alpha=0.2)

    def _draw_chunk_face_plan(self, ax, cave_network, cave_geometry):
        super()._draw_chunk_face_plan(ax, cave_network, cave_geometry)
        # Chunks at different heights overlap in plan; their IDs cannot all be
        # labelled at the same coordinates. The bar chart retains every ID.
        for label in list(ax.texts):
            label.remove()

    def _draw_slice_panel(self, ax, cave_geometry):
        grid = cave_geometry.voxel_grid
        rows = np.flatnonzero(self.profile.any(axis=1))
        ax.set_title("Voxel slices at fixed Y (not normal to each passage)")
        ax.axis("off")
        if not len(rows):
            return
        indices = rows[np.linspace(0.1 * (len(rows) - 1), 0.9 * (len(rows) - 1), 4).astype(int)]
        for i, index in enumerate(indices):
            section = np.zeros((grid.shape[0], grid.shape[2]), dtype=bool)
            for (x, y, z), tile in self.blocks(grid):
                if y <= index < y + tile.shape[1]:
                    section[x:x + tile.shape[0], z:z + tile.shape[2]] |= tile[:, index - y, :] >= grid.iso_level
            xx, zz = np.where(section)
            inset = ax.inset_axes([0.01 + i * 0.25, 0.1, 0.22, 0.78])
            if not len(xx):
                inset.axis("off")
                continue
            x0, x1 = max(0, xx.min() - 4), min(section.shape[0], xx.max() + 5)
            z0, z1 = max(0, zz.min() - 4), min(section.shape[1], zz.max() + 5)
            inset.imshow(section[x0:x1, z0:z1].T, origin="lower", cmap="magma",
                         interpolation="nearest", extent=(
                             grid.origin[0] + x0 * grid.voxel_size,
                             grid.origin[0] + x1 * grid.voxel_size,
                             grid.origin[2] + z0 * grid.voxel_size,
                             grid.origin[2] + z1 * grid.voxel_size))
            inset.set(title=f"Y = {grid.origin[1] + index * grid.voxel_size:.1f} m",
                      xlabel="X (m)", ylabel="Z (m)")
            inset.tick_params(labelsize=7)

    def _draw_presentation_mesh(self, ax, cave_geometry, poly_collection_cls):
        vertices = np.asarray(cave_geometry.assembled_vertices)
        faces = np.asarray(cave_geometry.assembled_faces)
        ax.add_collection3d(poly_collection_cls(vertices[faces], facecolors="#36978d",
                                               linewidth=0, shade=True))
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Complete saved isosurface")
        ax.set_box_aspect(high - low)
        ax.view_init(elev=45, azim=-15)
        ax.set_xticks([])
        ax.set_zticks([])
        ax.set_yticks(np.linspace(low[1], high[1], 3).round())
        ax.set_xlabel("")
        ax.set_zlabel("")
        size = high - low
        ax.text2D(0.5, 0.1, f"Physical extent (X × Y × Z): {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} m",
                  ha="center", transform=ax.transAxes, fontsize=9)


def render_saved_systems(host, network, path):
    """Show source provenance and every junction without overlapping labels."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    angle = np.radians(host.config.flow_angle_degrees)
    basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = np.array(host.config.seed_point)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    incoming = Counter(segment.end_node_id for segment in network.segments)
    outgoing = Counter(segment.start_node_id for segment in network.segments)
    fig, (ax, details) = plt.subplots(2, 1, figsize=(15, 8), layout="constrained",
                                    gridspec_kw={"height_ratios": [1.3, 1]})
    for segment in network.segments:
        points = (np.array([[p.x, p.y] for p in segment.points]) - origin) @ basis
        systems = segment.metadata["system_ids"]
        color = colors[systems[0] % len(colors)] if len(systems) == 1 else "#242B35"
        ax.plot(*points.T, color=color, lw=2)
    rows = []
    for node in sorted(network.nodes, key=lambda node: node.along_position):
        if incoming[node.node_id] <= 1 and outgoing[node.node_id] <= 1:
            continue
        xy = (np.array([node.x, node.y]) - origin) @ basis
        kind = "Merge" if incoming[node.node_id] > 1 else "Branch / split"
        ax.scatter(*xy, color="#242B35" if kind == "Merge" else "#9A4DB2",
                   marker="o" if kind == "Merge" else "D", edgecolor="white", zorder=5)
        rows.append([node.node_id, kind, f"{xy[0]:.1f}", incoming[node.node_id], outgoing[node.node_id]])
    handles = [Line2D([], [], color=colors[i % len(colors)], lw=2, label=f"System {i + 1}")
               for i in range(network.config.systems.count)]
    handles += [Line2D([], [], color="#242B35", lw=2, label="Shared passage"),
                Line2D([], [], marker="o", color="#242B35", ls="", label="Merge"),
                Line2D([], [], marker="D", color="#9A4DB2", ls="", label="Branch / split")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=6, frameon=False)
    ax.set(xlabel="Distance along host flow (m)", ylabel="Lateral distance (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    details.axis("off")
    # Interconnected runs can contain dozens of junctions; give each table row
    # enough space while retaining the complete connection inventory.
    if len(rows) > 14:
        fig.set_size_inches(15, 4.5 + 0.30 * len(rows))
    table = details.table(cellText=rows, colLabels=["Node", "Connection", "Along flow (m)", "Incoming", "Outgoing"],
                          cellLoc="center", loc="center", bbox=[0.1, 0.05, 0.8, 0.9])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_saved_footprint(network, sections, path):
    """Keep kilometre-long galleries legible using contiguous metric reaches."""
    if network.dominant_route_length <= 800:
        return render_topology_footprint(network, sections, path)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mask, x, y, spacing = section_footprint(network, sections)
    nodes = {node.node_id: node for node in network.nodes}
    first, last = [nodes[i] for i in (network.dominant_route_node_ids[0], network.dominant_route_node_ids[-1])]
    origin = np.array([first.x, first.y])
    along = np.array([last.x, last.y]) - origin
    along /= np.linalg.norm(along)
    cross = np.array([-along[1], along[0]])
    colors = {"trunk": "#0072B2", "island_arm": "#D55E00", "side_branch": "#009E73", "feeder": "#7570B3"}
    labels = {"trunk": "Main passage", "island_arm": "Island arms", "side_branch": "Blind branch", "feeder": "Source inlet"}
    first_station = 500 * np.floor(max(0, x[0]) / 500)
    count = max(1, int(np.ceil((x[-1] - first_station) / 500)))
    # Tiny end-cap margins can extend a lattice beyond the round route target.
    if count > 1 and x[-1] - (first_station + (count - 1) * 500) < 10:
        count -= 1
    fig, axes = plt.subplots(count, 1, figsize=(16, max(4, 2.1 * count)), squeeze=False)
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.99, hspace=0.5)
    used = {segment.metadata.get("topology_role", "trunk") for segment in network.segments}
    handles = [Patch(facecolor="#cad8dc", label="Section-envelope footprint")]
    handles.extend(Line2D([], [], color=colors[role], lw=2, label=labels[role]) for role in colors if role in used)
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    for i, ax in enumerate(axes.flat):
        lo = first_station + 500 * i
        hi = max(lo + 500, x[-1] + spacing) if i == count - 1 else lo + 500
        if i == 0:
            lo = min(lo, x[0] - spacing / 2)
        a, b = np.searchsorted(x, [lo - spacing, hi + spacing])
        b = min(len(x), b + 1)
        ax.imshow(mask[:, a:b], origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["white", "#cad8dc"]), vmin=0, vmax=1,
                  extent=[x[a] - spacing / 2, x[b - 1] + spacing / 2, y[0] - spacing / 2, y[-1] + spacing / 2])
        for segment in network.segments:
            points = np.array([[point.x, point.y] for point in segment.points]) - origin
            xx, yy = points @ along, points @ cross
            if xx.max() >= lo and xx.min() <= hi:
                ax.plot(xx, yy, color=colors[segment.metadata.get("topology_role", "trunk")], lw=1)
        ax.set(xlim=(lo, hi), ylim=(y[0] - 5, y[-1] + 5),
               xlabel="Downstream position (m)", ylabel="Lateral (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def render_imported_mesh_sheet(root, geometry, path):
    """Use complete Blender renders to avoid duplicating millions of triangles."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), layout="constrained")
    for ax, filename in zip(axes, ("top_down.png", "overview.png"), strict=True):
        ax.imshow(plt.imread(root / "previews" / filename))
        ax.axis("off")
    extent = np.ptp(np.asarray(geometry.assembled_vertices), axis=0)
    fig.supxlabel(
        f"Physical extent (X × Y × Z): {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} m",
        fontsize=10,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Stage figures require a completed run.")

    checkpoint_root = root / ".plume-checkpoints"
    metadata = json.loads((checkpoint_root / "host_field.json").read_text())
    store = StageCheckpointStore(checkpoint_root, metadata["fingerprint"])
    stages = (
        "host_field", "network", "section_field", "final_geometry",
        "final_floor_atlas", "geological_events",
    )
    artifacts = {}
    checkpoint_evidence = {}
    for stage in stages:
        artifacts[stage] = store.load(stage)
        if artifacts[stage] is None:
            raise ValueError(f"Missing, incompatible or damaged checkpoint: {stage}")
        checkpoint_evidence[stage] = json.loads((checkpoint_root / f"{stage}.json").read_text())
    host, network, sections, geometry, floor, events = (artifacts[key] for key in stages)

    # The recorded checkpoint fingerprint is deliberately used here. This is an
    # artifact renderer, not a request to resume generation under today's config.
    saved_network = json.loads((root / "stage_b_network.json").read_text())
    if network_semantic_hash(network) != saved_network["semantic_sha256"]:
        raise ValueError("Checkpoint network differs from the delivered network.")
    with tempfile.TemporaryDirectory(prefix="plume-figure-check-") as temporary:
        arrays_path, _ = export_section_artifact(sections, Path(temporary) / "sections")
        with np.load(arrays_path) as actual, np.load(root / "stage_c_sections.npz") as saved:
            if set(actual.files) != set(saved.files):
                raise ValueError("Checkpoint section fields differ from the saved artifact.")
            for key in saved.files:
                np.testing.assert_array_equal(actual[key], saved[key], err_msg=key)

    protected = [
        root / "run_manifest.json", root / "resolved_project_config.json",
        root / "stage_b_network.json", root / "stage_c_sections.npz",
        *root.glob("export_*/*.glb"), *root.glob("export_*/*.blend"),
    ]
    before = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    figures = []
    skipped = []

    def render(filename, title, caption, callback):
        print(f"Rendering {filename}", flush=True)
        path = root / filename
        callback(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Renderer did not write {filename}")
        figures.append(dict(file=filename, title=title, caption=caption, sha256=sha256_file(path)))

    render("stage_a_host_field.png", "A — Physical host field",
           "Terrain, cover and routing controls over the saved host domain.",
           lambda path: HostFieldPlotter().render(host, path))
    render("stage_b_cave_network.png", "B — Accepted network",
           "Accepted centreline, host influence, occupied widths and formation evidence.",
           lambda path: CaveNetworkPlotter().render(host, network, path))
    if network.config.systems.count > 1:
        render("stage_b_system_connections.png", "B — Multiple systems",
               "Source provenance, shared passages and all graph junctions. Branch/split markers include blind breakouts and divisions into parallel passages.",
               lambda path: render_saved_systems(host, network, path))
    if network.config.topology.style == "trunk_dominated":
        render("stage_bc_topology_footprint.png", "B–C — Passage footprint",
               "Plan view of the accepted network and sampled passage envelopes, before meshing. Long runs are divided into consecutive 500 m reaches with equal physical axis scaling.",
               lambda path: render_saved_footprint(network, sections, path))
    if network.config.topology.style == "interconnected":
        from plume_advanced.visualization.interconnected import render_interconnected
        render("stage_bc_interconnected.png", "B–C — Interconnected passages",
               "Actual section envelopes over shared host routing cost, in consecutive 500 m reaches with equal axis scaling. Circles mark merges; diamonds mark splits.",
               lambda path: render_interconnected(host, network, sections, path))
    has_history = (
        network.config.topology.generation_mode == "independent_growth"
        or (network.config.topology.style == "general" and network.config.systems.count == 1)
    )
    if has_history:
        render("stage_b_emplacement_history.png", "B — Formation phases",
               "Per-phase discharge, inactive intervals and passage reuse. These are procedural phases, not dated eruptions.",
               lambda path: render_emplacement_phase_activity(network, path))
    render("stage_c_section_field.png", "C — Cross-sections",
           "Sampled width, height, floor/roof profiles and longitudinal shape variation.",
           lambda path: InspectionSectionPlotter().render(network, sections, path))
    if has_history:
        render("stage_c_drained_pools.png", "C — Drained pools",
               "Pool location, widening, floor depression and transitions into adjacent passages.",
               lambda path: DrainedPoolPlotter().render(network, sections, path))
    render("stage_c_floor_map.png", "C — Final floor atlas",
           "Post-meshing floor samples, clearance and slope. This is a sampled map, not a continuous traversability certificate.",
           lambda path: FloorMapPlotter().render(floor, path, events, network))
    plotter = SavedGeometryPlotter(geometry)
    render("stage_d_geometry.png", "D — Volume and mesh diagnostics",
           "Actual saved voxel footprint, vertical envelope, chunks and fixed-Y slices. Overlapping branches can share a projected vertical envelope.",
           lambda path: plotter.render_debug(network, geometry, path))
    imported_views = [root / "previews" / name for name in ("top_down.png", "overview.png")]
    if all(path.is_file() for path in imported_views) and (root / "blender_import_check.json").is_file():
        render("stage_d_geometry_presentation.png", "D — Mesh overview",
               "Plan and perspective renders of the complete imported Blender mesh, with no decimation or geometry scaling. The scene exports are unchanged.",
               lambda path: render_imported_mesh_sheet(root, geometry, path))
        figures[-1]["source_images"] = {
            str(path.relative_to(root)): sha256_file(path) for path in imported_views
        }
    else:
        render("stage_d_geometry_presentation.png", "D — Mesh overview",
               "Saved cave geometry in plan and perspective with every triangle rendered and equal physical axis scaling; the scene exports are unchanged.",
               lambda path: plotter.render_presentation(network, geometry, path))
    render("stage_d_geometry_chunks.png", "D — Mesh chunks",
           "Spatial partition and face counts of the final mesh.",
           lambda path: plotter.render_chunks(network, geometry, path))
    if events.config.enabled:
        render("stage_e_geological_events.png", "E — Geological events",
               "Events actually present in the saved run.",
               lambda path: GeologicalEventPlotter().render(network, sections, events, path))
    else:
        skipped.append("Stage E was disabled for this run: no optional geological events or loose rocks were generated. No event-placement figure is shown.")
    render("stage_bc_evaluation_dashboard.png", "B–C — Diagnostic dashboard",
           "Topology and section measurements for this cave. No density sweep or reference-dataset comparison was run for this sheet.",
           lambda path: render_diagnostic_dashboard(
               network, sections, path, provenance={"scope": "saved run"},
               json_path=path.with_suffix(".json")))

    after = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    if after != before:
        raise RuntimeError("A delivered scene or canonical generation artifact changed during rendering.")
    record = dict(
        schema="plume.run-figures.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        mode="render_saved_checkpoints_only", generation_rerun=False,
        original_diagnostics_enabled=manifest["resolved_config"]["run"]["render_diagnostics"],
        checkpoint_fingerprint=store.fingerprint, checkpoints=checkpoint_evidence,
        protected_artifact_sha256=after, delivered_artifacts_unchanged=True,
        renderer_script_sha256=sha256_file(Path(__file__)), figures=figures, skipped=skipped,
    )
    (root / "stage_figures_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = [
        "# Stage figures — saved inspection run", "",
        "These figures use the saved stages of this exact cave. No new network or mesh was generated.", "",
        "[Inspection guide](README.md) · [Figure provenance](stage_figures_manifest.json)", "",
        "## Figure index", "",
        *[f"- [{figure['title']}]({figure['file']})" for figure in figures], "",
        *[f"{message}\n" for message in skipped],
    ]
    for figure in figures:
        lines.extend([f"## {figure['title']}", "", figure["caption"], "",
                      f"![{figure['title']}]({figure['file']})", ""])
    (root / "STAGE_FIGURES.md").write_text("\n".join(lines))
    print(f"Rendered {len(figures)} figures; index: {root / 'STAGE_FIGURES.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

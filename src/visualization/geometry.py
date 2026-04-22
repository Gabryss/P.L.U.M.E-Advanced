"""Visualization helpers for the constrained Stage-D geometry sweep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stages.geometry import CaveGeometry
from stages.network import CaveNetwork


@dataclass(frozen=True)
class GeometryPlotConfig:
    """Figure settings for geometry visualization."""

    figure_size: tuple[float, float] = (15.0, 11.0)
    dpi: int = 180


class GeometryPlotter:
    """Render a reviewable artifact for Stage-D1 swept geometry."""

    def __init__(self, config: GeometryPlotConfig | None = None) -> None:
        self.config = config or GeometryPlotConfig()

    def render(
        self,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=self.config.figure_size, constrained_layout=True)
        fig.suptitle("Stage D1 - Constrained Geometry Sweep", fontsize=16)
        axes = fig.subplots(2, 2, subplot_kw={})
        mesh_ax = fig.add_subplot(2, 2, 1, projection="3d")
        plan_ax = axes[0, 1]
        coverage_ax = axes[1, 0]
        summary_ax = axes[1, 1]
        fig.delaxes(axes[0, 0])

        self._draw_mesh_panel(mesh_ax, cave_geometry, Poly3DCollection)
        self._draw_plan_panel(plan_ax, cave_network, cave_geometry)
        self._draw_coverage_panel(coverage_ax, cave_network, cave_geometry)
        self._draw_summary_panel(summary_ax, cave_geometry)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_mesh_panel(self, ax, cave_geometry: CaveGeometry, poly_collection_cls) -> None:
        ax.set_title("Swept Meshes, Junction Patches, And Skylight")
        if not cave_geometry.meshes and not cave_geometry.junction_patches and not cave_geometry.skylights:
            ax.set_axis_off()
            return

        palette = ["#0f766e", "#2563eb", "#f59e0b", "#ef4444", "#7c3aed", "#22c55e"]
        for mesh in cave_geometry.meshes:
            vertices = np.array(mesh.vertices, dtype=float)
            polygons = [
                [vertices[index] for index in face]
                for face in mesh.faces[:: max(len(mesh.faces) // 600, 1)]
            ]
            collection = poly_collection_cls(
                polygons,
                facecolor=palette[mesh.mesh_id % len(palette)],
                edgecolor="#0f172a",
                linewidth=0.15,
                alpha=0.32,
            )
            ax.add_collection3d(collection)

        patch_palette = ["#fb7185", "#f97316", "#eab308", "#84cc16"]
        for patch in cave_geometry.junction_patches:
            vertices = np.array(patch.vertices, dtype=float)
            polygons = [
                [vertices[index] for index in face]
                for face in patch.faces[:: max(len(patch.faces) // 360, 1)]
            ]
            collection = poly_collection_cls(
                polygons,
                facecolor=patch_palette[patch.patch_id % len(patch_palette)],
                edgecolor="#7f1d1d",
                linewidth=0.12,
                alpha=0.40,
            )
            ax.add_collection3d(collection)

        skylight_palette = ["#c026d3", "#a21caf"]
        for skylight in cave_geometry.skylights:
            vertices = np.array(skylight.vertices, dtype=float)
            polygons = [
                [vertices[index] for index in face]
                for face in skylight.faces[:: max(len(skylight.faces) // 360, 1)]
            ]
            collection = poly_collection_cls(
                polygons,
                facecolor=skylight_palette[skylight.skylight_id % len(skylight_palette)],
                edgecolor="#581c87",
                linewidth=0.12,
                alpha=0.44,
            )
            ax.add_collection3d(collection)

        all_vertex_list = [
            vertex
            for mesh in cave_geometry.meshes
            for vertex in mesh.vertices
        ] + [
            vertex
            for patch in cave_geometry.junction_patches
            for vertex in patch.vertices
        ] + [
            vertex
            for skylight in cave_geometry.skylights
            for vertex in skylight.vertices
        ]
        all_vertices = np.array(all_vertex_list, dtype=float)
        x_values = all_vertices[:, 0]
        y_values = all_vertices[:, 1]
        z_values = all_vertices[:, 2]
        ax.set_xlim(float(x_values.min()), float(x_values.max()))
        ax.set_ylim(float(y_values.min()), float(y_values.max()))
        ax.set_zlim(float(z_values.min()), float(z_values.max()))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=26, azim=-62)

    def _draw_plan_panel(self, ax, cave_network: CaveNetwork, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Swept Span Coverage")
        node_lookup = {node.node_id: node for node in cave_network.nodes}
        for segment in cave_network.segments:
            if not segment.points:
                continue
            x_values = [point.x for point in segment.points]
            y_values = [point.y for point in segment.points]
            ax.plot(x_values, y_values, color="#cbd5e1", linewidth=1.0, alpha=0.75)

        swept_segment_ids = {mesh.segment_id for mesh in cave_geometry.meshes}
        for segment in cave_network.segments:
            if segment.segment_id not in swept_segment_ids:
                continue
            x_values = [point.x for point in segment.points]
            y_values = [point.y for point in segment.points]
            ax.plot(x_values, y_values, color="#0f766e", linewidth=1.8, alpha=0.95)

        excluded_segment_ids = set(cave_geometry.excluded_segment_ids)
        for segment in cave_network.segments:
            if segment.segment_id not in excluded_segment_ids:
                continue
            x_values = [point.x for point in segment.points]
            y_values = [point.y for point in segment.points]
            ax.plot(x_values, y_values, color="#ef4444", linewidth=1.2, linestyle="--", alpha=0.85)

        for junction in cave_network.junctions:
            if junction.junction_id not in {patch.junction_id for patch in cave_geometry.junction_patches}:
                continue
            ax.scatter(
                [junction.center_x],
                [junction.center_y],
                s=42,
                c="#fb7185",
                edgecolors="#7f1d1d",
                linewidths=0.4,
                alpha=0.9,
            )

        for skylight in cave_geometry.skylights:
            ax.scatter(
                [skylight.top_center[0]],
                [skylight.top_center[1]],
                s=54,
                c="#c026d3",
                edgecolors="#581c87",
                linewidths=0.5,
                alpha=0.95,
            )

        for node in cave_network.nodes:
            if node.node_id not in node_lookup:
                continue
            ax.scatter(node.x, node.y, s=8, c="#0f172a", alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_coverage_panel(self, ax, cave_network: CaveNetwork, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Geometry Elements")
        swept_counts: dict[int, int] = {}
        for mesh in cave_geometry.meshes:
            swept_counts[mesh.segment_id] = swept_counts.get(mesh.segment_id, 0) + mesh.face_count

        patch_counts = {
            f"J{patch.junction_id}": patch.face_count
            for patch in cave_geometry.junction_patches
        }
        skylight_counts = {
            f"K{skylight.skylight_id}": skylight.face_count
            for skylight in cave_geometry.skylights
        }

        if not swept_counts and not patch_counts and not skylight_counts:
            ax.axis("off")
            return

        segment_labels = [f"S{segment_id}" for segment_id in sorted(swept_counts)]
        segment_counts = [swept_counts[int(label[1:])] for label in segment_labels]
        patch_labels = sorted(patch_counts)
        patch_face_counts = [patch_counts[label] for label in patch_labels]
        skylight_labels = sorted(skylight_counts)
        skylight_face_counts = [skylight_counts[label] for label in skylight_labels]
        labels = segment_labels + patch_labels + skylight_labels
        counts = segment_counts + patch_face_counts + skylight_face_counts
        colors = (
            (["#2563eb"] * len(segment_labels))
            + (["#fb7185"] * len(patch_labels))
            + (["#c026d3"] * len(skylight_labels))
        )
        ax.bar(labels, counts, color=colors, alpha=0.85)
        ax.set_xlabel("Geometry element")
        ax.set_ylabel("Face count")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)

    def _draw_summary_panel(self, ax, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Stage D1 Summary")
        ax.axis("off")
        summary = cave_geometry.summary()
        lines = [
            f"Meshes: {int(summary['mesh_count'])}",
            f"Swept segments: {int(summary['swept_segment_count'])}",
            f"Junction patches: {int(summary['junction_patch_count'])}",
            f"Skylights: {int(summary['skylight_count'])}",
            f"Excluded segments: {int(summary['excluded_segment_count'])}",
            f"Excluded junctions: {int(summary['excluded_junction_count'])}",
            f"Vertices: {int(summary['vertex_count'])}",
            f"Faces: {int(summary['face_count'])}",
            "",
            "Current geometry excludes:",
            "- crossing / underpass segments",
            "",
            "Remaining work:",
            "- non-planar crossing geometry",
            "- better manifold junction stitching.",
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

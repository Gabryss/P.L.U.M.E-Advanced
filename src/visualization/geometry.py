"""Visualization helpers for the voxel Stage-D geometry."""

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
    """Render a reviewable artifact for voxel stamping and isosurface meshing."""

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
        fig.suptitle("Stage D - Voxel Density And Marching Cubes", fontsize=16)
        axes = fig.subplots(2, 2, subplot_kw={})
        mesh_ax = fig.add_subplot(2, 2, 1, projection="3d")
        plan_ax = axes[0, 1]
        voxel_ax = axes[1, 0]
        summary_ax = axes[1, 1]
        fig.delaxes(axes[0, 0])

        self._draw_mesh_panel(mesh_ax, cave_geometry, Poly3DCollection)
        self._draw_plan_panel(plan_ax, cave_network, cave_geometry)
        self._draw_voxel_panel(voxel_ax, cave_geometry)
        self._draw_summary_panel(summary_ax, cave_geometry)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_mesh_panel(self, ax, cave_geometry: CaveGeometry, poly_collection_cls) -> None:
        ax.set_title("Isosurface Mesh")
        vertices = np.array(cave_geometry.assembled_vertices, dtype=float)
        faces = cave_geometry.assembled_faces
        if len(vertices) == 0 or not faces:
            ax.set_axis_off()
            return

        step = max(len(faces) // 1200, 1)
        polygons = [[vertices[index] for index in face] for face in faces[::step]]
        collection = poly_collection_cls(
            polygons,
            facecolor="#0f766e",
            edgecolor="#0f172a",
            linewidth=0.08,
            alpha=0.38,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(float(vertices[:, 0].min()), float(vertices[:, 0].max()))
        ax.set_ylim(float(vertices[:, 1].min()), float(vertices[:, 1].max()))
        ax.set_zlim(float(vertices[:, 2].min()), float(vertices[:, 2].max()))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=28, azim=-64)

    def _draw_plan_panel(self, ax, cave_network: CaveNetwork, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Stamped Network Coverage")
        stamped_segment_ids = set(cave_geometry.stamped_segment_ids)
        for segment in cave_network.segments:
            if not segment.points:
                continue
            color = "#0f766e" if segment.segment_id in stamped_segment_ids else "#cbd5e1"
            linewidth = 1.7 if segment.segment_id in stamped_segment_ids else 0.9
            ax.plot(
                [point.x for point in segment.points],
                [point.y for point in segment.points],
                color=color,
                linewidth=linewidth,
                alpha=0.86,
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_voxel_panel(self, ax, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Carved Density Projection")
        density = cave_geometry.voxel_grid.density
        carved_projection = np.max(density, axis=2)
        ax.imshow(
            carved_projection.T,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xlabel("Voxel X")
        ax.set_ylabel("Voxel Y")

    def _draw_summary_panel(self, ax, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Stage D Summary")
        ax.axis("off")
        summary = cave_geometry.summary()
        lines = [
            f"Voxel size: {cave_geometry.voxel_grid.voxel_size:.2f}",
            f"Grid shape: {cave_geometry.voxel_grid.shape}",
            f"Chunk meshes: {int(summary['chunk_mesh_count'])}",
            f"Stamped segments: {int(summary['stamped_segment_count'])}",
            f"Stamped samples: {int(summary['stamped_sample_count'])}",
            f"Carved voxels: {int(summary['carved_voxel_count'])}",
            f"Voxel components: {int(summary['voxel_component_count'])}",
            f"Mesh components: {int(summary['component_count'])}",
            f"Vertices: {int(summary['vertex_count'])}",
            f"Faces: {int(summary['face_count'])}",
            "",
            "Pipeline:",
            "1. stamp capsule tunnels into density",
            "2. stamp widened junction volumes",
            "3. polygonize chunks for review",
            "4. export one welded cave mesh",
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

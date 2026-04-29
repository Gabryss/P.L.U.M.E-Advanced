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

    debug_figure_size: tuple[float, float] = (16.0, 12.0)
    presentation_figure_size: tuple[float, float] = (16.0, 9.0)
    dpi: int = 180


class GeometryPlotter:
    """Render Stage-D debug and presentation artifacts."""

    def __init__(self, config: GeometryPlotConfig | None = None) -> None:
        self.config = config or GeometryPlotConfig()

    def render(
        self,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
        output_path: str | Path,
    ) -> Path:
        """Render the technical Stage-D diagnostic sheet."""

        return self.render_debug(cave_network, cave_geometry, output_path)

    def render_debug(
        self,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(
            figsize=self.config.debug_figure_size,
            constrained_layout=True,
        )
        fig.suptitle("Stage D - Voxel Geometry Diagnostics", fontsize=16)
        grid = fig.add_gridspec(2, 3, width_ratios=(0.75, 1.15, 1.15))
        plan_ax = fig.add_subplot(grid[0, 0])
        chunk_ax = fig.add_subplot(grid[1, 0])
        profile_ax = fig.add_subplot(grid[0, 1:])
        slice_ax = fig.add_subplot(grid[1, 1:])

        self._draw_plan_footprint_panel(plan_ax, cave_network, cave_geometry)
        self._draw_profile_panel(profile_ax, cave_network, cave_geometry)
        self._draw_chunk_panel(chunk_ax, cave_network, cave_geometry)
        self._draw_slice_panel(slice_ax, cave_geometry)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def render_presentation(
        self,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(
            figsize=self.config.presentation_figure_size,
            constrained_layout=True,
        )
        fig.suptitle("Stage D - Connected Lava-Tube Geometry", fontsize=16)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.2, 1.0))
        plan_ax = fig.add_subplot(grid[0, 0])
        mesh_ax = fig.add_subplot(grid[0, 1], projection="3d")

        self._draw_presentation_plan(plan_ax, cave_network, cave_geometry)
        self._draw_presentation_mesh(mesh_ax, cave_geometry, Poly3DCollection)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def render_chunks(
        self,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
        output_path: str | Path,
    ) -> Path:
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=self.config.debug_figure_size,
            constrained_layout=True,
        )
        fig.suptitle("Stage D - Mesh Chunk Diagnostics", fontsize=16)

        self._draw_chunk_face_plan(axes[0, 0], cave_network, cave_geometry)
        self._draw_chunk_profile(axes[0, 1], cave_geometry)
        self._draw_chunk_face_bars(axes[1, 0], cave_geometry)
        self._draw_chunk_summary(axes[1, 1], cave_geometry)

        fig.savefig(output, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return output

    def _draw_plan_footprint_panel(
        self,
        ax,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
    ) -> None:
        ax.set_title("Carved Footprint vs. Network")
        footprint, extent = self._carved_footprint(cave_geometry)
        ax.imshow(
            footprint.T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.70,
            interpolation="nearest",
        )
        self._draw_network(ax, cave_network, linewidth=1.0, alpha=0.82)
        self._draw_junctions(ax, cave_network)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_profile_panel(
        self,
        ax,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
    ) -> None:
        ax.set_title("Longitudinal Carved Volume")
        y_values, z_min, z_max = self._carved_profile(cave_geometry)
        if len(y_values):
            ax.fill_between(
                y_values,
                z_min,
                z_max,
                color="#99f6e4",
                alpha=0.65,
                linewidth=0.0,
                label="carved voxel span",
            )
        for segment in cave_network.segments:
            if not segment.points:
                continue
            ax.plot(
                [point.y for point in segment.points],
                [point.elevation - point.cover_thickness for point in segment.points],
                color="#0f766e",
                linewidth=0.8,
                alpha=0.55,
            )
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.65)

    def _draw_chunk_panel(
        self,
        ax,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
    ) -> None:
        from matplotlib.patches import Rectangle

        ax.set_title("Meshed Chunk Coverage")
        footprint, extent = self._carved_footprint(cave_geometry)
        ax.imshow(
            footprint.T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.35,
            interpolation="nearest",
        )
        max_faces = max((mesh.face_count for mesh in cave_geometry.chunk_meshes), default=1)
        origin = np.array(cave_geometry.voxel_grid.origin, dtype=float)
        voxel_size = cave_geometry.voxel_grid.voxel_size
        for mesh in cave_geometry.chunk_meshes:
            x_start, x_end, y_start, y_end, _z_start, _z_end = mesh.grid_bounds
            x_min = origin[0] + x_start * voxel_size
            x_max = origin[0] + (x_end + 1) * voxel_size
            y_min = origin[1] + y_start * voxel_size
            y_max = origin[1] + (y_end + 1) * voxel_size
            intensity = mesh.face_count / max_faces
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=(0.06, 0.46, 0.42, 0.10 + 0.32 * intensity),
                edgecolor="#0f766e",
                linewidth=0.7,
            )
            ax.add_patch(rect)
        self._draw_network(ax, cave_network, linewidth=0.75, alpha=0.65)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_chunk_face_plan(
        self,
        ax,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
    ) -> None:
        from matplotlib.patches import Rectangle

        ax.set_title("Plan Chunks by Face Count")
        footprint, extent = self._carved_footprint(cave_geometry)
        ax.imshow(
            footprint.T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.40,
            interpolation="nearest",
        )
        origin = np.array(cave_geometry.voxel_grid.origin, dtype=float)
        voxel_size = cave_geometry.voxel_grid.voxel_size
        max_faces = max((mesh.face_count for mesh in cave_geometry.chunk_meshes), default=1)
        for mesh in cave_geometry.chunk_meshes:
            x_start, x_end, y_start, y_end, _z_start, _z_end = mesh.grid_bounds
            x_min = origin[0] + x_start * voxel_size
            x_max = origin[0] + (x_end + 1) * voxel_size
            y_min = origin[1] + y_start * voxel_size
            y_max = origin[1] + (y_end + 1) * voxel_size
            intensity = mesh.face_count / max_faces
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=(0.93, 0.31, 0.14, 0.16 + 0.38 * intensity),
                edgecolor="#7c2d12",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                (x_min + x_max) * 0.5,
                (y_min + y_max) * 0.5,
                str(mesh.chunk_id),
                ha="center",
                va="center",
                fontsize=7,
                color="#431407",
            )
        self._draw_network(ax, cave_network, linewidth=0.7, alpha=0.70)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    def _draw_chunk_profile(self, ax, cave_geometry: CaveGeometry) -> None:
        from matplotlib.patches import Rectangle

        ax.set_title("Chunk Y/Z Coverage")
        origin = np.array(cave_geometry.voxel_grid.origin, dtype=float)
        voxel_size = cave_geometry.voxel_grid.voxel_size
        carved = cave_geometry.voxel_grid.density >= cave_geometry.voxel_grid.iso_level
        max_faces = max((mesh.face_count for mesh in cave_geometry.chunk_meshes), default=1)
        y_values, z_min, z_max = self._carved_profile(cave_geometry)
        if len(y_values):
            ax.fill_between(
                y_values,
                z_min,
                z_max,
                color="#ccfbf1",
                alpha=0.55,
                linewidth=0.0,
            )
        for mesh in cave_geometry.chunk_meshes:
            _x_start, _x_end, y_start, y_end, z_start, z_end = mesh.grid_bounds
            chunk_carved = carved[:, y_start : y_end + 1, z_start : z_end + 1]
            if np.any(chunk_carved):
                _local_x, local_y, local_z = np.where(chunk_carved)
                y_min_index = y_start + int(local_y.min())
                y_max_index = y_start + int(local_y.max()) + 1
                z_min_index = z_start + int(local_z.min())
                z_max_index = z_start + int(local_z.max()) + 1
            else:
                y_min_index = y_start
                y_max_index = y_end + 1
                z_min_index = z_start
                z_max_index = z_end + 1
            y_min = origin[1] + y_min_index * voxel_size
            y_max = origin[1] + y_max_index * voxel_size
            z_min_world = origin[2] + z_min_index * voxel_size
            z_max_world = origin[2] + z_max_index * voxel_size
            intensity = mesh.face_count / max_faces
            rect = Rectangle(
                (y_min, z_min_world),
                y_max - y_min,
                z_max_world - z_min_world,
                facecolor=(0.06, 0.46, 0.42, 0.08 + 0.32 * intensity),
                edgecolor="#0f766e",
                linewidth=0.7,
            )
            ax.add_patch(rect)
            ax.text(
                (y_min + y_max) * 0.5,
                (z_min_world + z_max_world) * 0.5,
                str(mesh.chunk_id),
                ha="center",
                va="center",
                fontsize=7,
                color="#134e4a",
            )
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.65)

    @staticmethod
    def _draw_chunk_face_bars(ax, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Faces per Meshed Chunk")
        chunk_ids = [mesh.chunk_id for mesh in cave_geometry.chunk_meshes]
        face_counts = [mesh.face_count for mesh in cave_geometry.chunk_meshes]
        if not chunk_ids:
            ax.axis("off")
            return
        ax.bar(chunk_ids, face_counts, color="#0f766e", alpha=0.82)
        ax.set_xlabel("Chunk ID")
        ax.set_ylabel("Face count")
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.5, alpha=0.65)

    @staticmethod
    def _draw_chunk_summary(ax, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Chunk Summary")
        ax.axis("off")
        meshes = cave_geometry.chunk_meshes
        if not meshes:
            ax.text(0.02, 0.98, "No meshed chunks", va="top", ha="left")
            return
        face_counts = np.array([mesh.face_count for mesh in meshes], dtype=float)
        vertex_counts = np.array([mesh.vertex_count for mesh in meshes], dtype=float)
        densest = max(meshes, key=lambda mesh: mesh.face_count)
        sparsest = min(meshes, key=lambda mesh: mesh.face_count)
        lines = [
            f"Meshed chunks: {len(meshes)}",
            f"Chunk size: {cave_geometry.config.chunk_size} voxels",
            f"Faces total: {int(face_counts.sum())}",
            f"Faces mean: {face_counts.mean():.1f}",
            f"Faces median: {np.median(face_counts):.1f}",
            f"Vertices total: {int(vertex_counts.sum())}",
            "",
            f"Densest chunk: #{densest.chunk_id}",
            f"  faces: {densest.face_count}",
            f"  bounds: {densest.grid_bounds}",
            "",
            f"Sparsest chunk: #{sparsest.chunk_id}",
            f"  faces: {sparsest.face_count}",
            f"  bounds: {sparsest.grid_bounds}",
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

    def _draw_slice_panel(self, ax, cave_geometry: CaveGeometry) -> None:
        ax.set_title("Representative Cross Sections")
        density = cave_geometry.voxel_grid.density
        carved = density >= cave_geometry.voxel_grid.iso_level
        y_indices = np.where(np.any(carved, axis=(0, 2)))[0]
        if len(y_indices) == 0:
            ax.axis("off")
            return
        ax.axis("off")

        sample_indices = np.linspace(
            int(y_indices.min()),
            int(y_indices.max()),
            num=4,
            dtype=int,
        )
        origin = np.array(cave_geometry.voxel_grid.origin, dtype=float)
        voxel_size = cave_geometry.voxel_grid.voxel_size

        for panel_index, y_index in enumerate(sample_indices):
            section = carved[:, y_index, :]
            inset = ax.inset_axes(
                [
                    0.03 + 0.245 * panel_index,
                    0.13,
                    0.21,
                    0.74,
                ]
            )
            if not np.any(section):
                inset.axis("off")
                continue
            x_indices, z_indices = np.where(section)
            margin = 4
            x_min_index = max(int(x_indices.min()) - margin, 0)
            x_max_index = min(int(x_indices.max()) + margin + 1, section.shape[0])
            z_min_index = max(int(z_indices.min()) - margin, 0)
            z_max_index = min(int(z_indices.max()) + margin + 1, section.shape[1])
            cropped = section[x_min_index:x_max_index, z_min_index:z_max_index]
            x_min = origin[0] + x_min_index * voxel_size
            x_max = origin[0] + x_max_index * voxel_size
            z_min = origin[2] + z_min_index * voxel_size
            z_max = origin[2] + z_max_index * voxel_size
            inset.imshow(
                cropped.T,
                origin="lower",
                extent=(x_min, x_max, z_min, z_max),
                cmap="magma",
                alpha=0.82,
                interpolation="nearest",
                aspect="equal",
            )
            y_world = origin[1] + y_index * voxel_size
            inset.set_title(f"Y={y_world:.0f}", fontsize=9)
            inset.tick_params(axis="both", labelsize=7)
            inset.set_xlabel("X", fontsize=8)
            if panel_index == 0:
                inset.set_ylabel("Z", fontsize=8)

    def _draw_presentation_plan(
        self,
        ax,
        cave_network: CaveNetwork,
        cave_geometry: CaveGeometry,
    ) -> None:
        ax.set_title("Plan Footprint")
        footprint, extent = self._carved_footprint(cave_geometry)
        ax.imshow(
            footprint.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            alpha=0.78,
            interpolation="nearest",
        )
        self._draw_network(ax, cave_network, linewidth=1.2, alpha=0.85, color="#e0f2fe")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.set_facecolor("#030712")

    def _draw_presentation_mesh(self, ax, cave_geometry: CaveGeometry, poly_collection_cls) -> None:
        ax.set_title("Watertight Isosurface Preview")
        vertices = np.array(cave_geometry.assembled_vertices, dtype=float)
        faces = cave_geometry.assembled_faces
        if len(vertices) == 0 or not faces:
            ax.set_axis_off()
            return

        step = max(len(faces) // 9000, 1)
        polygons = [[vertices[index] for index in face] for face in faces[::step]]
        collection = poly_collection_cls(
            polygons,
            facecolor="#14b8a6",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.72,
        )
        ax.add_collection3d(collection)
        ax.set_xlim(float(vertices[:, 0].min()), float(vertices[:, 0].max()))
        ax.set_ylim(float(vertices[:, 1].min()), float(vertices[:, 1].max()))
        ax.set_zlim(float(vertices[:, 2].min()), float(vertices[:, 2].max()))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=32, azim=-58)

    @staticmethod
    def _draw_network(
        ax,
        cave_network: CaveNetwork,
        *,
        linewidth: float,
        alpha: float,
        color: str = "#0f766e",
    ) -> None:
        for segment in cave_network.segments:
            if not segment.points:
                continue
            ax.plot(
                [point.x for point in segment.points],
                [point.y for point in segment.points],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

    @staticmethod
    def _draw_junctions(ax, cave_network: CaveNetwork) -> None:
        for junction in cave_network.junctions:
            ax.scatter(
                [junction.center_x],
                [junction.center_y],
                s=22,
                c="#f97316",
                edgecolors="#7c2d12",
                linewidths=0.35,
                alpha=0.9,
            )

    @staticmethod
    def _carved_footprint(cave_geometry: CaveGeometry) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        grid = cave_geometry.voxel_grid
        carved = grid.density >= grid.iso_level
        footprint = np.max(carved, axis=2).astype(float)
        origin = np.array(grid.origin, dtype=float)
        extent = (
            float(origin[0]),
            float(origin[0] + grid.shape[0] * grid.voxel_size),
            float(origin[1]),
            float(origin[1] + grid.shape[1] * grid.voxel_size),
        )
        return footprint, extent

    @staticmethod
    def _carved_profile(cave_geometry: CaveGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = cave_geometry.voxel_grid
        carved = grid.density >= grid.iso_level
        y_indices = np.where(np.any(carved, axis=(0, 2)))[0]
        if len(y_indices) == 0:
            return np.array([]), np.array([]), np.array([])
        z_min = []
        z_max = []
        for y_index in y_indices:
            z_indices = np.where(np.any(carved[:, y_index, :], axis=0))[0]
            z_min.append(z_indices.min())
            z_max.append(z_indices.max())
        origin = np.array(grid.origin, dtype=float)
        y_values = origin[1] + y_indices * grid.voxel_size
        z_min_values = origin[2] + np.array(z_min, dtype=float) * grid.voxel_size
        z_max_values = origin[2] + np.array(z_max, dtype=float) * grid.voxel_size
        return y_values, z_min_values, z_max_values

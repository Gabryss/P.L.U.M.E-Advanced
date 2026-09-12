"""Metric inspection plots supporting both dense and sparse voxel storage."""
from __future__ import annotations

import numpy as np

from plume_advanced.visualization.geometry import GeometryPlotter
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
        stride = max(1, int(np.ceil(len(faces) / 100_000)))
        display_faces = faces[::stride]
        ax.add_collection3d(poly_collection_cls(vertices[display_faces], facecolors="#36978d",
                                               linewidth=0, shade=True))
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Saved isosurface" if stride == 1 else "Isosurface triangle sample (export unchanged)")
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



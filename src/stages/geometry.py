"""Stage D voxel stamping and marching-cubes-style isosurface generation."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from stages.geometry_types import CaveGeometry, GeometryChunkMesh, GeometryConfig, VoxelGrid
from stages.network import CaveNetwork
from stages.section_field import SectionField, SectionSample


class GeometryGenerator:
    """Build cave geometry by stamping a density grid and polygonizing it."""

    _CUBE_OFFSETS = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    _TETRAHEDRA = (
        (0, 5, 1, 6),
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
    )
    _TETRA_EDGES = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()

    def generate(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
    ) -> CaveGeometry:
        samples_by_segment = {
            segment_field.segment_id: segment_field.samples
            for segment_field in section_field.segment_fields
            if segment_field.samples
        }
        stamp_samples = [
            sample
            for samples in samples_by_segment.values()
            for sample in samples
        ]
        if not stamp_samples:
            voxel_grid = VoxelGrid(
                origin=(0.0, 0.0, 0.0),
                voxel_size=self.config.voxel_size,
                density=np.full((2, 2, 2), -1.0, dtype=np.float32),
                iso_level=self.config.iso_level,
            )
            return CaveGeometry(
                config=self.config,
                voxel_grid=voxel_grid,
                chunk_meshes=(),
                assembled_vertices=(),
                assembled_faces=(),
                component_count=0,
                stamped_sample_count=0,
                stamped_segment_ids=(),
            )

        voxel_grid = self._build_voxel_grid(samples_by_segment, cave_network)
        chunk_meshes = self._march_chunks(voxel_grid)
        assembled_vertices, assembled_faces = self._assemble_chunks(chunk_meshes)
        component_count = self._count_components(assembled_faces)
        return CaveGeometry(
            config=self.config,
            voxel_grid=voxel_grid,
            chunk_meshes=tuple(chunk_meshes),
            assembled_vertices=assembled_vertices,
            assembled_faces=assembled_faces,
            component_count=component_count,
            stamped_sample_count=len(stamp_samples),
            stamped_segment_ids=tuple(sorted(samples_by_segment)),
        )

    def _build_voxel_grid(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        cave_network: CaveNetwork,
    ) -> VoxelGrid:
        stamp_points = self._stamp_bounds_points(samples_by_segment)
        junction_stamp_points = self._junction_stamp_points(samples_by_segment, cave_network)
        stamp_points.extend(junction_stamp_points)
        margin = max(
            self.config.density_margin,
            max(max(radius_xy, radius_z) for _, radius_xy, radius_z in stamp_points) + self.config.voxel_size,
        )
        positions = np.array([position for position, _, _ in stamp_points], dtype=float)
        lower = positions.min(axis=0) - margin
        upper = positions.max(axis=0) + margin
        voxel_size = self.config.voxel_size
        shape = tuple(int(math.ceil((upper[axis] - lower[axis]) / voxel_size)) + 1 for axis in range(3))
        density = np.full(shape, -1.0, dtype=np.float32)

        for samples in samples_by_segment.values():
            self._stamp_sample_chain(
                density=density,
                origin=lower,
                samples=samples,
            )

        for center, radius_xy, radius_z in junction_stamp_points:
            self._stamp_ellipsoid(
                density=density,
                origin=lower,
                center=center,
                radius_xy=radius_xy,
                radius_z=radius_z,
            )

        return VoxelGrid(
            origin=tuple(float(value) for value in lower),
            voxel_size=voxel_size,
            density=density,
            iso_level=self.config.iso_level,
        )

    def _stamp_bounds_points(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
    ) -> list[tuple[np.ndarray, float, float]]:
        stamp_points: list[tuple[np.ndarray, float, float]] = []
        for samples in samples_by_segment.values():
            for sample in samples:
                position = np.array((sample.x, sample.y, sample.z), dtype=float)
                stamp_points.append((position, self._radius_xy(sample), self._radius_z(sample)))
        return stamp_points

    def _junction_stamp_points(
        self,
        samples_by_segment: dict[int, tuple[SectionSample, ...]],
        cave_network: CaveNetwork,
    ) -> list[tuple[np.ndarray, float, float]]:
        samples_by_junction: dict[int, list[SectionSample]] = defaultdict(list)
        for samples in samples_by_segment.values():
            for sample in samples:
                for influence in sample.junction_influences:
                    samples_by_junction[influence.junction_id].append(sample)

        stamp_points: list[tuple[np.ndarray, float, float]] = []
        for junction in cave_network.junctions:
            influenced_samples = samples_by_junction.get(junction.junction_id, [])
            if not influenced_samples:
                continue
            mean_z = float(np.mean([sample.z for sample in influenced_samples]))
            mean_height = float(np.mean([sample.tube_height for sample in influenced_samples]))
            sample_radius = max(self._radius_xy(sample) for sample in influenced_samples)
            blend_radius = max(junction.blend_length * 0.28, sample_radius)
            if junction.kind == "chamber":
                blend_radius *= self.config.chamber_radius_scale
            position = np.array((junction.center_x, junction.center_y, mean_z), dtype=float)
            stamp_points.append(
                (
                    position,
                    max(blend_radius, self.config.minimum_radius),
                    max(mean_height * 0.65, self.config.minimum_radius),
                )
            )
        return stamp_points

    def _radius_xy(self, sample: SectionSample) -> float:
        scale = self.config.tunnel_radius_scale
        if any(influence.kind == "chamber" for influence in sample.junction_influences):
            scale = max(scale, self.config.chamber_radius_scale)
        elif sample.junction_blend_weight > 0.0:
            scale = max(scale, self.config.junction_radius_scale)
        return max(sample.tube_width * 0.5 * scale, self.config.minimum_radius)

    def _radius_z(self, sample: SectionSample) -> float:
        return max(sample.tube_height * 0.5 * self.config.tunnel_radius_scale, self.config.minimum_radius)

    def _stamp_sample_chain(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        samples: tuple[SectionSample, ...],
    ) -> None:
        for start, end in zip(samples, samples[1:]):
            self._stamp_capsule(
                density=density,
                origin=origin,
                start=np.array((start.x, start.y, start.z), dtype=float),
                end=np.array((end.x, end.y, end.z), dtype=float),
                start_radius_xy=self._radius_xy(start),
                end_radius_xy=self._radius_xy(end),
                start_radius_z=self._radius_z(start),
                end_radius_z=self._radius_z(end),
            )
        for sample in (samples[0], samples[-1]):
            self._stamp_ellipsoid(
                density=density,
                origin=origin,
                center=np.array((sample.x, sample.y, sample.z), dtype=float),
                radius_xy=self._radius_xy(sample),
                radius_z=self._radius_z(sample),
            )

    def _stamp_capsule(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        start_radius_xy: float,
        end_radius_xy: float,
        start_radius_z: float,
        end_radius_z: float,
    ) -> None:
        segment = end - start
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared < 1e-9:
            self._stamp_ellipsoid(
                density=density,
                origin=origin,
                center=start,
                radius_xy=max(start_radius_xy, end_radius_xy),
                radius_z=max(start_radius_z, end_radius_z),
            )
            return

        radius = max(start_radius_xy, end_radius_xy, start_radius_z, end_radius_z)
        voxel_size = self.config.voxel_size
        lower = np.maximum(
            np.floor((np.minimum(start, end) - radius - origin) / voxel_size).astype(int),
            0,
        )
        upper = np.minimum(
            np.ceil((np.maximum(start, end) + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return

        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing="ij")

        point_x = x_grid - start[0]
        point_y = y_grid - start[1]
        point_z = z_grid - start[2]
        projection = (
            point_x * segment[0] + point_y * segment[1] + point_z * segment[2]
        ) / segment_length_squared
        projection = np.clip(projection, 0.0, 1.0)

        closest_x = start[0] + projection * segment[0]
        closest_y = start[1] + projection * segment[1]
        closest_z = start[2] + projection * segment[2]
        radius_xy = (1.0 - projection) * start_radius_xy + projection * end_radius_xy
        radius_z = (1.0 - projection) * start_radius_z + projection * end_radius_z
        dx = (x_grid - closest_x) / np.maximum(radius_xy, 1e-6)
        dy = (y_grid - closest_y) / np.maximum(radius_xy, 1e-6)
        dz = (z_grid - closest_z) / np.maximum(radius_z, 1e-6)
        stamp_density = 1.0 - np.sqrt(dx * dx + dy * dy + dz * dz)
        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, stamp_density.astype(np.float32), out=region)

    def _stamp_ellipsoid(
        self,
        *,
        density: np.ndarray,
        origin: np.ndarray,
        center: np.ndarray,
        radius_xy: float,
        radius_z: float,
    ) -> None:
        radius = max(radius_xy, radius_z)
        voxel_size = self.config.voxel_size
        lower = np.maximum(np.floor((center - radius - origin) / voxel_size).astype(int), 0)
        upper = np.minimum(
            np.ceil((center + radius - origin) / voxel_size).astype(int) + 1,
            np.array(density.shape, dtype=int),
        )
        if np.any(upper <= lower):
            return
        x_values = origin[0] + np.arange(lower[0], upper[0]) * voxel_size
        y_values = origin[1] + np.arange(lower[1], upper[1]) * voxel_size
        z_values = origin[2] + np.arange(lower[2], upper[2]) * voxel_size
        dx = (x_values[:, None, None] - center[0]) / max(radius_xy, 1e-6)
        dy = (y_values[None, :, None] - center[1]) / max(radius_xy, 1e-6)
        dz = (z_values[None, None, :] - center[2]) / max(radius_z, 1e-6)
        stamp_density = 1.0 - np.sqrt(dx * dx + dy * dy + dz * dz)
        region = density[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        np.maximum(region, stamp_density.astype(np.float32), out=region)

    def _march_chunks(self, voxel_grid: VoxelGrid) -> list[GeometryChunkMesh]:
        meshes: list[GeometryChunkMesh] = []
        chunk_size = max(int(self.config.chunk_size), 4)
        nx, ny, nz = voxel_grid.shape
        for x_start in range(0, nx - 1, chunk_size):
            x_end = min(x_start + chunk_size, nx - 1)
            for y_start in range(0, ny - 1, chunk_size):
                y_end = min(y_start + chunk_size, ny - 1)
                for z_start in range(0, nz - 1, chunk_size):
                    z_end = min(z_start + chunk_size, nz - 1)
                    chunk_density = voxel_grid.density[
                        x_start : x_end + 1,
                        y_start : y_end + 1,
                        z_start : z_end + 1,
                    ]
                    if (
                        np.all(chunk_density < voxel_grid.iso_level)
                        or np.all(chunk_density >= voxel_grid.iso_level)
                    ):
                        continue
                    mesh = self._march_chunk(
                        chunk_id=len(meshes),
                        voxel_grid=voxel_grid,
                        bounds=(x_start, x_end, y_start, y_end, z_start, z_end),
                    )
                    if mesh.faces:
                        meshes.append(mesh)
        return meshes

    def _march_chunk(
        self,
        *,
        chunk_id: int,
        voxel_grid: VoxelGrid,
        bounds: tuple[int, int, int, int, int, int],
    ) -> GeometryChunkMesh:
        x_start, x_end, y_start, y_end, z_start, z_end = bounds
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        density = voxel_grid.density
        origin = np.array(voxel_grid.origin, dtype=float)
        voxel_size = voxel_grid.voxel_size

        for ix in range(x_start, x_end):
            for iy in range(y_start, y_end):
                for iz in range(z_start, z_end):
                    corner_values = np.array(
                        [density[ix + dx, iy + dy, iz + dz] for dx, dy, dz in self._CUBE_OFFSETS],
                        dtype=float,
                    )
                    if np.all(corner_values < voxel_grid.iso_level) or np.all(corner_values >= voxel_grid.iso_level):
                        continue
                    corner_positions = np.array(
                        [
                            origin + np.array((ix + dx, iy + dy, iz + dz), dtype=float) * voxel_size
                            for dx, dy, dz in self._CUBE_OFFSETS
                        ],
                        dtype=float,
                    )
                    self._polygonize_cube(
                        corner_positions=corner_positions,
                        corner_values=corner_values,
                        iso_level=voxel_grid.iso_level,
                        vertices=vertices,
                        faces=faces,
                    )

        return GeometryChunkMesh(
            chunk_id=chunk_id,
            grid_bounds=bounds,
            vertices=tuple(vertices),
            faces=tuple(faces),
        )

    def _polygonize_cube(
        self,
        *,
        corner_positions: np.ndarray,
        corner_values: np.ndarray,
        iso_level: float,
        vertices: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
    ) -> None:
        for tetra in self._TETRAHEDRA:
            positions = corner_positions[list(tetra)]
            values = corner_values[list(tetra)]
            intersections: list[np.ndarray] = []
            for first, second in self._TETRA_EDGES:
                first_inside = values[first] >= iso_level
                second_inside = values[second] >= iso_level
                if first_inside == second_inside:
                    continue
                intersections.append(
                    self._interpolate_vertex(
                        positions[first],
                        positions[second],
                        values[first],
                        values[second],
                        iso_level,
                    )
                )

            if len(intersections) == 3:
                base = len(vertices)
                vertices.extend(tuple(float(value) for value in point) for point in intersections)
                faces.append((base, base + 1, base + 2))
            elif len(intersections) == 4:
                base = len(vertices)
                vertices.extend(tuple(float(value) for value in point) for point in intersections)
                faces.append((base, base + 1, base + 2))
                faces.append((base, base + 2, base + 3))

    @staticmethod
    def _interpolate_vertex(
        first_position: np.ndarray,
        second_position: np.ndarray,
        first_value: float,
        second_value: float,
        iso_level: float,
    ) -> np.ndarray:
        denominator = second_value - first_value
        if abs(denominator) < 1e-12:
            return (first_position + second_position) * 0.5
        ratio = float(np.clip((iso_level - first_value) / denominator, 0.0, 1.0))
        return first_position + ratio * (second_position - first_position)

    def _assemble_chunks(
        self,
        chunk_meshes: list[GeometryChunkMesh],
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        vertex_lookup: dict[tuple[int, int, int], int] = {}
        quantize = max(int(round(1.0 / max(self.config.weld_tolerance, 1e-12))), 1)
        for mesh in chunk_meshes:
            index_map: dict[int, int] = {}
            for local_index, vertex in enumerate(mesh.vertices):
                key = tuple(int(round(value * quantize)) for value in vertex)
                if key not in vertex_lookup:
                    vertex_lookup[key] = len(vertices)
                    vertices.append(vertex)
                index_map[local_index] = vertex_lookup[key]
            for a, b, c in mesh.faces:
                face = (index_map[a], index_map[b], index_map[c])
                if len(set(face)) == 3:
                    faces.append(face)
        return tuple(vertices), tuple(faces)

    @staticmethod
    def _count_components(faces: tuple[tuple[int, int, int], ...]) -> int:
        if not faces:
            return 0
        vertex_to_faces: dict[int, list[int]] = defaultdict(list)
        for face_index, face in enumerate(faces):
            for vertex_index in face:
                vertex_to_faces[vertex_index].append(face_index)
        visited: set[int] = set()
        component_count = 0
        for start_face in range(len(faces)):
            if start_face in visited:
                continue
            component_count += 1
            stack = [start_face]
            visited.add(start_face)
            while stack:
                face_index = stack.pop()
                for vertex_index in faces[face_index]:
                    for neighbor_face in vertex_to_faces[vertex_index]:
                        if neighbor_face in visited:
                            continue
                        visited.add(neighbor_face)
                        stack.append(neighbor_face)
        return component_count


__all__ = [
    "CaveGeometry",
    "GeometryChunkMesh",
    "GeometryConfig",
    "GeometryGenerator",
    "VoxelGrid",
]

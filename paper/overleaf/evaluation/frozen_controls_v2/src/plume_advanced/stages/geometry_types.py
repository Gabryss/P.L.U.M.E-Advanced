"""Shared dataclasses for Stage D voxel geometry generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
from scipy import ndimage

from plume_advanced.stages.events import GeologicalEventMesh


@dataclass(frozen=True)
class GeometryConfig:
    """Parameters controlling the Stage-D voxel and meshing pipeline."""

    random_seed: int | None = None
    voxel_size: float = 6.0
    resolution_policy: str = "fixed"
    resolution_quality: str = "standard"
    characteristic_passage_width_m: float = 12.0
    target_samples_across_passage: float = 2.0
    storage_mode: str = "auto"
    max_dense_voxels: int = 80_000_000
    density_margin: float = 30.0
    chunk_size: int = 64
    iso_level: float = 0.0
    tunnel_radius_scale: float = 1.2
    chamber_radius_scale: float = 1.7
    junction_radius_scale: float = 1.7
    minimum_radius: float = 3.5
    use_section_profiles: bool = True
    wall_roughness_amplitude: float = 0.18
    wall_roughness_frequency: float = 0.16
    wall_roughness_blend: float = 0.75
    floor_roughness_scale: float = 0.25
    roof_roughness_scale: float = 1.0
    # Inward accretion relief, in metres; zero preserves earlier scenarios.
    surface_wall_relief_m: float = 0.0
    surface_roof_relief_m: float = 0.0
    surface_floor_relief_m: float = 0.0
    surface_crust_relief_m: float = 0.0
    surface_feature_scale_m: float = 1.0
    surface_normal_filter_voxels: float = 1.2
    junction_irregularity_amplitude: float = 0.18
    junction_irregularity_frequency: float = 0.11
    structural_event_blend: float = 0.35
    weld_tolerance: float = 1e-5
    strict_texture_loading: bool = True
    embedded_texture_max_size: int = 1024
    cave_normal_scale: float = 2.0
    cave_smoothing_iterations: int = 4
    cave_displacement_scale_m: float = 0.12
    cave_displacement_midlevel: float = 0.5
    cave_diffuse_texture: str = "texture/dark_rock_8k/textures/dark_rock_diff_8k.jpg"
    cave_normal_texture: str = "texture/dark_rock_8k/textures/dark_rock_nor_gl_8k.exr"
    cave_roughness_texture: str = "texture/dark_rock_8k/textures/dark_rock_rough_8k.exr"
    cave_displacement_texture: str = "texture/dark_rock_8k/textures/dark_rock_disp_8k.png"

    @property
    def characteristic_samples_across_passage(self) -> float:
        """Return the resolved nominal passage sampling density."""

        return self.characteristic_passage_width_m / max(self.voxel_size, 1e-9)


class _DensitySurface(Protocol):
    """Surface-query contract shared by dense and sparse density storage."""

    @property
    def voxel_size(self) -> float: ...

    @property
    def iso_level(self) -> float: ...

    def sample_density(self, point: tuple[float, float, float] | np.ndarray) -> float: ...

    def surface_normal(
        self, point: tuple[float, float, float] | np.ndarray
    ) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class VoxelGrid:
    """Density field that stores the carved cave volume."""

    origin: tuple[float, float, float]
    voxel_size: float
    density: np.ndarray
    iso_level: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self.density.shape[0]),
            int(self.density.shape[1]),
            int(self.density.shape[2]),
        )

    @property
    def carved_voxel_count(self) -> int:
        return int(np.count_nonzero(self.density >= self.iso_level))

    @property
    def component_count(self) -> int:
        return _count_voxel_components(self)

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lower = np.asarray(self.origin, dtype=float)
        upper = lower + (np.asarray(self.shape, dtype=float) - 1.0) * self.voxel_size
        return (
            (float(lower[0]), float(lower[1]), float(lower[2])),
            (float(upper[0]), float(upper[1]), float(upper[2])),
        )

    def contains(self, point: tuple[float, float, float] | np.ndarray) -> bool:
        lower, upper = self.bounds
        position = np.asarray(point, dtype=float)
        return bool(
            np.all(position >= np.asarray(lower))
            and np.all(position <= np.asarray(upper))
        )

    def sample_density(
        self,
        point: tuple[float, float, float] | np.ndarray,
    ) -> float:
        """Trilinearly sample the generated cave density."""

        position = np.asarray(point, dtype=float)
        fractional = (position - np.asarray(self.origin, dtype=float)) / self.voxel_size
        maximum = np.asarray(self.shape, dtype=float) - 1.0
        if np.any(fractional < 0.0) or np.any(fractional > maximum):
            return float(self.iso_level - 1.0)

        lower = np.floor(fractional).astype(int)
        upper = np.minimum(lower + 1, np.asarray(self.shape, dtype=int) - 1)
        weight = fractional - lower
        x0, y0, z0 = lower
        x1, y1, z1 = upper
        wx, wy, wz = weight
        density = self.density

        c000 = float(density[x0, y0, z0])
        c100 = float(density[x1, y0, z0])
        c010 = float(density[x0, y1, z0])
        c110 = float(density[x1, y1, z0])
        c001 = float(density[x0, y0, z1])
        c101 = float(density[x1, y0, z1])
        c011 = float(density[x0, y1, z1])
        c111 = float(density[x1, y1, z1])
        c00 = c000 * (1.0 - wx) + c100 * wx
        c10 = c010 * (1.0 - wx) + c110 * wx
        c01 = c001 * (1.0 - wx) + c101 * wx
        c11 = c011 * (1.0 - wx) + c111 * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        return float(c0 * (1.0 - wz) + c1 * wz)

    def surface_normal(
        self: _DensitySurface,
        point: tuple[float, float, float] | np.ndarray,
    ) -> tuple[float, float, float]:
        """Estimate the inward-facing cave normal from the density gradient."""

        position = np.asarray(point, dtype=float)
        step = max(0.5 * self.voxel_size, 1e-6)
        gradient = np.empty(3, dtype=float)
        for axis in range(3):
            offset = np.zeros(3, dtype=float)
            offset[axis] = step
            gradient[axis] = (
                self.sample_density(position + offset)
                - self.sample_density(position - offset)
            ) / (2.0 * step)
        length = float(np.linalg.norm(gradient))
        if length <= 1e-12:
            return (0.0, 0.0, 1.0)
        normal = gradient / length
        return (float(normal[0]), float(normal[1]), float(normal[2]))

    def raycast_isosurface(
        self: _DensitySurface,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        max_distance: float,
        *,
        step: float | None = None,
    ) -> "SurfaceHit | None":
        """Find the first inside-to-solid isosurface crossing along a ray."""

        start = np.asarray(origin, dtype=float)
        ray = np.asarray(direction, dtype=float)
        ray_length = float(np.linalg.norm(ray))
        if ray_length <= 1e-12 or max_distance <= 0.0:
            return None
        ray /= ray_length
        increment = step or max(0.25 * self.voxel_size, 0.05)
        increment = min(max(float(increment), 0.01), float(max_distance))

        previous_distance = 0.0
        previous_density = self.sample_density(start)
        entered_void = previous_density >= self.iso_level
        distance = increment
        while distance <= max_distance + 1e-9:
            point = start + ray * min(distance, max_distance)
            density = self.sample_density(point)
            if not entered_void and density >= self.iso_level:
                entered_void = True
            elif entered_void and density < self.iso_level:
                low = previous_distance
                high = min(distance, max_distance)
                for _ in range(12):
                    middle = 0.5 * (low + high)
                    middle_density = self.sample_density(start + ray * middle)
                    if middle_density >= self.iso_level:
                        low = middle
                    else:
                        high = middle
                hit_distance = 0.5 * (low + high)
                hit_position = start + ray * hit_distance
                return SurfaceHit(
                    position=(
                        float(hit_position[0]),
                        float(hit_position[1]),
                        float(hit_position[2]),
                    ),
                    normal=self.surface_normal(hit_position),
                    distance=float(hit_distance),
                )
            previous_distance = min(distance, max_distance)
            previous_density = density
            if math.isclose(previous_distance, max_distance):
                break
            distance = min(distance + increment, max_distance)
        return None


@dataclass
class TiledVoxelGrid:
    """Sparse density field stored as overlapping active tiles."""

    origin: tuple[float, float, float]
    voxel_size: float
    global_shape: tuple[int, int, int]
    iso_level: float
    tile_size: int
    tiles: dict[tuple[int, int, int], np.ndarray]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.global_shape

    @property
    def storage_nbytes(self) -> int:
        return sum(tile.nbytes for tile in self.tiles.values())

    @property
    def carved_voxel_count(self) -> int:
        count = 0
        for key, tile in self.tiles.items():
            start = np.asarray(key, dtype=int) * self.tile_size
            stop = np.minimum(start + self.tile_size, np.asarray(self.shape) - 1)
            slices = tuple(
                slice(0, int(end - begin) + (1 if end == self.shape[axis] - 1 else 0))
                for axis, (begin, end) in enumerate(zip(start, stop, strict=True))
            )
            count += int(np.count_nonzero(tile[slices] >= self.iso_level))
        return count

    @property
    def component_count(self) -> int:
        return _count_tiled_components(self)

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lower = np.asarray(self.origin, dtype=float)
        upper = lower + (np.asarray(self.shape, dtype=float) - 1.0) * self.voxel_size
        return (
            (float(lower[0]), float(lower[1]), float(lower[2])),
            (float(upper[0]), float(upper[1]), float(upper[2])),
        )

    @property
    def active_tile_count(self) -> int:
        return len(self.tiles)

    def contains(self, point: tuple[float, float, float] | np.ndarray) -> bool:
        lower, upper = self.bounds
        position = np.asarray(point, dtype=float)
        return bool(np.all(position >= lower) and np.all(position <= upper))

    def tile_bounds(
        self,
        key: tuple[int, int, int],
    ) -> tuple[int, int, int, int, int, int]:
        start = np.asarray(key, dtype=int) * self.tile_size
        end = np.minimum(start + self.tile_size, np.asarray(self.shape) - 1)
        return (
            int(start[0]),
            int(end[0]),
            int(start[1]),
            int(end[1]),
            int(start[2]),
            int(end[2]),
        )

    def _density_at(self, index: np.ndarray) -> float:
        maximum_key = np.maximum(
            np.ceil((np.asarray(self.shape) - 1) / self.tile_size).astype(int) - 1,
            0,
        )
        primary = np.minimum(index // self.tile_size, maximum_key)
        axis_candidates: list[tuple[int, ...]] = []
        for axis in range(3):
            values = [int(primary[axis])]
            if index[axis] % self.tile_size == 0 and primary[axis] > 0:
                values.append(int(primary[axis] - 1))
            axis_candidates.append(tuple(values))
        for candidate in product(*axis_candidates):
            key = (int(candidate[0]), int(candidate[1]), int(candidate[2]))
            tile = self.tiles.get(key)
            if tile is None:
                continue
            local = index - np.asarray(key, dtype=int) * self.tile_size
            if np.all(local >= 0) and np.all(local < np.asarray(tile.shape)):
                return float(tile[tuple(local)])
        return float(self.iso_level - 1.0)

    def synchronize_halos(self) -> None:
        """Give every shared sample the value used by density queries.

        Independent tile stamping can disagree near rounded profile stations.
        The lexicographically greatest available tile owns an overlap, matching
        _density_at. Descending propagation covers faces, edges and corners,
        including sparse layouts with a missing primary tile.
        """
        offsets = tuple(product((-1, 0, 1), repeat=3))
        for key in sorted(self.tiles, reverse=True):
            source = self.tiles[key]
            start = np.asarray(key) * self.tile_size
            stop = start + np.asarray(source.shape)
            for offset in offsets:
                neighbor = tuple(k + d for k, d in zip(key, offset, strict=True))
                if neighbor >= key or neighbor not in self.tiles:
                    continue
                target = self.tiles[neighbor]
                target_start = np.asarray(neighbor) * self.tile_size
                lower = np.maximum(start, target_start)
                upper = np.minimum(stop, target_start + np.asarray(target.shape))
                if np.any(lower >= upper):
                    continue
                source_slice = tuple(slice(int(a), int(b)) for a, b in zip(
                    lower - start, upper - start, strict=True,
                ))
                target_slice = tuple(slice(int(a), int(b)) for a, b in zip(
                    lower - target_start, upper - target_start, strict=True,
                ))
                target[target_slice] = source[source_slice]

    def sample_density(
        self,
        point: tuple[float, float, float] | np.ndarray,
    ) -> float:
        position = np.asarray(point, dtype=float)
        fractional = (position - np.asarray(self.origin, dtype=float)) / self.voxel_size
        maximum = np.asarray(self.shape, dtype=float) - 1.0
        if np.any(fractional < 0.0) or np.any(fractional > maximum):
            return float(self.iso_level - 1.0)
        lower = np.floor(fractional).astype(int)
        upper = np.minimum(lower + 1, np.asarray(self.shape) - 1)
        weight = fractional - lower
        values = np.empty((2, 2, 2), dtype=float)
        for ix, x_index in enumerate((lower[0], upper[0])):
            for iy, y_index in enumerate((lower[1], upper[1])):
                for iz, z_index in enumerate((lower[2], upper[2])):
                    values[ix, iy, iz] = self._density_at(
                        np.asarray((x_index, y_index, z_index), dtype=int)
                    )
        wx, wy, wz = weight
        c00 = values[0, 0, 0] * (1.0 - wx) + values[1, 0, 0] * wx
        c10 = values[0, 1, 0] * (1.0 - wx) + values[1, 1, 0] * wx
        c01 = values[0, 0, 1] * (1.0 - wx) + values[1, 0, 1] * wx
        c11 = values[0, 1, 1] * (1.0 - wx) + values[1, 1, 1] * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        return float(c0 * (1.0 - wz) + c1 * wz)

    surface_normal = VoxelGrid.surface_normal
    raycast_isosurface = VoxelGrid.raycast_isosurface


@dataclass(frozen=True)
class SurfaceHit:
    """One intersection with the generated cave boundary."""

    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    distance: float


@dataclass(frozen=True)
class GeometryChunkMesh:
    """One mesh generated from a 3D density chunk."""

    chunk_id: int
    grid_bounds: tuple[int, int, int, int, int, int]
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


@dataclass(frozen=True)
class SurfaceTextureFrame:
    """Route-local frame used to map the cave wall without global stretching."""

    segment_id: int
    center: tuple[float, float, float]
    tangent: tuple[float, float, float]
    normal: tuple[float, float, float]
    binormal: tuple[float, float, float]
    longitudinal_m: float
    longitudinal_rate: float = 1.0
    profile_points: tuple[tuple[float, float], ...] = ()
    profile_perimeter_m: float = 0.0


@dataclass(frozen=True)
class CaveGeometry:
    """Stage-D output for voxel-stamped cave geometry."""

    config: GeometryConfig
    voxel_grid: VoxelGrid | TiledVoxelGrid
    chunk_meshes: tuple[GeometryChunkMesh, ...]
    assembled_vertices: tuple[tuple[float, float, float], ...]
    assembled_faces: tuple[tuple[int, int, int], ...]
    component_count: int
    stamped_sample_count: int
    stamped_segment_ids: tuple[int, ...]
    minimum_section_width_m: float = 0.0
    protected_route_points: tuple[tuple[float, float, float], ...] = ()
    surface_texture_frames: tuple[SurfaceTextureFrame, ...] = ()
    event_meshes: tuple[GeologicalEventMesh, ...] = ()
    structural_event_ids: tuple[int, ...] = ()
    # Additive Stage-D junction quality report.  A tuple keeps the public
    # dataclass immutable while allowing future metrics without schema breaks.
    junction_report: tuple[tuple[str, float], ...] = ()
    # Per-junction immutable records (key/value tuples) preserve local
    # outliers while keeping the existing summary API backwards compatible.
    junction_records: tuple[tuple[tuple[str, object], ...], ...] = ()
    stability_records: tuple[tuple[tuple[str, object], ...], ...] = ()
    preserved_pillar_columns: int = 0

    @property
    def meshes(self) -> tuple[GeometryChunkMesh, ...]:
        """Compatibility alias for callers that render/export geometry meshes."""

        return self.chunk_meshes

    def summary(self) -> dict[str, float]:
        summary = {
            "mesh_count": float(len(self.chunk_meshes)),
            "chunk_mesh_count": float(len(self.chunk_meshes)),
            "event_mesh_count": float(len(self.event_meshes)),
            "structural_event_count": float(len(self.structural_event_ids)),
            "junction_record_count": float(len(self.junction_records)),
            "stability_collapse_count": float(sum(
                bool(dict(record)["failed"]) for record in self.stability_records
            )),
            "stability_assessment_count": float(len(self.stability_records)),
            "preserved_pillar_column_count": float(self.preserved_pillar_columns),
            "stamped_segment_count": float(len(self.stamped_segment_ids)),
            "stamped_sample_count": float(self.stamped_sample_count),
            "voxel_size_m": float(self.voxel_grid.voxel_size),
            "characteristic_passage_samples": float(
                self.config.characteristic_samples_across_passage
            ),
            "minimum_section_width_samples": float(
                self.minimum_section_width_m
                / max(self.voxel_grid.voxel_size, 1e-9)
            ),
            "voxel_count": float(np.prod(self.voxel_grid.shape)),
            "density_memory_mib": float(
                getattr(
                    self.voxel_grid,
                    "storage_nbytes",
                    getattr(getattr(self.voxel_grid, "density", None), "nbytes", 0),
                )
                / (1024.0 * 1024.0)
            ),
            "active_tile_count": float(
                getattr(self.voxel_grid, "active_tile_count", 1)
            ),
            "carved_voxel_count": float(self.voxel_grid.carved_voxel_count),
            "voxel_component_count": float(self.voxel_grid.component_count),
            "component_count": float(self.component_count),
            "vertex_count": float(len(self.assembled_vertices)),
            "face_count": float(len(self.assembled_faces)),
            "event_vertex_count": float(sum(mesh.vertex_count for mesh in self.event_meshes)),
            "event_face_count": float(sum(mesh.face_count for mesh in self.event_meshes)),
            "export_vertex_count": float(
                len(self.assembled_vertices)
                + sum(mesh.vertex_count for mesh in self.event_meshes)
            ),
            "export_face_count": float(
                len(self.assembled_faces)
                + sum(mesh.face_count for mesh in self.event_meshes)
            ),
        }
        summary.update({str(key): float(value) for key, value in self.junction_report})
        return summary


def _count_voxel_components(voxel_grid: VoxelGrid) -> int:
    carved = voxel_grid.density >= voxel_grid.iso_level
    if not bool(np.any(carved)):
        return 0
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    _labels, component_count = ndimage.label(carved, structure=structure)
    return int(component_count)


def _count_tiled_components(voxel_grid: TiledVoxelGrid) -> int:
    """Count components using local labels joined across shared tile boundaries."""

    parent: dict[int, int] = {}
    boundary_labels: dict[tuple[int, int, int], int] = {}
    next_label = 1

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while value != root:
            previous = parent[value]
            parent[value] = root
            value = previous
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    for key, tile in voxel_grid.tiles.items():
        labels, count = ndimage.label(tile >= voxel_grid.iso_level, structure=structure)
        if count == 0:
            continue
        offset = next_label - 1
        for label in range(1, count + 1):
            parent[offset + label] = offset + label
        start = np.asarray(key, dtype=int) * voxel_grid.tile_size
        boundary = np.zeros(tile.shape, dtype=bool)
        for axis in range(3):
            low: list[slice | int] = [slice(None)] * 3
            high: list[slice | int] = [slice(None)] * 3
            low[axis] = 0
            high[axis] = -1
            boundary[tuple(low)] = True
            boundary[tuple(high)] = True
        for local in np.argwhere(boundary & (labels > 0)):
            position = start + local
            global_index = (
                int(position[0]),
                int(position[1]),
                int(position[2]),
            )
            label = offset + int(labels[tuple(local)])
            previous = boundary_labels.get(global_index)
            if previous is None:
                boundary_labels[global_index] = label
            else:
                union(previous, label)
        next_label += count
    return len({find(label) for label in parent})


__all__ = [
    "CaveGeometry",
    "GeometryChunkMesh",
    "GeometryConfig",
    "SurfaceTextureFrame",
    "SurfaceHit",
    "TiledVoxelGrid",
    "VoxelGrid",
]

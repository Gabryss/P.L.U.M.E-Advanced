"""Shared dataclasses for Stage D voxel geometry generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from stages.events import GeologicalEventMesh


@dataclass(frozen=True)
class GeometryConfig:
    """Parameters controlling the Stage-D voxel and meshing pipeline."""

    random_seed: int | None = None
    voxel_size: float = 6.0
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
    junction_irregularity_amplitude: float = 0.18
    junction_irregularity_frequency: float = 0.11
    weld_tolerance: float = 1e-5


@dataclass(frozen=True)
class VoxelGrid:
    """Density field that stores the carved cave volume."""

    origin: tuple[float, float, float]
    voxel_size: float
    density: np.ndarray
    iso_level: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.density.shape)

    @property
    def carved_voxel_count(self) -> int:
        return int(np.count_nonzero(self.density >= self.iso_level))

    @property
    def component_count(self) -> int:
        return _count_voxel_components(self)


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
class CaveGeometry:
    """Stage-D output for voxel-stamped cave geometry."""

    config: GeometryConfig
    voxel_grid: VoxelGrid
    chunk_meshes: tuple[GeometryChunkMesh, ...]
    assembled_vertices: tuple[tuple[float, float, float], ...]
    assembled_faces: tuple[tuple[int, int, int], ...]
    component_count: int
    stamped_sample_count: int
    stamped_segment_ids: tuple[int, ...]
    event_meshes: tuple[GeologicalEventMesh, ...] = ()

    @property
    def meshes(self) -> tuple[GeometryChunkMesh, ...]:
        """Compatibility alias for callers that render/export geometry meshes."""

        return self.chunk_meshes

    def summary(self) -> dict[str, float]:
        return {
            "mesh_count": float(len(self.chunk_meshes)),
            "chunk_mesh_count": float(len(self.chunk_meshes)),
            "event_mesh_count": float(len(self.event_meshes)),
            "stamped_segment_count": float(len(self.stamped_segment_ids)),
            "stamped_sample_count": float(self.stamped_sample_count),
            "voxel_count": float(np.prod(self.voxel_grid.shape)),
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


def _count_voxel_components(voxel_grid: VoxelGrid) -> int:
    carved = voxel_grid.density >= voxel_grid.iso_level
    if not bool(np.any(carved)):
        return 0
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    _labels, component_count = ndimage.label(carved, structure=structure)
    return int(component_count)


__all__ = [
    "CaveGeometry",
    "GeometryChunkMesh",
    "GeometryConfig",
    "VoxelGrid",
]

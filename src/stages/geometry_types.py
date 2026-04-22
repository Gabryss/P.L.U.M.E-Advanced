"""Shared dataclasses for Stage D geometry generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryConfig:
    """Parameters controlling the Stage-D geometry pipeline."""

    random_seed: int | None = None
    junction_exclusion_weight: float = 0.32
    minimum_span_samples: int = 3
    exclude_crossing_segments: bool = True
    exclude_underpass_segments: bool = True
    enable_junction_patches: bool = True
    junction_min_segments: int = 2
    junction_patch_scale: float = 0.32
    chamber_patch_scale: float = 0.46
    enable_skylight: bool = True
    skylight_bottom_radius_scale: float = 0.38
    skylight_top_radius_scale: float = 0.62
    skylight_ring_count: int = 7
    skylight_drift_ratio: float = 0.26
    skylight_rim_jaggedness: float = 0.18
    skylight_surface_margin: float = 0.8
    weld_tolerance: float = 1e-5


@dataclass(frozen=True)
class SegmentGeometrySpan:
    """One sweepable mesh span extracted from a segment section field."""

    mesh_id: int
    segment_id: int
    segment_kind: str
    is_connector: bool
    sample_indices: tuple[int, ...]
    connected_junction_ids: tuple[int, ...]
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    ring_vertex_count: int

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


@dataclass(frozen=True)
class JunctionGeometryPatch:
    """One stitched junction patch connecting multiple swept segment openings."""

    patch_id: int
    junction_id: int
    junction_kind: str
    segment_ids: tuple[int, ...]
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    ring_vertex_count: int

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


@dataclass(frozen=True)
class SkylightGeometry:
    """One skylight shaft event anchored to the highest internal cave sample."""

    skylight_id: int
    anchor_segment_id: int
    anchor_sample_index: int
    anchor_vertex: tuple[float, float, float]
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    ring_vertex_count: int
    top_center: tuple[float, float, float]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)


@dataclass(frozen=True)
class CaveGeometry:
    """Stage-D output for swept geometry, junction patches, and openings."""

    config: GeometryConfig
    meshes: tuple[SegmentGeometrySpan, ...]
    junction_patches: tuple[JunctionGeometryPatch, ...]
    skylights: tuple[SkylightGeometry, ...]
    assembled_vertices: tuple[tuple[float, float, float], ...]
    assembled_faces: tuple[tuple[int, int, int], ...]
    component_count: int
    excluded_segment_ids: tuple[int, ...]
    excluded_junction_ids: tuple[int, ...]

    def summary(self) -> dict[str, float]:
        vertex_count = (
            sum(mesh.vertex_count for mesh in self.meshes)
            + sum(patch.vertex_count for patch in self.junction_patches)
            + sum(skylight.vertex_count for skylight in self.skylights)
        )
        face_count = (
            sum(mesh.face_count for mesh in self.meshes)
            + sum(patch.face_count for patch in self.junction_patches)
            + sum(skylight.face_count for skylight in self.skylights)
        )
        swept_segment_ids = {mesh.segment_id for mesh in self.meshes}
        return {
            "mesh_count": float(len(self.meshes)),
            "swept_segment_count": float(len(swept_segment_ids)),
            "junction_patch_count": float(len(self.junction_patches)),
            "skylight_count": float(len(self.skylights)),
            "component_count": float(self.component_count),
            "excluded_segment_count": float(len(self.excluded_segment_ids)),
            "excluded_junction_count": float(len(self.excluded_junction_ids)),
            "vertex_count": float(vertex_count),
            "face_count": float(face_count),
        }

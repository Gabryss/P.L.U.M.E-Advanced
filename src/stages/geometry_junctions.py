"""Non-crossing junction patch generation for Stage D geometry."""

from __future__ import annotations

import numpy as np

from stages.geometry_types import GeometryConfig, JunctionGeometryPatch, SegmentGeometrySpan
from stages.network import CaveNetwork
from stages.section_field import SectionField, SectionSample, SegmentSectionField


class JunctionPatchBuilder:
    """Build non-crossing split/merge/chamber junction patches."""

    def __init__(self, config: GeometryConfig) -> None:
        self.config = config

    def build_patches(
        self,
        *,
        cave_network: CaveNetwork,
        section_field: SectionField,
        field_lookup: dict[int, SegmentSectionField],
        segment_lookup: dict[int, object],
        meshes: list[SegmentGeometrySpan],
        excluded_junction_ids: set[int],
    ) -> list[JunctionGeometryPatch]:
        swept_segment_ids = {mesh.segment_id for mesh in meshes}
        patches: list[JunctionGeometryPatch] = []
        for junction in cave_network.junctions:
            if len(junction.segment_ids) < self.config.junction_min_segments:
                continue

            boundary_samples: list[tuple[int, SectionSample, tuple[tuple[float, float, float], ...]]] = []
            for mesh in meshes:
                if mesh.segment_id not in junction.segment_ids:
                    continue
                if mesh.segment_id not in swept_segment_ids:
                    continue
                segment = segment_lookup[mesh.segment_id]
                if getattr(segment, "z_level", 0) != 0:
                    continue
                if getattr(segment, "kind", "") == "underpass":
                    continue
                segment_field = field_lookup.get(mesh.segment_id)
                if segment_field is None:
                    continue
                boundary = self._select_junction_boundary_sample(mesh, segment_field, junction)
                if boundary is None:
                    continue
                boundary_samples.append((mesh.segment_id, boundary[0], boundary[1]))

            if len(boundary_samples) < self.config.junction_min_segments:
                excluded_junction_ids.add(junction.junction_id)
                continue

            patch = self._build_junction_patch(
                patch_id=len(patches),
                junction=junction,
                boundary_samples=boundary_samples,
            )
            if patch is None:
                excluded_junction_ids.add(junction.junction_id)
                continue
            patches.append(patch)
        return patches

    def _select_junction_boundary_sample(
        self,
        mesh: SegmentGeometrySpan,
        segment_field: SegmentSectionField,
        junction,
    ) -> tuple[SectionSample, tuple[tuple[float, float, float], ...]] | None:
        sample_lookup = {sample.index: sample for sample in segment_field.samples}
        candidates: list[tuple[SectionSample, tuple[tuple[float, float, float], ...]]] = []
        if mesh.sample_indices:
            if mesh.sample_indices[0] in sample_lookup:
                candidates.append(
                    (
                        sample_lookup[mesh.sample_indices[0]],
                        tuple(mesh.vertices[: mesh.ring_vertex_count]),
                    )
                )
            if mesh.sample_indices[-1] in sample_lookup:
                candidates.append(
                    (
                        sample_lookup[mesh.sample_indices[-1]],
                        tuple(mesh.vertices[-mesh.ring_vertex_count :]),
                    )
                )
        weighted_candidates = [
            (
                sample,
                ring,
                max(
                    (
                        influence.weight
                        for influence in sample.junction_influences
                        if influence.junction_id == junction.junction_id
                    ),
                    default=0.0,
                ),
            )
            for sample, ring in candidates
        ]
        positive_candidates = [item for item in weighted_candidates if item[2] > 0.0]
        if positive_candidates:
            sample, ring, _weight = max(positive_candidates, key=lambda item: item[2])
            return sample, ring

        if not candidates:
            return None
        sample, ring = min(
            candidates,
            key=lambda item: float(
                np.hypot(item[0].x - junction.center_x, item[0].y - junction.center_y)
            ),
        )
        return sample, ring

    def _build_junction_patch(
        self,
        *,
        patch_id: int,
        junction,
        boundary_samples: list[tuple[int, SectionSample, tuple[tuple[float, float, float], ...]]],
    ) -> JunctionGeometryPatch | None:
        base_sample = boundary_samples[0][1]
        base_ring = boundary_samples[0][2]
        ring_vertex_count = len(base_ring)
        if ring_vertex_count < 3:
            return None
        if any(len(boundary_ring) != ring_vertex_count for _, _sample, boundary_ring in boundary_samples):
            return None

        center_z = float(np.mean([sample.z for _, sample, _boundary_ring in boundary_samples]))
        center = np.array([junction.center_x, junction.center_y, center_z], dtype=float)
        scale = (
            self.config.chamber_patch_scale
            if junction.kind == "chamber"
            else self.config.junction_patch_scale
        )

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        center_vertex_index: int | None = None
        segment_ids: list[int] = []

        for segment_id, sample, boundary_ring in boundary_samples:
            segment_ids.append(segment_id)
            inner_ring = self._build_inner_ring_vertices(sample, center, scale)
            start_offset = len(vertices)
            vertices.extend(boundary_ring)
            inner_offset = len(vertices)
            vertices.extend(inner_ring)

            for vertex_index in range(ring_vertex_count):
                next_vertex_index = (vertex_index + 1) % ring_vertex_count
                a = start_offset + vertex_index
                b = start_offset + next_vertex_index
                c = inner_offset + vertex_index
                d = inner_offset + next_vertex_index
                faces.append((a, c, b))
                faces.append((b, c, d))

            if center_vertex_index is None:
                center_vertex_index = len(vertices)
                vertices.append((float(center[0]), float(center[1]), float(center[2])))
            for vertex_index in range(ring_vertex_count):
                next_vertex_index = (vertex_index + 1) % ring_vertex_count
                a = inner_offset + vertex_index
                b = inner_offset + next_vertex_index
                faces.append((a, center_vertex_index, b))

        if center_vertex_index is None:
            return None

        return JunctionGeometryPatch(
            patch_id=patch_id,
            junction_id=junction.junction_id,
            junction_kind=junction.kind,
            segment_ids=tuple(sorted(set(segment_ids))),
            vertices=tuple(vertices),
            faces=tuple(faces),
            ring_vertex_count=ring_vertex_count,
        )

    @staticmethod
    def _build_inner_ring_vertices(
        sample: SectionSample,
        center: np.ndarray,
        scale: float,
    ) -> list[tuple[float, float, float]]:
        normal = np.array(sample.normal, dtype=float)
        binormal = np.array(sample.binormal, dtype=float)
        vertices: list[tuple[float, float, float]] = []
        for local_x, local_y in sample.profile_points[:-1]:
            vertex = center + scale * ((local_x * normal) + (local_y * binormal))
            vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
        return vertices

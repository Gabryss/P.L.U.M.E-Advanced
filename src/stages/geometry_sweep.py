"""Ordinary segment sweep logic for Stage D geometry."""

from __future__ import annotations

import numpy as np

from stages.geometry_types import GeometryConfig, SegmentGeometrySpan
from stages.section_field import SectionSample, SegmentSectionField


def build_ring_vertices(
    sample: SectionSample,
    profile_points: tuple[tuple[float, float], ...],
) -> list[tuple[float, float, float]]:
    """Lift local 2D profile points into world-space ring vertices."""

    normal = np.array(sample.normal, dtype=float)
    binormal = np.array(sample.binormal, dtype=float)
    center = np.array([sample.x, sample.y, sample.z], dtype=float)
    vertices: list[tuple[float, float, float]] = []
    for local_x, local_y in profile_points:
        vertex = center + (local_x * normal) + (local_y * binormal)
        vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
    return vertices


class SweepBuilder:
    """Build low-risk swept mesh spans from the section field."""

    def __init__(self, config: GeometryConfig) -> None:
        self.config = config

    def exclude_segment(
        self,
        segment,
        segment_field: SegmentSectionField,
        junction_lookup: dict[int, object],
    ) -> bool:
        if self.config.exclude_underpass_segments and segment.kind == "underpass":
            return True
        if self.config.exclude_crossing_segments and segment.z_level != 0:
            return True
        return False

    def build_sample_spans(
        self,
        segment_field: SegmentSectionField,
    ) -> tuple[tuple[int, ...], ...]:
        spans: list[tuple[int, ...]] = []
        current: list[int] = []
        for sample in segment_field.samples:
            if self.is_sweepable_sample(sample):
                current.append(sample.index)
                continue
            if len(current) >= self.config.minimum_span_samples:
                spans.append(tuple(current))
            current = []
        if len(current) >= self.config.minimum_span_samples:
            spans.append(tuple(current))
        return tuple(spans)

    def is_sweepable_sample(self, sample: SectionSample) -> bool:
        if sample.junction_blend_weight >= self.config.junction_exclusion_weight:
            return False
        return not any(influence.kind == "crossing" for influence in sample.junction_influences)

    def build_mesh_span(
        self,
        *,
        mesh_id: int,
        segment,
        segment_field: SegmentSectionField,
        sample_indices: tuple[int, ...],
        is_connector: bool = False,
    ) -> SegmentGeometrySpan | None:
        sample_lookup = {sample.index: sample for sample in segment_field.samples}
        samples = [sample_lookup[index] for index in sample_indices if index in sample_lookup]
        if len(samples) < self.config.minimum_span_samples:
            return None

        ring_profile = samples[0].profile_points[:-1]
        ring_vertex_count = len(ring_profile)
        if ring_vertex_count < 3:
            return None

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []

        for sample in samples:
            sample_ring_profile = sample.profile_points[:-1]
            if len(sample_ring_profile) != ring_vertex_count:
                return None
            vertices.extend(build_ring_vertices(sample, sample_ring_profile))

        for ring_index in range(len(samples) - 1):
            ring_offset = ring_index * ring_vertex_count
            next_offset = (ring_index + 1) * ring_vertex_count
            for vertex_index in range(ring_vertex_count):
                next_vertex_index = (vertex_index + 1) % ring_vertex_count
                a = ring_offset + vertex_index
                b = ring_offset + next_vertex_index
                c = next_offset + vertex_index
                d = next_offset + next_vertex_index
                faces.append((a, c, b))
                faces.append((b, c, d))

        return SegmentGeometrySpan(
            mesh_id=mesh_id,
            segment_id=segment.segment_id,
            segment_kind=segment.kind,
            is_connector=is_connector,
            sample_indices=sample_indices,
            connected_junction_ids=segment_field.connected_junction_ids,
            vertices=tuple(vertices),
            faces=tuple(faces),
            ring_vertex_count=ring_vertex_count,
        )

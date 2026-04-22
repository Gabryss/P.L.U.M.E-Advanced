"""Highest-point skylight event generation for Stage D geometry."""

from __future__ import annotations

import math

import numpy as np

from stages.geometry_sweep import build_ring_vertices
from stages.geometry_types import GeometryConfig, SkylightGeometry
from stages.section_field import SectionField, SectionSample


class SkylightBuilder:
    """Build one mostly vertical skylight at the highest cave centerline sample."""

    def __init__(self, config: GeometryConfig) -> None:
        self.config = config

    def build_skylights(
        self,
        *,
        section_field: SectionField,
        segment_lookup: dict[int, object],
        allowed_segment_ids: set[int] | None = None,
    ) -> list[SkylightGeometry]:
        if not self.config.enable_skylight:
            return []

        anchor = self._select_anchor(section_field, segment_lookup, allowed_segment_ids)
        if anchor is None:
            return []

        rng = np.random.default_rng(self.config.random_seed)
        skylight = self._build_skylight(skylight_id=0, anchor=anchor, rng=rng)
        return [] if skylight is None else [skylight]

    def _select_anchor(
        self,
        section_field: SectionField,
        segment_lookup: dict[int, object],
        allowed_segment_ids: set[int] | None,
    ) -> SectionSample | None:
        candidates: list[SectionSample] = []
        for segment_field in section_field.segment_fields:
            if allowed_segment_ids is not None and segment_field.segment_id not in allowed_segment_ids:
                continue
            segment = segment_lookup.get(segment_field.segment_id)
            if segment is None:
                continue
            if getattr(segment, "kind", "") == "underpass" or getattr(segment, "z_level", 0) != 0:
                continue
            for sample in segment_field.samples:
                if any(influence.kind == "crossing" for influence in sample.junction_influences):
                    continue
                if sample.junction_blend_weight >= 0.32:
                    continue
                candidates.append(sample)
        if not candidates:
            return None
        return max(candidates, key=lambda sample: sample.z)

    def _build_skylight(
        self,
        *,
        skylight_id: int,
        anchor: SectionSample,
        rng: np.random.Generator,
    ) -> SkylightGeometry | None:
        anchor_roof_vertex = self._anchor_roof_vertex(anchor)
        base_z = anchor.z + (0.46 * anchor.tube_height)
        top_z = anchor.surface_z + self.config.skylight_surface_margin
        if top_z <= base_z:
            return None

        ring_vertex_count = max(10, len(anchor.profile_points[:-1]) // 2)
        ring_count = max(self.config.skylight_ring_count, 4)

        drift_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        drift_direction = np.array(
            [math.cos(drift_angle), math.sin(drift_angle), 0.0],
            dtype=float,
        )
        base_center = np.array([anchor.x, anchor.y, base_z], dtype=float)
        total_height = top_z - base_z
        max_drift = self.config.skylight_drift_ratio * max(anchor.tube_width, anchor.tube_height)
        bottom_radius = self.config.skylight_bottom_radius_scale * max(anchor.tube_width, 1.0)
        top_radius = self.config.skylight_top_radius_scale * max(anchor.tube_width, 1.0)
        bottom_radius_y = bottom_radius * 0.78
        top_radius_y = top_radius * 0.84

        vertices: list[tuple[float, float, float]] = [anchor_roof_vertex]
        faces: list[tuple[int, int, int]] = []

        for ring_index in range(ring_count):
            t = (ring_index + 1) / ring_count
            smooth_t = t * t * (3.0 - 2.0 * t)
            ring_center = base_center + np.array([0.0, 0.0, total_height * t], dtype=float)
            ring_center += drift_direction * (max_drift * smooth_t * smooth_t)
            radius_x = (1.0 - smooth_t) * bottom_radius + smooth_t * top_radius
            radius_y = (1.0 - smooth_t) * bottom_radius_y + smooth_t * top_radius_y
            jaggedness = self.config.skylight_rim_jaggedness * smooth_t

            for vertex_index in range(ring_vertex_count):
                angle = (2.0 * math.pi * vertex_index) / ring_vertex_count
                harmonic = (
                    0.55 * math.sin((3.0 * angle) + 1.2)
                    + 0.30 * math.sin((5.0 * angle) - 0.4)
                    + 0.15 * math.sin((7.0 * angle) + 0.9)
                )
                radial_scale = 1.0 + (jaggedness * harmonic)
                x_coord = ring_center[0] + radial_scale * radius_x * math.cos(angle)
                y_coord = ring_center[1] + radial_scale * radius_y * math.sin(angle)
                z_coord = ring_center[2]
                vertices.append((float(x_coord), float(y_coord), float(z_coord)))

        first_ring_offset = 1
        for vertex_index in range(ring_vertex_count):
            next_vertex_index = (vertex_index + 1) % ring_vertex_count
            a = 0
            b = first_ring_offset + vertex_index
            c = first_ring_offset + next_vertex_index
            faces.append((a, b, c))

        for ring_index in range(ring_count - 1):
            ring_offset = 1 + (ring_index * ring_vertex_count)
            next_offset = 1 + ((ring_index + 1) * ring_vertex_count)
            for vertex_index in range(ring_vertex_count):
                next_vertex_index = (vertex_index + 1) % ring_vertex_count
                a = ring_offset + vertex_index
                b = ring_offset + next_vertex_index
                c = next_offset + vertex_index
                d = next_offset + next_vertex_index
                faces.append((a, c, b))
                faces.append((b, c, d))

        top_center_vertices = vertices[-ring_vertex_count:]
        top_center = (
            float(np.mean([vertex[0] for vertex in top_center_vertices])),
            float(np.mean([vertex[1] for vertex in top_center_vertices])),
            float(np.mean([vertex[2] for vertex in top_center_vertices])),
        )

        return SkylightGeometry(
            skylight_id=skylight_id,
            anchor_segment_id=anchor.segment_id,
            anchor_sample_index=anchor.index,
            anchor_vertex=anchor_roof_vertex,
            vertices=tuple(vertices),
            faces=tuple(faces),
            ring_vertex_count=ring_vertex_count,
            top_center=top_center,
        )

    @staticmethod
    def _anchor_roof_vertex(anchor: SectionSample) -> tuple[float, float, float]:
        ring_profile = anchor.profile_points[:-1]
        if not ring_profile:
            return (anchor.x, anchor.y, anchor.z)
        roof_point = max(ring_profile, key=lambda point: point[1])
        return build_ring_vertices(anchor, (roof_point,))[0]

"""Stage D orchestration for swept geometry, junction patches, and skylights."""

from __future__ import annotations

from collections import defaultdict

from stages.geometry_assembly import assemble_geometry
from stages.geometry_junctions import JunctionPatchBuilder
from stages.geometry_skylight import SkylightBuilder
from stages.geometry_sweep import SweepBuilder
from stages.geometry_types import (
    CaveGeometry,
    GeometryConfig,
    JunctionGeometryPatch,
    SegmentGeometrySpan,
    SkylightGeometry,
)
from stages.network import CaveNetwork
from stages.section_field import SectionField


class GeometryGenerator:
    """Build Stage-D geometry by orchestrating sweep, stitch, and opening passes."""

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()
        self.sweep_builder = SweepBuilder(self.config)
        self.junction_builder = JunctionPatchBuilder(self.config)
        self.skylight_builder = SkylightBuilder(self.config)

    def generate(
        self,
        cave_network: CaveNetwork,
        section_field: SectionField,
    ) -> CaveGeometry:
        segment_lookup = {segment.segment_id: segment for segment in cave_network.segments}
        junction_lookup = {junction.junction_id: junction for junction in cave_network.junctions}
        field_lookup = {
            segment_field.segment_id: segment_field for segment_field in section_field.segment_fields
        }
        primary_segment_ids = self._primary_component_segment_ids(cave_network)
        dominant_route_segment_ids = set(section_field.dominant_route_segment_ids)

        meshes: list[SegmentGeometrySpan] = []
        excluded_segment_ids: set[int] = set()
        excluded_junction_ids: set[int] = set()

        for segment_field in section_field.segment_fields:
            segment = segment_lookup[segment_field.segment_id]
            if segment.segment_id not in primary_segment_ids:
                excluded_segment_ids.add(segment.segment_id)
                excluded_junction_ids.update(segment_field.connected_junction_ids)
                continue
            if self.sweep_builder.exclude_segment(segment, segment_field, junction_lookup):
                excluded_segment_ids.add(segment.segment_id)
                excluded_junction_ids.update(segment_field.connected_junction_ids)
                continue

            spans = self.sweep_builder.build_sample_spans(segment_field)
            if not spans:
                connector_mesh = self._build_connector_span(
                    meshes=meshes,
                    segment=segment,
                    segment_field=segment_field,
                    dominant_route_segment_ids=dominant_route_segment_ids,
                )
                if connector_mesh is not None:
                    meshes.append(connector_mesh)
                    continue
                excluded_segment_ids.add(segment.segment_id)
                excluded_junction_ids.update(segment_field.connected_junction_ids)
                continue

            for sample_indices in spans:
                mesh = self.sweep_builder.build_mesh_span(
                    mesh_id=len(meshes),
                    segment=segment,
                    segment_field=segment_field,
                    sample_indices=sample_indices,
                )
                if mesh is not None:
                    meshes.append(mesh)

        junction_patches: list[JunctionGeometryPatch] = []
        if self.config.enable_junction_patches:
            junction_patches = self.junction_builder.build_patches(
                cave_network=cave_network,
                section_field=section_field,
                field_lookup=field_lookup,
                segment_lookup=segment_lookup,
                meshes=meshes,
                excluded_junction_ids=excluded_junction_ids,
            )

        skylights: list[SkylightGeometry] = self.skylight_builder.build_skylights(
            section_field=section_field,
            segment_lookup=segment_lookup,
            allowed_segment_ids=primary_segment_ids,
        )

        assembled_vertices, assembled_faces, component_count = assemble_geometry(
            config=self.config,
            meshes=meshes,
            junction_patches=junction_patches,
            skylights=skylights,
        )

        return CaveGeometry(
            config=self.config,
            meshes=tuple(meshes),
            junction_patches=tuple(junction_patches),
            skylights=tuple(skylights),
            assembled_vertices=assembled_vertices,
            assembled_faces=assembled_faces,
            component_count=component_count,
            excluded_segment_ids=tuple(sorted(excluded_segment_ids)),
            excluded_junction_ids=tuple(sorted(excluded_junction_ids)),
        )

    @staticmethod
    def _primary_component_segment_ids(cave_network: CaveNetwork) -> set[int]:
        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for segment in cave_network.segments:
            adjacency[segment.start_node_id].append((segment.end_node_id, segment.segment_id))
            adjacency[segment.end_node_id].append((segment.start_node_id, segment.segment_id))

        if cave_network.dominant_route_node_ids:
            start_node_id = cave_network.dominant_route_node_ids[0]
        elif cave_network.nodes:
            start_node_id = cave_network.nodes[0].node_id
        else:
            return set()

        visited_nodes = {start_node_id}
        stack = [start_node_id]
        component_segment_ids: set[int] = set()
        while stack:
            current = stack.pop()
            for neighbor_id, segment_id in adjacency.get(current, ()):
                component_segment_ids.add(segment_id)
                if neighbor_id in visited_nodes:
                    continue
                visited_nodes.add(neighbor_id)
                stack.append(neighbor_id)
        return component_segment_ids

    def _build_connector_span(
        self,
        *,
        meshes: list[SegmentGeometrySpan],
        segment,
        segment_field,
        dominant_route_segment_ids: set[int],
    ) -> SegmentGeometrySpan | None:
        if segment.kind in {"underpass", "spur"} or segment.z_level != 0:
            return None
        if (
            segment.segment_id not in dominant_route_segment_ids
            and not segment_field.connected_junction_ids
        ):
            return None
        sample_indices = tuple(sample.index for sample in segment_field.samples)
        if len(sample_indices) < self.config.minimum_span_samples:
            return None
        return self.sweep_builder.build_mesh_span(
            mesh_id=len(meshes),
            segment=segment,
            segment_field=segment_field,
            sample_indices=sample_indices,
            is_connector=True,
        )


__all__ = [
    "CaveGeometry",
    "GeometryConfig",
    "GeometryGenerator",
    "JunctionGeometryPatch",
    "SegmentGeometrySpan",
    "SkylightGeometry",
]

"""Stable machine-readable artifacts and semantic hashes for canonical stages."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.evaluation.metrics.sections import section_field_diagnostics
from plume_advanced.evaluation.provenance import quantized_array, semantic_hash, sha256_file
from plume_advanced.stages.events import GeologicalEventField
from plume_advanced.stages.geometry_types import CaveGeometry
from plume_advanced.stages.host_field import HostField
from plume_advanced.stages.network import CaveNetwork
from plume_advanced.stages.section_field import SectionField


def host_semantic_hash(host_field: HostField, *, tolerance: float = 1e-9) -> str:
    return semantic_hash(
        {
            name: quantized_array(values, tolerance)
            for name, values in {
                "x": host_field.x_coords,
                "y": host_field.y_coords,
                "elevation": host_field.elevation,
                "cover": host_field.cover_thickness,
                "fracture": host_field.fracture_intensity,
                "capacity": host_field.flow_capacity,
                "stability": host_field.roof_stability,
                "routing_cost": host_field.routing_cost,
            }.items()
        }
    )


def network_payload(network: CaveNetwork) -> dict[str, Any]:
    return {
        "schema": "plume.cave-network.v1",
        "nodes": [asdict(node) for node in sorted(network.nodes, key=lambda item: item.node_id)],
        "segments": [
            {
                "segment_id": segment.segment_id,
                "source_node_id": segment.start_node_id,
                "target_node_id": segment.end_node_id,
                "kind": segment.kind,
                "z_level": segment.z_level,
                "physical_length_m": segment.total_length,
                "mean_width_m": segment.mean_width,
                "mean_flux": segment.mean_flux,
                "temperature_start_k": segment.points[0].temperature_k if segment.points else None,
                "temperature_end_k": segment.points[-1].temperature_k if segment.points else None,
                "age_start_s": segment.points[0].age_s if segment.points else None,
                "age_end_s": segment.points[-1].age_s if segment.points else None,
                "grade": (
                    (segment.points[-1].elevation - segment.points[0].elevation)
                    / max(segment.total_length, 1e-12)
                    if segment.points
                    else None
                ),
                "metadata": segment.metadata,
                "centerline": [asdict(point) for point in segment.points],
            }
            for segment in sorted(network.segments, key=lambda item: item.segment_id)
        ],
        "junctions": [
            asdict(junction)
            for junction in sorted(network.junctions, key=lambda item: item.junction_id)
        ],
        "dominant_route_node_ids": list(network.dominant_route_node_ids),
        "summary_invariants": network_metrics(network),
    }


def network_semantic_hash(network: CaveNetwork, *, tolerance: float = 1e-6) -> str:
    payload = network_payload(network)
    for segment in payload["segments"]:
        for point in segment["centerline"]:
            for key, value in tuple(point.items()):
                if isinstance(value, float):
                    point[key] = int(round(value / tolerance))
    return semantic_hash(payload)


def export_network_artifact(network: CaveNetwork, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = network_payload(network)
    payload["semantic_sha256"] = network_semantic_hash(network)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def section_semantic_payload(section_field: SectionField, *, tolerance: float = 1e-6) -> list[Any]:
    result: list[Any] = []
    for segment_field in sorted(section_field.segment_fields, key=lambda item: item.segment_id):
        result.append(
            {
                "segment_id": segment_field.segment_id,
                "connected_junction_ids": list(segment_field.connected_junction_ids),
                "samples": [
                    {
                        "s": round(sample.segment_arc_length / tolerance),
                        "center": quantized_array((sample.x, sample.y, sample.z), tolerance),
                        "frame": quantized_array(
                            (sample.tangent, sample.normal, sample.binormal), tolerance
                        ),
                        "shape": quantized_array(
                            (
                                sample.tube_width,
                                sample.tube_height,
                                sample.floor_flatness,
                                sample.roof_arch,
                                sample.lateral_skew,
                                sample.junction_blend_weight,
                            ),
                            tolerance,
                        ),
                        "morphology": {
                            "regime": sample.morphology_regime,
                            "family_score": round(sample.morphology_family_score / tolerance),
                            "parent_segment_id": sample.parent_morphology_segment_id,
                            "junction_blend_length_m": round(sample.junction_blend_length_m / tolerance),
                        },
                        "flow_state": quantized_array(
                            (
                                sample.lava_flux,
                                sample.lava_temperature_k,
                                sample.lava_age_s,
                                sample.flow_maturity,
                            ),
                            tolerance,
                        ),
                        "profile": quantized_array(sample.profile_points, tolerance),
                    }
                    for sample in segment_field.samples
                ],
            }
        )
    return result


def section_semantic_hash(section_field: SectionField, *, tolerance: float = 1e-6) -> str:
    return semantic_hash(section_semantic_payload(section_field, tolerance=tolerance))


def export_section_artifact(
    section_field: SectionField,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    samples = [
        sample for segment_field in section_field.segment_fields for sample in segment_field.samples
    ]
    profiles = [np.asarray(sample.profile_points, dtype=float) for sample in samples]
    offsets = np.zeros(len(profiles) + 1, dtype=np.int64)
    if profiles:
        offsets[1:] = np.cumsum([profile.shape[0] for profile in profiles])
        profile_points = np.vstack(profiles)
    else:
        profile_points = np.empty((0, 2), dtype=float)
    centers = np.asarray([(sample.x, sample.y, sample.z) for sample in samples], dtype=float)
    floor_locations = np.empty_like(centers)
    for index, sample in enumerate(samples):
        profile = np.asarray(sample.profile_points, dtype=float)
        floor = profile[int(np.argmin(profile[:, 1]))]
        floor_locations[index] = (
            np.asarray((sample.x, sample.y, sample.z))
            + floor[0] * np.asarray(sample.normal)
            + floor[1] * np.asarray(sample.binormal)
        )
    np.savez_compressed(
        npz_path,
        segment_id=np.asarray([sample.segment_id for sample in samples], dtype=np.int64),
        arc_length_m=np.asarray([sample.segment_arc_length for sample in samples]),
        center_xyz_m=centers,
        tangent=np.asarray([sample.tangent for sample in samples], dtype=float),
        normal=np.asarray([sample.normal for sample in samples], dtype=float),
        binormal=np.asarray([sample.binormal for sample in samples], dtype=float),
        width_m=np.asarray([sample.tube_width for sample in samples]),
        height_m=np.asarray([sample.tube_height for sample in samples]),
        floor_flatness=np.asarray([sample.floor_flatness for sample in samples]),
        roof_arch=np.asarray([sample.roof_arch for sample in samples]),
        lateral_skew=np.asarray([sample.lateral_skew for sample in samples]),
        junction_influence=np.asarray([sample.junction_blend_weight for sample in samples]),
        lava_flux=np.asarray([sample.lava_flux for sample in samples]),
        lava_temperature_k=np.asarray([sample.lava_temperature_k for sample in samples]),
        lava_age_s=np.asarray([sample.lava_age_s for sample in samples]),
        flow_maturity=np.asarray([sample.flow_maturity for sample in samples]),
        morphology_family_score=np.asarray(
            [sample.morphology_family_score for sample in samples], dtype=float
        ),
        junction_blend_length_m=np.asarray(
            [sample.junction_blend_length_m for sample in samples], dtype=float
        ),
        morphology_regime=np.asarray([sample.morphology_regime for sample in samples]),
        parent_morphology_segment_id=np.asarray(
            [
                -1 if sample.parent_morphology_segment_id is None else sample.parent_morphology_segment_id
                for sample in samples
            ],
            dtype=np.int64,
        ),
        floor_location_xyz_m=floor_locations,
        profile_offsets=offsets,
        profile_points=profile_points,
    )
    metadata = {
        "schema": "plume.section-field.v1",
        "coordinate_system": {"length_unit": "metre", "up_axis": "Z", "handedness": "right"},
        "sample_count": len(samples),
        "segment_count": len(section_field.segment_fields),
        "sampling_policy": section_field.config.sampling_policy,
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "semantic_sha256": section_semantic_hash(section_field),
        "junction_influences": [
            [asdict(influence) for influence in sample.junction_influences] for sample in samples
        ],
        "diagnostics": section_field_diagnostics(section_field),
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, json_path


def geometry_semantic_hash(geometry: CaveGeometry, *, tolerance: float = 1e-5) -> str:
    vertices = np.asarray(geometry.assembled_vertices, dtype=float)
    faces = np.asarray(geometry.assembled_faces, dtype=np.int64)
    triangles = []
    if faces.size:
        quantized = np.rint(vertices / tolerance).astype(np.int64)
        for face in faces:
            triangle = sorted(tuple(int(value) for value in vertex) for vertex in quantized[face])
            triangles.append(triangle)
        triangles.sort()
    return semantic_hash({"triangles": triangles, "voxel_size": geometry.voxel_grid.voxel_size})


def export_geometry_report(geometry: CaveGeometry, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = geometry.summary()
    bounds = geometry.voxel_grid.bounds
    payload = {
        "schema": "plume.geometry-report.v1",
        "storage_mode_requested": geometry.config.storage_mode,
        "storage_mode_actual": "tiled" if hasattr(geometry.voxel_grid, "tiles") else "dense",
        "requested_voxel_size_m": geometry.config.voxel_size,
        "actual_voxel_size_m": geometry.voxel_grid.voxel_size,
        "nominal_samples_across_passage": geometry.config.characteristic_samples_across_passage,
        "actual_samples_across_minimum_section": summary["minimum_section_width_samples"],
        "dense_voxel_estimate": int(np.prod(geometry.voxel_grid.shape)),
        "processed_voxel_count": int(
            sum(tile.size for tile in geometry.voxel_grid.tiles.values())
            if hasattr(geometry.voxel_grid, "tiles")
            else np.prod(geometry.voxel_grid.shape)
        ),
        "chunk_count": len(geometry.chunk_meshes),
        "vertex_count": len(geometry.assembled_vertices),
        "face_count": len(geometry.assembled_faces),
        "component_count": geometry.component_count,
        "canonical_bbox_m": bounds,
        "collision_vertex_count": len(geometry.assembled_vertices),
        "collision_face_count": len(geometry.assembled_faces),
        "visual_vertex_count": len(geometry.assembled_vertices),
        "visual_face_count": len(geometry.assembled_faces),
        "canonical_geometry_semantic_sha256": geometry_semantic_hash(geometry),
        "summary": summary,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def event_semantic_hash(event_field: GeologicalEventField, *, tolerance: float = 1e-6) -> str:
    payload = []
    for event in sorted(event_field.events, key=lambda item: item.event_id):
        record = asdict(event)
        for key, value in tuple(record.items()):
            if isinstance(value, float):
                record[key] = None if not np.isfinite(value) else int(round(value / tolerance))
        payload.append(record)
    return semantic_hash(payload)


def export_event_report(
    event_field: GeologicalEventField,
    output_path: str | Path,
    *,
    invalidated_floor_cells: int = 0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for event in event_field.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    payload = {
        "schema": "plume.event-report.v1",
        "event_counts_by_type": counts,
        "accepted_prop_count": sum(
            event.kind in {"rock", "boulder"} for event in event_field.events
        ),
        "requested_prop_count": None,
        "rejected_prop_count": None,
        "rejection_reasons": {},
        "events": [asdict(event) for event in event_field.events],
        "route_preservation_enabled": event_field.config.preserve_rover_route,
        "minimum_measured_post_event_route_clearance_m": (
            event_field.minimum_rover_bypass_m
            if np.isfinite(event_field.minimum_rover_bypass_m)
            else None
        ),
        "invalidated_floor_atlas_cell_count": invalidated_floor_cells,
        "semantic_sha256": event_semantic_hash(event_field),
        "summary": event_field.summary(),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


__all__ = [
    "event_semantic_hash",
    "export_event_report",
    "export_geometry_report",
    "export_network_artifact",
    "export_section_artifact",
    "geometry_semantic_hash",
    "host_semantic_hash",
    "network_semantic_hash",
    "section_semantic_hash",
]

"""Generated-section extraction and sampling-fidelity metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from plume_advanced.evaluation.metrics.morphometry import contour_morphometry
from plume_advanced.stages.section_field import SectionField, SectionSample


def generated_section_records(
    section_field: SectionField,
    *,
    seed: int,
    world_id: str,
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for segment_field in section_field.segment_fields:
        for sample in segment_field.samples:
            metrics = contour_morphometry(sample.profile_points)
            records.append(
                {
                    "generated_seed": seed,
                    "generated_world_id": world_id,
                    "generated_segment_id": segment_field.segment_id,
                    "segment_arc_length_m": sample.segment_arc_length,
                    **metrics,
                }
            )
    return records


def section_surface_points(section_field: SectionField) -> np.ndarray:
    points: list[np.ndarray] = []
    for segment_field in section_field.segment_fields:
        for sample in segment_field.samples:
            points.append(_world_profile(sample))
    return np.vstack(points) if points else np.empty((0, 3), dtype=float)


def section_geometric_error(
    candidate: SectionField,
    reference: SectionField,
) -> dict[str, float]:
    candidate_points = section_surface_points(candidate)
    reference_points = section_surface_points(reference)
    if candidate_points.size == 0 or reference_points.size == 0:
        return {"mean_m": float("nan"), "rms_m": float("nan"), "p95_m": float("nan")}
    candidate_distances = cKDTree(candidate_points).query(reference_points, workers=1)[0]
    reference_distances = cKDTree(reference_points).query(candidate_points, workers=1)[0]
    distances = np.concatenate((candidate_distances, reference_distances))
    return {
        "mean_m": float(np.mean(distances)),
        "rms_m": float(np.sqrt(np.mean(np.square(distances)))),
        "p95_m": float(np.percentile(distances, 95.0)),
    }


def _world_profile(sample: SectionSample) -> np.ndarray:
    center = np.asarray((sample.x, sample.y, sample.z), dtype=float)
    normal = np.asarray(sample.normal, dtype=float)
    binormal = np.asarray(sample.binormal, dtype=float)
    profile = np.asarray(sample.profile_points, dtype=float)
    return center + profile[:, :1] * normal + profile[:, 1:] * binormal


__all__ = ["generated_section_records", "section_geometric_error", "section_surface_points"]

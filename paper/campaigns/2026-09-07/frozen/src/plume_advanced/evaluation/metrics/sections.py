"""Generated-section extraction and sampling-fidelity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from plume_advanced.evaluation.continuity import longitudinal_continuity
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry
from plume_advanced.stages.section_field import SectionField, SectionSample

PDC_FEATURES = (
    "width",
    "height",
    "aspect_ratio",
    "area",
    "compactness",
    "floor_residual",
    "roof_asymmetry",
)


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
                    # Canonical names are convenient for cross-dataset reports;
                    # the established *_m/*_norm fields remain untouched.
                    "width": metrics["width_m"],
                    "height": metrics["height_m"],
                    "area": metrics["area_m2"],
                    "floor_residual": metrics["floor_residual_norm"],
                    "roof_asymmetry": metrics["roof_asymmetry_norm"],
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


def pdc_comparable_section_summary(
    records: list[dict[str, Any]],
    *,
    features: tuple[str, ...] = PDC_FEATURES,
) -> dict[str, Any]:
    """Return deterministic Q1/median/Q3/IQR summaries for section records.

    Records can come from generated sections or the PDC loader. Features are
    read by canonical name first, then by the legacy metric key.
    """

    aliases = {
        "width": "width_m",
        "height": "height_m",
        "area": "area_m2",
        "floor_residual": "floor_residual_norm",
        "roof_asymmetry": "roof_asymmetry_norm",
    }
    result: dict[str, Any] = {}
    for feature in features:
        values = []
        for record in records:
            raw = record.get(feature, record.get(aliases.get(feature, feature)))
            if raw is None:
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values.append(number)
        ordered = np.sort(np.asarray(values, dtype=float))
        if not values:
            result[feature] = {"count": 0, "q1": None, "median": None, "q3": None, "iqr": None}
            continue
        q1, median, q3 = np.percentile(ordered, [25.0, 50.0, 75.0])
        result[feature] = {
            "count": int(ordered.size),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "iqr": float(q3 - q1),
        }
    return result


def section_longitudinal_continuity(section_field: SectionField) -> dict[str, Any]:
    """Measure per-segment shape evolution in physical arc length.

    Segment boundaries are kept separate because each segment has its own local
    arc-length origin; aggregating them would create an artificial jump.
    """

    per_segment: dict[str, Any] = {}
    for segment_field in section_field.segment_fields:
        records = []
        for sample in segment_field.samples:
            metrics = contour_morphometry(sample.profile_points)
            records.append(
                {
                    "arc_length": sample.segment_arc_length,
                    "width": metrics["width_m"],
                    "height": metrics["height_m"],
                    "aspect_ratio": metrics["aspect_ratio"],
                    "area": metrics["area_m2"],
                    "compactness": metrics["compactness"],
                    "floor_residual": metrics["floor_residual_norm"],
                    "roof_asymmetry": metrics["roof_asymmetry_norm"],
                }
            )
        per_segment[str(segment_field.segment_id)] = longitudinal_continuity(records)
    return {"schema_version": "1.0", "segments": per_segment}


def section_field_diagnostics(section_field: SectionField) -> dict[str, Any]:
    """Build the report payload consumed by the section artifact exporter."""

    records: list[dict[str, Any]] = []
    for segment_field in section_field.segment_fields:
        for sample in segment_field.samples:
            metrics = contour_morphometry(sample.profile_points)
            records.append(
                {
                    "segment_id": segment_field.segment_id,
                    "arc_length": sample.segment_arc_length,
                    **metrics,
                }
            )
    return {
        "schema_version": "1.0",
        "morphometry": pdc_comparable_section_summary(records),
        "longitudinal_continuity": section_longitudinal_continuity(section_field),
    }


section_morphometry_summary = pdc_comparable_section_summary
section_feature_summary = pdc_comparable_section_summary
section_diagnostics = pdc_comparable_section_summary


__all__ = [
    "generated_section_records",
    "section_geometric_error",
    "section_surface_points",
    "pdc_comparable_section_summary",
    "section_longitudinal_continuity",
    "section_morphometry_summary",
    "section_feature_summary",
    "section_diagnostics",
    "section_field_diagnostics",
]

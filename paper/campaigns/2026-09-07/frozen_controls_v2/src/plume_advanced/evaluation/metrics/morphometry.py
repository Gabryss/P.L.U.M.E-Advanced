"""Predeclared cross-section morphometry for generated and PDC contours."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from plume_advanced.evaluation.metrics.contours import (
    clean_contour,
    polygon_centroid,
    polygon_perimeter,
    signed_area,
)


def contour_morphometry(points: Any) -> dict[str, float]:
    """Measure a section while preserving its documented vertical axis.

    Floor residual is the normalized RMS residual to a least-squares line for
    boundary points in the lower 20% of section height. Roof asymmetry is the
    horizontal offset between the mean upper-half boundary position and the
    polygon centroid, normalized by width.
    """

    contour = clean_contour(points)
    vertices = contour[:-1]
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    width, height = maximum - minimum
    if width <= 0.0 or height <= 0.0:
        raise ValueError("contour width and height must be positive")
    area = abs(signed_area(vertices))
    perimeter = polygon_perimeter(contour)
    centroid_x, centroid_y = polygon_centroid(contour)
    hull = ConvexHull(vertices)
    hull_area = float(hull.volume)
    lower = vertices[vertices[:, 1] <= minimum[1] + 0.20 * height]
    if lower.shape[0] < 2:
        lower = vertices[np.argsort(vertices[:, 1])[: min(3, vertices.shape[0])]]
    design = np.column_stack((lower[:, 0], np.ones(lower.shape[0])))
    coefficients, *_ = np.linalg.lstsq(design, lower[:, 1], rcond=None)
    residual = lower[:, 1] - design @ coefficients
    floor_residual = float(np.sqrt(np.mean(np.square(residual))) / height)
    upper = vertices[vertices[:, 1] >= minimum[1] + 0.50 * height]
    roof_center_x = float(np.mean(upper[:, 0])) if upper.size else centroid_x
    roof_asymmetry = abs(roof_center_x - centroid_x) / width
    return {
        "width_m": float(width),
        "height_m": float(height),
        "aspect_ratio": float(width / height),
        "area_m2": float(area),
        "perimeter_m": float(perimeter),
        "equivalent_radius_m": float(math.sqrt(area / math.pi)),
        "compactness": float(4.0 * math.pi * area / max(perimeter**2, 1e-12)),
        "solidity": float(area / max(hull_area, 1e-12)),
        "centroid_vertical_norm": float((centroid_y - minimum[1]) / height),
        "centroid_horizontal_offset_norm": float(
            abs(centroid_x - 0.5 * (minimum[0] + maximum[0])) / width
        ),
        "floor_residual_norm": floor_residual,
        "roof_asymmetry_norm": float(roof_asymmetry),
    }


def ellipse_baseline(width: float, height: float, *, resolution: int = 64) -> np.ndarray:
    if width <= 0.0 or height <= 0.0 or resolution < 8:
        raise ValueError("ellipse baseline requires positive dimensions and resolution >= 8")
    angles = np.linspace(0.0, 2.0 * math.pi, resolution, endpoint=False)
    return np.column_stack((0.5 * width * np.cos(angles), 0.5 * height * np.sin(angles)))


__all__ = ["contour_morphometry", "ellipse_baseline"]

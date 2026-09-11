"""Validated cleanup and primitive operations for ordered 2-D contours."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class ContourError(ValueError):
    """Raised when a contour cannot support scientific measurements."""


def clean_contour(points: Any, *, close: bool = True) -> np.ndarray:
    contour = np.asarray(points, dtype=float)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ContourError("contour must have shape (N, 2)")
    if contour.shape[0] < 3:
        raise ContourError("contour must contain at least three points")
    if not bool(np.all(np.isfinite(contour))):
        raise ContourError("contour contains NaN or infinite coordinates")
    keep = np.ones(contour.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(contour, axis=0), axis=1) > 1e-12
    contour = contour[keep]
    if contour.shape[0] > 1 and np.linalg.norm(contour[0] - contour[-1]) <= 1e-12:
        contour = contour[:-1]
    if np.unique(contour, axis=0).shape[0] < 3:
        raise ContourError("contour has fewer than three distinct vertices")
    area = signed_area(contour)
    scale = max(float(np.ptp(contour[:, 0]) * np.ptp(contour[:, 1])), 1.0)
    if abs(area) <= 1e-12 * scale:
        raise ContourError("contour is degenerate or collinear")
    if area < 0.0:
        contour = contour[::-1].copy()
    if close:
        contour = np.vstack((contour, contour[0]))
    return contour


def signed_area(points: Any) -> float:
    contour = np.asarray(points, dtype=float)
    if contour.shape[0] > 1 and np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    x_coord = contour[:, 0]
    y_coord = contour[:, 1]
    return 0.5 * float(
        np.dot(x_coord, np.roll(y_coord, -1)) - np.dot(y_coord, np.roll(x_coord, -1))
    )


def polygon_perimeter(points: Any) -> float:
    contour = clean_contour(points)
    return float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum())


def polygon_centroid(points: Any) -> tuple[float, float]:
    contour = clean_contour(points)[:-1]
    x_coord = contour[:, 0]
    y_coord = contour[:, 1]
    cross = x_coord * np.roll(y_coord, -1) - np.roll(x_coord, -1) * y_coord
    area6 = 3.0 * float(np.sum(cross))
    if math.isclose(area6, 0.0):
        raise ContourError("contour centroid is undefined")
    centroid_x = float(np.sum((x_coord + np.roll(x_coord, -1)) * cross) / area6)
    centroid_y = float(np.sum((y_coord + np.roll(y_coord, -1)) * cross) / area6)
    return centroid_x, centroid_y


def self_intersection_count(points: Any) -> int:
    contour = clean_contour(points)
    count = 0
    segment_count = contour.shape[0] - 1
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            if second in {first, first + 1} or (first == 0 and second == segment_count - 1):
                continue
            if _segments_intersect(
                contour[first], contour[first + 1], contour[second], contour[second + 1]
            ):
                count += 1
    return count


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
        left = second - first
        right = third - first
        return float(left[0] * right[1] - left[1] * right[0])

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return first * second < 0.0 and third * fourth < 0.0


__all__ = [
    "ContourError",
    "clean_contour",
    "polygon_centroid",
    "polygon_perimeter",
    "self_intersection_count",
    "signed_area",
]

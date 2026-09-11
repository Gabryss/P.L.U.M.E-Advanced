import math

import numpy as np
import pytest

from plume_advanced.evaluation.metrics.contours import ContourError, clean_contour
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry


def test_rectangle_metrics_are_exact_and_orientation_invariant() -> None:
    rectangle = np.asarray(((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)))
    forward = contour_morphometry(rectangle)
    reversed_metrics = contour_morphometry(rectangle[::-1])

    assert forward["width_m"] == 4.0
    assert forward["height_m"] == 2.0
    assert forward["area_m2"] == 8.0
    assert forward["perimeter_m"] == 12.0
    assert forward["floor_residual_norm"] == pytest.approx(0.0, abs=1e-12)
    assert forward == pytest.approx(reversed_metrics)


def test_circle_approximation_has_compactness_near_one() -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    assert contour_morphometry(circle)["compactness"] == pytest.approx(1.0, rel=1e-4)


def test_cleanup_removes_duplicate_endpoint_and_rejects_invalid_contours() -> None:
    square = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    cleaned = clean_contour(square)
    assert cleaned.shape == (5, 2)
    assert np.array_equal(cleaned[0], cleaned[-1])

    with pytest.raises(ContourError):
        clean_contour(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    with pytest.raises(ContourError):
        clean_contour(((0.0, 0.0), (1.0, np.nan), (0.0, 1.0)))

"""Atlas failures must not corrupt texture mapping or move spatial triangles."""

import numpy as np
import pytest

from plume_advanced.stages.geometry_export import _repair_collapsed_uv_triangles


@pytest.mark.parametrize("shared", [False, True])
def test_collapsed_chart_is_metric_and_preserves_geometry_and_healthy_uvs(shared):
    vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [.003, .004, 0.], [0., .02, 0.]])
    mapping = np.arange(4, dtype=np.uint32)
    faces = np.array([[0, 1, 2], [0, 2, 3]] if shared else [[0, 1, 2]], dtype=np.uint32)
    # First triangle is collapsed; the second is a healthy, adjacent chart.
    uv = np.array([[0., 0.], [.001, 0.], [.002, 0.], [0., .005]])
    m, f, result = _repair_collapsed_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
    np.testing.assert_array_equal(vertices[m[f]], vertices[mapping[faces]])
    repaired = result[f[0]]
    world = vertices[faces[0]]
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        assert np.linalg.norm(repaired[a]-repaired[b])*4 == pytest.approx(np.linalg.norm(world[a]-world[b]), rel=1e-6)
    assert abs(np.linalg.det(repaired[1:]-repaired[0])) > 1e-12
    if shared:
        np.testing.assert_array_equal(result[f[1]], uv[faces[1]])
        assert len(m) == 6  # only the two shared corners need duplication
    else:
        np.testing.assert_array_equal(m, mapping)
        np.testing.assert_array_equal(f, faces)


def test_float32_rounding_collapse_is_detected_and_repair_is_repeatable():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    mapping = np.arange(3, dtype=np.uint32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    uv = np.array([[1e8, 1e8], [1e8+1, 1e8], [1e8, 1e8+1]])
    first = _repair_collapsed_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
    second = _repair_collapsed_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    mapped = first[2].astype(np.float32)[first[1][0]]
    assert abs(np.linalg.det(mapped[1:]-mapped[0])) == pytest.approx(1/16)


def test_healthy_atlas_is_exact_noop_and_degenerate_space_is_rejected():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    mapping = np.arange(3, dtype=np.uint32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    uv = vertices[:, :2].copy()
    actual = _repair_collapsed_uv_triangles(vertices, mapping, faces, uv, scale_m=1.)
    assert all(a is b for a, b in zip(actual, (mapping, faces, uv), strict=True))
    with pytest.raises(ValueError, match="spatially degenerate"):
        _repair_collapsed_uv_triangles(vertices*0, mapping, faces, uv*0, scale_m=1.)

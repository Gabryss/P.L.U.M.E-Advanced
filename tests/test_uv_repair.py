"""Atlas failures must not corrupt texture mapping or move spatial triangles."""

import numpy as np
import pytest

from plume_advanced.stages.geometry_export import _repair_metric_uv_triangles


@pytest.mark.parametrize("shared", [False, True])
def test_collapsed_chart_is_metric_and_preserves_geometry_and_healthy_uvs(shared):
    vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [.003, .004, 0.], [0., .02, 0.]])
    mapping = np.arange(4, dtype=np.uint32)
    faces = np.array([[0, 1, 2], [0, 2, 3]] if shared else [[0, 1, 2]], dtype=np.uint32)
    # First triangle is collapsed; the second is a healthy, adjacent chart.
    uv = np.array([[0., 0.], [.001, 0.], [.002, 0.], [0., .005]])
    m, f, result = _repair_metric_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
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
    first = _repair_metric_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
    second = _repair_metric_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    mapped = first[2].astype(np.float32)[first[1][0]]
    assert abs(np.linalg.det(mapped[1:]-mapped[0])) == pytest.approx(1/16)


def test_healthy_atlas_is_exact_noop_and_degenerate_space_is_rejected():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    mapping = np.arange(3, dtype=np.uint32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    uv = vertices[:, :2].copy()
    actual = _repair_metric_uv_triangles(vertices, mapping, faces, uv, scale_m=1.)
    assert all(a is b for a, b in zip(actual, (mapping, faces, uv), strict=True))
    with pytest.raises(ValueError, match="spatially degenerate"):
        _repair_metric_uv_triangles(vertices*0, mapping, faces, uv*0, scale_m=1.)


def test_severely_stretched_guide_chart_is_repaired_on_original_spatial_triangle():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., .1]])
    mapping = np.arange(3, dtype=np.uint32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    # Finite positive UV area alone does not detect this 1000:1 distortion.
    uv = np.array([[0., 0.], [1., 0.], [0., .001]])
    m, f, result = _repair_metric_uv_triangles(vertices, mapping, faces, uv, scale_m=4.)
    np.testing.assert_array_equal(vertices[m[f]], vertices[faces])
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        assert np.linalg.norm(result[a]-result[b])*4 == pytest.approx(
            np.linalg.norm(vertices[a]-vertices[b]), rel=1e-6)


def test_measured_rough_qem_patch_has_bounded_charts_without_smoothing_exported_rock():
    from pathlib import Path

    from plume_advanced.stages.geometry_export import _xatlas_metric_uvs
    from plume_advanced.stages.surface_frames import angle_weighted_vertex_normals, mesh_tangents
    from plume_advanced.validation import PortableAssetValidator

    with np.load(Path(__file__).parent / 'fixtures/geometry/qem_uv_chart_seed0.npz') as saved:
        vertices, faces = saved['vertices'], saved['faces']
    before_v, before_f = vertices.copy(), faces.copy()
    first = _xatlas_metric_uvs(vertices, faces, scale_m=4.)
    m, f, uv = first
    np.testing.assert_array_equal(vertices[m[f]], vertices[faces])
    np.testing.assert_array_equal(vertices, before_v)
    np.testing.assert_array_equal(faces, before_f)
    n = angle_weighted_vertex_normals(vertices, faces)[m]
    t = mesh_tangents(vertices[m], f, uv, n)
    validator = object.__new__(PortableAssetValidator)
    validator.manifest = {'cave': {'material': {'uv_scale_m': 4.}}}
    validator._geometry_arrays = lambda: (vertices[m], f, n, t, uv.astype(np.float32).astype(float))
    checks = validator._uv_checks()
    assert all(c.passed for c in checks), [(c.name, c.detail) for c in checks if not c.passed]
    second = _xatlas_metric_uvs(vertices, faces, scale_m=4.)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scale_m", [.5, 4., 32.])
def test_measured_small_qem_triangles_have_metric_uvs_and_correct_tangents(scale_m):
    from pathlib import Path

    from plume_advanced.stages.surface_frames import mesh_tangents
    from plume_advanced.validation import PortableAssetValidator

    with np.load(Path(__file__).parent / 'fixtures/geometry/qem_uv_slivers_seed0.npz') as saved:
        vertices, faces = saved['vertices'], saved['faces']
    original = vertices.copy()
    mapping, repaired_faces, uv = _repair_metric_uv_triangles(
        vertices, np.arange(len(vertices)), faces, np.zeros((len(vertices), 2)), scale_m=scale_m)
    positions = vertices[mapping]
    normals = np.tile([0., 0., -1.], (len(positions), 1))
    tangents = mesh_tangents(positions, repaired_faces, uv, normals)
    for face in repaired_faces:
        tri = positions[face]
        chart = uv.astype(np.float32).astype(float)[face]
        assert abs(np.linalg.det(chart[1:] - chart[0])) > 0.
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            assert np.linalg.norm(chart[a] - chart[b]) * scale_m == pytest.approx(
                np.linalg.norm(tri[a] - tri[b]), rel=1e-6)
        edge = tri[1] - tri[0]
        np.testing.assert_allclose(tangents[face, :3], np.tile(edge / np.linalg.norm(edge), (3, 1)))
    np.testing.assert_array_equal(vertices, original)
    # Exercise the portable UV gate as well as the producer, including a real
    # zero-area corruption. These independent checks use the exported arrays.
    validator = object.__new__(PortableAssetValidator)
    validator.manifest = {'cave': {'material': {'uv_scale_m': scale_m}}}
    validator._geometry_arrays = lambda: (positions, repaired_faces, normals, tangents, uv)
    checks = validator._uv_checks()
    assert next(c for c in checks if c.name == 'No collapsed UV triangles').passed
    uv[repaired_faces[0]] = 0.
    assert not next(c for c in validator._uv_checks() if c.name == 'No collapsed UV triangles').passed

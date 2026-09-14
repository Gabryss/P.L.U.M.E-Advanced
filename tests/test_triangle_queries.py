import numpy as np
import pytest
import trimesh

from plume_advanced.stages.triangle_queries import (
    TriangleIndex,
    segment_distances,
    segment_triangle_distances,
    vertical_clearances,
)


@pytest.mark.parametrize("start,end,distance", [
    ((0, 0, -1), (0, 0, 1), 0),
    ((0, 0, 1), (0, 0, 2), 1),
    ((-3, 0, 0), (3, 0, 0), 0),
    ((0, 0, 2), (0, 0, 2), 2),
])
def test_segment_triangle_crossings_and_degeneracy(start, end, distance):
    tri = np.array([[[-1., -1, 0], [1, -1, 0], [0, 1, 0]]])
    assert segment_triangle_distances(start, end, tri)[0] == pytest.approx(distance)


def test_distance_handles_parallel_and_point_segments():
    assert segment_distances(np.array([0., 0, 0]), np.array([1., 0, 0]),
                             np.array([2., 1, 0]), np.array([3., 1, 0])) == pytest.approx(2**.5)
    assert segment_distances(np.zeros(3), np.zeros(3), np.ones(3), np.ones(3)) == pytest.approx(3**.5)


def test_triangle_index_matches_brute_force_with_large_and_small_faces():
    small = trimesh.creation.icosphere(subdivisions=2, radius=.1)
    large = trimesh.creation.box(extents=(200, 1, 1))
    large.apply_translation([0, 10, 0])
    mesh = small+large
    points = np.random.default_rng(9).uniform([-110, -2, -2], [110, 12, 2], (80, 3))
    expected = trimesh.proximity.closest_point_naive(mesh, points)[1]
    index = TriangleIndex(mesh.vertices, mesh.faces)
    np.testing.assert_allclose(index.distances(points, batch_size=7), expected, atol=1e-10)


@pytest.mark.parametrize("height", [0., 1.])
def test_sweep_detects_thin_obstacle_between_clear_endpoints(height):
    # Endpoints are each 1 m from the sheet, but the complete sweep crosses it.
    tri = np.array([[0., -1, -1], [0, 1, -1], [0, 0, 1]])
    index = TriangleIndex(tri, [[0, 1, 2]])
    assert index.swept_capsule_distance([-1., 0, 0], [1., 0, 0], height, .25) == 0


def test_capsule_ribbon_detects_obstacle_between_its_upper_and_lower_axes():
    tri = np.array([[0., -.1, -.1], [0, .1, -.1], [0, 0, .1]])
    index = TriangleIndex(tri, [[0, 1, 2]])
    assert index.swept_capsule_distance([-1., 0, 0], [1., 0, 0], 2, .25) == 0


def test_sphere_sweep_reports_clearance_to_floor():
    mesh = trimesh.creation.box(extents=(10, 2, 2))
    index = TriangleIndex(mesh.vertices, mesh.faces)
    assert index.swept_capsule_distance([-2., 0, 0], [2., 0, 0], 0, 1.1) == pytest.approx(1)


def test_vertical_clearance_keeps_stacked_cavities_separate():
    lower = trimesh.creation.box(extents=(4, 4, 2))
    upper = trimesh.creation.box(extents=(4, 4, 2))
    upper.apply_translation([0, 0, 4])
    mesh = lower + upper
    points = np.array([[0., 0, .25], [0, 0, 2], [0, 0, 4.5], [0, 0, 1]])
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    rows = vertical_clearances(vertices, faces, points)
    assert [row['inside'] for row in rows] == [True, False, True, False]
    assert rows[0]['floor_distance_m'] == pytest.approx(1.25)
    assert rows[0]['roof_distance_m'] == pytest.approx(.75)
    assert rows[2]['floor_distance_m'] == pytest.approx(1.5)
    assert rows[2]['roof_distance_m'] == pytest.approx(.5)
    assert rows[1]['clearance_m'] is None
    assert rows[3]['clearance_m'] is None
    np.testing.assert_array_equal(vertices, mesh.vertices)
    np.testing.assert_array_equal(faces, mesh.faces)


def test_vertical_clearance_empty_probe_batch():
    mesh = trimesh.creation.box()
    assert vertical_clearances(mesh.vertices, mesh.faces, np.empty((0, 3))) == []

"""Actual triangle obstructions, deterministic repair and bounded failure."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.mesh_inspection import MeshInspectionError, inspect_surface
from plume_advanced.stages.route_placement import repair_vertical_path
from plume_advanced.stages.triangle_queries import TriangleIndex


def sweep(index, path, height=.5):
    return np.array([index.swept_capsule_distance(a, b, height-.5, .27)
                     for a, b in zip(path, path[1:])])


def fixture():
    with np.load(Path(__file__).parent/'fixtures/geometry/route_placement_seed42.npz') as data:
        return {name: data[name].copy() for name in data.files}


def test_rejected_seed_42_has_a_clear_corridor_without_changing_geometry():
    data = fixture()
    snapshot = {name: value.copy() for name, value in data.items()}
    index = TriangleIndex(data['vertices'], data['faces'])
    distances = sweep(index, data['path'])
    assert (distances < .27-1e-7).sum() == 4
    def repair():
        return repair_vertical_path(index, data['path'], data['lower'], data['upper'],
                                    distances, height=.5, width=.5, margin=.02)
    fixed, report = repair()
    assert report['passed'] and report['changed_stations'] > 0
    assert 0 < report['queries'] <= 20000
    assert np.all(sweep(index, fixed) >= .27-1e-7)
    assert np.array_equal(fixed[:, :2], data['path'][:, :2])
    assert np.array_equal(fixed[[0, -1]], data['path'][[0, -1]])
    assert np.all(fixed[:, 2] >= data['lower'])
    assert np.all(fixed[:, 2] <= data['upper'])
    again, repeated = repair()
    assert report == repeated and np.array_equal(again, fixed)
    for name in data:
        assert np.array_equal(data[name], snapshot[name])


@pytest.mark.parametrize('attempts,budget', [(0, 20000), (3, 0), (3, 1)])
def test_exhaustion_returns_original_path_without_partial_repair(attempts, budget):
    data = fixture()
    index = TriangleIndex(data['vertices'], data['faces'])
    fixed, report = repair_vertical_path(index, data['path'], data['lower'], data['upper'],
                                        sweep(index, data['path']), height=.5, width=.5,
                                        margin=.02, attempts=attempts, max_queries=budget)
    assert not report['passed']
    assert report['queries'] <= budget
    assert np.array_equal(fixed, data['path'])


def test_wall_cannot_be_repaired_by_moving_the_body_through_rock():
    wall = trimesh.creation.box(extents=(4, .01, 4))
    index = TriangleIndex(wall.vertices, wall.faces)
    path = np.array([(0, y, 0) for y in np.arange(-2, 2.01, .25)])
    fixed, report = repair_vertical_path(index, path, np.full(len(path), -.5),
                                        np.full(len(path), .5), sweep(index, path),
                                        height=.5, width=.5, margin=.02)
    assert not report['passed'] and report['queries'] > 0
    assert np.array_equal(fixed, path)


@pytest.mark.parametrize('interval', [(1., 0.), (np.nan, 1.)])
def test_invalid_vertical_interval_fails_without_search(interval):
    data = fixture()
    index = TriangleIndex(data['vertices'], data['faces'])
    data['lower'][1], data['upper'][1] = interval
    fixed, report = repair_vertical_path(index, data['path'], data['lower'], data['upper'],
                                        sweep(index, data['path']), height=.5, width=.5, margin=.02)
    assert not report['passed'] and report['queries'] == 0
    assert np.array_equal(fixed, data['path'])


def stepped_cave():
    # Concave prism: a roof step changes the midpoint suddenly but leaves a
    # continuous low corridor. All faces remain a closed genus-zero surface.
    polygon = [(-3, 0), (3, 0), (3, .7), (-.1, .7), (-.1, 2), (-3, 2)]
    vertices = [(x, y, z) for x in (-1, 1) for y, z in polygon]
    cap = np.array([(0, 1, 3), (1, 2, 3), (0, 3, 4), (0, 4, 5)])
    faces = cap.tolist() + (cap+6).tolist()
    for i in range(6):
        j = (i+1) % 6
        faces.extend([(i, j, j+6), (i, j+6, i+6)])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


@pytest.mark.parametrize('height,scale', [(.5, 1.), (1.2, 2.)])
def test_surface_gate_records_and_independently_verifies_vertical_repair(height, scale):
    mesh = stepped_cave()
    mesh.vertices[:, 2] *= scale
    path = np.array([(0, y, .3) for y in np.arange(-2, 2.01, .25)])
    kwargs = dict(required_paths=(path,), route_segment_ids=(7,),
                  route_height_m=height, route_width_m=.5, route_margin_m=.02)
    with pytest.raises(MeshInspectionError, match='capsule'):
        inspect_surface(mesh.vertices, mesh.faces, **kwargs, route_placement_repair_attempts=0)
    report = inspect_surface(mesh.vertices, mesh.faces, **kwargs)
    result = report['traversal']['paths'][0]
    assert result['placement_repair']['passed'] and result['placement_repair']['verified']
    assert result['passed'] and not result['blocked_edges']
    assert np.all(sweep(TriangleIndex(mesh.vertices, mesh.faces), result['center_path_m'], height) >= .27-1e-7)
    assert report == inspect_surface(mesh.vertices, mesh.faces, **kwargs)


def test_incorrect_search_success_is_rejected_by_independent_verification(monkeypatch):
    def incorrect(index, points, *args, **kwargs):
        return points, dict(passed=True, queries=0)
    monkeypatch.setattr('plume_advanced.stages.route_placement.repair_vertical_path', incorrect)
    mesh = stepped_cave()
    path = np.array([(0, y, .3) for y in np.arange(-2, 2.01, .25)])
    with pytest.raises(MeshInspectionError) as error:
        inspect_surface(mesh.vertices, mesh.faces, required_paths=(path,), route_segment_ids=(0,),
                        route_height_m=.5, route_width_m=.5)
    repair = error.value.report['traversal']['paths'][0]['placement_repair']
    assert not repair['passed'] and not repair['verified']


def test_search_budget_is_shared_by_all_paths_in_one_inspection():
    mesh = stepped_cave()
    path = np.array([(0, y, .3) for y in np.arange(-2, 2.01, .25)])
    kwargs = dict(route_height_m=.5, route_width_m=.5)
    report = inspect_surface(mesh.vertices, mesh.faces, required_paths=(path,), route_segment_ids=(0,), **kwargs)
    queries = report['traversal']['placement_queries']
    with pytest.raises(MeshInspectionError) as error:
        inspect_surface(mesh.vertices, mesh.faces, required_paths=(path, path), route_segment_ids=(0, 1),
                        route_placement_max_sweeps=queries, **kwargs)
    traversal = error.value.report['traversal']
    assert traversal['placement_queries'] == queries
    assert traversal['paths'][0]['passed'] and not traversal['paths'][1]['passed']


@pytest.mark.parametrize('name,maximum', [('route_placement_repair_attempts', 4),
                                       ('route_placement_max_sweeps', 200000)])
@pytest.mark.parametrize('value', [-1, True, 1.2, '3', float('nan')])
def test_placement_budget_configuration_is_strict(name, maximum, value):
    with pytest.raises(ValueError):
        GeometryConfig(**{name: value})
    with pytest.raises(ValueError):
        GeometryConfig(**{name: maximum+1})

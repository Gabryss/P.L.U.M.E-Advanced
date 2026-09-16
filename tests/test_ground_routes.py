"""Reference chassis limits, actual obstacles, bounded deterministic route repair."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from plume_advanced.acceptance import build_acceptance_policy
from plume_advanced.stages.box_queries import sweep_box, translating_box_hits
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.ground_routes import GroundRobot, inspect_ground_routes
from plume_advanced.stages.mesh_inspection import MeshInspectionError, inspect_surface
from plume_advanced.stages.triangle_queries import TriangleIndex


def corridor(floor=lambda x, y: 0., *, xs=None, ys=None, roof=3.):
    xs = np.linspace(-5, 5, 41) if xs is None else np.asarray(xs)
    ys = np.linspace(-2, 2, 17) if ys is None else np.asarray(ys)
    positions = np.array([(x, y, floor(x, y)) for x in xs for y in ys])
    n, stride = len(positions), len(ys)
    vertices = np.vstack([positions, positions * [1, 1, 0] + [0, 0, roof]])
    faces = []
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            a = i*stride+j
            for triangle in ((a, a+stride, a+1), (a+1, a+stride, a+stride+1)):
                faces.extend([triangle, tuple(k+n for k in triangle[::-1])])
    perimeter = ([i*stride for i in range(len(xs))]
                 + [n-stride+j for j in range(1, stride)]
                 + [i*stride+stride-1 for i in range(len(xs)-2, -1, -1)]
                 + list(range(stride-2, 0, -1)))
    for a, b in zip(perimeter, perimeter[1:]+perimeter[:1]):
        faces.extend([(a, a+n, b), (b, a+n, b+n)])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.fix_normals()
    assert mesh.is_watertight
    return mesh


def inspect(mesh, *, path=None, **kwargs):
    path = np.array([[-3., 0., 1.], [3., 0., 1.]]) if path is None else path
    return inspect_ground_routes(mesh.vertices, mesh.faces, (path,), (7,), **kwargs)


def test_flat_floor_checks_full_length_and_stores_continuous_native_witnesses():
    mesh = corridor()
    report = inspect(mesh)
    assert report['passed'] and report['robot'] == {
        'length_m': .7, 'width_m': .5, 'height_m': .5, 'margin_m': .02,
        'max_slope_deg': 20., 'max_step_m': .1, 'support_spacing_m': .1}
    route = report['paths'][0]
    assert route['samples'] == 61 and len(route['poses']) == 61
    assert len(route['sweeps']) == 60
    assert all(p['passed'] and abs(p['center_m'][2]-.273) < 1e-9 for p in route['poses'])
    assert all(np.allclose(s['half_extents_m'], [.37, .27, .27]) for s in route['sweeps'])
    assert report == inspect(mesh)


@pytest.mark.parametrize('degrees,passed', [(0, True), (19, True), (20, True), (21, False), (-21, False)])
def test_ramps_enforce_approved_slope_in_both_directions(degrees, passed):
    slope = np.tan(np.radians(degrees))
    mesh = corridor(lambda x, y: slope*x, roof=5)
    path = np.array([[-1., 0., 1.], [1., 0., 1.]])
    result = inspect(mesh, path=path, repair_attempts=0)
    assert result['passed'] == passed
    if passed:
        assert max(p['slope_deg'] for p in result['paths'][0]['poses']) == pytest.approx(abs(degrees))
    else:
        assert any(p.get('failure') == 'slope_limit' for p in result['paths'][0]['poses'])


@pytest.mark.parametrize('step,passed', [(0., True), (.15, False), (-.15, False)])
def test_steps_and_drops_cannot_pass_by_floating_above_the_floor(step, passed):
    mesh = corridor(lambda x, y: step if x > 0 else 0., xs=[-5, 0, .001, 5])
    result = inspect(mesh, repair_attempts=0)
    assert result['passed'] == passed


def test_step_limit_is_necessary_not_a_guarantee_of_chassis_clearance():
    mesh = corridor(lambda x, y: .06 if x > 0 else 0., xs=[-5, 0, .001, 5])
    result = inspect(mesh, repair_attempts=0)
    route = result['paths'][0]
    assert all(p['step_m'] < .1 for p in route['poses'])
    # A box cannot climb this abrupt edge along the proposed poses. Retaining
    # the collision rejection is required even though its height is below 10 cm.
    assert not result['passed'] and route['failed_edges']


def test_low_ceiling_blocks_long_body_even_when_floor_is_valid():
    mesh = corridor(roof=.52)
    report = inspect(mesh, path=np.array([[-1., 0., .3], [1., 0., .3]]), repair_attempts=0)
    assert not report['passed'] and report['paths'][0]['failed_edges']


def test_sampled_floor_hole_is_rejected():
    mesh = corridor(lambda x, y: -2 if -.2 <= x <= .2 else 0., xs=[-5, -.21, -.2, .2, .21, 5])
    report = inspect(mesh, repair_attempts=0)
    assert not report['passed']


def test_local_detour_is_reproducible_preserves_mesh_and_anchors():
    # A low, local ridge fails support/step checks, but either side is open.
    mesh = corridor(lambda x, y: .25 if abs(x) <= .3 and abs(y) <= .2 else 0.,
                    xs=[-5, -.31, -.3, .3, .31, 5], ys=[-2, -.21, -.2, .2, .21, 2])
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    path = np.array([[-4., 0., 1.], [4., 0., 1.]])
    failed = inspect(mesh, path=path, repair_attempts=0)
    assert not failed['passed']
    result = inspect(mesh, path=path)
    assert result['passed']
    repaired = result['paths'][0]
    assert repaired['placement_repair']['verified']
    assert repaired['placement_repair']['maximum_lateral_change_m'] > .25
    np.testing.assert_array_equal(np.asarray(repaired['plan_path_m'])[[0, -1]], path)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    assert result == inspect(mesh, path=path)


@pytest.mark.parametrize('budget', [0, 1, 10])
def test_budget_exhaustion_fails_closed(budget):
    report = inspect(corridor(), max_queries=budget)
    assert not report['passed'] and report['queries'] == budget
    assert 'budget exhausted' in report['failures'][-1]


def test_collision_sat_catches_thin_wall_between_clear_endpoints_and_length_corners():
    wall = trimesh.creation.box(extents=(.001, 4, 4))
    tri = wall.vertices[wall.faces]
    half = [.35, .25, .25]
    assert not translating_box_hits(tri, [-1, 0, 0], [-1, 0, 0], np.eye(3), half)
    assert translating_box_hits(tri, [-1, 0, 0], [1, 0, 0], np.eye(3), half)
    assert translating_box_hits(tri, [1, 0, 0], [-1, 0, 0], np.eye(3), half)
    assert translating_box_hits(tri, [.3, 0, 0], [.3, 0, 0], np.eye(3), half)
    assert not translating_box_hits(tri, [.3, 0, 0], [.3, 0, 0], np.eye(3), [.25]*3)


def test_rotating_box_catches_obstacle_between_clear_orientations():
    obstacle = trimesh.creation.box(extents=(.03, .03, .1), transform=trimesh.transformations.translation_matrix([.34, .34, 0]))
    index = TriangleIndex(obstacle.vertices, obstacle.faces)
    end = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    half = [.5, .05, .05]
    assert not translating_box_hits(index.triangles, [0, 0, 0], [0, 0, 0], np.eye(3), half)
    assert not translating_box_hits(index.triangles, [0, 0, 0], [0, 0, 0], end, half)
    assert sweep_box(index, [0, 0, 0], [0, 0, 0], np.eye(3), end, half) is None


@pytest.mark.parametrize('scale', [.001, 1., 1000.])
def test_continuous_translation_agrees_with_independent_barycentric_solver(scale):
    from scipy.optimize import linprog

    rng = np.random.default_rng(20260915 + int(np.log10(scale)))
    for _ in range(160):
        origin = rng.uniform(-1000, 1000, 3)
        triangle = rng.normal(size=(3, 3))
        start, end = rng.normal(size=(2, 3))
        basis = Rotation.random(random_state=rng).as_matrix()
        half = rng.uniform(.1, .9, 3)
        # A barycentric triangle point must equal a point in the moving box.
        # Normalize only the independent LP to avoid metre-dependent tolerance.
        matrix = np.zeros((4, 7))
        matrix[:3, :3], matrix[:3, 3], matrix[:3, 4:] = triangle.T, -(end-start), -basis
        matrix[3, :3] = 1.
        reference = linprog(np.zeros(7), A_eq=matrix, b_eq=np.r_[start, 1.],
                            bounds=[(0, 1)]*4 + [(-h, h) for h in half], method='highs')
        assert reference.status in (0, 2), reference.message
        for first, last in ((start, end), (end, start)):
            assert translating_box_hits([origin + scale*triangle], origin + scale*first,
                                        origin + scale*last, basis, scale*half) == (reference.status == 0)


def test_ground_failure_is_a_mandatory_surface_gate():
    mesh = corridor(lambda x, y: .5*x, roof=5)
    with pytest.raises(MeshInspectionError, match='ground route') as error:
        inspect_surface(mesh.vertices, mesh.faces, required_paths=([[-1, 0, 1], [1, 0, 1]],),
                        route_segment_ids=(7,), ground_robot=GroundRobot(), ground_repair_attempts=0)
    assert not error.value.report['ground_traversal']['passed']


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '20'])
@pytest.mark.parametrize('key', ['length_m', 'max_slope_deg', 'max_step_m', 'support_spacing_m'])
def test_invalid_reference_limits_rejected(key, value):
    with pytest.raises(ValueError):
        replace(GroundRobot(), **{key: value})


def test_approved_config_requires_clearance_and_serializes_limits():
    policy = build_acceptance_policy({'profile': 'simulation', 'require_ground_routes': True})
    assert policy.require_clearance and policy.robot_length_m == .7
    with pytest.raises(ValueError, match='positive route'):
        GeometryConfig(ground_robot_length_m=.7)
    cfg = GeometryConfig(ground_robot_length_m=.7, required_route_height_m=.5, required_route_width_m=.5)
    assert cfg.ground_max_step_m == .1


def test_junction_turns_are_checked_in_addition_to_independent_paths():
    mesh = corridor()
    paths = (np.array([[-2., 0., 1.], [0., 0., 1.]]), np.array([[0., 0., 1.], [0., 1., 1.]]))
    report = inspect_ground_routes(mesh.vertices, mesh.faces, paths, (1, 2))
    assert report['passed'] and len(report['junctions']) == 1
    junction = report['junctions'][0]
    assert junction['passed'] and junction['samples'] > 2 and junction['sweeps']
    assert junction['segment_ids'] == [1, 2]


def test_narrow_junction_rejects_turn_between_two_clear_approaches():
    from collections import Counter

    axis = [-3., -.28, .28, 3.]
    vertices = np.array([(x, y, z) for z in [0., 2.] for x in axis for y in axis])
    floor = []
    for i in range(3):
        for j in range(3):
            if i == 1 or j == 1:
                a = i*4+j
                floor.extend([(a, a+4, a+1), (a+1, a+4, a+5)])
    counts = Counter(tuple(sorted((t[i], t[(i+1) % 3]))) for t in floor for i in range(3))
    faces = floor + [tuple(v+16 for v in t[::-1]) for t in floor]
    for (a, b), count in counts.items():
        if count == 1:
            faces.extend([(a, b, b+16), (a, b+16, a+16)])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    assert mesh.is_watertight
    paths = (np.array([[-2., 0., 1.], [0., 0., 1.]]),
             np.array([[0., 0., 1.], [0., 2., 1.]]))
    result = inspect_ground_routes(mesh.vertices, mesh.faces, paths, (0, 1), repair_attempts=0)
    assert all(path['passed'] for path in result['paths'])
    assert result['junctions'] and not result['junctions'][0]['passed']
    assert not result['passed']


def test_vertical_section_center_connector_uses_incident_ground_heading():
    mesh = corridor()
    paths = (np.array([[-2., 0., 1.], [0., 0., 1.]]),
             np.array([[0., 0., 1.01], [1., 0., 1.01]]),
             np.array([[0., 0., 1.], [0., 0., 1.01]]))
    report = inspect_ground_routes(mesh.vertices, mesh.faces, paths, (0, 1, -2))
    assert report['passed'] and len(report['junctions']) == 2
    poses = report['paths'][-1]['poses']
    np.testing.assert_allclose(poses[0]['center_m'], poses[1]['center_m'])


def test_isolated_vertical_route_fails_without_warning_or_nan():
    mesh = corridor()
    with np.errstate(all='raise'):
        report = inspect_ground_routes(mesh.vertices, mesh.faces, ([[0, 0, 1], [0, 0, 1.01]],), (-1,))
    assert not report['passed']
    assert report['paths'][0]['placement_repair']['failure'] == 'No valid horizontal route heading'


def test_blocked_vertical_connector_does_not_normalize_a_zero_detour_heading():
    mesh = corridor(lambda x, y: .15 if x > 0 else 0., xs=[-5, 0, .001, 5])
    paths = (np.array([[-2., 0., 1.], [0., 0., 1.]]),
             np.array([[0., 0., 1.01], [1., 0., 1.01]]),
             np.array([[0., 0., 1.], [0., 0., 1.01]]))
    with np.errstate(all='raise'):
        report = inspect_ground_routes(mesh.vertices, mesh.faces, paths, (0, 1, -2))
    assert not report['passed']
    connector = report['paths'][-1]
    assert not connector['passed']
    assert connector['placement_repair']['failure'] == 'Anchored vertical connector has no lateral detour'
    assert connector['placement_repair']['windows'] == []

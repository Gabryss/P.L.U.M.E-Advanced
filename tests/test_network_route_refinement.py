"""Continuous routing proposals and bounded late capture, independent of seeds."""

from types import SimpleNamespace

import numpy as np
from scipy.optimize import check_grad
from scipy.sparse import csr_matrix

from plume_advanced.stages.network_front_capture import capture_path
from plume_advanced.stages.network_route_relaxation import relax_route, route_energy, sample_cost


def field_fixture():
    x, y = np.linspace(-10, 110, 61), np.linspace(-20, 20, 41)
    xx, yy = np.meshgrid(x, y)
    # A smooth, curved low-cost corridor. No artificial route noise is needed.
    field = 1 + .025 * (yy - 6*np.sin(np.pi*xx/100))**2
    return SimpleNamespace(x=x, y=y, width=4., cost_fields=field[None],
                           config=SimpleNamespace(quality=SimpleNamespace(maximum_uphill_grade=.08)),
                           sample_host=lambda points: np.column_stack((np.zeros(len(points)),
                                                                       np.ones((len(points), 2)))))


def test_bilinear_cost_derivative_and_route_energy_gradient():
    p = field_fixture()
    xy = np.array([[1.1, 2.3], [8.1, 3.3], [14.2, 4.1], [21.1, 3.6]])
    f = p.cost_fields[0]
    value, derivative = sample_cost(f, p.x, p.y, xy)
    for axis in (0, 1):
        offset = np.zeros_like(xy)
        offset[:, axis] = 1e-5
        finite = (sample_cost(f, p.x, p.y, xy+offset)[0] - value)/1e-5
        np.testing.assert_allclose(finite, derivative[:, axis], atol=1e-8)
    error = check_grad(lambda flat: route_energy(flat.reshape(-1, 2), f, p.x, p.y, .7)[0],
                       lambda flat: route_energy(flat.reshape(-1, 2), f, p.x, p.y, .7)[1].ravel(),
                       xy.ravel())
    assert error < 1e-4


def test_relaxation_follows_cost_without_moving_anchors_or_input():
    p = field_fixture()
    raw = np.column_stack((np.linspace(0, 100, 21), np.zeros(21)))
    before, cost_before = raw.copy(), p.cost_fields.copy()
    xy, report = relax_route(p, raw, 0)
    replay, repeated = relax_route(p, raw, 0)
    assert report == repeated and np.array_equal(xy, replay)
    assert np.array_equal(raw, before) and np.array_equal(p.cost_fields, cost_before)
    assert np.array_equal(xy[[0, -1]], raw[[0, -1]])
    assert report["accepted"] and report["energy_after"] < report["energy_before"]
    assert report["iterations"] <= 40
    assert report["evaluations"] <= 401
    assert np.linalg.norm(xy-raw, axis=1).max() <= 2*p.width + 1e-9
    assert xy[len(xy)//2, 1] > 3
    # Flip the host preference: the route must respond, not keep a seeded wiggle.
    p.cost_fields = p.cost_fields[:, ::-1].copy()
    mirrored, _ = relax_route(p, raw, 0)
    assert mirrored[len(xy)//2, 1] < -3


def test_continuous_proposal_rejects_invalid_strip_between_vertices():
    p = field_fixture()
    raw = np.column_stack((np.linspace(0, 100, 21), np.zeros(21)))

    def sample(points):
        values = np.column_stack((np.zeros(len(points)), np.ones((len(points), 2))))
        values[(points[:, 0] > 51) & (points[:, 0] < 53) & (points[:, 1] > 1), 1] = -1
        return values

    p.sample_host = sample
    xy, report = relax_route(p, raw, 0)
    assert np.array_equal(xy, raw)
    assert not report["accepted"] and report["reason"] == "host_constraint"


def test_relaxation_does_not_manufacture_curves_in_uniform_field():
    p = field_fixture()
    p.cost_fields[:] = 1
    raw = np.column_stack((np.linspace(0, 100, 21), np.zeros(21)))
    xy, report = relax_route(p, raw, 0)
    assert np.array_equal(xy, raw) and not report["accepted"]


def test_cost_holes_do_not_propagate_nan_to_route_or_audit():
    p = field_fixture()
    raw = np.column_stack((np.linspace(0, 100, 21), np.zeros(21)))
    p.cost_fields[0, 0, 0] = np.nan
    xy, report = relax_route(p, raw, 0)
    assert np.isfinite(xy).all()
    assert np.isfinite(report["energy_after"])
    p.cost_fields[:] = np.nan
    xy, report = relax_route(p, raw, 0)
    assert np.array_equal(xy, raw) and report["reason"] == "invalid_cost_field"


def capture_fixture():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [2., 2.], [3., 2.], [4., 2.]])
    p = SimpleNamespace(xy=xy, width=1., config=SimpleNamespace(
        quality=SimpleNamespace(maximum_turn_degrees=55, minimum_bend_radius_widths=.65)))
    graph = csr_matrix(([.2, 1.414, 1.414, 1., 1.], ([0, 0, 2, 3, 4], [1, 2, 3, 4, 5])), shape=(6, 6))
    return p, graph


def test_capture_looks_beyond_cheapest_dead_end_and_replays():
    p, graph = capture_fixture()
    args = (p, graph, 0, np.array([1., 0.]), 1., {4: [5]}, {0, 4, 5}, [4], 6.)
    route, work = capture_path(*args)
    assert route == [0, 2, 3, 4] and 0 < work <= 256
    assert (route, work) == capture_path(*args)
    assert capture_path(*args, maximum_expansions=1) == (None, 1)
    assert capture_path(*args, maximum_expansions=0) == (None, 0)


def test_capture_respects_remaining_length_occupancy_and_receiving_heading():
    p, graph = capture_fixture()
    def run(remaining=6., occupied=None):
        return capture_path(p, graph, 0, np.array([1., 0.]), 1., {4: [5]},
                            occupied or {0, 4, 5}, [4], remaining)[0]
    assert run(remaining=3) is None
    assert run(occupied={0, 2, 4, 5}) is None
    p.xy[5] = [2., 2.]
    assert run() is None


def test_capture_host_cost_guide_preserves_route_and_bounded_work():
    from scipy.sparse.csgraph import dijkstra

    p, graph = capture_fixture()
    guide = dijkstra(graph.T.tocsr(), indices=[4], min_only=True)
    args = (p, graph, 0, np.array([1., 0.]), 1., {4: [5]}, {0, 4, 5}, [4], 6.)
    route, work = capture_path(*args, cost_to_targets=guide)
    assert route == capture_path(*args)[0]
    assert 0 < work <= 256
    assert capture_path(*args, cost_to_targets=guide, maximum_expansions=1) == (None, 1)
    assert capture_path(*args, cost_to_targets=np.full(6, np.inf)) == (None, 0)

"""Clearance depends on physical junctions, not graph subdivision."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plume_advanced.stages.network import CaveNetworkGenerator, CavePoint, CaveSegment
from plume_advanced.stages.network_layers import NetworkLayersConfig, assess_layers
from plume_advanced.stages.network_neighborhoods import passage_conflicts
from plume_advanced.stages.network_quality import _crossings
from plume_advanced.stages.network_regional_geometry import AcceptedRoutes


def segment(sid, first, last, xy, width=6.):
    xy = np.array(xy, dtype=float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    return CaveSegment(sid, first, last, "backbone", 0,
        tuple(CavePoint(i, *p, 100., 0., 30., 1., .5, float(s), width)
              for i, (p, s) in enumerate(zip(xy, arc))),
        {"regional_start_layer": 0, "regional_end_layer": 0})


def separation(segments):
    network = SimpleNamespace(config=SimpleNamespace(layers=NetworkLayersConfig(enabled=True)),
                              segments=segments, nodes=[], backend_provenance={})
    checks = {}

    def check(name, passed, *_):
        checks[name] = bool(passed)

    assess_layers(network, None, check)
    return checks["layer_passage_separation"]


@pytest.mark.parametrize("pieces", [1, 2, 5, 12])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_junction_subdivision_preserves_clearance(pieces, reverse_order):
    a = segment(0, 0, 2, [(-100, 0), (0, 0)])
    b = segment(1, 1, 2, [(-100, -100), (-6, -3), (0, 0)])
    c = segment(2, 2, 3, [(0, 0), (100, 0)])
    assert separation([a, b, c])
    points = np.linspace([-6, -3], [0, 0], pieces + 1)
    nodes = [*range(4, 4 + pieces), 2]
    split = [a, segment(1, 1, 4, [(-100, -100), (-6, -3)]), c]
    split += [segment(3 + i, nodes[i], nodes[i+1], points[i:i+2]) for i in range(pieces)]
    if reverse_order:
        split.reverse()
    assert separation(split)


def test_unrelated_passage_is_not_exempt_inside_junction_region():
    a = segment(0, 0, 2, [(-100, 0), (0, 0)])
    b = segment(1, 1, 2, [(-100, -100), (0, 0)])
    c = segment(2, 2, 3, [(0, 0), (100, 0)])
    unrelated = segment(3, 4, 5, [(-5, 2), (5, 2)])
    assert separation([a, b, c])
    assert not separation([a, b, c, unrelated])


def test_connected_arms_still_fail_away_from_junction():
    a = segment(0, 0, 2, [(-100, 0), (0, 0)])
    b = segment(1, 1, 2, [(-100, -2), (-50, -2), (0, 0)])
    c = segment(2, 2, 3, [(0, 0), (100, 0)])
    assert not separation([a, b, c])


def test_degree_two_chain_cannot_hide_remote_self_approach():
    parts = [segment(0, 0, 1, [(0, 0), (100, 0)]),
             segment(1, 1, 2, [(100, 0), (100, 100), (0, 100), (0, 2)]),
             segment(2, 2, 3, [(0, 2), (80, 2)])]
    assert not separation(parts)


@pytest.mark.parametrize("split", [False, True])
def test_nonlocal_self_approach_rejected_before_and_after_subdivision(split):
    xy = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 2), (80, 2)]
    passages = ([segment(0, 0, 1, xy[:2]), segment(1, 1, 2, xy[1:5]),
                 segment(2, 2, 3, xy[4:])] if split else [segment(0, 0, 3, xy)])
    assert not separation(passages)


def test_straight_and_wide_bend_are_not_self_conflicts():
    assert separation([segment(0, 0, 1, [(0, 0), (200, 0)])])
    angle = np.linspace(0, np.pi, 200)
    assert separation([segment(0, 0, 1, np.c_[50 * np.cos(angle), 50 * np.sin(angle)])])


@pytest.mark.parametrize("height,conflict", [(0., True), (8., False)])
def test_self_approach_respects_actual_vertical_clearance(height, conflict):
    xy = np.array([(0, 0), (100, 0), (100, 100), (0, 100), (0, 2), (80, 2)])
    xyz = np.c_[xy, [0., 0., height, height, height, height]]
    passage = segment(0, 0, 1, xy)
    assert bool(passage_conflicts([(passage, xyz)], vertical_clearance=5.)) is conflict


def crossings(passages, width=None):
    chains = {s.segment_id: (np.array([[p.x, p.y, p.elevation] for p in s.points]),
                            np.array([p.width if width is None else width for p in s.points]))
              for s in passages}
    return _crossings(chains, SimpleNamespace(segments=passages))


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_local_junction_crossings_survive_arm_subdivision(split, reverse):
    a = segment(0, 0, 2, [(-100, 0), (0, 0)])
    c = segment(2, 2, 3, [(0, 0), (100, 0)])
    arm = ([segment(1, 1, 4, [(-100, -100), (-6, 3)]),
            segment(3, 4, 2, [(-6, 3), (0, 0)])] if split else
           [segment(1, 1, 2, [(-100, -100), (-6, 3), (0, 0)])])
    passages = [a, c, *arm]
    if reverse:
        passages.reverse()
    assert not crossings(passages)
    # Section screening can supply narrower profiles than the network widths.
    assert crossings(passages, width=.5)


def test_remote_crossing_near_junction_is_not_a_local_connection():
    passages = [segment(0, 0, 2, [(-100, 0), (0, 0)]),
                segment(1, 1, 2, [(-100, -100), (100, -100), (100, 100),
                                  (-15, 100), (-15, -20), (0, 0)]),
                segment(2, 2, 3, [(0, 0), (100, 0)])]
    assert any(np.allclose(hit["xy_m"], [-15, 0]) for hit in crossings(passages))


def test_accepted_route_subdivision_retains_repaired_polyline():
    repaired = segment(0, 0, 3, [(0, 0), (10, 4), (20, 7), (30, 0)])
    repaired = replace(repaired, metadata=dict(repaired.metadata,
        regional_cell_path=[0, 1, 2, 3], regional_cell_fractions=[0., .3, .6, 1.]))
    routes = AcceptedRoutes(SimpleNamespace(segments=[repaired]))
    planner = SimpleNamespace(xy=np.array([[0, 0], [10, 0], [20, 0], [30, 0]]))
    parts = [routes.assemble(planner, cells, 6.)[0] for cells in ([0, 1], [1, 2, 3])]
    assembled = np.r_[parts[0], parts[1][1:]]
    # Every repaired vertex remains on the exact original polyline; extra
    # interpolated vertices only subdivide it, never restore the grid path.
    old = np.array([[p.x, p.y] for p in repaired.points])
    assert all(np.min(np.linalg.norm(assembled[:, :2] - p, axis=1)) < 1e-9 for p in old)
    assert np.linalg.norm(np.diff(assembled[:, :2], axis=0), axis=1).sum() == pytest.approx(repaired.total_length)
    assert routes.position(planner, 1)[1] > 0


def test_new_attachment_does_not_refit_accepted_short_pieces():
    # Two cuts near an old bend used to refit its short middle piece into a
    # tighter turn, even though the accepted receiving curve had not changed.
    old = segment(0, 0, 1, [(0, 0), (4, 1), (8, 3), (12, 6)])
    following = segment(1, 1, 2, [(12, 6), (18, 12), (24, 17)])
    branch = segment(2, 1, 3, [(12, 6), (15, 10), (20, 30), (40, 40)])
    host = SimpleNamespace(sample=lambda x, y: SimpleNamespace(
        elevation=100., slope_degrees=0., cover_thickness=30.,
        roof_competence=1., growth_cost=.5))
    result = CaveNetworkGenerator._smooth_graph_routes(
        host, [old, following, branch], directed=True, preserved_segments={0, 1}
    )
    assert result[0] is old and result[1] is following
    assert result[2].points[0].x == 12 and result[2].points[0].y == 6
    assert np.diff([p.arc_length for p in result[2].points]).min() > 0


def test_repeated_canonical_cuts_keep_strict_arc_ordering():
    old = segment(0, 0, 3, [(0, 0), (10, 4), (20, 7), (30, 0)])
    metadata = dict(old.metadata, regional_cell_path=[0, 1, 2, 3],
                    regional_cell_fractions=[0., .3, .6, 1.])
    old = replace(old, metadata=metadata)
    planner = SimpleNamespace(xy=np.array([[0, 0], [10, 0], [20, 0], [30, 0]]))
    for _ in range(15):
        cache = AcceptedRoutes(SimpleNamespace(segments=[old]))
        points, fractions, preserved = cache.assemble(planner, [0, 1, 2, 3], 6.)
        assert preserved
        assert np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).min() > 1e-9
        old = replace(segment(0, 0, 3, points[:, :2]),
                      metadata=dict(metadata, regional_cell_fractions=fractions))


@pytest.mark.parametrize("samples", [2, 20, 80])
def test_close_passage_checks_use_local_width_not_segment_mean(samples):
    a = segment(0, 0, 1, np.column_stack((np.linspace(0, 100, samples), np.zeros(samples))))
    b = segment(1, 2, 3, [(0, 6), (10, 6), (100, 100)])
    a = replace(a, points=tuple(replace(p, width=2 + .08 * p.x) for p in a.points))
    # The close upstream part is narrow; the wide part is far away. Averaging
    # widths over the entire segment used to report an unrelated false clash.
    b = replace(b, points=tuple(replace(p, width=2.) for p in b.points))
    assert separation([a, b])

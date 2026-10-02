"""Constrained local search succeeds, fails safely and preserves accepted context."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import host_semantic_hash
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_gallery_growth import _points
from plume_advanced.stages.network_local_repair import repair_local_bends, search_curve
from plume_advanced.stages.network_quality import _turn_metrics, assess_network

ROOT = Path(__file__).resolve().parents[1]


def test_heading_search_routes_around_obstacle_reproducibly():
    def valid(xy):
        return (np.all(np.linalg.norm(xy - [15, 0], axis=1) > 4)
                and np.max(abs(xy[:, 1])) < 12 and np.min(xy[:, 0]) >= 0
                and np.max(xy[:, 0]) <= 30.1)
    a, b = [search_curve([0, 0], [30, 0], [1, 0], [1, 0], 5, valid,
                         lambda xy: 0) for _ in range(2)]
    assert a.reason == "local_heading_search" and 0 < a.expanded <= 1200
    assert b.expanded == a.expanded and np.array_equal(a.points, b.points)
    assert np.array_equal(a.points[[0, -1]], [[0, 0], [30, 0]])
    assert valid(a.points)
    angles, radii = _turn_metrics(a.points, np.ones(len(a.points)))
    assert min(radii) > .999 * 5 and max(angles) < 10
    assert abs((a.points[1]-a.points[0])[1]) < .02
    assert abs((a.points[-1]-a.points[-2])[1]) < .02


def test_blocked_search_has_explicit_reproducible_work_bound():
    def valid(xy):
        return bool(np.all(xy[:, 0] < 10))
    a, b = [search_curve([0, 0], [30, 0], [1, 0], [1, 0], 5, valid,
                         lambda xy: 0, maximum_expansions=20) for _ in range(2)]
    assert a.points is b.points is None
    assert a.reason == b.reason == "work_limit"
    assert a.expanded == b.expanded == 20


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_invalid_search_geometry_is_rejected(radius):
    with pytest.raises(ValueError, match="finite XY"):
        search_curve([0, 0], [30, 0], [1, 0], [1, 0], radius, lambda xy: True, lambda xy: 0)


def test_local_bend_repair_preserves_nodes_host_and_unaffected_geometry():
    c = load_project_config(ROOT / "config/regional-network.toml")
    host = HostFieldGenerator(c.host_field).generate()
    network = CaveNetworkGenerator(c.network).generate(host)
    cfg = replace(network.config, regional=replace(network.config.regional, branch_growth="front"))
    network = replace(network, config=cfg)
    # An isolated, sharp lateral perturbation in an already accepted route.
    segment = max(network.segments, key=lambda s: s.total_length)
    raw = np.array([[p.x, p.y] for p in segment.points])
    center = len(raw)//2
    direction = raw[center+1] - raw[center-1]
    raw[center] += 3 * np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    changed = replace(segment, points=_points(host, raw, np.array([p.width for p in segment.points])))
    broken = replace(network, segments=tuple(changed if s.segment_id == segment.segment_id else s for s in network.segments))
    report = assess_network(broken, host)
    failures = [c for c in report["checks"] if not c["passed"]]
    assert any(c["name"] == "bend_radius_relative_to_width" for c in failures)
    before = host_semantic_hash(host)
    a, b = [repair_local_bends(CaveNetworkGenerator(cfg), host, broken, 0, failures) for _ in range(2)]
    assert a.nodes == broken.nodes
    assert a.segments == b.segments
    assert host_semantic_hash(host) == before
    assert assess_network(a, host)["accepted"], a.backend_provenance["local_repair_history"]
    assert any(row["accepted"] for row in a.backend_provenance["local_repair_history"])
    for old, new in zip(broken.segments, a.segments):
        if old.segment_id != segment.segment_id:
            assert [(p.x, p.y) for p in old.points] == [(p.x, p.y) for p in new.points]


def test_zero_search_budget_still_allows_direct_fit_but_no_state_expansion():
    blocked = search_curve([0, 0], [30, 0], [1, 0], [1, 0], 5,
                           lambda xy: bool(np.all(xy[:, 0] < 10)), lambda xy: 0,
                           maximum_expansions=0)
    assert blocked.reason == "work_limit" and blocked.expanded == 0 and blocked.points is None
    direct = search_curve([0, 0], [30, 0], [1, 0], [1, 0], 5,
                          lambda xy: True, lambda xy: 0, maximum_expansions=0)
    assert direct.reason == "anchored_curve" and direct.expanded == 0

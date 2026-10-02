"""Real topology, flow and geometry checks for detail-stage passage encounters."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import network_semantic_hash
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkGenerator,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.network_acceptance import rebuild_network_geometry
from plume_advanced.stages.network_detail_junctions import (
    connect_overlaps,
    original_routes_preserved,
)
from plume_advanced.stages.network_layers import segment_xyz
from plume_advanced.stages.network_quality import _crossings, assess_network

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def crossing():
    cfg = load_project_config(ROOT / "config/regional-network.toml")
    config = replace(cfg.network, target_route_length_m=180.,
                     systems=replace(cfg.network.systems, count=2, require_merge=False, require_split=False))
    host = HostFieldGenerator(cfg.host_field).generate()
    host = replace(host, elevation=np.ones_like(host.elevation)*100.,
                   slope_degrees=np.zeros_like(host.slope_degrees),
                   growth_cost=np.ones_like(host.growth_cost),
                   cover_thickness=np.ones_like(host.cover_thickness)*50,
                   emplacement_thickness=np.ones_like(host.emplacement_thickness)*100.)
    y = np.linspace(-90, 90, 181)
    paths = [np.column_stack((-.35*y, y)), np.column_stack((.35*y, y))]
    nodes, segments = [], []
    for i, xy in enumerate(paths):
        for j, v in enumerate((xy[0], xy[-1])):
            nodes.append(CaveNode(2*i+j, *map(float, v), float(v[1]+90), float(v[0]), "entry" if j == 0 else "exit"))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        points = tuple(CavePoint(k, float(v[0]), float(v[1]), 100., 0., 50., 1., 1., float(s), 4., 1., 1400., 0.)
                       for k, (v, s) in enumerate(zip(xy, arc)))
        segments.append(CaveSegment(i, 2*i, 2*i+1, "backbone", 0, points,
            dict(source_system_id=i, regional_capacity=1., regional_start_potential=200.,
                 regional_end_potential=0., regional_route_type="source_route",
                 regional_start_layer=0, regional_end_layer=0)))
    network = CaveNetwork(config, tuple(nodes), tuple(segments), (), np.zeros_like(host.growth_cost, dtype=bool),
                          np.zeros_like(host.growth_cost), (0, 1), (), (), ())
    generator = CaveNetworkGenerator(config)
    network = rebuild_network_geometry(generator, host, network, list(network.segments))
    return generator, host, network


def test_crossing_becomes_one_shared_node_and_four_conserved_arms(crossing):
    generator, host, network = crossing
    original = network_semantic_hash(network)
    assert not assess_network(network, host)["accepted"]
    joined, audit = connect_overlaps(generator, host, network, {0})
    assert len(audit["junctions"]) == 1
    assert len(joined.nodes) == 5 and len(joined.segments) == 4
    node = joined.nodes[-1]
    assert sum(s.start_node_id == node.node_id for s in joined.segments) == 2
    assert sum(s.end_node_id == node.node_id for s in joined.segments) == 2
    assert joined.max_flow_conservation_error() < 1e-10
    assert original_routes_preserved(network, joined)
    checks = assess_network(joined, host)
    assert checks["accepted"], [c for c in checks["checks"] if not c["passed"]]
    for s in joined.segments:
        if s.start_node_id == node.node_id:
            assert s.metadata["contributing_system_ids"] == [0, 1]
            assert "source_system_id" not in s.metadata
    assert network_semantic_hash(network) == original
    replay, repeated = connect_overlaps(generator, host, network, {0})
    assert network_semantic_hash(replay) == network_semantic_hash(joined)
    assert repeated == audit
    unchanged, second = connect_overlaps(generator, host, joined, {s.segment_id for s in joined.segments})
    assert not second["junctions"] and unchanged is joined


@pytest.mark.parametrize("extended", [False, True])
def test_touching_envelopes_connect_without_centerline_crossing(crossing, extended):
    generator, host, network = crossing
    # A shallow approach touches the other passage and then diverges again.
    y = np.linspace(-90, 90, 361)
    x = 2+(.003*y*y if extended else 18*(1-np.exp(-(y/15)**2)))
    xy = np.column_stack((x, y))
    a = network.segments[0]
    a = replace(a, points=tuple(replace(p, x=0.) for p in a.points))
    a = replace(a, points=tuple(replace(p, arc_length=p.y+90) for p in a.points))
    b = network.segments[1]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    b = replace(b, points=tuple(replace(b.points[0], index=i, x=float(v[0]), y=float(v[1]), arc_length=float(s))
                                for i, (v, s) in enumerate(zip(xy, arc))))
    nodes = tuple(replace(n, x=s.points[k].x, y=s.points[k].y)
                  for s in (a, b) for n, k in ((network.nodes[s.start_node_id], 0), (network.nodes[s.end_node_id], -1)))
    network = rebuild_network_geometry(generator, host, replace(network, nodes=nodes), [a, b])
    chains = {s.segment_id: (np.array([(p.x, p.y) for p in s.points]), np.array([p.width for p in s.points]))
              for s in network.segments}
    assert not _crossings(chains, network)
    joined, audit = connect_overlaps(generator, host, network, {1})
    assert audit["junctions"]
    assert audit["junctions"][0]["contact_distance_m"] == pytest.approx(2., abs=.03)
    assert original_routes_preserved(network, joined)
    assert joined.max_flow_conservation_error() < 1e-10
    # An extended shared footprint is not automatically excused by a node.
    report = assess_network(joined, host)
    if extended:
        assert any(c["name"] == "regional_passage_clearance" and not c["passed"] for c in report["checks"])
    else:
        assert report["accepted"], [c for c in report["checks"] if not c["passed"]]


@pytest.mark.parametrize("case", ["separate_layers", "different_elevations", "opposed", "near_endpoint", "potential"])
def test_invalid_connections_leave_input_untouched(crossing, case):
    generator, host, network = crossing
    a, b = network.segments
    if case in {"separate_layers", "different_elevations"}:
        config = replace(network.config, layers=replace(network.config.layers, enabled=True))
        if case == "separate_layers":
            b = replace(b, metadata=dict(b.metadata, regional_start_layer=1, regional_end_layer=1))
        else:
            b = replace(b, metadata=dict(b.metadata, network_detail_vertical_offsets_m=[-2.]*len(b.points)))
        network = replace(network, config=config)
        generator = CaveNetworkGenerator(config)
    elif case == "opposed":
        b = replace(b, points=tuple(replace(p, arc_length=b.total_length-p.arc_length) for p in reversed(b.points)),
                    start_node_id=b.end_node_id, end_node_id=b.start_node_id)
    elif case == "near_endpoint":
        b = replace(b, points=tuple(replace(p, arc_length=p.arc_length-b.points[88].arc_length) for p in b.points[88:]))
    else:
        b = replace(b, metadata=dict(b.metadata, regional_start_potential=400., regional_end_potential=300.))
    network = replace(network, segments=(a, b))
    joined, audit = connect_overlaps(generator, host, network, {0})
    assert joined is network and not audit["junctions"]
    assert audit["rejected"]


def test_layer_junction_preserves_actual_z_and_host_elevations(crossing):
    generator, host, network = crossing
    config = replace(network.config, layers=replace(network.config.layers, enabled=True))
    network = replace(network, config=config)
    generator = CaveNetworkGenerator(config)
    joined, audit = connect_overlaps(generator, host, network, {0})
    assert len(audit["junctions"]) == 1
    node = joined.nodes[-1].node_id
    elevations = [segment_xyz(s, config.layers)[0 if s.start_node_id == node else -1, 2]
                  for s in joined.segments]
    assert np.ptp(elevations) < 1e-10
    assert all(p.elevation == 100. for s in joined.segments for p in s.points)
    checks = assess_network(joined, host)["checks"]
    assert all(c["passed"] for c in checks if c["name"] in {
        "layer_junction_elevation_continuity", "layer_host_thickness", "layer_passage_separation"})


def test_junction_budget_is_explicit(crossing):
    generator, host, network = crossing
    joined, audit = connect_overlaps(generator, host, network, {0}, maximum_junctions=0)
    assert joined is network and not audit["junctions"]
    assert audit["limit"] == 0


def test_cycle_creating_junction_is_not_committed(crossing):
    generator, host, network = crossing
    a, b = network.segments
    # The second route is already downstream of the first. Joining interiors
    # would create a directed return to the same node; never erase this guard.
    b = replace(b, start_node_id=a.end_node_id)
    network = replace(network, segments=(a, b))
    joined, audit = connect_overlaps(generator, host, network, {0})
    assert joined is network and not audit["junctions"]
    assert any("cycle" in item["reason"] for item in audit["rejected"])


def test_invalid_route_deletion_is_not_a_merge(crossing):
    _, _, network = crossing
    assert not original_routes_preserved(network, replace(network, segments=network.segments[:1]))


def test_detail_stage_commits_encounter_and_replays(crossing, monkeypatch):
    import plume_advanced.stages.network_detail as detail

    generator, host, network = crossing
    a, b = network.segments
    y = np.linspace(-90, 90, 361)
    new_x = 2+18*(1-np.exp(-(y/15)**2))
    old_x = new_x+8*np.exp(-(y/20)**2)
    old_x[[0, -1]] = new_x[[0, -1]]

    def points(s, x):
        arc = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(y))) ]
        return tuple(replace(s.points[0], index=i, x=float(x[i]), y=float(y[i]), arc_length=float(arc[i]))
                     for i in range(len(y)))

    a = replace(a, points=points(a, np.zeros_like(y)))
    changed = replace(b, points=points(b, new_x))
    b = replace(b, points=points(b, old_x))
    nodes = tuple(replace(n, x=s.points[k].x, y=s.points[k].y)
                  for s in (a, b) for n, k in ((network.nodes[s.start_node_id], 0), (network.nodes[s.end_node_id], -1)))
    # The coarse baseline must already be one connected network. Two separate
    # parallel passages are no longer accepted simply because their individual
    # geometries pass; close them with a downstream confluence before testing
    # the additional detail-stage encounter in the middle.
    nodes = tuple(replace(n, kind="junction") if n.kind == "exit" else n for n in nodes)
    nodes += (CaveNode(4, 10., 120., 210., 10., "junction"),
              CaveNode(5, 10., 150., 240., 10., "exit"))
    tails = []
    for sid, parent in enumerate((a, b), 2):
        t = np.linspace(0., 1., 161)
        tail_x = parent.points[-1].x + (10. - parent.points[-1].x) * t
        tail_y = 90. + 30.*t
        arc = np.r_[0., np.cumsum(np.hypot(np.diff(tail_x), np.diff(tail_y)))]
        tail_points = tuple(replace(parent.points[-1], index=i, x=float(x), y=float(y), arc_length=float(s))
                            for i, (x, y, s) in enumerate(zip(tail_x, tail_y, arc)))
        tails.append(replace(parent, segment_id=sid, start_node_id=parent.end_node_id,
                             end_node_id=4, points=tail_points,
                             metadata=dict(regional_capacity=1., regional_start_potential=0.,
                                           regional_end_potential=-30., regional_route_type="source_route",
                                           regional_start_layer=0, regional_end_layer=0)))
    stem_points = tuple(replace(a.points[-1], index=i, x=10., y=120.+i, arc_length=float(i))
                        for i in range(31))
    tails.append(replace(tails[0], segment_id=4, start_node_id=4, end_node_id=5,
                         points=stem_points, metadata=dict(tails[0].metadata,
                             regional_start_potential=-30., regional_end_potential=-60.)))
    config = replace(network.config, detail=detail.NetworkDetailConfig(enabled=True))
    generator = CaveNetworkGenerator(config)
    network = rebuild_network_geometry(generator, host, replace(network, nodes=nodes, config=config), [a, b, *tails])
    assessment = assess_network(network, host)
    assert assessment["accepted"], [c for c in assessment["checks"] if not c["passed"]]
    monkeypatch.setattr(detail, "_proposal", lambda s, c, h: (True if s.segment_id == 1 else None, {}))
    monkeypatch.setattr(detail, "_materialize", lambda *args: (changed, {}))
    joined = detail.refine_network(generator, host, network)
    report = joined.backend_provenance["detail"]
    assert report["status"] == "refined" and report["added_junctions"] == 1
    assert report["original_routes_preserved"] and not report["graph_preserved"]
    assert assess_network(joined, host)["accepted"]
    assert network_semantic_hash(detail.refine_network(generator, host, network)) == network_semantic_hash(joined)

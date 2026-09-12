"""Interactions must be graph connections with conserved, traceable flow."""

from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNetworkGenerator,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.network_quality import NetworkQualityConfig, assess_network
from plume_advanced.stages.network_systems import (
    NetworkSystemsConfig,
    assess_systems,
    plan_interactions,
    preferred_tracks,
)


def merge_split_network():
    config = CaveNetworkConfig(
        random_seed=42,
        base_passage_radius=2.5,
        minimum_passage_radius=1,
        maximum_passage_radius=5,
        target_route_length_m=500,
        source_flux=3,
        systems=NetworkSystemsConfig(
            count=2, minimum_shared_length_widths=8, minimum_independent_length_widths=8
        ),
    )
    nodes = [
        CaveNode(i, x, y, x, y, kind)
        for i, (x, y, kind) in enumerate(
            [
                (0, -30, "entry"),
                (0, 30, "entry"),
                (100, 0, "junction"),
                (300, 0, "junction"),
                (500, -30, "exit"),
                (500, 30, "exit"),
            ]
        )
    ]
    segments = []
    for sid, (start, end, ids, width) in enumerate(
        [
            (0, 2, [0], 5),
            (1, 2, [1], 5),
            (2, 3, [0, 1], 5),
            (3, 4, [0], 4),
            (3, 5, [1], 8),
        ]
    ):
        xy = np.linspace((nodes[start].x, nodes[start].y), (nodes[end].x, nodes[end].y), 25)
        arcs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        points = tuple(
            CavePoint(i, *p, 100 - 0.01 * arc, 1, 30, 0.8, 0.2, float(arc), width)
            for i, (p, arc) in enumerate(zip(xy, arcs))
        )
        metadata = {"system_ids": ids}
        if nodes[start].kind == "entry":
            metadata["source_system_id"] = ids[0]
        segments.append(CaveSegment(sid, start, end, "backbone", 0, points, metadata))
    generator = CaveNetworkGenerator(config)
    segments = generator._assign_conserved_flow(nodes, segments)
    route = generator._dominant_route(nodes, segments)
    return CaveNetwork(
        config,
        tuple(nodes),
        tuple(segments),
        (),
        np.ones((2, 2), bool),
        np.ones((2, 2)),
        route,
        (),
        (),
        (),
    )


def system_failures(network):
    failures = []
    assess_systems(
        network, lambda name, passed, *args: failures.append(name) if not passed else None
    )
    return set(failures)


def test_merge_combines_supply_once_and_split_divides_it():
    network = merge_split_network()
    segments = network.segments
    assert sum(s.mean_flux for s in segments[:2]) == pytest.approx(6)
    assert segments[2].mean_flux == pytest.approx(6)
    assert segments[3].mean_flux == pytest.approx(1.2)
    assert segments[4].mean_flux == pytest.approx(4.8)
    assert network.max_flow_conservation_error() < 1e-12
    assert not system_failures(network)
    CaveNetworkGenerator(network.config)._validate_generated_graph(
        list(network.nodes), list(segments), network.dominant_route_node_ids
    )
    assert network.dominant_route_node_ids[-1] == 5


def test_shared_geometry_is_single_edge_and_both_split_arms_inherit_sources():
    network = merge_split_network()
    assert sum(s.start_node_id == 2 and s.end_node_id == 3 for s in network.segments) == 1
    assert [s.metadata["contributing_system_ids"] for s in network.segments] == [
        [0],
        [1],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    assert [s.metadata["system_ids"] for s in network.segments[3:]] == [[0], [1]]


def test_thermal_state_is_flux_weighted_at_confluence():
    segments = merge_split_network().segments
    total = sum(s.mean_flux for s in segments[:2])
    expected_temperature = (
        sum(s.mean_flux * s.points[-1].temperature_k for s in segments[:2]) / total
    )
    expected_age = sum(s.mean_flux * s.points[-1].age_s for s in segments[:2]) / total
    assert segments[2].points[0].temperature_k == pytest.approx(expected_temperature)
    assert segments[2].points[0].age_s == pytest.approx(expected_age)
    for s in segments[3:]:
        assert s.points[0].temperature_k == pytest.approx(segments[2].points[-1].temperature_k)


@pytest.mark.parametrize("ids", [None, "0", [None], [0, 0], [0, "1"], [99]])
def test_malformed_membership_is_reported(ids):
    network = merge_split_network()
    segment = replace(
        network.segments[2], metadata=dict(network.segments[2].metadata, system_ids=ids)
    )
    network = replace(network, segments=network.segments[:2] + (segment,) + network.segments[3:])
    assert "system_identity_at_connections" in system_failures(network)


@pytest.mark.parametrize(
    "defect, expected",
    [
        ("identity", "system_identity_at_connections"),
        ("lineage", "system_source_lineage"),
        ("missing_source", "system_sources"),
        ("outflow", "system_total_discharge"),
        ("short_shared", "system_passage_persistence"),
    ],
)
def test_interaction_defects_fail_closed(defect, expected):
    network = merge_split_network()
    segments = list(network.segments)
    sid = 2
    metadata = dict(segments[sid].metadata)
    if defect == "identity":
        metadata["system_ids"] = [0]
    elif defect == "lineage":
        metadata["contributing_system_ids"] = [0]
    elif defect == "missing_source":
        sid = 0
        metadata = dict(segments[0].metadata, source_system_id=1)
    if defect == "outflow":
        sid = 4
        metadata = dict(segments[sid].metadata)
        segments[sid] = replace(
            segments[sid], points=tuple(replace(p, flux=p.flux * 2) for p in segments[sid].points)
        )
    elif defect == "short_shared":
        segments[sid] = replace(
            segments[sid],
            points=tuple(replace(p, arc_length=p.arc_length * 0.01) for p in segments[sid].points),
        )
    segments[sid] = replace(segments[sid], metadata=metadata)
    network = replace(network, segments=tuple(segments))
    assert expected in system_failures(network)
    assert expected in {c["name"] for c in assess_network(network)["checks"] if not c["passed"]}


def test_planner_emits_shared_passages_without_coincident_duplicates():
    along = np.arange(0.0, 801.0)
    separation = np.interp(along, [0, 150, 200, 350, 450, 800], [30, 30, 0, 0, 30, 30])
    tracks = np.array([-separation / 2, separation / 2])
    controls = NetworkSystemsConfig(
        count=2, minimum_shared_length_widths=25, minimum_independent_length_widths=30
    )
    nodes, records = plan_interactions(along, tracks, 1, controls)
    assert len([n for n in nodes if n[2] == "entry"]) == 2
    assert len([n for n in nodes if n[2] == "exit"]) == 2
    assert len(records) == 5  # two in, one shared, two out
    assert sum(len(r[-1]) == 2 for r in records) == 1
    for station in range(len(along) - 1):
        active_ids = [
            i for _, _, first, last, group in records if first <= station < last for i in group
        ]
        assert Counter(active_ids) == {0: 1, 1: 1}
    assert (nodes, records) == plan_interactions(along, tracks, 1, controls)


def test_brief_release_spike_does_not_split_shared_front():
    along = np.arange(501.0)
    gap = np.interp(along, [0, 100, 150, 200, 201, 204, 205, 500], [30, 30, 0, 0, 20, 20, 0, 0])
    _, records = plan_interactions(
        along, np.array([-gap / 2, gap / 2]), 1, NetworkSystemsConfig(count=2)
    )
    assert len(records) == 3  # merge and stay shared


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"count": 9},
        {"count": True},
        {"count": 2.5},
        {"merge_distance_widths": 8},
        {"source_spacing_widths": 2},
        {"split_confirmation_widths": 0},
        {"lateral_variation_widths": float("nan")},
        {"minimum_shared_length_widths": 2},
        {"require_merge": 1},
    ],
)
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError, match="network.systems"):
        NetworkSystemsConfig(**kwargs)


def test_config_round_trip_and_unknown_keys(tmp_path):
    path = tmp_path / "systems.toml"
    path.write_text("schema_version = 4\n[network.systems]\ncount = 3\nrequire_split = false\n")
    config = load_project_config(path)
    assert config.network.systems.count == 3
    assert config.network.systems.require_split is False
    path.write_text("schema_version = 4\n[network.systems]\ncounts = 3\n")
    with pytest.raises(ValueError, match="counts"):
        load_project_config(path)


def test_single_system_still_uses_existing_generator():
    from plume_advanced.stages.network_systems import generate_system_network

    with patch(
        "plume_advanced.stages.network_systems.generate_system_network",
        wraps=generate_system_network,
    ) as multi:
        generator = CaveNetworkGenerator(
            CaveNetworkConfig(quality=NetworkQualityConfig(enabled=False))
        )
        with patch.object(
            generator, "_build_flow_geometry", side_effect=RuntimeError("legacy path")
        ):
            with pytest.raises(RuntimeError, match="legacy path"):
                generator.generate(None)
        multi.assert_not_called()


def test_tracks_repeat_change_with_seed_and_respond_to_host():
    host = HostFieldGenerator(
        HostFieldConfig(
            random_seed=4,
            grid=GridConfig(width=1200, height=1600, nx=40, ny=60),
            seed_point=(0, -650),
            flow_angle_degrees=90,
        )
    ).generate()
    config = CaveNetworkConfig(
        random_seed=12, target_route_length_m=1200, systems=NetworkSystemsConfig(count=3)
    )
    generator = CaveNetworkGenerator(config)
    geometry = generator._build_flow_geometry(host)
    along, first, _ = preferred_tracks(generator, host, geometry)
    _, repeat, _ = preferred_tracks(generator, host, geometry)
    np.testing.assert_array_equal(first, repeat)
    changed = CaveNetworkGenerator(replace(config, random_seed=13))
    assert not np.array_equal(first, preferred_tracks(changed, host, geometry)[1])
    altered = replace(host, growth_cost=np.broadcast_to(np.linspace(0, 1, 40), (60, 40)).copy())
    assert not np.array_equal(first, preferred_tracks(generator, altered, geometry)[1])
    assert np.all(np.diff(first, axis=0) >= 0)
    assert along[-1] == pytest.approx(1200)


def test_impossible_host_fails_instead_of_stacking_sources():
    host = HostFieldGenerator(
        HostFieldConfig(
            grid=GridConfig(width=50, height=1000, nx=10, ny=30),
            seed_point=(0, -400),
            flow_angle_degrees=90,
        )
    ).generate()
    generator = CaveNetworkGenerator(CaveNetworkConfig(systems=NetworkSystemsConfig(count=3)))
    with pytest.raises(ValueError, match="fit|narrow"):
        preferred_tracks(generator, host, generator._build_flow_geometry(host))

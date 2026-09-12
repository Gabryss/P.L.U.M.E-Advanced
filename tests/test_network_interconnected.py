"""Interconnected morphology must survive physical envelopes and replay."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_interconnected import (
    InterconnectionConfig,
    route_preferences,
    spatial_metrics,
)
from plume_advanced.stages.network_quality import NetworkQualityError, assess_network
from plume_advanced.stages.network_systems import NetworkSystemsConfig, plan_interactions
from plume_advanced.stages.network_topology import NetworkTopologyConfig
from plume_advanced.stages.section_field import SectionFieldGenerator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    cfg = load_project_config(ROOT / "config/earth_short_interconnected.toml")
    host = HostFieldGenerator(cfg.host_field).generate()
    network = CaveNetworkGenerator(cfg.network).generate(host, section_config=cfg.section_field)
    return cfg, host, network, SectionFieldGenerator(cfg.section_field).generate(network)


def test_profiles_preserve_parallel_routes_and_actual_interactions(example):
    _, h, n, s = example
    report = assess_network(n, h, s)
    assert report["accepted"], [c for c in report["checks"] if not c["passed"]]
    m = spatial_metrics(n, s)
    assert m["parallel_fraction"] >= 0.4
    assert m["minimum_window_parallel_fraction"] >= 0.2
    events = n.backend_provenance["interaction_events"]
    assert {e["kind"] for e in events} == {"merge", "split"}
    assert max(e["station_m"] for e in events) > 0.5 * m["downstream_extent_m"]
    assert n.max_flow_conservation_error() < 1e-8


def test_replay_preserves_host_network_sections_and_decisions(example):
    cfg, host, n, s = example
    before = host_semantic_hash(host)
    again = CaveNetworkGenerator(cfg.network).generate(host, section_config=cfg.section_field)
    sections = SectionFieldGenerator(cfg.section_field).generate(again)
    assert network_semantic_hash(again) == network_semantic_hash(n)
    assert section_semantic_hash(sections) == section_semantic_hash(s)
    assert again.quality_report == n.quality_report
    assert host_semantic_hash(host) == before


def test_duplicate_labels_and_inflated_profiles_cannot_fake_parallelism(example):
    _, host, n, s = example
    axis = np.array(n.backend_provenance["flow_direction"])
    cross = np.array([-axis[1], axis[0]])
    # Physically superpose the arterial routes but retain every source label.
    collapsed = replace(
        n,
        segments=tuple(
            replace(
                segment,
                points=tuple(
                    replace(
                        p,
                        x=float((np.array([p.x, p.y]) - (np.array([p.x, p.y]) @ cross) * cross)[0]),
                        y=float((np.array([p.x, p.y]) - (np.array([p.x, p.y]) @ cross) * cross)[1]),
                    )
                    for p in segment.points
                ),
            )
            for segment in n.segments
        ),
    )
    assert spatial_metrics(collapsed)["parallel_fraction"] == 0
    inflated = replace(
        s,
        segment_fields=tuple(
            replace(
                f,
                samples=tuple(
                    replace(
                        p,
                        profile_points=tuple((x * 100, z) for x, z in p.profile_points),
                        tube_width=p.tube_width * 100,
                    )
                    for p in f.samples
                ),
            )
            for f in s.segment_fields
        ),
    )
    m = spatial_metrics(n, inflated)
    assert m["parallel_fraction"] < 0.4
    report = assess_network(n, host, inflated)
    assert not report["accepted"]
    assert any(
        c["name"] == "interconnected_parallel_fraction" and not c["passed"]
        for c in report["checks"]
    )


def test_same_host_different_seed_changes_routes_and_cost_changes_route(example):
    cfg, host, _, _ = example
    gen = CaveNetworkGenerator(cfg.network)
    geo = gen._build_flow_geometry(host)
    _, baseline, _ = route_preferences(gen, host, geo)
    _, other, _ = route_preferences(
        CaveNetworkGenerator(replace(cfg.network, random_seed=99)), host, geo
    )
    assert not np.allclose(baseline, other)
    bias = np.broadcast_to(
        np.linspace(0, 4, host.growth_cost.shape[1]), host.growth_cost.shape
    ).copy()
    _, shifted, _ = route_preferences(gen, replace(host, growth_cost=bias), geo)
    assert np.max(np.abs(baseline - shifted)) > 0.1


def test_inlet_only_parallelism_cannot_satisfy_downstream_coverage(example):
    _, h, n, _ = example
    axis = np.array(n.backend_provenance["flow_direction"])
    cross = np.array([-axis[1], axis[0]])
    origin = np.array(h.config.seed_point)
    segments = []
    for s in n.segments:
        points = []
        for p in s.points:
            xy = np.array([p.x, p.y]) - origin
            a, c = xy @ axis, xy @ cross
            fade = np.clip((180 - a) / 80, 0, 1)
            xy = origin + a * axis + c * fade * cross
            points.append(replace(p, x=float(xy[0]), y=float(xy[1])))
        segments.append(replace(s, points=tuple(points)))
    metrics = spatial_metrics(replace(n, segments=tuple(segments)))
    assert metrics["minimum_window_parallel_fraction"] == 0
    assert metrics["longest_single_run_fraction"] > 0.4


def test_repairs_keep_shared_tangents_and_monotone_downstream_coordinates(example):
    cfg, h, n, _ = example
    from plume_advanced.stages.network_acceptance import repair_network

    repaired = repair_network(CaveNetworkGenerator(n.config), h, n, 1)
    report = assess_network(repaired, h)
    wanted = {"interconnected_junction_angles", "interconnected_host_uphill_grade"}
    checks = [c for c in report["checks"] if c["name"] in wanted]
    assert len(checks) == 2 and all(c["passed"] for c in checks), checks
    assert (
        spatial_metrics(repaired)["parallel_fraction"]
        >= cfg.network.interconnection.minimum_parallel_fraction
    )


def test_reference_supply_cannot_hide_a_starved_phase_at_a_split(example):
    _, host, network, _ = example
    from plume_advanced.stages.network_interconnected import assess_interconnected

    event = next(
        e for e in network.backend_provenance["interaction_events"] if e["kind"] == "split"
    )
    selected = next(s.segment_id for s in network.segments if s.start_node_id == event["node_id"])
    changed = replace(
        network,
        segments=tuple(
            replace(
                s, metadata=dict(s.metadata, phase_fluxes=[0.0, *s.metadata["phase_fluxes"][1:]])
            )
            if s.segment_id == selected
            else s
            for s in network.segments
        ),
    )
    failures = []
    assess_interconnected(
        changed,
        host,
        None,
        lambda name, passed, *args: failures.append(name) if not passed else None,
    )
    assert "interconnected_split_supply" in failures


def test_multiple_host_corridors_are_deterministic_and_single_is_opt_in(example):
    cfg, _, _, _ = example
    a = HostFieldGenerator(cfg.host_field).generate()
    b = HostFieldGenerator(cfg.host_field).generate()
    assert host_semantic_hash(a) == host_semantic_hash(b)
    single = replace(cfg.host_field, corridor_count=1)
    c = HostFieldGenerator(single).generate()
    d = HostFieldGenerator(
        replace(single, corridor_spacing=123, corridor_lateral_variation=77)
    ).generate()
    assert host_semantic_hash(c) == host_semantic_hash(d)
    assert not np.allclose(c.elevation, a.elevation)
    assert not np.allclose(c.growth_cost, a.growth_cost)


def test_impossible_host_fails_closed_without_modification(example):
    cfg, host, _, _ = example
    gen = CaveNetworkGenerator(
        replace(cfg.network, quality=replace(cfg.network.quality, max_attempts=1, repair_passes=0))
    )
    geo = gen._build_flow_geometry(host)
    x, y = np.meshgrid(host.x_coords, host.y_coords)
    uphill = replace(host, elevation=x * geo.flow_x + y * geo.flow_y)
    before = uphill.elevation.copy()
    with pytest.raises(NetworkQualityError, match="No bounded forward route"):
        gen.generate(uphill, section_config=cfg.section_field)
    np.testing.assert_array_equal(uphill.elevation, before)


def test_event_spacing_is_local_to_the_participating_fronts():
    along = np.arange(201.0)
    tracks = np.broadcast_to(np.array([0.0, 2.0, 100.0, 102.0])[:, None], (4, len(along))).copy()
    controls = NetworkSystemsConfig(
        count=4,
        source_spacing_widths=10,
        merge_distance_widths=3,
        minimum_shared_length_widths=3,
        minimum_independent_length_widths=3,
        interaction_spacing_widths=30,
    )
    global_events, local_events = [], []
    plan_interactions(along, tracks, 1, controls, events=global_events)
    plan_interactions(along, tracks, 1, controls, events=local_events, local_spacing=True)
    assert global_events[1]["station_m"] - global_events[0]["station_m"] >= 30
    assert local_events[1]["station_m"] - local_events[0]["station_m"] < 30
    assert set(local_events[0]["after"][0]).isdisjoint(local_events[1]["after"][0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_parallel_fraction": 1},
        {"interaction_window_m": float("nan")},
        {"minimum_parallel_run_widths": 0},
        {"maximum_junction_angle_degrees": 90},
    ],
)
def test_invalid_interconnection_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        InterconnectionConfig(**kwargs)


def test_interconnected_requires_growth_not_the_layout_generator():
    with pytest.raises(ValueError, match="requires independent_growth"):
        NetworkTopologyConfig(style="interconnected")

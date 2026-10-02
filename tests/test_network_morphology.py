"""Localized morphology stays bounded, replayable and physically screened."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_payload,
    network_semantic_hash,
)
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_layer_routing import LayeredRegionalPlanner
from plume_advanced.stages.network_layers import NetworkLayersConfig, segment_depths, segment_xyz
from plume_advanced.stages.network_morphology import (
    branch_weights,
    branch_zones,
    morphology_metrics,
)
from plume_advanced.stages.network_quality import assess_network
from plume_advanced.stages.network_regional_routing import RegionalGrowthConfig

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_zone_scales_remain_bounded_and_probabilities_finite():
    regional = RegionalGrowthConfig(
        branch_localization=1, correlation_length_m=1e-9, maximum_branches=3
    )
    planner = SimpleNamespace(
        config=SimpleNamespace(regional=regional, random_seed=0, layers=NetworkLayersConfig()),
        geometry=SimpleNamespace(along_extent=800, seed_x=0, seed_y=0, flow_x=1, flow_y=0),
        xy=np.array([[1e9, 0], [2e9, 0]]),
        layer_ids=np.zeros(2, dtype=int),
    )
    zones = branch_zones(planner)
    assert len(zones) <= regional.maximum_branches
    p = branch_weights(planner, [0, 1], zones)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "name", ["branch_localization", "hierarchy_strength", "blind_branch_fraction"]
)
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_morphology_controls(name, value):
    with pytest.raises(ValueError, match="network.regional"):
        RegionalGrowthConfig(**{name: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("minimum_extent_fraction", 0.49),
        ("minimum_extent_fraction", 1.01),
        ("minimum_extent_fraction", float("nan")),
        ("spacing_variation", -0.1),
        ("spacing_variation", 1.1),
        ("spacing_variation", True),
    ],
)
def test_invalid_local_layers(name, value):
    with pytest.raises(ValueError, match="network.layers"):
        NetworkLayersConfig(**{name: value})


@pytest.fixture(scope="module")
def settings():
    c = load_project_config(ROOT / "config/varied-network.toml", seed_override=0)
    cfg = replace(c.network, regional=replace(c.network.regional, branches_per_km=8))
    return cfg, HostFieldGenerator(c.host_field).generate()


@pytest.fixture(scope="module")
def example(settings):
    cfg, host = settings
    return CaveNetworkGenerator(cfg).generate(host)


def test_zones_are_local_repeatable_and_layer_specific(settings):
    cfg, host = settings
    planner = LayeredRegionalPlanner(CaveNetworkGenerator(cfg), host)
    zones = branch_zones(planner)
    assert zones == branch_zones(planner)
    assert {z["layer"] for z in zones} == set(range(cfg.layers.count))
    candidates = np.arange(len(planner.xy))
    weights = branch_weights(planner, candidates, zones)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert weights.sum() == pytest.approx(1)
    assert weights.max() > 5 * weights.min()
    assert branch_weights(planner, candidates, []) is None
    assert len(set(planner.extents)) == cfg.layers.count
    assert planner.extents[-1] == pytest.approx(planner.geometry.along_extent)
    assert np.ptp(np.diff(planner.depths)) > 0.1
    assert (
        np.min(np.diff(planner.depths)) >= cfg.layers.passage_height_m + cfg.layers.minimum_rock_m
    )
    same = planner.layer_ids[planner.rows] == planner.layer_ids[planner.cols]
    along = (planner.xy - [planner.geometry.seed_x, planner.geometry.seed_y]) @ [
        planner.geometry.flow_x,
        planner.geometry.flow_y,
    ]
    assert np.all(
        along[planner.cols[same]]
        <= planner.extents[planner.layer_ids[planner.cols[same]]] + planner.step
    )


def test_disabling_layers_ignores_all_optional_layer_parameters(settings):
    cfg, host = settings
    from plume_advanced.stages.network_regional_routing import RegionalPlanner

    a = RegionalPlanner(
        CaveNetworkGenerator(replace(cfg, layers=replace(cfg.layers, enabled=False))), host
    )
    b = RegionalPlanner(CaveNetworkGenerator(replace(cfg, layers=NetworkLayersConfig())), host)
    assert np.array_equal(a.rows, b.rows)
    assert np.array_equal(a.weights, b.weights)
    assert a.sources == b.sources and a.goals == b.goals


def test_generated_widths_flows_and_semantics_pass_inspection(settings, example):
    cfg, host = settings
    report = assess_network(example, host)
    assert report["accepted"], [c for c in report["checks"] if not c["passed"]]
    assert example.max_flow_conservation_error() < 1e-8
    widths = np.array([s.mean_width for s in example.segments])
    assert np.ptp(widths) > 1
    roles = {s.metadata["regional_route_type"] for s in example.segments}
    assert {"bypass", "descending_ramp", "retained_trunk"} <= roles
    # Flow is a conserved formation-allocation proxy, including blind sinks.
    outgoing = {s.start_node_id for s in example.segments}
    terminal_flux = sum(s.mean_flux for s in example.segments if s.end_node_id not in outgoing)
    assert terminal_flux == pytest.approx(cfg.source_flux * cfg.systems.count)
    events = example.backend_provenance["growth_events"]
    assert (
        len(events)
        <= cfg.regional.attempts_per_branch * example.backend_provenance["requested_branches"]
    )


def test_reports_use_actual_layer_elevations(example):
    payload = network_payload(example)
    expected_uphill = 0
    for s, row in zip(sorted(example.segments, key=lambda x: x.segment_id), payload["segments"]):
        xyz = segment_xyz(s, example.config.layers)
        assert row["grade"] == pytest.approx((xyz[-1, 2] - xyz[0, 2]) / s.total_length)
        lengths = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        distance = lengths[np.diff(xyz[:, 2]) > 0].sum()
        expected_uphill += distance
        assert s.metadata["uphill_distance_m"] == pytest.approx(distance)
    assert example.summary()["uphill_distance_m"] == pytest.approx(expected_uphill)
    assert payload["morphology"]["loop_rank"] == example.summary()["loop_count"]
    assert sum(payload["morphology"]["junction_counts"]) == len(
        [
            n
            for n in example.nodes
            if sum(s.start_node_id == n.node_id for s in example.segments) > 1
            or sum(s.end_node_id == n.node_id for s in example.segments) > 1
        ]
    )
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("depths", [[4.5, 5, 6], [4.5, float("nan"), 28], [4.5, 18], [4.5, {}, 28]])
def test_invalid_depth_layout_is_rejected(settings, example, depths):
    _, host = settings
    s = example.segments[0]
    broken = replace(s, metadata=dict(s.metadata, regional_layer_depths_m=depths))
    assert np.isnan(segment_depths(broken, example.config.layers)).all()
    n = replace(example, segments=(broken, *example.segments[1:]))
    assert not assess_network(n, host)["accepted"]


def test_false_connection_labels_fail_inspection(settings, example):
    _, host = settings
    s = next(s for s in example.segments if s.metadata["regional_route_type"] == "descending_ramp")
    n = replace(
        example,
        segments=tuple(
            replace(t, metadata=dict(t.metadata, regional_route_type="bypass")) if t == s else t
            for t in example.segments
        ),
    )
    checks = {c["name"]: c["passed"] for c in assess_network(n, host)["checks"]}
    assert not checks["regional_route_semantics"]


def test_repair_cannot_qualify_a_candidate_that_skipped_branch_growth(settings, example):
    from plume_advanced.stages.network_acceptance import repair_network

    _, host = settings
    incomplete = replace(example, backend_provenance=dict(
        example.backend_provenance, regional_growth_completed=False,
    ))
    worker = CaveNetworkGenerator(example.config)
    repaired = repair_network(worker, host, incomplete, 0)
    for n in (incomplete, repaired):
        report = assess_network(n, host)
        assert not report["accepted"]
        assert not next(c["passed"] for c in report["checks"] if c["name"] == "regional_growth_completed")


def test_width_hierarchy_and_blind_taper_do_not_accumulate_under_flow_reassignment(
    settings, example
):
    cfg, _ = settings
    generator = CaveNetworkGenerator(replace(cfg, random_seed=example.config.random_seed))
    a = generator._assign_conserved_flow(list(example.nodes), list(example.segments))
    b = generator._assign_conserved_flow(list(example.nodes), a)
    assert [s.mean_flux for s in a] == [s.mean_flux for s in b]
    assert [[p.width for p in s.points] for s in a] == [[p.width for p in s.points] for s in b]
    s = replace(
        example.segments[0],
        metadata=dict(example.segments[0].metadata, regional_route_type="blind_branch"),
    )
    from plume_advanced.stages.network_morphology import width_profile

    w = width_profile(cfg, s, 1.0)
    assert w[-1] / w.max() <= cfg.quality.terminal_width_ratio


def test_cold_replay_and_immutable_host(settings, example):
    cfg, host = settings
    before = host_semantic_hash(host)
    repeated = CaveNetworkGenerator(cfg).generate(host)
    assert network_semantic_hash(repeated) == network_semantic_hash(example)
    assert repeated.backend_provenance == example.backend_provenance
    assert before == host_semantic_hash(host)
    code = """
from dataclasses import replace
from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.evaluation.artifacts import network_semantic_hash
c = load_project_config('config/varied-network.toml', seed_override=0)
cfg = replace(c.network, regional=replace(c.network.regional, branches_per_km=8))
print(network_semantic_hash(CaveNetworkGenerator(cfg).generate(HostFieldGenerator(c.host_field).generate())))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=150,
        env=dict(os.environ, PYTHONHASHSEED="427"),
    )
    assert result.stdout.strip() == network_semantic_hash(example)
    assert morphology_metrics(repeated) == morphology_metrics(example)

"""Regional routing contracts: host use, sustained connections and bounded repair."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import host_semantic_hash, network_semantic_hash
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import assess_network
from plume_advanced.stages.network_regional_routing import RegionalGrowthConfig, RegionalPlanner
from plume_advanced.stages.network_systems import GenerationDomainError
from plume_advanced.stages.network_topology import NetworkTopologyConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def settings():
    cfg = load_project_config(ROOT / "config/regional-network.toml")
    return cfg.network, HostFieldGenerator(cfg.host_field).generate()


@pytest.fixture(scope="module")
def example(settings):
    cfg, host = settings
    return CaveNetworkGenerator(cfg).generate(host)


@pytest.mark.parametrize(
    "field,value",
    [
        ("planning_step_m", 0),
        ("planning_step_m", True),
        ("planning_step_m", float("nan")),
        ("correlation_length_m", -1),
        ("route_variation", float("inf")),
        ("secondary_scale_weight", -1),
        ("secondary_scale_weight", 1.1),
        ("secondary_scale_weight", float("nan")),
        ("source_stagger_m", -1),
        ("source_stagger_m", True),
        ("source_lateral_jitter_m", -1),
        ("source_lateral_jitter_m", True),
        ("source_lateral_jitter_m", float("inf")),
        ("source_lateral_jitter_m", float("nan")),
        ("outlet_band_width_m", float("nan")),
        ("outlet_count", 0),
        ("outlet_count", 9),
        ("outlet_count", 2.5),
        ("outlet_count", True),
        ("extra_connections", -1),
        ("extra_connections", 9),
        ("extra_connections", 1.5),
        ("extra_connections", True),
        ("branches_per_km", -1),
        ("minimum_branch_length_m", 0),
        ("maximum_branch_length_m", 1),
        ("maximum_branches", 257),
        ("attempts_per_branch", 65),
        ("attempts_per_branch", False),
        ("maximum_grid_cells", 0),
        ("maximum_grid_cells", 1.5),
    ],
)
def test_invalid_controls_fail_early(field, value):
    with pytest.raises(ValueError, match="regional"):
        RegionalGrowthConfig(**{field: value})


def test_regional_requires_interconnected_mode():
    with pytest.raises(ValueError, match="requires interconnected"):
        NetworkTopologyConfig(generation_mode="regional_growth")


def test_staggered_sources_and_outlet_band_are_reproducible(settings):
    cfg, host = settings
    cfg = replace(cfg, regional=replace(cfg.regional, source_stagger_m=30., outlet_band_width_m=100.))
    before = host_semantic_hash(host)
    a = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    b = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    assert a.sources == b.sources and a.goal == b.goal
    g = a.geometry
    stations = (a.xy[a.sources] - [g.seed_x, g.seed_y]) @ [g.flow_x, g.flow_y]
    assert np.ptp(stations) > a.step
    cross = (a.xy[a.goal] - [g.seed_x, g.seed_y]) @ [g.cross_x, g.cross_y]
    assert abs(cross) <= 50
    assert all(np.isfinite(a.potential[s]) for s in a.sources)
    assert host_semantic_hash(host) == before


def test_source_stagger_cannot_consume_the_entire_route(settings):
    cfg, host = settings
    cfg = replace(cfg, regional=replace(cfg.regional, source_stagger_m=cfg.target_route_length_m))
    with pytest.raises(GenerationDomainError, match="Source stagger"):
        RegionalPlanner(CaveNetworkGenerator(cfg), host)


def test_irregular_inlets_replay_without_changing_host_or_cost_field(settings):
    cfg, host = settings
    before = host_semantic_hash(host)
    regular = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    varied = replace(cfg, regional=replace(cfg.regional, source_stagger_m=60.,
                                          source_lateral_jitter_m=10.))
    a = RegionalPlanner(CaveNetworkGenerator(varied), host)
    b = RegionalPlanner(CaveNetworkGenerator(varied), host)
    assert a.sources == b.sources and a.sources != regular.sources
    assert np.array_equal(a.cost, regular.cost)
    assert host_semantic_hash(host) == before
    g = a.geometry
    positions = a.xy[a.sources] - [g.seed_x, g.seed_y]
    assert np.ptp(positions @ [g.flow_x, g.flow_y]) > a.step
    assert np.ptp(np.diff(positions @ [g.cross_x, g.cross_y])) > a.step
    assert len(set(a.sources)) == cfg.systems.count
    assert all(np.isfinite(a.potential[s]) for s in a.sources)


def test_source_jitter_bounds_separation_and_seed_sensitivity(settings):
    cfg, host = settings
    planner = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    spacing = cfg.systems.source_spacing_widths * planner.width
    nominal = (np.arange(cfg.systems.count) - (cfg.systems.count - 1) / 2) * spacing
    stations = np.zeros(cfg.systems.count)
    assert np.array_equal(planner.jitter_source_offsets(nominal, stations), nominal)
    results = []
    for seed in range(20):
        planner.config = replace(cfg, random_seed=seed,
                                 regional=replace(cfg.regional, source_lateral_jitter_m=10.))
        shifted = planner.jitter_source_offsets(nominal, stations)
        assert np.all(abs(shifted - nominal) <= 10.)
        assert np.all(np.diff(shifted) >= 1.3 * planner.width + 1.1 * planner.step)
        results.append(tuple(shifted))
    assert len(set(results)) == 20


def test_impossible_jitter_layout_has_a_bounded_failure(settings):
    cfg, host = settings
    planner = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    planner.config = replace(cfg, regional=replace(cfg.regional, source_lateral_jitter_m=1.))
    with pytest.raises(ValueError, match="exhausted 64 proposals"):
        planner.jitter_source_offsets(np.full(cfg.systems.count, 1e6), np.zeros(cfg.systems.count))


def test_exit_selection_uses_reachable_routing_cost(settings):
    from scipy.sparse import csr_matrix

    cfg, host = settings
    planner = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    planner.config = replace(cfg, regional=replace(cfg.regional, outlet_band_width_m=100.))
    g = planner.geometry
    end = np.array([g.seed_x, g.seed_y]) + g.along_extent * np.array([g.flow_x, g.flow_y])
    left, centre, right = [planner.cell(end + d * np.array([g.cross_x, g.cross_y]))
                           for d in (-35, 0, 35)]
    # The cheapest receiving cell has no route from source 1 and must not win.
    rows, cols, costs = [], [], []
    for i, source in enumerate(planner.sources):
        for goal, cost in ((left, 2.), (centre, 8.), (right, 0.1)):
            if goal == right and i == 1:
                continue
            rows.append(source)
            cols.append(goal)
            costs.append(cost)
    graph = csr_matrix((costs, (rows, cols)), shape=planner.graph.shape)
    assert planner.select_outlet(graph, planner.sources, g.along_extent) == left
    costs = [20. if c == left else v for c, v in zip(cols, costs)]
    graph = csr_matrix((costs, (rows, cols)), shape=planner.graph.shape)
    assert planner.select_outlet(graph, planner.sources, g.along_extent) == centre


def test_graph_is_connected_supplied_and_has_sustained_reconnections(settings, example):
    cfg, host = settings
    report = assess_network(example, host)
    assert report["accepted"], [c for c in report["checks"] if not c["passed"]]
    assert example.max_flow_conservation_error() < 1e-8
    assert len(example.segments) - len(example.nodes) + 1 >= 2
    assert len([n for n in example.nodes if n.kind == "entry"]) == cfg.systems.count
    events = example.backend_provenance["growth_events"]
    accepted = [e for e in events if e["accepted"]]
    assert len(accepted) == example.backend_provenance["accepted_branches"]
    assert all(e["added_length_m"] >= cfg.regional.minimum_branch_length_m for e in accepted)
    assert any(not e["accepted"] for e in events)
    assert (
        len(events)
        <= cfg.regional.attempts_per_branch * example.backend_provenance["requested_branches"]
    )


def test_replay_preserves_decisions_and_immutable_host(settings, example):
    cfg, host = settings
    before = host_semantic_hash(host)
    original_state = np.random.get_state()
    np.random.seed(918)
    state = np.random.get_state()
    try:
        repeated = CaveNetworkGenerator(cfg).generate(host)
        assert network_semantic_hash(repeated) == network_semantic_hash(example)
        assert repeated.backend_provenance == example.backend_provenance
        assert host_semantic_hash(host) == before
        assert np.array_equal(np.random.get_state()[1], state[1])
    finally:
        np.random.set_state(original_state)


def test_seed_changes_routes_on_same_host(settings):
    cfg, host = settings
    p = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    other = RegionalPlanner(CaveNetworkGenerator(replace(cfg, random_seed=17)), host)
    assert not np.array_equal(p.cost, other.cost)
    assert not np.array_equal(p.potential, other.potential)


def test_secondary_scale_is_repeatable_and_changes_routing_field(settings):
    cfg, host = settings
    base = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    varied = replace(cfg, regional=replace(cfg.regional, secondary_scale_weight=0.35))
    a = RegionalPlanner(CaveNetworkGenerator(varied), host)
    b = RegionalPlanner(CaveNetworkGenerator(varied), host)
    assert np.array_equal(a.cost, b.cost)
    assert not np.array_equal(a.cost, base.cost)


def test_grid_budget_fails_before_large_allocation(settings):
    cfg, host = settings
    cfg = replace(cfg, regional=replace(cfg.regional, maximum_grid_cells=4))
    with pytest.raises(GenerationDomainError, match="maximum_grid_cells"):
        RegionalPlanner(CaveNetworkGenerator(cfg), host)


def test_sources_outside_host_are_not_silently_clamped(settings):
    cfg, host = settings
    p = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    with pytest.raises(GenerationDomainError, match="outside"):
        p.cell(np.array([host.x_coords[-1] + 1, 0]))


def test_coarse_edges_cannot_jump_a_thin_host_barrier(settings):
    cfg, host = settings
    cover = host.cover_thickness.copy()
    cover[len(host.y_coords) // 2, :] = -100
    blocked = replace(host, cover_thickness=cover)
    p = RegionalPlanner(CaveNetworkGenerator(cfg), blocked)
    with pytest.raises(GenerationDomainError, match="No viable"):
        p.primary_path(p.cell(np.array(host.config.seed_point)))


def test_routing_uses_host_obstacles_and_can_turn_transversely(settings):
    cfg, host = settings
    cover = np.ones_like(host.cover_thickness) * 30
    middle = len(host.y_coords) // 2
    cover[middle - 4 : middle + 4, host.x_coords > -65] = -100
    cover[middle + 12 : middle + 20, host.x_coords < 65] = -100
    # A level substrate isolates routing geometry from unrelated uphill limits.
    obstacle = replace(host, elevation=np.zeros_like(host.elevation), cover_thickness=cover)
    p = RegionalPlanner(CaveNetworkGenerator(cfg), obstacle)
    route = p.primary_path(p.cell(np.array(host.config.seed_point)))
    xy = p.xy[route]
    assert xy[:, 0].min() < -65
    delta = np.diff(xy, axis=0)
    assert np.any(abs(delta[:, 0]) > 2 * abs(delta[:, 1]))
    assert np.all(np.diff(p.potential[route]) < 0)


def test_smoothing_is_checked_against_current_host(settings, example):
    _, host = settings
    changed = replace(host, cover_thickness=np.zeros_like(host.cover_thickness))
    checks = {c["name"]: c for c in assess_network(example, changed)["checks"]}
    assert not checks["regional_viable_substrate"]["passed"]


def test_flow_duplication_and_lost_ancestry_fail_inspection(settings, example):
    _, host = settings
    segment = example.segments[0]
    broken = replace(
        segment,
        points=tuple(replace(p, flux=p.flux * 2) for p in segment.points),
        metadata=dict(segment.metadata, contributing_system_ids=[]),
    )
    network = replace(example, segments=(broken, *example.segments[1:]))
    checks = {c["name"]: c for c in assess_network(network, host)["checks"]}
    assert not checks["system_source_lineage"]["passed"]
    assert not checks["directed_connectivity_and_flow"]["passed"]


@pytest.mark.parametrize("option", ["detail"])
def test_unintegrated_network_options_remain_network_only(settings, option):
    cfg, host = settings
    cfg = replace(cfg, **{option: replace(getattr(cfg, option), enabled=True)})
    with pytest.raises(ValueError, match="network-only"):
        CaveNetworkGenerator(cfg).generate(host, section_config=object())


def test_single_layer_survey_network_is_screened_with_real_sections(tmp_path):
    from plume_advanced.stages.section_field import SectionFieldGenerator

    cfg = load_project_config(ROOT / "config/earth-survey-full.toml")
    reference = load_project_config(ROOT / "config/earth-survey-network.toml")
    assert cfg.network == reference.network
    assert not cfg.events.enabled and not cfg.events.enabled_kinds
    assert not cfg.acceptance.require_ground_routes
    host = HostFieldGenerator(cfg.host_field).generate()
    original_host = host_semantic_hash(host)
    network = CaveNetworkGenerator(cfg.network).generate(
        host, section_config=cfg.section_field, quality_report_path=tmp_path / "quality.json"
    )
    sections = SectionFieldGenerator(cfg.section_field).generate(network)
    assessment = assess_network(network, host, sections)
    assert assessment["accepted"], [c for c in assessment["checks"] if not c["passed"]]
    assert network.quality_report["scope"] == "network_and_sections"
    assert any(c["name"] == "section_profiles_finite_positive" for c in assessment["checks"])
    assert {s.segment_id for s in sections.segment_fields} == {s.segment_id for s in network.segments}
    assert len([n for n in network.nodes if n.kind == "entry"]) == cfg.network.systems.count
    assert host_semantic_hash(host) == original_host
    assert network.max_flow_conservation_error() < 1e-8


def test_branch_budget_zero_is_an_explicit_tree_mode(settings):
    cfg, host = settings
    cfg = replace(
        cfg,
        regional=replace(cfg.regional, branches_per_km=0),
        systems=replace(cfg.systems, require_split=False),
    )
    network = CaveNetworkGenerator(cfg)._generate_candidate(host)
    assert network.backend_provenance["growth_events"] == []
    assert network.backend_provenance["requested_branches"] == 0
    assert len(network.segments) == len(network.nodes) - 1


@pytest.mark.parametrize("seed,count", [(17, 3), (42, 3), (0, 5), (42, 5)])
def test_other_seeds_do_not_turn_sources_into_through_routes(seed, count):
    c = load_project_config(ROOT / "config/regional-network.toml", seed_override=seed)
    cfg = replace(c.network, systems=replace(c.network.systems, count=count))
    host = HostFieldGenerator(c.host_field).generate()
    n = CaveNetworkGenerator(cfg).generate(host)
    for source in [q for q in n.nodes if q.kind == "entry"]:
        assert not any(s.end_node_id == source.node_id for s in n.segments)
        assert sum(s.start_node_id == source.node_id for s in n.segments) == 1
    assert assess_network(n, host)["accepted"]
    assert n.summary()["shared_passage_length_m"] > 0


def test_network_cli_retains_reports_and_stops_before_sections(tmp_path):
    import json

    from plume_advanced.network_cli import main

    assert (
        main(["--config", str(ROOT / "config/regional-network.toml"), "--output", str(tmp_path)])
        == 0
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["scope"] == "host_and_network_only"
    assert summary["regional"]["accepted_branches"] > 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "resolved_config.json",
        "quality.json",
        "network.json",
        "summary.json",
        "progress.jsonl",
        "networks.png",
        "morphology.png",
        "viewer.html",
    }


def test_cold_replay_with_different_python_hash_seed(settings, example):
    import os
    import subprocess
    import sys

    code = """
from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.evaluation.artifacts import network_semantic_hash
c = load_project_config('config/regional-network.toml')
n = CaveNetworkGenerator(c.network).generate(HostFieldGenerator(c.host_field).generate())
print(network_semantic_hash(n))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env=dict(os.environ, PYTHONHASHSEED="397"),
    )
    assert result.stdout.strip() == network_semantic_hash(example)

"""Multiple termini remain supplied, reproducible sinks after routing and repair."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import host_semantic_hash, network_semantic_hash
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import assess_network
from plume_advanced.stages.network_regional_routing import RegionalPlanner
from plume_advanced.stages.network_systems import GenerationDomainError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenario():
    project = load_project_config(ROOT / "config/multi-outlet-network.toml")
    return project.network, HostFieldGenerator(project.host_field).generate()


@pytest.fixture(scope="module")
def generated(scenario):
    cfg, host = scenario
    return CaveNetworkGenerator(cfg).generate(host)


def test_selected_termini_replay_without_mutating_the_host(scenario):
    cfg, host = scenario
    before = host_semantic_hash(host)
    a = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    b = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    assert a.goals == b.goals and len(set(a.goals)) == 3
    assert np.array_equal(a.potential, b.potential)
    assert np.array_equal(a.predecessors, b.predecessors)
    assert np.all(a.potential[a.goals] == 0)
    assert not set(a.rows) & set(a.goals)
    assert np.all(a.potential[a.rows] > a.potential[a.cols])
    g = a.geometry
    cross = (a.xy[a.goals] - [g.seed_x, g.seed_y]) @ [g.cross_x, g.cross_y]
    assert np.all(np.diff(cross) >= max(3 * a.width, 2 * a.step))
    assert np.all(abs(cross) <= cfg.regional.outlet_band_width_m / 2)
    assert host_semantic_hash(host) == before


def test_unused_termini_receive_real_downstream_forks(scenario):
    cfg, host = scenario
    planner = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    feeders = [planner.primary_path(source) for source in planner.sources]
    # Regression: nearest-outlet routing alone starved requested termini.
    assert {path[-1] for path in feeders} != set(planner.goals)
    added = planner.terminal_continuations(feeders)
    occupied = {cell for path in feeders for cell in path}
    for path in added:
        assert path[0] in occupied and path[0] not in planner.sources
        assert not set(path[1:]) & occupied
        assert path[-1] in planner.goals
        assert np.all(np.diff(planner.potential[path]) < 0)
        assert np.linalg.norm(np.diff(planner.xy[path], axis=0), axis=1).sum() >= cfg.regional.minimum_branch_length_m
        assert all(planner.graph[a, b] > 0 for a, b in zip(path, path[1:]))
        occupied.update(path)
    assert {path[-1] for path in feeders + added} == set(planner.goals)


def test_multi_outlet_network_preserves_flow_dag_and_all_termini(scenario, generated):
    cfg, host = scenario
    report = assess_network(generated, host)
    assert report["accepted"], [c for c in report["checks"] if not c["passed"]]
    exits = {n.node_id for n in generated.nodes if n.kind == "exit"}
    assert len(exits) == cfg.regional.outlet_count
    assert len(generated.backend_provenance["outlet_cells"]) == len(exits)
    assert not exits & {s.start_node_id for s in generated.segments}
    assert generated.max_flow_conservation_error() < 1e-8
    metrics = network_metrics(generated)
    assert metrics["connected_component_count"] == 1
    repair = generated.backend_provenance["connectivity_repair"]
    # This host starts with two groups. Regression must exercise an actual
    # accepted metric connection, not just a later already-connected seed.
    assert repair["initial_components"] == 2 and repair["final_components"] == 1
    assert len(repair["connections"]) == 1
    assert repair["geometry_rejections"] > 0
    assert repair["expanded_states"] <= repair["state_limit"]
    assert metrics["merge_junction_count"] > 0
    assert metrics["split_junction_count"] > 0
    outgoing = {s.start_node_id for s in generated.segments}
    discharged = sum(s.mean_flux for s in generated.segments if s.end_node_id not in outgoing)
    assert discharged == pytest.approx(cfg.systems.count * cfg.source_flux)
    # Strict ordering proves acyclicity independently of the assessment flag.
    for segment in generated.segments:
        assert segment.metadata["regional_start_potential"] > segment.metadata["regional_end_potential"]


def test_exact_generated_replay(scenario, generated):
    cfg, host = scenario
    repeated = CaveNetworkGenerator(cfg).generate(host)
    assert network_semantic_hash(repeated) == network_semantic_hash(generated)
    assert repeated.backend_provenance == generated.backend_provenance


def test_missing_terminus_is_rejected(scenario, generated):
    _, host = scenario
    lost = next(n.node_id for n in generated.nodes if n.kind == "exit")
    nodes = tuple(replace(n, kind="terminal") if n.node_id == lost else n for n in generated.nodes)
    report = assess_network(replace(generated, nodes=nodes), host)
    assert not report["accepted"]
    assert not next(c for c in report["checks"] if c["name"] == "regional_terminal_count")["passed"]


def test_disconnected_node_is_rejected_even_with_all_termini_present(scenario, generated):
    _, host = scenario
    isolated = replace(generated.nodes[0], node_id=max(n.node_id for n in generated.nodes) + 1,
                       kind="junction")
    broken = replace(generated, nodes=generated.nodes + (isolated,))
    report = assess_network(broken, host)
    assert not report["accepted"]
    check = next(c for c in report["checks"] if c["name"] == "regional_connected_components")
    assert not check["passed"] and check["value"] == 2


def test_only_internal_feeder_phase_defers_connectivity(scenario, generated):
    from plume_advanced.stages.network_regional import _assess_feeders

    _, host = scenario
    isolated = replace(generated.nodes[0], node_id=max(n.node_id for n in generated.nodes) + 1,
                       kind="junction")
    broken = replace(generated, nodes=generated.nodes + (isolated,))
    temporary = _assess_feeders(broken, host, defer_connectivity=True)
    normal = _assess_feeders(broken, host)
    assert temporary['checks'] == [c for c in normal['checks']
                                   if c['name'] not in {'regional_connected_components', 'system_merge_opportunity'}]
    assert not normal['accepted']
    assert not assess_network(broken, host)['accepted']


def test_terminal_selection_respects_separate_host_catchments(scenario):
    cfg, host = scenario
    planner = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    planner.config = replace(cfg, regional=replace(cfg.regional, outlet_count=2))
    g = planner.geometry
    end = np.array([g.seed_x, g.seed_y]) + g.along_extent * np.array([g.flow_x, g.flow_y])
    left, right = [planner.cell(end + offset * np.array([g.cross_x, g.cross_y])) for offset in (-60, 60)]
    sources = planner.sources
    # No common reachable goal: each half of the source set has its own basin.
    graph = csr_matrix((np.ones(len(sources)), (sources, [left] * 3 + [right] * 3)),
                       shape=planner.graph.shape)
    assert planner.select_outlets(graph, sources, g.along_extent) == [left, right]
    blocked = csr_matrix((np.ones(3), (sources[:3], [left] * 3)), shape=planner.graph.shape)
    with pytest.raises(ValueError, match="separated termini"):
        planner.select_outlets(blocked, sources, g.along_extent)


def test_too_narrow_outlet_band_fails_without_collapsing_count(scenario):
    cfg, host = scenario
    cfg = replace(cfg, regional=replace(cfg.regional, outlet_band_width_m=5))
    with pytest.raises(ValueError, match="separated termini"):
        RegionalPlanner(CaveNetworkGenerator(cfg), host)


def test_zero_band_selects_an_automatic_multi_outlet_extent(scenario):
    cfg, host = scenario
    cfg = replace(cfg, regional=replace(cfg.regional, outlet_band_width_m=0))
    planner = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    assert len(set(planner.goals)) == cfg.regional.outlet_count


def test_multiple_outlets_and_layers_are_not_silently_combined(scenario):
    cfg, host = scenario
    cfg = replace(cfg, layers=replace(cfg.layers, enabled=True))
    with pytest.raises(GenerationDomainError, match="one layer"):
        RegionalPlanner(CaveNetworkGenerator(cfg), host)


def test_nonregional_mode_does_not_silently_ignore_multiple_outlets(scenario):
    cfg, host = scenario
    cfg = replace(cfg, topology=replace(cfg.topology, generation_mode="independent_growth"))
    with pytest.raises(ValueError, match="require regional_growth"):
        CaveNetworkGenerator(cfg).generate(host)


def test_multi_outlet_generation_cannot_skip_connectivity_inspection(scenario):
    cfg, host = scenario
    cfg = replace(cfg, quality=replace(cfg.quality, enabled=False))
    with pytest.raises(ValueError, match="require network.quality.enabled"):
        CaveNetworkGenerator(cfg).generate(host)

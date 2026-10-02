"""Optional layers must preserve host bounds, physical separation and replay."""

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import network_payload, network_semantic_hash
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_layer_routing import LayeredRegionalPlanner
from plume_advanced.stages.network_layers import NetworkLayersConfig, assess_layers, segment_xyz
from plume_advanced.stages.network_quality import assess_network
from plume_advanced.stages.network_systems import GenerationDomainError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    cfg = load_project_config(ROOT / "config/regional-multilayer.toml")
    host = HostFieldGenerator(cfg.host_field).generate()
    return cfg, host, CaveNetworkGenerator(cfg.network).generate(host)


def layer_checks(network, host=None):
    report = {}

    def check(name, passed, *_args):
        report[name] = bool(passed)

    assess_layers(network, host, check)
    return report


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", 1),
        ("preserve_layer_trunks", 1),
        ("connection_variation", -0.1),
        ("connection_variation", 1.1),
        ("connection_variation", float("nan")),
        ("count", 1),
        ("count", 5),
        ("count", True),
        ("spacing_m", 2),
        ("spacing_m", float("nan")),
        ("passage_height_m", 0),
        ("minimum_rock_m", -1),
        ("maximum_connection_grade", 0),
        ("maximum_connection_grade", 1.1),
        ("connection_opportunities_per_km", 33),
        ("connection_opportunities_per_km", float("inf")),
    ],
)
def test_invalid_layer_controls(field, value):
    with pytest.raises(ValueError, match="network.layers"):
        NetworkLayersConfig(**{field: value})


def test_layers_are_opt_in_and_do_not_change_single_layer_routes():
    c = load_project_config(ROOT / "config/regional-network.toml")
    assert not c.network.layers.enabled
    host = HostFieldGenerator(c.host_field).generate()
    # Compare the same artifact schema: descriptive metadata can evolve while
    # disabled layer settings must never alter the generated network.
    expected = network_semantic_hash(CaveNetworkGenerator(c.network).generate(host))
    cfg = replace(c.network, layers=NetworkLayersConfig(count=4, spacing_m=30))
    assert network_semantic_hash(CaveNetworkGenerator(cfg).generate(host)) == expected


def test_accepted_layers_have_continuous_descending_connections(example):
    c, host, network = example
    report = assess_network(network, host)
    assert report["accepted"], [x for x in report["checks"] if not x["passed"]]
    assert all(layer_checks(network, host).values())
    assert network.max_flow_conservation_error() < 1e-8
    connectors = [
        s
        for s in network.segments
        if s.metadata["regional_end_layer"] != s.metadata["regional_start_layer"]
    ]
    assert connectors
    for s in connectors:
        xyz = segment_xyz(s, c.network.layers)
        grade = np.diff(xyz[:, 2]) / np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        assert abs(grade).max() <= c.network.layers.maximum_connection_grade
    payload = network_payload(network)
    assert payload["layers"]["controls"]["enabled"]
    assert all(
        len(row["centerline_xyz_m"]) == len(row["centerline"]) for row in payload["segments"]
    )


def test_repeatable_layers_never_modify_any_host_array(example):
    c, host, network = example
    original = {
        f.name: getattr(host, f.name).copy()
        for f in fields(host)
        if isinstance(getattr(host, f.name), np.ndarray)
    }
    replay = CaveNetworkGenerator(c.network).generate(host)
    assert network_semantic_hash(replay) == network_semantic_hash(network)
    assert replay.backend_provenance == network.backend_provenance
    assert all(np.array_equal(getattr(host, name), value) for name, value in original.items())


def test_discontinuous_junction_elevations_are_rejected(example):
    _, _, n = example
    s = next(s for s in n.segments if s.metadata["regional_start_layer"] == 0)
    changed = replace(s, points=tuple(replace(p, elevation=p.elevation + 2) for p in s.points))
    broken = replace(
        n, segments=tuple(changed if q.segment_id == s.segment_id else q for q in n.segments)
    )
    assert not layer_checks(broken)["layer_junction_elevation_continuity"]


def test_steep_connection_is_rejected(example):
    _, _, n = example
    s = next(
        s
        for s in n.segments
        if s.metadata["regional_start_layer"] != s.metadata["regional_end_layer"]
    )
    first = s.points[0]
    compressed = replace(
        s,
        points=tuple(
            replace(p, x=first.x + 0.05 * (p.x - first.x), y=first.y + 0.05 * (p.y - first.y))
            for p in s.points
        ),
    )
    assert not layer_checks(replace(n, segments=(compressed,)))["layer_connection_grade"]


@pytest.mark.parametrize("overlap", [False, True])
def test_plan_overlap_uses_actual_height_not_layer_labels(example, overlap):
    _, _, n = example
    a = next(
        s
        for s in n.segments
        if s.metadata["regional_start_layer"] == s.metadata["regional_end_layer"] == 0
    )
    b = replace(
        a,
        segment_id=1000,
        start_node_id=1001,
        end_node_id=1002,
        z_level=1,
        metadata=dict(a.metadata, regional_start_layer=1, regional_end_layer=1),
        points=tuple(
            replace(p, elevation=p.elevation + (n.config.layers.spacing_m if overlap else 0))
            for p in a.points
        ),
    )
    # Separate node identifiers: same XY projection must not create a junction.
    nodes = {q.node_id: q for q in n.nodes}
    other = (
        replace(nodes[a.start_node_id], node_id=1001),
        replace(nodes[a.end_node_id], node_id=1002),
    )
    checks = layer_checks(replace(n, nodes=(*n.nodes, *other), segments=(a, b)))
    assert checks["layer_passage_separation"] is not overlap


def test_insufficient_rock_cannot_be_fixed_by_seed_retry(example):
    c, host, _ = example
    thin = replace(host, emplacement_thickness=np.full_like(host.emplacement_thickness, 4.0))
    with pytest.raises(GenerationDomainError, match="No viable multi-layer"):
        LayeredRegionalPlanner(CaveNetworkGenerator(c.network), thin)


def test_grid_budget_counts_all_layers_before_allocating(example):
    c, host, _ = example
    single = replace(c.network, layers=replace(c.network.layers, enabled=False))
    from plume_advanced.stages.network_regional_routing import RegionalPlanner

    cells = len(RegionalPlanner(CaveNetworkGenerator(single), host).xy)
    cfg = replace(c.network, regional=replace(c.network.regional, maximum_grid_cells=cells + 1))
    with pytest.raises(GenerationDomainError, match="maximum_grid_cells"):
        LayeredRegionalPlanner(CaveNetworkGenerator(cfg), host)


def test_too_few_sources_and_unsupported_modes_fail_config_validation(tmp_path):
    source = (ROOT / "config/regional-multilayer.toml").read_text()
    for text in (
        source.replace(
            'generation_mode = "regional_growth"', 'generation_mode = "independent_growth"'
        ),
        source.replace("count = 2", "count = 4"),
    ):
        path = tmp_path / "invalid.toml"
        path.write_text(text)
        with pytest.raises(ValueError, match="regional_growth|source per layer"):
            load_project_config(path)


def test_layered_cold_replay(example):
    import os
    import subprocess
    import sys

    code = """
from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.evaluation.artifacts import network_semantic_hash
c=load_project_config('config/regional-multilayer.toml')
print(network_semantic_hash(CaveNetworkGenerator(c.network).generate(HostFieldGenerator(c.host_field).generate())))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
        env=dict(os.environ, PYTHONHASHSEED="91"),
    )
    assert result.stdout.strip() == network_semantic_hash(example[2])


def test_direct_python_generation_cannot_silently_ignore_layers(example):
    c, host, _ = example
    cfg = replace(
        c.network, topology=replace(c.network.topology, generation_mode="independent_growth")
    )
    with pytest.raises(ValueError, match="Optional layers require regional_growth"):
        CaveNetworkGenerator(cfg).generate(host)


def test_missing_layer_metadata_fails_inspection(example):
    _, host, n = example
    s = n.segments[0]
    metadata = dict(s.metadata)
    del metadata["regional_start_layer"]
    broken = replace(n, segments=(replace(s, metadata=metadata), *n.segments[1:]))
    report = assess_network(broken, host)
    assert not report["accepted"]
    assert any(
        c["name"] == "layer_metadata_and_elevations" and not c["passed"] for c in report["checks"]
    )


@pytest.fixture(scope="module")
def complex_example():
    c = load_project_config(ROOT / "config/complex-network.toml", seed_override=0)
    h = HostFieldGenerator(c.host_field).generate()
    # Exercise the same retained trunks with a smaller bypass budget in unit tests.
    cfg = replace(c.network, regional=replace(c.network.regional, branches_per_km=4))
    return h, CaveNetworkGenerator(cfg).generate(h)


def test_persistent_layers_retain_outlets_connected_to_sources(complex_example):
    from plume_advanced.evaluation.metrics.network import network_metrics

    h, n = complex_example
    assert assess_network(n, h)["accepted"]
    assert network_metrics(n)["connected_component_count"] == 1
    assert n.summary()["loop_count"] >= 2
    assert n.max_flow_conservation_error() < 1e-8
    exits = {p.node_id for p in n.nodes if p.kind == "exit"}
    assert len(exits) == n.config.layers.count
    assert not exits & {s.start_node_id for s in n.segments}
    # Every level must have an uninterrupted same-level source-to-exit path.
    for layer in range(n.config.layers.count):
        reached = {p.node_id for p in n.nodes if p.kind == "entry"}
        previous = set()
        while reached != previous:
            previous = reached.copy()
            reached.update(
                s.end_node_id
                for s in n.segments
                if s.start_node_id in reached
                and s.metadata["regional_start_layer"] == layer
                and s.metadata["regional_end_layer"] == layer
            )
        assert reached & exits


def test_disconnect_layers_while_preserving_inlets_and_outlets_is_rejected(complex_example):
    """Intact per-layer trunks do not prove the whole cave is connected."""
    from plume_advanced.evaluation.metrics.network import network_metrics

    host, network = complex_example
    broken = replace(network, segments=tuple(
        s for s in network.segments
        if s.metadata["regional_start_layer"] == s.metadata["regional_end_layer"]
    ))
    assert broken.nodes == network.nodes
    assert network_metrics(broken)["connected_component_count"] > 1
    report = assess_network(broken, host)
    checks = {c['name']: c for c in report['checks']}
    assert checks['layer_retained_outlets']['passed']
    assert checks['layer_source_to_exit_trunks']['passed']
    assert not checks['regional_connected_components']['passed']
    assert not checks['layer_connections']['passed']
    assert not report['accepted']


def test_retained_exit_without_its_same_level_source_is_rejected(complex_example):
    _, network = complex_example
    upper_sources = {s.start_node_id for s in network.segments
                     if s.metadata["regional_start_layer"] == 0}
    damaged = replace(network, nodes=tuple(
        replace(n, kind="terminal") if n.kind == "entry" and n.node_id in upper_sources else n
        for n in network.nodes
    ))
    checks = layer_checks(damaged)
    assert checks["layer_retained_outlets"]
    assert not checks["layer_source_to_exit_trunks"]


def test_removing_retained_outlet_fails_inspection(complex_example):
    h, n = complex_example
    outlet = next(p.node_id for p in n.nodes if p.kind == "exit")
    broken = replace(
        n, nodes=tuple(replace(p, kind="junction") if p.node_id == outlet else p for p in n.nodes)
    )
    assert not layer_checks(broken, h)["layer_retained_outlets"]


def test_complex_replay_is_seeded_and_host_is_immutable(complex_example):
    from plume_advanced.evaluation.artifacts import host_semantic_hash

    h, n = complex_example
    before = host_semantic_hash(h)
    c = load_project_config(ROOT / "config/complex-network.toml", seed_override=0)
    cfg = replace(c.network, regional=replace(c.network.regional, branches_per_km=4))
    replay = CaveNetworkGenerator(cfg).generate(h)
    assert network_semantic_hash(replay) == network_semantic_hash(n)
    assert replay.quality_report == n.quality_report
    assert host_semantic_hash(h) == before


def test_varied_ramps_and_retained_outlets_obey_directed_potential():
    c = load_project_config(ROOT / "config/complex-network.toml", seed_override=0)
    h = HostFieldGenerator(c.host_field).generate()
    p = LayeredRegionalPlanner(CaveNetworkGenerator(c.network), h)
    a, b = p.rows, p.cols
    assert np.all(p.potential[a] > p.potential[b])
    assert not np.isin(a, p.goals).any()
    assert len(p.goals) == c.network.layers.count
    ramps = p.layer_ids[a] != p.layer_ids[b]
    lengths = np.linalg.norm(p.xy[b[ramps]] - p.xy[a[ramps]], axis=1)
    assert np.ptp(lengths) > 20
    replay = LayeredRegionalPlanner(CaveNetworkGenerator(c.network), h)
    assert np.array_equal(p.rows, replay.rows)
    assert np.array_equal(p.weights, replay.weights)


@pytest.mark.parametrize('layer_count', [2, 3])
def test_pipeline_reconnects_independent_layer_feeders_and_replays(monkeypatch, layer_count):
    """Inject disconnected, supplied trunks through the actual generation pipeline."""
    from scipy.sparse.csgraph import dijkstra

    from plume_advanced.evaluation.artifacts import host_semantic_hash
    from plume_advanced.evaluation.metrics.network import network_metrics
    from plume_advanced.stages import network_connectivity

    def independent_feeder(planner, source):
        goal = planner.goals[planner.layer_ids[source]]
        _, previous = dijkstra(planner.graph.T.tocsr(), indices=goal, return_predecessors=True)
        path = [source]
        while path[-1] != goal:
            other = int(previous[path[-1]])
            if other < 0:
                raise ValueError('Injected same-layer feeder cannot reach its outlet')
            path.append(other)
        return path

    original_connect = network_connectivity.connect_paths
    inspected = []

    def record_connections(planner, paths, **kwargs):
        before = [p[:] for p in paths]
        endpoints = list(planner.sources), list(planner.goals)
        result = original_connect(planner, paths, **kwargs)
        assert paths == before
        assert endpoints == (planner.sources, planner.goals)
        inspected.append(result[1])
        return result

    monkeypatch.setattr(LayeredRegionalPlanner, 'primary_path', independent_feeder)
    # All original same-level source-to-outlet paths are already supplied.
    monkeypatch.setattr(LayeredRegionalPlanner, 'retained_trunks', lambda *_: [])
    monkeypatch.setattr(network_connectivity, 'connect_paths', record_connections)
    c = load_project_config(ROOT / 'config/varied-network.toml', seed_override=42)
    cfg = replace(c.network, layers=replace(c.network.layers, count=layer_count),
                  regional=replace(c.network.regional, branches_per_km=0))
    host = HostFieldGenerator(c.host_field).generate()
    host_before = host_semantic_hash(host)
    network = CaveNetworkGenerator(cfg).generate(host)
    audit = network.backend_provenance['connectivity_repair']
    assert inspected and audit['initial_components'] == layer_count
    assert audit['final_components'] == network_metrics(network)['connected_component_count'] == 1
    assert len(audit['connections']) == layer_count - 1
    assert sum(c['ramp_count'] for c in audit['connections']) == layer_count - 1
    assert audit['expanded_states'] <= audit['state_limit']
    assert audit['searches'] <= 64 * (layer_count - 1)
    assert sum(n.kind == 'entry' for n in network.nodes) == cfg.systems.count
    assert sum(n.kind == 'exit' for n in network.nodes) == layer_count
    assert all(layer_checks(network, host).values())
    assert assess_network(network, host)['accepted']
    replay = CaveNetworkGenerator(cfg).generate(host)
    assert network_semantic_hash(replay) == network_semantic_hash(network)
    assert replay.backend_provenance == network.backend_provenance
    assert replay.quality_report == network.quality_report
    assert host_semantic_hash(host) == host_before

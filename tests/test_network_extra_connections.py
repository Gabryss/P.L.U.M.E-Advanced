"""Sparse optional links preserve the cave and never become a density quota."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import host_semantic_hash, network_semantic_hash
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import assess_network
from plume_advanced.stages.network_regional_geometry import AcceptedRoutes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def linked_example():
    config = load_project_config(ROOT / 'config/branching-layers.toml', seed_override=42)
    cfg = replace(config.network, quality=replace(config.network.quality, max_attempts=1))
    host = HostFieldGenerator(config.host_field).generate()
    before = host_semantic_hash(host)
    baseline = CaveNetworkGenerator(replace(cfg, regional=replace(cfg.regional, extra_connections=0))).generate(host)
    linked = CaveNetworkGenerator(cfg).generate(host)
    assert host_semantic_hash(host) == before
    return cfg, host, baseline, linked


def test_additional_links_preserve_ports_routes_and_candidate_seed(linked_example):
    cfg, host, baseline, linked = linked_example
    audit = linked.backend_provenance['extra_connections']
    assert audit['extra_requested'] == cfg.regional.extra_connections == 2
    assert audit['extra_accepted'] == 2
    assert [a['kind'] for a in audit['extra_attempts']] == ['descending_ramp', 'same_layer']
    assert all(a['accepted'] for a in audit['extra_attempts'])
    assert any(c['ramp_count'] == 1 for c in audit['connections'])
    assert linked.quality_report['selected_seed'] == baseline.quality_report['selected_seed']
    assert network_metrics(linked)['connected_component_count'] == 1
    assert network_metrics(linked)['cyclomatic_number'] == network_metrics(baseline)['cyclomatic_number'] + 2
    assert linked.backend_provenance['accepted_branches'] == baseline.backend_provenance['accepted_branches']
    assert assess_network(linked, host)['accepted']
    # Inlet/outlet coordinates survive although junction insertion renumbers nodes.
    def ports(network):
        return sorted((p.kind, p.x, p.y) for p in network.nodes if p.kind in {'entry', 'exit'})

    assert ports(linked) == ports(baseline)
    old, new = AcceptedRoutes(baseline), AcceptedRoutes(linked)
    assert old.edges.keys() <= new.edges.keys()
    for cell, position in old.positions.items():
        np.testing.assert_allclose(position, new.positions[cell], atol=1e-6, rtol=0)
    assert audit['expanded_states'] <= audit['state_limit'] == 16384
    assert audit['searches'] <= 128


def test_extra_connections_replay_exactly(linked_example):
    cfg, host, _, linked = linked_example
    replay = CaveNetworkGenerator(cfg).generate(host)
    assert network_semantic_hash(replay) == network_semantic_hash(linked)
    assert replay.quality_report == linked.quality_report


def test_failed_optional_fit_keeps_original_network_without_seed_retry(linked_example, monkeypatch):
    from plume_advanced.stages import network_regional

    cfg, host, baseline, _ = linked_example
    original = network_regional._network_from_paths

    def fail_optional(generator, host, planner, paths, sources, events, path_types, previous=None):
        if path_types[-1] == 'connection':
            raise ValueError('Injected optional connection fit failure')
        return original(generator, host, planner, paths, sources, events, path_types, previous=previous)

    monkeypatch.setattr(network_regional, '_network_from_paths', fail_optional)
    result = CaveNetworkGenerator(cfg).generate(host)
    audit = result.backend_provenance['extra_connections']
    assert audit['extra_accepted'] == 0
    assert audit['rejected_checks']['connection_construction'] > 0
    assert set(audit['construction_errors']) == {'Injected optional connection fit failure'}
    assert result.nodes == baseline.nodes and result.segments == baseline.segments
    assert result.quality_report['selected_attempt'] == 0
    assert assess_network(result, host)['accepted']


@pytest.mark.parametrize('invalid', ['quality', 'mode'])
def test_extra_links_cannot_silently_skip_inspection(invalid):
    config = load_project_config(ROOT / 'config/branching-layers.toml')
    cfg = replace(config.network, layers=replace(config.network.layers, enabled=False))
    cfg = (replace(cfg, quality=replace(cfg.quality, enabled=False)) if invalid == 'quality'
           else replace(cfg, topology=replace(cfg.topology, generation_mode='independent_growth')))
    with pytest.raises(ValueError, match='Extra connections require inspected regional_growth'):
        CaveNetworkGenerator(cfg).generate(None)

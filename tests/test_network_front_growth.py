"""Endpoint-free growth, conserved allocation and reproducible bounded births."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import host_semantic_hash, network_semantic_hash
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_front_growth import FrontGrower, branch_opportunities
from plume_advanced.stages.network_quality import assess_network
from plume_advanced.stages.network_regional_routing import RegionalGrowthConfig, RegionalPlanner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", ["", "arbitrary", True, None, 5])
def test_invalid_growth_method(value):
    with pytest.raises(ValueError, match="branch_growth"):
        RegionalGrowthConfig(branch_growth=value)


def test_births_are_repeatable_bounded_and_not_a_success_quota():
    c = load_project_config(ROOT / "config/varied-network.toml").network
    np.random.seed(123)
    before = np.random.get_state()
    counts = [branch_opportunities(replace(c, random_seed=seed), 800) for seed in range(30)]
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert counts == [branch_opportunities(replace(c, random_seed=seed), 800) for seed in range(30)]
    assert len(set(counts)) > 5
    assert all(0 <= n <= c.regional.maximum_branches for n in counts)
    assert branch_opportunities(replace(c, regional=replace(c.regional, branches_per_km=0)), 800) == 0
    assert branch_opportunities(replace(c, regional=replace(c.regional, branches_per_km=1e100)), 800) == c.regional.maximum_branches


@pytest.fixture(scope="module")
def example():
    c = load_project_config(ROOT / "config/varied-network.toml", seed_override=17)
    cfg = replace(c.network, layers=replace(c.network.layers, enabled=False))
    host = HostFieldGenerator(c.host_field).generate()
    before = host_semantic_hash(host)
    # Endpoint-free branches must never call the endpoint-constrained builder.
    with patch.object(RegionalPlanner, "detour", side_effect=AssertionError("fixed destination")):
        first = CaveNetworkGenerator(cfg).generate(host)
        second = CaveNetworkGenerator(cfg).generate(host)
    assert network_semantic_hash(first) == network_semantic_hash(second)
    assert first.backend_provenance == second.backend_provenance
    assert host_semantic_hash(host) == before
    return first, host


def test_front_can_both_merge_and_stop_without_prescribed_destinations(example):
    network, host = example
    p = network.backend_provenance
    accepted = [e for e in p["growth_events"] if e["accepted"]]
    assert {e["route_type"] for e in accepted} == {"bypass", "blind_branch"}
    assert all(e["destination_preselected"] is False for e in accepted)
    assert all(e["capture_started_m"] is None or e["capture_started_m"] >= network.config.regional.minimum_branch_length_m for e in accepted)
    assert any(e["termination"] == "local_capture" for e in accepted)
    for event in p["growth_events"]:
        assert event.get("capture_expanded_states", 0) <= 256 * event.get("capture_searches", 0)
        assert event.get("capture_searches", 0) <= 4
    assert p["route_relaxation"]
    for proposal in p["route_relaxation"]:
        assert proposal["iterations"] <= 40
        if proposal["accepted"]:
            assert proposal["energy_after"] < proposal["energy_before"]
            assert proposal["maximum_displacement_m"] <= 4*network.config.base_passage_radius + 1e-8
    assert not p["branch_budget_exhausted"]
    assert p["branch_opportunities"] == p["accepted_branches"] + p["unsuccessful_opportunities"]
    assert len(p["growth_events"]) <= p["branch_opportunities"] * min(4, network.config.regional.attempts_per_branch)
    assert assess_network(network, host)["accepted"]
    assert network.max_flow_conservation_error() < 1e-8
    # Reconnection is a real shared graph junction, not an unmodeled crossing.
    for event in accepted:
        if event["route_type"] == "bypass":
            ends = [s.end_node_id for s in network.segments if s.metadata.get("regional_branch_id") == event["branch_id"]]
            assert any(sum(s.end_node_id == node for s in network.segments) > 1 for node in ends)


def test_front_obeys_directed_host_graph_and_work_limit():
    c = load_project_config(ROOT / "config/varied-network.toml", seed_override=0)
    cfg = replace(c.network, layers=replace(c.network.layers, enabled=False))
    host = HostFieldGenerator(c.host_field).generate()
    p = RegionalPlanner(CaveNetworkGenerator(cfg), host)
    paths = [p.primary_path(source) for source in p.sources]
    front = FrontGrower(p)
    start = paths[0][20]
    a, b = (front.grow(start, paths, 0) for _ in range(2))
    assert a == b
    edges = set(zip(p.rows, p.cols))
    assert all((i, j) in edges for i, j in zip(a.path, a.path[1:]))
    assert all(p.potential[i] > p.potential[j] for i, j in zip(a.path, a.path[1:]))
    assert a.length_m <= cfg.regional.maximum_branch_length_m
    assert a.steps <= int(np.ceil(cfg.regional.maximum_branch_length_m / min(np.diff(p.x).min(), np.diff(p.y).min()))) + 2


def test_feeder_inspection_does_not_require_branches_before_they_can_grow():
    from plume_advanced.stages.network_regional import _assess_feeders

    c = load_project_config(ROOT / "config/regional-network.toml")
    host = HostFieldGenerator(c.host_field).generate()
    cfg = replace(c.network, systems=replace(c.network.systems, require_split=False),
                  regional=replace(c.network.regional, branches_per_km=0))
    tree = CaveNetworkGenerator(cfg).generate(host)
    # The final network must still satisfy the user's required split.
    required = replace(tree, config=replace(tree.config, systems=replace(tree.config.systems, require_split=True)))
    report = assess_network(required, host)
    assert any(c["name"] == "system_split_opportunity" and not c["passed"] for c in report["checks"])
    assert _assess_feeders(required, host)["accepted"]
    assert required.config.systems.require_split


def test_splitting_a_blind_arm_does_not_taper_its_upstream_piece(example):
    from plume_advanced.stages.network_morphology import width_profile

    network, _ = example
    arm = next(s for s in network.segments if s.metadata["regional_route_type"] == "blind_branch")
    upstream = replace(arm, metadata=dict(arm.metadata, regional_blind_terminal=False))
    terminal = replace(arm, metadata=dict(arm.metadata, regional_blind_terminal=True))
    a = width_profile(network.config, upstream, arm.mean_flux)
    b = width_profile(network.config, terminal, arm.mean_flux)
    assert b[-1] / max(b) < network.config.quality.terminal_width_ratio
    assert a[-1] / max(a) > network.config.quality.terminal_width_ratio


def test_total_search_budget_includes_rejected_branches(example):
    network, _ = example
    p = network.backend_provenance
    assert 0 <= p["local_search_expanded_states"] <= p["local_search_state_limit"]
    assert p["local_search_state_limit"] == 2400 * network.config.quality.repair_passes
    branch_work = sum(r["expanded_states"] for e in p["growth_events"] for r in e.get("local_repairs", []))
    assert branch_work <= p["local_search_expanded_states"]
    assert p["local_search_budget_exhausted"] == (p["local_search_expanded_states"] == p["local_search_state_limit"])


def test_rejected_candidates_keep_their_construction_audit(tmp_path):
    import json

    from test_network_quality import network_fixture

    from plume_advanced.stages.network_quality import NetworkQualityError

    c = load_project_config(ROOT / "config/varied-network.toml")
    cfg = replace(c.network, quality=replace(c.network.quality, max_attempts=2, repair_passes=0))
    provenance = dict(regional_growth_completed=False, local_search_expanded_states=20,
                      local_repair_history=[dict(strategy="work_limit", accepted=False)])
    candidate = replace(network_fixture(config=cfg), backend_provenance=provenance)
    failed = dict(accepted=False, checks=[dict(name="regional_growth_completed", passed=False,
                                               severity="error", segment_ids=[])])
    path = tmp_path / "quality.json"
    with patch.object(CaveNetworkGenerator, "_generate_candidate", return_value=candidate), patch(
            "plume_advanced.stages.network_acceptance.assess_network", return_value=failed):
        with pytest.raises(NetworkQualityError):
            CaveNetworkGenerator(cfg).generate(None, quality_report_path=path)
    audit = json.loads(path.read_text())
    assert len(audit["attempts"]) == 2
    assert audit["attempts"][0]["seed"] != audit["attempts"][1]["seed"]
    assert all(a["construction"] == provenance for a in audit["attempts"])

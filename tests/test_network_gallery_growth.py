"""Independent galleries must earn their topology and retain real phase ledgers."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import network_semantic_hash, section_semantic_hash
from plume_advanced.procedural import procedural_rng
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_acceptance import repair_network
from plume_advanced.stages.network_gallery_growth import (
    assess_gallery_history,
    independent_preferences,
    refresh_phase_discharge,
    trace_blind_breakout,
)
from plume_advanced.stages.network_quality import NetworkQualityError, assess_network
from plume_advanced.stages.network_systems import plan_interactions
from plume_advanced.stages.section_field import SectionFieldGenerator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    cfg = load_project_config(ROOT / "config/earth_independent_gallery.toml")
    host = HostFieldGenerator(cfg.host_field).generate()
    network = CaveNetworkGenerator(cfg.network).generate(host, section_config=cfg.section_field)
    return cfg, host, network, SectionFieldGenerator(cfg.section_field).generate(network)


def history_failures(network):
    failed = []
    assess_gallery_history(
        network, lambda name, passed, *a: failed.append(name) if not passed else None
    )
    return set(failed)


def test_independent_routes_preserve_graph_sections_and_phase_conservation(example):
    _, host, n, sections = example
    report = assess_network(n, host, sections)
    assert report["accepted"], [c for c in report["checks"] if not c["passed"]]
    assert len(report["checks"]) >= 50
    events = n.backend_provenance["interaction_events"]
    assert len({e["station_m"] for e in events if e["kind"] == "merge"}) >= 2
    assert any(e["kind"] == "split" for e in events)
    branches = [s for s in n.segments if s.metadata.get("topology_role") == "side_branch"]
    assert branches and any(s.metadata["reoccupied"] for s in branches)
    for s in branches:
        assert any(f == 0 for f in s.metadata["phase_fluxes"])
        assert s.metadata["active_phase_count"] < s.metadata["emplacement_phase_count"]
    assert any(j.metadata.get("chamber_type") == "drained_lava_pool" for j in n.junctions)


def test_replay_reproduces_network_profiles_and_rejection_history(example):
    cfg, host, n, sections = example
    repeat = CaveNetworkGenerator(cfg.network).generate(host, section_config=cfg.section_field)
    assert network_semantic_hash(repeat) == network_semantic_hash(n)
    assert section_semantic_hash(
        SectionFieldGenerator(cfg.section_field).generate(repeat)
    ) == section_semantic_hash(sections)
    assert n.quality_report == repeat.quality_report


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("dormant_flow", "phase_activity_ledger"),
        ("unbalanced", "phase_discharge_conservation"),
        ("chronology", "phase_chronology"),
        ("seed", "growth_source_streams"),
        ("event", "growth_interaction_evidence"),
        ("missing_event", "growth_interaction_evidence"),
    ],
)
def test_bad_history_cannot_pass(example, mutation, expected):
    _, _, n, _ = example
    segments = list(n.segments)
    branch = next(
        i for i, s in enumerate(segments) if s.metadata.get("topology_role") == "side_branch"
    )
    if mutation in {"dormant_flow", "unbalanced", "chronology"}:
        s = segments[branch]
        m = dict(s.metadata)
        if mutation == "chronology":
            m["active_phases"] = [-1, 3]
        else:
            m["phase_fluxes"] = list(m["phase_fluxes"])
            phase = next(
                i
                for i in range(len(m["phase_fluxes"]))
                if (i in m["active_phases"]) == (mutation == "unbalanced")
            )
            m["phase_fluxes"][phase] += 0.1
        segments[branch] = replace(s, metadata=m)
    elif mutation == "seed":
        i = next(i for i, s in enumerate(segments) if "source_system_id" in s.metadata)
        segments[i] = replace(segments[i], metadata=dict(segments[i].metadata, system_seed=0))
    else:
        provenance = dict(n.backend_provenance)
        events = [dict(e) for e in provenance["interaction_events"]]
        if mutation == "event":
            events[0]["preference_gap_m"] = 1000
        else:
            events.pop()
        provenance["interaction_events"] = events
        n = replace(n, backend_provenance=provenance)
    assert expected in history_failures(replace(n, segments=tuple(segments)))


def test_repairs_refresh_phase_discharge(example):
    _, host, n, _ = example
    # Stale flow history must not survive geometry repair.
    stale = replace(
        n,
        segments=tuple(
            replace(s, metadata=dict(s.metadata, phase_fluxes=[0] * 4)) for s in n.segments
        ),
    )
    repaired = repair_network(CaveNetworkGenerator(n.config), host, stale, 1)
    assert not history_failures(repaired)


def test_source_seed_and_host_change_preferences_and_event_positions(example):
    _, host, n, _ = example
    gen = CaveNetworkGenerator(n.config)
    geo = gen._build_flow_geometry(host)
    along, tracks, width = independent_preferences(gen, host, geo)
    other = CaveNetworkGenerator(replace(n.config, random_seed=n.config.random_seed + 1))
    a, t, _ = independent_preferences(other, host, geo)
    assert not np.allclose(tracks, t)
    first = []
    second = []
    plan_interactions(along, tracks, width, n.config.systems, events=first)
    plan_interactions(a, t, width, n.config.systems, events=second)
    assert [(e["kind"], e["station_m"]) for e in first] != [
        (e["kind"], e["station_m"]) for e in second
    ]
    # A spatial cost gradient affects routes even with exactly the same seed.
    bias = np.broadcast_to(
        np.linspace(0, 4, host.growth_cost.shape[1]), host.growth_cost.shape
    ).copy()
    _, changed, _ = independent_preferences(gen, replace(host, growth_cost=bias), geo)
    assert np.max(np.abs(tracks - changed)) > 0.1


def test_budget_is_shared_across_simultaneous_reopened_branches(example):
    _, _, n, _ = example
    segments = [
        replace(
            s,
            metadata=dict(
                s.metadata,
                active_phases=[0, 1, 2, 3],
                birth_phase=0,
                death_phase=3,
                reoccupied=False,
                initial_flux=100,
            ),
        )
        if s.metadata.get("topology_role") == "side_branch"
        else s
        for s in n.segments
    ]
    gen = CaveNetworkGenerator(n.config)
    result = refresh_phase_discharge(gen, list(n.nodes), segments)
    budget = (
        n.config.source_flux
        * n.config.systems.count
        * n.config.emplacement_history.phase_flux_budget_fraction
    )
    for phase in range(4):
        assert sum(
            s.metadata["phase_fluxes"][phase]
            for s in result
            if s.metadata.get("topology_role") == "side_branch"
        ) == pytest.approx(budget)
    assert not history_failures(replace(n, segments=tuple(result)))


def test_cooling_and_flux_change_breakout_geometry(example):
    _, host, n, _ = example
    parent = next(
        s for s in n.segments if s.metadata.get("topology_role") == "trunk" and len(s.points) > 10
    )
    gen = CaveNetworkGenerator(n.config)
    args = (host, parent, len(parent.points) // 2)
    warm, _ = trace_blind_breakout(gen, *args, procedural_rng(4, "test"), 0.9)
    cold_gen = CaveNetworkGenerator(replace(n.config, cooling_k_per_m=20))
    cold, reason = trace_blind_breakout(cold_gen, *args, procedural_rng(4, "test"), 0.9)
    small, _ = trace_blind_breakout(gen, *args, procedural_rng(4, "test"), 0.1)
    assert cold[-1].arc_length < warm[-1].arc_length
    assert reason == "cooled_below_retirement_temperature"
    assert small[0].width < warm[0].width


def test_reoccupation_affects_cross_sections(example):
    cfg, _, n, sections = example
    altered = replace(
        n,
        segments=tuple(
            replace(s, metadata=dict(s.metadata, active_phase_count=1))
            if s.metadata.get("reoccupied")
            else s
            for s in n.segments
        ),
    )
    assert section_semantic_hash(
        SectionFieldGenerator(cfg.section_field).generate(altered)
    ) != section_semantic_hash(sections)


def test_disabling_breakouts_fails_closed_when_branches_are_required(example):
    _, host, n, _ = example
    cfg = replace(
        n.config,
        quality=replace(n.config.quality, max_attempts=1, repair_passes=0),
        emplacement_history=replace(n.config.emplacement_history, breakout_probability=0),
    )
    with pytest.raises(NetworkQualityError):
        CaveNetworkGenerator(cfg).generate(host)


def test_unsupported_stacking_is_explicit(example):
    _, host, n, _ = example
    from plume_advanced.stages.network_gallery_growth import generate_gallery_growth

    cfg = replace(
        n.config,
        emplacement_history=replace(n.config.emplacement_history, stacked_lobe_fraction=0.2),
    )
    with pytest.raises(ValueError, match="one layer"):
        generate_gallery_growth(CaveNetworkGenerator(cfg), host)


def test_disabled_pools_and_reoccupation_are_respected(example):
    _, host, n, _ = example
    from plume_advanced.stages.network_gallery_growth import generate_gallery_growth

    cfg = replace(
        n.config,
        emplacement_history=replace(
            n.config.emplacement_history, drained_pool_enabled=False, reoccupation_probability=0
        ),
    )
    generated = generate_gallery_growth(CaveNetworkGenerator(cfg), host)
    assert all(j.metadata.get("chamber_type") != "drained_lava_pool" for j in generated.junctions)
    assert not any(s.metadata.get("reoccupied") for s in generated.segments)
    assert len(generated.nodes) < len(n.nodes)


def test_history_metrics_count_actual_activity_and_reused_flux(example):
    _, _, n, _ = example
    from plume_advanced.evaluation.metrics.emplacement import emplacement_metrics
    from plume_advanced.evaluation.metrics.network import network_metrics

    assert network_metrics(n)["breakout_event_count"] == 2
    assert n.summary()["breakout_event_count"] == 2
    rows = emplacement_metrics(n)["phase_activity"]
    for row in rows:
        phase = row["phase"]
        active = [s for s in n.segments if phase in s.metadata["active_phases"]]
        assert row["active_segment_count"] == len(active)
        assert row["allocated_flux"] == pytest.approx(
            sum(
                s.metadata["phase_fluxes"][phase]
                for s in active
                if s.metadata.get("topology_role") == "side_branch"
            )
        )
    bad = replace(
        n,
        config=replace(
            n.config,
            emplacement_history=replace(
                n.config.emplacement_history, phase_flux_budget_fraction=0.001
            ),
        ),
    )
    assert "phase_breakout_budget" in history_failures(bad)

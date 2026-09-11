"""Reject convincing-looking skeletons whose section footprints lose islands."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import network_semantic_hash, section_semantic_hash
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.network_quality import NetworkQualityError, assess_network
from plume_advanced.stages.network_topology import (
    NetworkTopologyConfig,
    section_footprint,
    topology_metrics,
)
from plume_advanced.stages.section_field import SectionFieldGenerator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    config = load_project_config(ROOT / "config/earth_valentine_topology.toml")
    host = HostFieldGenerator(config.host_field).generate()
    network = CaveNetworkGenerator(config.network).generate(
        host, section_config=config.section_field
    )
    sections = SectionFieldGenerator(config.section_field).generate(network)
    return config, host, network, sections


def failures(report):
    return {c["name"] for c in report["checks"] if not c["passed"]}


def test_reference_style_has_a_connected_gallery_and_two_surviving_islands(example):
    _, host, network, sections = example
    report = assess_network(network, host, sections)
    assert report["accepted"], failures(report)
    metrics = topology_metrics(network)
    assert metrics["trunk_length_fraction"] >= 0.65
    assert metrics["single_channel_fraction"] >= 0.55
    assert metrics["maximum_bypass_fraction"] <= 0.22
    checks = {c["name"]: c["value"] for c in report["checks"]}
    assert checks["section_footprint_connected"] == 1
    assert checks["section_footprint_islands"] == 2
    assert checks["section_island_clearance"] >= 1.52
    assert sections.summary()["unstable_section_count"] == 0
    assert network.max_flow_conservation_error() < 1e-12


def test_replay_includes_identical_profiles_and_decisions(example):
    config, host, original, sections = example
    repeat = CaveNetworkGenerator(config.network).generate(
        host, section_config=config.section_field
    )
    assert network_semantic_hash(repeat) == network_semantic_hash(original)
    assert section_semantic_hash(
        SectionFieldGenerator(config.section_field).generate(repeat)
    ) == section_semantic_hash(sections)
    assert original.quality_report == repeat.quality_report


def test_a_long_distant_bypass_fails_even_when_endpoints_still_match(example):
    _, host, network, _ = example
    segments = list(network.segments)
    sid = next(i for i, s in enumerate(segments) if s.metadata.get("island_id"))
    segment = segments[sid]
    xy = np.array([[p.x, p.y] for p in segment.points])
    t = np.linspace(0, 1, len(xy))
    chord = xy[-1] - xy[0]
    normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    xy += 150 * np.sin(np.pi * t)[:, None] * normal
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    points = tuple(
        replace(p, x=float(v[0]), y=float(v[1]), arc_length=float(a))
        for p, v, a in zip(segment.points, xy, arc)
    )
    segments[sid] = replace(segment, points=points)
    report = assess_network(replace(network, segments=tuple(segments)), host)
    assert {"maximum_bypass_fraction", "lateral_span_widths"} <= failures(report)


def test_sections_cannot_fill_in_the_rock_island(example):
    _, host, network, sections = example
    fields = []
    for field in sections.segment_fields:
        samples = tuple(
            replace(
                s,
                tube_width=s.tube_width * 5,
                profile_points=tuple((u * 5, v) for u, v in s.profile_points),
            )
            for s in field.samples
        )
        fields.append(replace(field, samples=samples))
    filled = replace(sections, segment_fields=tuple(fields))
    report = assess_network(network, host, filled)
    assert "section_island_clearance" in failures(report)
    assert "section_footprint_islands" in failures(report)


def test_disconnected_section_fragment_is_rejected(example):
    _, host, network, sections = example
    fields = list(sections.segment_fields)
    field = fields[-1]
    fields[-1] = replace(field, samples=tuple(replace(s, x=s.x + 100) for s in field.samples))
    report = assess_network(network, host, replace(sections, segment_fields=tuple(fields)))
    assert "section_footprint_connected" in failures(report)


def test_blind_tips_do_not_receive_a_second_multiplicative_taper(example):
    _, _, network, sections = example
    fields = {f.segment_id: f for f in sections.segment_fields}
    for segment in network.segments:
        if segment.metadata.get("topology_role") != "side_branch":
            continue
        field = fields[segment.segment_id]
        assert segment.points[-1].width / max(p.width for p in segment.points) <= 0.45
        assert (
            field.samples[-1].tube_width / max(s.tube_width for s in field.samples) <= 0.40 + 1e-9
        )
        assert field.samples[-1].tube_width > 0.6


def test_repairs_preserve_terminal_taper_without_cumulative_pinching(example):
    from plume_advanced.stages.network_acceptance import repair_network

    config, host, network, _ = example
    generator = CaveNetworkGenerator(config.network)
    original = {s.segment_id: s.points[-1].width for s in network.segments}
    for pass_index in range(3):
        network = repair_network(generator, host, network, pass_index)
        for segment in network.segments:
            if segment.metadata.get("topology_role") == "side_branch":
                assert segment.points[-1].width / max(p.width for p in segment.points) <= 0.45
                assert segment.points[-1].width >= 0.5 * original[segment.segment_id]


def test_lost_source_history_is_rejected(example):
    _, host, network, _ = example
    segments = list(network.segments)
    segments[0] = replace(
        segments[0], metadata=dict(segments[0].metadata, contributing_system_ids=[])
    )
    report = assess_network(replace(network, segments=tuple(segments)), host)
    assert "trunk_source_lineage" in failures(report)


@pytest.mark.parametrize("profile", [((float("nan"), 0),) * 3, (1.0, 2.0, 3.0)])
def test_invalid_profile_rejects_without_rasterization_crash(example, profile):
    _, host, network, sections = example
    fields = list(sections.segment_fields)
    samples = list(fields[0].samples)
    samples[0] = replace(samples[0], profile_points=profile)
    fields[0] = replace(fields[0], samples=tuple(samples))
    report = assess_network(network, host, replace(sections, segment_fields=tuple(fields)))
    assert {"section_profiles_finite_positive", "section_footprint_islands"} <= failures(report)


@pytest.mark.parametrize("count", [2, 3])
def test_several_sources_can_coalesce_into_the_dominant_gallery(example, count):
    config, host, _, _ = example
    cfg = replace(config.network, systems=replace(config.network.systems, count=count))
    network = CaveNetworkGenerator(cfg).generate(host, section_config=config.section_field)
    assert len([n for n in network.nodes if n.kind == "entry"]) == count
    assert len([n for n in network.nodes if n.kind == "exit"]) == 1
    assert any(
        s.metadata.get("contributing_system_ids") == list(range(count)) for s in network.segments
    )
    assert network.max_flow_conservation_error() < 1e-12


def test_impossible_feature_density_fails_closed_with_report(example, tmp_path):
    config, host, _, _ = example
    cfg = replace(
        config.network,
        topology=replace(config.network.topology, island_count=(8, 8)),
        quality=replace(config.network.quality, max_attempts=2),
    )
    report_path = tmp_path / "rejected.json"
    with pytest.raises(NetworkQualityError):
        CaveNetworkGenerator(cfg).generate(
            host, section_config=config.section_field, quality_report_path=report_path
        )
    import json

    report = json.loads(report_path.read_text())
    assert report["status"] == "rejected"
    assert not report["accepted"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"style": "valentine_exact"},
        {"island_count": (3, 1)},
        {"island_count": (0, 9)},
        {"side_branch_count": (1.0, 2.0)},
        {"island_length_widths": (0, 3)},
        {"island_half_span_widths": (1, float("nan"))},
        {"width_variation": 1.0},
        {"minimum_single_channel_fraction": 0},
        {"minimum_trunk_fraction": True},
    ],
)
def test_invalid_style_controls_rejected(kwargs):
    with pytest.raises(ValueError, match="network.topology"):
        NetworkTopologyConfig(**kwargs)


def test_topology_config_is_explicit_and_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[network.topology]\nstyle = "trunk_dominated"\nisland_count = [1, 2]\n')
    config = load_project_config(path)
    assert config.network.topology.island_count == (1, 2)
    path.write_text("[network.topology]\nislands = 2\n")
    with pytest.raises(ValueError, match="islands"):
        load_project_config(path)


def test_footprint_raster_is_repeatable(example):
    _, _, network, sections = example
    first, second = [section_footprint(network, sections) for _ in range(2)]
    np.testing.assert_array_equal(first[0], second[0])

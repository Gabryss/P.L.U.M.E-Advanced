"""Failure-oriented morphology tests and deterministic rejection/retry tests."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from plume_advanced.procedural import derive_subseed
from plume_advanced.stages.network import (
    CaveNetwork,
    CaveNetworkConfig,
    CaveNetworkGenerator,
    CaveNode,
    CavePoint,
    CaveSegment,
)
from plume_advanced.stages.network_quality import (
    NetworkQualityConfig,
    NetworkQualityError,
    assess_network,
    assess_sections,
)
from plume_advanced.stages.section_field import SectionFieldGenerator


def network_fixture(coords=None, *, config=None, kind="backbone", widths=None):
    coords = coords if coords is not None else [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)]
    config = config or CaveNetworkConfig(target_route_length_m=100, source_count=1)
    widths = widths if widths is not None else [5] * len(coords)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    nodes = tuple(
        CaveNode(i, *coords[j], float(arc[j]), 0, kind)
        for i, j, kind in ((0, 0, "entry"), (1, -1, "exit"))
    )
    points = tuple(
        CavePoint(
            i, *xy, 100 - 0.01 * s, 1, 30, 0.8, 1, float(s), float(w), 1, 1450 - 0.025 * s, s / 0.35
        )
        for i, (xy, s, w) in enumerate(zip(coords, arc, widths))
    )
    segment = CaveSegment(0, 0, 1, kind, 0, points, {})
    return CaveNetwork(
        config, nodes, (segment,), (), np.ones((2, 2), bool), np.ones((2, 2)), (0, 1), (), (), ()
    )


def failures(report):
    return {c["name"] for c in report["checks"] if not c["passed"]}


def test_straight_valid_network_passes_without_requiring_loops():
    report = assess_network(network_fixture())
    assert report["accepted"]
    assert len(report["checks"]) >= 13


def test_original_inspection_dogleg_is_rejected():
    import json

    case = json.loads(
        (Path(__file__).parent / "fixtures/network_quality/earth_dogleg.json").read_text()
    )
    n = network_fixture(case["xy_m"], widths=case["width_m"])
    assert {"local_turn_angle", "bend_radius_relative_to_width"} <= failures(assess_network(n))


def test_deterministic_repairs_fix_dogleg_without_changing_graph():
    from plume_advanced.stages.host_field import GridConfig, HostFieldConfig, HostFieldGenerator
    from plume_advanced.stages.network_acceptance import repair_network

    host = HostFieldGenerator(
        HostFieldConfig(grid=GridConfig(width=400, height=400, nx=24, ny=24))
    ).generate()
    coords = (
        [(x, 0) for x in range(51)]
        + [(50, y) for y in range(1, 31)]
        + [(x, 30) for x in range(51, 101)]
    )
    n = network_fixture(coords)
    g = CaveNetworkGenerator(n.config)
    first, second = [repair_network(g, host, n, 0) for _ in range(2)]
    assert first.nodes == n.nodes
    assert first.segments == second.segments
    report = assess_network(first, host)
    assert report["accepted"], failures(report)


def test_quality_can_only_be_bypassed_explicitly():
    config = CaveNetworkConfig(quality=NetworkQualityConfig(enabled=False))
    raw = network_fixture(config=config, widths=[12] * 5)
    with patch.object(CaveNetworkGenerator, "_generate_candidate", return_value=raw) as builder:
        assert CaveNetworkGenerator(config).generate(None) is raw
    assert builder.call_count == 1
    assert not raw.quality_report


def test_section_failure_retries_even_when_network_passes():
    config = CaveNetworkConfig(
        random_seed=17,
        target_route_length_m=100,
        source_count=1,
        quality=NetworkQualityConfig(max_attempts=2, repair_passes=0),
    )
    real_generate = SectionFieldGenerator.generate

    def make_sections(generator, network):
        sections = real_generate(generator, network)
        if network.config.random_seed == 17:
            field = sections.segment_fields[0]
            sections = replace(
                sections,
                segment_fields=(
                    replace(
                        field, samples=tuple(replace(s, z=s.z + 2 * s.x) for s in field.samples)
                    ),
                ),
            )
        return sections

    with (
        patch.object(
            CaveNetworkGenerator,
            "_generate_candidate",
            lambda worker, host: network_fixture(config=worker.config),
        ),
        patch.object(SectionFieldGenerator, "generate", make_sections),
    ):
        result = CaveNetworkGenerator(config).generate(
            None, section_config=SectionFieldGenerator().config
        )
    assert result.quality_report["selected_attempt"] == 1
    assert "section_grade" in failures(result.quality_report["attempts"][0])


def test_dogleg_rejected_at_both_sampling_densities():
    for step in (1, 2):
        coords = [(x, 0) for x in range(0, 51, step)]
        coords += [(50, y) for y in range(step, 31, step)]
        coords += [(x, 30) for x in range(50 + step, 101, step)]
        report = assess_network(network_fixture(coords))
        assert {"local_turn_angle", "bend_radius_relative_to_width"} <= failures(report)


def test_smooth_but_too_tight_hairpin_is_rejected():
    theta = np.linspace(0, np.pi, 101)
    coords = np.column_stack((3 * np.cos(theta), 3 * np.sin(theta)))
    report = assess_network(network_fixture(coords))
    assert "bend_radius_relative_to_width" in failures(report)
    assert "local_turn_angle" not in failures(report)


def test_large_rounded_dogleg_fails_route_scale_check():
    from scipy.ndimage import gaussian_filter1d
    coords = np.array([(x, 0) for x in range(101)] +
                      [(100, y) for y in range(1, 121)] +
                      [(x, 120) for x in range(101, 501)], dtype=float)
    coords = gaussian_filter1d(coords, 12, axis=0)
    report = assess_network(network_fixture(coords))
    assert "local_turn_angle" not in failures(report)
    assert "bend_radius_relative_to_width" not in failures(report)
    assert "arterial_transverse_runs" in failures(report)


def test_blind_terminal_and_saturated_widths_are_rejected():
    n = network_fixture(kind="stalled_lobe", widths=[12] * 5)
    assert {"blind_branch_taper", "width_cap_saturation"} <= failures(assess_network(n))
    tapered = network_fixture(kind="stalled_lobe", widths=[5, 5, 5, 4, 1.5])
    assert "blind_branch_taper" not in failures(assess_network(tapered))


def test_self_crossing_and_broken_endpoint_are_rejected():
    n = network_fixture([(0, 0), (75, 30), (25, 30), (100, 0)])
    assert "unmodeled_plan_crossings" in failures(assess_network(n))
    broken = replace(n, nodes=(replace(n.nodes[0], x=-10), n.nodes[1]))
    assert "graph_endpoint_agreement" in failures(assess_network(broken))


def test_width_jump_and_short_route_are_rejected():
    n = network_fixture([(0, 0), (1, 0), (2, 0)], widths=[2, 5, 2])
    assert {"width_gradient", "minimum_route_extent"} <= failures(assess_network(n))


def test_final_sections_are_screened_after_generation():
    n = network_fixture()
    sections = SectionFieldGenerator().generate(n)
    f = sections.segment_fields[0]
    broken = replace(f, samples=tuple(replace(s, z=s.z + 2 * s.x) for s in f.samples))
    assert "section_grade" in failures(
        assess_sections(n, replace(sections, segment_fields=(broken,)))
    )
    broken = replace(f, samples=(replace(f.samples[0], normal=(0, 0, 0)), *f.samples[1:]))
    assert "section_frames_orthonormal" in failures(
        assess_sections(n, replace(sections, segment_fields=(broken,)))
    )


def test_deterministic_retry_selects_same_candidate_and_report(tmp_path):
    config = CaveNetworkConfig(
        random_seed=17,
        target_route_length_m=100,
        source_count=1,
        quality=NetworkQualityConfig(max_attempts=3, repair_passes=0),
    )
    seeds = []

    def candidate(worker, host):
        seeds.append(worker.config.random_seed)
        return network_fixture(
            config=worker.config, widths=[12] * 5 if worker.config.random_seed == 17 else [5] * 5
        )

    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    with patch.object(CaveNetworkGenerator, "_generate_candidate", candidate):
        results = [
            CaveNetworkGenerator(config).generate(None, quality_report_path=p) for p in paths
        ]
    expected = derive_subseed(17, "network-quality-v1", 1)
    assert seeds == [17, expected, 17, expected]
    assert results[0].quality_report == results[1].quality_report
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert results[0].quality_report["selected_attempt"] == 1


def test_exhaustion_fails_closed_and_writes_report(tmp_path):
    config = CaveNetworkConfig(
        target_route_length_m=100, quality=NetworkQualityConfig(max_attempts=2, repair_passes=0)
    )
    with patch.object(
        CaveNetworkGenerator,
        "_generate_candidate",
        lambda worker, host: network_fixture(config=worker.config, widths=[12] * 5),
    ):
        with pytest.raises(NetworkQualityError) as error:
            CaveNetworkGenerator(config).generate(
                None, quality_report_path=tmp_path / "failed.json"
            )
    assert error.value.report["status"] == "rejected"
    assert len(error.value.report["attempts"]) == 2
    assert (tmp_path / "failed.json").exists()


def test_mesh_gate_runs_before_allocating_volume():
    from plume_advanced.stages.geometry import GeometryGenerator

    n = network_fixture(widths=[12] * 5)
    sections = SectionFieldGenerator().generate(n)
    with patch.object(
        GeometryGenerator, "_refine_profile_chain", side_effect=AssertionError("meshing started")
    ):
        with pytest.raises(NetworkQualityError):
            GeometryGenerator().build_base_volume(n, sections)


def test_mesh_gate_rejects_sections_changed_after_network_acceptance():
    from plume_advanced.stages.geometry import GeometryGenerator

    n = network_fixture()
    assert assess_network(n)["accepted"]
    sections = SectionFieldGenerator().generate(n)
    field = sections.segment_fields[0]
    field = replace(field, samples=tuple(replace(s, z=s.z + 2 * s.x) for s in field.samples))
    with pytest.raises(NetworkQualityError, match="section_grade"):
        GeometryGenerator().build_base_volume(n, replace(sections, segment_fields=(field,)))


def test_parallel_passage_overlap_uses_actual_vertical_clearance():
    n = network_fixture(config=CaveNetworkConfig(target_route_length_m=100, source_flux=2))
    branch = network_fixture([(0, 0), (25, 2), (50, 2), (75, 2), (100, 0)]).segments[0]
    n = replace(n, segments=(n.segments[0], replace(branch, segment_id=1, z_level=1)))
    sections = SectionFieldGenerator().generate(n)
    # Force equal elevations; a level label alone must never prove separation.
    first, second = sections.segment_fields
    first_floor = first.samples[0].floor_world_z

    def positioned(field, floor):
        return replace(
            field,
            samples=tuple(
                replace(s, floor_world_z=floor, roof_world_z=floor + 3, z=floor + 1.5)
                for s in field.samples
            ),
        )

    overlapping = replace(
        sections, segment_fields=(positioned(first, first_floor), positioned(second, first_floor))
    )
    assert "nonlocal_passage_overlap" in failures(assess_sections(n, overlapping))
    separated = replace(
        sections,
        segment_fields=(positioned(first, first_floor), positioned(second, first_floor + 10)),
    )
    assert "nonlocal_passage_overlap" not in failures(assess_sections(n, separated))


def test_near_identical_independent_bows_are_flagged_but_same_lobe_is_not():
    segments, nodes = [], []
    for i in range(3):
        x = np.linspace(0, 100, 33)
        coords = np.column_stack((x + 100 * i, 20 * np.sin(np.pi * x / 100)))
        s = network_fixture(coords, widths=[2] * len(coords), kind="anastomosis").segments[0]
        segments.append(
            replace(
                s,
                segment_id=i,
                start_node_id=i,
                end_node_id=i + 1,
                metadata={"lobe_path_id": str(i)},
            )
        )
        nodes.append(CaveNode(i, 100 * i, 0, 100 * i, 0, "entry" if i == 0 else "junction"))
    nodes.append(CaveNode(3, 300, 0, 300, 0, "exit"))
    n = replace(
        network_fixture(),
        nodes=tuple(nodes),
        segments=tuple(segments),
        dominant_route_node_ids=(0, 1, 2, 3),
    )
    assert "repeated_loop_shapes" in failures(assess_network(n))
    n = replace(
        n, segments=tuple(replace(s, metadata={"lobe_path_id": "one_lobe"}) for s in segments)
    )
    assert "repeated_loop_shapes" not in failures(assess_network(n))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_attempts=0),
        dict(max_attempts=1000),
        dict(repair_passes=-1),
        dict(enabled="false"),
        dict(maximum_turn_degrees=float("nan")),
    ],
)
def test_invalid_quality_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        NetworkQualityConfig(**kwargs)


def test_config_reads_nested_quality_and_rejects_misspelled_key(tmp_path):
    from plume_advanced.config import load_project_config

    path = tmp_path / "test.toml"
    path.write_text("[network.quality]\nmax_attempts = 3\nrepair_passes = 1\n")
    loaded = load_project_config(path)
    assert loaded.network.quality.max_attempts == 3
    assert loaded.network.quality.enabled
    path.write_text("[network.quality]\nmax_atempts = 3\n")
    with pytest.raises(ValueError):
        load_project_config(path)

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
from PIL import Image

from plume_advanced.evaluation.visualization.dashboard import (
    _route_profile_data,
    dashboard_payload,
    render_diagnostic_dashboard,
)
from plume_advanced.stages.network import CaveNetwork, CaveNetworkConfig, CaveNode, CavePoint, CaveSegment
from plume_advanced.config import load_project_config
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.stages.section_field import SectionField, SectionFieldConfig, SegmentSectionField, SectionSample
from plume_advanced.evaluation.metrics.sections import section_longitudinal_continuity
from plume_advanced.evaluation.metrics.network import network_metrics


def _network(density: float = 1.0) -> CaveNetwork:
    nodes = (CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"), CaveNode(1, 20.0, 0.0, 20.0, 0.0, "exit"))
    points = tuple(CavePoint(i, float(i) * 10.0, float(i % 2), float(i), 0.0, 2.0, 1.0, 0.0, float(i) * 10.0, 4.0, 1.0, 1400.0, float(i)) for i in range(3))
    segment = CaveSegment(0, 0, 1, "backbone", 0, points, {"emplacement_phase_count": 3})
    return CaveNetwork(CaveNetworkConfig(network_density=density), nodes, (segment,), (), np.zeros((2, 2), bool), np.zeros((2, 2)), (0, 1), (), (), ())


def test_dashboard_payload_is_deterministic_and_density_fields_ordered() -> None:
    base = {"node_count": 4, "segment_count": 6, "edge_count": 6, "main_route_length_m": 100.0, "cyclomatic_number": 2, "stacked_segment_count": 2, "vertical_capture_count": 1, "split_merge_region_count": 2, "connected_component_count": 1, "zero_flux_segment_count": 0, "normalized_topology": {"branch_segment_fraction": 0.5}}
    density = {3.0: [base, {**base, "cyclomatic_number": 4}], 0.0: {**base, "segment_count": 1, "edge_count": 1, "cyclomatic_number": 0, "stacked_segment_count": 0, "vertical_capture_count": 0, "split_merge_region_count": 0, "normalized_topology": {"branch_segment_fraction": 0.0}}}
    first = dashboard_payload(_network(), density_sweep=density)
    second = dashboard_payload(_network(), density_sweep={key: density[key] for key in reversed(list(density))})
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [row["density"] for row in first["density_sweep"]] == [0.0, 3.0]
    assert first["density_sweep"][1]["cyclomatic_per_km_p90"] > first["density_sweep"][1]["cyclomatic_per_km_p10"]
    assert first["sections"]["pdc_label"].startswith("generated-only")


def test_dashboard_png_and_json_sidecar_are_valid(tmp_path: Path) -> None:
    output, sidecar = render_diagnostic_dashboard(_network(), output_path=tmp_path / "dashboard.png", provenance={"seed": 7, "world": "earth"})
    assert output.is_file() and sidecar is not None and sidecar.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
    payload = json.loads(sidecar.read_text())
    assert payload["schema"] == "plume.evaluation-dashboard.v1"
    assert payload["provenance"]["seed"] == 7


def test_dashboard_rejects_non_calibration_pdc_partition() -> None:
    try:
        dashboard_payload(_network(), pdc_calibration={"partition": "evaluation", "records": []})
    except ValueError as error:
        assert "calibration" in str(error)
    else:
        raise AssertionError("confirmatory PDC partition was accepted")


def test_dashboard_pdc_panel_reports_calibration_quartiles() -> None:
    calibration = {
        "partition": "calibration",
        "records": [
            {"aspect_ratio": 1.0, "compactness": 0.6, "floor_residual_norm": 0.1, "roof_asymmetry_norm": 0.2},
            {"aspect_ratio": 1.5, "compactness": 0.8, "floor_residual_norm": 0.2, "roof_asymmetry_norm": 0.3},
            {"aspect_ratio": 2.0, "compactness": 0.9, "floor_residual_norm": 0.3, "roof_asymmetry_norm": 0.4},
        ],
    }
    payload = dashboard_payload(_network(), pdc_calibration=calibration)
    summary = payload["sections"]["pdc_calibration"]
    assert payload["sections"]["pdc_label"] == "PDC calibration caves"
    for feature in ("aspect_ratio", "compactness", "floor_residual", "roof_asymmetry"):
        assert summary[feature]["q1"] is not None
        assert summary[feature]["median"] is not None
        assert summary[feature]["q3"] is not None


def test_uphill_string_provenance_marks_unresolved_backbone() -> None:
    network = _network()
    segment = replace(network.segments[0], metadata={"grade_profile": "uphill_unresolved"})
    network = replace(network, segments=(segment,))
    report = network_metrics(network)
    assert report["uphill_provenance_by_kind"]["backbone"] == {"uphill_unresolved": 1}


def test_dashboard_accepts_bounded_seeded_stage_b_c_fixture(tmp_path: Path) -> None:
    project = load_project_config(Path("config/project.toml"))
    host_config = replace(
        project.host_field,
        grid=replace(project.host_field.grid, nx=60, ny=48, width=1800.0, height=1400.0),
    )
    host = HostFieldGenerator(host_config).generate()
    network_config = replace(
        project.network,
        random_seed=2,
        source_count=2,
        target_route_length_m=800.0,
        trace_max_steps=100,
        spur_count=1,
    )
    network = CaveNetworkGenerator(network_config).generate(host)
    section_config = replace(
        project.section_field,
        profile_resolution=12,
        maximum_sample_spacing=80.0,
        minimum_sample_spacing=40.0,
    )
    section_field = SectionFieldGenerator(section_config).generate(network)
    payload = dashboard_payload(network, section_field)
    assert payload["sections"]["continuity"]["segments"]
    output, _ = render_diagnostic_dashboard(network, section_field, tmp_path / "seeded.png")
    assert output.is_file()
    assert len(network.segments) > 0
    assert sum(len(item.samples) for item in section_field.segment_fields) > 0


def test_route_profile_follows_node_order_and_reports_physical_grade() -> None:
    nodes = (
        CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"),
        CaveNode(1, 10.0, 0.0, 10.0, 0.0, "junction"),
        CaveNode(2, 20.0, 0.0, 20.0, 0.0, "exit"),
    )

    def make(segment_id, start, end, elevations):
        points = tuple(
            CavePoint(
                index,
                start * 10.0 + index * 10.0,
                0.0,
                elevation,
                0.0,
                1.0,
                1.0,
                0.0,
                index * 10.0,
                2.0,
                1.0,
                1400.0,
                float(index),
            )
            for index, elevation in enumerate(elevations)
        )
        return CaveSegment(segment_id, start, end, "backbone", 0, points, {})

    network = CaveNetwork(
        CaveNetworkConfig(),
        nodes,
        (make(99, 0, 1, (10.0, 11.0)), make(1, 1, 2, (11.0, 9.0))),
        (),
        np.zeros((2, 2), bool),
        np.zeros((2, 2)),
        (0, 1, 2),
        (),
        (),
        (),
    )
    arc, elevation, grade = _route_profile_data(network)
    assert np.allclose(arc, (0.0, 10.0, 20.0))
    assert np.allclose(elevation, (10.0, 11.0, 9.0))
    assert grade[1] > 0.0 and grade[2] < 0.0


def test_section_continuity_keeps_duplicate_local_arc_origins_segment_local() -> None:
    profile = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0))

    def sample(segment_id, index, width):
        return SectionSample(
            index,
            segment_id,
            float(index * 10),
            float(index),
            0.0,
            0.0,
            1.0,
            2.0,
            1.0,
            1.0,
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            width,
            1.0,
            0.2,
            1.0,
            0.0,
            0.0,
            (),
            profile,
        )

    field = SectionField(
        SectionFieldConfig(),
        (
            SegmentSectionField(1, (), (sample(1, 0, 2.0), sample(1, 1, 2.2))),
            SegmentSectionField(2, (), (sample(2, 0, 9.0), sample(2, 1, 9.2))),
        ),
        (1, 2),
    )
    report = section_longitudinal_continuity(field)
    assert set(report["segments"]) == {"1", "2"}
    assert report["segments"]["1"]["metrics"]["width"]["count"] == 2
    assert report["segments"]["2"]["metrics"]["width"]["count"] == 2

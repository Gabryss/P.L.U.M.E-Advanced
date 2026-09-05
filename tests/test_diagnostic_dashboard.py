from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from plume_advanced.evaluation.visualization.dashboard import dashboard_payload, render_diagnostic_dashboard
from plume_advanced.stages.network import CaveNetwork, CaveNetworkConfig, CaveNode, CavePoint, CaveSegment


def _network(density: float = 1.0) -> CaveNetwork:
    nodes = (CaveNode(0, 0.0, 0.0, 0.0, 0.0, "entry"), CaveNode(1, 20.0, 0.0, 20.0, 0.0, "exit"))
    points = tuple(CavePoint(i, float(i) * 10.0, float(i % 2), float(i), 0.0, 2.0, 1.0, 0.0, float(i) * 10.0, 4.0, 1.0, 1400.0, float(i)) for i in range(3))
    segment = CaveSegment(0, 0, 1, "backbone", 0, points, {"emplacement_phase_count": 3})
    return CaveNetwork(CaveNetworkConfig(network_density=density), nodes, (segment,), (), np.zeros((2, 2), bool), np.zeros((2, 2)), (0, 1), (), (), ())


def test_dashboard_payload_is_deterministic_and_density_fields_ordered() -> None:
    density = {3.0: {"node_count": 4, "segment_count": 6, "edge_count": 6, "main_route_length_m": 100.0, "cyclomatic_number": 2, "stacked_segment_count": 2, "vertical_capture_count": 1, "split_merge_region_count": 2, "connected_component_count": 1, "zero_flux_segment_count": 0, "normalized_topology": {"branch_segment_fraction": 0.5}}, 0.0: {"node_count": 2, "segment_count": 1, "edge_count": 1, "main_route_length_m": 100.0, "cyclomatic_number": 0, "stacked_segment_count": 0, "vertical_capture_count": 0, "split_merge_region_count": 0, "connected_component_count": 1, "zero_flux_segment_count": 0, "normalized_topology": {"branch_segment_fraction": 0.0}}}
    first = dashboard_payload(_network(), density_sweep=density)
    second = dashboard_payload(_network(), density_sweep={key: density[key] for key in reversed(list(density))})
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [row["density"] for row in first["density_sweep"]] == [0.0, 3.0]
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

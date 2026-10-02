"""Portable viewers must preserve true network geometry and publish only selected data."""

import copy
import io
import json
import re
from types import SimpleNamespace

import pytest

from plume_advanced.network_viewer import (
    export_network_viewer,
    main,
    network_files,
    page_handler,
    viewer_network,
)


@pytest.fixture
def artifact():
    return {
        "schema": "plume.cave-network.v1",
        "layers": {"controls": {"enabled": True}},
        "nodes": [{"node_id": 3, "kind": "entry"}, {"node_id": 9, "kind": "terminal"}],
        "segments": [{
            "segment_id": 7, "source_node_id": 3, "target_node_id": 9,
            "physical_length_m": 10., "z_level": 0,
            "centerline": [{"x": 0., "y": 0., "elevation": 200., "width": 2.},
                           {"x": 6., "y": 0., "elevation": 199., "width": 3.}],
            "centerline_xyz_m": [[0., 0., 150.], [6., 0., 142.]],
            "metadata": {"regional_start_layer": 0, "regional_end_layer": 2,
                         "regional_route_type": "layer_ramp"},
        }],
    }


def test_viewer_preserves_samples_and_actual_layer_elevation(artifact):
    original = copy.deepcopy(artifact)
    result = viewer_network(artifact)
    assert result["segments"][0]["xyz"] == [[0., 0., 150.], [6., 0., 142.]]
    assert result["segments"][0]["widths"] == [2., 3.]
    assert result["segments"][0]["source"] == 3
    assert result["segments"][0]["target"] == 9
    assert result["layers"] == [0, 2]
    assert artifact == original


def test_missing_xyz_is_only_allowed_without_layers(artifact):
    del artifact["segments"][0]["centerline_xyz_m"]
    with pytest.raises(ValueError, match="Layered networks require"):
        viewer_network(artifact)
    artifact["layers"]["controls"]["enabled"] = False
    result = viewer_network(artifact)
    assert result["segments"][0]["xyz"][0][2] == 200.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "150"])
def test_invalid_coordinates_are_rejected(artifact, value):
    artifact["segments"][0]["centerline_xyz_m"][0][2] = value
    with pytest.raises(ValueError, match="finite numbers"):
        viewer_network(artifact)


@pytest.mark.parametrize("field,value", [
    ("source_node_id", 999), ("target_node_id", True), ("physical_length_m", -1),
    ("centerline_xyz_m", [[0, 0, 0]]), ("centerline_xyz_m", [[0, 0], [0, 1]]),
])
def test_invalid_passages_are_rejected(artifact, field, value):
    artifact["segments"][0][field] = value
    with pytest.raises(ValueError):
        viewer_network(artifact)


@pytest.mark.parametrize("width", [0, -1, float("nan")])
def test_invalid_widths_are_rejected(artifact, width):
    artifact["segments"][0]["centerline"][0]["width"] = width
    with pytest.raises(ValueError):
        viewer_network(artifact)


@pytest.mark.parametrize("collection", ["nodes", "segments"])
def test_duplicate_identifiers_are_rejected(artifact, collection):
    artifact[collection].append(copy.deepcopy(artifact[collection][0]))
    with pytest.raises(ValueError, match="unique integers"):
        viewer_network(artifact)


def test_selected_campaign_only_includes_accepted_cases_once(tmp_path):
    case = tmp_path / "accepted"
    case.mkdir()
    (case / "network.json").write_text("{}")
    (tmp_path / "campaign.json").write_text(json.dumps({"cases": [
        {"case": "accepted", "accepted": True}, {"case": "rejected", "accepted": False}]}))
    assert network_files([tmp_path, case]) == [case / "network.json"]


@pytest.mark.parametrize("case", ["../outside", "/outside"])
def test_campaign_cannot_publish_files_outside_selected_directory(tmp_path, case):
    (tmp_path / "campaign.json").write_text(json.dumps({"cases": [{"case": case, "accepted": True}]}))
    with pytest.raises(ValueError, match="inside"):
        network_files([tmp_path])


def test_empty_campaign_is_not_silently_published(tmp_path):
    (tmp_path / "campaign.json").write_text('{"cases": []}')
    with pytest.raises(ValueError, match="No accepted"):
        network_files([tmp_path])


def test_self_contained_html_keeps_untrusted_text_as_data(tmp_path, artifact):
    malicious = '</script><script>alert("injection")</script>'
    artifact["segments"][0]["metadata"]["regional_route_type"] = malicious
    (tmp_path / "network.json").write_text(json.dumps(artifact))
    (tmp_path / "resolved_config.json").write_text(json.dumps({
        "procedural_seed": 23, "host_field": {"flow_angle_degrees": 45},
        "private_unused_path": "/private/must-not-be-published"}))
    (tmp_path / "quality.json").write_text('{"accepted": true}')
    path = export_network_viewer([tmp_path], tmp_path / "viewer.html", title=malicious)
    content = path.read_text()
    embedded = re.search(r'<script id="network-data" type="application/json">(.*?)</script>', content).group(1)
    network = json.loads(embedded)["networks"][0]
    assert network["segments"][0]["role"] == malicious
    assert network["flowAngle"] == 45 and network["seed"] == 23 and network["quality"] is True
    assert malicious not in content and "/private/must-not-be-published" not in content
    assert "__DATA__" not in content and "__APP__" not in content
    assert '<script src=' not in content and '<link ' not in content
    assert "canvas.getContext('webgl'" in content


@pytest.mark.parametrize("path,code", [
    ("/", 200), ("/index.html?ignored=1", 200), ("/viewer.html", 200),
    ("/network.json", 404), ("/../private", 404), ("/%2e%2e/private", 404),
])
def test_http_handler_serves_only_the_selected_page(tmp_path, path, code):
    page = tmp_path / "index.html"
    page.write_bytes(b"the viewer")
    handler = object.__new__(page_handler(page))
    handler.path, handler.wfile = path, io.BytesIO()
    statuses, headers = [], []
    handler.send_response = statuses.append
    handler.send_error = statuses.append
    handler.send_header = lambda *values: headers.append(values)
    handler.end_headers = lambda: None
    handler.do_GET()
    assert statuses == [code]
    assert handler.wfile.getvalue() == (b"the viewer" if code == 200 else b"")
    if code == 200:
        assert ("X-Content-Type-Options", "nosniff") in headers


def test_cli_builds_page_without_starting_server(tmp_path, artifact, capsys):
    source, output = tmp_path / "network.json", tmp_path / "page/index.html"
    source.write_text(json.dumps(artifact))
    assert main(["--source", str(source), "--output", str(output)]) == 0
    assert output.is_file() and "Offline 3D viewer" in capsys.readouterr().out


def test_network_generation_automatically_writes_viewer(tmp_path, artifact, monkeypatch):
    from plume_advanced import network_cli
    from plume_advanced.visualization import regional

    host = object()
    network = SimpleNamespace(config=SimpleNamespace(random_seed=12),
                              summary=lambda: {}, backend_provenance={})
    monkeypatch.setattr(network_cli.HostFieldGenerator, "generate", lambda self: host)
    monkeypatch.setattr(network_cli.CaveNetworkGenerator, "generate", lambda *args, **kwargs: network)
    monkeypatch.setattr(network_cli, "host_semantic_hash", lambda value: "host-hash")
    monkeypatch.setattr(network_cli, "export_network_artifact", lambda value, path: path.write_text(json.dumps(artifact)))
    monkeypatch.setattr(regional, "render_network_comparison", lambda *args, **kwargs: None)
    assert network_cli.main(["--config", "config/short-single.toml", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "viewer.html").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["scope"] == "host_and_network_only"

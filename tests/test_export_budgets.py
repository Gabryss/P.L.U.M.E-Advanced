"""Budget limits fail before publishing and optional collider work is skipped."""

import json
from unittest.mock import patch

import pytest
from test_performance_regressions import _tiny_geometry

from plume_advanced.exporters import export_target_asset, prepare_export_scene
from plume_advanced.world import ExportConfig


def test_disabled_collision_skips_clustering():
    with patch(
        "plume_advanced.exporters.scene.simplified_collision_arrays",
        side_effect=AssertionError("unneeded"),
    ):
        scene = prepare_export_scene(_tiny_geometry(), generate_collision=False)
    assert scene.collision_faces.shape == (0, 3)


def test_triangle_budget_checked_before_surface_preparation(tmp_path):
    with patch(
        "plume_advanced.exporters.targets.prepare_export_scene",
        side_effect=AssertionError("unneeded"),
    ):
        with pytest.raises(ValueError, match="triangles"):
            export_target_asset(
                _tiny_geometry(), ExportConfig(max_visual_triangles=1), tmp_path / "export"
            )
    assert not (tmp_path / "export").exists()


def test_size_budget_failure_does_not_publish_partial_package(tmp_path):
    with pytest.raises(ValueError, match="size budget"):
        export_target_asset(_tiny_geometry(), ExportConfig(max_asset_bytes=16), tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not list(tmp_path.iterdir())


def test_export_report_measures_buffers_and_bytes(tmp_path):
    result = export_target_asset(
        _tiny_geometry(), ExportConfig(generate_collision=False), tmp_path / "export"
    )
    path = tmp_path / "export/export_size_report.json"
    report = json.loads(path.read_text())
    assert path in result.files
    assert report["visual_triangles"] == 4 and report["collision_triangles"] == 0
    assert report["cave_vertex_and_index_bytes"] > 0
    for file, size in report["files"].items():
        assert (tmp_path / "export" / file).stat().st_size == size

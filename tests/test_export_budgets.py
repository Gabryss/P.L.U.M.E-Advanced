"""Budget limits fail before publishing and optional collider work is skipped."""

import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import trimesh
from test_embedded_inspection import geometry
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


def test_dense_corridor_reduction_preserves_clearance_and_source():
    mesh = trimesh.creation.box(extents=(20, 4, 2))
    for _ in range(3):
        mesh = mesh.subdivide()
    cave = geometry(mesh, smoothing=0)
    before = np.asarray(cave.assembled_vertices).copy()
    cave = replace(cave, config=replace(cave.config,
        required_route_height_m=.5, required_route_width_m=.5),
        required_route_paths=(((-8., 0., 0.), (8., 0., 0.)),), route_path_segment_ids=(0,))
    scene = prepare_export_scene(cave, generate_collision=True,
                                max_visual_triangles=200, visual_max_error_m=.01)
    inspection = scene.inspection
    assert len(scene.canonical_visual["faces"]) <= 200 < len(mesh.faces)
    assert inspection["visual_reduction"]["enabled"]
    assert not inspection["visual_reduction"]["used_raw_fallback"]
    assert inspection["visual_attempts"][-1]["surface_deviation"]["passed"]
    assert inspection["visual"]["traversal"]["passed"]
    assert inspection["collision"]["inspection"]["traversal"]["passed"]
    np.testing.assert_array_equal(before, cave.assembled_vertices)


def test_reduced_surface_roundtrips_through_published_package(tmp_path):
    cave = geometry(trimesh.creation.icosphere(subdivisions=4, radius=2), smoothing=0)
    export_target_asset(cave, ExportConfig(max_visual_triangles=1500,
        visual_max_error_m=.01), tmp_path / "export")
    inspection = json.loads((tmp_path / "export/pipeline_inspection.json").read_text())
    size = json.loads((tmp_path / "export/export_size_report.json").read_text())
    assert size["visual_triangles"] <= 1500 < len(cave.assembled_faces)
    assert inspection["serialized"]["passed"]
    assert inspection["visual_reduction"]["attempts"][-1]["surface_deviation"]["passed"]


def test_unattainable_visual_budget_cannot_publish_an_inaccurate_mesh(tmp_path):
    cave = geometry(trimesh.creation.icosphere(subdivisions=2, radius=2), smoothing=0)
    with pytest.raises(ValueError, match="budget"):
        export_target_asset(cave, ExportConfig(max_visual_triangles=12,
            visual_max_error_m=.000001), tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("error", [-1., float("inf"), float("nan"), True, "0.1"])
def test_visual_error_budget_requires_finite_metres(error):
    with pytest.raises(ValueError, match="visual_max_error_m"):
        ExportConfig(visual_max_error_m=error)


def test_invalid_smoothing_is_rejected_before_expensive_uv_work(monkeypatch):
    from plume_advanced.stages import geometry_export

    cave = geometry(trimesh.creation.icosphere(subdivisions=2, radius=2), smoothing=2)
    atlas = geometry_export._xatlas_metric_uvs
    calls = []

    def wrong_smoothing(vertices, faces, *, iterations, **kwargs):
        return vertices + [0., 0., 1.] if iterations else vertices

    def measured_atlas(*args, **kwargs):
        calls.append(True)
        return atlas(*args, **kwargs)

    monkeypatch.setattr(geometry_export, '_smooth_visual_surface', wrong_smoothing)
    monkeypatch.setattr(geometry_export, '_xatlas_metric_uvs', measured_atlas)
    scene = prepare_export_scene(cave, generate_collision=False, visual_max_error_m=.01)
    assert len(calls) == 1
    assert not scene.inspection['visual_attempts'][0]['geometry_preflight']['passed']
    assert scene.inspection['visual_attempts'][-1]['surface_deviation']['passed']
    assert scene.geometry.config.cave_smoothing_iterations == 0


def test_successful_preflight_cannot_replace_full_surface_error_checks(monkeypatch):
    from plume_advanced.exporters import scene as scene_module
    from plume_advanced.stages import geometry_export

    cave = geometry(trimesh.creation.icosphere(subdivisions=2, radius=2), smoothing=1)
    calls = []

    def reject_final(*args, **kwargs):
        calls.append(True)
        return dict(passed=len(calls) == 1, max_sampled_error_m=.02, checked_points=1)

    monkeypatch.setattr(scene_module, 'surface_deviation', reject_final)
    monkeypatch.setattr(geometry_export, '_smooth_visual_surface', lambda vertices, *args, **kwargs: vertices)
    with pytest.raises(ValueError, match='All bounded visual repairs failed') as caught:
        prepare_export_scene(cave, generate_collision=False, visual_max_error_m=.01)
    assert caught.value.report['attempts'][0]['geometry_preflight']['passed']
    assert len(calls) == 3  # Preflight never replaces the check after UV preparation.

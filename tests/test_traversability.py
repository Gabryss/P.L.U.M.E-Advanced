"""Analytic terrain fixtures and exports exercise the ground-truth contract."""

import json
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from PIL import Image

from plume_advanced.traversability.config import TraversabilityConfig
from plume_advanced.traversability.export import TraversabilityBudgetError, export_traversability
from plume_advanced.traversability.raster import (
    cavity_fields,
    classify,
    obstacle_cells,
    vertical_hits,
)
from plume_advanced.traversability.request import from_paths


def box(z=0):
    mesh = trimesh.creation.box(extents=(8, 4, 2))
    mesh.apply_translation((0, 0, z + 1))
    return mesh


def request(layers=1, config=None, ramp=False):
    paths = {
        i: np.array([[-3, 0, 1 + i * 6, 4, 2], [3, 0, 1 + i * 6, 4, 2]], float)
        for i in range(layers)
    }
    segments = [
        (i, 2 * i, 2 * i + 1, dict(regional_start_layer=i, regional_end_layer=i))
        for i in range(layers)
    ]
    if ramp:
        paths[9] = np.array([[3, 0, 1, 4, 2], [-3, 0, 7, 4, 2]], float)
        segments.append((9, 1, 2, dict(regional_start_layer=0, regional_end_layer=1)))
    return from_paths(
        config or TraversabilityConfig(), segments, paths, layer_count=layers, layered=layers > 1
    )


def fields(mesh, z):
    origin = np.array([-4.0, -2.0])
    pixels, heights = vertical_hits(mesh.vertices, mesh.faces, origin, (16, 32), 0.25)
    return cavity_fields(pixels, heights, np.full((16, 32), z))


def test_flat_floor_passes_and_walls_exclude_the_footprint():
    floor, roof, uncertain = fields(box(), 1.0)
    np.testing.assert_allclose(floor, 0)
    np.testing.assert_allclose(roof, 2)
    result = classify(floor, roof, uncertain, np.zeros(floor.shape, bool), TraversabilityConfig())
    assert (result["status"][4:-4, 4:-4] == 1).all()
    assert (result["status"][0] == 2).all()
    assert result["reason_bits"][0, 16] & 1
    np.testing.assert_allclose(result["slope_deg"][4:-4, 4:-4], 0, atol=1e-7)


def test_stacked_cavities_are_separate_even_at_identical_xy():
    mesh = trimesh.util.concatenate([box(), box(6)])
    floor0, roof0, _ = fields(mesh, 1.0)
    floor1, roof1, _ = fields(mesh, 7.0)
    np.testing.assert_allclose(floor0, 0)
    np.testing.assert_allclose(roof0, 2)
    np.testing.assert_allclose(floor1, 6)
    np.testing.assert_allclose(roof1, 8)
    # A reference in solid rock must not attach to either cave.
    floor, roof, _ = fields(mesh, 4.0)
    assert np.isnan(floor).all() and np.isnan(roof).all()


def test_shared_triangle_edges_and_face_order_do_not_change_rays():
    mesh = box()
    a = fields(mesh, 1.0)
    mesh.faces = mesh.faces[::-1, ::-1]
    b = fields(mesh, 1.0)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("angle,passed", [(0, True), (15, True), (25, False)])
def test_ramps_are_not_mislabelled_as_steps(angle, passed):
    y, x = np.mgrid[:32, :64] * 0.1
    floor = x * np.tan(np.radians(angle))
    c = TraversabilityConfig(resolution_m=0.1)
    result = classify(floor, floor + 2, np.zeros(floor.shape, bool), np.zeros(floor.shape, bool), c)
    assert (result["status"][12:-12, 12:-12] == (1 if passed else 2)).all()
    np.testing.assert_allclose(result["slope_deg"][12:-12, 12:-12], angle, atol=1e-4)
    assert result["step_m"][16, 32] < 1e-6


def test_step_and_headroom_failures_have_distinct_reasons():
    floor = np.zeros((40, 80))
    floor[:, 40:] = 0.3
    roof = floor + 2
    c = TraversabilityConfig(resolution_m=0.1)
    result = classify(floor, roof, np.zeros(floor.shape, bool), np.zeros(floor.shape, bool), c)
    assert result["status"][20, 39] == 2
    assert result["reason_bits"][20, 39] & 4
    result = classify(
        floor, floor + 0.4, np.zeros(floor.shape, bool), np.zeros(floor.shape, bool), c
    )
    assert result["reason_bits"][20, 20] & 8


def test_small_prop_is_conservatively_blocked_only_on_its_own_layer():
    rock = trimesh.creation.box(extents=(0.01, 0.01, 0.3))
    rock.apply_translation((0.2, 0.2, 0.15))
    low, roof, _ = fields(box(), 1)
    a, height = obstacle_cells(
        [(rock.vertices, rock.faces)], low, roof, np.array([-4.0, -2.0]), 0.25
    )
    b, _ = obstacle_cells(
        [(rock.vertices, rock.faces)], low + 6, roof + 6, np.array([-4.0, -2.0]), 0.25
    )
    assert a.any() and not b.any()
    assert height.max() == pytest.approx(0.3)
    result = classify(low, roof, np.zeros(low.shape, bool), a, TraversabilityConfig())
    assert np.any(result["reason_bits"] & 16)


def test_unknown_is_not_free_and_propagates_through_reference_support():
    floor = np.zeros((24, 24))
    uncertain = np.zeros_like(floor, bool)
    uncertain[12, 12] = True
    result = classify(floor, floor + 2, uncertain, np.zeros_like(uncertain), TraversabilityConfig())
    assert result["status"][12, 12] == 255
    assert result["status"][12, 13] == 255
    assert result["status"][6, 6] == 1


def test_exports_have_layer_maps_ramp_records_units_legends_and_stable_arrays(tmp_path):
    mesh = trimesh.util.concatenate([box(), box(6)])
    req = request(2, ramp=True)
    # The ramp chart records unsupported areas; a schematic connection cannot make them free.
    for folder in ("first", "replay"):
        files = export_traversability(req, mesh.vertices, mesh.faces, tmp_path / folder)
        assert all(p.exists() for p in files)
    meta = json.loads((tmp_path / "first/manifest.json").read_text())
    assert [c["id"] for c in meta["charts"]] == ["layer_0", "layer_1", "ramp_9"]
    assert meta["charts"][-1]["from_layer"] == 0 and meta["charts"][-1]["to_layer"] == 1
    assert meta["charts"][0]["portals"][0]["connects_to"] == "ramp_9"
    assert meta["charts"][-1]["portals"][0]["connects_to"] == "layer_0"
    replay = json.loads((tmp_path / "replay/manifest.json").read_text())
    assert meta == replay
    assert meta["coordinates"]["units"] == "metres"
    assert not meta["robot_qualification"]
    for chart in meta["charts"]:
        a = np.load(tmp_path / "first" / chart["npz"], allow_pickle=False)
        b = np.load(tmp_path / "replay" / chart["npz"], allow_pickle=False)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
        image = np.asarray(Image.open(tmp_path / "first" / chart["occupancy"]))
        np.testing.assert_array_equal(image == 254, np.flipud(a["status"] == 1))
        assert tuple(a["status"].shape) == tuple(chart["shape_yx"])


def test_single_layer_exports_exactly_one_map(tmp_path):
    mesh = box()
    export_traversability(request(), mesh.vertices, mesh.faces, tmp_path)
    meta = json.loads((tmp_path / "manifest.json").read_text())
    assert len(meta["charts"]) == 1


def test_budget_is_checked_before_raster_allocation(tmp_path):
    mesh = box()
    with pytest.raises(TraversabilityBudgetError, match="grid needs"):
        export_traversability(
            request(config=TraversabilityConfig(max_cells=10)), mesh.vertices, mesh.faces, tmp_path
        )
    assert not tuple(tmp_path.glob("*.npz"))


@pytest.mark.parametrize(
    "values",
    [
        dict(enabled="yes"),
        dict(resolution_m=0),
        dict(resolution_m=float("nan")),
        dict(resolution_m=0.5),
        dict(robot_height_m=-1),
        dict(max_slope_deg=90),
        dict(max_cells=True),
        dict(max_step_m=0),
        dict(margin_m=-1),
        dict(resolution_m=0.001),
    ],
)
def test_invalid_map_controls_are_rejected(values):
    with pytest.raises(ValueError):
        TraversabilityConfig(**values)


def test_disabling_maps_does_not_write_anything(tmp_path):
    mesh = box()
    assert not export_traversability(
        request(config=replace(TraversabilityConfig(), enabled=False)),
        mesh.vertices,
        mesh.faces,
        tmp_path / "maps",
    )
    assert not (tmp_path / "maps").exists()


@pytest.mark.parametrize("collision", [False, True])
def test_maps_use_delivered_surface_and_are_registered_in_all_target_package(tmp_path, collision):
    from test_embedded_inspection import geometry

    from plume_advanced.exporters import export_target_asset
    from plume_advanced.world import ExportConfig

    cave = geometry(box().subdivide(), centers=((0, 0, 1),), smoothing=0)
    result = export_target_asset(
        cave,
        ExportConfig(target="all", file_format="auto", generate_collision=collision),
        tmp_path / "export",
        traversability=request(),
    )
    meta = json.loads((tmp_path / "export/traversability/manifest.json").read_text())
    assert meta["surface"]["kind"] == ("collision" if collision else "visual")
    shared = json.loads(result.primary_asset.read_text())["shared_files"]
    assert "traversability/layer_0.npz" in shared
    assert tmp_path / "export/traversability/layer_0.npz" in result.files
    inspection = json.loads((tmp_path / "export/pipeline_inspection.json").read_text())
    assert inspection["traversability"]["status"] == "complete"
    with np.load(tmp_path / "export/traversability/layer_0.npz") as data:
        assert (data["status"] == 1).any()
        np.testing.assert_allclose(data["floor_z_m"][np.isfinite(data["floor_z_m"])], 0)


def test_map_budget_does_not_discard_valid_cave(tmp_path):
    from test_embedded_inspection import geometry

    from plume_advanced.exporters import export_target_asset
    from plume_advanced.world import ExportConfig

    result = export_target_asset(
        geometry(box().subdivide(), centers=((0, 0, 1),), smoothing=0),
        ExportConfig(target="neutral", file_format="glb", generate_collision=False),
        tmp_path / "export",
        traversability=request(config=TraversabilityConfig(max_cells=10)),
    )
    assert result.primary_asset.is_file()
    meta = json.loads((tmp_path / "export/traversability/manifest.json").read_text())
    assert meta["status"] == "not_generated" and "grid needs" in meta["reason"]
    assert any("grid needs" in w for w in result.warnings)
    assert not tuple((tmp_path / "export/traversability").glob("*.npz"))


def test_unexpected_mapping_error_keeps_previous_package_intact(tmp_path, monkeypatch):
    from test_embedded_inspection import geometry

    from plume_advanced.exporters import export_target_asset
    from plume_advanced.traversability import export
    from plume_advanced.world import ExportConfig

    output = tmp_path / "export"
    output.mkdir()
    marker = output / "previous"
    marker.write_text("untouched")

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic unexpected map error")

    monkeypatch.setattr(export, "export_traversability", broken)
    with pytest.raises(RuntimeError, match="unexpected map error"):
        export_target_asset(
            geometry(box().subdivide(), centers=((0, 0, 1),), smoothing=0),
            ExportConfig(target="neutral", file_format="glb", generate_collision=False),
            output,
            traversability=request(),
        )
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "untouched"


def test_recipe_map_controls_are_strict_and_independent_of_qualification(tmp_path):
    from plume_advanced.config import load_project_config, project_config_manifest

    p = tmp_path / "recipe.toml"
    p.write_text(
        'recipe_version = 1\npreset = "preview"\n[traversability]\nresolution_m = 0.1\nrobot_height_m = 1.2\n'
    )
    project = load_project_config(p)
    assert project.traversability.resolution_m == 0.1
    assert project.traversability.robot_height_m == 1.2
    assert not project.acceptance.require_ground_routes
    assert project_config_manifest(project)["traversability"]["max_step_m"] == 0.1
    p.write_text(p.read_text() + "unknown_control = 4\n")
    with pytest.raises(ValueError, match="unknown_control"):
        load_project_config(p)


@pytest.mark.parametrize("output", [".", "..", "export_all/maps", "export_blender/maps"])
def test_saved_map_output_cannot_overwrite_source_or_packages(tmp_path, output):
    from plume_advanced.traversability.cli import main

    source = tmp_path / "run"
    (source / "export_blender").mkdir(parents=True)
    with pytest.raises(SystemExit) as caught:
        main(["--source", str(source), "--output", str(source / output)])
    assert caught.value.code == 2


@pytest.mark.parametrize("damage", ["array", "manifest", "slope_view"])
def test_pipeline_completion_rejects_changed_map_data(tmp_path, damage):
    from test_embedded_inspection import geometry

    from plume_advanced.exporters import export_target_asset
    from plume_advanced.pipeline.inspection import complete_inspection
    from plume_advanced.world import ExportConfig

    cave = geometry(box().subdivide(), centers=((0, 0, 1),), smoothing=0)
    result = export_target_asset(
        cave,
        ExportConfig(target="neutral", file_format="glb", generate_collision=False),
        tmp_path / "export",
        traversability=request(),
    )
    name = {"array": "layer_0.npz", "manifest": "manifest.json", "slope_view": "layer_0_slope.png"}[
        damage
    ]
    path = tmp_path / "export/traversability" / name
    path.write_bytes(path.read_bytes() + b"altered" if damage == "array" else b"{}")
    with pytest.raises(ValueError, match="Traversability"):
        complete_inspection(cave, result, {}, tmp_path)

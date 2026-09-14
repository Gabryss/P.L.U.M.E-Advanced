"""Inspection gates and bounded repairs execute inside ordinary generation/export."""

import numpy as np
import pytest
import trimesh

from plume_advanced.exporters import export_target_asset, targets
from plume_advanced.exporters import scene as scene_module
from plume_advanced.exporters.inspection import ExportInspectionError
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.stages.mesh_inspection import MeshInspectionError, inspect_surface
from plume_advanced.world import ExportConfig


def geometry(mesh=None, *, centers=((0.0, 0.0, 0.0),), smoothing=2, genus=0):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0) if mesh is None else mesh
    config = GeometryConfig(
        voxel_size=0.2,
        cave_smoothing_iterations=smoothing,
        cave_displacement_scale_m=0,
        cave_diffuse_texture="",
        cave_normal_texture="",
        cave_roughness_texture="",
        cave_displacement_texture="",
    )
    return CaveGeometry(
        config,
        VoxelGrid((-5, -5, -5), 0.2, np.ones((2, 2, 2)), 0),
        (),
        tuple(map(tuple, mesh.vertices)),
        tuple(map(tuple, mesh.faces)),
        1,
        1,
        (),
        route_centers=centers,
        expected_surface_genus=genus,
    )


def test_inspection_measures_actual_mesh_not_positive_source_density():
    cave = geometry()
    report = inspect_surface(
        cave.assembled_vertices, cave.assembled_faces, points=cave.route_centers, expected_genus=0
    )
    assert report["passed"] and report["measurements"][0]["clearance_m"] == pytest.approx(4.0)
    shifted = np.asarray(cave.assembled_vertices) + [0, 0, 10]
    with pytest.raises(MeshInspectionError) as caught:
        inspect_surface(shifted, cave.assembled_faces, points=cave.route_centers, expected_genus=0)
    assert caught.value.report["outside_centers"] == [0]


def test_vertical_intersections_distinguish_rock_island_from_air():
    torus = trimesh.creation.torus(major_radius=3.0, minor_radius=1.0)
    with pytest.raises(MeshInspectionError) as caught:
        inspect_surface(
            torus.vertices, torus.faces, points=((3.0, 0, 0), (0, 0, 0)), expected_genus=1
        )
    measurements = caught.value.report["measurements"]
    assert measurements[0]["inside"] and measurements[0]["clearance_m"] == pytest.approx(2.0)
    assert not measurements[1]["inside"]


def test_vertical_intersections_do_not_confuse_disjoint_air_layers():
    box = trimesh.creation.box(extents=(4, 4, 2))
    second = box.copy()
    second.apply_translation((0, 0, 6))
    mesh = box + second
    with pytest.raises(MeshInspectionError) as caught:
        inspect_surface(mesh.vertices, mesh.faces, points=((0, 0, 0), (0, 0, 3), (0, 0, 6)))
    rows = caught.value.report["measurements"]
    assert [row["inside"] for row in rows] == [True, False, True]
    assert rows[0]["clearance_m"] == rows[2]["clearance_m"] == 2.0


@pytest.mark.parametrize(
    "kind", ["open", "nonfinite", "degenerate", "winding", "index", "extra_shell"]
)
def test_mesh_gate_rejects_broken_surfaces(kind):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    if kind == "open":
        faces = faces[:-1]
    elif kind == "nonfinite":
        vertices[0, 0] = np.nan
    elif kind == "degenerate":
        vertices[faces[0, 0]] = vertices[faces[0, 1]]
    elif kind == "winding":
        faces[0] = faces[0, ::-1]
    elif kind == "index":
        faces[0, 0] = len(vertices)
    else:
        other = mesh.copy()
        other.apply_translation((5, 0, 0))
        vertices, faces = (mesh + other).vertices, (mesh + other).faces
    with pytest.raises(MeshInspectionError):
        inspect_surface(vertices, faces, expected_genus=0)


def test_exact_uv_seams_weld_but_real_seams_do_not():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    assert inspect_surface(vertices, faces, weld_seams=True, expected_genus=0)["passed"]
    vertices[0, 0] += 1e-7
    with pytest.raises(MeshInspectionError):
        inspect_surface(vertices, faces, weld_seams=True, expected_genus=0)


def test_intentional_event_obstruction_is_reported_not_silently_accepted():
    mesh = trimesh.creation.box()
    result = inspect_surface(mesh.vertices, mesh.faces, points=((0, 0, 3),), require_centers=False)
    assert result["passed"] and result["outside_centers"] == [0]
    assert result["warnings"] and not result["centers_required"]


def test_visual_repair_restores_passage_and_is_deterministic(monkeypatch):
    from plume_advanced.stages import geometry_export

    original = geometry()
    calls = []

    def bad_smoothing(vertices, faces, *, iterations, **kwargs):
        calls.append(iterations)
        return vertices + [0, 0, 10] if iterations else vertices.copy()

    monkeypatch.setattr(geometry_export, "_smooth_visual_surface", bad_smoothing)
    first = scene_module.prepare_export_scene(original, generate_collision=False)
    second = scene_module.prepare_export_scene(original, generate_collision=False)
    assert calls == [2, 0, 2, 0]
    assert original.config.cave_smoothing_iterations == 2
    assert first.geometry.config.cave_smoothing_iterations == 0
    attempts = first.inspection["visual_attempts"]
    assert not attempts[0]["accepted"] and attempts[1]["accepted"]
    assert attempts[0]["inspection"]["outside_centers"] == [0]
    assert first.inspection == second.inspection
    np.testing.assert_array_equal(
        first.canonical_visual["positions"], second.canonical_visual["positions"]
    )


@pytest.mark.parametrize("scale,accepted_scale", [(4.0, 2.0), (8.0, 0.0)])
def test_real_height_map_reduces_displacement_to_preserve_passage(
    tmp_path, scale, accepted_scale
):
    """Exercise actual baking/inspection, including half and zero amplitude repair."""
    from dataclasses import replace

    from PIL import Image

    height = tmp_path / "height.png"
    Image.new("L", (8, 8), 255).save(height)
    source_bytes = height.read_bytes()
    cave = geometry(centers=((0.0, 0.0, 1.3),), smoothing=0)
    cave = replace(cave, config=replace(
        cave.config, cave_displacement_texture=str(height), cave_displacement_scale_m=scale
    ))
    first, second = [
        scene_module.prepare_export_scene(cave, generate_collision=False) for _ in range(2)
    ]
    attempts = first.inspection["visual_attempts"]
    assert [r["displacement_scale_m"] for r in attempts] == (
        [scale, scale / 2] if accepted_scale else [scale, scale / 2, 0.0]
    )
    assert all(not r["accepted"] for r in attempts[:-1])
    assert attempts[-1]["accepted"]
    assert first.geometry.config.cave_displacement_scale_m == accepted_scale
    assert first.inspection["visual"]["measurements"][0]["inside"]
    assert first.inspection == second.inspection
    np.testing.assert_array_equal(first.canonical_visual["positions"],
                                  second.canonical_visual["positions"])
    assert cave.config.cave_displacement_scale_m == scale
    assert height.read_bytes() == source_bytes


def test_usd_points_round_trip_float32_and_single_ulp_corruption_is_rejected(tmp_path):
    from plume_advanced.exporters.inspection import _inspect_usda

    positions = np.array([
        [1016.2784423828125, -1005.1493530273438, 12.345678],
        [123456.789, -987.6543, 100.23456],
        [0.0000001234567, 253.3333, 1.2345],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    visual = dict(positions=positions, faces=faces)
    path = tmp_path / "cave.usd"

    def write(points):
        path.write_text("#usda 1.0\n" + "\n".join(targets._usda_mesh_lines(
            "CaveWall", points, faces, indent="", normals=np.ones_like(points)
        )))

    write(positions)
    _inspect_usda(path, visual)
    changed = positions.copy()
    changed[2, 0] = np.nextafter(changed[2, 0], np.float32(np.inf))
    write(changed)
    with pytest.raises(ValueError, match="Serialized USD points differs"):
        _inspect_usda(path, visual)


def test_visual_repair_exhaustion_preserves_previous_export(tmp_path, monkeypatch):
    from plume_advanced.stages import geometry_export

    calls = []

    def bad_smoothing(vertices, faces, **kwargs):
        calls.append(kwargs["iterations"])
        return vertices + [0, 0, 10]

    monkeypatch.setattr(geometry_export, "_smooth_visual_surface", bad_smoothing)
    output = tmp_path / "export"
    output.mkdir()
    (output / "previous.txt").write_text("keep")
    with pytest.raises(MeshInspectionError, match="All bounded visual repairs failed"):
        export_target_asset(
            geometry(), ExportConfig(target="neutral", generate_collision=False), output
        )
    assert calls == [2, 0]
    assert [path.name for path in output.iterdir()] == ["previous.txt"]


def test_programming_error_is_not_a_visual_retry(monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise TypeError("programming defect")

    monkeypatch.setattr(scene_module, "build_cave_visual_surface", broken)
    with pytest.raises(TypeError, match="programming defect"):
        scene_module.prepare_export_scene(geometry())
    assert len(calls) == 1


def test_closed_but_topologically_wrong_collider_falls_back(monkeypatch):
    torus = trimesh.creation.torus(major_radius=3.0, minor_radius=1.0)
    cave = geometry(torus, centers=((3.0, 0, 0),), smoothing=0, genus=1)
    wrong = trimesh.creation.icosphere(subdivisions=1)
    wrong.apply_translation((3, 0, 0))
    original = scene_module.simplified_collision_arrays

    def wrong_simplification(cave, **kwargs):
        if cave.config.collision_repair_attempts == 0:
            return original(cave, **kwargs)
        return wrong.vertices, wrong.faces

    monkeypatch.setattr(
        scene_module,
        "simplified_collision_arrays",
        wrong_simplification,
    )
    scene = scene_module.prepare_export_scene(cave)
    assert scene.inspection["collision"]["used_raw_fallback"]
    np.testing.assert_array_equal(scene.collision_faces, torus.faces)
    assert scene.inspection["collision"]["inspection"]["topology"]["genus"] == 1


def test_internal_collision_fallback_is_reported():
    # Clustering this long, thin volume collapses its entire cross section.
    # The low-level safety fallback must be visible in the exported evidence.
    mesh = trimesh.creation.box(extents=(1000.0, 0.5, 0.5))
    cave = geometry(mesh, smoothing=0)
    scene = scene_module.prepare_export_scene(cave)
    np.testing.assert_array_equal(scene.collision_vertices, mesh.vertices)
    np.testing.assert_array_equal(scene.collision_faces, mesh.faces)
    collision = scene.inspection["collision"]
    assert collision["used_raw_fallback"]
    assert collision["fallback_reason"]
    assert collision["inspection"]["passed"]


def test_internal_collision_fallback_reaches_run_quality_report(tmp_path):
    import json
    from dataclasses import replace

    from plume_advanced.pipeline.inspection import complete_inspection

    cave = geometry(smoothing=0)
    cave = replace(
        cave,
        config=replace(cave.config, voxel_size=2.0),
        voxel_grid=replace(cave.voxel_grid, voxel_size=2.0),
    )
    exported = export_target_asset(cave, ExportConfig(generate_collision=True), tmp_path / "export")
    quality, figure = complete_inspection(
        cave, exported, {"under_resolved_count": 0, "section_count": 1}, tmp_path
    )
    report = json.loads(quality.read_text())
    assert report["passed"] and figure.is_file()
    assert report["export_inspection"]["collision"]["used_raw_fallback"]
    assert report["export_inspection"]["collision"]["fallback_reason"]
    assert any("original collider" in warning for warning in report["warnings"])


@pytest.mark.parametrize("format", ["glb", "obj", "usd"])
def test_serialized_corruption_is_rejected_before_publication(tmp_path, monkeypatch, format):
    original = targets._export_target_asset_in_place

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        asset = result.primary_asset
        if format == "glb":
            data = asset.read_bytes()
            assert b'"doubleSided":false' in data
            asset.write_bytes(data.replace(b'"doubleSided":false', b'"doubleSided":true ', 1))
        elif format == "obj":
            data = asset.read_text()
            start = data.index("\nv ")
            end = data.index("\n", start + 1)
            asset.write_text(data[:start] + "\nv 999 999 999" + data[end:])
        else:
            data = asset.read_text()
            import re

            changed = re.sub(
                r"faceVertexIndices = \[\d+", "faceVertexIndices = [999", data, count=1
            )
            assert changed != data
            asset.write_text(changed)
        return result

    monkeypatch.setattr(targets, "_export_target_asset_in_place", corrupt)
    output = tmp_path / "export"
    output.mkdir()
    (output / "previous.txt").write_text("keep")
    config = ExportConfig(
        target="omniverse" if format == "usd" else "neutral",
        file_format=format,
        generate_collision=False,
    )
    with pytest.raises(ExportInspectionError):
        export_target_asset(geometry(), config, output)
    assert (output / "previous.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".export.staging-*"))


def test_package_gate_does_not_claim_final_run_provenance(tmp_path):
    import json

    from plume_advanced.validation import PortableAssetValidator

    result = export_target_asset(
        geometry(),
        ExportConfig(target="neutral", file_format="glb", generate_collision=False),
        tmp_path,
    )
    report = json.loads((tmp_path / "pipeline_inspection.json").read_text())
    assert report["passed"] and report["serialized"]["passed"]
    checks = next(
        row["checks"] for row in report["serialized"]["files"] if row["path"].endswith(".glb")
    )
    assert checks and all(check["category"] != "reproducibility" for check in checks)
    external = PortableAssetValidator(result.primary_asset, material_profile="neutral").validate()
    assert not all(check.passed for check in external if check.category == "reproducibility")


def test_protected_route_stays_mandatory_even_with_structural_events():
    mesh = trimesh.creation.box()
    with pytest.raises(MeshInspectionError) as caught:
        inspect_surface(
            mesh.vertices,
            mesh.faces,
            points=((0, 0, 0),),
            protected_points=((0, 0, 3),),
            require_centers=False,
        )
    assert caught.value.report["outside_centers"] == [1]
    assert caught.value.report["measurements"][1]["required"]


def test_run_acceptance_detects_asset_changed_since_staging(tmp_path):
    from plume_advanced.pipeline.inspection import complete_inspection

    cave = geometry()
    result = export_target_asset(
        cave,
        ExportConfig(target="neutral", file_format="glb", generate_collision=False),
        tmp_path / "export",
    )
    with result.primary_asset.open("ab") as stream:
        stream.write(b"changed after staging")
    with pytest.raises(ValueError, match="changed before run completion"):
        complete_inspection(cave, result, {}, tmp_path)
    assert not (tmp_path / "pipeline_quality_report.json").exists()


@pytest.mark.parametrize("slot", ["diffuse", "normal", "roughness"])
def test_partial_material_requires_its_configured_maps_only(tmp_path, slot):
    from dataclasses import replace

    from PIL import Image

    path = tmp_path / "tile.png"
    Image.new("RGB", (8, 8), (127, 127, 255)).save(path)
    cave = geometry()
    cave = replace(cave, config=replace(cave.config, **{f"cave_{slot}_texture": str(path)}))
    result = export_target_asset(
        cave,
        ExportConfig(target="neutral", file_format="glb", generate_collision=False),
        tmp_path / "package",
    )
    assert result.primary_asset.is_file()


@pytest.mark.parametrize(
    "slot,binding",
    [
        ("diffuse", "baseColorTexture"),
        ("normal", "normalTexture"),
        ("roughness", "metallicRoughnessTexture"),
    ],
)
def test_configured_partial_material_binding_cannot_disappear(tmp_path, monkeypatch, slot, binding):
    from dataclasses import replace

    from PIL import Image

    path = tmp_path / "tile.png"
    Image.new("RGB", (8, 8), (127, 127, 255)).save(path)
    cave = geometry()
    cave = replace(cave, config=replace(cave.config, **{f"cave_{slot}_texture": str(path)}))
    original = targets._export_target_asset_in_place

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        data = result.primary_asset.read_bytes()
        key = ('"' + binding + '"').encode()
        assert key in data
        result.primary_asset.write_bytes(data.replace(key, b'"' + b"x" * len(binding) + b'"', 1))
        return result

    monkeypatch.setattr(targets, "_export_target_asset_in_place", corrupt)
    with pytest.raises(ExportInspectionError, match=binding):
        export_target_asset(
            cave,
            ExportConfig(target="neutral", file_format="glb", generate_collision=False),
            tmp_path / "package",
        )
    assert not (tmp_path / "package").exists()


def test_actual_mesh_failure_participates_in_existing_surface_repairs(monkeypatch):
    from test_surface_acceptance import fixture

    from plume_advanced.stages import geometry as geometry_module
    from plume_advanced.stages.geometry import GeometryGenerator

    base = fixture()
    original = base.voxel_grid.density.copy()
    generator = GeometryGenerator(base.config)
    monkeypatch.setattr(generator, "_enforce_roof_stability", lambda *args: ())
    real_inspector = geometry_module.inspect_surface
    calls = []

    def one_damaged_mesh(vertices, faces, **kwargs):
        calls.append(True)
        inspected = np.asarray(vertices) + [0, 0, 1000] if len(calls) == 1 else vertices
        return real_inspector(inspected, faces, **kwargs)

    monkeypatch.setattr(geometry_module, "inspect_surface", one_damaged_mesh)
    result = generator._accept_base_surface(base, None, [], None)
    records = [dict(r) for r in result.surface_quality_records]
    assert any(not r["accepted"] and "inspection" in r for r in records)
    assert records[-1]["accepted"] and dict(result.mesh_inspection)["passed"]
    assert result.config.random_seed == base.config.random_seed
    np.testing.assert_array_equal(original, base.voxel_grid.density)


@pytest.mark.parametrize("format", ["obj", "usd"])
def test_serialized_collision_corruption_blocks_publication(tmp_path, monkeypatch, format):
    original = targets._export_target_asset_in_place
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if format == "obj":
            asset = next(p for p in result.files if p.name.endswith("_collision.obj"))
            text = asset.read_text()
            lines = text.splitlines()
            index = next(i for i, line in enumerate(lines) if line.startswith("v "))
            lines[index] = "v 999 999 999"
            asset.write_text("\n".join(lines)+"\n")
        else:
            asset = result.primary_asset
            text = asset.read_text()
            start = text.index('def Mesh "CaveCollision"')
            import re
            text = text[:start]+re.sub(r"faceVertexIndices = \[\d+", "faceVertexIndices = [999", text[start:], count=1)
            asset.write_text(text)
        return result
    monkeypatch.setattr(targets, "_export_target_asset_in_place", corrupt)
    with pytest.raises(ExportInspectionError):
        export_target_asset(geometry(), ExportConfig(target="omniverse" if format == "usd" else "neutral",
                            file_format=format, generate_collision=True), tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_usd_collider_midpoint_rounds_once_to_declared_float32(tmp_path):
    from plume_advanced.exporters.inspection import _inspect_usda
    midpoint = (float(np.float32(1.)) + float(np.nextafter(np.float32(1.), np.float32(2.))))/2
    vertices = np.array([[midpoint, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2]])
    path = tmp_path / "midpoint.usda"
    path.write_text("\n".join(targets._usda_mesh_lines("CaveCollision", vertices, faces, indent="")))
    _inspect_usda(path, dict(positions=vertices, faces=faces), object_name="CaveCollision")

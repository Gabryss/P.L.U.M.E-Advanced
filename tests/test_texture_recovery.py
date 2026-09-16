"""Texture faults exercise ordinary export, its journal, rollback and replay."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from plume_advanced.exporters import export_target_asset, targets
from plume_advanced.exporters.texture_recovery import (
    TextureRecoveryError,
    new_report,
    prepare_texture_assets,
)
from plume_advanced.identity import sha256_file
from plume_advanced.pipeline.inspection import complete_inspection
from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig, VoxelGrid
from plume_advanced.validation import GlbAsset
from plume_advanced.world import ExportConfig


@pytest.fixture
def cave(tmp_path):
    paths = {}
    for role, rgb in (
        ("diffuse", (60, 50, 40)),
        ("normal", (128, 128, 255)),
        ("roughness", (190, 190, 190)),
    ):
        path = tmp_path / f"{role}.png"
        Image.new("RGB", (32, 16), rgb).save(path)
        paths[f"cave_{role}_texture"] = str(path)
    config = GeometryConfig(
        cave_displacement_texture="",
        cave_displacement_scale_m=0,
        cave_smoothing_iterations=0,
        embedded_texture_max_size=16,
        **paths,
    )
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2)
    return CaveGeometry(
        config,
        VoxelGrid((-5, -5, -5), 0.2, np.ones((2, 2, 2)), 0),
        (),
        tuple(map(tuple, mesh.vertices)),
        tuple(map(tuple, mesh.faces)),
        1,
        1,
        (),
        route_centers=((0.0, 0.0, 0.0),),
        expected_surface_genus=0,
    )


def prepare(cave, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    report = new_report()
    effective = prepare_texture_assets(cave, tmp_path, report)
    return effective, report


def test_good_maps_are_bounded_copies_source_unchanged_and_replay_identical(cave, tmp_path):
    originals = {
        r: Path(getattr(cave.config, f"cave_{r}_texture")).read_bytes()
        for r in ("diffuse", "normal", "roughness")
    }
    first, report = prepare(cave, tmp_path / "first")
    second, replay = prepare(cave, tmp_path / "second")
    assert report == replay and report["outcome"] == "unchanged"
    for record in report["assets"]:
        assert record["source_size"] == [32, 16]
        assert record["prepared_size"] == [16, 8]  # retain aspect, never upscale
        assert (tmp_path / "first" / record["path"]).read_bytes() == (
            tmp_path / "second" / record["path"]
        ).read_bytes()
    for role, original in originals.items():
        assert Path(getattr(cave.config, f"cave_{role}_texture")).read_bytes() == original
    assert first.assembled_vertices is cave.assembled_vertices
    assert second.config.cave_normal_convention == "opengl"


@pytest.mark.parametrize("convention", ["opengl", "directx"])
def test_normal_vectors_repaired_and_convention_is_explicit(cave, tmp_path, convention):
    Image.new("RGB", (32, 16), (158, 168, 180)).save(cave.config.cave_normal_texture)
    cave = replace(cave, config=replace(cave.config, cave_normal_convention=convention))
    effective, report = prepare(cave, tmp_path / "prepared")
    normal = next(a for a in report["assets"] if a["role"] == "normal")
    assert normal["before"]["nonunit_vectors"] == 128
    assert normal["after"]["nonunit_vectors"] == 0
    assert normal["after"]["max_length_error"] < 0.01
    assert report["outcome"] == "repaired"
    with Image.open(effective.config.cave_normal_texture) as image:
        green = image.getpixel((0, 0))[1]
    assert (green < 128) == (convention == "directx")
    assert ("directx_to_opengl" in normal["repairs"]) == (convention == "directx")


def test_downsampling_normal_vectors_requires_renormalization(cave, tmp_path):
    pixels = np.zeros((16, 32, 3), dtype=np.uint8)
    pixels[:, ::2] = (230, 128, 204)
    pixels[:, 1::2] = (25, 128, 204)
    Image.fromarray(pixels).save(cave.config.cave_normal_texture)
    _, report = prepare(cave, tmp_path / "prepared")
    normal = next(a for a in report["assets"] if a["role"] == "normal")
    assert normal["before"]["nonunit_vectors"]
    assert normal["after"]["nonunit_vectors"] == 0


@pytest.mark.parametrize(
    "fault", ["missing", "corrupt", "truncated", "undefined", "grayscale", "unsupported"]
)
def test_irrecoverable_sources_fail_without_inventing_maps(cave, tmp_path, fault):
    path = Path(cave.config.cave_normal_texture)
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"not an image")
    elif fault == "truncated":
        path.write_bytes(path.read_bytes()[:45])
    elif fault == "undefined":
        Image.new("RGB", (32, 16), (128, 128, 128)).save(path)
    elif fault == "grayscale":
        Image.new("L", (32, 16), 128).save(path)
    else:
        renamed = path.with_suffix(".tiff")
        path.rename(renamed)
        cave = replace(cave, config=replace(cave.config, cave_normal_texture=str(renamed)))
    with pytest.raises(TextureRecoveryError) as caught:
        prepare(cave, tmp_path / "prepared")
    assert not caught.value.report["passed"]
    assert caught.value.report["failures"]
    assert not caught.value.report["package_attempts"]


def test_16bit_dark_roughness_is_not_saturated_or_autocontrasted(cave, tmp_path):
    Image.fromarray(np.full((16, 32), 128, dtype=np.uint16)).save(
        cave.config.cave_roughness_texture
    )
    effective, report = prepare(cave, tmp_path / "prepared")
    with Image.open(effective.config.cave_roughness_texture) as image:
        assert image.getextrema() == (0, 0)
    assert next(a for a in report["assets"] if a["role"] == "roughness")["source_mode"] in {
        "I;16",
        "I",
    }


def test_disabled_repairs_reject_invalid_normals_but_accept_good_maps(cave, tmp_path):
    cave = replace(cave, config=replace(cave.config, texture_repair_attempts=0))
    prepare(cave, tmp_path / "good")
    Image.new("RGB", (32, 16), (128, 128, 200)).save(cave.config.cave_normal_texture)
    with pytest.raises(TextureRecoveryError, match="texture_repair_attempts is zero"):
        prepare(cave, tmp_path / "bad")


@pytest.mark.parametrize("value", [-1, 2, 1.0, True, "1"])
def test_strict_repair_budget(value):
    with pytest.raises(ValueError, match="texture_repair_attempts"):
        GeometryConfig(texture_repair_attempts=value)


def test_convention_not_guessed():
    with pytest.raises(ValueError, match="cave_normal_convention"):
        GeometryConfig(cave_normal_convention="auto")


def test_partial_normal_material_is_repaired_without_inventing_other_maps(cave, tmp_path):
    cave = replace(
        cave, config=replace(cave.config, cave_diffuse_texture="", cave_roughness_texture="")
    )
    result = export_target_asset(
        cave, ExportConfig(target="blender", generate_collision=False), tmp_path / "export"
    )
    report = json.loads(
        next(p for p in result.files if p.name == "texture_recovery.json").read_text()
    )
    assert report["passed"] and len(report["assets"]) == 1
    assert not (tmp_path / "export/continuous_material").exists()


@pytest.mark.parametrize(
    "fault", ["missing_map", "stale_map", "stale_shader", "settings", "misbound", "wrap"]
)
def test_package_repair_reuses_geometry_and_retains_failed_inspection(
    cave, tmp_path, monkeypatch, fault
):
    bundle_writer = targets.write_projected_material_bundle
    calls = []

    def damage_once(source, output, **kwargs):
        paths = bundle_writer(source, output, **kwargs)
        calls.append(sha256_file(source))
        if len(calls) == 1:
            if fault == "missing_map":
                (output / "textures/cave_normal.png").unlink()
            elif fault == "stale_map":
                Image.new("RGB", (16, 8), (255, 128, 128)).save(output / "textures/cave_normal.png")
            elif fault == "stale_shader":
                (output / "unity/PlumeContinuousRock.shader").write_text("bad shader")
            elif fault == "settings":
                settings = json.loads((output / "settings.json").read_text())
                settings["tile_size_m"] *= 3
                (output / "settings.json").write_text(json.dumps(settings))
            else:
                _mutate_glb(source, fault)
        return paths

    monkeypatch.setattr(targets, "write_projected_material_bundle", damage_once)
    prepare_scene = targets.prepare_export_scene
    prepared = []

    def once(*args, **kwargs):
        prepared.append(True)
        return prepare_scene(*args, **kwargs)

    monkeypatch.setattr(targets, "prepare_export_scene", once)
    result = export_target_asset(
        cave, ExportConfig(target="blender", generate_collision=False), tmp_path / "export"
    )
    assert len(prepared) == 1 and len(calls) == 2 and calls[0] == calls[1]
    report = json.loads(
        next(p for p in result.files if p.name == "texture_recovery.json").read_text()
    )
    assert report["passed"] and report["outcome"] == "repaired"
    assert not report["package_attempts"][0]["passed"]
    assert report["package_attempts"][1]["passed"]
    assert report["package_attempts"][0]["failures"]
    complete_inspection(cave, result, dict(under_resolved_count=0), tmp_path)


def _mutate_glb(path, fault):
    import struct

    asset = GlbAsset(path)
    document = asset.document
    if fault == "wrap":
        document["samplers"][0]["wrapS"] = 33071
    else:
        _, _, primitive = asset.cave_primitive()
        material = document["materials"][primitive["material"]]
        material["normalTexture"]["index"] = material["pbrMetallicRoughness"]["baseColorTexture"][
            "index"
        ]
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    binary = bytes(asset.binary)
    binary += b"\0" * (-len(binary) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 28 + len(payload) + len(binary))
        + struct.pack("<I4s", len(payload), b"JSON")
        + payload
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )


@pytest.mark.parametrize("budget", [0, 1])
def test_exhaustion_preserves_existing_export(cave, tmp_path, monkeypatch, budget):
    cave = replace(cave, config=replace(cave.config, texture_repair_attempts=budget))
    writer = targets.write_projected_material_bundle
    calls = []

    def broken(*args, **kwargs):
        paths = writer(*args, **kwargs)
        next(p for p in paths if p.name == "cave_normal.png").unlink()
        calls.append(True)
        return paths

    monkeypatch.setattr(targets, "write_projected_material_bundle", broken)
    output = tmp_path / "previous"
    output.mkdir()
    (output / "keep.txt").write_text("previous accepted asset")
    with pytest.raises(TextureRecoveryError) as caught:
        export_target_asset(cave, ExportConfig(target="blender", generate_collision=False), output)
    assert len(calls) == budget + 1
    assert len(caught.value.report["package_attempts"]) == budget + 1
    assert not caught.value.report["passed"]
    assert [p.name for p in output.iterdir()] == ["keep.txt"]


def test_export_replay_and_portable_obj_paths(cave, tmp_path):
    cfg = ExportConfig(target="blender", generate_collision=False)
    first = export_target_asset(cave, cfg, tmp_path / "first")
    export_target_asset(cave, cfg, tmp_path / "second")
    for name in (first.primary_asset.name, "texture_recovery.json", "pipeline_inspection.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    for path in first.files:
        if path.suffix in {".json", ".mtl"}:
            assert ".staging-" not in path.read_text()
        if path.suffix == ".mtl":
            for line in path.read_text().splitlines():
                if line.startswith(("map_", "norm ")):
                    reference = (path.parent / line.split()[-1]).resolve()
                    assert reference.is_file() and reference.is_relative_to(tmp_path / "first")
    # Successful export does not make subsequently edited textures acceptable.
    normal = next(p for p in first.files if p.name == "cave_normal.png")
    normal.write_bytes(b"edited after inspection")
    with pytest.raises(ValueError, match="texture package changed"):
        complete_inspection(cave, first, dict(under_resolved_count=0), tmp_path)


def test_programming_error_is_not_retried(cave, tmp_path, monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise TypeError("programming error")

    monkeypatch.setattr(targets, "_export_target_asset_in_place", broken)
    with pytest.raises(TypeError, match="programming error"):
        export_target_asset(
            cave, ExportConfig(target="blender", generate_collision=False), tmp_path / "export"
        )
    assert len(calls) == 1


def test_shared_event_maps_are_prepared_once(cave, tmp_path):
    from plume_advanced.stages.events import GeologicalEventMesh

    maps = tuple(
        (role, getattr(cave.config, f"cave_{role}_texture"))
        for role in ("diffuse", "normal", "roughness")
    )
    events = tuple(
        GeologicalEventMesh(i, "rock", "basalt", (), (), material_maps=maps) for i in range(8)
    )
    cave = replace(cave, event_meshes=events)
    effective, report = prepare(cave, tmp_path / "prepared")
    assert len(report["assets"]) == 3
    assert all(len(a["users"]) == 9 for a in report["assets"])
    assert len(list((tmp_path / "prepared/texture_assets").iterdir())) == 3
    for event in effective.event_meshes:
        assert dict(event.material_maps)["normal"] == effective.config.cave_normal_texture


def test_displacement_precision_is_left_to_geometry_acceptance(cave, tmp_path):
    path = tmp_path / "height.png"
    Image.fromarray(np.array([[0, 128], [512, 65535]], dtype=np.uint16)).save(path)
    original = path.read_bytes()
    cave = replace(cave, config=replace(cave.config, cave_displacement_texture=str(path)))
    effective, report = prepare(cave, tmp_path / "prepared")
    assert effective.config.cave_displacement_texture == str(path)
    assert path.read_bytes() == original
    assert "Precision preserved" in report["assets"][-1]["processing"]


@pytest.mark.parametrize("kind", ["disk", "memory"])
def test_resource_failure_is_not_a_texture_retry(cave, tmp_path, monkeypatch, kind):
    import errno

    def failed(*args, **kwargs):
        if kind == "memory":
            raise MemoryError("budget")
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(Image.Image, "save", failed)
    with pytest.raises(MemoryError if kind == "memory" else OSError):
        prepare(cave, tmp_path / "prepared")


def test_repair_journal_matches_across_directories_even_after_missing_file(
    cave, tmp_path, monkeypatch
):
    writer = targets.write_projected_material_bundle
    counts = {}

    def damage_first(source, output, **kwargs):
        paths = writer(source, output, **kwargs)
        counts[output] = counts.get(output, 0) + 1
        if counts[output] == 1:
            (output / "textures/cave_normal.png").unlink()
        return paths

    monkeypatch.setattr(targets, "write_projected_material_bundle", damage_first)
    for name in ("first", "second"):
        export_target_asset(
            cave, ExportConfig(target="blender", generate_collision=False), tmp_path / name
        )
    assert (tmp_path / "first/texture_recovery.json").read_bytes() == (
        tmp_path / "second/texture_recovery.json"
    ).read_bytes()


def test_failed_texture_diagnosis_does_not_recommend_new_seed():
    from plume_advanced.evaluation.reliability_reports import diagnose

    error = TextureRecoveryError(dict(failures=["Missing normal map"], passed=False))
    diagnosis = diagnose(error, "export")
    assert diagnosis["category"] == "texture_rejected"


def test_export_cannot_erase_its_source_maps(cave, tmp_path):
    originals = {p.name: p.read_bytes() for p in tmp_path.glob("*.png")}
    with pytest.raises(TextureRecoveryError, match="outside the replaced export"):
        export_target_asset(
            cave, ExportConfig(target="blender", generate_collision=False), tmp_path
        )
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.png")} == originals


@pytest.mark.parametrize(
    "name", ["unity/material.shader", "body.hlsl", "unity/install.cs", "SETUP.txt"]
)
def test_shader_resources_participate_in_source_fingerprint(tmp_path, name):
    from plume_advanced.identity import package_source_hash

    (tmp_path / "__init__.py").write_text("")
    before = package_source_hash(tmp_path)
    resource = tmp_path / "material_assets" / name
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text("initial")
    initial = package_source_hash(tmp_path)
    assert initial != before
    resource.write_text("changed")
    assert package_source_hash(tmp_path) not in {before, initial}
    current = package_source_hash(tmp_path)
    cache = tmp_path / "material_assets/__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "local.pyc").write_bytes(b"cache")
    assert package_source_hash(tmp_path) == current

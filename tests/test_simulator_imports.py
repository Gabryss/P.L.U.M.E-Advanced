"""Import regressions discovered by the native Gazebo / Isaac smoke campaign."""

import hashlib
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from plume_advanced.exporters.targets import _gazebo_visuals, _relocate_obj_material_textures


@pytest.fixture
def gazebo_check():
    spec = importlib.util.spec_from_file_location(
        "gazebo_check", Path(__file__).resolve().parents[1] / "scripts/check_gazebo.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gazebo_pbr_keeps_event_and_cave_materials_separate(tmp_path):
    package = tmp_path / "cave"
    meshes = package / "meshes"
    meshes.mkdir(parents=True)
    textures = package / "materials/textures"
    textures.mkdir(parents=True)
    for name in ("cave.png", "rock.png", "normal.png", "roughness.png"):
        (textures / name).write_bytes(b"fixture")
    mesh = meshes / "cave.obj"
    mesh.write_text("o cave_wall\nusemtl wall\no event_0001_rock\nusemtl rock\n")
    material = mesh.with_suffix(".mtl")
    material.write_text(
        "newmtl wall\nKd 1 1 1\nPr 1\nPm 0\n"
        "map_Kd ../materials/textures/cave.png\n"
        "norm ../materials/textures/normal.png\n"
        "map_Pr ../materials/textures/roughness.png\n"
        "newmtl rock\nKd 0.4 0.3 0.2\nmap_Kd ../materials/textures/rock.png\n"
    )
    root = ET.fromstring("<link>" + "\n".join(
        _gazebo_visuals(mesh, material, "model://cave/meshes/cave.obj", "cave")
    ) + "</link>")
    visuals = root.findall("visual")
    assert len(visuals) == 2
    for visual, obj, image in zip(visuals, ("cave_wall", "event_0001_rock"), ("cave", "rock")):
        assert visual.findtext("geometry/mesh/submesh/name") == obj
        assert visual.findtext("geometry/mesh/submesh/center") == "false"
        assert visual.findtext("material/pbr/metal/albedo_map") == f"model://cave/materials/textures/{image}.png"
    assert visuals[0].findtext("material/pbr/metal/normal_map").endswith("/normal.png")
    assert visuals[0].findtext("material/pbr/metal/roughness_map").endswith("/roughness.png")
    assert visuals[1].find("material/pbr/metal/normal_map") is None


def test_norm_and_bump_relocate_to_the_same_single_file(tmp_path):
    meshes = tmp_path / "package/meshes"
    meshes.mkdir(parents=True)
    source = tmp_path / "normal.png"
    source.write_bytes(b"fixture")
    material = meshes / "cave.mtl"
    material.write_text(f"newmtl cave\nnorm {source}\nmap_Bump -bm 1 {source}\n")
    copied = _relocate_obj_material_textures(material, meshes.parent / "materials/textures")
    assert len(copied) == 1
    assert material.read_text().count("../materials/textures/normal.png") == 2
    assert str(source) not in material.read_text()


def test_native_gazebo_world_references_export_and_publishes_contacts(tmp_path, gazebo_check):
    model = tmp_path / "cave & rocks"
    world = ET.fromstring(gazebo_check.inspection_world(
        model, {"position": [1, 2, 3], "direction": [1, 0, 0], "floor_z": 2.2},
    ))
    assert world.findtext("world/include/uri") == model.as_uri()
    assert world.findtext(".//sensor[@type='contact']/contact/topic") == "/plume/check/contacts"
    assert world.findtext(".//sensor[@type='camera']/topic") == "/plume/check/image"
    assert world.find(".//mesh") is None  # No substitute or simplified test geometry.


@pytest.mark.parametrize("failure", [None, "absent_role", "missing_file", "external_uri", "path_escape"])
def test_native_gazebo_texture_receipts_require_complete_local_maps(tmp_path, gazebo_check, failure):
    package = tmp_path / "cave"
    package.mkdir()
    image = package / "map.png"
    image.write_bytes(b"texture fixture")
    outside = tmp_path / "external.png"
    outside.write_bytes(b"external fixture")
    root = ET.Element("sdf")
    metal = ET.SubElement(ET.SubElement(root, "pbr"), "metal")
    for role in ("albedo_map", "normal_map", "roughness_map"):
        if failure == "absent_role" and role == "normal_map":
            continue
        uri = "model://cave/map.png"
        if role == "normal_map":
            uri = {
                "missing_file": "model://cave/missing.png",
                "external_uri": outside.as_uri(),
                "path_escape": "model://cave/../external.png",
            }.get(failure, uri)
        ET.SubElement(metal, role).text = uri
    ET.ElementTree(root).write(package / "model.sdf")
    if failure:
        with pytest.raises(ValueError):
            gazebo_check.texture_receipts(package)
    else:
        receipts = gazebo_check.texture_receipts(package)
        assert len(receipts) == 3
        assert all(row["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest() for row in receipts)

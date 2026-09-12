"""Portable GLB validation regressions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from plume_advanced.stages.geometry_export import export_geometry_glb
from plume_advanced.stages.geometry_types import (
    CaveGeometry,
    GeometryConfig,
    VoxelGrid,
)
from plume_advanced.validation import (
    PortableAssetValidator,
    write_validation_reports,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("requested", [False, True, None])
def test_collision_sidecar_is_optional_only_when_explicitly_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: bool | None,
) -> None:
    mesh = trimesh.creation.box()
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="cave_wall", node_name="cave_wall")
    asset = tmp_path / "cave.glb"
    scene.export(asset)
    validator = PortableAssetValidator(asset)
    export = {} if requested is None else {"generate_collision": requested}
    validator.run_manifest = {"resolved_config": {"export": export}}
    monkeypatch.setattr(
        validator, "_geometry_arrays",
        lambda: (mesh.vertices, mesh.faces, None, None, None),
    )
    collision_check = next(
        check for check in validator._geometry_checks()
        if check.name == "Conservative collision sidecar"
    )
    assert collision_check.passed is (requested is False)


@pytest.mark.parametrize("package", ["export_neutral", "export_all/blender", "export_all/unity"])
def test_validator_discovers_run_manifest_for_export_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    output = tmp_path / "outputs"
    asset = output / package / "cave.glb"
    asset.parent.mkdir(parents=True)
    trimesh.creation.box().export(asset)
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "plume.run-manifest.v1",
        "status": "complete",
        "outputs": [{"path": str(asset.relative_to(output)), "sha256": _sha256(asset)}],
    }))
    monkeypatch.chdir(tmp_path)

    validator = PortableAssetValidator(asset.relative_to(tmp_path))

    assert validator.run_manifest_path == manifest
    assert all(check.passed for check in validator._reproducibility_checks())
    explicit = tmp_path / "explicit.json"
    explicit.write_text(manifest.read_text())
    assert PortableAssetValidator(asset, run_manifest_path=explicit).run_manifest_path == explicit


def test_portable_validator_checks_embedded_materials_and_displacement(
    tmp_path: Path,
) -> None:
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    diffuse = texture_dir / "diffuse.png"
    normal = texture_dir / "normal.png"
    roughness = texture_dir / "roughness.png"
    displacement = texture_dir / "displacement.png"
    Image.new("RGB", (4, 4), (82, 76, 68)).save(diffuse)
    Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
    Image.new("L", (4, 4), 210).save(roughness)
    height = Image.new("L", (4, 4))
    height.putdata(
        (
            0,
            64,
            128,
            255,
            32,
            96,
            160,
            224,
            64,
            128,
            192,
            255,
            0,
            80,
            176,
            240,
        )
    )
    height.save(displacement)

    frame = (0.25, 0.25, 0.25)
    geometry = CaveGeometry(
        config=GeometryConfig(
            cave_diffuse_texture=str(diffuse),
            cave_normal_texture=str(normal),
            cave_roughness_texture=str(roughness),
            cave_displacement_texture=str(displacement),
            cave_displacement_scale_m=0.05,
            cave_smoothing_iterations=0,
            embedded_texture_max_size=16,
        ),
        voxel_grid=VoxelGrid(
            origin=(0.0, 0.0, 0.0),
            voxel_size=1.0,
            density=np.zeros((2, 2, 2), dtype=np.float32),
            iso_level=0.0,
        ),
        chunk_meshes=(),
        assembled_vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        assembled_faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
        component_count=1,
        stamped_sample_count=1,
        stamped_segment_ids=(1,),
        route_centers=(frame,),
    )
    export_dir = tmp_path / "export_neutral"
    asset = export_geometry_glb(geometry, export_dir / "plume_cave_scene.glb")
    collision = asset.with_name(f"{asset.stem}_collision.obj")
    trimesh.Trimesh(
        vertices=np.asarray(geometry.assembled_vertices),
        faces=np.asarray(geometry.assembled_faces),
        process=False,
    ).export(collision)
    manifest = asset.with_suffix(".manifest.json")
    records = []
    for path in (asset, manifest, collision):
        records.append(
            {
                "path": str(path.relative_to(tmp_path)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "plume.run-manifest.v1",
                "status": "complete",
                "outputs": records,
            }
        ),
        encoding="utf-8",
    )

    validator = PortableAssetValidator(asset)
    updates: list[tuple[int, int, str]] = []
    checks = validator.validate(
        progress=lambda current, total, detail: updates.append(
            (current, total, detail)
        )
    )
    failed = [check for check in checks if not check.passed]
    assert not failed, [(check.name, check.detail) for check in failed]
    assert len(updates) == len(validator.PHASES)
    assert updates[-1] == (
        len(validator.PHASES),
        len(validator.PHASES),
        "Reproducibility",
    )

    json_report, markdown_report = write_validation_reports(
        asset,
        checks,
        tmp_path / "validation",
    )
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["valid"]
    assert "Baked displacement" not in markdown_report.read_text(encoding="utf-8")
    assert "Portable displacement contract" in markdown_report.read_text(
        encoding="utf-8"
    )

    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tampered_checks = validator.validate()
    hash_check = next(
        check
        for check in tampered_checks
        if check.name == "Generated output hashes"
    )
    assert not hash_check.passed
    assert "verified=2/3" in hash_check.detail
